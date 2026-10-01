#
# CEBRA: Consistent EmBeddings of high-dimensional Recordings using Auxiliary variables
# © Mackenzie W. Mathis & Steffen Schneider (v0.4.0+)
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/blob/main/LICENSE.md
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Multiobjective contrastive learning.

Starting in CEBRA 0.6.0, we have added support for subspace contrastive learning.
This is a method for training models that are able to learn multiple subspaces of the
feature space simultaneously.

Subspace contrastive learning requires to use specialized models and criterions.
This module specifies a test of classes required for training CEBRA models with multiple objectives.
The objectives are defined by the wrapper class :py:class:`cebra.models.multicriterions.MultiCriterions`.

Two solvers are currently implemented:

- :py:class:`cebra.solver.multiobjective.ContrastiveMultiobjectiveSolverxCEBRA`
- :py:class:`cebra.solver.multiobjective.SupervisedMultiobjectiveSolverxCEBRA`

See Also:
    :py:class:`cebra.solver.multiobjective.SupervisedMultiobjectiveSolverxCEBRA`
    :py:class:`cebra.solver.multiobjective.MultiObjectiveConfig`
    :py:class:`cebra.models.multicriterions.MultiCriterions`
"""

import logging
import time
import warnings
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Tuple

import literate_dataclasses as dataclasses
import numpy as np
import torch

import cebra
import cebra.data
import cebra.io
import cebra.models
import cebra.solver.single_session as cebra_solver_single
from cebra.solver import register
from cebra.solver.base import Solver
from cebra.solver.schedulers import Scheduler
from cebra.solver.util import Meter


class MultiObjectiveConfig:
    """Configuration class for setting up multi-objective learning with Cebra.



    Args:
        loader: Data loader used for configurations.
    """

    def __init__(self, loader):
        self.loader = loader
        self.total_info = []
        self.current_info = {}

    def _check_overwriting_key(self, key):
        if key in self.current_info:
            warnings.warn(
                "Configuration key already exists. Overwriting existing value. "
                "If you don't want to overwrite you should call push() before.")

    def _check_pushed_status(self):
        if "slice" not in self.current_info:
            raise RuntimeError(
                "Slice configuration is missing. Add it before pushing it.")
        if "distributions" not in self.current_info:
            raise RuntimeError(
                "Distributions configuration is missing. Add it before pushing it."
            )
        if "losses" not in self.current_info:
            raise RuntimeError(
                "Losses configuration is missing. Add it before pushing it.")

    def set_slice(self, start, end):
        """Select the index range of the embedding.

        The configured loss will be applied to the ``start:end`` slice of the
        embedding space. Make sure the selected dimensionality is appropriate
        for the chosen loss function and distribution.
        """
        self._check_overwriting_key("slice")
        self.current_info['slice'] = (start, end)

    def set_loss(self, loss_name, **kwargs):
        """Select the loss function to apply.

        Select a valid loss function from :py:mod:`cebra.models.criterions`.
        Common choices are:

        - `FixedEuclideanInfoNCE`
        - `FixedCosineInfoNCE`

        which can be passed as string values to ``loss_name``. The loss
        will be applied to the range specified with ``set_slice``.
        """
        self._check_overwriting_key("losses")
        self.current_info["losses"] = {"name": loss_name, "kwargs": kwargs}

    def set_distribution(self, distribution_name, **kwargs):
        """Select the distribution to sample from.

        The loss function specified in ``set_loss`` is applied to positive
        and negative pairs sampled from the specified distribution.
        """
        self._check_overwriting_key("distributions")
        self.current_info["distributions"] = {
            "name": distribution_name,
            "kwargs": kwargs
        }

    def push(self):
        """Add a slice/loss/distribution setting to the config.

        After calling all of ``set_slice``, ``set_loss``, ``set_distribution``,
        add this group to the config by calling this function.

        Once all configuration parts are pushed, call ``finalize`` to finish
        the configuration.
        """
        self._check_pushed_status()
        print(f"Adding configuration for slice: {self.current_info['slice']}")
        self.total_info.append(self.current_info)
        self.current_info = {}

    def finalize(self):
        """Finalize the multiobjective configuration."""
        self.losses = []
        self.feature_ranges = []
        self.feature_ranges_tuple = []

        for info in self.total_info:
            self._process_info(info)

        if len(set(self.feature_ranges_tuple)) != len(
                self.feature_ranges_tuple):
            raise RuntimeError(
                "Feature ranges are not unique. Please check again and remove the duplicates. "
                f"Feature ranges: {self.feature_ranges_tuple}")

        print("Creating MultiCriterion")
        self.criterion = cebra.models.MultiCriterions(losses=self.losses,
                                                      mode="contrastive")

    def _process_info(self, info):
        """
        Processes individual configuration info and updates the losses and feature ranges.

        Args:
            info (dict): The configuration info to process.
        """
        slice_info = info["slice"]
        losses_info = info["losses"]
        distributions_info = info["distributions"]

        self.losses.append(
            dict(indices=(slice_info[0], slice_info[1]),
                 contrastive_loss=dict(name=losses_info['name'],
                                       kwargs=losses_info['kwargs'])))

        self.feature_ranges.append(slice(slice_info[0], slice_info[1]))
        self.feature_ranges_tuple.append((slice_info[0], slice_info[1]))

        print(f"Adding distribution of slice: {slice_info}")
        self.loader.add_config(
            dict(distribution=distributions_info["name"],
                 kwargs=distributions_info["kwargs"]))


@dataclasses.dataclass
class MultiobjectiveSolverBase(cebra_solver_single.SingleSessionSolver):

    feature_ranges: List[slice] = None
    renormalize: bool = None
    log: Dict[Tuple,
              List[float]] = dataclasses.field(default_factory=lambda: ({}))
    use_sam: bool = False
    regularizer: torch.nn.Module = None
    metadata: Dict = dataclasses.field(default_factory=lambda: ({
        "timestamp": None,
        "batches_seen": None,
    }))

    def __post_init__(self):
        super().__post_init__()

        self.model = cebra.models.create_multiobjective_model(
            module=self.model,
            feature_ranges=self.feature_ranges,
            renormalize=self.renormalize,
        )

    def parameters(self, session_id: Optional[int] = None):
        """Iterate over all parameters."""
        super().parameters(session_id=session_id)

        for parameter in self.regularizer.parameters():
            yield parameter

    def fit(self,
            loader: cebra.data.Loader,
            valid_loader: cebra.data.Loader = None,
            *,
            valid_frequency: int = None,
            log_frequency: int = None,
            save_hook: Callable[[int, "Solver"], None] = None,
            scheduler_regularizer: "Scheduler" = None,
            scheduler_loss: "Scheduler" = None,
            logger: logging.Logger = None):
        """Train model for the specified number of steps.

        Args:
            loader: Data loader, which is an iterator over `cebra.data.Batch` instances.
                Each batch contains reference, positive and negative input samples.
            valid_loader: Data loader used for validation of the model.
            valid_frequency: The frequency for running validation on the ``valid_loader`` instance.
            logdir:  The logging directory for writing model checkpoints. The checkpoints
                can be read again using the `solver.load` function, or manually via loading the
                state dict.
            save_hook: callback. It will be called when we run validation.
            log_frequency: how frequent we log things.
            logger: logger to log progress. None by default.

        """

        def _run_validation():
            stats_val = self.validation(valid_loader, logger=logger)
            if save_hook is not None:
                save_hook(solver=self, step=num_steps)
            return stats_val

        self._set_fitted_params(loader)
        self.to(loader.device)

        iterator = self._get_loader(loader,
                                    logger=logger,
                                    log_frequency=log_frequency)
        self.model.train()
        for num_steps, batch in iterator:
            weights_regularizer = None
            if scheduler_regularizer is not None:
                weights_regularizer = scheduler_regularizer.get_weights(
                    step=num_steps)
                # NOTE(stes): Both SAM and Jacobian regularization is not yet supported.
                # For this, we need to re-implement the closure logic below (right now,
                # the closure function applies the non-regularized loss in the second
                # step, it is unclear if that is the correct behavior.
                assert not self.use_sam

            weights_loss = None
            if scheduler_loss is not None:
                weights_loss = scheduler_loss.get_weights()

            stats = self.step(batch,
                              weights_regularizer=weights_regularizer,
                              weights_loss=weights_loss)

            self._update_metadata(num_steps)
            iterator.set_description(stats)
            run_validation = (valid_loader
                              is not None) and (num_steps % valid_frequency
                                                == 0)
            if run_validation:
                _run_validation()

        #TODO
        #_run_validation()

    def _get_loader(self, loader, **kwargs):
        return super()._get_loader(loader)

    def _update_metadata(self, num_steps):
        self.metadata["timestamp"] = time.time()
        self.metadata["batches_seen"] = num_steps

    def compute_regularizer(self, predictions, inputs):
        regularizer = []
        for prediction in predictions:
            R = self.regularizer(inputs, prediction.reference)
            regularizer.append(R)

        return regularizer

    def create_closure(self, batch, weights_loss):

        def inner_closure():
            predictions = self._inference(batch)
            losses = self.criterion(predictions)

            if weights_loss is not None:
                assert len(weights_loss) == len(
                    losses
                ), "Number of weights should match the number of losses"
                losses = [
                    weight * loss for weight, loss in zip(weights_loss, losses)
                ]

            loss = sum(losses)
            loss.backward()
            return loss

        return inner_closure

    def step(self,
             batch: cebra.data.Batch,
             weights_loss: Optional[List[float]] = None,
             weights_regularizer: Optional[List[float]] = None) -> dict:
        """Perform a single gradient update with multiple objectives."""

        closure = None
        if self.use_sam:
            closure = self.create_closure(batch, weights_loss)

        if weights_regularizer is not None:
            assert isinstance(batch.reference, torch.Tensor)
            batch.reference.requires_grad_(True)

        predictions = self._inference(batch)
        losses = self.criterion(predictions)

        for i, loss_value in enumerate(losses):
            key = "loss_train", i
            self.log.setdefault(key, []).append(loss_value.item())

        if weights_loss is not None:
            losses = [
                weight * loss for weight, loss in zip(weights_loss, losses)
            ]

        loss = sum(losses)

        if weights_regularizer is not None:
            regularizer = self.compute_regularizer(predictions=predictions,
                                                   inputs=batch.reference)
            assert len(weights_regularizer) == len(regularizer) == len(losses)
            loss = loss + sum(
                weight * reg
                for weight, reg in zip(weights_regularizer, regularizer))

        loss.backward()
        self.optimizer.step(closure)
        self.optimizer.zero_grad()

        if weights_regularizer is not None:
            for i, (weight,
                    reg) in enumerate(zip(weights_regularizer, regularizer)):
                assert isinstance(weight, float)
                self.log.setdefault(("regularizer", i), []).append(reg.item())
                self.log.setdefault(("regularizer_weight", i),
                                    []).append(weight)

        if weights_loss is not None:
            for i, weight in enumerate(weights_loss):
                assert isinstance(weight, float) or isinstance(weight, int)
                self.log.setdefault(("loss_weight", i), []).append(weight)

        # add sum_loss_train
        self.log.setdefault(("sum_loss_train",), []).append(loss.item())
        return {"sum_loss_train": loss.item()}

    @torch.no_grad()
    def _compute_metrics(self):
        # NOTE: We set split_outputs = False when we compute
        # validation metrics, otherwise it returns a tuple
        # which led to a bug before.
        embeddings = {}
        self.model.set_split_outputs(False)
        for split in self.metrics.splits:
            embedding_tensor = self.transform(
                self.metrics.datasets[split].neural)
            embedding_np = embedding_tensor.cpu().numpy()
            assert embedding_np.shape[1] == self.model.num_output
            embeddings[split] = embedding_np

        self.model.set_split_outputs(True)
        return self.metrics.compute_metrics(embeddings)

    @torch.no_grad()
    def validation(
        self,
        loader: cebra.data.Loader,
        logger=None,
        weights_loss: Optional[List[float]] = None,
    ):
        loader.dataset.configure_for(self.model)
        iterator = self._get_loader(loader)

        self.model.eval()
        total_loss = Meter()

        losses_dict = {}
        for _, batch in iterator:
            predictions = self._inference(batch)
            losses = self.criterion(predictions)

            if weights_loss is not None:
                assert len(weights_loss) == len(
                    losses
                ), "Number of weights should match the number of losses"
                losses = [
                    weight * loss for weight, loss in zip(weights_loss, losses)
                ]

            total_loss.add(sum(losses).item())

            for i, loss_value in enumerate(losses):
                key = "loss_val", i
                losses_dict.setdefault(key, []).append(loss_value.item())

        losses_dict_mean = {k: np.mean(v) for k, v in losses_dict.items()}
        stats_val = {**losses_dict_mean}

        if self.metrics is not None:
            metrics = self._compute_metrics()
            stats_val.update(metrics)

        for key, value in stats_val.items():
            self.log.setdefault(key, []).append(value)

        if logger is not None:
            formatted_loss = ', '.join([
                f"{'_'.join(map(str, key))}:{value:.3f}"
                for key, value in stats_val.items()
                if key[0].startswith("loss")
            ])
            formatted_r2 = ', '.join([
                f"{'_'.join(map(str, key))}:{value:.3f}"
                for key, value in stats_val.items()
                if key[0].startswith("r2")
            ])
            logger.info(f"Val: {formatted_loss}")
            logger.info(f"Val: {formatted_r2}")

        # add sum_loss_valid
        sum_loss_valid = total_loss.average
        self.log.setdefault(("sum_loss_val",), []).append(sum_loss_valid)
        return stats_val


@register("supervised-solver-xcebra")
@dataclasses.dataclass
class SupervisedMultiobjectiveSolverxCEBRA(MultiobjectiveSolverBase):
    """Supervised neural network training using the MSE loss.

    This solver can be used as a baseline variant instead of the contrastive solver,
    :py:class:`cebra.solver.multiobjective.ContrastiveMultiobjectiveSolverxCEBRA`.
    """

    _variant_name = "supervised-solver-xcebra"

    def _inference(self, batch):
        """Compute predictions (discrete/continuous) for the batch."""
        pred_refs = self.model(batch.reference)
        prediction_batches = []
        for i, label_data in enumerate(batch.positive):
            prediction_batches.append(
                cebra.data.Batch(reference=pred_refs[i],
                                 positive=label_data,
                                 negative=None))
        return prediction_batches


@register("multiobjective-solver")
@dataclasses.dataclass
class ContrastiveMultiobjectiveSolverxCEBRA(MultiobjectiveSolverBase):
    """Multi-objective solver for CEBRA.

    This solver is used for training CEBRA models with multiple objectives.

    See Also:
        :py:class:`cebra.solver.multiobjective.SupervisedMultiobjectiveSolverxCEBRA`
        :py:class:`cebra.solver.multiobjective.MultiObjectiveConfig`
        :py:class:`cebra.models.multicriterions.MultiCriterions`
    """

    _variant_name = "contrastive-solver-xcebra"

    def _inference(self, batch: cebra.data.Batch) -> cebra.data.Batch:
        pred_refs = self.model(batch.reference)
        pred_negs = self.model(batch.negative)

        prediction_batches = []
        for i, positive in enumerate(batch.positive):
            pred_pos = self.model(positive)
            prediction_batches.append(
                cebra.data.Batch(pred_refs[i], pred_pos[i], pred_negs[i]))

        return prediction_batches


@register("unified-multiobjective-solver")
@dataclasses.dataclass
class UnifiedMultiobjectiveSolverxCEBRA(MultiobjectiveSolverBase):
    """Unified multi-objective solver for CEBRA.

    This solver is used for training CEBRA models with multiple objectives across
    multiple sessions using a single shared model. Designed to work with
    :py:class:`cebra.data.multiobjective.UnifiedContrastiveMultiObjectiveLoader`.

    Unlike the standard multiobjective solver which handles per-session models,
    this solver applies a single model to unified batches where reference, positive,
    and negative samples are already aligned across sessions.

    See Also:
        :py:class:`cebra.data.multiobjective.UnifiedContrastiveMultiObjectiveLoader`
        :py:class:`cebra.solver.multiobjective.MultiObjectiveConfig`
        :py:class:`cebra.models.multicriterions.MultiCriterions`
    """

    _variant_name = "unified-multiobjective-solver"

    def _set_fitted_params(self, loader: cebra.data.Loader):
        """Set parameters once the solver is fitted for unified multiobjective training.

        For unified training, num_sessions is the number of sessions in the unified dataset,
        and n_features is the total concatenated feature dimension across all sessions.

        Args:
            loader: Loader used to fit the solver.
        """
        self.num_sessions = loader.dataset.num_sessions
        self.n_features = loader.dataset.input_dimension

    def _check_is_inputs_valid(
        self,
        inputs: Optional[List[torch.Tensor]],
        session_id: Optional[int] = None
    ):
        """Check that the inputs can be inferred using the unified model.

        For unified training, inputs should be a list of per-session tensors,
        and the concatenated feature dimension must match the model's input dimension.

        Args:
            inputs: List of input tensors, one per session. Each has shape
                   (n_samples, num_features_session, ...).
            session_id: The target session ID (required for unified models).
        """
        if inputs is None:
            return

        if isinstance(inputs, list):
            inputs_shape = 0
            for i in range(len(inputs)):
                inputs_shape += inputs[i].shape[1]
        elif isinstance(inputs, torch.Tensor):
            raise NotImplementedError(
                "Unified models require a list of per-session tensors, not a single tensor."
            )
        else:
            raise NotImplementedError(
                f"Invalid input type: {type(inputs)}. Expected list of tensors."
            )

        if self.n_features != inputs_shape:
            raise ValueError(
                f"Invalid input shape: model requires an input of shape "
                f"(n_samples, {self.n_features}), but got total feature dimension "
                f"(n_samples, {inputs_shape}) from concatenated sessions."
            )

    def _check_is_session_id_valid(
        self,
        session_id: Optional[int] = None,
    ):
        """Check that the session ID provided is valid for the unified solver.

        The session ID must be non-null and between 0 and the number of sessions.

        Args:
            session_id: The session ID to check.
        """
        if session_id is None:
            raise RuntimeError(
                "No session_id provided: unified multiobjective model requires a session_id "
                "as the target session to use to align the sessions."
            )
        if session_id >= self.num_sessions or session_id < 0:
            raise RuntimeError(
                f"Invalid session_id {session_id}: session_id for the current unified model "
                f"must be between 0 and {self.num_sessions-1}."
            )

    def _get_model(self, session_id: Optional[int] = None):
        """Get the unified model (independent of session_id).

        For unified models, there is only one shared model regardless of session.
        The session_id is still validated but not used for model selection.

        Args:
            session_id: The session ID, required for validation.

        Returns:
            The shared unified model.
        """
        self._check_is_session_id_valid(session_id=session_id)
        self._check_is_fitted()
        return self.model

    def _inference(self, batch: cebra.data.Batch) -> cebra.data.Batch:
        """Compute predictions for unified multiobjective batch.

        For unified batches, reference, positive, and negative samples are already
        aligned across sessions (concatenated along the neuron dimension). We apply
        the single model to each and return per-objective prediction batches.

        Args:
            batch: A Batch with reference, positive (list), and negative tensors
                   already unified across sessions.

        Returns:
            A list of Batch objects, one per objective, with model predictions.
        """
        pred_refs = self.model(batch.reference)
        pred_negs = self.model(batch.negative)

        prediction_batches = []
        for i, positive in enumerate(batch.positive):
            pred_pos = self.model(positive)
            prediction_batches.append(
                cebra.data.Batch(
                    reference=pred_refs[i].unsqueeze(0).view(-1, pred_refs[i].shape[-1]),
                    positive=pred_pos[i].unsqueeze(0).view(-1, pred_pos[i].shape[-1]),
                    negative=pred_negs[i].unsqueeze(0).view(-1, pred_negs[i].shape[-1])
                ))

        return prediction_batches


    @torch.no_grad()
    def transform(self,
                  inputs: List[torch.Tensor],
                  labels: List[torch.Tensor],
                  pad_before_transform: bool = True,
                  session_id: Optional[int] = None,
                  batch_size: Optional[int] = 512,
                  split_outputs: bool = False) -> torch.Tensor:
        """Compute the embedding for the `session_id`th session of the dataset.

        For unified multiobjective training, the embedding is computed using the shared
        unified model, similar to UnifiedSolver. All sessions must be provided as they
        are used for sampling context across sessions.

        Args:
            inputs: The input signal for all sessions (list of tensors).
            labels: The auxiliary variables to use for sampling (list of tensors).
            pad_before_transform: Whether to pad the input before computing embeddings.
            session_id: The session ID for which to compute embeddings. Required for
                       multi-session models.
            batch_size: Batch size for computing embeddings (default 512).
            split_outputs: Whether to return per-objective outputs (True) or concatenated
                          output (False, default). If True, returns a tuple of embeddings
                          one per objective.

        Returns:
            If split_outputs=False: The output embedding for the session corresponding to
            the provided ID `session_id`, with shape (num_samples(session_id), output_dimension).
            If split_outputs=True: A tuple of embeddings, one per objective, with shapes
            (num_samples(session_id), output_dimension_objective_i).
        """
        if not isinstance(inputs, list):
            raise ValueError(
                f"Inputs to transform() should be a list, not {type(inputs)}.")

        self._check_is_fitted()

        if session_id is None:
            raise ValueError("Session ID is required for unified multiobjective models.")

        # Use the concrete model device (e.g. cuda:2), not a generic "cuda".
        transform_device_torch = next(self.model.parameters()).device
        transform_device = str(transform_device_torch)
        device_ctx = (
            torch.cuda.device(transform_device_torch)
            if transform_device_torch.type == "cuda"
            else nullcontext()
        )

        with device_ctx:
            # Build a unified dataset for sampling context
            dataset = cebra.data.UnifiedDataset(
                cebra.data.TensorDataset(
                    inputs[i], continuous=labels[i], offset=cebra.data.Offset(0, 1))
                for i in range(len(inputs))).to(transform_device)

            # Use UnifiedLoader for sampling
            loader = cebra.data.UnifiedLoader(dataset, num_steps=1).to(transform_device)

            # Set split_outputs to specified value
            self.model.set_split_outputs(split_outputs)

            # Compute embeddings in batches
            refs_data_batch_embeddings = []
            batch_range = range(0, len(dataset.get_session(session_id)), batch_size)
            if len(batch_range) < 2:
                raise ValueError(
                    "Not enough data to perform the batched transform. Please provide a larger dataset or reduce the batch size."
                )
            for batch_start in batch_range:
                batch_end = min(batch_start + batch_size,
                                len(dataset.get_session(session_id)))

                if batch_start == batch_range[-2]:  # one before last batch
                    last_start = batch_start
                    continue
                if batch_start == batch_range[-1]:  # last batch, likely incomplete
                    batch_start = last_start
                    batch_end = len(dataset.get_session(session_id))

                # Sample indices from all sessions aligned to the target session
                refs_idx_batch = loader.sampler.sample_all_sessions(
                    ref_idx=torch.arange(batch_start, batch_end),
                    session_id=session_id).to(transform_device)

                # Concatenate data across sessions
                refs_data_batch = torch.cat([
                    session[refs_idx_batch[s_id].cpu()]
                    for s_id, session in enumerate(dataset.iter_sessions())
                ], dim=1).squeeze()

                if len(self.model.get_offset()) < 2 and pad_before_transform:
                    pad_before_transform = False

                self.model.eval()
                # Get embedding with the specified split_outputs setting
                embedding = self._transform(model=self.model,
                                    inputs=refs_data_batch,
                                    pad_before_transform=pad_before_transform,
                                    offset=self.model.get_offset(),
                                    batch_size=batch_size)
                refs_data_batch_embeddings.append(embedding)


            # Reset split_outputs to True for subsequent training
            self.model.set_split_outputs(True)

            if split_outputs:
                # If split_outputs was True, we have tuples of embeddings
                # Stack them across batches per objective
                num_objectives = len(refs_data_batch_embeddings[0])
                return tuple(torch.cat([batch[i] for batch in refs_data_batch_embeddings], dim=0)
                            for i in range(num_objectives))
            else:
                # If split_outputs was False, we have concatenated tensors
                return torch.cat(refs_data_batch_embeddings, dim=0)


    def get_aligned_data(self,
                         inputs,
                         labels,
                         start: int,
                         end: int,
                         session_id: Optional[int] = None) -> torch.Tensor:
        """Get data aligned across sessions for the specified session ID and index range.
        
        Args:
            inputs: List of input tensors, one per session.
            labels: List of label tensors, one per session.
            start: Start index for the target session.
            end: End index for the target session.
            session_id: The target session ID to align to.
        
        Returns:
            A tensor of data aligned across sessions for the specified session ID and index range.
        """
        
        # Build a unified dataset for sampling context
        dataset = cebra.data.UnifiedDataset(
            cebra.data.TensorDataset(
                inputs[i], continuous=labels[i], offset=cebra.data.Offset(0, 1))
            for i in range(len(inputs))).to(self.device)

        # Use UnifiedLoader for sampling
        loader = cebra.data.UnifiedLoader(dataset, num_steps=1)

        # Sample indices from all sessions aligned to the target session
        refs_idx_batch = loader.sampler.sample_all_sessions(
            ref_idx=torch.arange(start, end),
            session_id=session_id).to(self.device)

        # Concatenate data across sessions
        refs_data_batch = torch.cat([
            session[refs_idx_batch[s_id].cpu()]
            for s_id, session in enumerate(dataset.iter_sessions())
        ], dim=1).squeeze()

        return refs_data_batch