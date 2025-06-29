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
"""Single session solvers embed a single pair of time series."""

import copy
from typing import Optional

import literate_dataclasses as dataclasses
import torch

import cebra
import cebra.data
import cebra.models
import cebra.solver.base as abc_
from cebra.solver import register


@register("single-session")
class SingleSessionSolver(abc_.Solver):
    """Single session training with a symmetric encoder.

    This solver assumes that reference, positive and negative samples
    are processed by the same features encoder and that a single session
    is provided to that encoder.
    """

    _variant_name = "single-session"

    def _set_fitted_params(self, loader: cebra.data.Loader):
        """Set parameters once the solver is fitted.

        In single session solver, the number of session is set to None and the number of
        features is set to the number of neurons in the dataset.

        Args:
            loader: Loader used to fit the solver.
        """
        #self.num_sessions = None
        self.n_features = loader.dataset.input_dimension

    def _check_is_inputs_valid(self, inputs: torch.Tensor, session_id: int):
        """Check that the inputs can be inferred using the selected model.

        Note: This method checks that the number of neurons in the input is
        similar to the input dimension to the selected model.

        Args:
            inputs: Data to infer using the selected model.
            session_id: The session ID, an :py:class:`int` between 0 and
                the number of sessions -1 for multisession, and set to
                ``None`` for single session.
        """
        super()._check_is_inputs_valid(inputs, session_id=session_id)
        if self.n_features != inputs.shape[1]:
            raise ValueError(
                f"Invalid input shape: model for session {session_id} requires an input of shape"
                f"(n_samples, {self.n_features}), got (n_samples, {inputs.shape[1]})."
            )

    def _check_is_session_id_valid(self, session_id: Optional[int] = None):
        """Check that the session ID provided is valid for the solver instance.

        The session ID must be null or equal to 0.

        Args:
            session_id: The session ID to check.
        """

        if session_id is not None and session_id > 0:
            raise RuntimeError(
                f"Invalid session_id {session_id}: single session models only takes an optional null session_id."
            )

    def _get_model(self, session_id: Optional[int] = None):
        """Get the model for the given session ID.

        Args:
            session_id: The session ID, an :py:class:`int` between 0 and
                the number of sessions -1 for multisession, and set to
                ``None`` for single session.

        Returns:
            The model for the given session ID.
        """
        self._check_is_session_id_valid(session_id=session_id)
        self._check_is_fitted()
        return self.model

    def _inference(self, batch: cebra.data.Batch) -> cebra.data.Batch:
        """Given a batch of input examples, computes the feature representation/embedding.

        Args:
            batch: The input data, not necessarily aligned across the batch
                dimension. This means that ``batch.index`` specifies the map
                between reference/positive samples, if not equal ``None``.

        Returns:
            Processed batch of data. While the input data might not be aligned
            across the sample dimensions, the output data should be aligned and
            ``batch.index`` should be set to ``None``.
        """
        batch.to(self.device)
        ref = self.model(batch.reference)
        pos = self.model(batch.positive)
        neg = self.model(batch.negative)
        return cebra.data.Batch(ref, pos, neg)

    def get_embedding(self, data: torch.Tensor) -> torch.Tensor:
        """Return the embedding of the given input data.

        Note:
            This function assumes that the input data is sliced
            according to the receptive field of the model. The input data
            needs to match ``batch x dims x len(self.model.get_offset())``
            which is internally reduced to ``batch x dims x 1``. The last
            dimension is squeezed, and the output is of shape ``time x features``.

        This function does *not* perform checks for correctness of the
        input.

        Args:
            data: The input data tensor of shape `batch_time x dims x time`

        Returns:
            The output data tensor of shape `batch_time x features`.

        TODO:
            - Check if implementing checks in this function would downgrade
              speed during training/inference.
        """
        return self.model(data)[0].T


@register("single-session-aux")
@dataclasses.dataclass
class SingleSessionAuxVariableSolver(SingleSessionSolver,
                                     abc_.AuxiliaryVariableSolver):
    """Single session training for reference and positive/negative samples.

    This solver processes reference samples with a model different from
    processing the positive and
    negative samples. Requires that the ``reference_model`` is initialized
    to be different from the ``model`` used to process the positive and
    negative samples.

    Besides using an asymmetric encoder for the same modality, this solver
    also allows for e.g. time-contrastive learning across modalities, by
    using a reference model on modality A, and a different model processing
    the signal from modality B.
    """

    _variant_name = "single-session-aux"
    reference_model: torch.nn.Module = None

    def __post_init__(self):
        super().__post_init__()
        if self.reference_model is None:
            # NOTE(stes): This should work, according to this thread
            # https://discuss.pytorch.org/t/can-i-deepcopy-a-model/52192/19
            # and create a true copy of the model.
            self.reference_model = copy.deepcopy(self.model)
            self.reference_model.to(self.model.device)

    def _get_model(self,
                   session_id: Optional[int] = None,
                   use_reference_model: bool = False):
        """Get the model for the given session ID.

        Args:
            session_id: The session ID, an :py:class:`int` between 0 and
                the number of sessions -1 for multisession, and set to
                ``None`` for single session.

        Returns:
            The model for the given session ID.
        """
        self._check_is_session_id_valid(session_id=session_id)
        self._check_is_fitted()
        if use_reference_model:
            model = self.reference_model[session_id]
        else:
            model = self.model[session_id]
        return model

    def _inference(self, batch: cebra.data.Batch) -> cebra.data.Batch:
        """Given a batch of input examples, computes the feature representation/embedding.

        The reference samples are processed with a different model than the
        positive and negative samples.

        Args:
            batch: The input data, not necessarily aligned across the batch
                dimension. This means that ``batch.index`` specifies the map
                between reference/positive samples, if not equal ``None``.

        Returns:
            Processed batch of data. While the input data might not be aligned
            across the sample dimensions, the output data should be aligned and
            ``batch.index`` should be set to ``None``.
        """
        batch.to(self.device)
        ref = self.reference_model(batch.reference)
        pos = self.model(batch.positive)
        neg = self.model(batch.negative)
        return cebra.data.Batch(ref, pos, neg)


@register("single-session-hybrid")
@dataclasses.dataclass
class SingleSessionHybridSolver(abc_.MultiobjectiveSolver, SingleSessionSolver):
    """Single session training, contrasting neural data against behavior."""

    _variant_name = "single-session-hybrid"

    def _inference(self, batch: cebra.data.Batch) -> cebra.data.Batch:
        """Given a batch of input examples, computes the feature representation/embedding.

        The samples are processed with both a time-contrastive module and a
        behavior-contrastive module, that are part of the same model.

        Args:
            batch: The input data, not necessarily aligned across the batch
                dimension. This means that ``batch.index`` specifies the map
                between reference/positive samples, if not equal ``None``.

        Returns:
            Processed batch of data. While the input data might not be aligned
            across the sample dimensions, the output data should be aligned and
            ``batch.index`` should be set to ``None``.
        """
        batch.to(self.device)
        behavior_ref = self.model(batch.reference)[0]
        behavior_pos = self.model(batch.positive[:int(len(batch.positive) //
                                                      2)])[0]
        behavior_neg = self.model(batch.negative)[0]
        time_pos = self.model(batch.positive[int(len(batch.positive) // 2):])[1]
        time_ref = self.model(batch.reference)[1]
        time_neg = self.model(batch.negative)[1]
        return cebra.data.Batch(behavior_ref, behavior_pos,
                                behavior_neg), cebra.data.Batch(
                                    time_ref, time_pos, time_neg)

    def _get_model(self, session_id: Optional[int] = None):
        """Get the model for the given session ID.

        Args:
            session_id: The session ID, an :py:class:`int` between 0 and
                the number of sessions -1 for multisession, and set to
                ``None`` for single session.

        Returns:
            The model for the given session ID.
        """
        self._check_is_session_id_valid(session_id=session_id)
        self._check_is_fitted()
        return self.model.module


@register("single-session-full")
@dataclasses.dataclass
class BatchSingleSessionSolver(SingleSessionSolver):
    """Optimize a model with batch gradient descent.

    Usage of this solver requires a sufficient amount of GPU memory. Using this solver
    is equivalent to using a single session solver with batch size set to dataset size,
    but requires less computation.
    """

    def fit(self, loader, *args, **kwargs):
        """TODO"""
        self.offset = loader.dataset.offset
        self.neural = loader.dataset.neural.T[None]
        if isinstance(self.model, cebra.models.ConvolutionalModelMixin):
            if self.offset is None:
                raise ValueError("Configure dataset, no offset found.")
            self._mode = "convolutional"
        else:
            self.offset = cebra.data.Offset(0, 1)
            self._mode = "fully_connected"
        super().fit(loader, *args, **kwargs)

    def get_embedding(self, data):
        """Compute the embedding of a full input dataset.

        For convolutional models that implement
        :py:class:`cebra.models.model.ConvolutionalModelMixin`),
        the embedding is computed via
        :py:meth:`.SingleSessionSolver.get_embedding`.

        For all other models, it is assumed that the data has shape
        ``(1, dim, time)`` and is transformed into ``(time, dim)``
        format.

        Args:
            data: The input data

        Returns:
            The output embedding of shape ``(time, dimension)``

        See Also:
            * :py:class:`cebra.models.model.ConvolutionalModelMixin`)
            * :py:meth:`.SingleSessionSolver.get_embedding`
        """
        if self._mode == "convolutional":
            return super().get_embedding(data)
        else:
            # data has shape (1, d, T)
            # output needs to be (T, d)
            return self.model(data[0].T)

    def _inference(self, batch: cebra.data.Batch) -> cebra.data.Batch:
        """Given a batch of input examples, computes the feature representation/embedding.

        Args:
            batch: The input data, not necessarily aligned across the batch
                dimension. This means that ``batch.index`` specifies the map
                between reference/positive samples, if not equal ``None``.

        Returns:
            Processed batch of data. While the input data might not be aligned
            across the sample dimensions, the output data should be aligned and
            ``batch.index`` should be set to ``None``.
        """
        outputs = self.get_embedding(self.neural)
        idc = batch.positive - self.offset.left >= len(outputs)
        batch.positive[idc] = batch.reference[idc]

        return cebra.data.Batch(
            outputs[batch.reference - self.offset.left],
            outputs[batch.positive - self.offset.left],
            outputs[batch.negative - self.offset.left],
        )
