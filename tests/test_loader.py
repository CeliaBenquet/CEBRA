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
import _util
import numpy as np
import pytest
import torch

import cebra.data
import cebra.io

BATCH_SIZE = 32
NUMS_NEURAL = [3, 4, 5]


class LoadSpeed:

    def __init__(self, loader):
        self.loader = loader

    def __call__(self):
        n = 0
        for batch in self.loader:
            n += 1
        assert batch.reference.device.type == self.loader.device
        assert n == len(self.loader)


class RandomDataset(cebra.data.SingleSessionDataset):

    def __init__(self, N=100, d=5, device="cpu"):
        super().__init__(device=device)
        self._cindex = torch.randint(0, 5, (N, d), device=device).float()
        self._dindex = torch.randint(0, 5, (N,), device=device).long()
        self.neural = self._data = torch.randn((N, d), device=device)

    @property
    def input_dimension(self):
        return self._data.shape[1]

    def __len__(self):
        return len(self._data)

    @property
    def continuous_index(self):
        return self._cindex

    @property
    def discrete_index(self):
        return self._dindex

    def __getitem__(self, index):
        return self._data[index]


def test_offset():
    offset = cebra.data.Offset(5, 4)
    assert offset.left == 5
    assert offset.right == 4
    assert offset.left_slice == slice(0, 5)
    assert len(offset) == 5 + 4

    offset = cebra.data.Offset(0, 4)
    assert offset.left == 0
    assert offset.right == 4
    assert offset.left_slice == slice(0, 0)
    assert len(offset) == 4

    offset = cebra.data.Offset(5)
    assert offset.left == 5
    assert offset.right == 5
    assert offset.left_slice == slice(0, 5)
    assert offset.right_slice == slice(-5, None)
    assert len(offset) == 5 * 2

    with pytest.raises(ValueError, match="Invalid.*right"):
        offset = cebra.data.Offset(5, 0)
    with pytest.raises(ValueError, match="Invalid.*right"):
        offset = cebra.data.Offset(0, 0)
    with pytest.raises(ValueError, match="Invalid.*number"):
        offset = cebra.data.Offset(5, 5, 5)
    with pytest.raises(ValueError, match="Invalid.*bounds"):
        offset = cebra.data.Offset(-2, 4)
    with pytest.raises(ValueError, match="Invalid.*bounds"):
        offset = cebra.data.Offset(4, -2)


def _assert_dataset_on_correct_device(loader, device):
    assert hasattr(loader, "dataset")
    assert hasattr(loader, "device")
    assert isinstance(loader.dataset, cebra.io.HasDevice)
    if isinstance(loader, cebra.data.SingleSessionDataset):
        assert loader.dataset.neural.device.type == device
    elif isinstance(loader, cebra.data.MultiSessionDataset):
        for session in loader.dataset.iter_sessions():
            assert session.neural.device.type == device


def test_demo_data():
    if not torch.cuda.is_available():
        pytest.skip("Test only possible with CUDA.")

    dataset = RandomDataset(N=100, device="cuda")
    assert dataset.neural.device.type == "cuda"
    dataset.to("cpu")
    assert dataset.neural.device.type == "cpu"


def _assert_device(first, second):

    def _to_str(val):
        if isinstance(val, torch.device):
            return val.type
        return val

    assert _to_str(first) == _to_str(second)


@_util.parametrize_device
@pytest.mark.parametrize(
    "data_name, loader_initfunc",
    [
        ("demo-discrete", cebra.data.DiscreteDataLoader),
        ("demo-continuous", cebra.data.ContinuousDataLoader),
        ("demo-mixed", cebra.data.MixedDataLoader),
        ("demo-continuous-multisession", cebra.data.MultiSessionLoader),
        ("demo-continuous-unified", cebra.data.UnifiedLoader),
    ],
)
def test_device(data_name, loader_initfunc, device):
    if not torch.cuda.is_available():
        pytest.skip("Test only possible with CUDA.")

    swap = {"cpu": "cuda", "cuda": "cpu"}
    other_device = swap.get(device)
    dataset = RandomDataset(N=100, device=other_device)

    loader = loader_initfunc(dataset, num_steps=10, batch_size=BATCH_SIZE)
    loader.to(device)
    assert loader.dataset == dataset
    _assert_device(loader.device, device)
    _assert_device(loader.dataset.device, device)

    _assert_device(loader.get_indices(10).reference.device, device)


@_util.parametrize_device
@pytest.mark.parametrize("prior", ("uniform", "empirical"))
def test_discrete(prior, device, benchmark):
    dataset = RandomDataset(N=100, device=device)
    loader = cebra.data.DiscreteDataLoader(
        dataset=dataset,
        num_steps=10,
        batch_size=8,
        prior=prior,
    )
    _assert_dataset_on_correct_device(loader, device)
    load_speed = LoadSpeed(loader)
    benchmark(load_speed)


@_util.parametrize_device
@pytest.mark.parametrize("conditional", ("time", "time_delta"))
def test_continuous(conditional, device, benchmark):
    dataset = RandomDataset(N=100, d=5, device=device)
    loader = cebra.data.ContinuousDataLoader(
        dataset=dataset,
        num_steps=10,
        batch_size=8,
        conditional=conditional,
    )
    _assert_dataset_on_correct_device(loader, device)
    load_speed = LoadSpeed(loader)
    benchmark(load_speed)


def _check_attributes(obj, is_list=False):
    if is_list:
        for obj_ in obj:
            _check_attributes(obj_, is_list=False)
    elif isinstance(obj, cebra.data.Batch) or isinstance(
            obj, cebra.data.BatchIndex):
        assert hasattr(obj, "positive")
        assert hasattr(obj, "negative")
        assert hasattr(obj, "reference")
    else:
        raise TypeError()


@_util.parametrize_device
@pytest.mark.parametrize(
    "data_name, loader_initfunc",
    [
        ("demo-discrete", cebra.data.DiscreteDataLoader),
        ("demo-continuous", cebra.data.ContinuousDataLoader),
        ("demo-mixed", cebra.data.MixedDataLoader),
    ],
)
def test_singlesession_loader(data_name, loader_initfunc, device):
    data = cebra.datasets.init(data_name)
    data.to(device)
    loader = loader_initfunc(data, num_steps=10, batch_size=BATCH_SIZE)
    _assert_dataset_on_correct_device(loader, device)

    index = loader.get_indices(100)
    _check_attributes(index)

    for batch in loader:
        _check_attributes(batch)
        assert len(batch.positive) == BATCH_SIZE


@_util.parametrize_device
@pytest.mark.parametrize(
    "data_name, loader_initfunc",
    [
        ("demo-continuous-multisession",
         cebra.data.ContinuousMultiSessionDataLoader),
        ("demo-discrete-multisession",
         cebra.data.DiscreteMultiSessionDataLoader),
    ],
)
def test_multisession_loader(data_name, loader_initfunc, device):
    data = cebra.datasets.init(data_name)
    data.to(device)
    loader = loader_initfunc(data, num_steps=10, batch_size=BATCH_SIZE)

    _assert_dataset_on_correct_device(loader, device)

    # Check the sampler
    assert hasattr(loader, "sampler")
    ref_idx = loader.sampler.sample_prior(1000)
    assert len(ref_idx) == len(NUMS_NEURAL)
    for session in range(len(NUMS_NEURAL)):
        assert ref_idx[session].max(
        ) < cebra.datasets.demo._DEFAULT_NUM_TIMEPOINTS
    pos_idx, idx, idx_rev = loader.sampler.sample_conditional(ref_idx)

    assert pos_idx is not None
    assert idx is not None
    assert idx_rev is not None

    batch = next(iter(loader))
    for i, n_neurons in enumerate(NUMS_NEURAL):
        assert batch[i].reference.shape == (BATCH_SIZE, n_neurons, 10)

    def _mix(array, idx):
        shape = array.shape
        n, m = shape[:2]
        mixed = array.reshape(n * m, -1)[idx]
        print(mixed.shape, array.shape, idx.shape)
        return mixed.reshape(shape)

    def _process(batch, feature_dim=1):
        """Given list_i[(N,d_i)] batch, return (#session, N, feature_dim) tensor"""
        return torch.stack(
            [b.reference.flatten(1).mean(dim=1, keepdims=True) for b in batch],
            dim=0).repeat(1, 1, feature_dim)

    dummy_prediction = _process(batch, feature_dim=6)
    assert dummy_prediction.shape == (3, BATCH_SIZE, 6)
    _mix(dummy_prediction, batch[0].index)

    index = loader.get_indices(100)
    #print(index[0])
    #print(type(index))
    _check_attributes(index, is_list=False)

    for batch in loader:
        _check_attributes(batch, is_list=True)
        for session_batch in batch:
            assert len(session_batch.positive) == BATCH_SIZE


@_util.parametrize_device
@pytest.mark.parametrize(
    "data_name, loader_initfunc",
    [
        ("demo-continuous-unified", cebra.data.UnifiedLoader),
    ],
)
def test_unified_loader(data_name, loader_initfunc, device):
    data = cebra.datasets.init(data_name)
    data.to(device)
    loader = loader_initfunc(data, num_steps=10, batch_size=BATCH_SIZE)

    _assert_dataset_on_correct_device(loader, device)

    # Check the sampler
    num_samples = 100
    assert hasattr(loader, "sampler")
    ref_idx = loader.sampler.sample_all_uniform_prior(num_samples)
    assert ref_idx.shape == (len(NUMS_NEURAL), num_samples)
    assert isinstance(ref_idx, np.ndarray)

    for session in range(len(NUMS_NEURAL)):
        assert ref_idx[session].max(
        ) < cebra.datasets.demo._DEFAULT_NUM_TIMEPOINTS
    pos_idx = loader.sampler.sample_conditional(ref_idx)
    assert pos_idx.shape == (len(NUMS_NEURAL), num_samples)

    for session in range(len(NUMS_NEURAL)):
        ref_idx = torch.from_numpy(
            loader.sampler.sample_all_uniform_prior(
                num_samples=num_samples)[session])
        assert ref_idx.shape == (num_samples,)
        all_ref_idx = loader.sampler.sample_all_sessions(ref_idx=ref_idx,
                                                         session_id=session)
        assert all_ref_idx.shape == (len(NUMS_NEURAL), num_samples)
        assert isinstance(all_ref_idx, torch.Tensor)
        for i in range(len(all_ref_idx)):
            assert all_ref_idx[i].max(
            ) < cebra.datasets.demo._DEFAULT_NUM_TIMEPOINTS

    for i in range(len(all_ref_idx)):
        pos_idx = loader.sampler.sample_conditional(all_ref_idx)
        assert pos_idx.shape == (len(NUMS_NEURAL), num_samples)

    # Check the batch
    batch = next(iter(loader))
    assert batch.reference.shape == (BATCH_SIZE, sum(NUMS_NEURAL), 10)
    assert batch.positive.shape == (BATCH_SIZE, sum(NUMS_NEURAL), 10)
    assert batch.negative.shape == (BATCH_SIZE, sum(NUMS_NEURAL), 10)

    index = loader.get_indices(100)
    _check_attributes(index, is_list=False)
