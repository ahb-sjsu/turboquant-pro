"""use_gpu=True must fall back to NumPy when no CUDA device is visible.

An importable CuPy is not enough to take the GPU path. With CuPy installed and
no visible device (no GPU, or CUDA_VISIBLE_DEVICES=""), the old check
``use_gpu and _HAS_CUPY`` chose the GPU path and failed with cudaErrorNoDevice.
"""

import numpy as np

import turboquant_pro.core as core
from turboquant_pro.core import TurboQuantKV
from turboquant_pro.cuda_kernels import cuda_device_available


def test_cuda_device_available_is_a_bool() -> None:
    assert isinstance(cuda_device_available(), bool)


def test_use_gpu_follows_device_availability() -> None:
    tq = TurboQuantKV(head_dim=64, n_heads=1, bits=3, use_gpu=True, seed=0)
    assert tq._gpu == cuda_device_available()


def test_no_device_falls_back_to_numpy(monkeypatch) -> None:
    monkeypatch.setattr(core, "cuda_device_available", lambda: False)
    tq = TurboQuantKV(head_dim=64, n_heads=1, bits=3, use_gpu=True, seed=0)
    assert tq._gpu is False
    x = np.random.default_rng(0).standard_normal((1, 1, 3, 64)).astype(np.float32)
    assert tq.decompress(tq.compress(x)).shape == x.shape
