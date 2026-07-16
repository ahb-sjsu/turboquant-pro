"""Native fp8 passthrough vs the code-space oracle."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "float8_e4m3fn"):
    pytest.skip("torch lacks float8_e4m3fn", allow_module_level=True)

from tqp_trtllm.native import FP8NativeKV  # noqa: E402
from tqp_trtllm.plugin import FP8KVQuantizer  # noqa: E402

RNG = np.random.default_rng(5)


def _keys():
    off = RNG.uniform(-4, 4, size=(1, 4, 1, 64))
    return (off + RNG.standard_normal((1, 4, 96, 64))).astype(np.float32)


def test_native_matches_code_space_oracle():
    x = _keys()
    want = FP8KVQuantizer().decompress(FP8KVQuantizer().compress(x))
    got = FP8NativeKV().decompress(FP8NativeKV().compress(x)).cpu().numpy()
    # RNE cast vs nearest-table lookup: identical except exact ties
    assert np.allclose(got, want, atol=1e-6), float(np.abs(got - want).max())


def test_native_storage_is_half_of_fp16():
    x = _keys()
    c = FP8NativeKV().compress(x)
    assert c.data.dtype == torch.float8_e4m3fn
    fp16_bytes = x.size * 2
    assert c.nbytes() < 0.55 * fp16_bytes


def test_roundtrip_error_bounded():
    x = _keys()
    got = FP8NativeKV().decompress(FP8NativeKV().compress(x)).cpu().numpy()
    rel = np.linalg.norm(got - x) / np.linalg.norm(x)
    assert rel < 0.05  # fp8 with per-head scale is near-lossless on keys


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_native_on_cuda_device():
    x = torch.as_tensor(_keys()).cuda()
    c = FP8NativeKV().compress(x)
    assert c.data.is_cuda and c.data.dtype == torch.float8_e4m3fn
    out = FP8NativeKV().decompress(c)
    assert out.is_cuda
    assert torch.allclose(
        out,
        torch.as_tensor(
            FP8KVQuantizer().decompress(FP8KVQuantizer().compress(x.cpu().numpy()))
        ).cuda(),
        atol=1e-6,
    )
