"""Conformance + semantics for the code-space FP8/NVFP4 KV plugins."""

from __future__ import annotations

import numpy as np
import pytest
from tqp_trtllm.plugin import _E2M1, _E4M3, FP8KVQuantizer, NVFP4KVQuantizer

from turboquant_pro.plugin_conformance import assert_conformance
from turboquant_pro.plugins import native_dtype

RNG = np.random.default_rng(9)


def _keys():
    off = RNG.uniform(-4, 4, size=(1, 4, 1, 64))
    return (off + RNG.standard_normal((1, 4, 96, 64))).astype(np.float32)


def test_e4m3_table_sane():
    assert len(_E4M3) == 253 and np.all(np.diff(_E4M3) > 0)
    assert _E4M3[-1] == 448.0 and _E4M3[0] == -448.0  # e4m3fn max magnitude


def test_fp8_conformance_affine_pass():
    report = assert_conformance(FP8KVQuantizer(), _keys(), rel_err_max=0.30)
    assert report.results["affine"].startswith("pass"), report


def test_nvfp4_conformance_block_granular_pass():
    report = assert_conformance(NVFP4KVQuantizer(), _keys(), rel_err_max=0.40)
    assert report.results["affine"].startswith("pass"), report


def test_nvfp4_grid_and_blocks():
    q = NVFP4KVQuantizer()
    c = q.compress(_keys())
    mu, w, grid = q.grid_params(c)
    assert w.shape == (4, 96, 64) and np.array_equal(grid, _E2M1)
    x = (_E2M1[[8, 10, 14, 7] * 4] * 2.0).reshape(1, 1, 1, 16)
    got = q.decompress(q.compress(x))
    assert np.allclose(got, x, atol=1e-6)


def test_native_dtype_declared():
    assert native_dtype(FP8KVQuantizer()) == "float8_e4m3fn"
    assert native_dtype(NVFP4KVQuantizer()) is None


def test_ml_dtypes_crosscheck():
    ml = pytest.importorskip("ml_dtypes")
    all_vals = np.arange(256, dtype=np.uint8).view(ml.float8_e4m3fn).astype(np.float32)
    finite = np.unique(all_vals[np.isfinite(all_vals)])
    assert np.allclose(finite, np.unique(_E4M3))
