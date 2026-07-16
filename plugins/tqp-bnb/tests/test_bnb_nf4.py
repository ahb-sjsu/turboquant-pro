"""Conformance + semantics tests for the bnb_nf4 plugin (run in this
package's own CI; needs only turboquant-pro + numpy)."""

from __future__ import annotations

import numpy as np
import pytest
from tqp_bnb.plugin import SPEC_NF4, BnbNF4Quantizer

from turboquant_pro.plugin_conformance import assert_conformance
from turboquant_pro.plugins import Quantizer, register

RNG = np.random.default_rng(3)


def test_conformance_kv_block():
    q = BnbNF4Quantizer()
    x = RNG.standard_normal((1, 4, 96, 64)).astype(np.float32)
    report = assert_conformance(q, x, rel_err_max=0.30)
    assert report.results["affine"].startswith("pass"), report
    assert report.results["packed"].startswith("skip")


def test_block_granular_affine_matches_decompress():
    """Milestone 2: the (H, S, D) block-granular weight reproduces
    decompress exactly -- the gate that makes fused decode inheritable."""
    q = BnbNF4Quantizer(blocksize=64)
    x = RNG.standard_normal((1, 4, 96, 64)).astype(np.float32)
    c = q.compress(x)
    mu, w, grid = q.grid_params(c)
    assert mu.shape == (4, 64) and w.shape == (4, 96, 64)
    dense = mu[:, None, :] + w * grid[q.codes(c)[0]]
    assert np.allclose(dense[None], q.decompress(c), atol=1e-6)


def test_non_kv_shape_degrades():
    q = BnbNF4Quantizer()
    c = q.compress(RNG.standard_normal((128, 64)).astype(np.float32))
    assert q.grid_params(c) is None


def test_protocol_and_spec():
    assert isinstance(BnbNF4Quantizer(), Quantizer)
    with pytest.raises(ValueError, match="already registered"):
        register(SPEC_NF4)  # only if a prior import registered it
        register(SPEC_NF4)


def test_blockwise_semantics_exact_table_values():
    q = BnbNF4Quantizer(blocksize=8)
    # values exactly on the scaled table reconstruct exactly
    tab = np.asarray(
        [q._table[i] for i in [0, 3, 7, 8, 12, 15, 1, 14]], dtype=np.float32
    )
    x = (tab * 2.5).reshape(1, 1, 1, 8)
    got = q.decompress(q.compress(x))
    assert np.allclose(got, x, atol=1e-6)


def test_tail_padding_roundtrip():
    q = BnbNF4Quantizer(blocksize=64)
    x = RNG.standard_normal((1, 1, 1, 100)).astype(np.float32)  # not divisible
    got = q.decompress(q.compress(x))
    assert got.shape == x.shape and np.isfinite(got).all()


def test_crosscheck_against_bitsandbytes():
    bnb = pytest.importorskip("bitsandbytes")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("bnb 4-bit needs CUDA")
    x = RNG.standard_normal((256, 64)).astype(np.float32)
    qt, state = bnb.functional.quantize_4bit(
        torch.tensor(x, device="cuda"), blocksize=64, quant_type="nf4"
    )
    want = bnb.functional.dequantize_4bit(qt, state).cpu().numpy()
    got = BnbNF4Quantizer(blocksize=64).decompress(
        BnbNF4Quantizer(blocksize=64).compress(x)
    )
    assert np.allclose(got, want, atol=2e-3), float(np.abs(got - want).max())


# ------------------------------------------------------------------ #
# LLM.int8 adapter                                                    #
# ------------------------------------------------------------------ #


def _outlier_block(H=4, S=48, D=64, n_out=3):
    x = RNG.standard_normal((1, H, S, D)).astype(np.float32)
    for h in range(H):
        for d in RNG.choice(D, n_out, replace=False):
            x[0, h, :, d] *= 12.0  # emergent-feature column
    return x


def test_int8_conformance_full_affine_and_csr():
    from tqp_bnb.plugin import LLMInt8Quantizer

    q = LLMInt8Quantizer()
    report = assert_conformance(q, _outlier_block(), rel_err_max=0.30)
    assert report.results["affine"].startswith("pass"), report
    assert report.results["csr"] == "pass", report


def test_int8_outlier_channels_fp16_exact():
    from tqp_bnb.plugin import LLMInt8Quantizer

    q = LLMInt8Quantizer()
    x = _outlier_block()
    c = q.compress(x)
    got = q.decompress(c)
    mask = c.outlier_mask  # (H, D)
    for h, d in zip(*np.nonzero(mask)):
        assert np.allclose(got[0, h, :, d], x[0, h, :, d].astype(np.float16), atol=1e-3)
    assert mask.sum() >= 3


def test_int8_no_outliers_csr_none():
    from tqp_bnb.plugin import LLMInt8Quantizer

    q = LLMInt8Quantizer(outlier_threshold=1e9)
    c = q.compress(RNG.standard_normal((1, 2, 16, 32)).astype(np.float32))
    assert q.outlier_csr(c) is None
    assert not c.outlier_mask.any()
