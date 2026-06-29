"""
Tests for the KVQuant and KIVI competitor baselines added to the LongBench harness
(benchmarks/kvquant_matrix/tq_paper_lb_shard.py).

These run on CPU with synthetic tensors only -- the build box has torch-CPU but NO CUDA
and NO models, so the GPU acceptance gate (reproducing ~21.06 qasper on Llama-2-7B with
calibrate_kvquant) is NOT exercised here. We test the quantization math that backs the
baselines:

  1. Fisher-weighted ``_kmeans_1d`` beats uniform at equal levels, and weighting toward
     high-sensitivity samples lowers the error there.
  2. nearest-centroid roundtrip is idempotent and bounded.
  3. KIVI 2-bit asymmetric beats symmetric 2-bit on DC-offset data.
  4. ``qdq_key_block`` dispatches "kvquant" (with an injected fake calibration cache)
     and "kivi" on a synthetic key tensor without error and preserves shape.

Usage:
    python -m pytest tests/test_kvquant_kivi.py -q
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch

# ------------------------------------------------------------------ #
# Load the (non-package) harness module by file path.                #
# ------------------------------------------------------------------ #
_HARNESS = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "kvquant_matrix"
    / "tq_paper_lb_shard.py"
)


def _load_harness():
    os.environ.setdefault("SHARD_ID", "0")
    os.environ.setdefault("NUM_SHARDS", "1")
    os.environ.setdefault("TAG", "pytest")
    spec = importlib.util.spec_from_file_location("tq_paper_lb_shard", _HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tq = _load_harness()


def _uniform_recon(x: torch.Tensor, nlev: int) -> torch.Tensor:
    """Per-row asymmetric uniform quantize+dequant at ``nlev`` levels. x: (R, n)."""
    mn = x.amin(1, keepdim=True)
    mx = x.amax(1, keepdim=True)
    sc = (mx - mn).clamp_min(1e-8) / (nlev - 1)
    return ((x - mn) / sc).round().clamp(0, nlev - 1) * sc + mn


# ------------------------------------------------------------------ #
# 1. Fisher-weighted k-means                                          #
# ------------------------------------------------------------------ #


class TestWeightedKMeans:
    def test_kmeans_beats_uniform_at_equal_levels(self) -> None:
        """Unweighted k-means must achieve lower MSE than uniform at the same #levels on
        Gaussian-with-DC-offset channels (non-uniform codes pack the dense region)."""
        torch.manual_seed(0)
        R, n, nlev = 8, 600, 8
        offset = torch.arange(R, dtype=torch.float32).view(R, 1) * 3.0
        x = torch.randn(R, n) + offset  # per-row DC offset

        cent = tq._kmeans_1d(x, nlev, iters=12)
        km_recon = tq._dequant_to_centroids(x, cent)
        uni_recon = _uniform_recon(x, nlev)

        km_mse = ((x - km_recon) ** 2).mean().item()
        uni_mse = ((x - uni_recon) ** 2).mean().item()
        assert (
            km_mse < uni_mse
        ), f"kmeans {km_mse:.4f} should beat uniform {uni_mse:.4f}"

    def test_weighting_reduces_error_on_sensitive_samples(self) -> None:
        """A small cluster of high-sensitivity (high-Fisher-weight) samples should be
        reconstructed better when those samples are up-weighted in the Lloyd update.
        Compare WEIGHTED MSE of weighted vs unweighted k-means -> weighted wins."""
        torch.manual_seed(1)
        R, nlev, m = 4, 2, 200  # 2 codes, 3 equal clusters -> a code MUST be shared
        # Three equal dense clusters at -5, 0, +5. With only 2 codes plain k-means lumps
        # the middle (0) cluster in with a neighbour (error ~ 2.5 there). The middle is
        # the high-sensitivity region: up-weighting it pulls a code onto 0.
        lo = -5.0 + 0.05 * torch.randn(R, m)
        mid = 0.0 + 0.05 * torch.randn(R, m)  # the sensitive cluster
        hi = 5.0 + 0.05 * torch.randn(R, m)
        x = torch.cat([lo, mid, hi], dim=1)  # (R, 3m)

        w = torch.ones_like(x)
        w[:, m : 2 * m] = 50.0  # Fisher weight concentrated on the middle (sensitive)
        sens = mid  # alias used by the sensitive-region error check below

        cent_unw = tq._kmeans_1d(x, nlev, iters=20)  # ignores the tiny far cluster
        cent_w = tq._kmeans_1d(x, nlev, iters=20, w=w)  # allocates a code to it

        rec_unw = tq._dequant_to_centroids(x, cent_unw)
        rec_w = tq._dequant_to_centroids(x, cent_w)

        # Weighted MSE (the objective the sensitivity weighting actually targets).
        wmse_unw = (w * (x - rec_unw) ** 2).sum().item() / w.sum().item()
        wmse_w = (w * (x - rec_w) ** 2).sum().item() / w.sum().item()
        assert wmse_w < wmse_unw, f"weighted {wmse_w:.4f} should beat {wmse_unw:.4f}"

        # And specifically the error ON the sensitive (middle) samples must drop.
        err_unw = ((sens - rec_unw[:, m : 2 * m]) ** 2).mean().item()
        err_w = ((sens - rec_w[:, m : 2 * m]) ** 2).mean().item()
        assert err_w < err_unw

    def test_unweighted_path_unchanged(self) -> None:
        """w=None must reproduce the original Lloyd update bit-for-bit (additive)."""
        torch.manual_seed(2)
        x = torch.randn(5, 300)
        a = tq._kmeans_1d(x, 8, iters=10)
        b = tq._kmeans_1d(x, 8, iters=10, w=None)
        assert torch.equal(a, b)


# ------------------------------------------------------------------ #
# 2. Nearest-centroid roundtrip                                       #
# ------------------------------------------------------------------ #


class TestCentroidRoundtrip:
    def test_idempotent_and_bounded(self) -> None:
        torch.manual_seed(3)
        R, n, k = 6, 400, 8
        cent = torch.randn(R, k)
        x = torch.randn(R, n) * 2.0

        q1 = tq._dequant_to_centroids(x, cent)
        q2 = tq._dequant_to_centroids(
            q1, cent
        )  # re-quantizing centroid values is a no-op
        assert torch.allclose(q1, q2, atol=1e-6)

        lo = cent.amin(1, keepdim=True)
        hi = cent.amax(1, keepdim=True)
        assert (q1 >= lo - 1e-6).all() and (q1 <= hi + 1e-6).all()


# ------------------------------------------------------------------ #
# 3. KIVI asymmetric vs symmetric on DC-offset data                   #
# ------------------------------------------------------------------ #


class TestKiviAsymmetric:
    def test_asym_beats_sym_2bit_on_dc_offset(self) -> None:
        torch.manual_seed(4)
        B, H, n, D = 1, 2, 64, 16
        g = n  # single group spanning the block
        # Strong per-channel DC offset that symmetric abs-max wastes its range on.
        offset = (torch.arange(D, dtype=torch.float32) - D / 2.0) * 4.0
        x = torch.randn(B, H, n, D) + offset.view(1, 1, 1, D)

        asym = tq._quant_uniform_asym_group(x, bits=2, g=g)
        sym = tq._quant_uniform_sym_group(x, bits=2, g=g)

        mse_asym = ((x - asym) ** 2).mean().item()
        mse_sym = ((x - sym) ** 2).mean().item()
        assert mse_asym < mse_sym, f"asym {mse_asym:.4f} should beat sym {mse_sym:.4f}"


# ------------------------------------------------------------------ #
# 4. qdq_key_block dispatch smoke test                                #
# ------------------------------------------------------------------ #


class TestKeyBlockDispatch:
    @pytest.fixture(autouse=True)
    def _reset_globals(self):
        saved = {
            k: getattr(tq, k)
            for k in (
                "CODEBOOK",
                "KB",
                "G",
                "SINK",
                "OUT_FRAC",
                "_CUR_LAYER",
                "_KVQ_CACHE",
            )
        }
        yield
        for k, v in saved.items():
            setattr(tq, k, v)

    def test_dispatch_kvquant(self) -> None:
        B, H, n, D, nlev = 1, 2, 32, 4, 8
        x = torch.randn(B, H, n, D)
        # Inject a fake calibration cache: per-channel (H*D) centroids.
        cent = torch.sort(torch.randn(H * D, nlev), dim=1).values
        tq._KVQ_CACHE = {0: cent}
        tq._CUR_LAYER = 0
        tq.CODEBOOK = "kvquant"
        tq.SINK = 0
        tq.OUT_FRAC = 0.0

        out = tq.qdq_key_block(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_dispatch_kvquant_with_outliers(self) -> None:
        """OUT_FRAC outlier path (dense-sparse fp16) must still preserve shape."""
        B, H, n, D, nlev = 1, 2, 32, 4, 8
        x = torch.randn(B, H, n, D)
        cent = torch.sort(torch.randn(H * D, nlev), dim=1).values
        tq._KVQ_CACHE = {0: cent}
        tq._CUR_LAYER = 0
        tq.CODEBOOK = "kvquant"
        tq.SINK = 0
        tq.OUT_FRAC = 0.1

        out = tq.qdq_key_block(x)
        assert out.shape == x.shape

    def test_dispatch_kivi(self) -> None:
        B, H, n, D = 1, 2, 48, 16
        x = torch.randn(B, H, n, D)
        tq.CODEBOOK = "kivi"
        tq.KB = 2
        tq.G = 32
        tq.SINK = 0
        tq.OUT_FRAC = 0.0

        out = tq.qdq_key_block(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
