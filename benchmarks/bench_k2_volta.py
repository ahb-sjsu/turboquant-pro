# TurboQuant Pro: benchmark for the Volta K2 fused key-score kernel.
# Copyright (c) 2026 Andrew H. Bond. MIT License.
"""Benchmark the K2 fused per-channel key-score decode on a Volta GV100.

Compares the fused kernel (:func:`turboquant_pro.volta_kernels.k2_key_scores`)
against two baselines that both touch an ``(H,S,D)`` fp32 intermediate:

  * ``decompress+attend`` -- rebuild K then ``einsum(q, K)``
  * ``einsum(codes)``     -- the current :mod:`kv_fused_pck` path,
                            ``einsum(w, grid[codes])`` (materializes ``grid[codes]``)

The kernel touches only the 1-byte codes. Run on an uncontended GV100 for
meaningful numbers::

    CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_k2_volta.py

Measured on a Quadro GV100 (uncontended): ~12--20x over decompress+attend and
~5x over einsum(codes) at ``S in {4k, 8k}``, exact to fp32 rounding.
"""

from __future__ import annotations

import time

import numpy as np

from turboquant_pro.per_channel_kv import _NF4
from turboquant_pro.volta_kernels import k2_key_scores

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None


def _bench_case(H, S, D, reps=100):
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 16, size=(H, S, D), dtype=np.uint8)
    q = rng.standard_normal((H, D)).astype(np.float32)
    weight = (0.5 + rng.random((H, D))).astype(np.float32)
    mu = (0.1 * rng.standard_normal((H, D))).astype(np.float32)
    w = (q * weight).astype(np.float32)
    bias = (q * mu).sum(1).astype(np.float32)

    ref = bias[:, None] + np.einsum("hd,hsd->hs", w, _NF4[codes])

    d_codes, d_w, d_bias = cp.asarray(codes), cp.asarray(w), cp.asarray(bias)
    d_grid, d_q = cp.asarray(_NF4), cp.asarray(q)
    d_weight, d_mu = cp.asarray(weight), cp.asarray(mu)

    err = float(
        cp.abs(k2_key_scores(d_codes, d_w, d_bias, d_grid) - cp.asarray(ref)).max()
    )
    print(f"[correct] H={H} S={S} D={D}  kernel max abs err = {err:.2e}")

    def kernel():
        k2_key_scores(d_codes, d_w, d_bias, d_grid)

    def decompress_attend():
        K = d_mu[:, None, :] + d_weight[:, None, :] * d_grid[d_codes]
        cp.einsum("hd,hsd->hs", d_q, K)

    def einsum_codes():
        cp.einsum(
            "hd,hsd->hs", d_w, d_grid[d_codes]
        )  # noqa: F841 (+bias omitted; timing)

    results = {}
    for name, fn in [
        ("kernel", kernel),
        ("decompress+attend", decompress_attend),
        ("einsum(codes)", einsum_codes),
    ]:
        fn()
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        cp.cuda.Stream.null.synchronize()
        ms = (time.perf_counter() - t0) / reps * 1e3
        results[name] = ms
        gbps = H * S * D / (ms * 1e-3) / 1e9
        print(f"  {name:20s} {ms:8.3f} ms   (codes {gbps:6.1f} GB/s)")
    base = results["decompress+attend"]
    print(
        f"  -> kernel speedup: {base / results['kernel']:.1f}x vs decompress+attend, "
        f"{results['einsum(codes)'] / results['kernel']:.1f}x vs einsum(codes)"
    )


def main():
    if cp is None:
        raise SystemExit("this benchmark needs CuPy on a CUDA device")
    print("gpu", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
    for H, S, D in [(8, 4096, 128), (32, 4096, 128), (32, 8192, 128)]:
        _bench_case(H, S, D)
        print()


if __name__ == "__main__":
    main()
