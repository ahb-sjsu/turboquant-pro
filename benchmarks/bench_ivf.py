# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""IVF coarse-partition benchmark: recall@k vs fraction of the corpus scanned.

Shows the point of the layer — high recall while touching a small fraction of the
rows — and that the adaptive A*-style stop (nprobe=None) matches a well-chosen fixed
nprobe without having to pick one. Ground truth is the brute-force ADC top-k (the
ranking IVF is trying to reproduce), so recall==1.0 means "identical to scanning
everything".

    python benchmarks/bench_ivf.py --n 200000 --dim 64 --out-dim 32 --bits 4
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from turboquant_pro import ADCIndex, IVFIndex, PCAMatryoshka


def _corpus(n, dim, rng):
    rank = dim // 2
    basis = rng.standard_normal((rank, dim))
    coeffs = rng.standard_normal((n, rank)) * np.linspace(1.0, 0.3, rank)
    return (coeffs @ basis + 0.05 * rng.standard_normal((n, dim))).astype(np.float32)


def _recall(got, ref, k):
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(got[:, :k], ref)]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="IVF recall vs scan-fraction benchmark.")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--out-dim", type=int, default=32)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--nlist", type=int, default=None)
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(0)
    corpus = _corpus(args.n, args.dim, rng)
    q = corpus[rng.choice(args.n, size=args.queries, replace=False)]

    # Ground truth = brute-force ADC top-k over the whole corpus.
    pca = PCAMatryoshka(input_dim=args.dim, output_dim=args.out_dim)
    pca.fit(corpus[:200_000])
    adc = ADCIndex(pca.with_quantizer(bits=args.bits)).add(corpus)
    t0 = time.perf_counter()
    ref, _ = adc.search(q, k=args.k)
    brute_qps = round(len(q) / (time.perf_counter() - t0), 1)

    ivf = IVFIndex.create(
        corpus, output_dim=args.out_dim, bits=args.bits, nlist=args.nlist
    )
    st = ivf.stats()
    print(json.dumps({"phase": "build", "n": args.n, **st, "brute_qps": brute_qps}))

    def run(label, **kw):
        ivf.search(q[:8], k=args.k, **kw)  # warm
        t0 = time.perf_counter()
        ids, _, stats = ivf.search(q, k=args.k, return_stats=True, **kw)
        dt = time.perf_counter() - t0
        frac = float(np.mean([s.scan_fraction for s in stats]))
        cells = float(np.mean([s.cells_probed for s in stats]))
        print(
            json.dumps(
                {
                    "mode": label,
                    f"recall_at_{args.k}": round(_recall(ids, ref, args.k), 4),
                    "scan_fraction": round(frac, 4),
                    "cells_probed": round(cells, 1),
                    "nlist": st["nlist"],
                    "speedup_vs_brute": round(1.0 / max(frac, 1e-9), 1),
                    "qps": round(len(q) / dt, 1),
                }
            )
        )

    for nprobe in (1, 2, 4, 8, 16, 32):
        if nprobe <= st["nlist"]:
            run(f"fixed nprobe={nprobe}", nprobe=nprobe)
    for beta in (0.75, 0.5, 0.25):
        run(
            f"adaptive weighted beta={beta}",
            nprobe=None,
            bound="weighted",
            radius_scale=beta,
        )
    run("adaptive admissible (exact)", nprobe=None, bound="admissible")


if __name__ == "__main__":
    raise SystemExit(main())
