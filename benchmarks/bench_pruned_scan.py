"""Pruned two-pass ADC scan vs the v2 kernel on real embeddings (docs/PREREG_pruned_scan.md).

    python bench_pruned_scan.py --data-root /data --out /data/kbench/pruned_scan.json

For each registered configuration (arm, d', bits):
  1. calibration: build the index (seed 0), search the calibration queries with v2 and with
     every grid point (prefix fraction x z); pick the grid point with the largest speedup
     whose recall@k against v2 is >= CAL_RECALL (ties: larger z, then larger prefix);
  2. evaluation: for seeds 0, 1, 2 rebuild the index, search the held-out queries with v2 and
     with the chosen point; record recall@k against v2, survivor fraction, and paired wall time
     (minimum of TIMING_REPS repetitions, same process, same threads).
Both k = 10 and k = 50 are measured; calibration is done separately for each k.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np

GRID_PREFIX = (0.125, 0.25, 0.5)
GRID_Z = (2.0, 3.0, 4.0)
SEEDS = (0, 1, 2)
KS = (10, 50)
CAL_RECALL = 0.999
TIMING_REPS = 3
CONFIGS = (
    # arm, corpus parts, dim_out, bits
    ("wiki1024-1m", "wiki1024", 1, 1024, 2),
    ("wiki1024-1m", "wiki1024", 1, 1024, 4),
    ("wiki1024-1m", "wiki1024", 1, 256, 4),
    ("dbpedia-3large-1536", "dbpedia3072", 1, 1536, 2),
    ("dbpedia-3large-1536", "dbpedia3072", 1, 1536, 4),
    ("dbpedia-3large-1536", "dbpedia3072", 1, 384, 4),
    ("wiki1024-5m", "wiki1024", 5, 1024, 4),
)


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)


def load_arm(root, folder, parts):
    """Corpus parts, calibration queries (500) and held-out queries (1000), all disjoint."""
    corpus = [
        np.load(f"{root}/{folder}/part_{i:03d}.npy", mmap_mode="r")
        for i in range(parts)
    ]
    if folder == "wiki1024":
        extra = np.load(f"{root}/{folder}/part_{parts:03d}.npy", mmap_mode="r")
        cal, ev = extra[:500], extra[500:1500]
    else:  # dbpedia: corpus part_000 is 990k rows; queries.npy holds 1000 held-out rows
        cal = corpus[0][-500:]
        corpus = [corpus[0][:-500]]
        ev = np.load(f"{root}/{folder}/queries.npy")
    return (
        corpus,
        normalize(np.asarray(cal, np.float32)),
        normalize(np.asarray(ev, np.float32)),
    )


def build(corpus, dim_out, bits, seed):
    from turboquant_pro import ADCIndex, PCAMatryoshka

    dim = corpus[0].shape[1]
    rng = np.random.default_rng(seed)
    n0 = len(corpus[0])
    fit = normalize(
        np.asarray(
            corpus[0][np.sort(rng.choice(n0, min(n0, 100_000), replace=False))],
            np.float32,
        )
    )
    pca = PCAMatryoshka(input_dim=dim, output_dim=dim_out)
    pca.fit(fit)
    pipe = pca.with_quantizer(bits=bits, seed=seed)
    n = sum(len(p) for p in corpus)
    codes = np.empty((n, dim_out), np.uint8)
    cnorm = np.empty(n, np.float32)
    vrnorm = np.empty(n, np.float32)
    pos = 0
    for part in corpus:
        for s in range(0, len(part), 250_000):
            blk = normalize(np.asarray(part[s : s + 250_000], np.float32))
            sub = ADCIndex(pipe).add(blk)
            e = pos + len(blk)
            codes[pos:e], cnorm[pos:e], vrnorm[pos:e] = (
                sub._codes,
                sub._cnorm,
                sub._vrnorm,
            )
            pos = e
    ix = ADCIndex(pipe)
    ix._codes, ix._cnorm, ix._vrnorm = codes, cnorm, vrnorm
    return ix


def timed(fn):
    best, out = float("inf"), None
    for _ in range(TIMING_REPS):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


def recall_vs(ref, got, k):
    return float(
        np.mean(
            [len(np.intersect1d(ref[i, :k], got[i, :k])) / k for i in range(len(ref))]
        )
    )


def run_config(ix, queries, k, prefix, z):
    t_v2, (ids_v2, _) = timed(lambda: ix.search(queries, k=k))
    t_pr, (ids_pr, _) = timed(lambda: ix.search(queries, k=k, prune=(prefix, z)))
    return dict(
        prefix=prefix,
        z=z,
        recall=round(recall_vs(ids_v2, ids_pr, k), 5),
        survivor_fraction=round(float(np.mean(ix.last_survivors)) / ix.size, 5),
        t_v2=round(t_v2, 3),
        t_pruned=round(t_pr, 3),
        speedup=round(t_v2 / t_pr, 3),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", nargs="*", help="indexes into CONFIGS (default: all)")
    a = ap.parse_args()
    from turboquant_pro import _adc

    assert _adc.is_available() and hasattr(_adc.load(), "search_pruned")
    results = []
    chosen = [CONFIGS[int(i)] for i in a.configs] if a.configs else list(CONFIGS)
    for arm, folder, parts, dim_out, bits in chosen:
        corpus, cal, ev = load_arm(a.data_root, folder, parts)
        rec = dict(
            arm=arm,
            dim_out=dim_out,
            bits=bits,
            n=sum(len(p) for p in corpus),
            calibration={},
            evaluation={},
        )
        ix = build(corpus, dim_out, bits, SEEDS[0])
        for k in KS:
            grid = [run_config(ix, cal, k, p, z) for p in GRID_PREFIX for z in GRID_Z]
            ok = [g for g in grid if g["recall"] >= CAL_RECALL]
            pick = (
                max(ok, key=lambda g: (g["speedup"], g["z"], g["prefix"]))
                if ok
                else None
            )
            rec["calibration"][str(k)] = dict(grid=grid, chosen=pick)
            print(arm, dim_out, bits, "k", k, "chosen", pick, flush=True)
        del ix  # one index at a time: holding the calibration index while building the next
        for seed in SEEDS:
            ix_s = build(corpus, dim_out, bits, seed)
            for k in KS:
                pick = rec["calibration"][str(k)]["chosen"]
                if pick is None:
                    continue
                r = run_config(ix_s, ev, k, pick["prefix"], pick["z"])
                rec["evaluation"].setdefault(str(k), []).append(dict(seed=seed, **r))
                print(arm, dim_out, bits, "seed", seed, "k", k, r, flush=True)
            del ix_s
        results.append(rec)
        with open(a.out + ".tmp", "w") as f:
            json.dump(
                dict(
                    env=dict(
                        host=platform.node(), threads=os.environ.get("OMP_NUM_THREADS")
                    ),
                    results=results,
                ),
                f,
                indent=1,
            )
        os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
