# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""
Heat-taper vs hard truncation: a pre-registered theory test.

PREDICTION (registered before the first run, from Theorem `thm:heat` of the
companion paper, https://github.com/ahb-sjsu/the-angular-observer): hard
truncation at ``output_dim`` is the *plain-filter* move, and plain-filter
distances equalize as dimensions accumulate; exponential tail suppression is
what buys uniformity-in-truncation. Therefore, AT MATCHED TOTAL BITS, a soft
exponential taper (a full-precision head plus a tail of dimensions at
exponentially decaying bit width) should beat hard truncation on rank
stability (pairwise-distance Spearman, recall@10, certified tau floor)
across operating points, most visibly at small budgets -- flattening the
cliff that makes ``suggest_output_dim`` sensitive. If hard truncation ties,
that is a real boundary on the heat-filter analogy and gets documented as
such (either outcome is informative).

Protocol (scheme-level, CPU, deterministic):
  * synthetic corpus with a concentrated power-law spectrum
    (lambda_j ~ j^-1.2, the sentence-encoder regime where PCA-Matryoshka is
    the right tool per RESULTS_glove.md);
  * one PCA basis fit at max width; each config is a per-dimension bit
    schedule on the PCA coordinates (0 bits = dropped dim), quantized with
    a uniform per-dim min/max grid, reconstructed via inverse_transform;
  * matched total bits per vector across configs at every budget;
  * metrics vs exact original-space cosine: recall@10 (single-stage),
    pairwise-distance Spearman over 200 anchors, and the distribution-free
    rank certificate (kappa, tau floor) -- the floor from
    turboquant_pro.rank_certificate, dogfooding the shipped theorem.

Run:  python benchmarks/heat_taper_experiment.py
Emits a table and JSON between ###RESULTS_JSON_START###/END### markers.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from turboquant_pro.pca import PCAMatryoshka  # noqa: E402
from turboquant_pro.rank_certificate import (  # noqa: E402
    certificate_from_embeddings,
)

N_CORPUS, N_QUERY, DIM = 5000, 500, 1024
MAX_WIDTH = 512
BUDGETS = [512, 768, 1024, 1536]  # total bits per vector
ANCHORS = 200
SEED = 42


def make_corpus(n: int, dim: int, alpha: float = 1.2, seed: int = SEED):
    """Gaussian data with power-law covariance spectrum lambda_j ~ j^-alpha."""
    rng = np.random.default_rng(seed)
    lam = (np.arange(1, dim + 1, dtype=np.float64)) ** (-alpha)
    g = rng.standard_normal((n, dim))
    return (g * np.sqrt(lam)[None, :]).astype(np.float32)


def taper_schedule(budget: int, head_bits: int = 4, tail_fracs=(0.5, 0.3, 0.2)):
    """Exponentially decaying tail: head @ head_bits, then 3/2/1-bit bands.

    The tail bands hold ``tail_fracs`` of the head's dimension count at
    3, 2, 1 bits (a geometric bit decay); the head size is solved so the
    total equals ``budget`` bits.
    """
    # head*4 + head*(0.5*3 + 0.3*2 + 0.2*1) = budget
    tail_cost = sum(f * b for f, b in zip(tail_fracs, (3, 2, 1)))
    head = int(round(budget / (head_bits + tail_cost)))
    bits = [head_bits] * head
    for frac, b in zip(tail_fracs, (3, 2, 1)):
        bits += [b] * int(round(head * frac))
    # trim/pad 1-bit dims to hit the budget exactly
    while sum(bits) > budget:
        bits.pop()
    while sum(bits) < budget:
        bits.append(1)
    return bits[:MAX_WIDTH]


def hard_schedule(budget: int, bits: int):
    """Hard truncation: budget/bits dims at uniform precision."""
    return [bits] * min(budget // bits, MAX_WIDTH)


def quantize_schedule(z: np.ndarray, bits: list[int], lo: np.ndarray, hi: np.ndarray):
    """Per-dim uniform min/max quantization of PCA coords under a schedule."""
    k = len(bits)
    zq = np.zeros_like(z)
    for j, b in enumerate(bits):
        if b <= 0:
            continue
        levels = 2**b
        step = max((hi[j] - lo[j]) / (levels - 1), 1e-30)
        codes = np.clip(np.round((z[:, j] - lo[j]) / step), 0, levels - 1)
        zq[:, j] = codes * step + lo[j]
    zq[:, k:] = 0.0
    return zq


def recall_at_10(exact_scores: np.ndarray, approx_scores: np.ndarray) -> float:
    gt = np.argsort(-exact_scores, axis=1)[:, :10]
    ap = np.argsort(-approx_scores, axis=1)[:, :10]
    return float(np.mean([len(set(gt[q]) & set(ap[q])) / 10 for q in range(len(gt))]))


def spearman_pairs(exact: np.ndarray, recon: np.ndarray, seed: int = SEED) -> float:
    rng = np.random.default_rng(seed)
    anc = rng.choice(len(exact), size=ANCHORS, replace=False)
    iu = np.triu_indices(ANCHORS, k=1)

    def cosd(x):
        u = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)
        return (1.0 - u @ u.T)[iu]

    a, b = cosd(exact[anc].astype(np.float64)), cosd(recon[anc].astype(np.float64))
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    x = make_corpus(N_CORPUS + N_QUERY, DIM)
    corpus, queries = x[:N_CORPUS], x[N_CORPUS:]

    pca = PCAMatryoshka(input_dim=DIM, output_dim=MAX_WIDTH)
    pca.fit(corpus)
    zc = np.asarray(pca.transform(corpus), dtype=np.float64)
    lo, hi = zc.min(axis=0), zc.max(axis=0)

    qn = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-30)
    cn = corpus / np.maximum(np.linalg.norm(corpus, axis=1, keepdims=True), 1e-30)
    exact_scores = qn @ cn.T

    rows = []
    for budget in BUDGETS:
        configs = {
            "hard@4bit": hard_schedule(budget, 4),
            "hard@3bit": hard_schedule(budget, 3),
            "taper(4/3/2/1)": taper_schedule(budget, 4),
        }
        for name, bits in configs.items():
            zq = quantize_schedule(zc, bits, lo, hi)
            recon = np.asarray(pca.inverse_transform(zq), dtype=np.float32)
            rn = recon / np.maximum(np.linalg.norm(recon, axis=1, keepdims=True), 1e-30)
            approx_scores = qn @ rn.T
            cert = certificate_from_embeddings(corpus, recon, seed=SEED)
            row = dict(
                budget_bits=budget,
                config=name,
                n_dims=len(bits),
                total_bits=int(sum(bits)),
                recall_at_10=round(recall_at_10(exact_scores, approx_scores), 4),
                spearman=round(spearman_pairs(corpus, recon), 4),
                kappa=round(cert.kappa, 3),
                tau_floor=round(cert.tau_floor, 4),
            )
            rows.append(row)
            print(
                f"B={budget:5d}  {name:15s} dims={row['n_dims']:3d} "
                f"recall@10={row['recall_at_10']:.4f} "
                f"spearman={row['spearman']:.4f} "
                f"kappa={row['kappa']:.2f} tau_floor={row['tau_floor']:.3f}",
                flush=True,
            )

    # Verdict per budget: taper vs best hard config
    print("\nVerdict (taper - best hard, per budget):")
    verdicts = []
    for budget in BUDGETS:
        sub = [r for r in rows if r["budget_bits"] == budget]
        taper = next(r for r in sub if r["config"].startswith("taper"))
        hard = max(
            (r for r in sub if r["config"].startswith("hard")),
            key=lambda r: r["recall_at_10"],
        )
        d_rec = taper["recall_at_10"] - hard["recall_at_10"]
        d_sp = taper["spearman"] - hard["spearman"]
        verdicts.append(
            dict(budget=budget, d_recall=round(d_rec, 4), d_spearman=round(d_sp, 4))
        )
        print(
            f"B={budget:5d}  d_recall={d_rec:+.4f}  d_spearman={d_sp:+.4f}  "
            f"({'taper' if d_rec > 0 else 'hard'} wins recall)",
            flush=True,
        )

    print("###RESULTS_JSON_START###")
    print(json.dumps(dict(rows=rows, verdicts=verdicts), indent=1))
    print("###RESULTS_JSON_END###")


if __name__ == "__main__":
    main()
