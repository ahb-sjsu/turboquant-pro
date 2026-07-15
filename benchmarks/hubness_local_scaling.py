# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""
Hubness quotient for ADCIndex: is the alpha=1 lesson also the practical fix?

Motivation (companion paper, section `Radius is density, angle is geometry';
its moral-probe measured rho(radius, kNN in-degree) = 0.58): in embedding kNN
spaces the radius/density coordinate is largely hubness -- density, not
meaning -- and the Coifman-Lafon alpha=1 normalization that quotients density
is the same operation that isolates geometry. The retrieval analogue is
LOCAL-SCALING SCORE CORRECTION: estimate a per-vector density proxy at index
build (mean/std of similarity to a fixed sample of the corpus, computable
from the compressed reconstructions) and correct candidate scores by it. Hub
vectors flooding candidate lists is a known single-stage recall killer at
high compression; this benchmark row answers whether the principled fix is
also the practical one.

Design note (a first version of this experiment was discarded for a
construction error worth recording): if ground truth is defined by cosine on
the OBSERVED anisotropic vectors, the hubs are part of the truth and any
hubness correction must lose by construction. The honest design mirrors the
real situation (embedding anisotropy / "all-but-the-top"): the semantic
signal is the centered part, the shared mean direction is nuisance, and the
hubness that nuisance induces is genuinely spurious relative to semantics.

Protocol (deterministic, CPU): corpus = signal (power-law spectrum,
lambda_j ~ j^-1.2) + a common mean offset (the nuisance); GROUND TRUTH =
top-10 by cosine in SIGNAL space; retrieval sees only the observed
(offset-carrying, L2-normalized) vectors through PCA-128 + 2-bit TurboQuant
(aggressive operating point) in ADCIndex. Variants share the identical
candidate fetch (top 10x5 by raw ADC score) and differ only in candidate
scoring:

  * baseline    -- raw ADC scores;
  * centered    -- score_i - mu_i  (mu_i = mean sim of corpus item i to a
                   fixed 256-sample of reconstructed corpus vectors);
  * zscored     -- (score_i - mu_i) / s_i (local scaling proper);
  * +rerank x5  -- exact rerank on observed vectors (the production
                   protocol; note it is exact in OBSERVED space, so it
                   cannot remove the nuisance the quotient targets).

Also reports hubness itself: the k-occurrence distribution of corpus items
across all query top-10 lists (max occupancy and Fisher skewness), with and
without the quotient.

Run:  python benchmarks/hubness_local_scaling.py
Emits a table and JSON between ###RESULTS_JSON_START###/END### markers.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from turboquant_pro.adc_index import ADCIndex  # noqa: E402
from turboquant_pro.pca import PCAMatryoshka, PCAMatryoshkaPipeline  # noqa: E402
from turboquant_pro.pgvector import TurboQuantPGVector  # noqa: E402

N_CORPUS, N_QUERY, DIM = 5000, 500, 1024
PCA_DIM, BITS = 128, 2
K, OVERSAMPLE = 10, 5
DENSITY_SAMPLE = 256
SEED = 42


def make_corpus(n: int, dim: int, alpha: float = 1.2, seed: int = SEED):
    """Signal (power-law spectrum) + common mean offset (the nuisance).

    Returns (observed, signal): retrieval sees ``observed``; ground truth is
    cosine in ``signal`` space (semantics before anisotropy).
    """
    rng = np.random.default_rng(seed)
    lam = (np.arange(1, dim + 1, dtype=np.float64)) ** (-alpha)
    signal = rng.standard_normal((n, dim)) * np.sqrt(lam)[None, :]
    mean = rng.standard_normal(dim) * np.sqrt(lam.mean()) * 3.0
    observed = signal + mean[None, :]
    return observed.astype(np.float32), signal.astype(np.float32)


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)


def occupancy_stats(top_lists: np.ndarray, n_corpus: int) -> dict:
    occ = np.bincount(top_lists.ravel(), minlength=n_corpus).astype(np.float64)
    mu, sd = occ.mean(), occ.std()
    skew = float(((occ - mu) ** 3).mean() / max(sd, 1e-30) ** 3)
    return {"max_occupancy": int(occ.max()), "skewness": round(skew, 2)}


def recall(top_lists: np.ndarray, gt: np.ndarray) -> float:
    return float(
        np.mean([len(set(gt[q]) & set(top_lists[q])) / K for q in range(len(gt))])
    )


def main():
    observed, signal = make_corpus(N_CORPUS + N_QUERY, DIM)
    corpus, queries = unit(observed[:N_CORPUS]), unit(observed[N_CORPUS:])
    sig_c, sig_q = signal[:N_CORPUS], signal[N_CORPUS:]

    pca = PCAMatryoshka(input_dim=DIM, output_dim=PCA_DIM)
    pca.fit(corpus)
    tq = TurboQuantPGVector(dim=PCA_DIM, bits=BITS, seed=SEED)
    pipeline = PCAMatryoshkaPipeline(pca=pca, quantizer=tq)
    idx = ADCIndex(pipeline).add(corpus)

    # Two ground truths = the two regimes of the density dissociation:
    #   signal   -- cosine in SIGNAL space (nuisance-mean regime: the
    #               anisotropy is NOT part of the truth);
    #   observed -- cosine on the observed vectors (density-is-truth
    #               regime: hubs are genuine neighbors).
    # L2-normalizing observed vectors makes rerank's inner product equal to
    # observed-space cosine, the standard protocol.
    gts = {
        "signal (mean = nuisance)": np.argsort(-(unit(sig_q) @ unit(sig_c).T), axis=1)[
            :, :K
        ],
        "observed (mean = truth)": np.argsort(-(queries @ corpus.T), axis=1)[:, :K],
    }

    # Per-vector density proxy from the compressed-domain reconstructions.
    recon = unit(np.asarray(pipeline.decompress_batch(pipeline.compress_batch(corpus))))
    rng = np.random.default_rng(SEED)
    sample = recon[rng.choice(N_CORPUS, size=DENSITY_SAMPLE, replace=False)]
    sims = recon @ sample.T
    mu_i = sims.mean(axis=1)
    s_i = np.maximum(sims.std(axis=1), 1e-6)

    # One shared candidate fetch for all single-stage variants.
    cand_idx, cand_scores = idx.search(queries, k=K * OVERSAMPLE, rerank=0)

    def top10(corrected: np.ndarray) -> np.ndarray:
        order = np.argsort(-corrected, axis=1)[:, :K]
        return np.take_along_axis(cand_idx, order, axis=1)

    variants = {
        "baseline (raw ADC)": top10(cand_scores),
        "centered (score - mu_i)": top10(cand_scores - mu_i[cand_idx]),
        "zscored ((score-mu)/s)": top10((cand_scores - mu_i[cand_idx]) / s_i[cand_idx]),
    }
    # rerank>0 returns indices only (see ADCIndex.search docstring)
    rr_idx = idx.search(queries, k=K, rerank=OVERSAMPLE, originals=corpus)
    variants[f"+rerank x{OVERSAMPLE} (exact)"] = rr_idx

    rows = []
    for gt_name, gt in gts.items():
        print(f"\n-- ground truth: {gt_name} --", flush=True)
        block = []
        for name, lists in variants.items():
            occ = occupancy_stats(lists, N_CORPUS)
            row = dict(
                ground_truth=gt_name,
                variant=name,
                recall_at_10=round(recall(lists, gt), 4),
                **occ,
            )
            block.append(row)
            print(
                f"{name:26s} recall@10={row['recall_at_10']:.4f}  "
                f"max_occ={row['max_occupancy']:4d}  skew={row['skewness']:6.2f}",
                flush=True,
            )
        base = block[0]["recall_at_10"]
        best = max(block[1:3], key=lambda r: r["recall_at_10"])
        print(
            f"verdict: best quotient ({best['variant']}) vs baseline: "
            f"{best['recall_at_10'] - base:+.4f} recall@10 single-stage "
            f"(rerank reference: {block[3]['recall_at_10']:.4f})",
            flush=True,
        )
        rows += block

    print("###RESULTS_JSON_START###")
    print(
        json.dumps(
            dict(
                config=dict(
                    n=N_CORPUS,
                    dim=DIM,
                    pca_dim=PCA_DIM,
                    bits=BITS,
                    oversample=OVERSAMPLE,
                ),
                rows=rows,
            ),
            indent=1,
        )
    )
    print("###RESULTS_JSON_END###")


if __name__ == "__main__":
    main()
