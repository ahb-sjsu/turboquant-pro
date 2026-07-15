# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Public-data replication of the hubness quotient (RESULTS_hubness_local_scaling.md).

The synthetic dissociation established: quotient density only when it is
nuisance for the task truth. On public embeddings the retrieval ground truth
is defined on the OBSERVED vectors (exact original-space cosine), so the
dissociation itself predicts a plain density quotient must lose here -- the
data's density is part of the truth. The refined public-data hypothesis is
therefore about the density component that IS nuisance by construction:
**the compression-induced density shift**. Aggressive quantization moves
per-vector density (mu_recon != mu_orig, creating/erasing hubs the original
space never had), and only that shift should be corrected:

    score'_i = score_i - (mu_recon_i - mu_orig_i)

Both densities are computable at index build (originals are in hand then);
nothing extra is needed at query time. Variants (identical candidate fetch):

  * baseline        -- raw ADC scores;
  * plain centered  -- score - mu_recon (the synthetic regime-2 control:
                       predicted to LOSE, per the dissociation);
  * delta centered  -- score - (mu_recon - mu_orig) (predicted >= baseline);
  * +rerank x5      -- exact rerank reference.

Data: BGE-M3 (public model) over WikiText-2 (public text), 1024-d
unnormalized CLS embeddings; PCA-128 + 2-bit TurboQuant in ADCIndex.

Usage:
    EMB=path/to/bge_wikitext.npz python benchmarks/hubness_public.py
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

N_QUERY, PCA_DIM, BITS = 500, 128, 2
K, OVERSAMPLE, DENSITY_SAMPLE = 10, 5, 256
SEED = 42


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
    emb_path = os.environ.get("EMB", "bge_wikitext.npz")
    x = np.load(emb_path)["embeddings"].astype(np.float32)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(x))
    corpus, queries = unit(x[perm[N_QUERY:]]), unit(x[perm[:N_QUERY]])
    n = len(corpus)
    print(f"corpus={corpus.shape} queries={queries.shape} from {emb_path}", flush=True)

    pca = PCAMatryoshka(input_dim=corpus.shape[1], output_dim=PCA_DIM)
    pca.fit(corpus)
    tq = TurboQuantPGVector(dim=PCA_DIM, bits=BITS, seed=SEED)
    pipeline = PCAMatryoshkaPipeline(pca=pca, quantizer=tq)
    idx = ADCIndex(pipeline).add(corpus)

    gt = np.argsort(-(queries @ corpus.T), axis=1)[:, :K]  # observed-space GT

    # Density in both spaces, same anchor sample (computable at build time).
    recon = unit(np.asarray(pipeline.decompress_batch(pipeline.compress_batch(corpus))))
    sample_ids = rng.choice(n, size=DENSITY_SAMPLE, replace=False)
    mu_recon = (recon @ recon[sample_ids].T).mean(axis=1)
    mu_orig = (corpus @ corpus[sample_ids].T).mean(axis=1)
    delta = mu_recon - mu_orig
    print(
        f"density shift |delta|: mean={np.abs(delta).mean():.4f} "
        f"max={np.abs(delta).max():.4f}",
        flush=True,
    )

    cand_idx, cand_scores = idx.search(queries, k=K * OVERSAMPLE, rerank=0)

    def top10(corrected: np.ndarray) -> np.ndarray:
        order = np.argsort(-corrected, axis=1)[:, :K]
        return np.take_along_axis(cand_idx, order, axis=1)

    variants = {
        "baseline (raw ADC)": top10(cand_scores),
        "plain centered (- mu_recon)": top10(cand_scores - mu_recon[cand_idx]),
        "delta centered (- (mu_r - mu_o))": top10(cand_scores - delta[cand_idx]),
    }
    variants[f"+rerank x{OVERSAMPLE} (exact)"] = idx.search(
        queries, k=K, rerank=OVERSAMPLE, originals=corpus
    )

    rows = []
    for name, lists in variants.items():
        occ = occupancy_stats(lists, n)
        row = dict(variant=name, recall_at_10=round(recall(lists, gt), 4), **occ)
        rows.append(row)
        print(
            f"{name:34s} recall@10={row['recall_at_10']:.4f}  "
            f"max_occ={row['max_occupancy']:4d}  skew={row['skewness']:6.2f}",
            flush=True,
        )

    base = rows[0]["recall_at_10"]
    print(
        f"\nVerdict: plain {rows[1]['recall_at_10'] - base:+.4f} "
        f"(dissociation predicts <= 0), "
        f"delta {rows[2]['recall_at_10'] - base:+.4f} "
        f"(hypothesis predicts >= 0); rerank ref {rows[3]['recall_at_10']:.4f}",
        flush=True,
    )

    print("###RESULTS_JSON_START###")
    print(
        json.dumps(
            dict(
                config=dict(n=n, pca_dim=PCA_DIM, bits=BITS, oversample=OVERSAMPLE),
                density_shift_mean_abs=round(float(np.abs(delta).mean()), 4),
                rows=rows,
            ),
            indent=1,
        )
    )
    print("###RESULTS_JSON_END###")


if __name__ == "__main__":
    main()
