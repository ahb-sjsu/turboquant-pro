# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Public-data replication of the heat-taper experiment (RESULTS_heat_taper.md).

Same pre-registered prediction and identical battery as
``heat_taper_experiment.py`` (which used a synthetic power-law corpus), run
on real sentence-encoder embeddings: **BGE-M3 (public model) over WikiText-2
(public text)**, 1024-d CLS embeddings, unnormalized -- the
concentrated-spectrum regime the prediction targets.

Usage:
    EMB=path/to/bge_wikitext.npz python benchmarks/heat_taper_public.py

The npz holds ``embeddings`` (n, 1024) float32 (see the generation recipe in
RESULTS_heat_taper.md). Emits the same table/JSON as the synthetic run.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from heat_taper_experiment import (  # noqa: E402
    BUDGETS,
    MAX_WIDTH,
    SEED,
    hard_schedule,
    quantize_schedule,
    recall_at_10,
    spearman_pairs,
    taper_schedule,
)

from turboquant_pro.pca import PCAMatryoshka  # noqa: E402
from turboquant_pro.rank_certificate import (  # noqa: E402
    certificate_from_embeddings,
)

N_QUERY = 500


def main():
    emb_path = os.environ.get("EMB", "bge_wikitext.npz")
    x = np.load(emb_path)["embeddings"].astype(np.float32)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(x))
    corpus, queries = x[perm[N_QUERY:]], x[perm[:N_QUERY]]
    dim = corpus.shape[1]
    print(f"corpus={corpus.shape} queries={queries.shape} from {emb_path}", flush=True)

    pca = PCAMatryoshka(input_dim=dim, output_dim=MAX_WIDTH)
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
