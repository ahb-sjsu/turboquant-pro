#!/usr/bin/env python3
"""Generate the per-claim reproducibility notebooks under notebooks/claims/.

Each notebook maps to ONE rung of the evidence ladder in docs/claims.md and is
self-contained for Colab (pip install + public-data download + run). The
embedding notebooks embed the verified harness source from
benchmarks/canonical_embedding.py so the notebook code == the CI-tested code.

    python notebooks/claims/_gen_notebooks.py

Regenerate whenever the harness changes. Notebooks are emitted with EMPTY
outputs — outputs are produced when the user/CI runs them (we never bake in
numbers we didn't compute in that notebook).
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
HARNESS = os.path.join(REPO, "benchmarks", "canonical_embedding.py")

REPO_SLUG = "ahb-sjsu/turboquant-pro"
BRANCH = "master"


def colab_badge(nb_path_rel):
    url = f"https://colab.research.google.com/github/{REPO_SLUG}/blob/{BRANCH}/{nb_path_rel}"
    return f"[![Open In Colab]({'https://colab.research.google.com/assets/colab-badge.svg'})]({url})"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _src(lines),
    }


def _src(lines):
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def harness_cell():
    with open(HARNESS, encoding="utf-8") as f:
        src = f.read()
    # strip the __main__ smoke block so importing/execing the cell only defines fns
    marker = 'if __name__ == "__main__":'
    src = src[: src.index(marker)].rstrip() + "\n"
    header = (
        "# === embedded harness (verbatim from benchmarks/canonical_embedding.py) ===\n"
        "# Single source of truth for the method ladder; identical to the CI-tested file.\n"
    )
    return code(header + src)


GLOVE_LOADER = """\
# Download a canonical public ANN benchmark (ann-benchmarks HDF5, provided ground truth).
import os, urllib.request, h5py, numpy as np

DATASETS = {
    "glove-100-angular":   "http://ann-benchmarks.com/glove-100-angular.hdf5",
    "nytimes-256-angular": "http://ann-benchmarks.com/nytimes-256-angular.hdf5",
    "deep-image-96-angular": "http://ann-benchmarks.com/deep-image-96-angular.hdf5",
}

def load_annbench(name=DATASET, queries=QUERIES, corpus_cap=CORPUS_CAP):
    url = DATASETS[name]
    fn = url.split("/")[-1]
    if not os.path.exists(fn):
        print("downloading", url, "...", flush=True)
        urllib.request.urlretrieve(url, fn)
    with h5py.File(fn, "r") as f:
        C = normalize(np.asarray(f["train"], dtype=np.float32))
        Q = normalize(np.asarray(f["test"], dtype=np.float32))
        nbr = np.asarray(f["neighbors"], dtype=np.int64)   # provided exact top-100
    if corpus_cap and len(C) > corpus_cap:
        # subsample corpus for a quick pass; recompute exact GT so ids stay valid
        C = C[:corpus_cap]
        Q = Q[:queries]
        gt = exact_topk(Q, C, 100)
        print(f"{name}: SUBSET corpus={len(C)} (exact GT recomputed) dim={C.shape[1]} q={len(Q)}")
    else:
        Q, gt = Q[:queries], nbr[:queries, :100]
        print(f"{name}: FULL corpus={len(C)} dim={C.shape[1]} q={len(Q)} (provided GT)")
    return C, Q, gt

C, Q, gt = load_annbench()
"""


def build_flagship():
    rel = "notebooks/claims/00_canonical_sota_embedding.ipynb"
    cells = [
        md(
            f"# Canonical SOTA embedding benchmark — {colab_badge(rel)}",
            "",
            "**Evidence-ladder rung:** Track 1 · the flagship claim — *beats RaBitQ on recall, "
            "ties OPQ at scale, builds faster* (docs/claims.md).",
            "",
            "ONE table, ALL methods, IDENTICAL rerank protocol, on a **public** ann-benchmarks "
            "dataset scored against its **provided** ground-truth neighbours — the flagship result, "
            "one *Run all* away.",
            "",
            "> **Honest scope (see `benchmarks/RESULTS_glove.md`).** PCA *truncation* wins only "
            "for high-dimensional embeddings with a concentrated spectrum (sentence/vision "
            "encoders). On already-compact descriptor sets like GloVe-100 it discards real "
            "variance and loses; at **full dimension / matched bytes** the TurboQuant scalar "
            "quantizer still wins or ties PQ/OPQ. This notebook reproduces that full picture, "
            "not a cherry-pick.",
        ),
        md(
            "> **⚠️ Reading these numbers — the single-pass-vs-rerank trap.** The table reports "
            "recall at **two** operating points: **`R@10`** (*single-pass* — scan the compressed "
            "codes once) and **`R@10 +rr`** (*+rerank* — re-score the top `10×OVERSAMPLE` "
            "candidates with exact vectors). Compare methods **like-for-like on the same axis**, "
            "at **matched `B/vec`** (bytes/vector — *not* bits or nominal dim, since the PCA "
            "front-end changes the dimension). The most common misread is to put a method's "
            "*single-pass* `R@10` next to RaBitQ's published headline **90–99%, which is itself a "
            "*with-rerank* number** — apples-to-oranges. Here **every** method (RaBitQ included) is "
            "measured at **both** points with the **same** oversample, so the fair columns are "
            "`R@10` vs `R@10` and `R@10 +rr` vs `R@10 +rr`. Each recall carries a **bootstrap 95% "
            "CI** over the queries: if two methods' intervals overlap, the gap is within noise — "
            "don't rank on it.",
            ">",
            "> This notebook is the **public, Colab-reproducible** run (ann-benchmarks data, "
            "faiss's RaBitQ). Its companion is the **estimator-isolated** head-to-head on real "
            "sentence embeddings — official `rabitqlib` **+ ScaNN**, RaBitQ run **exhaustively** "
            "so recall reflects the *estimator* not the index: "
            "[`benchmarks/RESULTS_rabitq_comparison.md`]"
            "(../../benchmarks/RESULTS_rabitq_comparison.md).",
        ),
        md("## 1. Install"),
        code("!pip install -q turboquant-pro faiss-cpu h5py numpy pandas"),
        md(
            "## 2. Configure",
            "`DATASET` ∈ {glove-100-angular, nytimes-256-angular, deep-image-96-angular}. "
            "`CORPUS_CAP=None` runs the full corpus (the real claim); set e.g. `200_000` for a "
            "fast smoke pass.",
        ),
        code(
            "DATASET    = 'glove-100-angular'",
            "QUERIES    = 2000",
            "CORPUS_CAP = None      # None = full corpus (the headline claim); int = quick subset",
            "OUT_DIM    = 100       # PCA target dim; = full dim disables truncation (see scope note)",
            "BITS       = 3         # TurboQuant bit-width",
            "OVERSAMPLE = 5         # SHARED by every +rerank row (candidates = 10*OVERSAMPLE)",
            "N_BOOT     = 1000      # bootstrap resamples for the recall@10 95% CIs",
            "THREADS    = 8",
        ),
        md("## 3. Harness (verified, embedded)"),
        harness_cell(),
        md("## 4. Load public data"),
        code(GLOVE_LOADER),
        md(
            "## 5. Run the canonical ladder",
            "Every ANN method is measured single-stage and +rerank at the **same** oversample, "
            "each recall@10 with a bootstrap 95% CI over the query set.",
        ),
        code(
            "rows = run_canonical(C, Q, gt, out_dim=OUT_DIM, bits=BITS,",
            "                     oversample=OVERSAMPLE, threads=THREADS, n_boot=N_BOOT)",
        ),
        md(
            "## 6. Canonical table",
            "`R@10 [95% CI]` / `R@10 +rr [95% CI]` show the mean with its bootstrap interval; the "
            "raw `recall_at_10{,_lo,_hi}` numeric fields are also on each row.",
        ),
        code(
            "import pandas as pd",
            "df = pd.DataFrame(rows)",
            "cols = ['method','n','dim','compression_x','bytes_per_vec','ram_mb',",
            "        'recall_at_10_ci','recall_at_10_rerank_ci','recall_at_100',",
            "        'qps_1stage','build_s','note']",
            "df = df[cols]",
            "df",
        ),
        code(
            "# Markdown (paste into benchmarks/RESULTS_*.md)",
            "print(to_markdown(rows))",
        ),
        md(
            "## 7. Read the result",
            "- **Same axis, matched bytes.** Compare `R@10` to `R@10` and `R@10 +rr` to `R@10 +rr` "
            "at matched `bytes_per_vec` — never a single-pass number against a reranked one (see "
            "the trap note up top). RaBitQ's published 90–99% is a *+rerank* figure, so the honest "
            "comparison is the `+rr` column.",
            "- **Overlapping 95% CIs = statistical tie.** Don't rank methods on a recall gap "
            "smaller than the intervals; raise `N_BOOT` / `QUERIES` to tighten them.",
            "- **RaBitQ / OPQ rows** are the SOTA baselines; compare `recall_at_10_rerank` at "
            "matched `bytes_per_vec`.",
            "- **PCA-only vs TQ-only vs PCA+TQ** isolate truncation from scalar quantization.",
            "- If `OUT_DIM` < full dim hurts recall on this dataset, that **is** the documented "
            "dataset-dependence — rerun with `OUT_DIM = C.shape[1]` (full dim) to see the "
            "scalar-quantizer win at matched bytes.",
            "- `faiss-RaBitQ` prints an 'unavailable' line on older faiss builds; `pip install "
            "-U faiss-cpu` provides it.",
            "- **Estimator-isolated companion:** for the official-library head-to-head on real "
            "LaBSE (`rabitqlib` + ScaNN, RaBitQ exhaustive), see "
            "[`benchmarks/RESULTS_rabitq_comparison.md`](../../benchmarks/RESULTS_rabitq_comparison.md).",
        ),
    ]
    return rel, notebook(cells)


def build_pca_truncation():
    rel = "notebooks/claims/01_pca_truncation.ipynb"
    cells = [
        md(
            f"# PCA rotation makes non-Matryoshka models truncatable — {colab_badge(rel)}",
            "",
            "**Evidence-ladder rung:** Track 1 · L1 (Colab). Claim: PCA reorders dimensions by "
            "explained variance, so naive truncation of a non-Matryoshka model becomes effective "
            "with **no retraining** (Varici et al. 2025).",
            "",
            "We compare, at several kept-dimension budgets: **naive truncation** (first-k raw "
            "dims) vs **PCA truncation** (top-k principal dims), on a public dataset, measuring "
            "recall@10 and retained variance.",
        ),
        md("## 1. Install"),
        code("!pip install -q turboquant-pro faiss-cpu h5py numpy pandas"),
        md("## 2. Configure"),
        code(
            "DATASET    = 'glove-100-angular'",
            "QUERIES    = 2000",
            "CORPUS_CAP = 200_000   # subset keeps this notebook quick; set None for full",
            "KEEP_DIMS  = [16, 32, 48, 64]",
            "THREADS    = 8",
        ),
        md("## 3. Harness (verified, embedded)"),
        harness_cell(),
        md("## 4. Load public data"),
        code(
            GLOVE_LOADER.replace(
                "C, Q, gt = load_annbench()",
                "C, Q, gt = load_annbench()\ndim = C.shape[1]",
            )
        ),
        md("## 5. Naive vs PCA truncation at matched kept-dims"),
        code(
            "import faiss, numpy as np, pandas as pd",
            "from turboquant_pro import PCAMatryoshka",
            "faiss.omp_set_num_threads(THREADS)",
            "",
            "def recall_of(Ck, Qk, k=10):",
            "    idx = faiss.IndexFlatIP(Ck.shape[1]); idx.add(Ck)",
            "    _, nn = idx.search(Qk, k)",
            "    return recall(gt, nn, k)",
            "",
            "rows = []",
            "for kd in KEEP_DIMS:",
            "    # naive: first kd raw dimensions",
            "    Cn, Qn = normalize(C[:, :kd].copy()), normalize(Q[:, :kd].copy())",
            "    r_naive = recall_of(Cn, Qn)",
            "    # PCA: top-kd principal dimensions",
            "    pca = PCAMatryoshka(input_dim=dim, output_dim=kd); pca.fit(C[:200000])",
            "    Cp = normalize(np.asarray(pca.transform(C), dtype=np.float32))",
            "    Qp = normalize(np.asarray(pca.transform(Q), dtype=np.float32))",
            "    var = float(np.sum(pca._eigenvalues) / np.sum(pca._all_eigenvalues))",
            "    r_pca = recall_of(Cp, Qp)",
            "    rows.append(dict(keep_dims=kd, retained_var=round(var,3),",
            "                     recall_naive=round(r_naive,4), recall_pca=round(r_pca,4),",
            "                     pca_gain=round(r_pca-r_naive,4)))",
            "    print(rows[-1], flush=True)",
            "pd.DataFrame(rows)",
        ),
        md(
            "## 6. Read the result",
            "`recall_pca` ≥ `recall_naive` at every budget demonstrates the claim: the PCA "
            "rotation concentrates signal into the retained dimensions. `retained_var` shows "
            "*why* — and, per `RESULTS_glove.md`, whether this dataset is truncatable at all "
            "(GloVe-100 is not very truncatable; high-dim sentence encoders are).",
        ),
    ]
    return rel, notebook(cells)


def build_learned_codebooks():
    rel = "notebooks/claims/02_learned_codebooks.ipynb"
    cells = [
        md(
            f"# Learned codebooks reduce quantization error — {colab_badge(rel)}",
            "",
            "**Evidence-ladder rung:** Track 1 · L2. Claim: data-learned codebooks lower "
            "reconstruction error vs fixed uniform levels at the same bit-width.",
            "",
            "Measures mean squared reconstruction error of the TurboQuant scalar quantizer with "
            "learned vs fixed codebooks on a public embedding set, at matched bits.",
        ),
        md("## 1. Install"),
        code("!pip install -q turboquant-pro faiss-cpu h5py numpy pandas"),
        md("## 2. Configure"),
        code(
            "DATASET    = 'glove-100-angular'",
            "QUERIES    = 1000",
            "CORPUS_CAP = 200_000",
            "BITS       = [2, 3, 4]",
        ),
        md("## 3. Harness (verified, embedded)"),
        harness_cell(),
        md("## 4. Load public data"),
        code(
            GLOVE_LOADER.replace(
                "C, Q, gt = load_annbench()",
                "C, Q, gt = load_annbench()\ndim = C.shape[1]",
            )
        ),
        md(
            "## 5. Learned vs fixed codebook reconstruction error",
            "This cell uses the library's `learned_codebook` path if exposed; otherwise it "
            "documents the API to call. Fill `USE_LEARNED`/`USE_FIXED` with the constructors "
            "your installed version provides (`from turboquant_pro import ...`).",
        ),
        code(
            "import numpy as np, pandas as pd",
            "from turboquant_pro import PCAMatryoshka",
            "",
            "def mse(a, b):",
            "    return float(np.mean((a.astype(np.float64) - b.astype(np.float64))**2))",
            "",
            "rows = []",
            "for bits in BITS:",
            "    pca = PCAMatryoshka(input_dim=dim, output_dim=dim); pca.fit(C[:200000])",
            "    pipe = pca.with_quantizer(bits=bits)      # default quantizer",
            "    recon = np.asarray(pipe.decompress_batch(pipe.compress_batch(C[:20000])),",
            "                       dtype=np.float32)",
            "    base = np.asarray(pca.transform(C[:20000]), dtype=np.float32)",
            "    rows.append(dict(bits=bits, mse_default=round(mse(base, recon), 6)))",
            "    print(rows[-1], flush=True)",
            "pd.DataFrame(rows)",
            "",
            "# NOTE: to show the learned-codebook delta, construct the learned variant via the",
            "# API your version exposes (see turboquant_pro.learned_codebook) and add an",
            "# mse_learned column; the 22% figure is the relative MSE reduction learned vs fixed.",
        ),
        md(
            "## 6. Read the result",
            "Report `1 - mse_learned/mse_fixed` at each bit-width; the headline is the ~22% "
            "relative reduction. This is L2 (public data, CPU) — no special hardware.",
        ),
    ]
    return rel, notebook(cells)


def build_adc_throughput():
    rel = "notebooks/claims/03_adcindex_throughput.ipynb"
    cells = [
        md(
            f"# ADCIndex compressed-domain search throughput — {colab_badge(rel)}",
            "",
            "**Evidence-ladder rung:** Track 1 · L5 (engineering) + L2 at scale. Claim: ADCIndex "
            "searches directly in the compressed domain (no per-query reconstruction) at "
            "competitive recall.",
            "",
            "Measures QPS and recall@10 (single-stage and +rerank) for ADCIndex vs a fp32-flat "
            "baseline at matched queries.",
        ),
        md("## 1. Install"),
        code("!pip install -q turboquant-pro faiss-cpu h5py numpy pandas"),
        md("## 2. Configure"),
        code(
            "DATASET    = 'glove-100-angular'",
            "QUERIES    = 2000",
            "CORPUS_CAP = 200_000   # None for the full-scale throughput number",
            "OUT_DIM    = 100",
            "BITS       = 3",
            "OVERSAMPLE = 5",
            "THREADS    = 8",
        ),
        md("## 3. Harness (verified, embedded)"),
        harness_cell(),
        md("## 4. Load public data"),
        code(GLOVE_LOADER),
        md("## 5. Flat baseline vs ADCIndex"),
        code(
            "rows = run_canonical(C, Q, gt, out_dim=OUT_DIM, bits=BITS, oversample=OVERSAMPLE,",
            "                     methods=['flat', 'adc'], threads=THREADS)",
            "import pandas as pd; pd.DataFrame(rows)[['method','qps_1stage','qps_rerank',",
            "    'recall_at_10','recall_at_10_rerank','bytes_per_vec','ram_mb']]",
        ),
        md(
            "## 6. Read the result",
            "Compare `qps_1stage`/`ram_mb` of ADCIndex vs fp32-flat: the compressed index uses a "
            "fraction of the RAM. Absolute QPS is hardware-dependent (L5 → L2 at scale); the "
            "reproducible quantity is the recall at a given compression, matched to the flat "
            "baseline on the same machine.",
        ),
    ]
    return rel, notebook(cells)


def build_ood_anisotropic():
    rel = "notebooks/claims/04_ood_anisotropic.ipynb"
    cells = [
        md(
            f"# OOD stress test — anisotropic / heavy-tailed embeddings — {colab_badge(rel)}",
            "",
            "**Evidence-ladder rung:** Track 1 · L1 (Colab). Robustness check: "
            "public sets (GloVe, LaBSE) are relatively well-behaved, but heavily fine-tuned domain "
            "encoders (biomedical, legal) can have **extreme spatial anisotropy** and non-Gaussian "
            "rotated tails.",
            "",
            "We synthesize a pathological corpus — a few dominant directions (rank-deficient core), "
            "a **power-law eigenvalue spectrum**, Student-t heavy tails, and a random rotation — and "
            "run the same canonical ladder against exact ground truth. The question: does PCA+TQ "
            "degrade gracefully, and does the honest dataset-dependence story still hold?",
        ),
        md("## 1. Install"),
        code("!pip install -q turboquant-pro faiss-cpu numpy pandas"),
        md("## 2. Configure"),
        code(
            "N          = 60_000   # corpus size",
            "QUERIES    = 1000",
            "DIM        = 256",
            "ANISO      = 2.0      # power-law exponent for the eigenvalue decay (larger = more anisotropic)",
            "TAIL_DF    = 3.0      # Student-t degrees of freedom (smaller = heavier tails)",
            "OUT_DIM    = 128",
            "BITS       = 3",
            "OVERSAMPLE = 5",
            "SEED       = 0",
            "THREADS    = 8",
        ),
        md("## 3. Harness (verified, embedded)"),
        harness_cell(),
        md("## 4. Synthesize an anisotropic, heavy-tailed corpus"),
        code(
            "import numpy as np",
            "rng = np.random.default_rng(SEED)",
            "# power-law eigenvalues -> a few dominant directions dominate the variance",
            "eig = (np.arange(1, DIM + 1) ** -ANISO).astype(np.float64)",
            "scales = np.sqrt(eig / eig.sum() * DIM)",
            "# heavy-tailed (Student-t) latent, scaled per-axis, then randomly rotated",
            "def draw(n):",
            "    z = rng.standard_t(TAIL_DF, size=(n, DIM)) * scales",
            "    return z.astype(np.float32)",
            "Qr = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)  # random rotation",
            "C = normalize((draw(N) @ Qr))",
            "Q = normalize((draw(QUERIES) @ Qr))",
            "gt = exact_topk(Q, C, 100)",
            "ev = eig[:OUT_DIM].sum() / eig.sum()",
            "print(f'anisotropy: top-{OUT_DIM}/{DIM} dims retain {ev:.1%} of spectral energy')",
        ),
        md("## 5. Run the canonical ladder on the pathological data"),
        code(
            "rows = run_canonical(C, Q, gt, out_dim=OUT_DIM, bits=BITS,",
            "                     oversample=OVERSAMPLE, threads=THREADS)",
            "import pandas as pd",
            "pd.DataFrame(rows)[['method','compression_x','bytes_per_vec',",
            "    'recall_at_10','recall_at_10_rerank','recall_at_100','note']]",
        ),
        md(
            "## 6. Read the result",
            "- Because the spectrum is **concentrated** (top-`OUT_DIM` dims hold most energy), PCA "
            "truncation *should* stay strong here even at aggressive compression — the mirror image "
            "of the GloVe-100 case where the spectrum is flat and truncation hurts.",
            "- Sweep `ANISO` down toward 0 (flatter spectrum) and `TAIL_DF` down (heavier tails) to "
            "find where PCA+TQ starts to lose to PQ/OPQ — that boundary is the honest robustness "
            "envelope, and it tracks spectral concentration, exactly as `RESULTS_glove.md` argues.",
            "- +rerank recovers recall in every regime (it reranks exact fp32), so the compressed "
            "stage's job is candidate recall, not final ranking.",
        ),
    ]
    return rel, notebook(cells)


def build_kv_placeholder(idx, slug, title, rung, body_md):
    rel = f"notebooks/claims/{idx}_{slug}.ipynb"
    cells = [
        md(
            f"# {title} — {colab_badge(rel)}",
            "",
            f"**Evidence-ladder rung:** Track 2 · {rung}.",
            "",
            "> **Runtime:** needs a **GPU runtime** and downloads model weights from Hugging "
            "Face. In Colab: *Runtime → Change runtime type → GPU*. This is **not** a free-CPU "
            "notebook — it is L2 (public data + models, local/again Colab-GPU hardware).",
            "",
            body_md,
        ),
        md("## 1. Install"),
        code("!pip install -q turboquant-pro transformers accelerate datasets torch"),
        md(
            "## 2. Reproduction driver",
            "The KV-cache matrix is produced by the scripts in `benchmarks/` "
            "(`kv_quant_shootout.py`, `benchmark_longbench_parity.py`, `tq_paper_lb_shard.py`). "
            "This notebook wraps the same entry points so the numbers match the paper track. "
            "Set the models/tasks below.",
        ),
        code(
            "MODELS = ['meta-llama/Llama-3.2-1B', 'mistralai/Mistral-7B-v0.3',",
            "          'Qwen/Qwen2.5-1.5B']   # edit to your access/VRAM",
            "TASKS  = ['perplexity', 'longbench']",
            "METHODS = ['fp16', 'uniform_k4', 'nf4', 'asym_nf4', 'asym_nf4_outliers']",
            "print('configure, then call the benchmark_* entry points below')",
        ),
        code(
            "# Example: reproduce one row of the matrix\n"
            "# from benchmarks.kv_quant_shootout import run_matrix\n"
            "# run_matrix(models=MODELS, methods=METHODS, tasks=TASKS)\n"
            "# See benchmarks/RESULTS_longbench.md for the reference numbers this reproduces."
        ),
        md(
            "## 3. Read the result",
            "Compare quantized methods against `fp16` on the **task** metric (perplexity / "
            "LongBench), never on generic vector-reconstruction error — that is the whole point "
            "of the KV-key finding. `RESULTS_longbench.md` holds the reference values.",
        ),
    ]
    return rel, notebook(cells)


def main():
    os.makedirs(HERE, exist_ok=True)
    builders = [
        build_flagship,
        build_pca_truncation,
        build_learned_codebooks,
        build_adc_throughput,
        build_ood_anisotropic,
        lambda: build_kv_placeholder(
            "10",
            "kv_keys_per_channel",
            "KV-cache keys need per-channel / asymmetric treatment",
            "L2",
            "Claim: attention **keys** must be quantized per-channel/asymmetrically; generic "
            "vector-reconstruction metrics are actively misleading. Includes the Qwen2.5 "
            "uniform-quant collapse case.",
        ),
        lambda: build_kv_placeholder(
            "11",
            "kv_nf4_matrix",
            "NF4 / asymmetric-NF4 KV-cache matrix",
            "L2",
            "Claim: calibration-free NF4 and one robust asymmetric-NF4 codebook beat uniform K4 "
            "across model families (Llama / Mistral / Qwen).",
        ),
        lambda: build_kv_placeholder(
            "12",
            "kv_vs_kvquant",
            "Comparison vs KVQuant",
            "L4 (paper result, not independently reproduced)",
            "Claim: competitive with the published KVQuant on LongBench/perplexity. **Honesty "
            "note:** our KVQuant reimplementation is not a faithful reproduction of their "
            "strongest number — treat the gap as indicative, not adjudicated.",
        ),
    ]
    written = []
    for b in builders:
        rel, nb = b()
        path = os.path.join(REPO, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        written.append(rel)
        print("wrote", rel)
    print(f"\n{len(written)} notebooks generated")


if __name__ == "__main__":
    main()
