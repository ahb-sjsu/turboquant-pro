#!/home/claude/env/bin/python
"""IEEE TAI Reviewer Experiments for PCA-Matryoshka Paper.

Runs 6 experiments requested by reviewers:
1. E5-large-v2 (second non-Matryoshka model)
2. STS Benchmark (public benchmark)
3. Out-of-domain generalization
4. Expanded retrieval evaluation (200 queries)
5. Cross-lingual UDHR retrieval
6. Eigenspectrum comparison

Usage:
    python3 run_reviewer_experiments.py --all
    python3 run_reviewer_experiments.py --exp e5
    python3 run_reviewer_experiments.py --exp sts
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from batch_probe import ThermalController

# Global thermal controller — caps CPU temp at 80°C using Kalman-filtered PID
# CPU0 runs hot on Z840 with ethics embedding on GPU1 also generating heat
_thermal = ThermalController(target_temp=80.0, max_threads=6, min_threads=2, verbose=True)


def _apply_thermal_threads():
    """Set PyTorch and OS thread count from thermal controller."""
    n = _thermal.get_threads()
    torch.set_num_threads(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    return n

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# DB config
DB_DSN = os.environ.get("ATLAS_DB_DSN", "dbname=atlas user=claude")


def get_db_embeddings(n=10000, table="chunks"):
    """Fetch n embeddings from the database."""
    import psycopg2
    log.info(f"Fetching {n} embeddings from {table}...")
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()
    cur.execute(f"SELECT embedding FROM {table} WHERE embedding IS NOT NULL ORDER BY random() LIMIT %s", (n,))
    rows = cur.fetchall()
    conn.close()
    parsed = []
    for r in rows:
        val = r[0]
        if isinstance(val, str):
            # pgvector returns '[0.1,0.2,...]' string or bare CSV
            val = val.strip("[]")
            parsed.append(np.fromstring(val, sep=",", dtype=np.float32))
        elif isinstance(val, (list, tuple)):
            parsed.append(np.array(val, dtype=np.float32))
        elif isinstance(val, (bytes, memoryview)):
            parsed.append(np.frombuffer(val, dtype=np.float32))
        else:
            parsed.append(np.array(val, dtype=np.float32))
    embeddings = np.array(parsed)
    log.info(f"Fetched {len(embeddings)} embeddings, shape: {embeddings.shape}")
    return embeddings


def get_db_texts(n=10000, table="chunks"):
    """Fetch n texts from the database."""
    import psycopg2
    log.info(f"Fetching {n} texts from {table}...")
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()
    cur.execute(f"SELECT content FROM {table} WHERE embedding IS NOT NULL ORDER BY random() LIMIT %s", (n,))
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]


def pca_fit(embeddings, n_components=None):
    """Fit PCA on embeddings, return components and explained variance."""
    from sklearn.decomposition import PCA
    if n_components is None:
        n_components = embeddings.shape[1]
    log.info(f"Fitting PCA with {n_components} components on {len(embeddings)} vectors...")
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    return pca


def pca_truncate_cosine(embeddings, pca, k):
    """Compute mean cosine sim between original and PCA-truncated vectors."""
    projected = embeddings @ pca.components_[:k].T  # (N, k)
    reconstructed = projected @ pca.components_[:k]  # (N, d)
    # Add back mean
    reconstructed += pca.mean_

    # Cosine similarity
    norms_orig = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms_recon = np.linalg.norm(reconstructed, axis=1, keepdims=True)
    cos_sims = np.sum(embeddings * reconstructed, axis=1) / (
        norms_orig.squeeze() * norms_recon.squeeze() + 1e-10
    )
    return float(np.mean(cos_sims))


def naive_truncate_cosine(embeddings, k):
    """Compute mean cosine sim between original and naively truncated."""
    truncated = embeddings.copy()
    truncated[:, k:] = 0  # Zero out trailing dims

    norms_orig = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms_trunc = np.linalg.norm(truncated, axis=1, keepdims=True)
    cos_sims = np.sum(embeddings * truncated, axis=1) / (
        norms_orig.squeeze() * norms_trunc.squeeze() + 1e-10
    )
    return float(np.mean(cos_sims))


def recall_at_k(embeddings, pca, dims, k=10, n_queries=200):
    """Compute recall@k for PCA-truncated vs full embeddings."""
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-10)

    # Full-dim similarities for ground truth
    query_indices = np.random.choice(len(normed), min(n_queries, len(normed)), replace=False)
    queries_full = normed[query_indices]

    # Ground truth top-k
    full_sims = queries_full @ normed.T
    # Set self-similarity to -inf
    for i, qi in enumerate(query_indices):
        full_sims[i, qi] = -np.inf
    gt_topk = np.argsort(-full_sims, axis=1)[:, :k]

    # PCA-truncated search
    centered = embeddings - pca.mean_
    projected = centered @ pca.components_[:dims].T
    proj_norms = np.linalg.norm(projected, axis=1, keepdims=True)
    proj_normed = projected / (proj_norms + 1e-10)

    query_proj = proj_normed[query_indices]
    pca_sims = query_proj @ proj_normed.T
    for i, qi in enumerate(query_indices):
        pca_sims[i, qi] = -np.inf
    pca_topk = np.argsort(-pca_sims, axis=1)[:, :k]

    # Recall per query
    recalls = []
    for i in range(len(query_indices)):
        gt_set = set(gt_topk[i])
        pca_set = set(pca_topk[i])
        recalls.append(len(gt_set & pca_set) / k)

    mean_recall = float(np.mean(recalls))
    std_recall = float(np.std(recalls))
    se_recall = std_recall / np.sqrt(len(recalls))
    return mean_recall, se_recall, recalls


# ============================================================
# Experiment 1: E5-large-v2
# ============================================================
def run_e5_experiment():
    """Test PCA-Matryoshka on E5-large-v2 (second non-Matryoshka model)."""
    log.info("=" * 60)
    log.info("EXPERIMENT 1: E5-large-v2")
    log.info("=" * 60)

    from sentence_transformers import SentenceTransformer

    # Get texts from ethics corpus
    texts = get_db_texts(n=10000)
    log.info(f"Got {len(texts)} texts")

    # Embed with E5-large-v2
    log.info("Loading E5-large-v2...")
    # GPU0 has ~1.6GB free — E5-large-v2 is ~1.3GB, try GPU first
    _device = "cuda:0" if torch.cuda.is_available() and torch.cuda.mem_get_info(0)[0] > 1_400_000_000 else "cpu"
    log.info(f"Using device: {_device}")
    model = SentenceTransformer("intfloat/e5-large-v2", device=_device)
    log.info(f"E5-large-v2 loaded, dim={model.get_sentence_embedding_dimension()}")

    # E5 requires "query: " or "passage: " prefix
    prefixed = [f"passage: {t[:512]}" for t in texts]
    _apply_thermal_threads()
    log.info("Embedding texts with E5-large-v2 (CPU, this will take a while)...")
    e5_embeddings = model.encode(prefixed, batch_size=16, show_progress_bar=True,
                                  normalize_embeddings=True)
    e5_embeddings = np.array(e5_embeddings, dtype=np.float32)
    log.info(f"E5 embeddings shape: {e5_embeddings.shape}")

    # Also get BGE-M3 embeddings for comparison
    bge_embeddings = get_db_embeddings(n=10000)

    results = {"model": "E5-large-v2", "dim": int(e5_embeddings.shape[1]),
               "n_vectors": len(e5_embeddings)}

    # Fit PCA on E5 embeddings
    pca_e5 = pca_fit(e5_embeddings)
    pca_bge = pca_fit(bge_embeddings)

    # Compare truncation at various dims
    dims_to_test = [128, 256, 384, 512]
    results["truncation"] = []
    for d in dims_to_test:
        if d >= e5_embeddings.shape[1]:
            continue
        naive_cos = naive_truncate_cosine(e5_embeddings, d)
        pca_cos = pca_truncate_cosine(e5_embeddings, pca_e5, d)
        improvement = (pca_cos - naive_cos) / naive_cos * 100

        naive_bge = naive_truncate_cosine(bge_embeddings, d)
        pca_bge_cos = pca_truncate_cosine(bge_embeddings, pca_bge, d)

        entry = {
            "dims": d,
            "e5_naive": round(naive_cos, 4),
            "e5_pca": round(pca_cos, 4),
            "e5_improvement": round(improvement, 1),
            "bge_naive": round(naive_bge, 4),
            "bge_pca": round(pca_bge_cos, 4),
        }
        log.info(f"  dims={d}: E5 naive={naive_cos:.4f} PCA={pca_cos:.4f} "
                 f"(+{improvement:.1f}%)  BGE naive={naive_bge:.4f} PCA={pca_bge_cos:.4f}")
        results["truncation"].append(entry)

    # Eigenspectrum for E5
    var_explained_e5 = np.cumsum(pca_e5.explained_variance_ratio_)
    var_explained_bge = np.cumsum(pca_bge.explained_variance_ratio_)
    results["eigenspectrum"] = {
        "e5_var_128": round(float(var_explained_e5[127]), 4),
        "e5_var_256": round(float(var_explained_e5[255]), 4),
        "e5_var_384": round(float(var_explained_e5[383]), 4),
        "e5_var_512": round(float(var_explained_e5[511]), 4),
        "bge_var_128": round(float(var_explained_bge[127]), 4),
        "bge_var_256": round(float(var_explained_bge[255]), 4),
        "bge_var_384": round(float(var_explained_bge[383]), 4),
        "bge_var_512": round(float(var_explained_bge[511]), 4),
    }
    # Save raw eigenvalues for plotting
    results["eigenvalues_e5"] = pca_e5.explained_variance_ratio_.tolist()[:100]
    results["eigenvalues_bge"] = pca_bge.explained_variance_ratio_.tolist()[:100]

    # Recall@10 for E5
    results["recall"] = []
    for d in dims_to_test:
        if d >= e5_embeddings.shape[1]:
            continue
        mean_r, se_r, _ = recall_at_k(e5_embeddings, pca_e5, d, k=10, n_queries=200)
        results["recall"].append({
            "dims": d, "recall_at_10": round(mean_r, 4), "se": round(se_r, 4)
        })
        log.info(f"  E5 recall@10 at dims={d}: {mean_r:.4f} ± {se_r:.4f}")

    out_path = RESULTS_DIR / "exp1_e5_large_v2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {out_path}")
    return results


# ============================================================
# Experiment 2: STS Benchmark
# ============================================================
def run_sts_experiment():
    """Evaluate on STS Benchmark (public benchmark)."""
    log.info("=" * 60)
    log.info("EXPERIMENT 2: STS Benchmark")
    log.info("=" * 60)

    from sentence_transformers import SentenceTransformer
    from scipy.stats import spearmanr

    # Download STS Benchmark
    import urllib.request
    import csv
    import io

    sts_url = "https://raw.githubusercontent.com/facebookresearch/SentEval/main/data/downstream/STS/STSBenchmark/sts-test.csv"
    log.info("Downloading STS Benchmark test set...")
    try:
        response = urllib.request.urlopen(sts_url, timeout=30)
        sts_data = response.read().decode("utf-8")
    except Exception as e:
        # Fallback: try alternate URL
        log.warning(f"Primary URL failed: {e}, trying alternate...")
        sts_url2 = "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2017/track5.en-en.tsv"
        try:
            response = urllib.request.urlopen(sts_url2, timeout=30)
            sts_data = response.read().decode("utf-8")
        except Exception as e2:
            log.error(f"Could not download STS data: {e2}")
            # Generate synthetic STS-like data
            log.info("Generating synthetic STS-like data for benchmarking...")
            return _run_sts_synthetic()

    # Parse STS data
    sentences1, sentences2, scores = [], [], []
    for line in sts_data.strip().split("\n"):
        parts = line.strip().split("\t")
        if len(parts) >= 7:
            try:
                score = float(parts[4])
                s1, s2 = parts[5], parts[6]
                sentences1.append(s1)
                sentences2.append(s2)
                scores.append(score)
            except (ValueError, IndexError):
                continue

    if len(sentences1) < 100:
        log.warning(f"Only {len(sentences1)} STS pairs found, trying alternate parse...")
        sentences1, sentences2, scores = [], [], []
        for line in sts_data.strip().split("\n"):
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                try:
                    score = float(parts[0])
                    s1, s2 = parts[1], parts[2]
                    sentences1.append(s1)
                    sentences2.append(s2)
                    scores.append(score)
                except (ValueError, IndexError):
                    continue

    log.info(f"Loaded {len(sentences1)} STS pairs")
    if len(sentences1) < 50:
        return _run_sts_synthetic()

    # Embed with BGE-M3
    log.info("Loading BGE-M3...")
    # BGE-M3 ~2.3GB won't fit in remaining VRAM; CPU with thermal throttling
    _apply_thermal_threads()
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")

    all_texts = sentences1 + sentences2
    _apply_thermal_threads()
    log.info(f"Embedding {len(all_texts)} sentences...")
    all_embs = model.encode(all_texts, batch_size=32, show_progress_bar=True,
                             normalize_embeddings=True)
    all_embs = np.array(all_embs, dtype=np.float32)

    embs1 = all_embs[:len(sentences1)]
    embs2 = all_embs[len(sentences1):]

    gold_scores = np.array(scores)

    # Fit PCA
    pca = pca_fit(all_embs)

    results = {"benchmark": "STSBenchmark", "n_pairs": len(sentences1)}
    results["correlations"] = []

    # Full dim baseline
    full_cos = np.sum(embs1 * embs2, axis=1) / (
        np.linalg.norm(embs1, axis=1) * np.linalg.norm(embs2, axis=1) + 1e-10
    )
    rho_full, p_full = spearmanr(gold_scores, full_cos)
    results["correlations"].append({
        "method": "Full (1024d)",
        "spearman_rho": round(float(rho_full), 4),
        "p_value": float(p_full),
    })
    log.info(f"Full 1024d: Spearman ρ = {rho_full:.4f}")

    # Test PCA-Matryoshka at various dims
    for d in [128, 256, 384, 512]:
        # PCA truncated
        centered1 = embs1 - pca.mean_
        centered2 = embs2 - pca.mean_
        proj1 = centered1 @ pca.components_[:d].T
        proj2 = centered2 @ pca.components_[:d].T
        pca_cos = np.sum(proj1 * proj2, axis=1) / (
            np.linalg.norm(proj1, axis=1) * np.linalg.norm(proj2, axis=1) + 1e-10
        )
        rho_pca, p_pca = spearmanr(gold_scores, pca_cos)

        # Naive truncated
        naive1 = embs1[:, :d]
        naive2 = embs2[:, :d]
        naive_cos = np.sum(naive1 * naive2, axis=1) / (
            np.linalg.norm(naive1, axis=1) * np.linalg.norm(naive2, axis=1) + 1e-10
        )
        rho_naive, p_naive = spearmanr(gold_scores, naive_cos)

        entry = {
            "dims": d,
            "pca_spearman": round(float(rho_pca), 4),
            "naive_spearman": round(float(rho_naive), 4),
            "improvement_pct": round((rho_pca - rho_naive) / abs(rho_naive) * 100, 1),
        }
        results["correlations"].append(entry)
        log.info(f"  dims={d}: PCA ρ={rho_pca:.4f}, Naive ρ={rho_naive:.4f}")

    out_path = RESULTS_DIR / "exp2_sts_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {out_path}")
    return results


def _run_sts_synthetic():
    """Fallback STS evaluation using sentence similarity from the corpus."""
    log.info("Running synthetic STS evaluation using ethics corpus pairs...")
    embeddings = get_db_embeddings(n=10000)
    pca = pca_fit(embeddings)

    # Create synthetic pairs by using nearby vectors as "similar" and distant as "dissimilar"
    n_pairs = 500
    idx1 = np.random.choice(len(embeddings), n_pairs, replace=False)
    embs_query = embeddings[idx1]

    # Full-dim similarities as "gold standard"
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-10)
    full_sims = embs_query @ normed.T / (np.linalg.norm(embs_query, axis=1, keepdims=True) + 1e-10)

    # Pick pairs with varying similarity
    pairs_sim = []
    for i in range(n_pairs):
        top_idx = np.argsort(-full_sims[i])
        # Pick one similar and one random
        if len(top_idx) > 5:
            pairs_sim.append(float(full_sims[i, top_idx[np.random.randint(1, 20)]]))
            pairs_sim.append(float(full_sims[i, top_idx[np.random.randint(500, 5000)]]))

    results = {"benchmark": "Synthetic-STS (corpus-based)", "n_pairs": len(pairs_sim),
               "note": "Could not download STS Benchmark; using corpus-based proxy"}
    out_path = RESULTS_DIR / "exp2_sts_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


# ============================================================
# Experiment 3: Out-of-domain generalization
# ============================================================
def run_ood_experiment():
    """Test PCA learned on ethics corpus applied to out-of-domain data."""
    log.info("=" * 60)
    log.info("EXPERIMENT 3: Out-of-Domain Generalization")
    log.info("=" * 60)

    from sentence_transformers import SentenceTransformer

    # Fit PCA on ethics corpus embeddings
    ethics_embs = get_db_embeddings(n=10000)
    pca_ethics = pca_fit(ethics_embs)

    # Generate out-of-domain texts (tech/science topics, very different from ethics)
    ood_texts = [
        "The TCP/IP protocol stack enables reliable communication across heterogeneous networks.",
        "Quantum entanglement allows particles to share states regardless of distance.",
        "The Krebs cycle produces ATP through oxidative phosphorylation in mitochondria.",
        "Machine learning models use gradient descent to minimize loss functions.",
        "Continental drift explains how tectonic plates move over geological time.",
        "The Higgs boson was discovered at CERN in 2012 confirming the Standard Model.",
        "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
        "The Fourier transform decomposes signals into constituent frequencies.",
        "DNA replication occurs during the S phase of the cell cycle.",
        "Black holes form when massive stars collapse under gravitational force.",
        "The Navier-Stokes equations describe the motion of viscous fluids.",
        "Cryptocurrency uses blockchain technology for decentralized transactions.",
        "CRISPR-Cas9 enables precise gene editing by targeting specific DNA sequences.",
        "Moore's Law predicted transistor density doubling approximately every two years.",
        "Superconductors exhibit zero electrical resistance below critical temperature.",
        "Neural networks use backpropagation to adjust weights during training.",
        "The double helix structure of DNA was discovered by Watson and Crick.",
        "Relativity shows that mass and energy are equivalent through E=mc².",
        "Compiler optimization techniques include loop unrolling and dead code elimination.",
        "The human genome contains approximately 3 billion base pairs.",
        "Quantum computing uses qubits that can exist in superposition states.",
        "Plate tectonics explains earthquakes through subduction and transform faults.",
        "Docker containers provide lightweight virtualization for application deployment.",
        "The central dogma of molecular biology describes DNA to RNA to protein.",
        "Hubble's Law relates galaxy recession velocity to distance from Earth.",
        "Recursive algorithms solve problems by breaking them into smaller subproblems.",
        "The Standard Model describes fundamental particles and three of four forces.",
        "Convolutional neural networks excel at image recognition tasks.",
        "Mitosis divides a cell's nucleus into two identical daughter nuclei.",
        "The traveling salesman problem is NP-hard and has no known polynomial solution.",
        "Antibiotics work by disrupting bacterial cell wall synthesis or protein production.",
        "The halting problem proves that no algorithm can determine if any program terminates.",
        "Protein folding is governed by thermodynamics and hydrophobic interactions.",
        "RSA encryption relies on the difficulty of factoring large prime numbers.",
        "Climate change is driven primarily by greenhouse gas emissions from fossil fuels.",
        "Graph neural networks process data structured as nodes and edges.",
        "The uncertainty principle limits simultaneous knowledge of position and momentum.",
        "Kubernetes orchestrates containerized applications across clusters of machines.",
        "Evolution through natural selection was proposed by Darwin and Wallace.",
        "The PageRank algorithm determines web page importance from link structure.",
        "Ribosomes translate mRNA codons into amino acid sequences during protein synthesis.",
        "Hash tables provide average O(1) time complexity for insertion and lookup.",
        "Gravitational waves were first detected by LIGO in 2015.",
        "Transfer learning allows models trained on one task to perform related tasks.",
        "The citric acid cycle is a key metabolic pathway in aerobic organisms.",
        "Public key cryptography enables secure communication without shared secrets.",
        "The Big Bang theory explains the origin and expansion of the universe.",
        "Generative adversarial networks pit two neural networks against each other.",
        "Enzymes are biological catalysts that lower activation energy for reactions.",
        "MapReduce enables distributed processing of large datasets across clusters.",
    ]

    # Also get some news/current events style texts
    news_texts = [
        "The Federal Reserve announced a quarter-point interest rate cut today.",
        "SpaceX successfully landed its Starship booster for the third time.",
        "The World Health Organization declared the new variant a concern.",
        "Tesla reported record quarterly earnings driven by energy storage growth.",
        "The United Nations General Assembly voted on the climate resolution.",
        "Amazon announced plans to open a new fulfillment center in Ohio.",
        "The Supreme Court ruled on the constitutionality of the digital privacy act.",
        "Researchers at MIT developed a new battery technology using sodium.",
        "The Olympics committee selected the host city for the 2036 games.",
        "Apple unveiled its latest augmented reality headset at WWDC.",
        "The European Central Bank maintained its current interest rate policy.",
        "NASA's James Webb Space Telescope captured new images of distant galaxies.",
        "A major earthquake struck the Pacific coast measuring 7.2 magnitude.",
        "Google announced advances in its quantum computing research program.",
        "The International Space Station completed its twenty-fifth year in orbit.",
        "A new study linked social media use to changes in adolescent brain development.",
        "The G7 summit concluded with agreements on trade and climate policy.",
        "Microsoft reported strong cloud revenue growth in its quarterly earnings.",
        "Volcanic activity increased at several sites along the Pacific Ring of Fire.",
        "The census results showed significant population shifts to southern states.",
    ]

    all_ood = ood_texts + news_texts
    _apply_thermal_threads()
    log.info(f"Embedding {len(all_ood)} out-of-domain texts with BGE-M3...")

    # BGE-M3 ~2.3GB won't fit in remaining VRAM; CPU with thermal throttling
    _apply_thermal_threads()
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    ood_embs = model.encode(all_ood, batch_size=32, show_progress_bar=True,
                             normalize_embeddings=True)
    ood_embs = np.array(ood_embs, dtype=np.float32)
    log.info(f"OOD embeddings shape: {ood_embs.shape}")

    # Fit PCA on OOD data itself for comparison
    pca_ood = pca_fit(ood_embs)

    results = {"experiment": "Out-of-Domain", "n_ood_vectors": len(all_ood),
               "pca_trained_on": "ethics_corpus_10k"}

    results["truncation"] = []
    for d in [128, 256, 384, 512]:
        # Ethics PCA applied to OOD
        cos_ethics_pca = pca_truncate_cosine(ood_embs, pca_ethics, d)
        # OOD PCA (ideal, fitted on same data)
        cos_ood_pca = pca_truncate_cosine(ood_embs, pca_ood, d)
        # Naive
        cos_naive = naive_truncate_cosine(ood_embs, d)
        # Transfer gap
        transfer_gap = cos_ood_pca - cos_ethics_pca

        entry = {
            "dims": d,
            "naive": round(cos_naive, 4),
            "pca_ethics": round(cos_ethics_pca, 4),
            "pca_ood_ideal": round(cos_ood_pca, 4),
            "transfer_gap": round(transfer_gap, 4),
            "improvement_over_naive": round((cos_ethics_pca - cos_naive) / cos_naive * 100, 1),
        }
        results["truncation"].append(entry)
        log.info(f"  dims={d}: naive={cos_naive:.4f}  ethics_PCA={cos_ethics_pca:.4f}  "
                 f"ideal_PCA={cos_ood_pca:.4f}  gap={transfer_gap:.4f}")

    # Variance explained comparison
    var_ethics = np.cumsum(pca_ethics.explained_variance_ratio_)
    var_ood = np.cumsum(pca_ood.explained_variance_ratio_)
    results["variance_explained"] = {
        "ethics_256": round(float(var_ethics[255]), 4),
        "ethics_384": round(float(var_ethics[383]), 4),
        "ood_256": round(float(var_ood[min(255, len(var_ood)-1)]), 4),
        "ood_384": round(float(var_ood[min(383, len(var_ood)-1)]), 4),
    }

    out_path = RESULTS_DIR / "exp3_out_of_domain.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {out_path}")
    return results


# ============================================================
# Experiment 4: Expanded Retrieval (200 queries)
# ============================================================
def run_expanded_retrieval():
    """Run recall@10 with 200 queries and confidence intervals."""
    log.info("=" * 60)
    log.info("EXPERIMENT 4: Expanded Retrieval (200 queries)")
    log.info("=" * 60)

    embeddings = get_db_embeddings(n=50000)
    pca = pca_fit(embeddings)

    results = {"experiment": "Expanded Retrieval", "n_vectors": len(embeddings),
               "n_queries": 200, "k": 10}
    results["configs"] = []

    for d in [128, 256, 384, 512]:
        mean_r, se_r, per_query = recall_at_k(embeddings, pca, d, k=10, n_queries=200)

        # Compute confidence intervals
        recalls_arr = np.array(per_query)
        ci_95_low = np.percentile(recalls_arr, 2.5)
        ci_95_high = np.percentile(recalls_arr, 97.5)
        median_r = float(np.median(recalls_arr))

        entry = {
            "dims": d,
            "mean_recall": round(mean_r, 4),
            "se": round(se_r, 4),
            "median_recall": round(median_r, 4),
            "ci_95": [round(ci_95_low, 4), round(ci_95_high, 4)],
            "min_recall": round(float(np.min(recalls_arr)), 4),
            "max_recall": round(float(np.max(recalls_arr)), 4),
        }
        results["configs"].append(entry)
        log.info(f"  PCA-{d}: recall@10 = {mean_r:.4f} ± {se_r:.4f}  "
                 f"95% CI [{ci_95_low:.4f}, {ci_95_high:.4f}]")

    # Also test with TurboQuant 3-bit combined
    results["combined"] = []
    for d in [128, 256, 384, 512]:
        # PCA truncation + simulated 3-bit quantization
        centered = embeddings - pca.mean_
        projected = centered @ pca.components_[:d].T

        # Simulate 3-bit scalar quantization
        mins = projected.min(axis=0)
        maxs = projected.max(axis=0)
        ranges = maxs - mins + 1e-10
        n_levels = 2**3  # 8 levels for 3-bit
        quantized = np.round((projected - mins) / ranges * (n_levels - 1))
        dequantized = quantized / (n_levels - 1) * ranges + mins

        # Recall with quantized vectors
        norms = np.linalg.norm(dequantized, axis=1, keepdims=True)
        normed = dequantized / (norms + 1e-10)

        query_indices = np.random.choice(len(normed), 200, replace=False)
        queries = normed[query_indices]

        # Ground truth from full embeddings
        full_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        full_normed = embeddings / (full_norms + 1e-10)
        full_sims = full_normed[query_indices] @ full_normed.T
        for i, qi in enumerate(query_indices):
            full_sims[i, qi] = -np.inf
        gt_topk = np.argsort(-full_sims, axis=1)[:, :10]

        comp_sims = queries @ normed.T
        for i, qi in enumerate(query_indices):
            comp_sims[i, qi] = -np.inf
        comp_topk = np.argsort(-comp_sims, axis=1)[:, :10]

        recalls = []
        for i in range(200):
            gt_set = set(gt_topk[i])
            comp_set = set(comp_topk[i])
            recalls.append(len(gt_set & comp_set) / 10)

        mean_r = float(np.mean(recalls))
        se_r = float(np.std(recalls) / np.sqrt(len(recalls)))
        ratio = 1024 * 32 / (d * 3)

        entry = {
            "method": f"PCA-{d} + TQ3",
            "compression_ratio": round(ratio, 1),
            "mean_recall": round(mean_r, 4),
            "se": round(se_r, 4),
        }
        results["combined"].append(entry)
        log.info(f"  PCA-{d}+TQ3 ({ratio:.1f}x): recall@10 = {mean_r:.4f} ± {se_r:.4f}")

    out_path = RESULTS_DIR / "exp4_expanded_retrieval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {out_path}")
    return results


# ============================================================
# Experiment 5: Cross-lingual UDHR
# ============================================================
def run_crosslingual_experiment():
    """Test cross-lingual retrieval with UDHR texts."""
    log.info("=" * 60)
    log.info("EXPERIMENT 5: Cross-Lingual UDHR Retrieval")
    log.info("=" * 60)

    from sentence_transformers import SentenceTransformer

    # UDHR Article 1 in multiple languages
    udhr_articles = {
        "en": "All human beings are born free and equal in dignity and rights. They are endowed with reason and conscience and should act towards one another in a spirit of brotherhood.",
        "fr": "Tous les êtres humains naissent libres et égaux en dignité et en droits. Ils sont doués de raison et de conscience et doivent agir les uns envers les autres dans un esprit de fraternité.",
        "es": "Todos los seres humanos nacen libres e iguales en dignidad y derechos y, dotados como están de razón y conciencia, deben comportarse fraternalmente los unos con los otros.",
        "de": "Alle Menschen sind frei und gleich an Würde und Rechten geboren. Sie sind mit Vernunft und Gewissen begabt und sollen einander im Geist der Brüderlichkeit begegnen.",
        "zh": "人人生而自由，在尊严和权利上一律平等。他们赋有理性和良心，并应以兄弟关系的精神相对待。",
        "ar": "يولد جميع الناس أحراراً متساوين في الكرامة والحقوق، وقد وهبوا عقلاً وضميراً وعليهم أن يعامل بعضهم بعضاً بروح الإخاء.",
        "ja": "すべての人間は、生れながらにして自由であり、かつ、尊厳と権利とについて平等である。人間は、理性と良心とを授けられており、互いに同胞の精神をもって行動しなければならない。",
        "ru": "Все люди рождаются свободными и равными в своём достоинстве и правах. Они наделены разумом и совестью и должны поступать в отношении друг друга в духе братства.",
        "pt": "Todos os seres humanos nascem livres e iguais em dignidade e em direitos. Dotados de razão e de consciência, devem agir uns para com os outros em espírito de fraternidade.",
        "hi": "सभी मनुष्यों को गौरव और अधिकारों के मामले में जन्मजात स्वतन्त्रता और समानता प्राप्त है। उन्हें बुद्धि और अन्तरात्मा की देन प्राप्त है और परस्पर उन्हें भाईचारे के भाव से बर्ताव करना चाहिये।",
        "ko": "모든 인간은 태어날 때부터 자유로우며 그 존엄과 권리에 있어 동등하다. 인간은 천부적으로 이성과 양심을 부여받았으며 서로 형제애의 정신으로 행동하여야 한다.",
        "he": "כל בני האדם נולדו חופשיים ושווים בכבודם ובזכויותיהם. הם חוננו בתבונה ובמצפון, לפיכך חייבם לנהוג איש ברעהו ברוח של אחוה.",
    }

    # UDHR Article 3 for a second set
    udhr_article3 = {
        "en": "Everyone has the right to life, liberty and security of person.",
        "fr": "Tout individu a droit à la vie, à la liberté et à la sûreté de sa personne.",
        "es": "Todo individuo tiene derecho a la vida, a la libertad y a la seguridad de su persona.",
        "de": "Jeder hat das Recht auf Leben, Freiheit und Sicherheit der Person.",
        "zh": "人人有权享有生命、自由和人身安全。",
        "ar": "لكل فرد الحق في الحياة والحرية وسلامة شخصه.",
        "ja": "すべて人は、生命、自由及び身体の安全に対する権利を有する。",
        "ru": "Каждый человек имеет право на жизнь, на свободу и на личную неприкосновенность.",
        "pt": "Todo ser humano tem direito à vida, à liberdade e à segurança pessoal.",
        "hi": "प्रत्येक व्यक्ति को जीवन, स्वतन्त्रता और वैयक्तिक सुरक्षा का अधिकार है।",
        "ko": "모든 사람은 생명과 신체의 자유와 안전에 대한 권리를 가진다.",
        "he": "לכל אדם הזכות לחיים, לחירות ולביטחון הנפש.",
    }

    # A distractor set (different UDHR articles)
    udhr_article5 = {
        "en": "No one shall be subjected to torture or to cruel, inhuman or degrading treatment or punishment.",
        "fr": "Nul ne sera soumis à la torture, ni à des peines ou traitements cruels, inhumains ou dégradants.",
        "es": "Nadie será sometido a torturas ni a penas o tratos crueles, inhumanos o degradantes.",
        "de": "Niemand darf der Folter oder grausamer, unmenschlicher oder erniedrigender Behandlung oder Strafe unterworfen werden.",
    }

    all_texts = []
    all_labels = []  # (article_num, language)

    for lang, text in udhr_articles.items():
        all_texts.append(text)
        all_labels.append(("art1", lang))
    for lang, text in udhr_article3.items():
        all_texts.append(text)
        all_labels.append(("art3", lang))
    for lang, text in udhr_article5.items():
        all_texts.append(text)
        all_labels.append(("art5", lang))

    _apply_thermal_threads()
    log.info(f"Embedding {len(all_texts)} UDHR texts...")
    # BGE-M3 ~2.3GB won't fit in remaining VRAM; CPU with thermal throttling
    _apply_thermal_threads()
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    embs = model.encode(all_texts, batch_size=32, normalize_embeddings=True)
    embs = np.array(embs, dtype=np.float32)

    # Fit PCA on ethics corpus
    ethics_embs = get_db_embeddings(n=10000)
    pca = pca_fit(ethics_embs)

    results = {"experiment": "Cross-Lingual UDHR", "n_texts": len(all_texts),
               "languages": list(udhr_articles.keys())}

    # For each English query, find which articles are retrieved
    # Test: query with English Art 1, should retrieve Art 1 in other languages
    def crosslingual_accuracy(embs_matrix, labels, query_lang="en"):
        """For each article in query_lang, check if same article in other langs is retrieved."""
        # Get query indices (English texts)
        query_indices = [i for i, (art, lang) in enumerate(labels) if lang == query_lang]
        correct = 0
        total = 0

        for qi in query_indices:
            art_num = labels[qi][0]
            query = embs_matrix[qi:qi+1]
            sims = (query @ embs_matrix.T).squeeze()
            sims[qi] = -np.inf  # exclude self

            # Get top-5
            top5 = np.argsort(-sims)[:5]
            # Check if same article in different language is in top 5
            same_article_others = [i for i, (a, l) in enumerate(labels)
                                   if a == art_num and l != query_lang]
            if same_article_others:
                retrieved = set(top5.tolist())
                target_set = set(same_article_others)
                hits = len(retrieved & target_set)
                correct += hits
                total += min(len(target_set), 5)

        return correct / total if total > 0 else 0

    results["retrieval"] = []

    # Full dimensional
    acc_full = crosslingual_accuracy(embs, all_labels)
    results["retrieval"].append({"method": "Full (1024d)", "accuracy": round(acc_full, 4)})
    log.info(f"Full 1024d cross-lingual accuracy: {acc_full:.4f}")

    for d in [128, 256, 384, 512]:
        # PCA truncated
        centered = embs - pca.mean_
        projected = centered @ pca.components_[:d].T
        proj_norms = np.linalg.norm(projected, axis=1, keepdims=True)
        proj_normed = projected / (proj_norms + 1e-10)
        acc_pca = crosslingual_accuracy(proj_normed, all_labels)

        # Naive truncated
        naive = embs[:, :d]
        naive_norms = np.linalg.norm(naive, axis=1, keepdims=True)
        naive_normed = naive / (naive_norms + 1e-10)
        acc_naive = crosslingual_accuracy(naive_normed, all_labels)

        entry = {
            "dims": d,
            "pca_accuracy": round(acc_pca, 4),
            "naive_accuracy": round(acc_naive, 4),
        }
        results["retrieval"].append(entry)
        log.info(f"  dims={d}: PCA acc={acc_pca:.4f}  Naive acc={acc_naive:.4f}")

    out_path = RESULTS_DIR / "exp5_crosslingual_udhr.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {out_path}")
    return results


# ============================================================
# Experiment 6: Eigenspectrum comparison
# ============================================================
def run_eigenspectrum_experiment():
    """Compare eigenvalue spectra of BGE-M3 and E5-large-v2."""
    log.info("=" * 60)
    log.info("EXPERIMENT 6: Eigenspectrum Comparison")
    log.info("=" * 60)

    from sentence_transformers import SentenceTransformer

    # Get texts
    texts = get_db_texts(n=10000)

    # BGE-M3 embeddings from DB
    bge_embs = get_db_embeddings(n=10000)
    pca_bge = pca_fit(bge_embs)

    # E5-large-v2
    log.info("Loading E5-large-v2...")
    # GPU0 has ~1.6GB free — E5-large-v2 is ~1.3GB, try GPU first
    _device = "cuda:0" if torch.cuda.is_available() and torch.cuda.mem_get_info(0)[0] > 1_400_000_000 else "cpu"
    log.info(f"Using device: {_device}")
    model = SentenceTransformer("intfloat/e5-large-v2", device=_device)
    prefixed = [f"passage: {t[:512]}" for t in texts]
    _apply_thermal_threads()
    log.info("Embedding with E5-large-v2...")
    e5_embs = model.encode(prefixed, batch_size=16, show_progress_bar=True,
                            normalize_embeddings=True)
    e5_embs = np.array(e5_embs, dtype=np.float32)
    pca_e5 = pca_fit(e5_embs)

    results = {"experiment": "Eigenspectrum Comparison"}

    # Full eigenvalue arrays (first 100 for plotting)
    results["bge_m3"] = {
        "model": "BAAI/bge-m3",
        "dim": int(bge_embs.shape[1]),
        "eigenvalues": pca_bge.explained_variance_ratio_.tolist(),
        "cumulative_var": np.cumsum(pca_bge.explained_variance_ratio_).tolist(),
    }
    results["e5_large_v2"] = {
        "model": "intfloat/e5-large-v2",
        "dim": int(e5_embs.shape[1]),
        "eigenvalues": pca_e5.explained_variance_ratio_.tolist(),
        "cumulative_var": np.cumsum(pca_e5.explained_variance_ratio_).tolist(),
    }

    # Summary stats
    cum_bge = np.cumsum(pca_bge.explained_variance_ratio_)
    cum_e5 = np.cumsum(pca_e5.explained_variance_ratio_)

    milestones = [64, 128, 256, 384, 512]
    results["milestones"] = []
    for m in milestones:
        if m < len(cum_bge) and m < len(cum_e5):
            entry = {
                "dims": m,
                "bge_var_explained": round(float(cum_bge[m-1]), 4),
                "e5_var_explained": round(float(cum_e5[m-1]), 4),
            }
            results["milestones"].append(entry)
            log.info(f"  dims={m}: BGE var={cum_bge[m-1]:.4f}  E5 var={cum_e5[m-1]:.4f}")

    # Power law fit: λ_k ∝ k^{-α}
    def fit_power_law(eigenvalues, n_fit=200):
        """Fit power law decay to eigenvalues."""
        k = np.arange(1, min(n_fit, len(eigenvalues)) + 1)
        log_k = np.log(k)
        log_lam = np.log(eigenvalues[:len(k)] + 1e-20)
        # Linear regression in log-log space
        coeffs = np.polyfit(log_k, log_lam, 1)
        return -coeffs[0]  # alpha (negative slope)

    alpha_bge = fit_power_law(pca_bge.explained_variance_)
    alpha_e5 = fit_power_law(pca_e5.explained_variance_)
    results["power_law"] = {
        "bge_alpha": round(float(alpha_bge), 3),
        "e5_alpha": round(float(alpha_e5), 3),
    }
    log.info(f"Power law decay: BGE α={alpha_bge:.3f}  E5 α={alpha_e5:.3f}")

    # Effective dimensionality (number of dims for 95% variance)
    eff_dim_bge = int(np.searchsorted(cum_bge, 0.95)) + 1
    eff_dim_e5 = int(np.searchsorted(cum_e5, 0.95)) + 1
    results["effective_dim"] = {
        "bge_95pct": eff_dim_bge,
        "e5_95pct": eff_dim_e5,
    }
    log.info(f"Effective dim (95% var): BGE={eff_dim_bge}  E5={eff_dim_e5}")

    out_path = RESULTS_DIR / "exp6_eigenspectrum.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {out_path}")
    return results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="IEEE TAI Reviewer Experiments")
    parser.add_argument("--exp", type=str, default="all",
                        help="Experiment to run: e5, sts, ood, retrieval, crosslingual, eigenspectrum, all")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    args = parser.parse_args()

    np.random.seed(42)
    start = time.time()

    # Pre-set conservative thread count before controller starts
    torch.set_num_threads(4)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    # Start thermal controller — monitors CPU temp and adjusts thread count
    _thermal.start()
    _apply_thermal_threads()
    log.info(f"Thermal controller started (target=80°C, max_threads=10)")

    experiments = {
        "e5": ("E5-large-v2", run_e5_experiment),
        "sts": ("STS Benchmark", run_sts_experiment),
        "ood": ("Out-of-Domain", run_ood_experiment),
        "retrieval": ("Expanded Retrieval", run_expanded_retrieval),
        "crosslingual": ("Cross-Lingual UDHR", run_crosslingual_experiment),
        "eigenspectrum": ("Eigenspectrum", run_eigenspectrum_experiment),
    }

    if args.all or args.exp == "all":
        to_run = list(experiments.keys())
    else:
        to_run = [args.exp]

    all_results = {}
    for key in to_run:
        if key not in experiments:
            log.error(f"Unknown experiment: {key}")
            continue
        name, func = experiments[key]
        log.info(f"\n{'='*60}")
        log.info(f"Starting: {name}")
        log.info(f"{'='*60}")
        try:
            _apply_thermal_threads()  # Adjust threads before each experiment
            result = func()
            all_results[key] = result
            log.info(f"Completed: {name}")
        except Exception as e:
            log.error(f"Failed: {name}: {e}", exc_info=True)
            all_results[key] = {"error": str(e)}

    # Save combined results
    combined_path = RESULTS_DIR / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    _thermal.stop()
    log.info(f"Thermal summary: {_thermal.summary()}")

    elapsed = time.time() - start
    log.info(f"\nAll experiments completed in {elapsed:.0f}s")
    log.info(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
