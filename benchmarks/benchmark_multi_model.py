"""
Multi-model PCA-Matryoshka evaluation on STS-Benchmark corpus.

Uses 15K+ unique sentences from STS-B (a standard NLP benchmark)
embedded by multiple models, with fixed PCA dimensions for fair
comparison across architectures.

Usage:
    python benchmarks/benchmark_multi_model.py
"""

from __future__ import annotations

import time

import numpy as np


def load_stsb_sentences(max_sentences: int = 15000) -> list[str]:
    """Load unique sentences from the STS-Benchmark dataset."""
    from datasets import load_dataset

    ds = load_dataset("mteb/stsbenchmark-sts")
    sentences: set[str] = set()
    for split_name in ["train", "validation", "test"]:
        for row in ds[split_name]:
            sentences.add(row["sentence1"])
            sentences.add(row["sentence2"])
            if len(sentences) >= max_sentences:
                break
    return sorted(sentences)[:max_sentences]


def benchmark_model(
    model_name: str,
    corpus: list[str],
    pca_dim: int,
) -> dict | None:
    """Run PCA-Matryoshka evaluation on one embedding model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  sentence-transformers not installed, skipping")
        return None

    import gc

    print(f"  Loading {model_name}...", end=" ", flush=True)
    try:
        model = SentenceTransformer(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"FAILED: {e}")
        return None

    dim = model.get_embedding_dimension()
    print(f"dim={dim}, encoding {len(corpus)} sentences...", end=" ", flush=True)

    t0 = time.perf_counter()
    embeddings = model.encode(corpus, show_progress_bar=False, batch_size=64)
    embeddings = np.array(embeddings, dtype=np.float32)
    encode_time = time.perf_counter() - t0
    print(f"{encode_time:.1f}s")

    # Free model memory before running benchmarks
    del model
    gc.collect()

    n = len(embeddings)
    results = {"model": model_name, "dim": dim, "n_vectors": n, "pca_dim": pca_dim}

    from turboquant_pro.pca import PCAMatryoshka
    from turboquant_pro.pgvector import TurboQuantPGVector

    # Use first 80% for PCA fitting, last 20% for evaluation
    n_train = int(n * 0.8)
    train_embs = embeddings[:n_train]
    test_embs = embeddings[n_train:]
    n_test = len(test_embs)

    # 1. Naive truncation
    truncated = test_embs[:, :pca_dim]
    sims_naive = []
    for i in range(n_test):
        a = test_embs[i]
        b_pad = np.zeros(dim, dtype=np.float32)
        b_pad[:pca_dim] = truncated[i]
        denom = np.linalg.norm(a) * np.linalg.norm(b_pad) + 1e-30
        sims_naive.append(float(np.dot(a, b_pad) / denom))
    results["naive_cosine"] = round(float(np.mean(sims_naive)), 4)

    # 2. PCA-Matryoshka truncation
    effective_pca_dim = min(pca_dim, dim - 1, n_train - 1)
    pca = PCAMatryoshka(input_dim=dim, output_dim=effective_pca_dim)
    pca.fit(train_embs)
    sims_pca = []
    for emb in test_embs:
        reduced = pca.transform(emb)
        reconstructed = pca.inverse_transform(reduced)
        denom = np.linalg.norm(emb) * np.linalg.norm(reconstructed) + 1e-30
        sims_pca.append(float(np.dot(emb, reconstructed) / denom))
    results["pca_cosine"] = round(float(np.mean(sims_pca)), 4)
    results["pca_improvement"] = round(
        results["pca_cosine"] - results["naive_cosine"], 4
    )

    # 3. TQ3 (full dim, no PCA)
    tq = TurboQuantPGVector(dim=dim, bits=3, seed=42)
    sims_tq = []
    for emb in test_embs:
        c = tq.compress_embedding(emb)
        r = tq.decompress_embedding(c)
        denom = np.linalg.norm(emb) * np.linalg.norm(r) + 1e-30
        sims_tq.append(float(np.dot(emb, r) / denom))
    results["tq3_cosine"] = round(float(np.mean(sims_tq)), 4)

    # 4. PCA + TQ3 combined
    pipeline = pca.with_quantizer(bits=3, seed=42)
    sims_combined = []
    for emb in test_embs:
        c = pipeline.compress(emb)
        r = pipeline.decompress(c)
        denom = np.linalg.norm(emb) * np.linalg.norm(r) + 1e-30
        sims_combined.append(float(np.dot(emb, r) / denom))
    results["pca_tq3_cosine"] = round(float(np.mean(sims_combined)), 4)

    # 5. Learned codebook
    from turboquant_pro.learned_codebook import fit_codebook

    lq = fit_codebook(train_embs, dim=dim, bits=3, seed=42)
    sims_learned = []
    for emb in test_embs:
        c = lq.compress_embedding(emb)
        r = lq.decompress_embedding(c)
        denom = np.linalg.norm(emb) * np.linalg.norm(r) + 1e-30
        sims_learned.append(float(np.dot(emb, r) / denom))
    results["learned_cosine"] = round(float(np.mean(sims_learned)), 4)
    results["learned_delta"] = round(
        results["learned_cosine"] - results["tq3_cosine"], 4
    )

    return results


def main():
    models = [
        ("all-MiniLM-L6-v2", 192),  # 384-dim -> PCA-192 (50%)
        ("all-mpnet-base-v2", 384),  # 768-dim -> PCA-384 (50%)
        ("BAAI/bge-small-en-v1.5", 192),  # 384-dim -> PCA-192 (50%)
        ("BAAI/bge-base-en-v1.5", 384),  # 768-dim -> PCA-384 (50%)
    ]

    print("=" * 95)
    print("MULTI-MODEL PCA-MATRYOSHKA EVALUATION")
    print("=" * 95)

    print("Loading STS-Benchmark corpus...")
    corpus = load_stsb_sentences(15000)
    print(f"Corpus: {len(corpus)} unique sentences from STS-B")
    print(f"Models: {len(models)}")
    print("Split: 80% train (PCA fitting) / 20% test (evaluation)")
    print()

    all_results = []
    for model_name, pca_dim in models:
        result = benchmark_model(model_name, corpus, pca_dim)
        if result:
            all_results.append(result)
        print()

    if not all_results:
        print("No models evaluated successfully.")
        return

    # Summary table
    print("=" * 95)
    print("RESULTS")
    print("=" * 95)
    print(
        f"{'Model':>25s}  {'Dim':>4s}  {'PCA':>4s}  "
        f"{'Naive':>7s}  {'PCA':>7s}  {'Improv':>7s}  "
        f"{'TQ3':>7s}  {'Learned':>7s}  {'L.delta':>8s}  "
        f"{'PCA+TQ3':>7s}"
    )
    print("-" * 95)

    for r in all_results:
        print(
            f"{r['model']:>25s}  {r['dim']:>4d}  {r['pca_dim']:>4d}  "
            f"{r['naive_cosine']:>7.4f}  "
            f"{r['pca_cosine']:>7.4f}  "
            f"{r['pca_improvement']:>+7.4f}  "
            f"{r['tq3_cosine']:>7.4f}  "
            f"{r['learned_cosine']:>7.4f}  "
            f"{r['learned_delta']:>+8.4f}  "
            f"{r['pca_tq3_cosine']:>7.4f}"
        )

    print()
    print("Notes:")
    print(f"  - Corpus: {len(corpus)} sentences from STS-Benchmark")
    print("  - Train/test split: 80/20 (PCA fit on train, eval on test)")
    print("  - PCA dims: 50% of input dim for all models")
    print("  - Cosine similarity: per-vector fidelity (see paper for")
    print("    recall@10 evaluation on production-scale corpus)")


if __name__ == "__main__":
    main()
