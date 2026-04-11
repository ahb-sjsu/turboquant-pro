#!/usr/bin/env python3
"""
Reranking experiment for ALL methods — apples-to-apples comparison.

Addresses reviewer criticism: if reranking is the recommended practice,
benchmark ALL methods with reranking, not just ours.

Run on Atlas via: sudo -u postgres python3 reranking_all_methods.py
"""
import time
import numpy as np
import psycopg2


N_QUERIES = 50
TOP_K = 10
OVERSAMPLE_FACTORS = [1, 2, 5, 10]
DIM = 1024
SEED = 42
SAMPLE_SIZE = 50000


def brute_force_search(query, corpus, corpus_norms, k):
    """Exact brute-force top-k by cosine similarity."""
    sims = corpus @ query / (corpus_norms * np.linalg.norm(query) + 1e-30)
    return np.argsort(-sims)[:k]


def compressed_search_with_reranking(
    queries, corpus, compressed_corpus, compressed_norms,
    compressed_queries, gt, factors, top_k
):
    """Run search on compressed, rerank with exact, for multiple oversample factors."""
    results = {}
    for factor in factors:
        fetch_k = top_k * factor
        total_recall = 0.0
        for i in range(len(queries)):
            # Search with compressed
            sims = compressed_corpus @ compressed_queries[i] / (
                compressed_norms * np.linalg.norm(compressed_queries[i]) + 1e-30
            )
            candidates = np.argsort(-sims)[:fetch_k]
            # Rerank with exact original vectors
            exact_sims = corpus[candidates] @ queries[i] / (
                np.linalg.norm(corpus[candidates], axis=1) * np.linalg.norm(queries[i]) + 1e-30
            )
            reranked = [candidates[j] for j in np.argsort(-exact_sims)[:top_k]]
            total_recall += len(set(reranked) & gt[i]) / top_k
        results[factor] = total_recall / len(queries)
    return results


def main():
    print("=" * 80)
    print("ALL-METHODS RERANKING BENCHMARK (production data)")
    print("=" * 80)

    conn = psycopg2.connect(dbname="atlas", user="postgres")
    cur = conn.cursor()

    print(f"\nFetching {SAMPLE_SIZE} embeddings...")
    t0 = time.time()
    cur.execute(
        "SELECT embedding::float4[] FROM ethics_chunks "
        "ORDER BY random() LIMIT %s", (SAMPLE_SIZE,)
    )
    rows = cur.fetchall()
    embeddings = np.array([r[0] for r in rows], dtype=np.float32)
    print(f"  Fetched {len(embeddings)} in {time.time()-t0:.1f}s")

    queries = embeddings[:N_QUERIES]
    corpus = embeddings[N_QUERIES:]
    corpus_norms = np.linalg.norm(corpus, axis=1)
    n_corpus = len(corpus)

    # Ground truth
    print("Computing ground truth...")
    gt = []
    for q in queries:
        idx = brute_force_search(q, corpus, corpus_norms, TOP_K)
        gt.append(set(idx.tolist()))

    from turboquant_pro.pgvector import TurboQuantPGVector
    from turboquant_pro.pca import PCAMatryoshka

    methods = []

    # --- Scalar int8 (simulate: round to int8 range) ---
    print("\nScalar int8...")
    # Simulate int8: scale to [-128, 127], round, scale back
    corpus_min = corpus.min(axis=0)
    corpus_max = corpus.max(axis=0)
    scale = 255.0 / (corpus_max - corpus_min + 1e-30)
    int8_corpus = np.round((corpus - corpus_min) * scale).astype(np.float32) / scale + corpus_min
    int8_queries = np.round((queries - corpus_min) * scale).astype(np.float32) / scale + corpus_min
    int8_norms = np.linalg.norm(int8_corpus, axis=1)
    methods.append(("Scalar int8", "4x", int8_corpus, int8_norms, int8_queries))

    # --- TQ3 uniform ---
    print("TQ3 uniform...")
    tq = TurboQuantPGVector(dim=DIM, bits=3, seed=SEED)
    tq3_corpus = np.empty_like(corpus)
    for i in range(n_corpus):
        tq3_corpus[i] = tq.decompress_embedding(tq.compress_embedding(corpus[i]))
    tq3_queries = np.empty_like(queries)
    for i in range(N_QUERIES):
        tq3_queries[i] = tq.decompress_embedding(tq.compress_embedding(queries[i]))
    tq3_norms = np.linalg.norm(tq3_corpus, axis=1)
    methods.append(("TQ3 uniform", "10.5x", tq3_corpus, tq3_norms, tq3_queries))

    # --- PCA-384 + TQ3 ---
    print("PCA-384 + TQ3...")
    pca384 = PCAMatryoshka(input_dim=DIM, output_dim=384)
    pca384.fit(corpus[:5000])
    pipe384 = pca384.with_quantizer(bits=3, seed=SEED)
    pca384_corpus = np.empty((n_corpus, DIM), dtype=np.float32)
    for i in range(n_corpus):
        pca384_corpus[i] = pipe384.decompress(pipe384.compress(corpus[i]))
    pca384_queries = np.empty((N_QUERIES, DIM), dtype=np.float32)
    for i in range(N_QUERIES):
        pca384_queries[i] = pipe384.decompress(pipe384.compress(queries[i]))
    pca384_norms = np.linalg.norm(pca384_corpus, axis=1)
    methods.append(("PCA-384+TQ3", "27.7x", pca384_corpus, pca384_norms, pca384_queries))

    # --- PCA-256 + TQ3 ---
    print("PCA-256 + TQ3...")
    pca256 = PCAMatryoshka(input_dim=DIM, output_dim=256)
    pca256.fit(corpus[:5000])
    pipe256 = pca256.with_quantizer(bits=3, seed=SEED)
    pca256_corpus = np.empty((n_corpus, DIM), dtype=np.float32)
    for i in range(n_corpus):
        pca256_corpus[i] = pipe256.decompress(pipe256.compress(corpus[i]))
    pca256_queries = np.empty((N_QUERIES, DIM), dtype=np.float32)
    for i in range(N_QUERIES):
        pca256_queries[i] = pipe256.decompress(pipe256.compress(queries[i]))
    pca256_norms = np.linalg.norm(pca256_corpus, axis=1)
    methods.append(("PCA-256+TQ3", "41x", pca256_corpus, pca256_norms, pca256_queries))

    # --- Binary quantization ---
    print("Binary quantization...")
    bin_corpus = np.where(corpus > 0, 1.0, -1.0).astype(np.float32)
    bin_queries = np.where(queries > 0, 1.0, -1.0).astype(np.float32)
    bin_norms = np.linalg.norm(bin_corpus, axis=1)
    methods.append(("Binary", "32x", bin_corpus, bin_norms, bin_queries))

    # --- Product quantization (simulate via PCA-64 + TQ3 as proxy) ---
    print("PQ proxy (PCA-64+TQ3)...")
    pca64 = PCAMatryoshka(input_dim=DIM, output_dim=64)
    pca64.fit(corpus[:5000])
    pipe64 = pca64.with_quantizer(bits=3, seed=SEED)
    pq_corpus = np.empty((n_corpus, DIM), dtype=np.float32)
    for i in range(n_corpus):
        pq_corpus[i] = pipe64.decompress(pipe64.compress(corpus[i]))
    pq_queries = np.empty((N_QUERIES, DIM), dtype=np.float32)
    for i in range(N_QUERIES):
        pq_queries[i] = pipe64.decompress(pipe64.compress(queries[i]))
    pq_norms = np.linalg.norm(pq_corpus, axis=1)
    methods.append(("PQ proxy", "256x", pq_corpus, pq_norms, pq_queries))

    # Run all methods
    print("\n" + "=" * 80)
    print(f"RESULTS (corpus={n_corpus}, queries={N_QUERIES}, k={TOP_K})")
    print("=" * 80)
    header = f"\n{'Method':>20s} {'Ratio':>6s}"
    for f in OVERSAMPLE_FACTORS:
        header += f"  fetch={TOP_K*f:>4d}"
    print(header)
    print("-" * 72)

    for name, ratio, comp_corpus, comp_norms, comp_queries in methods:
        results = compressed_search_with_reranking(
            queries, corpus, comp_corpus, comp_norms,
            comp_queries, gt, OVERSAMPLE_FACTORS, TOP_K
        )
        line = f"{name:>20s} {ratio:>6s}"
        for f in OVERSAMPLE_FACTORS:
            line += f"  {results[f]:>8.1%}"
        print(line)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
