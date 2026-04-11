#!/usr/bin/env python3
"""
Reranking experiment on 2.4M production ethics_chunks corpus.

Run on Atlas via: python3 reranking_production.py

Measures recall@10 with oversampling + exact reranking for:
- TQ3 uniform (10.5x compression)
- PCA-384 + TQ3 (27.7x compression)
"""
import time
import numpy as np
import psycopg2

# Config
N_QUERIES = 50
TOP_K = 10
OVERSAMPLE_FACTORS = [1, 2, 5, 10]
DIM = 1024
PCA_DIM = 384
BITS = 3
SEED = 42
SAMPLE_SIZE = 50000  # subsample for brute-force ground truth

def main():
    print("=" * 80)
    print("RERANKING EXPERIMENT ON 2.4M PRODUCTION CORPUS")
    print("=" * 80)

    conn = psycopg2.connect(dbname="atlas", user="postgres")
    cur = conn.cursor()

    # Fetch a sample of embeddings for ground truth computation
    print(f"\nFetching {SAMPLE_SIZE} embeddings for ground truth...")
    t0 = time.time()
    cur.execute(
        "SELECT id, embedding::float4[] FROM ethics_chunks "
        "ORDER BY random() LIMIT %s",
        (SAMPLE_SIZE,)
    )
    rows = cur.fetchall()
    print(f"  Fetched {len(rows)} rows in {time.time()-t0:.1f}s")

    ids = [r[0] for r in rows]
    embeddings = np.array([r[1] for r in rows], dtype=np.float32)
    n = len(embeddings)
    print(f"  Shape: {embeddings.shape}")

    # Split: first N_QUERIES are queries, rest is corpus
    queries = embeddings[:N_QUERIES]
    query_ids = ids[:N_QUERIES]
    corpus = embeddings[N_QUERIES:]
    corpus_ids = ids[N_QUERIES:]

    # Ground truth: exact brute-force on full corpus sample
    print(f"\nComputing ground truth (brute force, {len(corpus)} vectors)...")
    corpus_norms = np.linalg.norm(corpus, axis=1)
    gt = []
    for i in range(N_QUERIES):
        q = queries[i]
        sims = corpus @ q / (corpus_norms * np.linalg.norm(q) + 1e-30)
        topk_idx = np.argsort(-sims)[:TOP_K]
        gt.append(set(topk_idx.tolist()))

    # --- TQ3 uniform ---
    print("\nCompressing corpus with TQ3 uniform...")
    try:
        from turboquant_pro.pgvector import TurboQuantPGVector
        from turboquant_pro.pca import PCAMatryoshka
    except ImportError:
        # If turboquant not installed on server, use pip
        import subprocess
        subprocess.run(["pip3", "install", "turboquant-pro"], check=True)
        from turboquant_pro.pgvector import TurboQuantPGVector
        from turboquant_pro.pca import PCAMatryoshka

    tq = TurboQuantPGVector(dim=DIM, bits=BITS, seed=SEED)
    tq_compressed = np.empty_like(corpus)
    for i in range(len(corpus)):
        c = tq.compress_embedding(corpus[i])
        tq_compressed[i] = tq.decompress_embedding(c)
    tq_compressed_norms = np.linalg.norm(tq_compressed, axis=1)

    # Compress queries too
    tq_queries = np.empty_like(queries)
    for i in range(N_QUERIES):
        c = tq.compress_embedding(queries[i])
        tq_queries[i] = tq.decompress_embedding(c)

    print("  Measuring TQ3 recall with reranking...")
    tq_results = {}
    for factor in OVERSAMPLE_FACTORS:
        fetch_k = TOP_K * factor
        total_recall = 0.0
        for i in range(N_QUERIES):
            # Search with compressed representations
            sims = tq_compressed @ tq_queries[i] / (
                tq_compressed_norms * np.linalg.norm(tq_queries[i]) + 1e-30
            )
            candidates = np.argsort(-sims)[:fetch_k]
            # Rerank with exact original vectors
            exact_sims = corpus[candidates] @ queries[i] / (
                np.linalg.norm(corpus[candidates], axis=1) * np.linalg.norm(queries[i]) + 1e-30
            )
            reranked = [candidates[j] for j in np.argsort(-exact_sims)[:TOP_K]]
            total_recall += len(set(reranked) & gt[i]) / TOP_K
        tq_results[factor] = total_recall / N_QUERIES

    # --- PCA-384 + TQ3 ---
    print("\nFitting PCA-384 and compressing...")
    pca = PCAMatryoshka(input_dim=DIM, output_dim=PCA_DIM)
    pca.fit(corpus[:5000])  # fit on subsample
    pipeline = pca.with_quantizer(bits=BITS, seed=SEED)

    pca_compressed = np.empty((len(corpus), DIM), dtype=np.float32)
    for i in range(len(corpus)):
        c = pipeline.compress(corpus[i])
        pca_compressed[i] = pipeline.decompress(c)
    pca_compressed_norms = np.linalg.norm(pca_compressed, axis=1)

    pca_queries = np.empty((N_QUERIES, DIM), dtype=np.float32)
    for i in range(N_QUERIES):
        c = pipeline.compress(queries[i])
        pca_queries[i] = pipeline.decompress(c)

    print("  Measuring PCA-384+TQ3 recall with reranking...")
    pca_results = {}
    for factor in OVERSAMPLE_FACTORS:
        fetch_k = TOP_K * factor
        total_recall = 0.0
        for i in range(N_QUERIES):
            sims = pca_compressed @ pca_queries[i] / (
                pca_compressed_norms * np.linalg.norm(pca_queries[i]) + 1e-30
            )
            candidates = np.argsort(-sims)[:fetch_k]
            exact_sims = corpus[candidates] @ queries[i] / (
                np.linalg.norm(corpus[candidates], axis=1) * np.linalg.norm(queries[i]) + 1e-30
            )
            reranked = [candidates[j] for j in np.argsort(-exact_sims)[:TOP_K]]
            total_recall += len(set(reranked) & gt[i]) / TOP_K
        pca_results[factor] = total_recall / N_QUERIES

    # Print results
    print("\n" + "=" * 80)
    print(f"RESULTS (corpus={len(corpus)}, queries={N_QUERIES}, k={TOP_K})")
    print("=" * 80)
    print(f"\n{'Method':>20s}", end="")
    for f in OVERSAMPLE_FACTORS:
        print(f"  fetch={TOP_K*f:>4d}", end="")
    print()
    print("-" * 60)

    print(f"{'TQ3 (10.5x)':>20s}", end="")
    for f in OVERSAMPLE_FACTORS:
        print(f"  {tq_results[f]:>8.1%}", end="")
    print()

    print(f"{'PCA-384+TQ3 (27.7x)':>20s}", end="")
    for f in OVERSAMPLE_FACTORS:
        print(f"  {pca_results[f]:>8.1%}", end="")
    print()

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
