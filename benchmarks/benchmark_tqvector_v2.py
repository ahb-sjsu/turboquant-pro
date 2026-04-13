#!/usr/bin/env python3
"""tqvector benchmark v2 — optimized insert via pgvector-python adapter
and execute_batch, plus real-embedding support.

Key wins over v1:
  - pgvector.psycopg2.register_vector(): numpy -> pgvector wire in C,
    no Python TSV buffer building
  - psycopg2.extras.execute_batch with chunks of 1000 for tqvector
    (where compression runs inside PG via tq_compress on float4[])
  - Optional real embeddings from sentence-transformers (MiniLM/BGE)
    instead of synthetic Gaussian

Run on Atlas as postgres user:
    sudo -u postgres /home/claude/env/bin/python3 -u \\
        /home/claude/bench_v2.py --dims 768 1536 --scale 100000 --queries 30
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
import psycopg2
import psycopg2.extras

try:
    from pgvector.psycopg2 import register_vector
    PGVECTOR_ADAPTER = True
except ImportError:
    PGVECTOR_ADAPTER = False


def make_synthetic(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)
    return x


def make_real_embeddings(n: int, seed: int = 42) -> np.ndarray:
    """Generate real embeddings from sentence-transformers/all-MiniLM-L6-v2
    on random WikiText snippets. Returns (n, 384) normalized float32.
    """
    from sentence_transformers import SentenceTransformer  # lazy
    import os
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Use WikiText test as source
    wiki_path = "/home/claude/datasets/wikitext/wiki.test.raw"
    text = open(wiki_path).read()
    # Split into ~50-word chunks
    words = text.split()
    chunks = []
    rng = np.random.default_rng(seed)
    while len(chunks) < n:
        start = rng.integers(0, max(1, len(words) - 50))
        chunks.append(" ".join(words[start:start + 50]))
    chunks = chunks[:n]
    emb = model.encode(chunks, batch_size=256, show_progress_bar=True,
                       normalize_embeddings=True)
    return emb.astype(np.float32)


def exact_top_k(queries: np.ndarray, corpus: np.ndarray, k: int) -> np.ndarray:
    sims = queries @ corpus.T
    order = np.argpartition(-sims, kth=k, axis=1)[:, :k]
    for i in range(queries.shape[0]):
        order[i] = order[i][np.argsort(-sims[i, order[i]])]
    return order


def recall_at_k(exact: np.ndarray, approx: np.ndarray) -> float:
    k = exact.shape[1]
    n = exact.shape[0]
    return sum(len(set(exact[i]) & set(approx[i])) / k for i in range(n)) / n


def insert_vector(cur, table: str, corpus: np.ndarray, chunk: int = 2000) -> float:
    """Insert corpus into vector(dim) table via pgvector adapter.

    Uses execute_values with multi-VALUES to amortize round-trips.
    """
    t0 = time.perf_counter()
    rows = [(i, v) for i, v in enumerate(corpus)]
    psycopg2.extras.execute_values(
        cur,
        f"INSERT INTO {table}(id, v) VALUES %s",
        rows,
        page_size=chunk,
    )
    return time.perf_counter() - t0


def insert_tqvector(cur, table: str, corpus: np.ndarray, bits: int,
                     chunk: int = 2000) -> float:
    """Insert corpus into tqvector table. Uses tq_compress() in SQL on
    float4[] passed via execute_values (multi-row VALUES).
    """
    t0 = time.perf_counter()
    rows = [(i, v.tolist(), bits) for i, v in enumerate(corpus)]
    # Template controls how each row is serialized in VALUES
    psycopg2.extras.execute_values(
        cur,
        f"INSERT INTO {table}(id, v) VALUES %s",
        rows,
        template="(%s, tq_compress(%s::float4[], %s))",
        page_size=chunk,
    )
    return time.perf_counter() - t0


def run_cell(conn, dim: int, scale: int, storage: str,
             corpus: np.ndarray, queries: np.ndarray,
             exact_top: np.ndarray, k: int) -> dict:
    table = f"bench_{storage}_{dim}"
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table};")
        if storage == "vector":
            cur.execute(f"CREATE TABLE {table} (id int, v vector({dim}));")
        elif storage.startswith("tqvector_"):
            cur.execute(f"CREATE TABLE {table} (id int, v tqvector);")
        else:
            raise ValueError(storage)
    conn.commit()

    print(f"    {storage}: inserting {scale} rows...", flush=True)
    with conn.cursor() as cur:
        if storage == "vector":
            insert_seconds = insert_vector(cur, table, corpus)
        else:
            bits = int(storage.split("_")[1])
            insert_seconds = insert_tqvector(cur, table, corpus, bits)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(f"SELECT pg_total_relation_size('{table}');")
        size_bytes = cur.fetchone()[0]

    # Queries
    latencies = []
    approx_ids = np.zeros((len(queries), k), dtype=np.int64)
    for qi in range(len(queries)):
        q = queries[qi]
        t_q = time.perf_counter()
        with conn.cursor() as cur:
            if storage == "vector":
                cur.execute(
                    f"SELECT id FROM {table} ORDER BY v <=> %s LIMIT %s",
                    (q, k),
                )
            else:
                bits = int(storage.split("_")[1])
                cur.execute(
                    f"SELECT id FROM {table} "
                    f"ORDER BY v <=> tq_compress(%s::float4[], %s) LIMIT %s",
                    (q.tolist(), bits, k),
                )
            rows = cur.fetchall()
        latencies.append(time.perf_counter() - t_q)
        ids = [r[0] for r in rows]
        while len(ids) < k:
            ids.append(-1)
        approx_ids[qi] = ids[:k]

    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table};")
    conn.commit()

    recall = recall_at_k(exact_top, approx_ids)
    return {
        "storage": storage, "dim": dim, "scale": scale,
        "size_bytes": size_bytes,
        "bytes_per_vec": size_bytes / scale,
        "insert_seconds": insert_seconds,
        "insert_rate": scale / insert_seconds,
        "query_p50_ms": statistics.median(latencies) * 1000,
        "query_p95_ms": sorted(latencies)[int(0.95 * len(latencies))] * 1000,
        "recall_at_10": recall,
        "n_queries": len(queries),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dims", type=int, nargs="+", default=[768])
    p.add_argument("--scale", type=int, default=100_000)
    p.add_argument("--queries", type=int, default=30)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--storages", nargs="+",
        default=["vector", "tqvector_4", "tqvector_3", "tqvector_2"],
    )
    p.add_argument("--output", type=str, default="/tmp/tqv_v2.json")
    p.add_argument("--real", action="store_true",
                   help="Use sentence-transformers MiniLM embeddings (384-dim only)")
    args = p.parse_args()

    if not PGVECTOR_ADAPTER:
        print("WARNING: pgvector-python not available, falling back slower")

    conn = psycopg2.connect(
        host="/var/run/postgresql", dbname="atlas", user="postgres",
    )
    if PGVECTOR_ADAPTER:
        register_vector(conn)
    # psycopg2 defaults to autocommit=False; setting it after register_vector
    # would error because register_vector ran a SELECT to query pg_type.

    results = []
    for dim in args.dims:
        print(f"\n=== dim={dim} scale={args.scale} ===", flush=True)
        if args.real and dim != 384:
            print(f"  skip real mode for dim!=384", flush=True)
            continue
        t0 = time.perf_counter()
        if args.real:
            corpus = make_real_embeddings(args.scale, args.seed)
        else:
            corpus = make_synthetic(args.scale, dim, args.seed)
        queries = make_synthetic(args.queries, dim, args.seed + 1) if not args.real \
            else make_real_embeddings(args.queries, args.seed + 1)
        print(f"  corpus ready ({time.perf_counter() - t0:.1f}s)", flush=True)
        t0 = time.perf_counter()
        exact_top = exact_top_k(queries, corpus, args.top_k)
        print(f"  exact top-k ({time.perf_counter() - t0:.1f}s)", flush=True)
        for storage in args.storages:
            try:
                row = run_cell(conn, dim, args.scale, storage,
                               corpus, queries, exact_top, args.top_k)
                print(f"  {storage}: {row}", flush=True)
                results.append(row)
            except Exception as e:
                conn.rollback()
                print(f"  {storage}: FAILED {e}", flush=True)
                import traceback
                traceback.print_exc()
                results.append({"storage": storage, "dim": dim, "error": str(e)})

    conn.close()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\n| dim | storage | bytes/vec | insert r/s | q_p50 ms | q_p95 ms | recall@10 |")
    print("|---:|---|---:|---:|---:|---:|---:|")
    for r in results:
        if "error" in r:
            print(f"| {r.get('dim','?')} | {r['storage']} | ERR | ERR | ERR | ERR | ERR |")
            continue
        print(f"| {r['dim']} | {r['storage']} | {r['bytes_per_vec']:.1f} | "
              f"{r['insert_rate']:.0f} | {r['query_p50_ms']:.1f} | "
              f"{r['query_p95_ms']:.1f} | {r['recall_at_10']:.4f} |")


if __name__ == "__main__":
    main()
