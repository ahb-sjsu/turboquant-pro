#!/usr/bin/env python3
"""tqvector benchmark designed to run on Atlas itself (not via SSH).

Uses psycopg2 over Unix socket + binary COPY for efficient data loading.
Much faster than SSH-based variants since no SFTP TSV round-trip and no
per-query psql spawn.

Run as:
    sudo -u postgres /home/claude/env/bin/python3 \
        /home/claude/bench_tqvector_atlas_local.py \
        --dims 768 1536 --scale 100000 --queries 20
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import time

import numpy as np
import psycopg2


def make_synthetic(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)
    return x


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


def vec_literal(v: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.6g}" for x in v) + "]"


def run_cell(
    conn,
    dim: int,
    scale: int,
    storage: str,
    corpus: np.ndarray,
    queries: np.ndarray,
    exact_top: np.ndarray,
    k: int,
) -> dict:
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

    # Stage raw fp32 into a temp table via binary COPY-like path.
    # psycopg2 copy_expert with text format + pgvector array literal is simplest.
    print(f"    {storage}: inserting {scale} rows...", flush=True)
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(f"CREATE TEMP TABLE staging_{table} (id int, v_raw float4[]);")
        # Stream TSV into staging via copy_expert
        buf = io.StringIO()
        for i, v in enumerate(corpus):
            buf.write(f"{i}\t" + "{" + ",".join(f"{float(x):.6g}" for x in v) + "}\n")
        buf.seek(0)
        cur.copy_expert(
            f"COPY staging_{table}(id, v_raw) FROM STDIN "
            f"WITH (FORMAT text, DELIMITER E'\\t')",
            buf,
        )
        if storage == "vector":
            cur.execute(
                f"INSERT INTO {table}(id, v) "
                f"SELECT id, v_raw::vector FROM staging_{table};"
            )
        else:
            bits = int(storage.split("_")[1])
            cur.execute(
                f"INSERT INTO {table}(id, v) "
                f"SELECT id, tq_compress(v_raw, {bits}) FROM staging_{table};"
            )
    conn.commit()
    insert_seconds = time.perf_counter() - t0

    with conn.cursor() as cur:
        cur.execute(f"SELECT pg_total_relation_size('{table}');")
        size_bytes = cur.fetchone()[0]

    # Query timing — single persistent connection, no psql spawn
    latencies = []
    approx_ids = np.zeros((len(queries), k), dtype=np.int64)
    for qi in range(len(queries)):
        q = queries[qi]
        q_lit = vec_literal(q)
        if storage == "vector":
            sql = (
                f"SELECT id FROM {table} "
                f"ORDER BY v <=> '{q_lit}'::vector LIMIT {k};"
            )
        else:
            bits = int(storage.split("_")[1])
            q_arr = "ARRAY[" + ",".join(f"{float(x):.6g}" for x in q) + "]::float4[]"
            sql = (
                f"SELECT id FROM {table} "
                f"ORDER BY v <=> tq_compress({q_arr}, {bits}) "
                f"LIMIT {k};"
            )
        t_q = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(sql)
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
        "storage": storage,
        "dim": dim,
        "scale": scale,
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
    p.add_argument("--queries", type=int, default=20)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--storages",
        nargs="+",
        default=["vector", "tqvector_4", "tqvector_3", "tqvector_2"],
    )
    p.add_argument("--output", type=str, default="/tmp/tqv_bench_atlas.json")
    args = p.parse_args()

    conn = psycopg2.connect(
        host="/var/run/postgresql",  # Unix socket
        dbname="atlas",
        user="postgres",
    )
    conn.autocommit = False

    results = []
    for dim in args.dims:
        print(f"\n=== dim={dim} scale={args.scale} ===", flush=True)
        print("  generating corpus...", flush=True)
        corpus = make_synthetic(args.scale, dim, args.seed)
        queries = make_synthetic(args.queries, dim, args.seed + 1)
        print("  computing exact top-k ground truth...", flush=True)
        exact_top = exact_top_k(queries, corpus, args.top_k)
        for storage in args.storages:
            try:
                row = run_cell(
                    conn,
                    dim,
                    args.scale,
                    storage,
                    corpus,
                    queries,
                    exact_top,
                    args.top_k,
                )
                print(f"  {storage}: {row}", flush=True)
                results.append(row)
            except Exception as e:
                conn.rollback()
                print(f"  {storage}: FAILED {e}", flush=True)
                results.append({"storage": storage, "dim": dim, "error": str(e)})

    conn.close()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(
        "\n\n| dim | storage | bytes/vec | insert r/s | q_p50 ms | q_p95 ms | recall@10 |"
    )
    print("|---:|---|---:|---:|---:|---:|---:|")
    for r in results:
        if "error" in r:
            continue
        print(
            f"| {r['dim']} | {r['storage']} | {r['bytes_per_vec']:.1f} | "
            f"{r['insert_rate']:.0f} | {r['query_p50_ms']:.1f} | "
            f"{r['query_p95_ms']:.1f} | {r['recall_at_10']:.4f} |"
        )


if __name__ == "__main__":
    main()
