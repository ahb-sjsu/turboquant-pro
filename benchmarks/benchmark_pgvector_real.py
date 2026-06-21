#!/usr/bin/env python3
"""pgvector (fp32) vs tqvector (compressed) on REAL embeddings, inside PostgreSQL.

Demonstrates turboquant-pro's SQL-native path: compressed vectors are stored
*and searched* in Postgres via the `tqvector` type and the `<=>` operator -- no
decompression, no separate ANN service.

Run on Atlas as the postgres user (it has the venv + tqvector/vector extensions):
  sudo -u postgres /home/claude/env/bin/python3 benchmark_pgvector_real.py \\
      --npy /tmp/labse_bench.npy --corpus 50000 --queries 500
"""

import argparse
import io
import json
import statistics
import time

import numpy as np
import psycopg2


def exact_topk(Q, C, k):
    sims = Q @ C.T
    idx = np.argpartition(-sims, k, axis=1)[:, :k]
    for i in range(len(Q)):
        idx[i] = idx[i][np.argsort(-sims[i, idx[i]])]
    return idx


def recall(gt, ap, k):
    return float(
        np.mean([len(set(gt[i, :k]) & set(ap[i, :k])) / k for i in range(len(gt))])
    )


def vlit(v):
    return "[" + ",".join(f"{float(x):.6g}" for x in v) + "]"


def alit(v):
    return "ARRAY[" + ",".join(f"{float(x):.6g}" for x in v) + "]::float4[]"


def run_cell(conn, storage, C, Q, gt, k, dim):
    table = f"bench_real_{storage}"
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {table}")
    if storage == "vector":
        cur.execute(f"CREATE TABLE {table}(id int, v vector({dim}))")
    else:
        cur.execute(f"CREATE TABLE {table}(id int, v tqvector)")
    conn.commit()

    t0 = time.perf_counter()
    cur.execute(f"CREATE TEMP TABLE stg_{table}(id int, v_raw float4[])")
    buf = io.StringIO()
    for i, v in enumerate(C):
        buf.write(f"{i}\t{{" + ",".join(f"{float(x):.6g}" for x in v) + "}\n")
    buf.seek(0)
    cur.copy_expert(
        f"COPY stg_{table}(id,v_raw) FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')",
        buf,
    )
    if storage == "vector":
        cur.execute(
            f"INSERT INTO {table}(id,v) SELECT id, v_raw::vector FROM stg_{table}"
        )
    else:
        bits = int(storage.split("_")[1])
        cur.execute(
            f"INSERT INTO {table}(id,v) SELECT id, tq_compress(v_raw,{bits}) FROM stg_{table}"
        )
    conn.commit()
    insert_s = time.perf_counter() - t0
    cur.execute(f"SELECT pg_total_relation_size('{table}')")
    size = cur.fetchone()[0]

    lat = []
    approx = np.zeros((len(Q), k), dtype=np.int64)
    for qi in range(len(Q)):
        q = Q[qi]
        if storage == "vector":
            sql = f"SELECT id FROM {table} ORDER BY v <=> '{vlit(q)}'::vector LIMIT {k}"
        else:
            bits = int(storage.split("_")[1])
            sql = f"SELECT id FROM {table} ORDER BY v <=> tq_compress({alit(q)},{bits}) LIMIT {k}"
        t = time.perf_counter()
        cur.execute(sql)
        rows = cur.fetchall()
        lat.append(time.perf_counter() - t)
        ids = [r[0] for r in rows] + [-1] * k
        approx[qi] = ids[:k]
    cur.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()
    return dict(
        storage=storage,
        bytes_per_vec=size / len(C),
        insert_s=round(insert_s, 2),
        q_p50_ms=round(statistics.median(lat) * 1000, 2),
        q_p95_ms=round(sorted(lat)[int(0.95 * len(lat))] * 1000, 2),
        recall_at_10=round(recall(gt, approx, k), 4),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True)
    ap.add_argument("--corpus", type=int, default=50000)
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--storages",
        nargs="+",
        default=["vector", "tqvector_4", "tqvector_3", "tqvector_2"],
    )
    ap.add_argument("--json", default="/tmp/pgreal.json")
    a = ap.parse_args()

    X = np.load(a.npy, mmap_mode="r").astype(np.float32)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
    Q = X[: a.queries].copy()
    C = X[a.queries : a.queries + a.corpus].copy()
    dim = X.shape[1]
    print(f"corpus={len(C)} dim={dim} queries={len(Q)}", flush=True)
    gt = exact_topk(Q, C, 100)

    conn = psycopg2.connect(host="/var/run/postgresql", dbname="atlas", user="postgres")
    res = []
    for s in a.storages:
        try:
            r = run_cell(conn, s, C, Q, gt, a.k, dim)
            print(r, flush=True)
            res.append(r)
        except Exception as e:
            conn.rollback()
            print(s, "FAILED", e, flush=True)
    conn.close()

    json.dump(res, open(a.json, "w"), indent=2)
    print(
        "\n| storage | bytes/vec | comp x | insert s | q_p50 ms | q_p95 ms | recall@10 |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in res:
        print(
            f"| {r['storage']} | {r['bytes_per_vec']:.1f} | {dim*4/r['bytes_per_vec']:.1f} "
            f"| {r['insert_s']} | {r['q_p50_ms']} | {r['q_p95_ms']} | {r['recall_at_10']} |"
        )


if __name__ == "__main__":
    main()
