#!/usr/bin/env python3
"""Comprehensive tqvector benchmark against pgvector baselines.

For a matrix of (dim, scale, storage type), measures:
  - Storage bytes per vector (via pg_total_relation_size)
  - Insert throughput (rows/sec)
  - Recall@10 vs brute-force fp32 cosine
  - Query latency (p50, p95) at top-10

Storage types compared:
  vector      -- pgvector fp32 (baseline)
  halfvec     -- pgvector fp16 (2x smaller)
  tqvector_2  -- TurboQuant 2-bit
  tqvector_3  -- TurboQuant 3-bit
  tqvector_4  -- TurboQuant 4-bit

Runs everything via SSH + psql against atlas.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import statistics
import time

import numpy as np
import paramiko


ATLAS_HOST = "100.68.134.21"
ATLAS_USER = "claude"
ATLAS_PASSWORD = "roZes9090!~"


def ssh_connect() -> paramiko.SSHClient:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ATLAS_HOST, username=ATLAS_USER, password=ATLAS_PASSWORD, timeout=30)
    return ssh


def psql_exec(ssh: paramiko.SSHClient, sql: str, dbname: str = "atlas",
              timeout: int = 600) -> tuple[int, str]:
    """Run a SQL command on Atlas as postgres user. Returns (rc, output)."""
    cmd = f"sudo -n -u postgres psql -d {dbname} -v ON_ERROR_STOP=1 -At -c $'{sql}'"
    t = ssh.get_transport()
    chan = t.open_session()
    chan.settimeout(timeout)
    chan.exec_command(cmd)
    out = b""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            break
        if chan.recv_ready():
            out += chan.recv(65536)
        if chan.recv_stderr_ready():
            out += chan.recv_stderr(65536)
        time.sleep(0.05)
    while chan.recv_ready():
        out += chan.recv(65536)
    while chan.recv_stderr_ready():
        out += chan.recv_stderr(65536)
    return chan.recv_exit_status(), out.decode("utf-8", errors="replace")


def psql_copy_from(ssh: paramiko.SSHClient, table: str, tsv_text: str,
                   timeout: int = 900) -> tuple[int, str]:
    """COPY FROM STDIN via a shell pipe.

    We write tsv_text to a temp file on Atlas, then COPY from it.
    TSV format: id<tab>array_literal<newline>.
    """
    # Write TSV to temp file on Atlas
    sftp = ssh.open_sftp()
    remote_path = f"/tmp/tq_copy_{random.randint(0, 10**9)}.tsv"
    with sftp.file(remote_path, "w") as f:
        f.write(tsv_text)
    sftp.close()

    sql = (f"\\copy {table}(id, v_raw) FROM '{remote_path}' WITH "
           f"(FORMAT text, DELIMITER E'\\t')")
    cmd = f"sudo -n -u postgres psql -d atlas -v ON_ERROR_STOP=1 -c \"{sql}\""

    t = ssh.get_transport()
    chan = t.open_session()
    chan.settimeout(timeout)
    chan.exec_command(cmd)
    out = b""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            break
        if chan.recv_ready():
            out += chan.recv(65536)
        if chan.recv_stderr_ready():
            out += chan.recv_stderr(65536)
        time.sleep(0.05)
    while chan.recv_ready():
        out += chan.recv(65536)
    while chan.recv_stderr_ready():
        out += chan.recv_stderr(65536)
    # Clean up
    ssh.exec_command(f"rm -f {remote_path}")
    return chan.recv_exit_status(), out.decode("utf-8", errors="replace")


def vec_literal(v: np.ndarray) -> str:
    """Render a numpy float32 vector as a pgvector literal '[x,y,z]'."""
    return "[" + ",".join(f"{float(x):.6g}" for x in v) + "]"


def make_synthetic(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Gaussian synthetic embeddings, unit-normalized."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x /= np.maximum(norms, 1e-30)
    return x


def exact_top_k(queries: np.ndarray, corpus: np.ndarray, k: int) -> np.ndarray:
    """Return shape (n_queries, k) top-k indices by cosine similarity.

    Corpus and queries assumed already unit-normalized.
    """
    sims = queries @ corpus.T
    order = np.argpartition(-sims, kth=k, axis=1)[:, :k]
    # sort within top-k
    for i in range(queries.shape[0]):
        top_i = order[i]
        order[i] = top_i[np.argsort(-sims[i, top_i])]
    return order


def recall_at_k(exact_rankings: np.ndarray, approx_rankings: np.ndarray) -> float:
    """Mean recall: |exact ∩ approx| / k averaged over queries."""
    k = exact_rankings.shape[1]
    n = exact_rankings.shape[0]
    total = 0.0
    for i in range(n):
        inter = len(set(exact_rankings[i]) & set(approx_rankings[i]))
        total += inter / k
    return total / n


def run_cell(ssh, dim: int, scale: int, storage: str, corpus: np.ndarray,
             queries: np.ndarray, exact_top: np.ndarray, k: int,
             n_query_probe: int) -> dict:
    """Run one (dim, scale, storage) cell. Returns metrics dict."""
    table = f"bench_{storage}_{dim}"
    # Drop + create
    psql_exec(ssh, f"DROP TABLE IF EXISTS {table};")

    if storage == "vector":
        col_type = f"vector({dim})"
        create_sql = f"CREATE TABLE {table} (id int, v {col_type});"
        insert_fn = lambda v: vec_literal(v)
        psql_exec(ssh, create_sql)
    elif storage == "halfvec":
        col_type = f"halfvec({dim})"
        create_sql = f"CREATE TABLE {table} (id int, v {col_type});"
        insert_fn = lambda v: vec_literal(v)
        psql_exec(ssh, create_sql)
    elif storage.startswith("tqvector_"):
        bits = int(storage.split("_")[1])
        # Compress via tq_compress inside PG: table stores raw fp32, with generated
        # tqvector column. Simpler: INSERT raw, build column via trigger... actually
        # do it all in SQL: CREATE + INSERT from a float4[] staging.
        col_type = "tqvector"
        create_sql = (f"CREATE TABLE {table} (id int, v {col_type});")
        insert_fn = lambda v: vec_literal(v)  # not used; we go via float4[]
        psql_exec(ssh, create_sql)
        # Store raw then compress
    else:
        raise ValueError(f"Unknown storage: {storage}")

    # Insert via COPY from a temp staging table of float4[]
    # Strategy for all types: stage as vector(dim) temp table, then INSERT INTO target.
    # For tqvector we call tq_compress in the INSERT. For vector/halfvec we cast.
    staging = f"{table}_staging"
    psql_exec(ssh, f"DROP TABLE IF EXISTS {staging};")
    psql_exec(ssh, f"CREATE TEMP TABLE {staging} (id int, v_raw float4[]);")

    # Build a TSV payload. Each line: id<tab>{x,y,z}
    def arr_literal(v: np.ndarray) -> str:
        return "{" + ",".join(f"{float(x):.6g}" for x in v) + "}"

    # Use COPY FROM directly (no SFTP to avoid overhead). We pipe stdin.
    sftp = ssh.open_sftp()
    remote_path = f"/tmp/stage_{random.randint(0, 10**9)}.tsv"
    with sftp.file(remote_path, "w") as f:
        for i, v in enumerate(corpus):
            f.write(f"{i}\t{arr_literal(v)}\n")
    sftp.close()

    rc, out = psql_exec(
        ssh,
        f"CREATE TEMP TABLE {staging}(id int, v_raw float4[]); "
        f"\\copy {staging} FROM \\'{remote_path}\\' WITH (FORMAT text, DELIMITER E\\\\t);",
        timeout=900,
    )
    # psql_exec uses -c so temp tables won't persist between invocations; use psql script
    # Simpler: psql -f a file on remote
    script_path = f"/tmp/load_{random.randint(0, 10**9)}.sql"
    sftp = ssh.open_sftp()
    with sftp.file(script_path, "w") as f:
        f.write(f"CREATE TEMP TABLE {staging}(id int, v_raw float4[]);\n")
        f.write(f"\\copy {staging} FROM '{remote_path}' WITH (FORMAT text, DELIMITER E'\\t');\n")
        if storage == "vector":
            f.write(f"INSERT INTO {table}(id, v) SELECT id, v_raw::vector FROM {staging};\n")
        elif storage == "halfvec":
            f.write(f"INSERT INTO {table}(id, v) SELECT id, v_raw::vector::halfvec FROM {staging};\n")
        elif storage.startswith("tqvector_"):
            bits = int(storage.split("_")[1])
            f.write(f"INSERT INTO {table}(id, v) SELECT id, tq_compress(v_raw, {bits}) FROM {staging};\n")
    sftp.close()

    t0 = time.perf_counter()
    cmd = (f"sudo -n -u postgres psql -d atlas -v ON_ERROR_STOP=1 "
           f"-f {script_path} 2>&1")
    t = ssh.get_transport()
    chan = t.open_session()
    chan.settimeout(1800)
    chan.exec_command(cmd)
    out = b""
    deadline = time.time() + 1800
    while time.time() < deadline:
        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            break
        if chan.recv_ready(): out += chan.recv(65536)
        if chan.recv_stderr_ready(): out += chan.recv_stderr(65536)
        time.sleep(0.1)
    while chan.recv_ready(): out += chan.recv(65536)
    while chan.recv_stderr_ready(): out += chan.recv_stderr(65536)
    rc = chan.recv_exit_status()
    insert_seconds = time.perf_counter() - t0
    ssh.exec_command(f"rm -f {script_path} {remote_path}")
    if rc != 0:
        return {"error": out.decode("utf-8", errors="replace")[:500]}

    # Storage size
    rc, size_out = psql_exec(
        ssh,
        f"SELECT pg_total_relation_size(\\'{table}\\');"
    )
    size_bytes = int(size_out.strip().splitlines()[-1]) if size_out.strip() else 0

    # Query: run n_query_probe random queries, ORDER BY cosine distance LIMIT k.
    # For recall, compare against exact_top.
    # To keep SQL round-trips low we issue each query separately and time it.
    latencies = []
    approx_ids = np.zeros((n_query_probe, k), dtype=np.int64)

    for qi in range(n_query_probe):
        q = queries[qi]
        q_lit = vec_literal(q)
        if storage in ("vector", "halfvec"):
            # cast query to column type for operator compat
            cast = "vector" if storage == "vector" else "halfvec"
            sql = (f"SELECT id FROM {table} "
                   f"ORDER BY v <=> '{q_lit}'::{cast} "
                   f"LIMIT {k};")
        else:
            # tqvector
            sql = (f"SELECT id FROM {table} "
                   f"ORDER BY v <=> tq_compress(ARRAY{list(map(float, q))}::float4[], "
                   f"{int(storage.split('_')[1])}) "
                   f"LIMIT {k};")
        t0 = time.perf_counter()
        rc, out = psql_exec(ssh, sql.replace("'", "\\'"), timeout=120)
        dt = time.perf_counter() - t0
        latencies.append(dt)
        ids = []
        for line in out.strip().splitlines():
            s = line.strip()
            if s.isdigit():
                ids.append(int(s))
        while len(ids) < k:
            ids.append(-1)
        approx_ids[qi] = ids[:k]

    # Drop table
    psql_exec(ssh, f"DROP TABLE IF EXISTS {table};")

    recall = recall_at_k(exact_top[:n_query_probe], approx_ids)
    return {
        "storage": storage,
        "dim": dim,
        "scale": scale,
        "size_bytes": size_bytes,
        "bytes_per_vec": size_bytes / scale if scale else 0,
        "insert_seconds": insert_seconds,
        "insert_rate": scale / insert_seconds if insert_seconds else 0,
        "query_p50_ms": statistics.median(latencies) * 1000 if latencies else 0,
        "query_p95_ms": (sorted(latencies)[int(0.95 * len(latencies))] * 1000) if latencies else 0,
        "recall_at_10": recall,
        "n_queries": n_query_probe,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[768, 1024])
    parser.add_argument("--scale", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--storages",
        nargs="+",
        default=["vector", "tqvector_4", "tqvector_3", "tqvector_2"],
    )
    parser.add_argument("--output", type=str, default="/tmp/tqvector_bench.json")
    args = parser.parse_args()

    ssh = ssh_connect()
    results = []
    for dim in args.dims:
        print(f"\n=== dim={dim} scale={args.scale} ===")
        corpus = make_synthetic(args.scale, dim, args.seed)
        queries = make_synthetic(args.queries, dim, args.seed + 1)
        exact_top = exact_top_k(queries, corpus, args.top_k)
        for storage in args.storages:
            print(f"  -> {storage}")
            try:
                row = run_cell(
                    ssh, dim, args.scale, storage, corpus, queries,
                    exact_top, args.top_k, args.queries,
                )
                print(f"     {row}")
                results.append(row)
            except Exception as e:
                print(f"     FAILED: {e}")
                results.append({
                    "storage": storage, "dim": dim, "scale": args.scale,
                    "error": str(e),
                })
    ssh.close()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Markdown table
    print("\n\n| dim | storage | bytes/vec | insert r/s | q_p50 ms | q_p95 ms | recall@10 |")
    print("|---:|---|---:|---:|---:|---:|---:|")
    for r in results:
        if "error" in r:
            print(f"| {r.get('dim', '?')} | {r.get('storage', '?')} | ERR | ERR | ERR | ERR | ERR |  ({r['error'][:80]})")
        else:
            print(f"| {r['dim']} | {r['storage']} | {r['bytes_per_vec']:.1f} | "
                  f"{r['insert_rate']:.0f} | {r['query_p50_ms']:.1f} | "
                  f"{r['query_p95_ms']:.1f} | {r['recall_at_10']:.4f} |")


if __name__ == "__main__":
    main()
