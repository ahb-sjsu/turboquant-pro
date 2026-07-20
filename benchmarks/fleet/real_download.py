# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Download a real web-scale embedding corpus onto Atlas /archive.

CohereLabs/wikipedia-2023-11-embed-multilingual-v3 (en): ~41M Wikipedia
paragraph embeddings, 1024-d Embed-V3 fp32. Direct per-parquet-file downloads
(hf_hub_download + pyarrow) rather than `datasets` streaming: the streaming
client dies on long unauthenticated sessions and cannot resume without
re-decoding everything. This path is resumable per corpus shard and decodes
at pyarrow speed. Writes 1M-row fp32 .npy shards + a held-out query set from
rows past the corpus cap.
"""

import glob
import os
import time

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

OUT = os.environ.get("TQP_REAL_OUT", "/archive/tqp_real/wiki1024")
CAP_ROWS = int(os.environ.get("TQP_REAL_ROWS", str(10_000_000)))
SHARD = 1_000_000
QN = 1000
REPO = "CohereLabs/wikipedia-2023-11-embed-multilingual-v3"
CONFIG = os.environ.get("TQP_REAL_CONFIG", "en")

os.makedirs(OUT, exist_ok=True)
if os.path.exists(f"{OUT}/DOWNLOAD_DONE"):
    print("already complete", flush=True)
    raise SystemExit(0)

files = sorted(
    f
    for f in list_repo_files(REPO, repo_type="dataset")
    if f.startswith(f"{CONFIG}/") and f.endswith(".parquet")
)
print(f"{REPO}/{CONFIG}: {len(files)} parquet files", flush=True)

buf: np.ndarray | None = None
fill = 0
shard = int(len(glob.glob(f"{OUT}/part_*.npy")))  # resume after complete shards
total = shard * SHARD
skip = total  # rows already banked -> skip this many stream rows
queries: list[np.ndarray] = []
qdone = 0
t0 = time.time()
print(f"resuming at shard {shard} ({total} rows banked)", flush=True)

for f in files:
    if total >= CAP_ROWS and qdone >= QN:
        break
    local = hf_hub_download(REPO, f, repo_type="dataset")
    col = pq.read_table(local, columns=["emb"])["emb"]
    emb = np.stack(col.to_numpy(zero_copy_only=False)).astype(np.float32)
    os.unlink(local)  # keep the HF cache from accumulating 40 GB
    if buf is None:
        buf = np.empty((SHARD, emb.shape[1]), dtype=np.float32)
        with open(f"{OUT}/META.txt", "w") as fh:
            fh.write(f"{REPO} {CONFIG} dim={emb.shape[1]}\n")
    i = 0
    if skip > 0:
        adv = min(skip, len(emb))
        skip -= adv
        i = adv
    while i < len(emb):
        if total < CAP_ROWS:
            take = min(SHARD - fill, len(emb) - i, CAP_ROWS - total)
            buf[fill : fill + take] = emb[i : i + take]
            fill += take
            total += take
            i += take
            if fill == SHARD:
                p = f"{OUT}/part_{shard:03d}.npy"
                np.save(p + ".tmp.npy", buf)
                os.replace(p + ".tmp.npy", p)
                print(
                    f"shard {shard} done ({total} rows, "
                    f"{total / max(time.time() - t0, 1):.0f} rows/s)",
                    flush=True,
                )
                shard += 1
                fill = 0
        elif qdone < QN:
            take = min(QN - qdone, len(emb) - i)
            queries.append(emb[i : i + take])
            qdone += take
            i += take
        else:
            break

np.save(f"{OUT}/queries.npy", np.concatenate(queries)[:QN])
with open(f"{OUT}/DOWNLOAD_DONE", "w") as fh:
    fh.write(f"{total} rows, {shard} shards, {QN} queries\n")
print(f"DOWNLOAD_DONE rows={total} shards={shard}", flush=True)
