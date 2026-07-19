# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Stream a real web-scale embedding corpus onto Atlas /archive.

CohereLabs/wikipedia-2023-11-embed-multilingual-v3 (en): ~41M Wikipedia
paragraph embeddings, 1024-d Embed-V3 fp32 — a public, real-distribution
corpus for the pilot that answers "does the synthetic-corpus structure
transfer to real embeddings?". (The older 22-12 sets are script-style
datasets that `datasets` 4.x can no longer load.) Writes 1M-row fp32 .npy
shards (corpus) + a held-out query set drawn AFTER the corpus cap (real
queries, not corpus members). Idempotent per shard.
"""

import os
import time

import numpy as np

OUT = os.environ.get("TQP_REAL_OUT", "/archive/tqp_real/wiki1024")
CAP_ROWS = int(os.environ.get("TQP_REAL_ROWS", str(20_000_000)))
SHARD = 1_000_000
QN = 1000
DATASET = "CohereLabs/wikipedia-2023-11-embed-multilingual-v3"
CONFIG = os.environ.get("TQP_REAL_CONFIG", "en")

os.makedirs(OUT, exist_ok=True)
done_marker = f"{OUT}/DOWNLOAD_DONE"
if os.path.exists(done_marker):
    print("already complete", flush=True)
    raise SystemExit(0)

from datasets import load_dataset  # noqa: E402

ds = load_dataset(DATASET, CONFIG, split="train", streaming=True)
it = ds.iter(batch_size=50_000)

first = next(it)
dim = int(np.asarray(first["emb"][0]).shape[-1])
print(f"dataset {DATASET}/{CONFIG}, dim={dim}", flush=True)
with open(f"{OUT}/META.txt", "w") as f:
    f.write(f"{DATASET} {CONFIG} dim={dim}\n")

buf = np.empty((SHARD, dim), dtype=np.float32)
fill = 0
shard = 0
total = 0
queries: list[np.ndarray] = []
qdone = 0
t0 = time.time()
from itertools import chain  # noqa: E402

for batch in chain([first], it):
    emb = np.asarray(batch["emb"], dtype=np.float32)
    i = 0
    while i < len(emb):
        if total < CAP_ROWS:
            take = min(SHARD - fill, len(emb) - i, CAP_ROWS - total)
            buf[fill : fill + take] = emb[i : i + take]
            fill += take
            total += take
            i += take
            if fill == SHARD:
                p = f"{OUT}/part_{shard:03d}.npy"
                if not os.path.exists(p):
                    np.save(p + ".tmp.npy", buf)
                    os.replace(p + ".tmp.npy", p)
                rate = total / max(time.time() - t0, 1)
                print(
                    f"shard {shard} done ({total} rows, {rate:.0f} rows/s)", flush=True
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
    if total >= CAP_ROWS and qdone >= QN:
        break

np.save(f"{OUT}/queries.npy", np.concatenate(queries)[:QN])
with open(done_marker, "w") as f:
    f.write(f"{total} rows, {shard} shards, {QN} queries\n")
print(f"DOWNLOAD_DONE rows={total} shards={shard}", flush=True)
