# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Regenerate this server's cold-tier originals as per-shard files.

The corpus is seed-defined, so the cold store can be (re)built without
touching the indexes: one ``g_XXXXX.npy`` per global shard, exactly one
writer per file. Idempotent — complete files are skipped.
"""

import os

import numpy as np
from fleet_common import (
    DIM,
    ORIG_DIR,
    SHARD_ROWS,
    SHARDS_PER_SERVER,
    gen_block,
    orig_path,
    write_original,
)

SID = int(os.environ["TQP_SERVER_ID"])
EXPECT = SHARD_ROWS * DIM * 4  # payload bytes; header adds a little

os.makedirs(ORIG_DIR, exist_ok=True)
for j in range(SHARDS_PER_SERVER):
    g = SID * SHARDS_PER_SERVER + j
    p = orig_path(g)
    if os.path.exists(p) and os.path.getsize(p) >= EXPECT:
        print(f"orig {j + 1}/{SHARDS_PER_SERVER} (g={g}) exists, skipped", flush=True)
        continue
    write_original(g, gen_block(g))
    print(f"orig {j + 1}/{SHARDS_PER_SERVER} (g={g}) written", flush=True)
print("ORIG_DONE", flush=True)
