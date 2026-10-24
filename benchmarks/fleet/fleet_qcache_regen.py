# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""Query set of the regenerated-index run (tag 1trg): two hundred queries from two hundred shards.

The first two query sets drew 25 rows from each of four shards, and every shard has its own
random subspace, so they cover four subspaces of the generator. This set draws one row from each
of two hundred shards:

  rows 0..99     non-members: row 0 of shards 400000..400099, seeds no corpus shard uses
                 (the corpus is shards 0..199999), so no query is in the index
  rows 100..199  corpus rows: row 0 of shard 400 * (5 i + 2) + 200, i = 0..99, one shard on
                 each of the servers 2, 7, ..., 497, so every query is in the index on a
                 different server

Row 0 of a shard is the first row of ``gen_block_bands``, taken as ``fleet_qcache.py`` takes it.
Writes ``queries1trg.npy`` and ``queries1trg.json`` (the shard and server of every row) to the
directory given. Deterministic.
"""

import json
import os
import sys

import numpy as np
from fleet_common import gen_block_bands

SHARDS_PER_SERVER = 400
NONMEMBER = [400_000 + i for i in range(100)]
MEMBER = [SHARDS_PER_SERVER * (5 * i + 2) + 200 for i in range(100)]


def row0(gshard):
    bands = gen_block_bands(gshard)
    first = next(bands)[:1].copy()
    bands.close()  # removes the spill file
    return first


out_dir = sys.argv[1]
q = np.concatenate([row0(g) for g in NONMEMBER + MEMBER])
meta = {
    "tag": "1trg",
    "rows": [
        {"row": i, "shard": g, "member": i >= 100}
        | ({"server": g // SHARDS_PER_SERVER} if i >= 100 else {})
        for i, g in enumerate(NONMEMBER + MEMBER)
    ],
}
np.save(os.path.join(out_dir, "queries1trg.npy"), q)
with open(os.path.join(out_dir, "queries1trg.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f)
print(f"wrote {q.shape}", flush=True)
