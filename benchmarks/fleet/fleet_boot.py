# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Fleet bootstrap (one job, runs first): the artifacts every node shares.

- global shard 0 on CephFS -> the **shared PCA basis** every ``write_shard``
  reuses (cross-server scores comparable) and the coordinator's pipeline
- the **global coarse quantizer** (``nlist=2048`` k-means on shard 0's
  quantized directions) + its radius file — copied verbatim by every server so
  the whole fleet probes identical cells
- the sparse-created cold-tier ``originals.npy`` the builders fill in parallel
"""

import os

import numpy as np
from fleet_common import (
    BITS,
    BOOT,
    NLIST,
    ORIG_DIR,
    OUT_DIM,
    RESULTS,
    gen_block,
)

from turboquant_pro import ShardedIndex

os.makedirs(RESULTS, exist_ok=True)

print("generating global shard 0", flush=True)
block = gen_block(0)
meta = ShardedIndex.write_shard(
    BOOT, block, 0, output_dim=OUT_DIM, bits=BITS, keep_originals=False
)
sh = ShardedIndex.finalize_manifest(BOOT, [meta])
print("fitting global coarse quantizer", flush=True)
sh.build_ivf(nlist=NLIST)
print("coarse:", np.load(f"{BOOT}/coarse_centroids.npy").shape, flush=True)

os.makedirs(ORIG_DIR, exist_ok=True)
print("cold store dir:", ORIG_DIR, flush=True)
print("BOOT_DONE", flush=True)
