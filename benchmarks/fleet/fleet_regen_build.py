# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""Rebuild one global shard into a directory, under whatever OpenBLAS kernel this process runs.

The corpus rows are bit-identical under every kernel (``gen_block_bands`` measured equal under
the Haswell, Sandybridge and Nehalem kernels on 2026-10-24). The shard built from them is not:
the rotation into the shared basis, the 4-bit quantization and the cell assignment are BLAS
products, and OpenBLAS picks its kernel from the CPU at load time. On one shard the Haswell and
Nehalem kernels put 9 of 5,000,000 rows in different cells and differ in 48 of 120,000,000 code
bytes. The released index was built on whatever node each build job landed on, so each of its
shards carries the arithmetic of one kernel.

The kernel can only be chosen before numpy loads (``OPENBLAS_CORETYPE``), so
``fleet_regen_score.py`` runs this script once per candidate kernel and keeps the build whose
cell files hash equal to the released shard's.

Usage: python fleet_regen_build.py <global shard> <shard index on server> <out dir>
"""

import os
import shutil
import sys

import numpy as np
from fleet_common import BOOT, SHARD_ROWS, drop_page_cache, gen_block_bands

from turboquant_pro import ShardedIndex

g, j, d = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
BOOT = os.environ.get("TQP_BOOT", BOOT)
shutil.rmtree(d, ignore_errors=True)
os.makedirs(d)
m = ShardedIndex.write_shard_streaming(
    d,
    gen_block_bands(g),
    j,
    ids_start=g * SHARD_ROWS,
    basis_from=f"{BOOT}/shard_00000.tqe",
)
drop_page_cache(os.path.join(d, m["path"]))
sh = ShardedIndex.finalize_manifest(d, [m])
sh.build_ivf(centroids=np.load(f"{BOOT}/coarse_centroids.npy"))
shutil.copy(f"{BOOT}/coarse_radius.npy", os.path.join(d, "coarse_radius.npy"))
print("BUILD_DONE", m["path"], flush=True)
