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

The build code changed while the index was built, and two of its changes alter the shard. The
first servers (0 to 7) were begun on 2026-08-04 while each shard was still written as one block
(``write_shard`` over ``gen_block``); the streaming write (``write_shard_streaming`` over
``gen_block_bands``, 8f42543) replaced it that night, so those servers hold shards of both. The
two writes round the rotated rows differently. Separately, f724489 decodes each shard in chunks for its cell assignment, which
moves boundary rows on some shards. The write is the ``write`` argument here, and the library
version is whichever ``turboquant_pro`` the caller puts on the path (``fleet_regen_score.py``
supplies the ``sharded_index.py`` of 0d5178b, before the chunking, and of f99d0d5, after it; no
other library file the build reads changed between 74ae141 and the build's end). Shards 2000 and 1
reproduce only from the block write, and shard 2000 only before the chunking (Atlas, 2026-10-25).

The kernel can only be chosen before numpy loads (``OPENBLAS_CORETYPE``), so
``fleet_regen_score.py`` runs this script once per candidate and keeps the build whose cell files
hash equal to the released shard's.

Usage: python fleet_regen_build.py <global shard> <shard index on server> <out dir> <stream|block>
"""

import os
import shutil
import sys

import numpy as np
from fleet_common import BOOT, SHARD_ROWS, drop_page_cache, gen_block, gen_block_bands

from turboquant_pro import ShardedIndex

g, j, d, write = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
BOOT = os.environ.get("TQP_BOOT", BOOT)
BASIS = f"{BOOT}/shard_00000.tqe"
shutil.rmtree(d, ignore_errors=True)
os.makedirs(d)
if write == "block":
    m = ShardedIndex.write_shard(
        d,
        gen_block(g),
        j,
        ids=np.arange(g * SHARD_ROWS, (g + 1) * SHARD_ROWS, dtype=np.int64),
        basis_from=BASIS,
        keep_originals=False,
    )
elif write == "stream":
    m = ShardedIndex.write_shard_streaming(
        d, gen_block_bands(g), j, ids_start=g * SHARD_ROWS, basis_from=BASIS
    )
else:
    raise SystemExit(f"unknown write {write!r}")
drop_page_cache(os.path.join(d, m["path"]))
sh = ShardedIndex.finalize_manifest(d, [m])
# As fleet_build.py assigned: resumable, against the global coarse quantizer.
sh.build_ivf(centroids=np.load(f"{BOOT}/coarse_centroids.npy"), resume=True)
shutil.copy(f"{BOOT}/coarse_radius.npy", os.path.join(d, "coarse_radius.npy"))
print("BUILD_DONE", m["path"], flush=True)
