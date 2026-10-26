# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Cache the seeded query set so exempt-class jobs need not derive it.

``queries()`` calls ``gen_block`` once per query shard, a 5M x dim generation of about
640 MB, and slicing the first rows off it keeps the whole block alive through the view, so
four query shards held four blocks at once and the job needed 8 GiB. The query rows are
the first rows of each shard's stream, and the stream draws every coefficient before any
noise, so the first band of ``gen_block_bands`` gives those rows exactly (the same early
stop ``fleet_rerank_gen.py`` verified byte for byte against ``gen_block``) at a peak near
0.1 GiB. The set is deterministic; this job computes it once, in the exempt class, and
every later job loads the cache.
"""

import os

import numpy as np
from fleet_common import QUERIES_PER_SHARD, QUERY_SHARDS, RESULTS, gen_block_bands


def head_rows(gshard: int, n: int) -> np.ndarray:
    out, got = [], 0
    bands = gen_block_bands(gshard)
    for band in bands:
        out.append(band[: n - got].copy())
        got += len(out[-1])
        if got >= n:
            break
    bands.close()  # removes the spill file
    return np.concatenate(out)


out = f"{RESULTS}/{os.environ.get('TQP_QCACHE_NAME', 'queries10b.npy')}"
if os.path.exists(out):
    print("query cache exists", flush=True)
else:
    q = np.concatenate([head_rows(g, QUERIES_PER_SHARD) for g in QUERY_SHARDS])
    np.save(out + ".tmp.npy", q)
    os.replace(out + ".tmp.npy", out)
    print(f"wrote {out} shape={q.shape}", flush=True)
print("QCACHE_DONE", flush=True)
