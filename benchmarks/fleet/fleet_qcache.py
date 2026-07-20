# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Cache the seeded query set so exempt-class coordinators need not derive it.

``queries()`` calls ``gen_block`` once per query shard — each a 5M x dim
generation (~640 MB plus temporaries) — which OOMs a cpu<=1/2Gi serving-window
coordinator. The set is deterministic, so a right-sized batch job computes it
once and every coordinator loads the cache. (The rows cannot be shortcut:
truncating the generation changes the RNG stream for the noise term, so the
sample would no longer match what the reference jobs scanned.)
"""

import os

import numpy as np
from fleet_common import RESULTS, queries

out = f"{RESULTS}/{os.environ.get('TQP_QCACHE_NAME', 'queries10b.npy')}"
if os.path.exists(out):
    print("query cache exists", flush=True)
else:
    q = queries()
    np.save(out + ".tmp.npy", q)
    os.replace(out + ".tmp.npy", out)
    print(f"wrote {out} shape={q.shape}", flush=True)
print("QCACHE_DONE", flush=True)
