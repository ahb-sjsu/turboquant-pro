# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""10B measurement coordinator: routed sweep vs the scattered exact reference.

No cold store at 10B (rerank/true-recall are established at 1B), so the
serving window carries only the cheap phases: merge the 8 pre-computed
full-scan reference partials (exact — shared basis), readiness-barrier the
fleet, then the routed IVF sweep on a 100-query subset of the 500-query
reference set. Serving pods are exempt-class; the window is minutes.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fleet_common import BOOT, RESULTS, queries


from turboquant_pro.distributed import (
    Router,
    build_cell_placement,
    scatter_gather,
    scatter_gather_routed,
)
from turboquant_pro.nats_transport import nats_transport

K = 10
# Per-request shard fan-out INSIDE each server. At 250 shards/server the scan
# path re-opens every shard per request (thread-safety), so a request is
# dominated by ~250 serialized cold opens — measured at ~2% CPU, i.e. pure I/O
# latency. Threads overlap those opens even on a 1-CPU pod, which is the whole
# point of the knob.
WORKERS = int(os.environ.get("TQP_WORKERS", "16"))
N_SRV = int(os.environ.get("TQP_N_SERVERS", "8"))
NQ_ROUTED = int(os.environ.get("TQP_NQ_ROUTED", "100"))
PREFIX = os.environ.get("TQP_EXPORT_PREFIX", "server10b_")
NATS_URL = os.environ.get("NATS_URL", "nats://atlas-nats:4222")


def recall(got, ref):
    return float(np.mean([len(set(a[:K]) & set(b[:K])) / K for a, b in zip(got, ref)]))


def _failed(fut) -> bool:
    try:
        fut.result()
        return False
    except Exception as e:  # a probe that never answered
        print(f"  probe failed: {type(e).__name__}: {e}", flush=True)
        return True


# Query set from the cache (fleet_qcache.py): deriving it here would OOM the
# exempt-class pod — gen_block is ~640 MB per query shard.
qcache = f"{RESULTS}/{os.environ.get('TQP_QCACHE_NAME', 'queries10b.npy')}"
q = np.load(qcache) if os.path.exists(qcache) else queries()
res: dict = {"nq_reference": len(q), "nq_routed": NQ_ROUTED, "n_servers": N_SRV}

print("[1/3] merging scattered full-scan reference", flush=True)
parts = [np.load(f"{RESULTS}/ref10b_part_{i}.npz") for i in range(N_SRV)]
ids = np.concatenate([p["ids"] for p in parts], axis=1)
scs = np.concatenate([p["scores"] for p in parts], axis=1)
scs = np.where(np.isfinite(scs), scs, -np.inf)
order = np.argsort(-scs, axis=1)[:, :K]
ref_ids = np.take_along_axis(ids, order, axis=1)
res["reference"] = {
    "per_server_wall_s": [float(p["wall_s"]) for p in parts],
}
print(json.dumps(res["reference"]), flush=True)

qr = q[:NQ_ROUTED]
ref_r = ref_ids[:NQ_ROUTED]
shared = os.path.dirname(BOOT)
eps = [str(i) for i in range(N_SRV)]
mans = [f"{shared}/{PREFIX}{i:03d}/manifest.json" for i in range(N_SRV)]
placement = build_cell_placement(mans, eps)
router = Router(BOOT, placement, pipeline_manifest=f"{BOOT}/manifest.json")
transport = nats_transport(NATS_URL, timeout=7200.0)

print("[2/3] readiness barrier", flush=True)
# Probe budget scales with shards/server: even an nprobe=1 request walks every
# shard in the range (250 at 10B vs 50 at 1B), so a 15 s probe times out on a
# perfectly healthy fleet.
# Probe every endpoint CONCURRENTLY. A request costs the same whether it
# carries 1 query or 100 (the coordinator batches all queries into one request
# per server), and at 250 shards/server that fixed cost is minutes — so a
# serial barrier over 8 endpoints cannot finish inside any sane deadline,
# while a concurrent one costs a single request's time.
probe_s = float(os.environ.get("TQP_PROBE_S", "1800"))
probe_t = nats_transport(NATS_URL, timeout=probe_s)
t0 = time.time()
with ThreadPoolExecutor(max_workers=len(eps)) as ex:
    futs = {
        ex.submit(
            scatter_gather, qr[:1], 1, [ep], probe_t, nprobe=1, workers=WORKERS
        ): ep
        for ep in eps
    }
    down = [ep for f, ep in futs.items() if _failed(f)]
probe_t.close()
if down:
    raise RuntimeError(f"fleet endpoints never came up: {sorted(down, key=int)}")
print(f"fleet ready: {len(eps)} endpoints in {time.time() - t0:.0f}s", flush=True)

print("[3/3] routed IVF sweep", flush=True)
res["routed"] = {}
for nprobe in (32, 128):
    touched, _ = router.servers_for(qr, nprobe)
    t0 = time.time()
    got, _ = scatter_gather_routed(
        qr, K, router, transport, nprobe=nprobe, workers=WORKERS, max_parallel=8
    )
    res["routed"][str(nprobe)] = {
        "wall_s": round(time.time() - t0, 1),
        "qps": round(len(qr) / (time.time() - t0), 2),
        "recall_vs_adc_fullscan": recall(got, ref_r),
        "servers_touched": len(touched),
    }
    print(json.dumps({str(nprobe): res["routed"][str(nprobe)]}), flush=True)

transport.close()
with open(f"{RESULTS}/fleet_run_10B.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print("RESULT_JSON " + json.dumps(res), flush=True)
print("COORD_DONE", flush=True)
