# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Fleet coordinator: the end-to-end measurement over the live shard-servers.

Runs in-cluster (reaches the pool at ``nats://atlas-nats:4222``), mounts only
the shared CephFS volume. Acceptance is rank fidelity throughout — recall of
shortlists against reference rankings, plus **true** fp32 recall against a
one-pass ground truth computed from the cold store. Never reconstruction
cosine.

Phases (ordered so the shard-servers are needed only for a short, busy
window; ``TQP_GT_ONLY=1`` runs phase 1 alone, with no fleet up at all):

1. true fp32 top-10 per query — one blocked pass over the cold store (cached)
2. fleet full-scan ADC reference (exact scatter-gather merge at 1B)
3. routed IVF sweep (nprobe 32 / 128): recall vs both references, wall time,
   servers touched (with round-robin range placement every server holds every
   cell, so the router's fan-out equals the fleet — reported, not hidden)
4. tiered rerank: wide routed shortlist re-scored from the cold store —
   the ADC-vs-truth ceiling break, measured at 1B

``TQP_LOGICAL_PER_POD=L`` targets the L-per-pod logical fleet (1T/5T
endpoint-count shape) written by ``fleet_split.py``.
"""

import json
import os
import sys
import time

import numpy as np
from fleet_common import (
    BOOT,
    N_SERVERS,
    RESULTS,
    SHARD_ROWS,
    SHARDS_PER_SERVER,
    ShardedNpyStore,
    orig_path,
    queries,
)

from turboquant_pro.distributed import (
    Router,
    build_cell_placement,
    scatter_gather,
    scatter_gather_routed,
)
from turboquant_pro.nats_transport import nats_transport
from turboquant_pro.rerank_tier import rerank_candidates

K = 10
NATS_URL = os.environ.get("NATS_URL", "nats://atlas-nats:4222")
# Logical fleet topology: L>1 multiplexes L shard-servers per PVC-holding pod
# (the 1T/5T *shape* — endpoint count — without 1T of storage).
L = int(os.environ.get("TQP_LOGICAL_PER_POD", "1"))
N_LOGICAL = N_SERVERS * L
# Cap in-flight requests: at large L the endpoints share 4 pods, and running
# every logical server's scan at once OOMs the pods (16 concurrent
# full-scan buffer sets per 2-CPU pod). The merge is exact regardless of
# request batching, so a bounded fan-out changes nothing but memory.
MAXPAR = min(N_LOGICAL, int(os.environ.get("TQP_MAX_PARALLEL", "12")))


def recall(got: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean([len(set(a[:K]) & set(b[:K])) / K for a, b in zip(got, ref)]))


q = queries()
res: dict = {"nq": len(q), "n_servers": N_SERVERS, "n_logical": N_LOGICAL}

print("[1/4] true fp32 ground truth from the cold store", flush=True)
t0 = time.time()
gt_cache = f"{RESULTS}/gt.npy"
gt_parts = [f"{RESULTS}/gt_part_{i}.npz" for i in range(N_SERVERS)]
if os.path.exists(gt_cache):
    gt = np.load(gt_cache)
    print("ground truth loaded from cache", flush=True)
elif all(os.path.exists(p) for p in gt_parts):
    parts = [np.load(p) for p in gt_parts]
    ids = np.concatenate([p["ids"] for p in parts], axis=1)
    scs = np.concatenate([p["scores"] for p in parts], axis=1)
    order = np.argsort(-scs, axis=1)[:, :K]
    gt = np.take_along_axis(ids, order, axis=1)
    np.save(gt_cache, gt)
    print("ground truth merged from partials", flush=True)
else:
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-30)
    best_sc = np.full((len(q), K), -np.inf, dtype=np.float32)
    best_id = np.full((len(q), K), -1, dtype=np.int64)
    SLAB = 1_000_000  # keeps the (nq, slab) score temporaries ~2 GB, not ~9 GB
    for g in range(N_SERVERS * SHARDS_PER_SERVER):
        # Sequential read, not mmap — CephFS mmap faults are ~10x slower.
        arr = np.load(orig_path(g))
        for s in range(0, len(arr), SLAB):
            chunk = np.asarray(arr[s : s + SLAB])
            cn = chunk / np.maximum(np.linalg.norm(chunk, axis=1, keepdims=True), 1e-30)
            sc = (qn @ cn.T).astype(np.float32)  # (nq, slab)
            part = np.argpartition(-sc, K - 1, axis=1)[:, :K]
            msc = np.concatenate(
                [best_sc, np.take_along_axis(sc, part, axis=1)], axis=1
            )
            mid = np.concatenate([best_id, part + g * SHARD_ROWS + s], axis=1)
            order = np.argsort(-msc, axis=1)[:, :K]
            best_sc = np.take_along_axis(msc, order, axis=1)
            best_id = np.take_along_axis(mid, order, axis=1)
        if (g + 1) % 20 == 0:
            print(f"  gt {g + 1}/{N_SERVERS * SHARDS_PER_SERVER}", flush=True)
    gt = best_id
    np.save(gt_cache, gt)
res["ground_truth"] = {"wall_s": round(time.time() - t0, 1)}
print(json.dumps(res["ground_truth"]), flush=True)
if os.environ.get("TQP_GT_ONLY"):
    print("COORD_DONE", flush=True)
    sys.exit(0)

shared = os.path.dirname(BOOT)
eps = [str(i) for i in range(N_LOGICAL)]
if L == 1:
    mans = [f"{shared}/server_{i:03d}/manifest.json" for i in range(N_SERVERS)]
else:
    mans = [
        f"{shared}/server_{t // L:03d}/server_L{L}_{t:03d}.manifest.json"
        for t in range(N_LOGICAL)
    ]
placement = build_cell_placement(mans, eps)
router = Router(BOOT, placement, pipeline_manifest=f"{BOOT}/manifest.json")
transport = nats_transport(NATS_URL, timeout=2400.0)

# Readiness barrier: at large L the workers take minutes to spawn; requesting
# an unsubscribed subject just times out. Probe every endpoint (1-query,
# nprobe=1) on a short-timeout transport until the whole fleet answers.
probe_t = nats_transport(NATS_URL, timeout=15.0)
pending = set(eps)
deadline = time.time() + 900
while pending and time.time() < deadline:
    for ep in sorted(pending, key=int):
        try:
            scatter_gather(q[:1], 1, [ep], probe_t, nprobe=1)
            pending.discard(ep)
        except Exception:
            pass
    if pending:
        print(f"  waiting for {len(pending)} endpoints", flush=True)
        time.sleep(10)
probe_t.close()
if pending:
    raise RuntimeError(f"fleet endpoints never came up: {sorted(pending, key=int)}")
print(f"fleet ready: {len(eps)} endpoints", flush=True)

print("[2/4] fleet full-scan ADC reference", flush=True)
t0 = time.time()
ref_ids, _ = scatter_gather(q, K, eps, transport, max_parallel=MAXPAR, workers=1)
res["fullscan"] = {
    "wall_s": round(time.time() - t0, 1),
    "true_recall": recall(ref_ids, gt),
}
print(json.dumps(res["fullscan"]), flush=True)

print("[3/4] routed IVF sweep", flush=True)
res["routed"] = {}
for nprobe in (32, 128):
    touched, _ = router.servers_for(q, nprobe)
    t0 = time.time()
    ids, _ = scatter_gather_routed(
        q, K, router, transport, nprobe=nprobe, workers=1, max_parallel=MAXPAR
    )
    res["routed"][str(nprobe)] = {
        "wall_s": round(time.time() - t0, 1),
        "qps": round(len(q) / (time.time() - t0), 2),
        "recall_vs_adc_fullscan": recall(ids, ref_ids),
        "true_recall": recall(ids, gt),
        "servers_touched": len(touched),
    }
    print(json.dumps({str(nprobe): res["routed"][str(nprobe)]}), flush=True)

print("[4/4] tiered rerank from the cold store", flush=True)
t0 = time.time()
wide, _ = scatter_gather_routed(
    q, K * 10, router, transport, nprobe=128, workers=1, max_parallel=MAXPAR
)
t_short = time.time() - t0
store = ShardedNpyStore()
t0 = time.time()
rr_ids, _ = rerank_candidates(q, wide, K, store)
res["rerank"] = {
    "shortlist_wall_s": round(t_short, 1),
    "rerank_wall_s": round(time.time() - t0, 1),
    "true_recall_adc_only": res["routed"]["128"]["true_recall"],
    "true_recall_reranked": recall(rr_ids, gt),
}
print(json.dumps(res["rerank"]), flush=True)

transport.close()
out_name = "fleet_run.json" if L == 1 else f"fleet_run_L{L}.json"
with open(f"{RESULTS}/{out_name}", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print("RESULT_JSON " + json.dumps(res), flush=True)
print("COORD_DONE", flush=True)
