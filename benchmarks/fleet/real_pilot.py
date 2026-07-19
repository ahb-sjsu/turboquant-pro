# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Real-corpus pilot: does the synthetic triplet transfer to real embeddings?

Runs locally on Atlas over the downloaded Wikipedia Embed-V3 corpus
(1024-d fp32, real distribution, real held-out queries): build a v3 sharded
index (dim 1024 -> PCA 256 -> 4-bit), then measure the same triplet as the 1B
fleet run — recall of routed IVF vs the exact ADC full-scan, ADC-only recall
vs true fp32 ground truth, and tiered-rerank recall vs truth (originals ARE
the downloaded shards). Prints per-phase JSON + RESULT_JSON.

Thermal discipline: bounded threads (OMP), streaming build; the box's rule of
<=20 workers is respected by numpy's thread cap here.
"""

import glob
import json
import os
import time

import numpy as np

from turboquant_pro import ShardedIndex
from turboquant_pro.rerank_tier import rerank_candidates

SRC = os.environ.get("TQP_REAL_OUT", "/archive/tqp_real/wiki1024")
IDX = os.environ.get("TQP_REAL_IDX", "/archive/tqp_real/wiki1024_idx")
OUT_DIM = int(os.environ.get("TQP_REAL_OUT_DIM", "256"))
BITS = int(os.environ.get("TQP_REAL_BITS", "4"))
NLIST = int(os.environ.get("TQP_REAL_NLIST", "4096"))
NQ = int(os.environ.get("TQP_REAL_NQ", "500"))
K = 10

parts = sorted(glob.glob(f"{SRC}/part_*.npy"))
assert parts, f"no corpus shards under {SRC}"
q = np.load(f"{SRC}/queries.npy")[:NQ].astype(np.float32)
res: dict = {
    "corpus_shards": len(parts),
    "nq": len(q),
    "out_dim": OUT_DIM,
    "bits": BITS,
    "nlist": NLIST,
}


def blocks():
    for p in parts:
        yield np.load(p)


manifest = f"{IDX}/manifest.json"
if not os.path.exists(manifest):
    print("[1/4] building v3 sharded index", flush=True)
    t0 = time.time()
    ShardedIndex.create_streaming(
        blocks(), IDX, output_dim=OUT_DIM, bits=BITS, keep_originals=False
    )
    sh = ShardedIndex.open(manifest)
    sh.build_ivf(nlist=NLIST)
    res["build_s"] = round(time.time() - t0, 1)
else:
    print("[1/4] index exists, reusing", flush=True)
    sh = ShardedIndex.open(manifest)
n_rows = sh.n_rows
nbytes = sum(os.path.getsize(p) for p in glob.glob(f"{IDX}/*") if os.path.isfile(p))
res["n_rows"] = int(n_rows)
res["bytes_per_row"] = round(nbytes / n_rows, 2)
print(json.dumps({"rows": n_rows, "bytes_per_row": res["bytes_per_row"]}), flush=True)

print("[2/4] true fp32 ground truth (one pass over the corpus)", flush=True)
t0 = time.time()
qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-30)
best_sc = np.full((len(q), K), -np.inf, dtype=np.float32)
best_id = np.full((len(q), K), -1, dtype=np.int64)
off = 0
for p in parts:
    chunk = np.load(p)
    for s in range(0, len(chunk), 250_000):
        cb = chunk[s : s + 250_000]
        cn = cb / np.maximum(np.linalg.norm(cb, axis=1, keepdims=True), 1e-30)
        sc = (qn @ cn.T).astype(np.float32)
        part = np.argpartition(-sc, K - 1, axis=1)[:, :K]
        msc = np.concatenate([best_sc, np.take_along_axis(sc, part, axis=1)], axis=1)
        mid = np.concatenate([best_id, part + off + s], axis=1)
        order = np.argsort(-msc, axis=1)[:, :K]
        best_sc = np.take_along_axis(msc, order, axis=1)
        best_id = np.take_along_axis(mid, order, axis=1)
    off += len(chunk)
gt = best_id
res["gt_s"] = round(time.time() - t0, 1)
print(json.dumps({"gt_s": res["gt_s"]}), flush=True)


def recall(got, ref):
    return float(np.mean([len(set(a[:K]) & set(b[:K])) / K for a, b in zip(got, ref)]))


print("[3/4] exact ADC full-scan reference + routed sweep", flush=True)
t0 = time.time()
ref_ids, _ = sh.search(q, k=K)
res["fullscan"] = {
    "wall_s": round(time.time() - t0, 1),
    "true_recall": recall(ref_ids, gt),
}
print(json.dumps(res["fullscan"]), flush=True)
res["ivf"] = {}
for nprobe in (64, 256):
    t0 = time.time()
    ids, _ = sh.search(q, k=K, nprobe=nprobe, workers=8)
    res["ivf"][str(nprobe)] = {
        "wall_s": round(time.time() - t0, 1),
        "recall_vs_adc_fullscan": recall(ids, ref_ids),
        "true_recall": recall(ids, gt),
    }
    print(json.dumps({str(nprobe): res["ivf"][str(nprobe)]}), flush=True)

print("[4/4] tiered rerank from the original shards", flush=True)


class LocalStore:
    """fetch(ids) straight from the downloaded fp32 shards (memmap is fine
    on local storage — the CephFS pathologies do not apply)."""

    def __init__(self):
        self._maps = [np.load(p, mmap_mode="r") for p in parts]
        self._sizes = np.array([len(m) for m in self._maps])
        self._starts = np.concatenate([[0], np.cumsum(self._sizes)[:-1]])

    def fetch(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        out = np.empty((len(ids), q.shape[1]), dtype=np.float32)
        which = np.searchsorted(self._starts, ids, side="right") - 1
        for w in np.unique(which):
            m = which == w
            out[m] = self._maps[w][ids[m] - self._starts[w]]
        return out


t0 = time.time()
wide, _ = sh.search(q, k=K * 10, nprobe=256, workers=8)
rr_ids, _ = rerank_candidates(q, wide, K, LocalStore())
res["rerank"] = {
    "wall_s": round(time.time() - t0, 1),
    "true_recall_adc_only": res["ivf"]["256"]["true_recall"],
    "true_recall_reranked": recall(rr_ids, gt),
}
print(json.dumps(res["rerank"]), flush=True)

with open(f"{SRC}/real_pilot_result.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print("RESULT_JSON " + json.dumps(res), flush=True)
print("PILOT_DONE", flush=True)
