# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""The rerank bound at 10^12 rows, from regenerated floats and no new scan.

Inputs are the wide shortlists (``fleet_rerank_prep.py``) and the regenerated float rows
(``fleet_rerank_gen.py``). Every shortlist is rescored by cosine against the float query,
the metric ``fleet_gt.py`` used for ground truth at 1B, and cut to ten. Three numbers a set:

  adc_float_agreement   share of the ADC top-10 that survives the float rerank of its own
                        shortlist. The float top-10 of the whole corpus can only lose
                        members relative to the float top-10 of a subset that contains the
                        ADC top-10, so this is an UPPER BOUND on ADC-only true recall.
  rerank_transfer       share of the routed shortlist's float top-10 that is also in the
                        reference shortlist's float top-10: what routing at that width
                        costs after reranking, relative to the exact compressed scan.
  cos_adc10 / cos_float10   mean cosine of the two top-10s to the query, the score gap
                        the rerank closes inside the shortlist.

Neither is true recall. True recall needs the float top-10 over all 10^12 rows, which no
scan here produced; the 1B run measured it from a cold store and the JSON keeps the
distinction. Writes ``rerank{TAG}_bound.json`` and prints RESULT_JSON.
"""

import glob
import json
import os

import numpy as np
from fleet_common import QUERIES_PER_SHARD, QUERY_SHARDS, RESULTS, SHARD_ROWS

K = 10
TAG = os.environ.get("TQP_RUN_TAG", "1t")
N_GEN = int(os.environ.get("TQP_RGEN_N", "100"))
QNAME = os.environ.get("TQP_QCACHE_NAME", f"queries{TAG}.npy")


def normalize(x):
    x = np.asarray(x, np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)


def recall(got, ref):
    return np.array([len(set(a[:K]) & set(b[:K])) / K for a, b in zip(got, ref)])


def dist(v):
    v = np.asarray(v, float)
    return {
        "mean": round(float(v.mean()), 4),
        "min": round(float(v.min()), 3),
        "p10": round(float(np.percentile(v, 10)), 3),
        "queries_at_1": int((v >= 1.0 - 1e-12).sum()),
        "n": int(len(v)),
    }


sl = np.load(f"{RESULTS}/rerank{TAG}_shortlists.npz")
sets = sorted(k[4:] for k in sl.files if k.startswith("ids_"))
sets = ["ref"] + sorted((s for s in sets if s != "ref"), key=int)

paths = sorted(glob.glob(f"{RESULTS}/rerank{TAG}_vec_*.npz"))
assert len(paths) == N_GEN, (len(paths), N_GEN)
parts = [np.load(p) for p in paths]
vid = np.concatenate([p["ids"] for p in parts]).astype(np.int64)
vec = np.concatenate([p["vecs"] for p in parts]).astype(np.float32)
order = np.argsort(vid)
vid, vec = vid[order], vec[order]
need = np.load(f"{RESULTS}/rerank{TAG}_rows.npy").astype(np.int64)
assert (
    len(vid) == len(need) and (vid == need).all()
), "regenerated rows differ from the list"
vn = normalize(vec)


def fetch(ids):
    pos = np.searchsorted(vid, ids)
    assert (vid[pos] == ids).all()
    return vn[pos]


q = np.load(f"{RESULTS}/{QNAME}").astype(np.float32)
qn = normalize(q)
nq = len(q)
self_ids = np.concatenate(
    [g * SHARD_ROWS + np.arange(QUERIES_PER_SHARD) for g in QUERY_SHARDS]
)
assert len(self_ids) == nq, (len(self_ids), nq)

res = {
    "n_rows": 500 * 2_000_000_000,
    "nq": int(nq),
    "k": K,
    "wide": int(sl["ids_ref"].shape[1]),
    "unique_rows_regenerated": int(len(vid)),
    "metric": "cosine on regenerated float rows, as fleet_gt.py at 1B",
    "shortlist": "top-WIDE by ADC score of the merge of 500 per-server top-10 partials, "
    "not the exact ADC top-WIDE (a server may hold more than ten of those)",
    "not_true_recall": "the float top-10 over all rows is unknown here; "
    "adc_float_agreement is an upper bound on ADC-only true recall",
    "sets": {},
}
float10 = {}
for s in sets:
    ids = sl[f"ids_{s}"]
    cos = np.einsum("qd,qwd->qw", qn, fetch(ids.ravel()).reshape(nq, -1, q.shape[1]))
    o = np.argsort(-cos, axis=1, kind="stable")[:, :K]
    f10 = np.take_along_axis(ids, o, axis=1)
    a10 = ids[:, :K]
    float10[s] = f10
    cos_a = cos[:, :K].mean(axis=1)
    cos_f = np.take_along_axis(cos, o, axis=1).mean(axis=1)
    res["sets"][s] = {
        "adc_float_agreement": dist(recall(a10, f10)),
        "cos_adc10": round(float(cos_a.mean()), 5),
        "cos_float10": round(float(cos_f.mean()), 5),
        "self_in_adc10": int((a10 == self_ids[:, None]).any(axis=1).sum()),
        "self_in_float10": int((f10 == self_ids[:, None]).any(axis=1).sum()),
    }
    if s != "ref":
        res["sets"][s]["rerank_transfer"] = dist(recall(f10, float10["ref"]))
        res["sets"][s]["adc_recall_vs_ref_adc10"] = round(
            float(recall(a10, sl["ids_ref"][:, :K]).mean()), 4
        )
    print(json.dumps({s: res["sets"][s]}), flush=True)

tmp = f"{RESULTS}/rerank{TAG}_bound.json.tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
os.replace(tmp, f"{RESULTS}/rerank{TAG}_bound.json")
print("RESULT_JSON " + json.dumps(res), flush=True)
print("RERANK_DONE", flush=True)
