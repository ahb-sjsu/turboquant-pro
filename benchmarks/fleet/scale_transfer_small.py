# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""Scale transfer from 10^8 rows: routed recall at 10^12 predicted from corpora of 10^8 to 10^9.

The preregistration is docs/PREREG_scale_transfer_small.md; this script is the procedure it names.

A calibration corpus is a set of shards of the trillion-row index, rebuilt from their seeds with
the fleet's own bootstrap: the same basis (shard_00000.tqe), the same global coarse quantizer
(coarse_centroids.npy) and the same radius file (coarse_radius.npy), which the fleet copies over
every server's local radius so that every server probes the same cells. A corpus of m shards is
therefore the fleet's index restricted to those shards, routed by the fleet's router, and it is
searched as one ShardedIndex, as each fleet server searched its own 400 shards.

Calibration design: three permutations of 200 shards drawn without replacement from the 200000
global shards (seed PERM_SEED); for each, the first 20, 40, 100 and 200 shards, that is 10^8,
2*10^8, 5*10^8 and 10^9 rows. Recall at ten of routing against the exact scan of the same corpus,
per query, averaged over the permutations, is the calibration curve per width.

Phases (run in order on a machine with the bootstrap and the fleet code on the path):
  build     rebuild every drawn shard from its seed with write_shard_streaming, one thread, and
            assign its cells against the global coarse quantizer; idempotent per shard
  measure   for one query set and one (permutation, size): exact and routed top ten per query;
            idempotent per output file
  curve     per-query recall for every (permutation, size) of a query set, and the calibration
            curve; for --dev also the fit and every model's prediction beside the measured 1e12
            value of the member-query run
  grade     after the prediction for a query set is committed: per-query recall at 1e12 from
            the fleet's 500 per-server partials of the registered run (read with the fleet's
            scale_transfer.py), and every model graded by the paired bootstrap of
            docs/PREREG_scale_transfer.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

SHARD_ROWS = 5_000_000
N_GLOBAL_SHARDS = 200_000
N_STAR = 10**12
PERM_SEED = 20261001
N_PERM = 3
SIZES = (20, 40, 100, 200)
K = 10
WIDTHS = (16, 32, 64, 128, 256)
BLOCK = 65536
MODELS = ("M0", "M1", "M2")
BOOT = 2000
BOOT_SEED = 7
# member-query run, all 500 servers, from record/1t/post (score_1T.log, probe_1T.log)
MEMBER_1E12 = {16: 0.952, 32: 0.989, 64: 0.997, 128: 0.999, 256: 1.000}


def permutations() -> list[list[int]]:
    rng = np.random.default_rng(PERM_SEED)
    return [draw(rng) for _ in range(N_PERM)]


def draw(rng) -> list[int]:
    return [int(g) for g in rng.choice(N_GLOBAL_SHARDS, max(SIZES), replace=False)]


def all_shards() -> list[int]:
    return sorted({g for p in permutations() for g in p})


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# build                                                                       #
# --------------------------------------------------------------------------- #


def build_one(g: int, pool: str, boot: str) -> None:
    from fleet_common import gen_block_bands

    from turboquant_pro import ShardedIndex

    d = os.path.join(pool, f"g{g:06d}")
    done = os.path.join(d, "DONE")
    if os.path.exists(done):
        return
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    m = ShardedIndex.write_shard_streaming(
        d,
        gen_block_bands(g),
        0,
        ids_start=g * SHARD_ROWS,
        basis_from=os.path.join(boot, "shard_00000.tqe"),
    )
    sh = ShardedIndex.finalize_manifest(d, [m])
    sh.build_ivf(centroids=np.load(os.path.join(boot, "coarse_centroids.npy")))
    open(done, "w").write(json.dumps({"g": g, "meta": m}))


def assemble(perm_idx: int, size: int, pool: str, work: str, boot: str) -> str:
    """A ShardedIndex over the first `size` shards of a permutation, by symlinks into the pool,
    with the bootstrap's centroids and radius, returned as its manifest path."""
    from turboquant_pro import ShardedIndex

    d = os.path.join(work, f"p{perm_idx}_m{size}")
    man = os.path.join(d, "manifest.json")
    if os.path.exists(os.path.join(d, "READY")):
        return man
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    metas = []
    for j, g in enumerate(permutations()[perm_idx][:size]):
        # absolute: a relative link target resolves against the link's own directory
        src = os.path.abspath(os.path.join(pool, f"g{g:06d}"))
        for suffix in (".tqe", ".ivf.off.npy", ".ivf.memb.npy"):
            os.symlink(
                os.path.join(src, "shard_00000" + suffix),
                os.path.join(d, f"shard_{j:05d}" + suffix),
            )
        meta = json.load(open(os.path.join(src, "DONE")))["meta"]
        metas.append(dict(meta, path=f"shard_{j:05d}.tqe"))
    sh = ShardedIndex.finalize_manifest(d, metas)
    # every sidecar exists, so this records the ivf block without re-assigning any row
    sh.build_ivf(
        centroids=np.load(os.path.join(boot, "coarse_centroids.npy")), resume=True
    )
    shutil.copy(
        os.path.join(boot, "coarse_radius.npy"), os.path.join(d, "coarse_radius.npy")
    )
    open(os.path.join(d, "READY"), "w").close()
    return man


# --------------------------------------------------------------------------- #
# measure                                                                     #
# --------------------------------------------------------------------------- #


def measure(
    qname: str,
    queries: np.ndarray,
    perm_idx: int,
    size: int,
    pool: str,
    work: str,
    boot: str,
    out_dir: str,
) -> str:
    from turboquant_pro import ShardedIndex

    out = os.path.join(out_dir, f"{qname}_p{perm_idx}_m{size}.npz")
    if os.path.exists(out):
        return out
    man = assemble(perm_idx, size, pool, work, boot)
    sh = ShardedIndex.open(man, mmap=True, max_open_shards=8)
    res = {}
    t = time.time()
    ids, sc = sh.search(queries, k=K, block=BLOCK)
    res["ref_ids"], res["ref_scores"] = ids, sc
    res["ref_wall_s"] = time.time() - t
    for p in WIDTHS:
        t = time.time()
        ids, sc = sh.search(queries, k=K, nprobe=p, workers=1)
        res[f"p{p}_ids"], res[f"p{p}_scores"] = ids, sc
        res[f"p{p}_wall_s"] = time.time() - t
    tmp = out + ".tmp.npz"
    np.savez(tmp, **res)
    os.replace(tmp, out)
    return out


def recall(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.array(
        [len(set(a[a >= 0]) & set(b[b >= 0])) / K for a, b in zip(got, ref)]
    )


# --------------------------------------------------------------------------- #
# curve and models                                                            #
# --------------------------------------------------------------------------- #


def fit_predict(n: np.ndarray, r: np.ndarray, n_star: float = N_STAR) -> dict:
    x, xs = np.log(n), np.log(n_star)
    out = {"M0": float(r[-1])}
    b, a = np.polyfit(x, np.log(np.clip(1.0 - r, 1e-6, None)), 1)
    out["M1"] = float(1.0 - np.exp(a + b * xs))
    b2, a2 = np.polyfit(x, r, 1)
    out["M2"] = float(min(1.0, a2 + b2 * xs))
    return out


def curve(qname: str, out_dir: str) -> dict:
    n = np.array(SIZES, float) * SHARD_ROWS
    per_query = {}
    for p in WIDTHS:
        rows = []
        for m in SIZES:
            acc = 0.0
            for i in range(N_PERM):
                z = np.load(os.path.join(out_dir, f"{qname}_p{i}_m{m}.npz"))
                acc = acc + recall(z[f"p{p}_ids"], z["ref_ids"])
            rows.append(acc / N_PERM)
        per_query[p] = np.array(rows)  # (sizes, nq)
    rec = {"query_set": qname, "sizes_rows": [int(v) for v in n], "widths": {}}
    for p, m in per_query.items():
        c = m.mean(axis=1)
        rec["widths"][str(p)] = {
            "curve": dict(zip(map(str, map(int, n)), map(float, c))),
            "prediction_1e12": fit_predict(n, c),
        }
    return rec, per_query


def grade(qname: str, out_dir: str, pred_path: str, results: str, tag: str) -> dict:
    """Grade a committed prediction against the registered run's value at 1e12.

    The calibration queries and the run's queries are the same rows in the same order, so
    each bootstrap resample recomputes the calibration curve, every model's prediction and the
    measured value on one draw of queries, and the spread of the difference is the noise.
    """
    import scale_transfer as fleet

    pred = json.load(open(pred_path))
    _, per_query = curve(qname, out_dir)
    n = np.array(SIZES, float) * SHARD_ROWS
    all_srv = list(range(fleet.N_SERVERS))
    ref = fleet.load(results, f"ref{tag}_part_{{s}}.npz", all_srv)[:2]
    ref_top = fleet.top10(*ref, all_srv)
    rng = np.random.default_rng(BOOT_SEED)
    rec = {
        "query_set": qname,
        "tag": tag,
        "prediction_file_sha256": sha(pred_path),
        "script_sha256": sha(os.path.abspath(__file__)),
        "widths": {},
    }
    for p, reg in pred["registered_models"].items():
        routed = fleet.load(results, f"ivf{tag}_p{p}_part_{{s}}.npz", all_srv)[:2]
        meas_q = fleet.per_query_recall(fleet.top10(*routed, all_srv), ref_top)
        cal_q = per_query[int(p)]  # (sizes, nq)
        nq = len(meas_q)
        assert cal_q.shape[1] == nq, "calibration and run query counts differ"
        measured = float(meas_q.mean())
        diffs = {m: [] for m in MODELS}
        for _ in range(BOOT):
            idx = rng.integers(0, nq, nq)
            pb = fit_predict(n, cal_q[:, idx].mean(axis=1))
            mb = meas_q[idx].mean()
            for m in MODELS:
                diffs[m].append(pb[m] - mb)
        graded = {}
        for m in MODELS:
            err = pred["widths"][p]["prediction_1e12"][m] - measured
            se = float(np.std(diffs[m], ddof=1))
            graded[m] = {
                "prediction": pred["widths"][p]["prediction_1e12"][m],
                "error": err,
                "se_of_difference": se,
                "pass": bool(abs(err) <= 2 * se) if se > 0 else bool(abs(err) < 1e-12),
            }
        rec["widths"][p] = {
            "measured_1e12": measured,
            "models": graded,
            "registered_model": reg,
            "H1_pass": graded[reg]["pass"],
            "H2_beats_no_change": abs(graded[reg]["error"])
            < abs(graded["M0"]["error"]),
        }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("list", "build", "measure", "curve", "grade"))
    ap.add_argument("--boot", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument(
        "--out", required=True, help="directory for measurement files and JSON"
    )
    ap.add_argument(
        "--shards", help="build: comma-separated global shard ids (default all)"
    )
    ap.add_argument("--qname")
    ap.add_argument("--queries", help="measure: .npy of the query set")
    ap.add_argument("--perm", type=int)
    ap.add_argument("--size", type=int)
    ap.add_argument(
        "--dev", action="store_true", help="curve: add the member-run comparison"
    )
    ap.add_argument(
        "--registered",
        help='curve: JSON map width -> registered model, e.g. {"32":"M2","128":"M1"}',
    )
    ap.add_argument("--prediction", help="grade: the committed prediction JSON")
    ap.add_argument("--results", help="grade: directory of the run's 500 partials")
    ap.add_argument("--tag", default="1tnm", help="grade: the registered run's tag")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.phase == "list":
        print(json.dumps({"permutations": permutations(), "all_shards": all_shards()}))
    elif a.phase == "build":
        todo = [int(x) for x in a.shards.split(",")] if a.shards else all_shards()
        for g in todo:
            build_one(g, a.pool, a.boot)
            print("built", g, flush=True)
    elif a.phase == "measure":
        q = np.load(a.queries)
        print(
            measure(a.qname, q, a.perm, a.size, a.pool, a.work, a.boot, a.out),
            flush=True,
        )
    elif a.phase == "grade":
        rec = grade(a.qname, a.out, a.prediction, a.results, a.tag)
        path = os.path.join(a.out, f"scale_transfer_small_grade_{a.qname}.json")
        json.dump(rec, open(path, "w"), indent=1)
        print("GRADE_JSON " + json.dumps(rec), flush=True)
    else:
        rec, _ = curve(a.qname, a.out)
        rec["script_sha256"] = sha(os.path.abspath(__file__))
        if a.registered:
            rec["registered_models"] = json.loads(a.registered)
        if a.dev:
            for p, w in rec["widths"].items():
                meas = MEMBER_1E12[int(p)]
                w["measured_1e12"] = meas
                w["error"] = {mo: w["prediction_1e12"][mo] - meas for mo in MODELS}
        path = os.path.join(a.out, f"scale_transfer_small_{a.qname}.json")
        json.dump(rec, open(path, "w"), indent=1)
        print("CURVE_JSON " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    sys.exit(main())
