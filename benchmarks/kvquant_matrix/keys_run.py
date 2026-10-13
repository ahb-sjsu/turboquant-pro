#!/usr/bin/env python3
"""Runner of observer-advantage Part II on one owned GPU (Atlas), resumable per cell.

    python keys_run.py --root /archive/ahb-sjsu/keys --models tierB --arms priority --gpu 1

For every (model, arm) in keys_grid order it runs the LongBench cell through the
harness's own ``main()`` (``keys_cell.py``) and then the WikiText-2 perplexity
(``wikitext_ppl.py`` with CHUNKS_OUT), each under a thermal watchdog: no cell starts
above 75 C, a running cell is paused above 83 C and resumed below 72 C. A cell whose
outputs are complete is skipped, so the runner can be stopped and restarted. A cell
is never rerun because of its result. Atlas is shared: no cell starts while any other
process holds the GPU, so the campaign yields to other work between cells.

--arms: ``priority`` (G0, fp16, then every arm a registered verdict reads),
``all`` (priority, then every reported arm), or a comma list.
--docs N limits each task to documents idx % (200 // N) == 0 (smoke only; the
registered runs use all documents).

--grid part5 runs observer-advantage Part V's grid (part5_grid.py,
docs/PREREG_full_consumer_kv.md) through the same cells; the default is Part II's
registered grid, whose behaviour is unchanged. A calibrated head or Fisher weight is
written to its own file per model (alloc_calib_<alloc>.pt), never to the shared
basis_calib.pt that Part II's calibrated arms use.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import keys_grid as KG  # noqa: E402

if "--grid" in sys.argv and sys.argv[sys.argv.index("--grid") + 1] == "part5":
    import part5_grid as KG  # noqa: E402,F811

PY = sys.executable


def _lines() -> dict:
    """Every env line this grid can run (Part V adds its harness baselines)."""
    return KG.all_lines() if hasattr(KG, "all_lines") else {**KG.ARMS, **KG.G0_ARMS}


def matched_aatc_bits(root: str, mk: str) -> tuple[float, float]:
    """(AATC's average code bits, u3's mean stored key bits per element) for AATC at
    u3's bytes (part5_grid.BASELINE_ARMS): u3's mean over its qasper documents minus
    AATC's per-token scale bits. u3 must have run on this model."""
    rows = []
    for f in sorted(glob.glob(os.path.join(root, mk, "u3", "qasper.*.jsonl"))):
        rows += [json.loads(x) for x in open(f, encoding="utf-8")]
    bits = [r["key_bits"]["total"] for r in rows if r.get("key_bits")]
    if not bits:
        raise SystemExit(f"{mk}: AATC at u3's bytes needs u3's qasper cell first")
    ref = sum(bits) / len(bits)
    return ref - KG.AATC_SCALE_BITS, ref


def write_baseline(cell: str, arm: str, ref_bits: float) -> None:
    rows = []
    for f in sorted(glob.glob(os.path.join(cell, "qasper.*.jsonl"))):
        rows += [json.loads(x) for x in open(f, encoding="utf-8")]
    bits = [r["key_bits"]["total"] for r in rows if r.get("key_bits")]
    json.dump({"baseline": arm, "matched": {"reference": "u3", "arm_bits": sum(bits) / len(bits),
                                            "reference_bits": ref_bits, "kind": "stored key bits"}},
              open(os.path.join(cell, "baseline.json"), "w"), indent=1)


def _harness_arm(a: str) -> bool:
    """An arm this harness runs (Part V's baselines have their own runners)."""
    return a in KG.ARMS or a in KG.G0_ARMS


def priority_arms() -> list[str]:
    head = list(KG.G0_ARMS) + ["fp16"]
    scored = []
    for pairs in KG.COMPARISONS.values():
        for trio in pairs:
            scored += [a for a in trio if a not in scored and a not in head]
    return head + [a for a in scored if _harness_arm(a)]


def all_arms() -> list[str]:
    out = priority_arms()
    for trio in KG.REPORTED:
        out += [a for a in trio if a not in out and _harness_arm(a)]
    return out + [a for a in KG.ARMS if a not in out]


def gpu_temp(gpu: int) -> int:
    q = ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits",
         "-i", str(gpu)]
    return int(subprocess.check_output(q).decode().strip())


def gpu_busy(gpu: int) -> bool:
    """Another process holds this GPU (Atlas is shared: never start on top of it)."""
    q = ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"]
    return bool(subprocess.check_output(q).decode().split())


OOM_RETRIES = int(os.environ.get("KEYS_OOM_RETRIES", "6"))
# KEYS_OOM_RETRIES=0 on a dedicated cluster pod: no other job can claim its GPU, so an
# out-of-memory start is deterministic and a retry would only idle the GPU (Part V).
OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError")


def watched(cmd: list[str], env: dict, log: str, gpu: int) -> int:
    """Run ``cmd`` on a free, cool GPU; an out-of-memory start is retried.

    Starting only on a free GPU still races another job that claims it in the
    same minute. That is an operational failure, which the registration reruns
    unchanged, so the runner waits for the GPU and tries again instead of
    leaving the cell for a later pass."""
    for attempt in range(OOM_RETRIES + 1):
        rc = _watched_once(cmd, env, log, gpu)
        if rc == 0:
            return rc
        text = open(log, encoding="utf-8", errors="replace").read()
        if not any(m in text for m in OOM_MARKERS) or attempt == OOM_RETRIES:
            return rc
        os.replace(log, f"{log}.oom{attempt}")
        time.sleep(120)
    return rc


def _watched_once(cmd: list[str], env: dict, log: str, gpu: int) -> int:
    while gpu_busy(gpu) or gpu_temp(gpu) > 75:
        time.sleep(30)
    with open(log, "w") as fo:
        p = subprocess.Popen(cmd, env=env, stdout=fo, stderr=subprocess.STDOUT, cwd=HERE)
        paused = False
        while p.poll() is None:
            t = gpu_temp(gpu)
            if not paused and t > 83:
                p.send_signal(signal.SIGSTOP)
                paused = True
            elif paused and t < 72:
                p.send_signal(signal.SIGCONT)
                paused = False
            time.sleep(10)
    return p.returncode


def lb_done(cell: str, tasks, nsh: int) -> bool:
    for t in tasks:
        f = f"{cell}/{t}.0.jsonl"
        if not os.path.exists(f):
            return False
        n = sum(1 for _ in open(f, encoding="utf-8"))
        want = len(range(0, 200, nsh)) if t != "multifieldqa_en" else len(range(0, 150, nsh))
        if n < want:
            return False
    return True


def ppl_done(cell: str) -> bool:
    return os.path.exists(f"{cell}/ppl.done")


def environment(gpu: int) -> dict:
    """What a cell ran on. The scorer compares an arm only with a reference that ran
    on the same GPU product and software (greedy decoding is not bit-stable across
    them), so every cell records it."""
    import importlib.metadata as md

    q = ["nvidia-smi", "-i", str(gpu), "--query-gpu=name", "--format=csv,noheader"]
    try:
        name = subprocess.check_output(q).decode().strip()
    except Exception:  # noqa: BLE001
        name = "unknown"
    pkgs = {}
    for p in ("torch", "transformers", "tokenizers", "accelerate"):
        try:
            pkgs[p] = md.version(p)
        except md.PackageNotFoundError:
            pkgs[p] = None
    return {"gpu": name, **pkgs}


def record_env(cell: str, env: dict) -> None:
    path = f"{cell}/env.json"
    old = json.load(open(path)) if os.path.exists(path) else None
    if old and old != env:
        raise SystemExit(
            f"{cell}: started on {old}, now {env}; a cell must finish where it began"
        )
    json.dump(env, open(path, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--models", default="tierB")
    ap.add_argument("--arms", default="priority")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--docs", type=int, default=0)
    ap.add_argument("--ppl-chunks", type=int, default=0, help="0 = every chunk")
    ap.add_argument("--lbroot", default=os.path.expanduser("~/item4c/LongBench/LongBench"))
    ap.add_argument("--datadir", default="/archive/longbench/data")
    ap.add_argument("--grid", default="keys", choices=("keys", "part5"))
    ap.add_argument("--aatc-text", default="/data/kv5/aatc_text.jsonl",
                    help="AATC's staged calibration text (part5_aatc_text.py)")
    ap.add_argument("--model-dir", default="",
                    help="load <model-dir>/<model key> instead of the hub id (NRP pods)")
    a = ap.parse_args()

    models = {"tierA": KG.TIER_A, "tierB": KG.TIER_B, "all": KG.MODELS}.get(a.models)
    if models is None:
        models = {m: KG.MODELS[m] for m in a.models.split(",")}
    arms = {"priority": priority_arms, "all": all_arms}.get(a.arms)
    arms = arms() if arms else a.arms.split(",")
    nsh = 200 // a.docs if a.docs else 1
    os.makedirs(a.root, exist_ok=True)
    log = open(os.path.join(a.root, "run.log"), "a")

    def say(msg):
        log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        log.flush()
        print(f"[run] {msg}", flush=True)  # progress in the pod log (Part V telemetry)

    try:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        head = os.environ["HARNESS_COMMIT"][:7]  # a pod's code tar carries no .git
    say(f"=== start harness={head} models={list(models)} arms={len(arms)} docs={a.docs or 'all'}")
    for mk, hf in models.items():
        for arm in arms:
            line = _lines()[arm]
            tasks = KG.G0_TASKS if arm in KG.G0_ARMS else KG.TASKS
            cell = os.path.join(a.root, mk, arm)
            os.makedirs(cell, exist_ok=True)
            src = os.path.join(a.model_dir, mk) if a.model_dir else hf
            env = {**os.environ, "MODEL": src, "MODEL_KEY": mk, "TAG": arm,
                   "DATASETS": ",".join(tasks), "SHARD_ID": "0", "NUM_SHARDS": str(nsh),
                   "OUT_DIR": cell, "LBROOT": a.lbroot, "DATADIR": a.datadir,
                   "CUDA_VISIBLE_DEVICES": str(a.gpu),
                   "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                   "BASIS_CALIB": os.path.join(a.root, mk, "basis_calib.pt")}
            env.update(dict(kv.split("=", 1) for kv in line.split()))
            ref_bits = None
            if env.get("AATC"):  # Part V's AATC baseline: its calibration and its budget
                kind = "whiten" if env.get("AATC_TRANSFORM", "1") == "1" else "native"
                env["AATC_CALIB"] = os.path.join(a.root, mk, f"aatc_calib_{kind}.pt")
                env["AATC_TEXT"] = a.aatc_text
                if env.get("AATC_KEY_BITS") == "match":
                    b, ref_bits = matched_aatc_bits(a.root, mk)
                    env["AATC_KEY_BITS"] = f"{b:.6f}"
            if env.get("KEY_ALLOC") in ("head", "fisher"):  # Part V: one file per weight
                env["BASIS_CALIB"] = os.path.join(a.root, mk, f"alloc_calib_{env['KEY_ALLOC']}.pt")
            env["RESUME"] = "1"
            record_env(cell, environment(a.gpu))
            if not lb_done(cell, tasks, nsh):
                t0 = time.time()
                rc = watched([PY, os.path.join(HERE, "keys_cell.py")], env,
                             f"{cell}/lb.log", a.gpu)
                say(f"{mk} {arm} lb rc={rc} {time.time() - t0:.0f}s")
            if not ppl_done(cell):
                t0 = time.time()
                penv = {**env, "CHUNKS_OUT": f"{cell}/ppl_chunks.jsonl",
                        "MAXCHUNKS": str(a.ppl_chunks), "OMP_NUM_THREADS": "8"}
                if a.grid == "part5":  # Part V reports per-chunk KL to fp16 (same process)
                    penv["KL_REF"] = "1"
                rc = watched([PY, os.path.join(HERE, "wikitext_ppl.py")], penv,
                             f"{cell}/ppl.log", a.gpu)
                if rc == 0:
                    # the limits it ran with: Part V's scorer counts only full cells
                    open(f"{cell}/ppl.done", "w").write(json.dumps(
                        {"harness": head, "maxchunks": a.ppl_chunks, "docs": a.docs}))
                ppl = [x for x in open(f"{cell}/ppl.log", encoding="utf-8") if x.startswith("PPL")]
                say(f"{mk} {arm} ppl rc={rc} {time.time() - t0:.0f}s {ppl[-1].strip() if ppl else ''}")
            if ref_bits is not None and lb_done(cell, tasks, nsh) and ppl_done(cell):
                write_baseline(cell, arm, ref_bits)
    say("=== done")


if __name__ == "__main__":
    main()
