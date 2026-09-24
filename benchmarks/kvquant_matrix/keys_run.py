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
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import keys_grid as KG  # noqa: E402

PY = sys.executable


def priority_arms() -> list[str]:
    head = list(KG.G0_ARMS) + ["fp16"]
    scored = []
    for pairs in KG.COMPARISONS.values():
        for trio in pairs:
            scored += [a for a in trio if a not in scored]
    return head + scored


def all_arms() -> list[str]:
    out = priority_arms()
    for trio in KG.REPORTED:
        out += [a for a in trio if a not in out]
    return out + [a for a in KG.ARMS if a not in out]


def gpu_temp(gpu: int) -> int:
    q = ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits",
         "-i", str(gpu)]
    return int(subprocess.check_output(q).decode().strip())


def gpu_busy(gpu: int) -> bool:
    """Another process holds this GPU (Atlas is shared: never start on top of it)."""
    q = ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"]
    return bool(subprocess.check_output(q).decode().split())


OOM_RETRIES = 6
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

    head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE).decode().strip()
    say(f"=== start harness={head} models={list(models)} arms={len(arms)} docs={a.docs or 'all'}")
    for mk, hf in models.items():
        for arm in arms:
            line = {**KG.ARMS, **KG.G0_ARMS}[arm]
            tasks = KG.G0_TASKS if arm in KG.G0_ARMS else KG.TASKS
            cell = os.path.join(a.root, mk, arm)
            os.makedirs(cell, exist_ok=True)
            env = {**os.environ, "MODEL": hf, "MODEL_KEY": mk, "TAG": arm,
                   "DATASETS": ",".join(tasks), "SHARD_ID": "0", "NUM_SHARDS": str(nsh),
                   "OUT_DIR": cell, "LBROOT": a.lbroot, "DATADIR": a.datadir,
                   "CUDA_VISIBLE_DEVICES": str(a.gpu),
                   "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                   "BASIS_CALIB": os.path.join(a.root, mk, "basis_calib.pt")}
            env.update(dict(kv.split("=", 1) for kv in line.split()))
            if not lb_done(cell, tasks, nsh):
                t0 = time.time()
                rc = watched([PY, os.path.join(HERE, "keys_cell.py")], env,
                             f"{cell}/lb.log", a.gpu)
                say(f"{mk} {arm} lb rc={rc} {time.time() - t0:.0f}s")
            if not ppl_done(cell):
                t0 = time.time()
                penv = {**env, "CHUNKS_OUT": f"{cell}/ppl_chunks.jsonl",
                        "MAXCHUNKS": str(a.ppl_chunks), "OMP_NUM_THREADS": "8"}
                rc = watched([PY, os.path.join(HERE, "wikitext_ppl.py")], penv,
                             f"{cell}/ppl.log", a.gpu)
                if rc == 0:
                    open(f"{cell}/ppl.done", "w").write(json.dumps({"harness": head}))
                ppl = [x for x in open(f"{cell}/ppl.log", encoding="utf-8") if x.startswith("PPL")]
                say(f"{mk} {arm} ppl rc={rc} {time.time() - t0:.0f}s {ppl[-1].strip() if ppl else ''}")
    say("=== done")


if __name__ == "__main__":
    main()
