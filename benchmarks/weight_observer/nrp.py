"""Part III on NRP Nautilus (namespace ssu-atlas-ai) through nats-bursting. Run on Atlas.

    python -m weight_observer.nrp setup  [--dry-run]    # CPU: pinned venv tar on the volume
    python -m weight_observer.nrp code --commit SHA      # CPU: the pinned harness as one tar
    python -m weight_observer.nrp stage --commit SHA     # CPU: model weights + WikiText text
    python -m weight_observer.nrp run --commit SHA --models qwen2.5-0.5b [--dry-run]

The GET G3c discipline (experiments/G3c/nrp/submit.py), scored in ``preflight``:
CPU jobs sit in the exempt class (1 CPU, 2 GiB); GPU pods install and download nothing (the
venv, weights, text and code are staged by CPU jobs); requests equal limits; ephemeral
storage is declared; nothing sleeps; a GPU pod's CPU and memory come from the MEASURED usage
of an earlier run of that model or, for a model never run, from the pilot's measurement
scaled by parameter count (the pilot itself is sized by a stated model and runs under the
utilization guard); each model runs on one GPU product. ``watch`` deletes a GPU job whose
utilization stays under 40% for three samples after its warm-up.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, "/home/claude/src/nats-bursting/python")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

NS = "ssu-atlas-ai"
PVC = "tqp-rbq-data"
APP = "tqp-wo"
BATCH = "tqp-weight-observer"
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"
CPU_ZONE = {"topology.kubernetes.io/zone": "ucsd-nrp"}
GPU_PRODUCT = "NVIDIA-A10"
REPO = "https://github.com/ahb-sjsu/turboquant-pro"
PINS = "transformers==4.56.1 accelerate==1.14.0 safetensors numpy huggingface_hub"
ROOT = "/data/wo"
STATE = "/archive/ahb-sjsu/tqp_weight_observer"
OBSERVATIONS = os.path.join(STATE, "observations.json")
MODELS = {  # key: (hf id, parameters in billions)
    "qwen2.5-0.5b": (
        "Qwen/Qwen2.5-0.5B",
        0.5,
    ),  # pilot: wiring and sizing, never scored
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B", 1.5),
    "llama3.2-3b": ("unsloth/Llama-3.2-3B", 3.2),
}
nl = chr(10)

SETUP = f"""set -euo pipefail
export PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1
if [ -f {ROOT}/env/env.tar ]; then echo "env exists"; exit 0; fi
python -m venv --system-site-packages /tmp/venv
/tmp/venv/bin/pip install -q --no-cache-dir {PINS}
/tmp/venv/bin/python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
/tmp/venv/bin/pip freeze > /tmp/venv/freeze.txt
mkdir -p {ROOT}/env && tar -cf {ROOT}/env/env.tar.tmp -C /tmp venv
mv {ROOT}/env/env.tar.tmp {ROOT}/env/env.tar
df -h /data | tail -1
echo SETUP_DONE
"""


def stage_script(commit: str, what: str) -> str:
    """A CPU job that stages one thing with the pinned code: 'text' or a model key."""
    head = f"""set -euo pipefail
export PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
tar -xf {ROOT}/env/env.tar -C /tmp
mkdir -p /tmp/code && tar -xf {ROOT}/code/{commit}.tar -C /tmp/code
export PYTHONPATH=/tmp/code
"""
    if what == "text":
        return head + nl.join(
            [
                "/tmp/venv/bin/pip install -q --no-cache-dir datasets==4.3.0",
                f"/tmp/venv/bin/python -m weight_observer.stage_text --dest {ROOT}/text",
                "",
            ]
        )
    hf = MODELS[what][0]
    return head + (
        "/tmp/venv/bin/python -m weight_observer.stage_models "
        f"--model-id {hf} --dest {ROOT}/models/{what}" + nl
    )


def code_script(commit: str) -> str:
    tar = f"{ROOT}/code/{commit}.tar"
    return f"""set -euo pipefail
if [ -f {tar} ]; then echo "already staged: {tar}"; exit 0; fi
git clone -q --filter=blob:none --no-checkout {REPO} /tmp/src
git -C /tmp/src sparse-checkout set --no-cone benchmarks/weight_observer
git -C /tmp/src checkout -q {commit}
test "$(git -C /tmp/src rev-parse HEAD)" = "{commit}"
mkdir -p {ROOT}/code
tar -cf {tar}.tmp.$$ -C /tmp/src/benchmarks weight_observer
mv {tar}.tmp.$$ {tar}
echo "staged {tar}"
"""


def run_script(commit: str, key: str, tag: str = "", pilot_env: str = "") -> str:
    """``tag`` and ``pilot_env`` (e.g. WO_RATES=...,WO_PER_RATE=...) are for pilots only."""
    exports = " ".join(pilot_env.split(",")) if pilot_env else ""
    out = f"{key}-{tag}" if tag else key
    return f"""set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{"export " + exports if exports else ""}
tar -xf {ROOT}/env/env.tar -C /tmp
mkdir -p /tmp/code && tar -xf {ROOT}/code/{commit}.tar -C /tmp/code
export PATH=/tmp/venv/bin:$PATH PYTHONPATH=/tmp/code
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m weight_observer.run --model-key {key} --model-path {ROOT}/models/{key} \\
    --text {ROOT}/text --out {ROOT}/runs/{out}
echo RUN_DONE {key}
"""


# ----------------------------------------------------------------------------- sizing


def measured(key: str):
    """(cpu cores mean, mem GiB mean, mem GiB peak) of a finished run of this model."""
    try:
        obs = json.load(open(OBSERVATIONS, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    rows = [o for n, o in obs.items() if n.startswith(f"wo-run-{key.replace('.', '')}")]
    rows = [o for o in rows if o.get("mean_cpu_cores") and o.get("mean_mem_gib")]
    if not rows:
        return None
    return (
        min(o["mean_cpu_cores"] for o in rows),
        min(o["mean_mem_gib"] for o in rows),
        max(o.get("peak_mem_gib") or o["mean_mem_gib"] for o in rows),
    )


def request(key: str):
    """(cpu, mem GiB, why). Measured if this model ran; else the pilot scaled by size; the
    pilot itself by a stated model (host RAM holds two fp16 copies while loading)."""
    m = measured(key)
    if m:
        cpu, mean, peak = m
        mem = max(int(peak * 1.25 + 0.999), 1)
        return (
            max(1, round(cpu / 0.6)),
            mem,
            f"measured: {cpu:.2f} cores, {mean:.1f}/{peak:.1f} GiB",
        )
    pilot = measured("qwen2.5-0.5b")
    if key != "qwen2.5-0.5b":
        if not pilot:
            raise SystemExit(f"{key}: the pilot has not been measured yet")
        cpu, mean, peak = pilot
        s = MODELS[key][1] / MODELS["qwen2.5-0.5b"][1]
        mem = int((peak - 1.5) * s + 1.5 + 0.999) + 1
        return max(1, round(cpu / 0.6)), mem, f"pilot scaled x{s:.1f}"
    return 2, 6, "pilot model: 2 x 1 GB fp16 in host RAM while loading, + 3 GiB runtime"


def preflight(desc, gpu: bool) -> list:
    bad = []
    r = desc.resources
    if not r.ephemeral_storage:
        bad.append("ephemeral-storage not declared")
    if desc.backoff_limit != 0:
        bad.append("backoff_limit is not 0")
    script = " ".join(desc.command)
    if re.search(r"\bsleep\b", script):
        bad.append("the command contains sleep")
    if gpu and any(
        w in script
        for w in ("pip install", "git clone", "snapshot_download", "apt-get")
    ):
        bad.append("a GPU job installs or downloads")
    if not gpu and (float(r.cpu) > 1 or float(str(r.memory).rstrip("Gi")) > 2):
        bad.append("a CPU job outside the exempt class")
    return bad


def descriptor(name, script, cpu, mem_gib, eph, role, gpu=0, image=IMAGE):
    from nats_bursting import JobDescriptor, Resources, Volume

    return JobDescriptor(
        name=name,
        image=image,
        command=["/bin/bash", "-lc", script],
        resources=Resources(
            cpu=str(cpu), memory=f"{mem_gib}Gi", gpu=gpu, ephemeral_storage=eph
        ),
        labels={"app": APP, "atlas.io/batch": BATCH, "atlas.io/role": role},
        node_selector=(
            {"nvidia.com/gpu.product": GPU_PRODUCT} if gpu else dict(CPU_ZONE)
        ),
        backoff_limit=0,
        volumes=[Volume(name="data", mount_path="/data", claim_name=PVC)],
    )


def kubectl(*args):
    return subprocess.run(
        ["kubectl", "-n", NS, "--request-timeout=60s", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def submit(desc) -> None:
    from nats_bursting import Client

    kubectl("delete", "job", desc.name, "--ignore-not-found")
    t0 = time.time()
    with Client() as c:
        c.submit(desc)
    for _ in range(60):
        r = kubectl(
            "get", "job", desc.name, "-o", "jsonpath={.metadata.creationTimestamp}"
        )
        if r.returncode == 0 and r.stdout.strip():
            ts = (
                time.mktime(time.strptime(r.stdout.strip(), "%Y-%m-%dT%H:%M:%SZ"))
                - time.timezone
            )
            if ts < t0 - 120:
                raise SystemExit(
                    f"{desc.name}: stale creationTimestamp {r.stdout.strip()}"
                )
            print(desc.name, "created", r.stdout.strip(), flush=True)
            return
        time.sleep(5)
    raise SystemExit(f"{desc.name}: no job appeared")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("setup", "stage", "code", "run"))
    ap.add_argument("--commit", default="")
    ap.add_argument("--models", default="")
    ap.add_argument("--tag", default="", help="pilot runs only: output and job suffix")
    ap.add_argument("--pilot-env", default="", help="pilot only: K=V,K=V overrides")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    items = []
    if a.cmd == "setup":
        items.append((descriptor("wo-setup", SETUP, 1, 2, "12Gi", "setup"), False))
    elif a.cmd == "code":
        if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
            raise SystemExit("--commit must be a full sha")
        items.append(
            (
                descriptor(
                    f"wo-code-{a.commit[:12]}",
                    code_script(a.commit),
                    1,
                    2,
                    "2Gi",
                    "code",
                    image="python:3.12",  # has git; the PyTorch image does not
                ),
                False,
            )
        )
    elif a.cmd == "stage":
        if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
            raise SystemExit(
                "--commit must be a full sha (the staging code is pinned too)"
            )
        for what in ("text", *MODELS):
            name = f"wo-stage-{what.replace('.', '')}"
            items.append(
                (
                    descriptor(
                        name, stage_script(a.commit, what), 1, 2, "12Gi", "stage"
                    ),
                    False,
                )
            )
    elif a.cmd == "run":
        if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
            raise SystemExit("--commit must be a full sha")
        for key in a.models.split(","):
            cpu, mem, why = request(key)
            d = descriptor(
                f"wo-run-{key.replace('.', '')}" + (f"-{a.tag}" if a.tag else ""),
                run_script(a.commit, key, a.tag, a.pilot_env),
                cpu,
                mem,
                "20Gi",
                "run",
                gpu=1,
            )
            print(d.name, cpu, f"{mem}Gi", GPU_PRODUCT, "|", why)
            items.append((d, True))
    bad = {d.name: preflight(d, g) for d, g in items}
    if any(bad.values()):
        raise SystemExit(
            "PREFLIGHT VETO " + json.dumps({k: v for k, v in bad.items() if v})
        )
    print(f"{a.cmd}: {len(items)} job(s) pass preflight")
    if a.dry_run:
        return 0
    for d, _ in items:
        submit(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
