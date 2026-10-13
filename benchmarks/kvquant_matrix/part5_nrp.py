#!/usr/bin/env python3
"""Observer-advantage Part V on NRP Nautilus A10 pods, through nats-bursting. Run on Atlas.

    python part5_nrp.py setup [--dry-run]               # CPU: pinned venv tar on the volume
    python part5_nrp.py code  --commit SHA              # CPU: the pinned harness as one tar
    python part5_nrp.py stage --commit SHA --what text  # CPU: WikiText-2 (HF datasets cache)
    python part5_nrp.py stage --commit SHA --what lb    # CPU: LongBench data + config, hash-checked
    python part5_nrp.py stage --commit SHA --what qwen2.5-1.5b-instruct   # CPU: one model
    python part5_nrp.py smoke --commit SHA --model qwen2.5-1.5b-instruct [--dry-run]
    python part5_nrp.py fetch --commit SHA --model qwen2.5-1.5b-instruct --what smoke
    python part5_nrp.py set-run --commit SHA --model qwen2.5-7b-instruct --wave gate [--dry-run]
    python part5_nrp.py fetch-set --commit SHA --model qwen2.5-7b-instruct --what run

docs/PREREG_full_consumer_kv.md, section 6. The discipline is Part III's
(``weight_observer/nrp.py``, whose preflight, submission and kubectl helpers this reuses):
CPU jobs sit in the exempt class (1 CPU, 2 GiB); GPU pods install and download nothing
(the venv, weights, text and LongBench inputs are staged by CPU jobs); requests equal
limits; ephemeral storage is declared; nothing sleeps; every model runs on an A10. A GPU
pod copies its checkpoint to local disk while the environment unpacks, then runs Part II's
runner (``keys_run.py --grid part5``) over the arms. Before submitting a GPU job, start the
Atlas-side watchers for app ``tqp-kv5``: ``part5_watch_gpu.sh`` and
``benchmarks/nrp/utilization_guard.py --selector app=tqp-kv5 --heartbeat <STATE>/guard.hb``.

The staged LongBench inputs must hash to the files Part II scored (recorded below from
Atlas, 2026-10-12); a mismatch fails the staging job.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))  # benchmarks/, for weight_observer
from weight_observer import nrp as W  # noqa: E402  (preflight, submit, kubectl)

import part5_grid as P  # noqa: E402

NS = W.NS
PVC = W.PVC
APP = "tqp-kv5"
BATCH = "tqp-part5-kv"
IMAGE = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime"  # torch 2.10.0, Part II's
CODE_IMAGE = "python:3.12"  # the pytorch image has no git (Part III's code jobs use this)
GPU_PRODUCT = "NVIDIA-A10"
REPO = W.REPO
# Part II's environment (Atlas, 2026-10-12), torch from the image.
PINS = (
    "transformers==5.5.0 accelerate==1.13.0 datasets==4.3.0 sentencepiece==0.2.1 "
    "tokenizers==0.22.2 huggingface_hub==1.11.0 safetensors==0.7.0 protobuf"
)
ROOT = "/data/kv5"
STATE = "/archive/ahb-sjsu/tqp_part5"

# ---------------------------------------------------------------- linstor volume sets (Tier A)
# A 7B checkpoint read from CephFS idles an A10 for minutes: the smoke copied 3.1 GB in 4 min
# (13 MB/s, 2026-10-13), and the GTC twin measured CephFS-east at 15-38 MB/s against
# 149 MB/s per linstor-unl volume and 578 MB/s for four read in parallel (2026-10-03,
# gtc-prototype twin/ieip/nrp.py, whose layout this follows). Each Tier A model gets its own
# SET of RWO linstor-unl volumes in zone unl (18 A10 nodes), one per weight file, so its GPU
# pod reads every shard at once with transformers' parallel loader; volume 0 also holds the
# environment, the code, the LongBench and WikiText inputs and the run's outputs, so a pod
# reads nothing from another region. Shard sizes (GB) from the Hub at the revisions Atlas's
# cache holds (2026-10-13); hashes in part5_expect.json, from the same cache.
LIN_CLASS = "linstor-unl"
UNL = {"topology.kubernetes.io/zone": "unl"}
SHARDS_GB = {
    "llama2-7b-chat-4k": (9.98, 3.50),
    "mistral-7b-instruct": (4.94, 5.00, 4.54),
    "qwen2.5-7b-instruct": (3.95, 3.86, 3.86, 3.56),
}
VOL0_EXTRA_GIB = 6  # env tar, code tars, inputs, HF datasets cache, outputs
EXPECT_FILE = os.path.join(HERE, "part5_expect.json")


def set_volumes(key: str) -> list:
    """[(pvc name, GiB, mount path)] of a model's volume set: one per weight file."""
    out = []
    for k, gb in enumerate(SHARDS_GB[key]):
        # shard bytes in GiB, +12% and 2 GiB for the filesystem (ext4 reserves ~5%) and slack
        gib = int(gb * 1.12 / 1.074 + 2) + (VOL0_EXTRA_GIB if k == 0 else 0)
        out.append((f"kv5-{short(key)}-w{k}", gib, f"/w{k}"))
    return out


def pvc_yaml(key: str) -> str:
    return "".join(
        f"""---
apiVersion: v1
kind: PersistentVolumeClaim
metadata: {{name: {n}, namespace: {NS}, labels: {{app: {APP}, atlas.io/batch: {BATCH}}}}}
spec:
  accessModes: [ReadWriteOnce]
  storageClassName: {LIN_CLASS}
  resources: {{requests: {{storage: {g}Gi}}}}
"""
        for n, g, _ in set_volumes(key)
    )


def stage_set_script(commit: str, key: str) -> str:
    """A CPU job in zone unl: the smoke's environment, the code and the inputs from the shared
    volume onto volume 0, then the model's weight files across the set, each verified against
    Atlas's cache hashes, and the parallel read speed the GPU pod will see."""
    import json

    exp = json.load(open(EXPECT_FILE, encoding="utf-8"))[key]
    shards = ",".join(f"{m}/shards" for _, _, m in set_volumes(key))
    return f"""set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
mkdir -p /w0/code /w0/lb /w0/hf
timeout 1800 cp {ROOT}/env/env.tar /w0/env.tar.part && mv /w0/env.tar.part /w0/env.tar
timeout 600 cp {ROOT}/code/{commit}.tar /w0/code/{commit}.tar
timeout 600 cp -r {ROOT}/lb/. /w0/lb/
timeout 1800 cp -r {ROOT}/hf/datasets /w0/hf/
tar -xf /w0/env.tar -C /tmp
mkdir -p /tmp/code && tar -xf /w0/code/{commit}.tar -C /tmp/code
cat > /tmp/expect.json <<'EOF'
{json.dumps(exp["files"])}
EOF
timeout 7200 /tmp/venv/bin/python /tmp/code/kvquant_matrix/part5_stage_model.py \\
    --model-id {exp["model_id"]} --revision {exp["revision"]} --dest /w0/model \\
    --shard-dests {shards} --expect /tmp/expect.json
df -h {" ".join(m for _, _, m in set_volumes(key))}
echo STAGE_SET_DONE {key}
"""


# A set GPU pod's stated shape (its first run is its measurement): the checkpoint goes
# straight to the GPU (DIRECT_LOAD=1), so the host holds the CUDA context and libraries, the
# tokenized inputs and the runner (~3-4 GiB); two cores for the loader's threads. The local
# disk holds the verified fp16 copy every cell loads (part5_cache_fp16.py, ~15 GB at 7B).
SET_SHAPE = {"cpu": 2, "mem_gib": 6, "eph": "30Gi"}
SET_SMOKE_DEADLINE = 4 * 3600


def set_smoke_script(commit: str, key: str) -> str:
    """The wiring smoke on a 7B model from its linstor set: every harness arm on documents
    0, 50, 100 and 150 and 8 perplexity chunks, under pod_guard. It also measures what the
    prereg asks of a 7B model: the Fisher calibration's peak GPU memory at PART5_CALIB_SEQ
    (printed by the harness), the load time from the set, and the pod's utilization."""
    out = f"/w0/runs/{commit[:12]}/set-smoke"
    return f"""set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/w0/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HARNESS_COMMIT={commit}
export DIRECT_LOAD=1 KEYS_OOM_RETRIES=0  # loads straight to the GPU; an OOM never retries in a pod
t() {{ echo "[t] $(date -u +%T) $*"; }}
t start
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
tar -xf /w0/env.tar -C /tmp
mkdir -p /tmp/code && tar -xf {ROOT}/code/{commit}.tar -C /tmp/code
mkdir -p /tmp/model/{key} && cp /w0/model/* /tmp/model/{key}/
ln -s /w?/shards/*.safetensors /tmp/model/{key}/
t env and model links ready
export PATH=/tmp/venv/bin:$PATH
cd /tmp/code/kvquant_matrix
# One load from the set, saved as fp16 on local disk and verified bit-identical
# (part5_cache_fp16.py); every cell then loads that copy. Both run under pod_guard.
python pod_guard.py --deadline {SET_SMOKE_DEADLINE} --heartbeat {out}/heartbeat.json -- \\
  bash -c "python part5_cache_fp16.py --src /tmp/model/{key} --dest /tmp/model16/{key} && \\
  python keys_run.py --grid part5 --root {out} --models {key} --arms all --gpu 0 \\
    --docs 4 --ppl-chunks 8 --lbroot /w0/lb/LongBench --datadir /w0/lb/data \\
    --model-dir /tmp/model16"
t work done
echo SET_SMOKE_DONE {key}
"""


# ---------------------------------------------------------------- registered runs (Tier A)
# docs/PART5_RUN_DESIGN.md. One pod per (model, wave) of part5_grid.WAVES; a model's waves run
# one after another on its set (RWO: one pod per set at a time), the three models in
# parallel. Shape from the 7B set-smoke's measurement (utilization guard, 2026-10-13):
# mean 0.89 cores and 4.64 GiB, peak 5.63 GiB; memory = peak x 1.25, so the mean sits at
# 66% of the request and the peak inside it; two cores keep the mean at 45%.
RUN_SHAPE = {"cpu": 2, "mem_gib": 7, "eph": "30Gi"}
# Hours per wave on Qwen2.5-7B, scaled from the smoke's per-document and per-chunk times
# (200 documents per task, 146 perplexity chunks) plus a pod's ~10 min of setup and fp16
# caching. The deadline is 1.5 x the estimate: a pod that reaches it exits, and because
# cells resume per document the next pod of the same wave continues where it stopped.
WAVE_HOURS = {
    "gate": 2.6, "g0": 3.4, "verdict": 2.7, "head": 3.8,
    "reported3": 2.2, "reported2": 5.6, "reported4": 5.6, "head2": 3.8, "head4": 3.8,
    # two u3-cost arms each, plus AATC's three calibration passes (~10 min)
    "baselines": 1.8, "fidelity": 1.8,
}
RUN_DIR = "part5"


SMOKE_LIMITS = "--docs 4 --ppl-chunks 8"  # a wave's wiring check: never scored


def run_root(commit: str, smoke: bool = False) -> str:
    return f"/w0/runs/{commit[:12]}/{'smoke' if smoke else RUN_DIR}"


def run_script(commit: str, key: str, wave: str, smoke: bool = False) -> str:
    """One wave of the registered grid on one Tier A model, from its linstor set: every
    document and every perplexity chunk, under pod_guard, whose telemetry is also kept in
    the run root beside the cells (pod_<wave>.log), so it survives the pod. ``smoke`` runs
    the wave's arms on 4 documents per task and 8 chunks in a separate root, which the
    scorer never reads (its cells record the limits)."""
    out = run_root(commit, smoke)
    arms = ",".join(P.WAVES[wave])
    deadline = int(WAVE_HOURS[wave] * 1.5 * 3600) if not smoke else 2 * 3600
    limits = f" {SMOKE_LIMITS}" if smoke else ""
    return f"""set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/w0/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HARNESS_COMMIT={commit}
export DIRECT_LOAD=1 KEYS_OOM_RETRIES=0  # loads straight to the GPU; an OOM never retries in a pod
t() {{ echo "[t] $(date -u +%T) $*"; }}
t start {key} wave={wave}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
tar -xf /w0/env.tar -C /tmp
mkdir -p /tmp/code && tar -xf {ROOT}/code/{commit}.tar -C /tmp/code
mkdir -p /tmp/model/{key} && cp /w0/model/* /tmp/model/{key}/
ln -s /w?/shards/*.safetensors /tmp/model/{key}/
mkdir -p {out}/{key}
t env and model links ready
export PATH=/tmp/venv/bin:$PATH
cd /tmp/code/kvquant_matrix
python pod_guard.py --deadline {deadline} --heartbeat {out}/{key}/heartbeat_{wave}.json -- \\
  bash -c "python part5_cache_fp16.py --src /tmp/model/{key} --dest /tmp/model16/{key} && \\
  python keys_run.py --grid part5 --root {out} --models {key} --arms {arms} --gpu 0{limits} \\
    --lbroot /w0/lb/LongBench --datadir /w0/lb/data --model-dir /tmp/model16" \\
  2>&1 | tee -a {out}/{key}/pod_{wave}.log
t work done
echo RUN_DONE {key} {wave}
"""


def set_busy(key: str) -> list:
    """Unfinished Jobs of this project on ``key``'s volume set. Its volumes are
    single-attach, and a same-name submit deletes the old Job first, so a second pod on
    a busy set is refused here, before anything is deleted."""
    import json as _json

    r = W.kubectl("get", "jobs", "-l", f"app={APP}", "-o", "json")
    if r.returncode != 0:
        raise SystemExit(f"cannot list jobs: {r.stderr.strip()}")
    busy = []
    for j in _json.loads(r.stdout)["items"]:
        name = j["metadata"]["name"]
        conds = {c["type"] for c in j.get("status", {}).get("conditions", []) if c["status"] == "True"}
        if name.endswith(f"-{short(key)}") or f"-{short(key)}-" in name:
            if not conds & {"Complete", "Failed"}:
                busy.append(name)
    return busy


def fetch_set_script(commit: str, key: str, what: str = "set-smoke") -> str:
    """A CPU job beside the set: its outputs as one base64 gzip tar on stdout. Volume
    0 is single-attach, so this runs after the GPU pod has finished."""
    return f"""set -euo pipefail
cd /w0/runs/{commit[:12]}/{what}/{key}
echo FETCH_BEGIN
timeout 600 tar -czf - . | base64 -w0
echo
echo FETCH_END
"""
GUARD_HEARTBEAT = os.path.join(STATE, "guard.hb")
# AATC's calibration text (part5_aatc_text.py) as staged on Atlas, 2026-10-13: 2,600 rows.
AATC_TEXT_SHA256 = "c1dc9d173135c1c700288282b1c04422908f9ea0129e05b159a964d7d4de3dbe"
LB_COMMIT = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"  # THUDM/LongBench, Part II's checkout
LB_SHA256 = {
    "data/trec.jsonl": "20a88e14f7442f58a39082cdd7396f1093ea8162da0a4ea7ead7d6757514c8f4",
    "data/triviaqa.jsonl": "ed2529de2e10b12c00f49981870c23c7c51667069cdd8d0740e1423ff337d7fa",
    "data/qasper.jsonl": "29aa07d2a63f36f4fb8e8cd200a3428ee3126d750bde6af1c8f9bc41c2366854",
    "LongBench/config/dataset2prompt.json": "56d22ad4f382169c2b8a11ff4c982a4a1bea096c8152b0f0b85b64686b157c30",
    "LongBench/config/dataset2maxlen.json": "72966b3c0933e214591637fb085798c5e687ebff4ddaab5d99bbc31120532022",
}
# GPU pod shape, the smoke's pilot model stated before it is measured: the harness loads
# the checkpoint to host memory first (low_cpu_mem_usage, then .cuda()), 3.1 GB at 1.5B,
# plus the runtime; one core drives the GPU. Registered models are sized from the smoke's
# measurement scaled by size (the prereg).
SMOKE_SHAPE = {"cpu": 1, "mem_gib": 6, "eph": "12Gi"}
SMOKE_DEADLINE = 7200  # seconds; about 30 arms of 6 documents and 4 chunks each
nl = chr(10)


def setup_script() -> str:
    return f"""set -euo pipefail
export PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1
if [ -f {ROOT}/env/env.tar ]; then echo "env exists"; exit 0; fi
# The 2.10 image's Python is the system 3.12 without ensurepip (kv5-setup failed on a
# plain venv, 2026-10-13): the venv borrows the image's pip and torch from the system
# site-packages, and pip installs into the venv (inside a venv it ignores the system's
# externally-managed marker).
python -m venv --without-pip --system-site-packages /tmp/venv
/tmp/venv/bin/python -m pip --version
/tmp/venv/bin/python -m pip install -q --no-cache-dir {PINS}
/tmp/venv/bin/python -c "import torch, transformers, datasets; print(torch.__version__, transformers.__version__, datasets.__version__)"
/tmp/venv/bin/python -m pip freeze > /tmp/venv/freeze.txt
mkdir -p {ROOT}/env && tar -cf {ROOT}/env/env.tar.tmp -C /tmp venv
mv {ROOT}/env/env.tar.tmp {ROOT}/env/env.tar
echo SETUP_DONE
"""


def code_script(commit: str) -> str:
    tar = f"{ROOT}/code/{commit}.tar"
    return f"""set -euo pipefail
if [ -f {tar} ]; then echo "already staged: {tar}"; exit 0; fi
git clone -q --filter=blob:none --no-checkout {REPO} /tmp/src
git -C /tmp/src sparse-checkout set --no-cone benchmarks/kvquant_matrix benchmarks/weight_observer
git -C /tmp/src checkout -q {commit}
test "$(git -C /tmp/src rev-parse HEAD)" = "{commit}"
mkdir -p {ROOT}/code
tar -cf {tar}.tmp.$$ -C /tmp/src/benchmarks kvquant_matrix weight_observer
mv {tar}.tmp.$$ {tar}
echo "staged {tar}"
"""


def _cpu_head(commit: str) -> str:
    return f"""set -euo pipefail
export PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1 HF_HOME={ROOT}/hf
tar -xf {ROOT}/env/env.tar -C /tmp
mkdir -p /tmp/code && tar -xf {ROOT}/code/{commit}.tar -C /tmp/code
export PYTHONPATH=/tmp/code
"""


def stage_script(commit: str, what: str) -> str:
    if what == "text":
        return _cpu_head(commit) + (
            '/tmp/venv/bin/python -c "from datasets import load_dataset; '
            "[load_dataset('wikitext', 'wikitext-2-raw-v1', split=s) for s in ('train', 'test')]; "
            "print('TEXT_DONE')\"" + nl
        )
    if what == "lb":
        checks = nl.join(
            f'echo "{h}  {ROOT}/lb/{p}" >> /tmp/lb.sha256' for p, h in LB_SHA256.items()
        )
        return _cpu_head(commit) + f"""mkdir -p /tmp/lb {ROOT}/lb/data {ROOT}/lb/LongBench/config
/tmp/venv/bin/python -c "from huggingface_hub import hf_hub_download as d; print(d('THUDM/LongBench', 'data.zip', repo_type='dataset', local_dir='/tmp/lb'))"
cd /tmp/lb && python -c "import zipfile; zipfile.ZipFile('data.zip').extractall('.')"
for t in trec triviaqa qasper; do cp /tmp/lb/data/$t.jsonl {ROOT}/lb/data/; done
# the image has no git: the config files at the pinned commit, by raw URL (hash-checked below)
for f in dataset2prompt dataset2maxlen; do
  python -c "import urllib.request as u; u.urlretrieve('https://raw.githubusercontent.com/THUDM/LongBench/{LB_COMMIT}/LongBench/config/$f.json', '{ROOT}/lb/LongBench/config/$f.json')"
done
: > /tmp/lb.sha256
{checks}
sha256sum -c /tmp/lb.sha256
echo LB_DONE
"""
    if what == "aatc-text":  # AATC's calibration text, hash-checked against Atlas's
        return _cpu_head(commit) + (
            "cd /tmp/code/kvquant_matrix && /tmp/venv/bin/python part5_aatc_text.py "
            f"--out {ROOT}/aatc_text.jsonl --expect {AATC_TEXT_SHA256}" + nl
        )
    if what in P.MODELS:
        return _cpu_head(commit) + (
            "/tmp/venv/bin/python -m weight_observer.stage_models "
            f"--model-id {P.MODELS[what]} --dest {ROOT}/models/{what}" + nl
        )
    raise SystemExit(f"unknown stage target {what!r}")


def smoke_script(commit: str, key: str) -> str:
    """The wiring smoke (prereg section 6): every harness arm on documents 0, 50, 100 and
    150 of each task and 8 perplexity chunks (more than the prereg's 2 and 4, so a cell keeps
    the A10 busy past its process start). Nothing is compared from it."""
    out = f"{ROOT}/runs/{commit[:12]}/smoke"
    return f"""set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME={ROOT}/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HARNESS_COMMIT={commit}
t() {{ echo "[t] $(date -u +%T) $*"; }}
t start
mkdir -p /tmp/code /tmp/model && tar -xf {ROOT}/code/{commit}.tar -C /tmp/code
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cp -r {ROOT}/models/{key} /tmp/model/{key} &
cp_model=$!
tar -xf {ROOT}/env/env.tar -C /tmp
t env unpacked
wait $cp_model
t checkpoint copied
export PATH=/tmp/venv/bin:$PATH
cd /tmp/code/kvquant_matrix
# pod_guard: telemetry every 30 s to the log and {out}/heartbeat.json, a hard deadline,
# and the pod ends itself if its GPU idles after warm-up (nothing outside deletes it).
python pod_guard.py --deadline {SMOKE_DEADLINE} --heartbeat {out}/heartbeat.json -- \\
  python keys_run.py --grid part5 --root {out} --models {key} --arms all --gpu 0 \\
    --docs 4 --ppl-chunks 8 --lbroot {ROOT}/lb/LongBench --datadir {ROOT}/lb/data \\
    --model-dir /tmp/model
t work done
echo SMOKE_DONE {key}
"""


def fetch_script(commit: str, key: str, what: str) -> str:
    """A run directory as one base64 gzip tar on stdout, read back with ``kubectl logs``."""
    if what != "smoke":
        raise SystemExit(f"unknown output {what!r}")
    src = f"{ROOT}/runs/{commit[:12]}/smoke/{key}"
    return f"""set -euo pipefail
cd {src}
echo FETCH_BEGIN
tar -czf - . | base64 -w0
echo
echo FETCH_END
"""


def descriptor(name, script, cpu, mem_gib, eph, role, gpu=0, image=IMAGE, extra=(),
               zone=None, shared=True):
    """``extra``: (pvc, mount path) pairs beside the shared volume (``shared``) at /data;
    ``zone``: a node selector pinning the pod beside its volumes (a linstor set)."""
    from nats_bursting import JobDescriptor, Resources, Volume

    vols = [Volume(name="data", mount_path="/data", claim_name=PVC)] if shared else []
    vols += [Volume(name=f"w{i}", mount_path=m, claim_name=c) for i, (c, m) in enumerate(extra)]
    sel = dict(zone) if zone else ({} if gpu else dict(W.CPU_ZONE))
    if gpu:
        sel["nvidia.com/gpu.product"] = GPU_PRODUCT
    return JobDescriptor(
        name=name,
        image=image,
        command=["/bin/bash", "-lc", script],
        resources=Resources(cpu=str(cpu), memory=f"{mem_gib}Gi", gpu=gpu, ephemeral_storage=eph),
        labels={"app": APP, "atlas.io/batch": BATCH, "atlas.io/role": role},
        node_selector=sel,
        backoff_limit=0,
        volumes=vols,
    )


def short(key: str) -> str:
    return key.replace(".", "").replace("-instruct", "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cmd",
        choices=(
            "setup", "code", "stage", "smoke", "fetch", "volumes", "stage-set",
            "set-smoke", "fetch-set", "set-run",
        ),
    )
    ap.add_argument("--apply", action="store_true", help="volumes: kubectl apply the PVCs")
    ap.add_argument("--commit", default="")
    ap.add_argument("--what", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--wave", default="", help="set-run: a wave of part5_grid.WAVES")
    ap.add_argument("--smoke", action="store_true", help="set-run: 4 documents, 8 chunks")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "volumes":  # PVCs come from kubectl apply; only Jobs go through nats-bursting
        keys = [a.model] if a.model else list(SHARDS_GB)
        yaml = "".join(pvc_yaml(k) for k in keys)
        print(yaml)
        if a.apply:
            r = subprocess.run(["kubectl", "-n", NS, "--request-timeout=60s", "apply", "-f", "-"],
                               input=yaml, capture_output=True, text=True, timeout=180)
            print(r.stdout, r.stderr)
            return r.returncode
        return 0
    if a.cmd not in ("setup",) and len(a.commit) != 40:
        raise SystemExit("--commit takes the full 40-character SHA")
    gpu = a.cmd in ("smoke", "set-smoke", "set-run")
    if a.cmd == "setup":
        desc = descriptor("kv5-setup", setup_script(), 1, 2, "10Gi", "setup")
    elif a.cmd == "code":
        desc = descriptor(
            f"kv5-code-{a.commit[:8]}", code_script(a.commit), 1, 2, "2Gi", "code", image=CODE_IMAGE
        )
    elif a.cmd == "stage":
        name = f"kv5-stage-{short(a.what)}"
        desc = descriptor(name, stage_script(a.commit, a.what), 1, 2, "20Gi", "stage")
    elif a.cmd == "smoke":
        if a.model not in P.MODELS:
            raise SystemExit(f"unknown model {a.model!r}")
        if not W.guard_alive(GUARD_HEARTBEAT):
            raise SystemExit(f"the utilization guard is not beating ({GUARD_HEARTBEAT})")
        s = SMOKE_SHAPE
        desc = descriptor(
            f"kv5-smoke-{short(a.model)}",
            smoke_script(a.commit, a.model),
            s["cpu"], s["mem_gib"], s["eph"], "smoke", gpu=1,
        )
    elif a.cmd in ("set-smoke", "fetch-set", "set-run"):
        if a.model not in SHARDS_GB:
            raise SystemExit(f"no volume set for {a.model!r}")
        vols = [(n, m) for n, _, m in set_volumes(a.model)]
        if not a.dry_run and set_busy(a.model):
            raise SystemExit(f"{a.model}'s set is in use by {set_busy(a.model)}")
        if a.cmd == "set-run":
            if a.wave not in P.WAVES:
                raise SystemExit(f"unknown wave {a.wave!r}; one of {list(P.WAVES)}")
            if not W.guard_alive(GUARD_HEARTBEAT):
                raise SystemExit(f"the utilization guard is not beating ({GUARD_HEARTBEAT})")
            s = RUN_SHAPE
            if a.wave == "fidelity" and a.model != P.FIDELITY_MODEL:
                raise SystemExit(f"the fidelity check runs on {P.FIDELITY_MODEL} only")
            kind = "smoke" if a.smoke else "run"
            desc = descriptor(
                f"kv5-{kind}-{a.wave}-{short(a.model)}",
                run_script(a.commit, a.model, a.wave, a.smoke),
                s["cpu"], s["mem_gib"], s["eph"], "run", gpu=1, extra=vols, zone=UNL,
            )
        elif a.cmd == "set-smoke":
            if not W.guard_alive(GUARD_HEARTBEAT):
                raise SystemExit(f"the utilization guard is not beating ({GUARD_HEARTBEAT})")
            s = SET_SHAPE
            desc = descriptor(
                f"kv5-setsmoke-{short(a.model)}", set_smoke_script(a.commit, a.model),
                s["cpu"], s["mem_gib"], s["eph"], "smoke", gpu=1, extra=vols, zone=UNL,
            )
        else:
            what = {"run": RUN_DIR, "smoke": "smoke"}.get(a.what, "set-smoke")
            desc = descriptor(
                f"kv5-fetchset-{short(a.model)}", fetch_set_script(a.commit, a.model, what),
                1, 2, "2Gi", "fetch", extra=vols[:1], zone=UNL, shared=False,
            )
    elif a.cmd == "stage-set":
        if a.model not in SHARDS_GB:
            raise SystemExit(f"no volume set for {a.model!r}")
        desc = descriptor(
            f"kv5-stageset-{short(a.model)}", stage_set_script(a.commit, a.model), 1, 2, "24Gi",
            "stage", extra=[(n, m) for n, _, m in set_volumes(a.model)], zone=UNL,
        )
    else:
        desc = descriptor(
            f"kv5-fetch-{short(a.model)}", fetch_script(a.commit, a.model, a.what), 1, 2, "2Gi", "fetch"
        )
    bad = W.preflight(desc, gpu)
    if bad:
        raise SystemExit(f"preflight failed: {bad}")
    print(f"{desc.name}: {desc.resources} on {desc.node_selector} (preflight clean)")
    if a.dry_run:
        print(desc.command[-1])
        return 0
    W.submit(desc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
