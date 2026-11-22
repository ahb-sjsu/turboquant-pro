# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""The Mistral-7B and Gemma-3-4B parts of E1 to E4 on NRP, through nats-bursting.

Runs on Atlas (kubectl and the nats-bursting controller are there). One volume set of
four linstor volumes in zone unl holds everything; the code and the inputs made on
Atlas (prep.py) ride in ConfigMaps. Every Job is submitted through nrp-kit, whose
preflight checks it first.

    python driver.py volumes [--apply]     # the volume set (kubectl apply)
    python driver.py configmaps [--apply]  # code, stripe plans, tokens
    python driver.py submit stage|gguf|rest|e1 [--dry-run]
    python driver.py fetch OUT_DIR         # /w0/results back to Atlas

The Jobs, in order (the volumes are single-attach, so one at a time):

  stage  CPU, small class. The GPU pods' environment (a venv on the GPU image), then
         Mistral-7B-v0.1 and Gemma-3-4B-it from the Hub, hash-checked, striped, and
         checked against Atlas's reference stripes.
  gguf   CPU, small class. llama.cpp at Atlas's commit, Mistral's f16 GGUF and the four
         quantized ones, then E4's weights phase (llamacpp_factorial.py).
  rest   GPU (A10), small class: E2 (row protection), E4's functional phase and E3 on
         Mistral, then E3 on Gemma. Model, copies and base log-probabilities stay on
         the GPU, so the host holds only the runtime.
  e1     GPU (A10), 1 CPU and 16 GiB: E1 on Mistral. Each of its 21 conditions starts
         from a full copy of the linear weights, which the A10 has no room for beside
         the model, so the copy is pinned in host memory (13.0 GiB).
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)  # the experiments directory
WORK = "/archive/experiments/tqp-tmlr-operator/nrp/work"
NRP_KIT = "/home/claude/src/nrp-kit"
sys.path[:0] = [NRP_KIT, "/home/claude/src/nats-bursting/python"]

from nrp_kit import config, kube, podscript, submit  # noqa: E402
from nrp_kit.volumes import VolumeSet  # noqa: E402

APP = "tqp-tmlr"
BATCH = "tqp-tmlr-operator"
ZONE = {"topology.kubernetes.io/zone": "unl"}
A10 = {**ZONE, "nvidia.com/gpu.product": "NVIDIA-A10"}
GPU_IMAGE = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime"  # torch 2.10.0, as Atlas
BUILD_IMAGE = "python:3.12"  # git and a C++ compiler, for llama.cpp
FETCH_IMAGE = "python:3.12-slim"
# Atlas's environment (/home/claude/env, 2026-11-22), torch from the image
PINS = (
    "transformers==5.5.0 accelerate==1.13.0 sentencepiece==0.2.1 tokenizers==0.22.2 "
    "huggingface_hub==1.11.0 safetensors==0.7.0 numpy==2.2.6 protobuf"
)
LLAMA_CPP = "a1cfb645307edc61a89e41557f290f441043d3c2"  # b8639, Atlas's build
# Volume 0 also holds both environments, llama.cpp and the five GGUF files (~26.5 GB).
SET = VolumeSet(
    prefix="tmlr-w",
    n=4,
    gib0=46,
    gibk=10,
    storage_class="linstor-unl",
    zone="unl",
    app=APP,
)
VOLS = ",".join(f"/w{k}" for k in range(SET.n))
CODE_FILES = [
    os.path.join(CODE, f)
    for f in (
        "exp_io.py",
        "agreement_reference_repeats.py",
        "k_row_protection_controls.py",
        "score_error_decomposition.py",
        "llamacpp_factorial.py",
    )
] + [
    os.path.join(HERE, f) for f in ("loader.py", "stripe.py", "prep.py", "stage_pod.py")
]
CODE_FILES.append(os.path.join(NRP_KIT, "nrp_kit", "load_model.py"))
MISTRAL = {
    "NB_MODEL": "mistralai/Mistral-7B-v0.1",
    "NB_MODEL_DIR": "/w0/mistral/model",
    "NB_STRIPES": "/w?/mistral/s/*.safetensors",
    "NB_TOKENS": "/tokens/tokens_mistral.json",
    "NB_OUT": "/w0/results",
}
GEMMA = {
    "NB_MODEL": "unsloth/gemma-3-4b-it",
    "NB_MODEL_DIR": "/w0/gemma/model",
    "NB_STRIPES": "/w?/gemma/s/*.safetensors",
    "NB_TOKENS": "/tokens/tokens_gemma.json",
    "NB_OUT": "/w0/results",
    "NB_DTYPE": "bfloat16",
}
E4 = {
    "NB_TAG": "Mistral-7B-v0.1",
    "NB_ARM_DIRS": ",".join(f"/w{k}/e4" for k in range(SET.n)),
    "NB_WEIGHTS": "/w0/results/e4_weights_Mistral-7B-v0.1.json",
}
PREAMBLE = """export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NB_THREADS=1
t() { echo "[t] $(date -u +%T) $*"; }
t start
"""


def env(d: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in d.items())


def stage_script() -> list[str]:
    return podscript.script(PREAMBLE + f"""df -h {VOLS.replace(",", " ")}
export HF_HOME=/tmp/hf HF_HUB_DISABLE_XET=1
if [ ! -f /w0/env/READY ]; then
  if [ -e /w0/env ]; then mv /w0/env /w0/env.stale.$(date +%s); fi
  python -m venv --without-pip --system-site-packages /w0/env/venv
  timeout 1800 /w0/env/venv/bin/python -m pip install -q --no-cache-dir {PINS}
  /w0/env/venv/bin/python -m pip freeze > /w0/env/freeze.txt
  /w0/env/venv/bin/python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
  touch /w0/env/READY
fi
t environment ready
cd /code
timeout 7200 /w0/env/venv/bin/python stage_pod.py --plan /plans/plan_mistral.json --root mistral --vols {VOLS}
t mistral staged
timeout 7200 /w0/env/venv/bin/python stage_pod.py --plan /plans/plan_gemma.json --root gemma --vols {VOLS}
t gemma staged
df -h {VOLS.replace(",", " ")}
echo STAGE_DONE
""")  # noqa: E501


def gguf_script() -> list[str]:
    tag = E4["NB_TAG"]
    g = "/w0/gguf"
    quant = "\n".join(
        f"""if [ ! -s {g}/{tag}-{lv}{s}.gguf ]; then
  timeout 5400 $LQ {flag}{g}/{tag}-f16.gguf {g}/{tag}-{lv}{s}.gguf.part {lv} 1
  mv {g}/{tag}-{lv}{s}.gguf.part {g}/{tag}-{lv}{s}.gguf
fi
t {lv}{s} done"""
        for lv in ("Q2_K", "Q3_K_M")
        for s, flag in (("", ""), ("-pure", "--pure "))
    )
    return podscript.script(PREAMBLE + f"""L=/w0/llama.cpp
LQ=$L/build/bin/llama-quantize
if [ ! -x $LQ ]; then
  [ -d $L/.git ] || timeout 1800 git clone -q --filter=blob:none https://github.com/ggml-org/llama.cpp $L
  git -C $L checkout -q {LLAMA_CPP}
  git -C $L log -1 --format='llama.cpp %H %cd'
  if [ ! -f /w0/gvenv/READY ]; then
    if [ -e /w0/gvenv ]; then mv /w0/gvenv /w0/gvenv.stale.$(date +%s); fi
    python -m venv /w0/gvenv
    timeout 1800 /w0/gvenv/bin/pip install -q --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
    timeout 1800 /w0/gvenv/bin/pip install -q --no-cache-dir numpy==2.2.6 transformers==5.5.0 sentencepiece==0.2.1 tokenizers==0.22.2 safetensors==0.7.0 huggingface_hub==1.11.0 protobuf tqdm pyyaml requests cmake
    /w0/gvenv/bin/pip freeze > /w0/gvenv/freeze.txt
    touch /w0/gvenv/READY
  fi
  timeout 1800 /w0/gvenv/bin/cmake -S $L -B $L/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
  timeout 7200 /w0/gvenv/bin/cmake --build $L/build --target llama-quantize -j 1
fi
t llama-quantize built
mkdir -p {g} /tmp/mhf
ln -sf /w0/mistral/model/* /tmp/mhf/
ln -sf /w?/mistral/s/*.safetensors /tmp/mhf/
if [ ! -s {g}/{tag}-f16.gguf ]; then
  timeout 7200 /w0/gvenv/bin/python $L/convert_hf_to_gguf.py /tmp/mhf --outtype f16 --outfile {g}/{tag}-f16.gguf.part
  mv {g}/{tag}-f16.gguf.part {g}/{tag}-f16.gguf
fi
t f16 converted
{quant}
ls -la {g}
cd /code
if [ ! -s {E4["NB_WEIGHTS"]} ]; then
  {env(MISTRAL)} {env(E4)} NB_PHASE=weights NB_BASE='{MISTRAL["NB_STRIPES"]}' NB_GGUF_DIR={g} NB_LLAMA_CPP=$L timeout 10800 /w0/gvenv/bin/python llamacpp_factorial.py
fi
t weights phase done
(cd {g} && timeout 3600 sha256sum *.gguf | tee /w0/results/gguf_sha256_{tag}.txt)
echo GGUF_DONE
""")  # noqa: E501


def gpu_preamble() -> str:
    return PREAMBLE + """export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/w0/env/venv/bin/python
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
cd /code
"""


def rest_script() -> list[str]:
    r = "/w0/results"
    return podscript.script(
        gpu_preamble() + f"""if [ ! -s {r}/k_row_protection_Mistral-7B-v0.1.json ]; then
  {env(MISTRAL)} NB_RESIDENT=1 timeout 3600 $PY k_row_protection_controls.py
fi
t E2 done
if [ ! -s {r}/llamacpp_factorial_Mistral-7B-v0.1.json ]; then
  {env(MISTRAL)} {env(E4)} NB_PHASE=functional NB_RESIDENT=1 timeout 3600 $PY llamacpp_factorial.py
fi
t E4 functional done
if [ ! -s {r}/score_error_Mistral-7B-v0.1.json ]; then
  {env(MISTRAL)} NB_SAMPLES=8 timeout 1800 $PY score_error_decomposition.py
fi
t E3 mistral done
if [ ! -s {r}/score_error_gemma-3-4b-it.json ]; then
  {env(GEMMA)} NB_SAMPLES=8 timeout 1800 $PY score_error_decomposition.py
fi
t E3 gemma done
echo REST_DONE
"""  # noqa: E501
    )


def e1_script() -> list[str]:
    r = "/w0/results"
    return podscript.script(
        gpu_preamble()
        + f"""if [ ! -s {r}/agreement_repeats_Mistral-7B-v0.1.json ]; then
  {env(MISTRAL)} NB_SAMPLES=48 timeout 3600 $PY agreement_reference_repeats.py
fi
t E1 done
echo E1_DONE
"""
    )


def volumes():
    from nats_bursting import Volume

    v = [
        Volume(name=f"w{k}", mount_path=m, claim_name=c)
        for k, (c, m) in enumerate(SET.mounts())
    ]
    v += [
        Volume(name="code", mount_path="/code", config_map="tmlr-code", read_only=True),
        Volume(
            name="plans", mount_path="/plans", config_map="tmlr-plans", read_only=True
        ),
        Volume(
            name="tokens",
            mount_path="/tokens",
            config_map="tmlr-tokens",
            read_only=True,
        ),
    ]
    return v


SHAPES = {  # name: (image, cpu, memory, gpu, ephemeral, script, node selector)
    "stage": (GPU_IMAGE, "1", "2Gi", 0, "14Gi", stage_script, ZONE),
    "gguf": (BUILD_IMAGE, "1", "2Gi", 0, "6Gi", gguf_script, ZONE),
    "rest": (GPU_IMAGE, "1", "2Gi", 1, "4Gi", rest_script, A10),
    "e1": (GPU_IMAGE, "1", "16Gi", 1, "4Gi", e1_script, A10),
}
PILOT = {
    "rest": (
        "first pod of its class; nrp-kit's measured LLM GPU shape (1 CPU, 2 GiB: "
        "load_model of a 7B, host peak 1.55 GiB); model, weight copies and base "
        "log-probabilities all stay on the GPU"
    ),
    "e1": (
        "first pod of its class; 16 GiB = a pinned 13.0 GiB copy of Mistral-7B's "
        "linear weights (6.98B fp16) + ~1.7 GiB runtime, reached seconds after the "
        "load, so steady use is ~92% of the request; 1 CPU drives the GPU"
    ),
}


def descriptor(job: str):
    from nats_bursting import JobDescriptor, Resources

    image, cpu, mem, gpu, eph, script, sel = SHAPES[job]
    return JobDescriptor(
        name=f"tmlr-{job}",
        image=image,
        command=script(),
        resources=Resources(cpu=cpu, memory=mem, gpu=gpu, ephemeral_storage=eph),
        labels={"app": APP, "atlas.io/batch": BATCH},
        node_selector=dict(sel),
        backoff_limit=0,
        volumes=volumes(),
    )


def configmap(name: str, files: list[str], apply: bool) -> None:
    args = ["create", "configmap", name] + [f"--from-file={f}" for f in files]
    site = config.load()
    r = kube.kubectl(site, *args, "--dry-run=client", "-o", "yaml")
    if r.returncode:
        raise SystemExit(r.stderr)
    size = sum(os.path.getsize(f) for f in files)
    print(f"{name}: {len(files)} files, {size / 1024:.0f} KiB")
    if size > 900 * 1024:
        raise SystemExit(f"{name} is too large for a ConfigMap")
    if apply:
        r = kube.kubectl(
            site,
            "apply",
            "--server-side",
            "--force-conflicts",
            "-f",
            "-",
            stdin=r.stdout,
        )
        print(r.stdout.strip() or r.stderr.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("volumes")
    v.add_argument("--apply", action="store_true")
    c = sub.add_parser("configmaps")
    c.add_argument("--apply", action="store_true")
    s = sub.add_parser("submit")
    s.add_argument("job", choices=sorted(SHAPES))
    s.add_argument("--dry-run", action="store_true")
    f = sub.add_parser("fetch")
    f.add_argument("out")
    a = ap.parse_args()
    site = config.load()
    if a.cmd == "volumes":
        print(SET.manifest(site))
        if a.apply:
            print(SET.apply(site))
    elif a.cmd == "configmaps":
        configmap("tmlr-code", CODE_FILES, a.apply)
        configmap(
            "tmlr-plans",
            [os.path.join(WORK, f"plan_{k}.json") for k in ("mistral", "gemma")],
            a.apply,
        )
        configmap(
            "tmlr-tokens",
            [os.path.join(WORK, f"tokens_{k}.json") for k in ("mistral", "gemma")],
            a.apply,
        )
    elif a.cmd == "submit":
        d = descriptor(a.job)
        kw = {"pilot_reason": PILOT[a.job]} if a.job in PILOT else {}
        if a.dry_run:
            from nrp_kit import preflight

            d.labels = {**site.labels, **d.labels}
            gpus = kube.gpu_pods(site) if d.resources.gpu else None
            bad = preflight.check(site, d, gpus_in_use=gpus, **kw)
            print("PREFLIGHT", "ok" if not bad else bad)
            print("labels", d.labels, "selector", d.node_selector)
            print(d.command[2])
            print("command bytes", len(d.command[2].encode()))
        else:
            submit.submit(site, d, **kw)
    elif a.cmd == "fetch":
        from nats_bursting import JobDescriptor, Resources, Volume
        from nrp_kit.fetch import fetch

        def make(name, command):
            return JobDescriptor(
                name=name,
                image=FETCH_IMAGE,
                command=command,
                resources=Resources(cpu="1", memory="1Gi", ephemeral_storage="2Gi"),
                labels={"app": APP, "atlas.io/batch": BATCH},
                node_selector=dict(ZONE),
                backoff_limit=0,
                volumes=[
                    Volume(name="w0", mount_path="/w0", claim_name=SET.names()[0])
                ],
            )

        print(fetch(site, make, "/w0/results", "tmlr-results", a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
