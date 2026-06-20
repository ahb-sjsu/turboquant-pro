# Running `benchmark_edge.py` on a Jetson Nano (original / B01, 4 GB)

This is the **original Maxwell Jetson Nano** (Tegra X1, compute 5.3, 4 GB shared
LPDDR4, ~10 W) — e.g. the Waveshare B01 eMMC kit — **not** an Orin Nano. That
distinction drives everything below.

## What works, and what doesn't (be honest in the paper)

| Aspect | Reality on this board |
|---|---|
| OS / Python | JetPack 4.6.x → Ubuntu 18.04, **Python 3.6**. `turboquant-pro` needs **3.9+** → use miniforge (below). |
| GPU path | Maxwell is **below** the repo's "Volta+" CUDA path → TurboQuant runs the **CPU (numpy) path**. That's fine; it still measures real ARM throughput + board energy. |
| Energy | **No NVML on Tegra.** `PowerSampler` auto-uses `tegrastats` (the `POM_5V_IN` rail). Real J/tok. |
| Storage | eMMC is only **16 GB** and JetPack already fills most of it. Put miniforge + models on a **USB SSD/stick**. |
| Speed | Low tok/s (10 W, CPU). That's the *point*: a 4 GB / 10 W device hosting a model **only because TQ shrinks it** is the AIoT story. |

## Setup

```bash
# 1. Max performance + stable clocks (so power/throughput are repeatable)
sudo nvpmodel -m 0        # 10 W MAXN
sudo jetson_clocks

# 2. miniforge (aarch64) for Python 3.10 -- install onto external storage if eMMC is full
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b -p /mnt/usb/miniforge   # or ~/miniforge
source /mnt/usb/miniforge/etc/profile.d/conda.sh
conda create -y -n edge python=3.10 numpy && conda activate edge

# 3. Get the code (run from source -- avoids pulling heavy optional deps on the Nano)
git clone https://github.com/ahb-sjsu/turboquant-pro && cd turboquant-pro
```

## Measure (fills Table III's Nano rows)

```bash
# tegrastats backend is auto-selected; --energy now yields real J/tok
PYTHONPATH=. python benchmarks/benchmark_edge.py \
    --models llama-3.2-1b llama-3.2-3b --contexts 2048 8192 \
    --bits 3 --gen 128 --energy --json out/edge_jetson_nano.json
```

This reports the KV-cache compression budget, KV-path throughput, the 4 GB
device-fit result, and **measured board power** (avg W → J for the workload).
`PowerSampler.backend` will read `tegrastats`.

## End-to-end tok/s + J/tok (fills Table III) — `benchmark_e2e.py`

`benchmark_edge.py` measures the KV path; `benchmark_e2e.py` runs a **real
generation** and reports decode tok/s, J/tok (tegrastats-measured), and peak
memory — one Table III row per run.

```bash
pip install llama-cpp-python        # CPU build; compiles on aarch64

# Get a small GGUF onto external storage (example: a 1B at ~3-bit)
#   e.g. llama-3.2-1b Q3_K  ->  ./models/llama-3.2-1b-q3_k.gguf

PYTHONPATH=. python -m benchmarks.benchmark_e2e \
    --backend llama-cpp --model ./models/llama-3.2-1b-q3_k.gguf \
    --quantizer gguf-q3_k --device "Jetson Nano 4GB" \
    --threads 4 --n-tokens 128 --energy --json out/e2e_nano_1b.json
```

It prints a paste-ready `Device & Model & quantizer & tok/s & J/tok` line. Expect
**low tok/s** (10 W CPU) — that's the AIoT point, a model that only fits in 4 GB
because it's compressed.

**Be honest about provenance:** `--quantizer gguf-q3_k` is a GGUF *baseline*, not
TurboQuant's own quantizer. If/when you export TurboQuant-quantized weights into a
runnable format, label that run `--quantizer turboquant-3bit` so the J/tok claim
is unambiguous. The paper must state which produced the running weights.
