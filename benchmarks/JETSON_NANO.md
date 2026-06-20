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

## For end-to-end model tok/s + J/tok (optional, stronger)

The microbenchmark measures the KV path, not a full forward pass. For a real
generated-token rate and energy, run an actual small model and wrap it in the
same `PowerSampler`:

- Easiest on this board: **llama.cpp** (compiles on aarch64, runs a 1B/3B GGUF on
  CPU). Start it, and in parallel capture `tegrastats` over the generation window;
  divide energy by tokens generated.
- Or import `PowerSampler` from `benchmark_edge.py` and wrap a `transformers`
  `model.generate(...)` loop.

Report which quantizer produced the running model (TurboQuant-exported weights
vs. a GGUF baseline) so the J/tok claim is unambiguous.
