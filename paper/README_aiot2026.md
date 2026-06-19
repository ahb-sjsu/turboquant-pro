# TurboQuant @ Edge — IEEE AIoT 2026 paper

`turboquant_aiot2026.tex` — IEEE conference (`IEEEtran`) skeleton for the
Track-3 (edge LLM deployment) submission. Deadline: **Aug 1, 2026**.

## Status
- Established numbers (27× embedding @ 99.8% recall@10; −22% codebook error;
  5.3× KV @ 3-bit; the 8 GB fit result) and the analytical memory budget are in.
- Cells marked `\tbd{}` (red) are **on-device measurements still to fill** —
  decode throughput and energy-per-token per device.

## Fill the device results
Run on each target (Jetson Orin Nano/NX, Raspberry Pi 5, a consumer GPU):
```bash
python benchmarks/benchmark_edge.py --models llama-3.2-1b llama-3.2-3b qwen2.5-7b \
    --contexts 2048 8192 32768 --bits 3 --gen 128 --energy --json out/edge_<device>.json
```
Then transcribe tok/s and J/tok into Table III (`\label{tab:device}`).

## Build
```bash
pdflatex turboquant_aiot2026.tex && pdflatex turboquant_aiot2026.tex
```
