# `tqp` — the turboquant-pro CLI

`tqp` surfaces the library's instruments behind one command. It ships with the
package (console script) and adds no dependencies — the interactive subcommands
are pure stdlib + numpy; `tqp trace` additionally needs `[torch]` + transformers.

> **Status:** Phase 1 of [`docs/turboquant_pro_next_level_roadmap.md`](turboquant_pro_next_level_roadmap.md).
> Implemented today: `version`, `plugin list`, `plugin conformance`, `trace`,
> `probe`, `monitor`. `plan` / `certify` / `replay` are declared but stubbed —
> they print their roadmap phase and exit 2, so the surface is visible without
> overclaiming.

## Install

```bash
pip install turboquant-pro          # gives you `tqp`
pip install 'turboquant-pro[torch]' transformers   # additionally enables `tqp trace`
```

## Commands

### `tqp version`
Prints the installed version.

### `tqp plugin list [--target T] [-v]`
Lists registered quantizer plugins (in-tree **and** entry-point plugins), with
tier and targets. `--target` filters (`weight` / `kv_key` / `kv_value` /
`embedding`); `-v` shows descriptions.

```
NAME         TIER    TARGETS
per_channel  beta    kv_key
polar        stable  kv_value
```

### `tqp plugin conformance [names...] [--target T] [--heads H --seq S --dim D | --shape a,b,...]`
Runs the container-contract conformance kit (`run_conformance`) on the given
plugins (default: all registered), reporting `pass` / `skip` / `FAIL` per check.
The default sample is a canonical KV block `(1, H, S, D)` with a per-head DC
offset — matched to the in-tree KV plugins — and the quantizer is built with
`head_dim=D, n_heads=H` to match it. Use `--shape` for non-KV plugins. Exit code
is non-zero if any plugin fails.

```bash
tqp plugin conformance                 # all plugins, default KV block
tqp plugin conformance per_channel     # one plugin
tqp plugin conformance --shape 512,128 # custom sample shape
```

### `tqp trace <hf-model> [--target weight|kv_activation] [--prefer auto|structural|fx] [-v] [--trust-remote-code]`
Traces a Hugging Face model's operator regimes and maps each tensor to its (A2)
quantization discipline. The architecture is built on the **meta device** (real
module structure and names, zero materialized weights), so tracing even a 7B
model costs no download or RAM. Prints regime and discipline-family
distributions; `-v` adds the per-tensor table.

```bash
tqp trace meta-llama/Llama-3.2-1B --target kv_activation
```
```
# meta-llama/Llama-3.2-1B  (target=kv_activation, prefer=auto, fx-traced=..., tensors=...)
regime distribution:
  linear_residual  ...
  softmax_score    ...
(A2) discipline family distribution:
  per_channel      ...
  symmetric        ...
```

### `tqp probe [--npy PATH | --demo {isotropic,dc_offset}] [--consumer cosine|l2|attention_logits] [--bits N] [--queries PATH] [--seed N] [--json]`
The **(A2) consumer-metric probe** (`a2_probe.probe_quotient`): given a sample
batch of the vectors you intend to quantize, it applies the polar (per-vector
norm + direction) and per-channel (affine) family proxies at a matched bit
budget and reports which one preserves the *declared consumer's* ranking
(Spearman agreement of cosine / L2 / attention-logit scores). This is the check
that catches the v1.2.0 KV-keys class at calibration time, where reconstruction
cosine looks fine but attention-logit ranking collapses. Input is a `.npy`
array `(n, d)` (arrays with more axes are flattened to `(-1, d)` — rows are
last-axis vectors, the KV convention). `--demo` substitutes a **labeled
synthetic** batch for a quick look. Exit 0 on success; 2 on a usage/data error.

```bash
tqp probe --npy keys.npy --consumer attention_logits   # attention keys
tqp probe --demo dc_offset --consumer attention_logits --json
```
```
consumer=attention_logits  bits=4
  spearman(polar)        = 0.9911
  spearman(per_channel)  = 0.9924
=> recommend: per_channel
```
It selects a family at calibration time — validate the shipped path end-to-end,
and pair with `tqp monitor` for radial drift in production.

### `tqp monitor --original PATH --reconstructed PATH [--floor F] [--window N] [--format json|prometheus|text]`
Feeds original/reconstructed `.npy` pairs through
`monitor.QualityMonitor.record_batch` and emits `metrics_dict()` — mean/min/p95
cosine, drift flags, the (A2) tangential fraction — as JSON, Prometheus
text-exposition, or a human table. **Exit code is a gate:** 0 when mean cosine
is at or above `--floor`, 1 when it falls below, 2 on a load/shape error.

```bash
tqp monitor --original o.npy --reconstructed r.npy --format prometheus
```
```
# TYPE turboquant_quality_mean_cosine gauge
turboquant_quality_mean_cosine 0.9999
# TYPE turboquant_quality_is_healthy gauge
turboquant_quality_is_healthy 1
```

## Design notes
- **No new runtime dep.** `tqp` is `argparse`; the core install stays numpy-only.
- **Honest surface.** Unbuilt subcommands are visible but exit 2 with a roadmap
  pointer — never a silent no-op or a fake success.
- **`turboquant-pro`** remains the existing AutoConfig entry point; `tqp` is the
  new unified surface.
