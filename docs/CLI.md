# `tqp` — the turboquant-pro CLI

`tqp` surfaces the library's instruments behind one command. It ships with the
package (console script) and adds no dependencies — the interactive subcommands
are pure stdlib + numpy; `tqp trace` additionally needs `[torch]` + transformers.

> **Status:** Phases 1–4 of [`docs/turboquant_pro_next_level_roadmap.md`](turboquant_pro_next_level_roadmap.md).
> The full pipeline `trace → plan → compress → certify → replay → monitor` is
> live: `version`, `plugin list`/`conformance`, `trace`, `probe`, `plan`,
> `certify`, `verify`, `replay`, `monitor`, `index`. No stubs remain.
>
> **Coherence rule.** Every command's acceptance signal is **rank fidelity /
> the (A2) consumer metric / a distribution-free certificate** — never
> reconstruction cosine on its own. Cosine appears only as a *labelled secondary
> diagnostic*, and where it is the base signal (`monitor`) it is guarded by the
> (A2) tangential-fraction / radial-drift statistics, because cosine can read
> ~0.97 while the ranking the consumer actually uses collapses
> ([`docs/KV_KEYS_FINDING.md`](KV_KEYS_FINDING.md)).

## Install

> `tqp` ships in **1.8.0**; the current PyPI release is **1.9.0** (larger-than-RAM
> sharded / memory-mapped search + index format v3) — `pip install turboquant-pro`
> gives you the console script. Add `[torch]` for `tqp trace`.

```bash
pip install turboquant-pro          # gives you `tqp` (1.8.0+)
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

### `tqp monitor --original PATH --reconstructed PATH [--floor F] [--window N] [--tangential-floor T] [--format json|prometheus|text]`
Feeds original/reconstructed `.npy` pairs through
`monitor.QualityMonitor.record_batch` and emits `metrics_dict()` — mean/min/p95
cosine, drift flags, the (A2) tangential fraction — as JSON, Prometheus
text-exposition, or a human table. **Exit code is a gate on `is_healthy`:** 0
when healthy, 1 when not, 2 on a load/shape error. Health requires **both** the
cosine floor (`--floor`) *and* (A2) noncollapse — a self-calibrating guard on
downward tangential drift, plus the optional hard level gate `--tangential-floor`
— so a stream sliding into the norm-dominated regime (where angular quantization
damages ranking while cosine still reads fine) is reported unhealthy, never
healthy-on-cosine-alone.

```bash
tqp monitor --original o.npy --reconstructed r.npy --format prometheus
```
```
# TYPE turboquant_quality_mean_cosine gauge
turboquant_quality_mean_cosine 0.9999
# TYPE turboquant_quality_is_healthy gauge
turboquant_quality_is_healthy 1
```

### `tqp certify --original PATH --reconstructed PATH [--metric cosine|l2] [--anchors N] [--seed N] [--min-tau T] [--task STR] [--task-kind KIND] [--environment] [--limitation STR ...] [--html FILE] [--out FILE] [--format json|text]`
Emits a **distribution-free rank certificate** (`rank_certificate`) as a
machine-readable `certificate.json`. Given original and reconstructed embedding
`.npy` matrices (same row order), it samples anchor pairs, measures the robust
distortion `kappa` and the corpus concentration `mu_hat`, and reports the
*guaranteed* floors `Kendall tau >= 1 − 2·mu_hat` and `Spearman rho >= 1 − 3·mu_hat`
— with **no distributional assumptions**. The JSON carries provenance (schema +
version, tool version, UTC timestamp, per-input shape/dtype/sha256, params) so a
certificate is reproducible and auditable.

Optional, additive envelope (does not bump `schema_version`; see
[CERTIFICATE_SPEC.md](CERTIFICATE_SPEC.md)): `--task "recall@10 >= 0.995"` declares
the downstream consumer (`--task-kind`, default `retrieval`, records its kind),
`--environment` stamps the run's software/hardware/git state, `--limitation "…"`
(repeatable) records scope caveats, and `--html report.html` writes a readable
human report alongside the JSON.

**Exit code is a gate:** 0 when the certificate certifies a positive floor (or
`tau_floor >= --min-tau` when given), 1 when it is vacuous / below the floor
(the corpus needs exact reranking), 2 on a load/shape error. The JSON is still
written even when the gate fails.

```bash
tqp certify --original emb.npy --reconstructed emb_q.npy --out certificate.json
tqp certify --original emb.npy --reconstructed emb_q.npy --min-tau 0.8   # CI gate
```
```
metric=cosine  anchors=200  pairs=19900
  kappa (robust distortion) = 1.0148
  Kendall  tau  floor       >= 0.8671
  Spearman rho  floor       >= 0.8006
=> certifies Kendall tau >= 0.8671, Spearman rho >= 0.8006 (distribution-free)
```
A vacuous certificate (`tau_floor <= 0`, seen on distance-concentrated corpora)
is itself the signal: single-stage rank fidelity can't be certified, so exact
reranking is mandatory.

### `tqp verify CERTIFICATE.json [--original PATH --reconstructed PATH] [--atol A] [--rtol R] [--out FILE] [--format text|json]`
Checks a `certificate.json` **that someone else emitted** — the trust primitive
`certify` was missing. Two layers:

- **Schema / self-consistency (always):** the schema and version are recognized,
  the required fields are present, `passed` is a boolean, and the recorded rank
  statistics are inside their valid ranges (`tau_floor`, `spearman_floor` in
  `[-1, 1]`; `kappa >= 0`; `n_pairs` a positive int). Runs offline, no data
  needed — catches a truncated or hand-edited certificate.
- **Independent recompute (when `--original`/`--reconstructed` are given):**
  re-hashes the two `.npy` inputs and compares to the recorded `sha256`, then
  **re-runs the certification with the certificate's own params** (metric,
  anchors, seed) and confirms the recomputed `kappa`/`mu_hat`/`tau_floor`/
  `spearman_floor` match the recorded values within `--atol`/`--rtol`. This is
  the reproduction check: same inputs + same params must reproduce the claim.

**Exit code is a gate:** 0 when verified, 1 when the schema is malformed *or* a
recompute mismatches (bad hash or drifted floor), 2 on a read error. `certify`
writes certificates; `verify` is how a third party trusts one.

```bash
tqp verify certificate.json                                   # schema/self-consistency only
tqp verify certificate.json --original emb.npy --reconstructed emb_q.npy   # full reproduction
```

### `tqp plan embeddings --embeddings PATH [--target STR] [--max-bytes-per-vector N] [--sample N] [--seed N] [--out FILE] [--format json|text]`
Task-aware embedding-compression planner. Runs `auto_compress` to sweep the
PCA/bit/rotation recipe space, then — for the recommended recipe — computes a
**rank-certificate preview** (`certificate_from_embeddings`): the acceptance
signal is the distribution-free Kendall/Spearman floor, *not* reconstruction
cosine (which is reported only as a labelled diagnostic). Emits `plan.json` with
the recommended recipe, the Pareto `alternatives` (each with bytes/vector), the
certificate preview, `risk_flags`, and a `tqp certify` reproduction command.
`--max-bytes-per-vector` constrains the recommendation to a byte budget.

> Scope: `auto_compress` ranks the frontier on the target metric — a **measured
> `recall@k`** when the target is a recall target (the default is
> `recall@10 >= 0.90`), else cosine/ratio for reconstruction-only checks. The
> plan's overall *acceptance* signal is the rank certificate; cosine is only a
> labelled diagnostic. A vacuous preview → exit 1 + "exact reranking required"
> (single-stage rank fidelity can't be certified on this corpus).

```bash
tqp plan embeddings --embeddings emb.npy --target 'recall@10 >= 0.90' --out plan.json
```

### `tqp plan kv --model NAME [--target quality|balanced|compression|extreme] [--context N] [--out FILE] [--format json|text]`
KV-cache policy planner over `AutoConfig`. Resolves the model from the built-in
registry (no network) or a HuggingFace path (needs `transformers`), applies the
target preset, and emits `kv_plan.json`: key/value bit policy, RoPE-awareness,
head/layer geometry, estimated cache size + compression ratio, and `risk_flags`
(e.g. keys below the 4-bit default surface the KV-keys risk).

```bash
tqp plan kv --model qwen2.5-7b --target balanced --context 32768 --out kv_plan.json
```

### `tqp replay <claim|all> [--claims claims.yaml] [--track T] [--full] [--list] [--dry-run] [--cwd DIR] [--out FILE] [--json]`
Executes claim reproductions from `claims.yaml`. Each claim with a `command`
runs through a shared harness that writes a normalized `results.json`, which is
checked against the claim's `expected` ranges (`*_min` / `*_max` bound the
like-named metric); claims without a command are `manual` reference entries
(surfaced by `--list`). Emits a report with a per-claim `verdict`
(`reproduced` / `regressed` / `error` / `manual` / `dry_run`) and a drift-class
hint on failure. **Exit code gates:** 0 if nothing regressed/errored, 1
otherwise, 2 on a usage/parse error. Needs PyYAML (`pip install
'turboquant-pro[yaml]'`).

> `command`/`full_command` run through the shell — `claims.yaml` is a trusted
> in-repo artifact; review before replaying an untrusted copy.

```bash
tqp replay --list                       # the claim ledger
tqp replay track1_recall_smoke          # CPU, seconds: recall@10 >= 0.80 @ >10x
tqp replay all --track embedding --json
```

### `tqp index <create|add|delete|compact|migrate|search|certify|drift|info>`
The production vector-index lifecycle for Track 1 — a persisted, compressed ADC
search index (PCA-Matryoshka + TurboQuant) in the versioned, CRC-checked **TQIX**
container. Every section is CRC32-guarded, so a flipped byte is a clean
`IndexCorruptionError`, never silent bad data; writes are atomic.

```bash
tqp index create --embeddings emb.npy --out index.tqe --output-dim 64 --bits 3
tqp index add    index.tqe --embeddings new.npy          # append, same basis, no refit
tqp index delete index.tqe --ids 12,88,90                # tombstone by external id
tqp index compact index.tqe                              # drop tombstoned rows, reclaim bytes
tqp index migrate index.tqe --to-version 2               # v1 (positional ids) -> v2 (ids+tombstones)
tqp index search index.tqe --queries q.npy --k 10 --rerank 10   # exact-rerank two-stage
tqp index certify index.tqe --min-tau 0.5                # rank certificate over stored originals
tqp index drift  index.tqe --embeddings recent.npy       # is the PCA basis stale?
tqp index info   index.tqe                               # container + stats
```

**At scale** (indexes too large to load into RAM):

```bash
tqp index search index.tqe --queries q.npy --mmap --block 262144     # memory-mapped, bounded-RAM search
tqp index create --embeddings big.npy --out corpus.shards --shard-size 1000000   # sharded (shared basis)
tqp index search corpus.shards --queries q.npy --k 10 --rerank 10 --mmap          # fan-out over shards
```

`--mmap` memory-maps the big arrays and streams the codes in row-blocks (`--block`),
so peak memory is `O(n_queries × block)` regardless of index size (search-only —
mutations need an in-RAM open). `--shard-size` writes a **sharded** index: `--out` is
a directory of `shard_*.tqe` + `manifest.json` sharing one PCA basis, and a directory
or manifest path is searched as a shard set with the per-shard top-k merged globally.

- **Ids are external and stable.** `create`/`add` assign monotonic ids (or take
  `--ids`); they survive `compact` (rows are dropped, ids are not renumbered).
- **Exact rerank + certify need the originals.** `create` stores fp32 originals
  by default (`--no-originals` to skip); without them, rerank degrades to the
  compressed reconstruction and `certify` errors.
- **Acceptance is the rank certificate / recall**, never reconstruction cosine.
  `certify` emits a `turboquant-pro/index-certificate` doc; `--min-tau` gates the
  exit code on the Kendall-τ floor. `drift` exits 1 when the basis is stale.
- **Format versions.** v1 = positional ids (read-only-ish; no deletes); v2 adds
  explicit ids + a tombstone bitmap; **v3** bit-packs sub-byte codes at slot
  granularity (2 codes/byte at 3–4 bits, 4/byte at 2-bit) and elides
  arange-reconstructible ids + empty tombstones — a **lossless** re-encoding
  (rankings bit-identical to v2), ~1.7× smaller `--no-originals` indexes
  (24.1 B/row all-in vs 41 B/row in v2 at 2M rows / 4-bit). v1/v2 files keep
  opening; `migrate` (`TQEIndex.migrate(3)`) upgrades in place.

## Design notes
- **One acceptance metric, everywhere.** Rank fidelity / (A2) consumer metric /
  distribution-free certificate — cosine is only ever a guarded, labelled
  diagnostic. This is the coherence rule the whole surface obeys.
- **No new runtime dep.** `tqp` is `argparse`; the core install stays numpy-only.
- **Honest surface.** Unbuilt subcommands are visible but exit 2 with a roadmap
  pointer — never a silent no-op or a fake success.
- **`turboquant-pro`** remains the existing AutoConfig entry point; `tqp` is the
  new unified surface.
