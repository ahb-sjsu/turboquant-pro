# Fairness & validation plan — turning "we grade ourselves fairly" into "others checked"

The repo's strongest asset is honesty of evidence. This plan closes the gaps that
keep that honesty *self-reported*: thin error bars, single-metric KV claims, and
breadth that is one-dataset/one-family deep. It is written as a **next-release
(1.8.0) hygiene-and-validation cycle**, mapped to concrete GPU/CPU jobs sized to
quota. Compute runs on the internal GPU box (CPU-side embedding/retrieval) and on
NRP Nautilus (GPU jobs); nothing is run on a laptop.

Status legend: ✅ done · 🔄 in progress / spec ready · ⏭ next.

---

## 1. Error bars everywhere

**Principle:** no headline number without a stated uncertainty. Single-seed
numbers get a bootstrap or multi-seed interval; "treat sub-0.01 as noise" prose
is replaced by an actual interval.

- ✅ **Canonical embedding tables** — percentile bootstrap 95% CI over the query
  set on every recall@10 (single-pass and +rerank). `benchmarks/canonical_embedding.py`
  (`recall_per_query` + `bootstrap_ci`, `n_boot`/`boot_seed`); rendered as
  `R@10 [95% CI]`. Cost `O(n_boot · n_queries)`, corpus-size-independent.
- 🔄 **Backfill CIs into the published `RESULTS_*.md`** (LaBSE 199k, gutenberg 1M,
  glove, rabitq comparison). Re-run the canonical harness with `n_boot=2000`,
  paste the CI columns. *Where:* internal box, CPU. *Cost:* minutes–1h each.
- 🔄 **Multi-seed for index-randomized methods** (rotation seed, IVF training) —
  5 seeds, report mean ± CI, so a sub-0.01 gap is visibly a tie.
- ⏭ **KV perplexity intervals** — perplexity is currently one number per
  (model, method). Report a CI over wikitext-2 shards (block bootstrap) and over
  ≥3 seeds. *Where:* NRP GPU. Folds into §2.

**Exit:** every table in `README.md` / `RESULTS_*.md` / `COMPREHENSIVE_ANALYSIS.md`
carries an interval or an explicit "exact/deterministic" note.

---

## 2. KV quality beyond fake-quant perplexity

**Gap (reviewer):** the KV claim is matched-bit fake-quant perplexity on
wikitext-2 — honest and predictive, but one metric on one axis. Add a real
downstream task and a real decode.

- ⏭ **Downstream task accuracy** — run **LongBench** (and one QA/retrieval task)
  under quantized KV, comparing each scheme to fp16 on the *task* metric, never on
  reconstruction error. Entry points exist: `benchmarks/benchmark_longbench_parity.py`,
  `benchmarks/tq_paper_lb_shard.py`. *Models:* Llama-3.2-1B/3B, Mistral-7B-v0.3,
  Qwen2.5-1.5B/7B. *Where:* NRP GPU (A10×8 default fleet; the reserved L40 pool is
  effectively unavailable to us). *Sizing:* per-model sharded Jobs, ≤8×A10, must
  sustain >40% util (batch the shards).
- ⏭ **Real deployed decode** — wire quantized KV into an actual `generate()` loop
  (not just fake-quant of a teacher-forced pass) and measure the same task
  metrics, so the claim covers the deployed path, not only the scoring path.
- ⏭ **CI on the KV numbers** — shard/seed bootstrap (§1).

**Exit:** the KV headline reads "matched-bit, fake-quant *and* real-decode, on
perplexity *and* LongBench, across N model families, with CIs."

---

## 3. Real-model validation of the v1.7 operator-sensitivity claims

**Gap (reviewer):** the MoE/SSM boundaries are measured on **synthetic**
operators — a strong pre-registration, not yet a systems claim. Close it with one
real MoE and one real SSM experiment through the *shipped* diagnostics.

- ✅ **MoE routing (real router logits).** Done on **Mixtral-8x7B** (top-2) and
  **OLMoE-1B-7B** (top-8): real router logits through `routing_sensitivity`, gate
  weights quantized, expert-set flip rate measured vs margin. The margin-
  concentration holds — 10.7× low/high at 4-bit on Mixtral (vs the paper's 12.4×),
  saturating on OLMoE's near-zero top-8 margins. `benchmarks/validate_{mixtral,olmoe}_routing.py`,
  `docs/model_cards/moe_routing.md`. *Ran on:* the internal box, GPU1 + CPU offload.
- ✅ **SSM decay (real recurrences).** Done on **Mamba-790m**: decays through
  `state_decay_sensitivity`, end-to-end WikiText-2 perplexity with the decay
  quantized in log-τ vs linear basis — linear collapses to ~10¹⁰, native `A_log`
  stays near baseline (14.44 vs 11.65). `benchmarks/validate_mamba_decay.py`,
  `docs/model_cards/ssm_decay.md`.

Both have landed: the operator boundaries are now real-model results, promoted to
`docs/model_cards/` and the `moe_routing_margin` / `ssm_decay_basis` claims.

**Exit:** one MoE + one SSM real-model result, each with CIs, promoted from
`docs/notes/operator_sensitivity_ssm_moe.md` to a README claim row in `CLAIMS.md`.

---

## 4. Prove the plugin contract: one external plugin + a hardware matrix

**Gap (reviewer):** the plugin architecture is right but aspirational until an
*out-of-tree* recipe passes conformance.

- ⏭ **`tqp-bnb`** — a tiny external package wrapping **bitsandbytes NF4/FP4** to
  the `Quantizer` protocol (the design doc's identified closest relative). Must
  pass `plugin_conformance.run_conformance` **and** one instrument smoke
  (rank-certificate + (A2) probe). Ships in its own repo; referenced here.
- ⏭ **`instrument_conformance` smoke** (in-tree) — the separate corpus-shaped
  certificate/(A2) check the design doc now points to (§2.2 reconciliation).
  Small, CPU, in-tree; unblocks the `tqp-bnb` smoke.
- ⏭ **Hardware portability matrix** — the public artifact that convinces:
  | rung | what | where |
  |---|---|---|
  | CPU conformance | plugin + format conformance suites | CI |
  | CUDA torch reference | `torch_decode` vs `fused_decode` exactness | NRP A10/A100 |
  | fused-decode exactness | `kv_fused_pck` compute-on-codes == reconstruct | internal GV100 + NRP |
  | public benchmark, HW-tagged | canonical ladder, tagged by device | NRP |
  Closes the backend P1 "remaining: NRP batch-Job suite on A100, MPS/ROCm numbers."

**Exit:** `docs/PLUGINS.md` links a green external-plugin CI badge; `CLAIMS.md`
has a hardware-tagged benchmark row.

---

## Sequencing & guardrails

**Order:** §1 backfill (cheap, CPU, immediate trust) → §4 `instrument_conformance`
+ `tqp-bnb` (unblocks external credibility) → §3 real-model MoE/SSM (the exciting
research closure) → §2 KV downstream (largest GPU spend) → §4 hardware matrix.

**NRP guardrails:** default to **A10×8** pods (L40 pool is reserved elsewhere);
GPU pods must sustain **>40% util** (batch shards, don't idle); size CPU-only
helper pods to the ignored range; check cluster policy before every submission;
never approach ban thresholds.

**Internal-box guardrails:** thermal-capped worker pools; long jobs go through the
scheduler, not ad-hoc; never reboot/kill others' processes; suspend co-tenant
workloads politely before a heavy run.

**Release story:** land as **1.8.0** — a hygiene-and-validation release. Master
already carries `1.8.0`; each ⏭ item that lands flips a `CLAIMS.md` row from
"synthetic/experimental" to "validated, with CI, on hardware X."

**What stays deferred:** PyPI Trusted Publishing (the project deliberately uses a
scoped token for releases — revisit only if provenance becomes a hard requirement).
