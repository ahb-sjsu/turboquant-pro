# Claim replay guide — reproduce the headline numbers

Every claim this project makes — in the README, [`CLAIMS.md`](../../CLAIMS.md), or
[`docs/claims.md`](../claims.md) — is a row in [`claims.yaml`](../../claims.yaml), and a
CI test ([`tests/test_claims_ledger.py`](../../tests/test_claims_ledger.py)) fails if the
two ledgers disagree. **Not every row is executable.** Two are (`tqp replay` runs them and
gates on floors); the rest are *reference* rows that point at the notebook, script, or
results file that reproduces them, and the ledger says plainly which ones rest on data that
is not in the repo (`reported`) or were withdrawn (`retracted`).

```bash
pip install "turboquant-pro[yaml]"
tqp replay --list                 # the whole ledger, with status per row
tqp replay embedding_glove_recall # run one claim and gate on its expected ranges
tqp replay all --track embedding  # run every executable claim in a track
```

## How a claim works

Each entry is either **executable** (has a `command`) or a **reference** (points at a
notebook / model card / script you run yourself). An executable claim's `command`
writes a normalized `results.json`; the runner checks it against the claim's
`expected` ranges (`*_min` / `*_max` bound the like-named metric) and reports a
verdict: `reproduced` / `regressed` / `error` / `manual` / `dry_run`. Exit code is 0
only if nothing regressed.

> `command` runs through the shell — `claims.yaml` is a trusted in-repo artifact.
> Review it before replaying an untrusted copy.

## Status vocabulary

| status | meaning |
|---|---|
| `executable` | `tqp replay <id>` runs it and gates on declared floors |
| `reproducible` | one *Run all* of the linked notebook on public data reproduces it |
| `needs-local-run` | reproducible on public data, but full scale / timing needs your hardware; no committed result at that scale |
| `partial` | notebook runs; one variant is still unfilled |
| `experimental` | needs GPU + model weights; engineering/preview |
| `reported` | measured once on data **not in the repo** (private sample or internal deployment); real, but not replayable from here |
| `retracted` | published, then found irreproducible; kept visible on purpose |

## The claims

| Claim | Track | What it proves | Status / how |
|---|---|---|---|
| `embedding_glove_recall` | embedding | ~9.6× compression at recall@10 ≈ 0.999 on real GloVe-100 (1.18M) | **executable**, CI-gated on a hermetic subset — the only CI-gated retrieval number |
| `track1_recall_smoke` | embedding | PCA+TQ preserves recall@10 at >10× in seconds on CPU | **executable** (not CI-gated) |
| `embedding_labse_32x_headline` | embedding | the README headline: 32× at recall@10 ≈ 0.999, ties OPQ, beats RaBitQ, ~20× faster build than OPQ | **reported** — private 199k LaBSE sample; [results file](../../benchmarks/RESULTS_labse_199k.md) |
| `embedding_beats_rabitq_ties_opq` | embedding | the same comparison on public ann-benchmarks data | **needs-local-run** — notebook; no committed RaBitQ row on public data |
| `embedding_27x_high_recall` | embedding | up to 27.7× at high recall on concentrated-spectrum encoders | **reported** — private BGE-M3 / LaBSE corpora |
| `pca_truncation_no_retrain` | embedding | PCA rotation makes non-Matryoshka models truncatable | **reproducible** — notebook 01 |
| `learned_codebooks_error_reduction` | embedding | learned codebooks cut quantization MSE ~22% | **partial** — notebook 02 |
| `adcindex_throughput` | embedding | ADCIndex reproduces the reconstruct ranking; QPS local | **reproducible** — notebook 03 |
| `embedding_114x_pipeline_storage` | embedding | up to 114× pipeline-storage accounting | **reported** — dataset-dependent, private corpora |
| `ood_anisotropic_robustness` | embedding | graceful degradation on anisotropic / heavy-tailed data | **reproducible** — notebook 04 (synthetic) |
| `kv_keys_per_channel` | operator | keys need per-channel/asym-NF4; PolarQuant collapses them | **experimental** — [model card](../model_cards/attention_keys.md) |
| `kv_nf4_matrix` | operator | asym-NF4 is near-fp16 across families; symmetric NF4 collapses high-GQA | **experimental** — notebook 11 |
| `kv_longgen_nf4a_degradation` | operator | (was) asym-NF4 degrades at long generation | **retracted** 2026-08-15 — [re-validation](../../benchmarks/RESULTS_longgen_revalidation.md) |
| `kv_vs_kvquant` | operator | comparison vs KVQuant / KIVI re-implementations | **experimental** — notebook 12 |
| `kv_fused_decode_speedup` | operator | fused decode kernel speedup | **experimental** — microbenchmark |
| `moe_routing_margin` | operator | routing fragility is carried by the margin (real OLMoE, Mixtral) | **experimental** — [model card](../model_cards/moe_routing.md) + `benchmarks/validate_olmoe_routing.py` |
| `ssm_decay_basis` | operator | SSM decay needs the native A_log basis (real Mamba) | **experimental** — [model card](../model_cards/ssm_decay.md) + `benchmarks/validate_mamba_decay.py` |

## The one you can run in 30 seconds

```bash
tqp replay embedding_glove_recall --small
# runs benchmarks/canonical_glove.py on a bundled tiny *real* GloVe subset,
# gates recall_at_10_rerank >= 0.95 and compression_ratio >= 9.5
```

The same claim's `full_command` runs the real 1.18M-vector GloVe on your own
hardware (`--full`). A **canonical artifact bundle** from one such run — results,
a provenance-stamped certificate, hardware, and the exact command — is committed at
[`benchmarks/artifacts/embedding_glove_recall/`](../../benchmarks/artifacts/embedding_glove_recall/MANIFEST.md).
The hermetic `--small` path is what CI gates on every push, so the central Track-1
claim can never silently regress. The floors are deliberately loose because the
bundled subset is tiny; the committed artifact is where the ~0.999 figure lives. See the
[user guide](user_guide.md) to build an index of your own, and
[certification](certification.md) for how the recall floor is guaranteed.
