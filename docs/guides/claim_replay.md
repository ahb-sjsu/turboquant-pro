# Claim replay guide — reproduce the headline numbers

Every headline number in this project is a row in [`claims.yaml`](../../claims.yaml),
and `tqp replay` runs it. No screenshots, no trust-me: the claims are executable.

```bash
pip install "turboquant-pro[yaml]"
tqp replay --list                 # the whole ledger
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

## The claims

| Claim | Track | What it proves | How |
|---|---|---|---|
| `embedding_glove_recall` | embedding | ~9.6× compression at recall@10 ≈ 0.999 on real GloVe-100 (1.18M) | **executable**, CI-gated on a hermetic subset |
| `track1_recall_smoke` | embedding | PCA+TQ preserves recall@10 at >10× in seconds on CPU | **executable** |
| `embedding_27x_high_recall` | embedding | up to 27× at high recall on concentrated-spectrum encoders | notebook |
| `kv_keys_per_channel` | operator | keys need per-channel/asym-NF4; PolarQuant collapses them | [model card](../model_cards/attention_keys.md) |
| `moe_routing_margin` | operator | routing fragility is carried by the margin (real OLMoE) | [model card](../model_cards/moe_routing.md) + `benchmarks/validate_olmoe_routing.py` |
| `ssm_decay_basis` | operator | SSM decay needs the native A_log basis (real Mamba) | [model card](../model_cards/ssm_decay.md) + `benchmarks/validate_mamba_decay.py` |

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
claim can never silently regress. See the
[user guide](user_guide.md) to build an index of your own, and
[certification](certification.md) for how the recall floor is guaranteed.
