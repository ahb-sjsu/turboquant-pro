# Results: observer advantage, Part II (attention keys)

Registration: `docs/PREREG_observer_advantage_keys.md` (registered #209; Amendments 1 and 2
before any verdict cell ran, 3 before K3's control cells ran, 4 to 8 after the verdicts were
computed, each marked as such in the amendment log). Harness:
`benchmarks/kvquant_matrix/` at `cd82d72` (Llama-2-7B and Tier B on Atlas, GV100) and tag
`keys-colab-v1` (Mistral-7B and Qwen2.5-7B on Google Colab, A100 40 GB), same package versions
(torch 2.10.0, transformers 5.5.0). Scorer: `score_keys.py` at master `507b902`, run on Atlas
on one scoring root, `/archive/ahb-sjsu/keys_score_20260926_final`: the Atlas campaign root
copied and the Colab Drive results copied as one archive (sha256
`d1a54758f7e6b40bfaee8fa3995f46f10b892c8ae62a9bab1fe29033d077c3fb`), taken when Colab's last
finished arm was Qwen2.5-7B `u2_R` (2026-09-26 23:41 UTC). Scorer output:
`benchmarks/kvquant_matrix/results/results_keys_20260926.json`.

**Status: FINAL_WITH_POSTHOC_EXPLANATION** (the scorer's `verdict_status`, Amendments 5 and 6).
Every verdict cell ran and verified on every Tier A model. The reported arms were still running
(Colab and Atlas) when this was written; the section on them is a snapshot and is refreshed when
both campaigns finish. The verdicts cannot change: their cells are complete.

Units. qasper: the paired mean difference of per-document F1 (points), arm minus reference.
Perplexity: the paired mean difference of per-chunk NLL per token, reference minus arm, so
**positive means the arm is better** on both. Intervals: 95% percentile bootstrap, 10,000
resamples, seed 0. A difference counts (**better** / **worse**) only when its interval excludes 0
and it exceeds twice the reference's one-ulp jitter floor.

## Gates

| gate | Llama-2-7B | Mistral-7B | Qwen2.5-7B |
|---|---|---|---|
| G0, every stage exact through the identity codebook (8 arms) | PASS | PASS | PASS |
| G1, `nf4a` reproduces the recorded matrix | PASS | FAIL_EXPLAINED_POSTHOC | PASS |
| G1 detail: qasper now vs recorded; perplexity now vs recorded | 21.45 vs 20.81; 6.964 vs 6.97 | 29.81 vs 28.74; 5.949 vs 5.955 | 42.09 vs 41.91; 7.4965 vs 7.499 |

Mistral's scored `nf4a` cell (Colab A100) missed G1's 1.0-point qasper tolerance by 0.07. The
direct test of Amendment 7 ran the same cell on Atlas's GV100, the recording hardware class:
qasper 29.71, which meets G1. The two GPUs differ by 0.09 on this cell, more than the miss, so the
failure is within cross-GPU variation (Amendment 8). **Not explained:** today's `nf4a` scores
above the recorded matrix on all three models (qasper +0.64, +0.97 to +1.07, +0.18; perplexity
within 0.1%). The offset lies within G1's tolerance on the recording hardware, and because every
comparison below is made within today's harness and one environment, it moves arms and
references alike. Amendment 4 first attributed the failure to the A100's numerics; the direct test
contradicted that, and Amendment 8 records the correction and the flaw in the test's design.

## Verdicts

| id | arm vs reference | Llama-2-7B | Mistral-7B | Qwen2.5-7B | verdict |
|---|---|---|---|---|---|
| **K1** (primary) | `nf4a_O` vs `nf4a_bm` | q +1.52 [−0.06, +3.37]; nll −0.61e-3 | q −0.41; nll −1.00e-3 **worse** | q −1.49; nll −1.21e-3 | **FAILS** |
| K1_low | `u3_O` vs `u3` | q +0.61; nll −0.65e-3 | q +1.70 [−0.19, +3.68]; nll −0.43e-3 | q +0.86; nll −0.33e-3 | **FAILS** |
| K2 | `nf4a_O` vs `nf4a_Ofor` | q −0.01; nll +8.31e-3 | q −0.51; nll +3.13e-3 | q +0.91; nll +17.04e-3 | **FAILS** |
| K3 | `nf4a_O` vs each of `nf4a_P`, `nf4a_R`, `nf4a_H` | better on neither endpoint against any control | worse on nll against `nf4a_R` | better on neither | **DOES NOT HOLD** |
| K4 | `u3_read` vs `u3` | q +0.57; nll +2.66e-3 **better** | q +0.83; nll +2.19e-3 **better** | q +3.19 [+0.04, +6.61] **better**; nll +2.13e-3 **better** | **INCONCLUSIVE** |
| K4b | `u3_read` vs `u3_key` | q +0.55; nll +1.16e-3 | q −1.42; nll +2.42e-3 **better** | q +0.97; nll +4.92e-3 | **FAILS** |

Where no judgement is written, the difference did not count. K2's perplexity differences favour
the fitted observer over the foreign one on all three models (+3.1e-3 to +17.0e-3), but each sits
inside twice its floor, which is large for these arms.

**The observer basis for keys is refuted.** Coding keys in the observer's coordinates is not
better than the shipped native-channel coding at matched stored bytes (K1), at 3 bits either
(K1_low), is not specifically the observer's doing (K2), and is not better than a random or
Hadamard rotation or the keys' own PCA (K3). On Mistral it is materially worse in perplexity than
the byte-matched native arm.

**Allocation by reads is promising but not established (K4).** Spending a fixed 3-bit budget by
the observer's read energy in native channels lowers perplexity materially on all three models,
but improves qasper materially only on Qwen2.5-7B; K4 needs both endpoints on two models, so it is
inconclusive. The read term itself is not shown to matter beyond key variance (K4b).

## Reported, not scored (snapshot)

Tier A, arm vs reference, qasper and NLL, judgement where it counted (b better, w worse; — not
yet run). Llama-2-7B's reported arms run last on Atlas, after Tier B.

| arm vs reference | Llama-2-7B | Mistral-7B | Qwen2.5-7B |
|---|---|---|---|
| `nf4a_O` vs `nf4a` (unmatched bytes) | q +1.31; nll −0.8e-3 w | q +0.05; nll −0.9e-3 w | q +0.45; nll −0.6e-3 |
| `nf4a_Oey` vs `nf4a_O` | — | q +0.12; nll +0.1e-3 | q −0.45; nll −0.0e-3 |
| `nf4a_Ocal` vs `nf4a_O` | — | q −0.37; nll −0.6e-3 | q +0.92; nll −0.5e-3 |
| `nf4a_bm` vs `nf4a` | q −0.21; nll −0.2e-3 | q +0.45; nll +0.1e-3 | q +1.94; nll +0.6e-3 |
| `u4_O` vs `u4` | — | q +1.25; nll −0.0e-3 | q +1.73; nll +0.2e-3 |
| `u4_R` vs `u4` | — | q +2.28 b; nll +0.3e-3 | q +1.69; nll −0.2e-3 |
| `u4_read` vs `u4` | — | q +2.38 b; nll +0.5e-3 b | q +2.01; nll +1.4e-3 b |
| `u4_key` vs `u4` | — | q +1.36; nll −0.1e-3 | q +1.74; nll +0.0e-3 |
| `u4_O_read` vs `u4` | — | q +1.67 b; nll +0.5e-3 b | q +1.62; nll +1.3e-3 b |
| `u3_R` vs `u3` | — | q +1.69; nll +0.6e-3 | q +1.69; nll −1.2e-3 |
| `u3_key` vs `u3` | q +0.02; nll +1.5e-3 b | q +2.25 b; nll −0.2e-3 | q +2.22; nll −2.8e-3 w |
| `u3_O_read` vs `u3` | — | q +1.63; nll +3.1e-3 b | q +1.29; nll +2.5e-3 b |
| `u2_O` vs `u2` | — | q +4.14 b; nll −7.2e-3 w | q +1.48; nll +1.8e-3 |
| `u2_R` vs `u2` | — | q +1.96; nll +6.0e-3 b | q +1.12; nll −9.9e-3 w |
| `u2_read` vs `u2` | — | q +0.68; nll −3.7e-3 w | in flight, excluded |
| `u2_key` vs `u2` | — | q +1.42; nll −3.8e-3 w | — |
| `u2_O_read` vs `u2` | — | q +2.98 b; nll +16.0e-3 b | — |

A cell counts only when complete (200 documents in each task and every perplexity chunk);
Qwen2.5-7B `u2_read` and Qwen2.5-1.5B `u2_O_read` were in flight and are excluded. Tier B:
Qwen2.5-1.5B has 18 complete reported pairs (5 better on perplexity, 3 worse, none counting on
qasper); Llama-3.2-3B has not started.

**Stored bits per key element** (Mistral-7B; the values depend on the model's head layout, for
example Llama-2-7B stores 5.60 for `nf4a` and 6.23 for the dense-basis arms; every model's values
are in the scorer output):
native `nf4a` 5.59, byte-matched and every dense-basis arm 6.03 (`nf4a_bm`, `nf4a_O`, `nf4a_P`,
...), `u4` 5.59, `u3` 4.61, `u2` 3.63, and allocation arms 0.001 above their uniform family.

**The stated expectations (not bars).**
- *O's gain, if any, is larger at 3 and 2 bits than at 4.* No basis-only O arm counts as better
  on both endpoints at any width; at 2 bits on Mistral O helps qasper (+4.14) and hurts
  perplexity. The arms that combine O with read allocation (`u4_O_read`, `u2_O_read`) do count on
  both endpoints on Mistral, which is consistent with the read-allocation pattern.
- *Random rotations help the uniform codebook and hurt `nf4a`.* Mixed for the uniform codebook:
  `u4_R` helps qasper on Mistral, `u2_R` helps perplexity on Mistral and hurts it on Qwen. For
  `nf4a` this snapshot has no rotation-against-native comparison; K3 shows only that `nf4a_R` is
  never materially worse than `nf4a_O`.
- *Read allocation beats key allocation where a few channels dominate `Q·K`.* On perplexity at
  3 bits, read beats key on Mistral (K4b); not shown on the other two.

## What it means, stated with its limits

The prereg fixed the consequences in advance (section 7). Its "all FAIL" row does not apply
exactly, because K4 is inconclusive rather than failed, so both of its parts are stated here.
**The observer claim for key coordinates is withdrawn:** turboquant-pro keeps native-channel key
coding, and the platform vision stops presenting an observer basis as a KV-cache stage. **Key
allocation stays uniform** in what ships, because K4 did not hold. Read-weighted allocation in
native channels is the one observer-shaped idea with support here: better perplexity on all
three models at 3 bits, on the two models measured so far at 4 bits, and on qasper on
Qwen2.5-7B at 3 bits.
That support was found on the models that measured it, so it is a hypothesis for a new
registration on fresh models, not a finding. The claims are scoped to three 7B instruction
models, LongBench trec, triviaqa and qasper, WikiText-2 perplexity, and one harness whose `nf4a`
sits slightly above its own historical record for a reason not yet identified.
