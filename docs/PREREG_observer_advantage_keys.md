# Pre-registration: observer advantage, Part II (attention keys)

**Status: REGISTERED on commit to master.** Part I (`PREREG_observer_advantage.md`, section 8)
requires this registration to be committed before any Part I hypothesis is scored, so no Part I
verdict can shape these bars. At registration no Part II cell had run on a scored model; the only
runs were the wiring smoke on Qwen2.5-1.5B-Instruct (2 documents per task, 4 perplexity chunks,
section 6), whose numbers set nothing below. Supersedes the review drafts in
`ethicalfinite/docs/tqp-platform/PREREG_observer_advantage_keys_DRAFT.md`.

> **The question.** turboquant-pro quantizes KV keys with per-channel asymmetric NF4 (`nf4a`),
> 2% fp16 outliers, a 4-token fp16 sink and a 128-token fp16 hot window. Attention logits are
> inner products `q·k`, so Part I's observer basis applies to keys, with the queries of the heads
> reading a KV head as the consumer. **Does coding keys in the observer's coordinates, or
> spending their bits by the observer's reads, improve on the shipped native-channel coding?**

## 0. What the project already establishes

- **Per-channel structure is why the shipped quantizer works.** Keys carry a per-channel DC offset
  and a few channels dominate `Q·K` (`docs/model_cards/attention_keys.md`). Per-vector
  normalization destroys perplexity while reading 0.995 cosine; a zero-centred grid (symmetric NF4)
  collapses high-GQA models (Qwen2.5-7B qasper 43.77 to 4.69, `results_matrix.json`).
- **A basis change mixes channels.** A rotation or whitening before a per-channel codec spreads a
  dominant channel's offset across coordinates (the harness documents the same for RoPE). This is
  the central risk, and the reason no direction is predicted with confidence for the basis arms.
  A loss is informative: it would place the observer in bit allocation or codec choice rather
  than in the coordinates the codec sees. The allocation arms (K4) test exactly that alternative
  while staying in native channels.
- **Headroom is small at 4 bits.** `nf4a` sits within 1.25 / 0.69 / 1.86 qasper points and 0.4% /
  0.2% / 0.6% perplexity of fp16 on the three primary models. Any 4-bit gain is bounded by those
  gaps, so the design also registers a low-bit comparison (K1_low, 3-bit uniform) where the
  question has room to be answered.
- **Calibration-free is a stated strength** (`ROADMAP_best_tool.md`), and a read operator for a
  long context is better estimated over the whole sequence than over a calibration prefix
  (readscope C-11c). The registered O is therefore fitted **online from the prefill's own keys and
  queries**; the calibrated O is run and reported.
- **Acceptance endpoints for keys:** WikiText-2 perplexity and LongBench task scores, qasper the
  outlier-sensitive task. Reconstruction error and cosine are not acceptance metrics.

## 1. Decisions recorded from the review draft

The draft listed five open questions. Each option was weighed and one chosen; every option not
chosen as primary is still run, as a reported arm.

| # | question | options | decision and reason |
|---|---|---|---|
| 1 | reference for O | native `nf4a`; P | **native at matched stored bytes** (`nf4a_bm`) is the primary reference: the claim a user acts on is "better than what ships", and a dense basis costs D×D fp16 per head per sequence, so an unmatched native comparison would favour O by those bytes. P, R and H are the basis controls (K3); native unmatched is reported. |
| 2 | fit | online prefill; calibrated | **online prefill registered**, calibrated (`nf4a_Ocal`) reported: keeps the calibration-free property and follows the README's whole-sequence guidance. |
| 3 | models | three 7B; add 1.5B/3B | **all five**: the three 7B models (MHA, GQA 4:1, GQA 7:1) are Tier A and carry every verdict; Qwen2.5-1.5B and Llama-3.2-3B are Tier B, reported, same arms. No pilot result sets any bar. |
| 4 | materiality | fixed points; noise units | **noise units**: each codebook family has a jitter arm (every key moved one fp16 ulp, seeded); a difference counts only if its paired bootstrap interval excludes 0 **and** it exceeds twice the jitter arm's distance from the reference. A plain repeat was the draft's floor; the smoke showed runs are bit-deterministic, so a repeat would measure zero, while fp16 inference moves perplexity by ~0.1% under one-ulp key changes. |
| 5 | native-channel allocation | include or not | **include**, with its control: water-filling by the observer's read energy in native channels (`read`), against water-filling by key variance alone (`key`), and composed with O (`O_read`). |

## 2. The stages (harness knobs)

`benchmarks/kvquant_matrix/key_coding.py`, around the unchanged codebooks of
`tq_paper_lb_shard.py`. Defaults are the shipped path, whose artifact hash is unchanged.

| knob | values | meaning |
|---|---|---|
| `KEY_BASIS` | native, P, **O**, Oey, O_foreign, R, H | coordinates the codebook sees, per (layer, KV head). O = Part I's balanced maps (`observer_advantage.cell.balanced_o`, identical by test) with `S` the settled keys' second moment and `C` the pooled second moment of the post-RoPE prefill queries of that head's readers. Oey = Eckart–Young split. O_foreign = O with `C` from the next KV group. R = seeded random orthogonal, H = random-sign Hadamard (both regenerable, zero stored bytes). |
| `BASIS_FIT` | **prefill**, calib | moments from this prompt, or once per model from 16 WikiText-2 train sequences |
| `KEY_ALLOC` | **uniform**, read, key | bits per coded coordinate at a fixed D×KEY_BITS per head, integer water-filling (exact greedy) on `E[(Aq)_j²]·Var((Bk)_j)` (read) or `Var((Bk)_j)` (key), widths 1..8 |
| `BYTE_MATCH` | 0, 1 | extra fp16 outliers whose net bits equal one D×D fp16 basis |
| `KEY_JITTER` | 0, 1 | every settled key moved one fp16 ulp up or down (seeded): the floor arm |

Keys are coded as `z = Bk` and held as `k + B⁻¹(Q(z) − z)`, equal to `B⁻¹Q(z)` in exact
arithmetic, so only the quantization error passes through the inverse; sink and outliers are kept
fp16 in the coded coordinates. Every prediction and perplexity chunk records stored bits per settled key element by
component (codes, codebook metadata, outliers at 32 bits, basis, allocation table).

## 3. Arms, models, endpoints

The grid is `benchmarks/kvquant_matrix/keys_grid.py`, imported by the runner and the scorer.

- **`nf4a` family (4-bit, shipped codebook):** `nf4a`, `nf4a_jit`, `nf4a_bm`, **`nf4a_O`**,
  `nf4a_Oey`, `nf4a_Ofor`, `nf4a_P`, `nf4a_R`, `nf4a_H`, `nf4a_Ocal`.
- **`uniform` family at 4, 3 and 2 bits:** `u{b}`, `u{b}_jit`, `u{b}_O`, `u{b}_R`, `u{b}_read`,
  `u{b}_key`, `u{b}_O_read`.
- **`fp16`** (`NOQUANT=1`) and the **G0 arms**: the identity codebook through every basis and
  through `O_read`.
- **Models.** Tier A: Llama-2-7B-chat, Mistral-7B-Instruct-v0.2, Qwen2.5-7B-Instruct. Tier B:
  Qwen2.5-1.5B-Instruct, Llama-3.2-3B-Instruct.
- **Endpoints.** LongBench trec, triviaqa, qasper (all 200 documents, greedy, official metric,
  per-sample) exactly as in `results_matrix.json`; WikiText-2 test perplexity, SEQLEN 2048, every
  chunk, per-chunk NLL. qasper and perplexity carry the bars; trec and triviaqa are reported.

## 4. Gates

- **G0, the injection is exact.** Each identity-codebook arm must reproduce `g0_native`, the
  identity-codebook arm with no stage, bit for bit: every trec prediction identical and perplexity
  equal to 1e-6. The reference is not `fp16`, because values are quantized in every quantized arm,
  identity-key arms included (the smoke measured `g0_native` 1.2% above `fp16` in perplexity on
  Qwen2.5-1.5B, the 4-bit value path). A basis or allocation whose gate fails scores nothing, in any
  comparison. The orientation of the inverse, which the identity cannot see, is pinned by a unit
  test (a known coded-space error must come back as `B⁻¹δ`).
- **G1, the harness has not drifted.** On Tier A, `nf4a` must reproduce the recorded
  `results_matrix.json` qasper within 1.0 point and perplexity within 1%. A failure stops scoring
  until explained; it is reported either way.
- **Cell verification.** The scorer reads a cell only if its config sidecars agree with each other
  and with the arm's registered env line, including the key-coding record.

## 5. Hypotheses and bars

For arm X against reference Y, paired over documents (qasper) and chunks (perplexity):
mean difference with a 95% percentile bootstrap interval (10,000 resamples, seed 0), and the floor
`|mean(Y_jit) − mean(Y)|`, where `Y_jit` is the reference's family jitter arm. **Better** = interval above 0 and mean above twice the floor;
**worse** symmetrically. Verdicts use Tier A only and require G0 for every arm involved.

| id | X vs Y | HOLDS | FAILS |
|---|---|---|---|
| **K1** (primary) | `nf4a_O` vs `nf4a_bm` | better on qasper **and** perplexity in ≥ 2 of 3 models, worse on either in none | better in none; "reversed" if worse in ≥ 2 |
| **K1_low** | `u3_O` vs `u3` | same rule | same |
| **K2** (it is the observer) | `nf4a_O` vs `nf4a_Ofor` | same rule | same |
| **K3** (not any basis) | `nf4a_O` vs each of `nf4a_P`, `nf4a_R`, `nf4a_H` | better on qasper or perplexity in ≥ 2 models, against every control | otherwise |
| **K4** (observer allocation, native channels) | `u3_read` vs `u3` | same rule as K1 | same |
| **K4b** (the read term matters) | `u3_read` vs `u3_key` | same rule | same |

Anything else is INCONCLUSIVE. **Reported, not scored:** every other pair in
`keys_grid.REPORTED` (all bases and allocations at 4, 3 and 2 bits against their family's native,
Oey and O_cal against O, byte-matched against unmatched native), Tier B throughout, trec and
triviaqa, and stored bits per element for every arm.

**Stated expectations (not bars).** O's gain, if any, is larger at 3 and 2 bits than at 4.
Random rotations help the uniform codebook and hurt `nf4a`, which relies on native channel
offsets. Read allocation beats key allocation where a few channels dominate `Q·K`.

## 6. Execution

- Runner: the harness's own `main()` per (model, arm, task) cell, the 2026-08-08 re-validation
  pattern, with the Atlas thermal discipline or on NRP GPU pods sized by the resource policy.
  Order: G0 and `fp16`, then the arms of K1 to K4, then the reported arms.
- **Smoke before registration (wiring only).** Qwen2.5-1.5B-Instruct, trec and qasper documents
  0 and 100, 4 perplexity chunks, every stage combination. It changed the design in three places,
  all recorded above: the G0 reference (`g0_native`, not `fp16`), the floor (jitter, not repeat),
  and reconstruction (the residual form, after the direct form failed an exact G0). It also caught
  a jitter defect (a one-step move of a zero's bit pattern is a NaN encoding) that returned NaN
  perplexity; fixed and pinned by a test. After the fixes all eight G0 arms reproduced `g0_native`
  exactly and every arm ran to completion with finite output. No comparison was computed from
  these runs, and they are not part of any.
- **Shared hardware.** Atlas GPU 1 is shared; the runner starts no cell while another process
  holds the GPU, so other work is never slowed by more than the cell in flight.
- A cell is never rerun because of its result; an operational failure is rerun unchanged.
- Scorer: `python score_keys.py --root <dir> --lbroot <LongBench> --out results_keys.json`,
  written into `RESULTS_observer_advantage_keys.md` with every verdict as it falls.

## 7. What each outcome would mean for the platform

| outcome | product consequence |
|---|---|
| K1 or K1_low HOLDS, K2 and K3 HOLD | observer basis becomes an opt-in key transform stage, fitted from the prefill, at the byte cost stated |
| K4 HOLDS, K1 fails | the observer enters key compression through allocation, not coordinates; `read_allocation` in native channels becomes the key-path planner stage |
| all FAIL | keys stay native-channel, uniform-allocation; the observer claim for KV is withdrawn in the vision document |

## 8. Amendment log

- **Amendment 1, 2026-09-24, before any registered cell completed. Operational; nothing computed
  changes.** The first Tier A cell (Llama-2-7B, `g0_native`) waited for the shared GPU, started
  when it was free, and ran out of memory because another job claimed the GPU in the same minute.
  The runner now retries an out-of-memory start after waiting for the GPU again (up to six
  times), keeping each failed attempt's log. This is the rerun-unchanged rule of section 6 applied
  automatically.
- **Amendment 2, 2026-09-24, before any verdict cell ran. Operational; nothing computed changes.**
  One GV100 would take most of a week for Tier A, so the models are split across hardware, one GPU
  product per model: Llama-2-7B and Tier B on Atlas (GV100), Mistral-7B and Qwen2.5-7B on Google
  Colab A100 (`keys_colab.ipynb`, the same package versions as Atlas: torch 2.10.0, transformers
  5.5.0). Greedy decoding is not bit-stable across GPUs or library versions, so every cell now
  records its GPU and package versions (`env.json`), a cell refuses to finish anywhere but where it
  began, and the scorer compares an arm only with a reference and floor arm recorded in the same
  environment; a comparison across environments is treated as a failed gate. Cells resume inside a
  task after a disconnect (`RESUME=1`): each document's prediction and each perplexity chunk is an
  independent forward, so resuming does not change what is computed.
