# Equal Weight Error, Unequal Effect: Quantization Sensitivity at the Consuming Operator

Status: **rejected by TMLR without review; being revised for IEEE Transactions on Neural Networks
and Learning Systems (TNNLS).** The author passed on TMLR's message on 2026-11-22: submission
14863 "has been rejected without further review. It was deemed to be unlikely to meet one or
both of TMLR criterion". The message gives no reason. The author chose TNNLS the same day, and
the folder and the paper's files were renamed from `tmlr_operator` and `main` to
`tnnls_operator`. The source is now in IEEE's two-column journal class (`IEEEtran`), with the author named,
a 250-word abstract, index terms, table captions above the tables, the AI-use statement as the
Acknowledgment, and the end matter in IEEE's order (appendices, acknowledgment, references). It
builds to 17 pages. The TNNLS author information found on 2026-11-22 (an undated copy) charges
for pages beyond 13, so the length is the author's to decide. `tmlr.sty` and `tmlr.bst` are no
longer used. An unpublished Zenodo draft of the preprint exists (deposition 24454776); the author
publishes it or not.

The third draft folds in experiments E1 to E6 (run on Atlas and NRP by another session,
committed in b581fc3) and answers a review of the second draft:

- **Intervals.** Table 1 and the row-protection controls carry 95% intervals from resampling
  whole sequences. The reference perturbation is repeated over three seed pairs and three
  dither amplitudes.
- **Rababah et al.** A factorial on two models (Table 3) shows that the weight-statistic
  ordering under the shipped llama.cpp presets follows their bit allocation, and that at equal
  bits the output-level ordering puts value and output first. The abstract says the paper does
  not refute their ranking under its own protocol.
- **Row protection, a result against the second draft.** At a matched 3.625 bits on held-out
  text the longest-wavelength eighth removes 89% of the key damage on Llama-3.2-3B, far more
  than a random eighth (6% to 37%). The eighth with the largest weight error removes 96%, and
  wins on Mistral-7B too. The second draft's sentence that the tested weight statistics do not
  locate these rows is withdrawn at row granularity.
- **Routers.** The ratio R of error spread to margin is measured. 40.9% of Mixtral tokens have
  R < 1 at 4 bits and none of them flips; 0.003% of OLMoE tokens do.
- **Decay.** 99.8% of channels land on the linear grid's top level; a second, per-row linear
  grid also collapses. The collapsed perplexity depends on float rounding of the top level, so
  the paper says "above 10^9".
- **Manuscript.** Links are colored, not boxed. The setup sentence and the abstract's router and
  decay sentences are rewritten. The tracer is called a proposal.

**Reproduction.** `reproduce.ipynb` rebuilds every number in the paper from the committed
records and compares it with the printed value. Every comparison passes when executed (the
count is in the notebook's last cell). It does not rerun the models. Build it with `scripts/build_notebook.py`.
`lean/OperatorChecks.lean` checks the paper's identities in Lean 4.32.2 with Mathlib v4.32.2:
20 theorems, no `sorry`, standard axioms only (six of them are the inclusions that Whitney's
conditions (a) and (b) reduce to for the arrangement of logit ties); `lean/BUILD-RECORD.txt` has the hash and the
command. It was built on Atlas.

- **Tracer (E7, 2026-11-22).** Measured on eight library architectures built from their configs
  and four small traceable modules (`experiments/tracer_accuracy.py`, CPU). torch.fx tracing
  fails on all eight library architectures, so there the tracer is name rules only (200 of 200
  right, against truth read from the same names). On the small modules with uninformative names
  the graph pass lifts 9 of 13 to 12 of 13. Two defects found: the scan's backward walk tags the
  input map as state decay, and the `in_proj` name rule mislabels a recurrence's input map.

- **Second state-space checkpoint (2026-11-22).** Mamba-370m, run on Atlas CPU: the same
  pattern (log grid 18.94 against a baseline of 13.99; both linear grids collapse). Record:
  `benchmarks/results_mamba_decay_census_mamba-370m.json`. It is the same architecture, so it
  is a replication within Mamba.
- **Common form and strata.** A section states sensitivity as the output metric pulled back to
  the weights, and treats the selection operators as a Whitney stratification by ties (Figure 1,
  `scripts/make_fig_strata.py`), with a linear flip law checked against the noise sweep.

- **A different recurrent family (2026-11-22).** RWKV-4 169M on Atlas CPU
  (`benchmarks/rwkv_decay_grid.py`): baseline 31.32, log grid 32.32, grid uniform in the rate
  46.83, grid uniform in the time constant 166.8. The ordering of the bases replicates; the
  collapse does not, and the paper says the collapse is specific to Mamba.
- **A linear grid on the time constant (2026-11-22).** Both Mamba checkpoints on Atlas CPU
  (`benchmarks/mamba_decay_tau_grid.py`). One range per tensor: 71.3 and 162.5 (6.1 and 11.6
  times the baselines), a large loss and not a collapse. One range per row: 13.60 and 16.48
  (+17% and +18%), below the per-tensor log grid's +24% and +35%, though not at equal cost in
  stored ranges. The paper now says the collapse belongs to grids uniform in the rate, not to
  linear grids in general. A log grid with a range per row was not run.
- **Table 2 intervals (2026-11-22).** A second run recorded each sequence's KL
  (`experiments/results_matched_bit/per_sequence/`, commit bada529); Table 2 reports its ratios
  with paired sequence-bootstrap intervals. Every first-run ratio lies inside its interval.
- **Fresh-context review of a8a2d08 (2026-11-22).** No error in the mathematics. One data error:
  the OLMoE margins and noise sweep are in the router's probabilities, not its logits; the paper
  now marks those cells and says so. A rerun on logits is owed. Overclaims corrected (the
  sub-4-bit instability belongs to the per-channel quantizer; three of five weight statistics
  reproduce the reported ordering; the wavelength hypothesis was committed with its result).

- **Routing on logits (2026-11-22).** Both routers rerun with logits recomputed from the router
  input and gate weights (`experiments/routing_margin_logits.py`, commit 3029c87; records
  `benchmarks/routing_margin_logits_<model>.json` and every decision's margin in
  `routing_margins_<model>.npz`). Table 8's margin and noise rows are now in logits for both
  models, and the probability-space cells are gone. The linear law is tested with the exact
  average over the margins at noise down to 0.01 of the median margin: measured and predicted
  flip rates agree within sampling error there, and the measured rate rises to 1.4 to 1.8 times
  the pair law at one to two median margins.

- **Reviewer pass (2026-11-22, notes passed on by the author).** Rababah et al., first version:
  the Section 5 suggestion (higher precision for query and key, more aggressive compression of
  value and output) is confirmed against arXiv:2607.08734v1, read through a fetched HTML copy.
  The first version has no per-projection Q2_K allocation, so that attribution is removed and
  the allocation rests on our own reading of the GGUF tensor types. The tracer audit and the
  pair-law comparison moved to Appendices A and B. The routing table is now Table 8 and the
  tracer table Table 9.

- **A second quantizer (E8, 2026-11-22).** Llama-3.2-3B on Atlas GPU 1
  (`experiments/k_quantizer_control.py`): a per-row scale that minimizes the row's squared error
  (clipping factor on a grid, 0.25 to 1). It reverses the 4-bit ordering (QK/VO 1.20 [1.11,
  1.29] against 0.38 under absmax; key KL 0.068 against 0.011, with a smaller weight error),
  removes most of the 3-bit key failure (0.166 against 0.781), and keeps the order of the
  row-protection criteria with smaller effects (weight error 70%, activation 58%, longest
  wavelength 29%, random 9% to 19%). The paper's 4-bit claim is now scoped to the absmax scale.
  Mistral-7B was not run: Atlas GPU 1 had 8 GB free.

- **Intervals on Llama-3.2-3B (2026-11-22).** The factorial, the score-error decomposition and
  the wavelength probe were rerun with per-sequence values (commit 6ad3fd9); each rerun
  reproduces its committed record. Tables 3 and 4 and the decomposition paragraph now carry
  paired sequence-bootstrap intervals for Llama. The shipped mix at Q2_K is 0.91 [0.84, 1.02],
  not distinguishable from parity. Mistral-7B has none yet (GPU memory on Atlas).
- **Title (2026-11-22, owner's choice).** "Equal Weight Error, Unequal Effect: Quantization
  Sensitivity at the Consuming Operator". The abstract, the introduction and the paragraph on
  what would count against the claim follow the second-quantizer result.

**Revisions of 2026-11-22, after further review notes passed on by the author:**

- **The clipped scale on a second model.** On Qwen2.5-1.5B the 4-bit ratio goes from 0.28 to
  0.76 and does not reverse (Llama-3.2-3B: 0.38 to 1.20). The paper no longer calls the reversal
  general. Under the clipped scale the activation-error criterion beats weight error on Qwen
  (81% against 67%), the reverse of Llama, so the paper claims no order between the two.
- **GPTQ in place of rounding** (new Section 5.4, `experiments/k_gptq_control.py`), on both
  models. With the absmax scale value and output dominate at 4 and 3 bits (ratios 0.38 to 0.44),
  and the 3-bit failures of rounding disappear. With the clipped scale query and key dominate in
  all four cells (1.25 to 1.58). On the key projection, equal layer-output error gives a 9 times
  different KL on Llama.
- **The tracer's graph pass** now reads the autograd graph of one forward pass
  (`turboquant_pro/operator_trace.py`, 8 new tests). With no parameter name read it labels 200
  of 200 scored tensors on the eight library architectures, 30 of them from graph evidence, and
  16 of 16 on RWKV-4, which was held back while the pass was written. Symbolic tracing still
  fails on all nine.
- **Decay study.** Window-bootstrap intervals for the Mamba grids that do not collapse
  (`benchmarks/mamba_decay_intervals.py`). A log grid with a range per row costs 2%, against 17%
  and 18% for the time-constant grid at the same storage. With exact endpoints and no clamp
  (`benchmarks/mamba_decay_rate_safe.py`) the per-tensor grid uniform in the rate still collapses
  both checkpoints, at 1.3e10 and 3.8e9, so the collapse belongs to the grid. The earlier 32-bit
  reconstruction clamped 17% and 23% of the channels. The per-row pass of the endpoint-exact run
  on Mamba-790m had not finished at this commit; its record holds the per-tensor result only.
- **Text.** The limitations no longer deny the clipped-scale control. The introduction credits
  GPTQ's layer-output objective. Figure 1(b) says the linear write lacks the softmax's
  row-constant invariance. Section 5.3 states that the row rankings were recomputed under the
  clipped quantizer.
- The notebook passes 867 of 867 comparisons plus the endpoint checks added at this commit.

**Still not done:** the per-row endpoint-exact number for Mamba-790m; Mistral-7B under the
clipped scale and under GPTQ (no free GPU memory on the workstation); intervals for the routing
study and for the Mistral-7B rows of the factorial and the wavelength probe; data figures; an author biography, if TNNLS wants one at submission; the multiple-choice tasks of Rababah et al. The E5 records carry a
field, `logit_recompute_rel_err`, that compares logits with a softmax and validates nothing; the
script is fixed (cc2c7bb) but the records were not rerun.

It is the focused paper cut from the foundational overview (`paper/foundational/`). It holds
only the results that appear in no other paper (the overlap check of 2026-11-21):

| Section | Result | Foundational section |
|---|---|---|
| 3 | decision-level agreement and a reference perturbation | C3b, sec:behavioral |
| 4 | matched-bit projection sensitivity, the sub-4-bit key failure, the rotary-row fix | C3b, sec:behavioral |
| 5 | operator tracer (design and unit tests only) | C3c, sec:infer |
| 6 | routing margin, Mixtral-8x7B and OLMoE-1B-7B | C4a, sec:hybrid |
| 7 | state-decay basis, Mamba-790m | C4b, sec:hybrid |

Left out on purpose: the two-axis (A2) probe (belongs to Paper IV, geometric-observation), the
density dissociation (overlaps Paper III), the keys finding and asym-NF4 (cited as motivation
only), and the synthetic gate and recurrence tables (no committed record). The retrieval
section (the rank floor and the reversed taper prediction, foundational C3a and C4d) was in
the first draft (db443c5) and was cut on 2026-11-21 by the owner's decision, to keep the paper
on attention, routing and recurrence. Those two results stay clean, so they are free for
another paper.

## Build

```
latexmk -pdf tnnls_operator.tex
```

MiKTeX on the laptop. 17 pages in the IEEE class, 0 undefined references, 0 overfull boxes
(2026-11-22).

## Where each number comes from

`scripts/derive_numbers.py` and `scripts/derive_ratios.py` read the JSON records below and print
every figure in Tables 1 to 5 and Section 7. Copy the records into one directory and run the
scripts there. They were run on Atlas on 2026-11-21, and `scripts/numbers_derived.txt` is their
output, with the sha256 prefix of each record.

| Paper | Record |
|---|---|
| Table 1 | `experiments/results_matched_bit/behavioral_{Qwen2.5-1.5B-Instruct,Mistral-7B-v0.1,Llama-3.2-3B}.json` |
| Table 2, Section 4.1 text | `experiments/results_matched_bit/matched_bit_{Qwen2.5-1.5B-Instruct,gemma-3-4b-it,Mistral-7B-v0.1,Llama-3.2-3B}.json` |
| Table 3 | `experiments/results_matched_bit/k_wavelength_{Llama-3.2-3B,Mistral-7B-v0.1}.json` |
| rotary-row fix (Section 4.1) | `experiments/results_matched_bit/validate_rope_aware_k_Llama-3.2-3B.json` |
| Table 5 | `benchmarks/results_mixtral_routing.json`, `benchmarks/results_olmoe_routing.json` |
| Section 7 | `benchmarks/results_mamba_decay.json` |
| intro, 675 times | `docs/KV_KEYS_FINDING.md` (10643 / 15.77) |
| Table 1 intervals, reference repeats | `experiments/results_matched_bit/agreement_repeats_{Qwen2.5-1.5B-Instruct,Mistral-7B-v0.1,Llama-3.2-3B}.json` |
| Table 3 (factorial) | `experiments/results_matched_bit/llamacpp_factorial_{Llama-3.2-3B,Mistral-7B-v0.1}.json` |
| score-error decomposition | `experiments/results_matched_bit/score_error_{Qwen2.5-1.5B-Instruct,gemma-3-4b-it,Mistral-7B-v0.1,Llama-3.2-3B}.json` |
| Table 5 (row protection) | `experiments/results_matched_bit/k_row_protection_{Llama-3.2-3B,Mistral-7B-v0.1}.json` |
| Table 7, rows on R | `benchmarks/router_error_margin_{Mixtral-8x7B-Instruct-v0.1,OLMoE-1B-7B-0924}.json` |
| Section 7 census | `benchmarks/results_mamba_decay_census.json` |

`scripts/derive_e_numbers.py` prints every figure taken from the E1 to E6 records, with the
sha256 prefix of each record (LF form); `scripts/numbers_derived_e.txt` is its output.

## Open items for the owner

1. **References once marked `% verify`** were checked on 2026-11-22. QuantMoE-Bench: the
   arXiv record now carries that name in the title and a fifth author, Zhen Tan (entry
   corrected). Quamba: ICLR 2025, five authors (confirmed). Quamba2: ICML 2025, PMLR 267,
   pages 10411 to 10427, six authors (pages added). OLMoE: ICLR 2025 (confirmed). RWKV:
   Findings of EMNLP 2023, pages 14048 to 14077 (entry changed from the preprint). The
   llama.cpp tensor types stated in Section 5.1 match `src/llama-quant.cpp`.
2. **Rababah et al., v1 against v2.** v2 (read in full, 2026-11-21) still ranks Q/K above V/O,
   uses weight moments and weight-space divergences under llama.cpp legacy and K-quant schemes on
   Llama-3.2-3B, Vicuna-7B, Mistral-7B and Llama-3.1-8B, and has no appendix, no per-projection
   bit table and no "compress V and O harder" advice. The paper attributes the Q2_K bit split
   (Q and K at 2 bits, V at 4, O at 3) and that advice to v1, from our note's reading. Check both
   against the v1 PDF, and the bit split against llama.cpp's tensor-type rules.
3. **"Stated in advance."** The wavelength prediction (Section 4.1) is recorded in an analysis
   note, not in a sealed registration. The paper says so. Confirm from git history that it was
   written before the run, or soften further.
4. **Figures.** The draft has tables only. Candidates: flip ratio against noise scale for both
   routers, and the KL ratio against bit width for the four models. Build them on Atlas from
   the records.
5. **Anonymized supplement.** `python scripts/build_supplement.py OUT_DIR --zip FILE` builds
   it (2026-11-22): 84 files, 3.3 MB, the code, the records, the notebook, the derivation
   scripts and the Lean file, with the library modules shipped as the package `opquant`. The
   build fails if any file holds an identifying string. The notebook, run from the unpacked
   zip alone, passes 662 of 662. Not shipped: the cluster-staging helpers, which depend on a
   package that names the author. One record's `gpu` field is rewritten. Read the zip once
   before uploading it.
6. **Anonymous citation.** `anon2026overview` is the foundational overview (if it is posted as a
   preprint). Replace it at camera-ready. If the overview later goes to an archival venue, it
   must cite this paper, not repeat its results.
7. **AI-use statement** in the paper. Check that its wording matches what you will attest.

## Experiments the review notes called for (run 2026-11-21 and 22)

All six are run and their records are committed (b581fc3). The paper reports each.

| | Experiment | Answers | Models |
|---|---|---|---|
| E1 | Record per-sequence outcomes; paired confidence intervals by resampling sequences; repeat the dithered 8-bit reference over several seeds and amplitudes | notes 1 (Section 3, Table 1) | Qwen2.5-1.5B, Mistral-7B, Llama-3.2-3B |
| E2 | Row protection at a matched 3.625 bits per key weight: longest-wavelength eighth against a random eighth (5 or more seeds), the shortest-wavelength eighth, and eighths chosen by weight error and by a declared sensitivity statistic; held-out WikiText-2 sequences; also the replication on held-out data | note 4, the paper's centerpiece | Llama-3.2-3B, Mistral-7B control |
| E3 | Decompose each perturbed score row into row-constant and centered parts, for Q and K weight perturbations; relate the centered part to the measured KL | note 2 (softmax hypothesis) | the four matched-bit models |
| E4 | Repeat the original comparison one factor at a time: llama.cpp bit mix against equal bits under one scheme, and weight statistics against functional KL on the same quantized models | note 3 | Llama-3.2-3B, Mistral-7B |
| E5 | Distribution of centered gate-logit error divided by boundary margin, by layer, under 4- and 3-bit per-tensor gate quantization | note 5 | Mixtral-8x7B, OLMoE-1B-7B |
| E6 | Count Mamba decay channels that round to the linear grid's top level, and their decay-rate change; add a linear grid that preserves slow channels | note 6 | Mamba-790m (CPU) |

E2's outcome: a random eighth does not protect as well as the longest-wavelength eighth, and the
weight-error eighth protects better than both.

## Errors found in the repo records while drafting (fixed in 125cc8b)

- `benchmarks/RESULTS_heat_taper.md` said the taper loses "0.9 to 3.6" recall points on BGE-M3.
  Its own table gives 0.9 to 3.0.
- `docs/notes/projection_sensitivity_deconfounded.md` gave the 4-bit range as 2.3 to 3.8 times
  (the records give 2.35 to 3.78), said the 2-bit ratios "drift toward parity" (Qwen stays at
  0.27), and gave "74 to 92%" for 2-bit flips (the records give 68 to 100%).
- `docs/notes/operator_trace.md`, `turboquant_pro/operator_trace.py` and
  `paper/foundational/main.tex` said value and output are "2.3 to 6 times" more sensitive
  across three architectures. At 4 bits it is 2.4 to 3.8 times across four. The 6 is Qwen at
  3 bits.
- `docs/model_cards/moe_routing.md` reconciled against foundational-paper numbers (12.4 and 3.5
  times, 45% and 87%) that the paper no longer contains.
