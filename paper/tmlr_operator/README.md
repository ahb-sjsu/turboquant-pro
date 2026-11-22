# Quantization Sensitivity Follows the Consuming Operator (TMLR draft)

Status: **third draft, 2026-11-22, not submitted.** Target venue: TMLR (rolling submission,
double-blind, any length, accepts on evidence and interest, not novelty). The draft is the
anonymous submission build.

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

**Still not done:** a rerun of the OLMoE noise sweep on logits; a control with a clipped or
error-minimizing quantizer for the 3-bit key failure; verification of the two attributions to
the first version of Rababah et al.; an anonymized copy of the scripts for the supplement; data
figures; intervals for Table 2, the factorial, the score-error decomposition, the routing
study and the decay study; the multiple-choice tasks of Rababah et al. The E5 records carry a
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
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

MiKTeX on the laptop. 13 pages, 0 undefined references, 0 overfull boxes (2026-11-22, third draft).
`tmlr.sty` and `tmlr.bst` are copied from `geometric-observation/paper/`.

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

1. **References marked `% verify`** in `references.bib`: QuantMoE-Bench authors and identifier,
   the Quamba and Quamba2 venues, the OLMoE venue, and the llama.cpp K-quant tensor-type
   assignment.
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
5. **Anonymized supplement.** Code and records must ship without the library's name. The text
   already avoids it, and uses torch.fx as the only tool name.
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
