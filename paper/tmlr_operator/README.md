# Quantization Sensitivity Follows the Consuming Operator (TMLR draft)

Status: **second draft, 2026-11-21, not submitted.** Target venue: TMLR (rolling submission,
double-blind, any length, accepts on evidence and interest, not novelty). The draft is the
anonymous submission build.

The second draft applies the owner's review notes of 2026-11-21 that need no new data. It is
centered on the matched-bit study and the rotary-row protection, as the notes recommend. The
reference perturbation is descriptive and no longer a noise floor, the z-scores are gone, the
claim against Rababah et al. is narrowed to an ordering under a declared protocol and reconciled
with their v2, and the softmax account is a hypothesis. The cached-key motivation keeps the
finite-rate reading, SpQR and Dutta et al. are credited for what they already showed, and the
OLMoE margin explanation is withdrawn (absolute margins are scale-free under absmax). The Mamba
grid endpoints are stated, and each table entry is marked measured or default. The experiments
the notes call for are listed below and are not run.

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

MiKTeX on the laptop. 10 pages, 0 undefined references, 0 overfull boxes (2026-11-21, second draft).
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

## Experiments the review notes call for (not run)

All need GPU time (Atlas GPU 1 or NRP) except E6, which runs on CPU. None has been started.

| | Experiment | Answers | Models |
|---|---|---|---|
| E1 | Record per-sequence outcomes; paired confidence intervals by resampling sequences; repeat the dithered 8-bit reference over several seeds and amplitudes | notes 1 (Section 3, Table 1) | Qwen2.5-1.5B, Mistral-7B, Llama-3.2-3B |
| E2 | Row protection at a matched 3.625 bits per key weight: longest-wavelength eighth against a random eighth (5 or more seeds), the shortest-wavelength eighth, and eighths chosen by weight error and by a declared sensitivity statistic; held-out WikiText-2 sequences; also the replication on held-out data | note 4, the paper's centerpiece | Llama-3.2-3B, Mistral-7B control |
| E3 | Decompose each perturbed score row into row-constant and centered parts, for Q and K weight perturbations; relate the centered part to the measured KL | note 2 (softmax hypothesis) | the four matched-bit models |
| E4 | Repeat the original comparison one factor at a time: llama.cpp bit mix against equal bits under one scheme, and weight statistics against functional KL on the same quantized models | note 3 | Llama-3.2-3B, Mistral-7B |
| E5 | Distribution of centered gate-logit error divided by boundary margin, by layer, under 4- and 3-bit per-tensor gate quantization | note 5 | Mixtral-8x7B, OLMoE-1B-7B |
| E6 | Count Mamba decay channels that round to the linear grid's top level, and their decay-rate change; add a linear grid that preserves slow channels | note 6 | Mamba-790m (CPU) |

The note-4 centerpiece hinges on E2. If a random eighth protects about as well, the result is
"protecting any eighth at 8 bits suffices" and the rotary localization becomes descriptive.

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
