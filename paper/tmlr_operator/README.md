# Quantization Sensitivity Follows the Consuming Operator (TMLR draft)

Status: **first draft, 2026-11-21, not submitted.** Target venue: TMLR (rolling submission,
double-blind, any length, accepts on evidence and interest, not novelty). The draft is the
anonymous submission build.

It is the focused paper cut from the foundational overview (`paper/foundational/`). It holds
only the results that appear in no other paper (the overlap check of 2026-11-21):

| Section | Result | Foundational section |
|---|---|---|
| 3 | decision-level agreement with a noise floor | C3b, sec:behavioral |
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

MiKTeX on the laptop. 9 pages, 0 undefined references, 0 overfull boxes (2026-11-21, after the retrieval cut).
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
2. **The Q2_K bit assignment** (Q and K at 2 bits, V at 4, O at 3) is quoted from our note's
   reading of Rababah et al.'s appendix. Check it against their appendix and against llama.cpp.
   Their paper has a v2 (2026-08-21). Confirm that v2 still ranks Q/K above V/O.
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
