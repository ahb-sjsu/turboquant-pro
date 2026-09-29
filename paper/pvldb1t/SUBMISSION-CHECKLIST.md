# Submission checklist, PVLDB Volume 20 (VLDB 2027), Scalable Data Science

Prepared 2026-09-29. Every venue fact below was read from the VLDB 2027 site on that date.

## Venue facts, verified

| fact | value | source |
|---|---|---|
| Category | Scalable Data Science (SDS), one of four equal research-track categories | https://vldb.org/2027/call-for-research-track.html |
| Page limit | "up to 8 pages excluding references"; "All content, including any appendices and acknowledgements but excluding the references, must fit on the given number of pages" | https://vldb.org/2027/submission-guidelines.html |
| Review | single-blind; "authors MUST include their names and affiliations on the first page" | https://vldb.org/2027/submission-guidelines.html |
| Deadlines | 1st of each month, 5:00 PM Pacific, with mandatory abstract by the 25th of the previous month; Volume 20 runs April 1, 2026 to March 1, 2027 | https://vldb.org/2027/submission-guidelines.html |
| Template | official PVLDB LaTeX template, https://github.com/vldbproceedings/VLDB-Template (acmart + pvldb.sty, pinned copies in this directory) | https://vldb.org/2027/formatting-guidelines.html |
| Supplementary material | authors "must submit supplemental material" placed in "a publicly accessible archival repository" with a URL given at submission | https://vldb.org/2027/submission-guidelines.html |
| SDS scope | systems "in the real world, with a special focus on different dimensions of scalability, such as data size, ..., or degree of parallelism"; deployed-solution papers describe the problem, design choices, implementation challenges and lessons learned | https://vldb.org/2027/call-for-research-track.html (category text as indexed by search) |
| NRP acknowledgement | required NSF award list CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019 | https://nrp.ai/documentation/userdocs/start/policies/ |

Not verified on the site and left as is: whether the abstract-deadline rule applies to SDS
exactly as to regular papers (the guidelines state it once for all categories), and whether an
availability URL in the PDF footer is mandatory (the template provides it and this paper sets it).

## State of the manuscript

- Built clean with pdflatex + bibtex, 7 pages including references, no overfull boxes, no
  undefined references, so the body is under the 8-page limit with room. The rerank-bound
  paragraph and table (item 1 below) will add a few lines.
- Figures, all TikZ/pgfplots and self-contained: Figure 1 fleet diagram, Figure 2 recall by
  corpus size (kept), Figure 3 predicted-versus-measured recall over probe width, Figure 4
  per-server wall-time ECDF by phase, Figure 5 run timeline. Figures 3 and 4 and Table 3 read
  data files produced by `figdata/fig_data.py` from the record logs; rerun it after any change
  to the record.
- Related work carries 19 references, each verified against its publisher page, DBLP or arXiv
  on 2026-09-29. None was dropped as unverifiable; the DiskANN and NeurIPS'21 competition
  entries already in refs.bib were checked and are now cited.
- Lean 4 / Mathlib check under `lean/`: nine theorems build on Atlas at Lean v4.32.2, axioms
  recorded in `lean/AXIOMS.txt` (propext, Quot.sound, Classical.choice only). Cited in
  Sections 4.2 and 7.
- Tone grep (enforce, violat, killed, died, ban, threshold, utilization, stopped, bug, honest,
  hostnames): clean except the author email line. No cluster hostname appears.
- House prose scan: no colons, semicolons or em dashes in prose; no bullet lists in the body.

## Pending results (from the fleet session, 2026-09-29)

- **NESTED-SCALE PENDING** (Section 4.1, Figure 2): recall against the exact scan of the first k
  and of seeded random subsets of k servers of the one 10^12 index, k = 1..500, widths 16..256
  (`benchmarks/fleet/record/1t/post/nested_1T.log`, `nested1t.json`). It is the scale series in
  which only N changes.
- **NONMEMBER-QUERIES PENDING** (Section 4.3): 100 queries from shards 200000, 250000, 300000 and
  350000, which no corpus shard uses, so no query is a corpus row or has a home shard; reference
  scan plus routed 32 and 128 over all 500 servers, run tag 1tnm (`score_1Tnm.log`). It answers the
  reviewer's query-locality question. The per-volume index hashes it writes
  (`hash1tnm_part_*.json`) belong in Section 9.

Both markers are LaTeX comments in main.tex; delete each when its result is in the text.

## Reviewer pass, 2026-09-29

Done: placeholders came from a single-pass build (full pdflatex, bibtex, pdflatex, pdflatex
leaves none); 95 percent bootstrap intervals over queries at 32 and 128 probes
(`figdata/fig_data.py`, `recall_ci95` in `stats.json`, per-query values reconstructed from the
recorded summary and checked against it); the query recipe at every scale point stated in the
abstract, introduction and Section 4.3; the hubness paragraph rewritten (attributed mechanism,
and why it does not act on recall against the scan); the stability explained from the
reachability identity; the routed wall times marked as job cost, not search cost; Figure 1
volume label; Table 1 and Table 2 labels; Section 5 pilot names; one mention of the ninety-poll
wait; the rerank bound in the abstract and conclusion. Not done: a dispersed-query experiment at
10^11 (superseded by the non-member run at 10^12).

## Owner items before upload

1. **Rerank-bound numbers.** Section 7 carries the marker RERANK-BOUND NUMBERS PENDING in the
   paragraph and in Table 5's caption, and Table 5 is a skeleton (sets ref, 16, 32, 64, 128,
   256; rows survival and transfer). The parent session fills these from
   `rerank1t_bound.json` when the job lands, then deletes both markers. The abstract does not
   mention the bound and need not.
2. **Final read** of Sections 1, 7 and 9 for voice.
3. **Author block.** Single author as drafted. Add co-authors, if any, in the acmart author
   block on page 1 (single-blind, names required).
4. **NRP wording.** The acknowledgement uses the NSF award list from the NRP policy page and a
   plain thanks. NRP_SCALE_REQUEST.md commits only to "credit NRP in the paper", so there is no
   other required wording; confirm the platform paper citation (Weitzel et al., PEARC '25) is
   wanted in the introduction, where it now sits.
5. **Citations to confirm.** `douze2024faiss` (arXiv 2401.08281) and `yu2025dsann`
   (arXiv 2510.17326) are preprints; `simhadri2024bigann23` (arXiv 2409.17424) likewise. Replace
   with published versions if they have appeared by the submission month.
6. **Availability URL.** `\vldbavailabilityurl` points at `benchmarks/fleet` on GitHub. The
   guidelines ask for an archival repository; mint a Zenodo record of the repository at the
   submission commit and put that DOI URL in the macro.
7. **DOI and pages macros** (`\vldbdoi`, `\vldbpages`) stay as placeholders until camera-ready.
8. **Abstract by the 25th** of the month before the chosen deadline, in the submission system.

## Files to upload

- `main.pdf` (the paper), built from `main.tex`, `refs.bib`, `pvldb.sty`, `acmart.cls`,
  `ACM-Reference-Format.bst`, and the data files under `figdata/`.
- Supplementary material URL: the repository (archival DOI per item 6), which carries
  `benchmarks/fleet/` (scripts, driver, record logs), `paper/pvldb1t/figdata/fig_data.py`
  and `paper/pvldb1t/lean/`.
