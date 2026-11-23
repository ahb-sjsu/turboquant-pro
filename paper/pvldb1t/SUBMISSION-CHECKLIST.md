# Submission checklist, PVLDB Volume 20 (VLDB 2027), Experiment, Analysis & Benchmark

Prepared 2026-09-29. Every venue fact below was read from the VLDB 2027 site on that date.

**One paper, not a pair (2026-10-24).** The Scalable Data Science companion `../pvldb1t-sds`
was merged into this paper after an outside review and will not be submitted. Its source and
PDF are kept as history and still carry figures this paper corrects: 152 six-CPU exact scans
(there were 164), wall times that include pod start-up (Table `tab:cost` here is search time
only, start-up excluded), 1349 "CPU-hours" (1349 hours is search time summed over servers, and
Section `sec:cost` keeps it apart from the 2077 core-hours the jobs held), and unqualified
retry safety and seed regeneration (Section `sec:run`, "Safety of repeated jobs", states the
two weaker cases, and Section `sec:regen` says a rebuild matches only under the BLAS kernel,
thread count and write mode of the first build, and that 400 of the 200000 shards never did).
The sections dated 2026-09-29 below describe the pair. Where they differ from the entries of
2026-10-24 and 2026-11-22, the later entries hold.

## Venue facts, verified

| fact | value | source |
|---|---|---|
| Category | Experiment, Analysis & Benchmark, one of four equal research-track categories (this row read Scalable Data Science until the 2026-10-24 merge) | https://vldb.org/2027/call-for-research-track.html |
| Page limit | 12 pages excluding references for this category (8 for Scalable Data Science); "All content, including any appendices and acknowledgements but excluding the references, must fit on the given number of pages" | https://vldb.org/2027/submission-guidelines.html |
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

## VLDB rules checked against the manuscript, 2026-09-29

Read from https://vldb.org/2027/submission-guidelines.html and the official template
(github.com/vldbproceedings/VLDB-Template, main at 39c95f5) on 2026-09-29.

- **Target round: abstract by Sun Oct 25, 2026; paper by Sun Nov 1, 2026, 5:00 PM Pacific.** The
  Oct 1 round closed to new papers when its abstract deadline passed on Sep 25.
- **Category suffix.** A non-regular paper carries its category in brackets at the end of the
  title, "both in the paper file and in the CMT submission title". Since the split of 2026-09-29 the title is
  "Routed Recall Does Not Degrade at a Trillion Rows [Experiment, Analysis & Benchmark]" (retitled 2026-10-23 from "Recall Does Not Move ..."). Type it exactly
  so in CMT.
- **Mandatory first-page blocks** (PVLDB Reference Format, license footer, Artifact
  Availability) are emitted by `\vldbtopmatter`, which the template says to call right after
  `\maketitle`. The draft did not call it, so all three were missing (a desk-rejection cause).
  Fixed, and checked in the built PDF.
- **Template files** pvldb.sty, acmart.cls (v2.19) and ACM-Reference-Format.bst are
  byte-identical to the official template's current versions.
- **Page limit.** SDS is 8 pages excluding references, and appendices and acknowledgements count.
  The body ends on page 7 (references start on page 7). Letter paper, all fonts embedded.
- **Single-blind.** Author name and affiliation on page 1, as required.
- **Artifacts.** "a public GitHub repository" is an acceptable archival repository, so the
  availability URL may stay on GitHub. A Zenodo DOI is optional, not required.
- **Category, decided 2026-09-29: two papers.** This paper is Experiment, Analysis & Benchmark
  (12 pages excluding references; "required to make available all experimental data and related
  software", which the repository does). The build and pool sections moved to the companion
  `paper/pvldb1t-sds` [Scalable Data Science]. Both go in the Nov 1 round (the cap is two papers
  per author per month), each cites the other as under submission, and both are declared in CMT.
  See `../pvldb1t-sds/SUBMISSION-CHECKLIST.md`. [Superseded 2026-10-24: the companion was merged
  into this paper. One submission, no cross-citation, nothing to declare as concurrent.]
- **Page budget.** The body now ends on page 5 of 12. The two pending results below are the
  planned additions.
- **Concurrent review.** Neither paper can be under review elsewhere while PVLDB considers it,
  so do not also send either to SIGMOD or ICDE in the same window.

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

## Final pass, 2026-10-24

- Both pending results are in the text and both markers are gone. The non-member run (tag 1tnm)
  is in the abstract, introduction, Section 4.4, the threats table, the limits and the
  conclusion: 0.969 at 32 probes (95 percent interval 0.954 to 0.982) and 0.999 at 128 (0.997 to
  1.000), reference neighbours on 404 servers, at most 1 percent on any one, a median of 8 servers
  per query. Per-query values from `benchmarks/fleet/fleet_nonmember_analysis.py`, output in
  `benchmarks/fleet/record/1t/post/analysis1tnm_partials.log`. The across-scale claims are now
  stated for corpus-row queries.
- Title in CMT, exactly: `Routed Recall Does Not Degrade at a Trillion Rows [Experiment, Analysis
  & Benchmark]`. Plain-text abstract for CMT in `CMT-abstract.txt`.
- The 152 exact scans that ran at six CPUs and eight gigabytes are stated where the job shape is
  (abstract, Table 1 caption). Their median wall per server, 2660 s, matches the 2664 s of the 348
  one-CPU scans. [Corrected later: there were 164 six-CPU scans, and their median scan call was
  2573 s against 2757 s for the 336 at one CPU. `main.tex` Section `sec:run` states this.]
- Items 5 and 6 done. Every reference was checked against its publisher, DBLP, Crossref or arXiv
  and corrected; Faiss and Big ANN'23 now cite their published versions. Availability URL is the
  software DOI 10.5281/zenodo.23608470; the data DOI is 10.5281/zenodo.23608434. Both Zenodo
  records are DRAFTS until the owner publishes them, and must be published before Nov 1 so the
  DOIs resolve.
- Builds clean at 7 pages, no undefined references, no overfull boxes.

## Pass of 2026-11-22, for the Dec 1 round (abstract by Nov 25)

- Section `sec:cost` now accounts for the regenerated-index run (tag 1trg). Its 581 completed
  cluster jobs spent 4984 hours in their shard loops, a median of 9.0 hours per server, all at one CPU
  and two gigabytes except 73 whole-write rebuild jobs at three. The drivers logged 677
  submissions, and the Atlas workstation's 46 partials add 439 hours of process time. Every figure comes from
  `figdata/cost_1trg.py`, which reads only `benchmarks/fleet/record/1trg/` (the final merge's
  per-partial walls and the driver logs). It was run on Atlas, and its output is
  `figdata/cost_1trg.txt`. Reproducibility names the script.
- Section `sec:regen`, the 0.005 gap. The sentence no longer says that most of the four-shard gap
  "came from" the subspaces. It now says what the data show: the lead is less than half the width
  of either interval, so this measurement cannot separate a corpus row's advantages from none,
  and the four-shard gap of 0.020 is consistent with the choice of four subspaces alone.
- "every number" (abstract, introduction twice) replaced by claims about the record: the tables and
  figures are computed from it, and the results can be recomputed from it. `CMT-abstract.txt`
  changed to match the LaTeX abstract.
- Intro paragraph overfull (2.86 pt) fixed. Build: 11 pages, 0 undefined references, 0 overfull or
  underfull lines. Banned-word, tone, colour and eaten-backslash checks clean.

Owner items for this round:

1. **Abstract in CMT by Wed Nov 25**, title exactly `Routed Recall Does Not Degrade at a Trillion
   Rows [Experiment, Analysis & Benchmark]`, abstract text from `CMT-abstract.txt`.
2. **Zenodo DOIs do not resolve yet.** On 2026-11-22 the public API returned 404 for both
   10.5281/zenodo.23608470 (software, the PDF's availability URL) and 10.5281/zenodo.23608434
   (data), so neither is published (the 2026-10-24 entry above records both as drafts). The software draft dates from 2026-10-24, before the 1trg
   record and cost script were committed. Update it to the submission commit, then publish both
   before Dec 1.
3. Final read of Sections 1, 7 and 9 for voice (unchanged from earlier passes).

## Review notes of 2026-11-22 (five items), checked against the record

- **Table 2, "see text" at 10^9.** The 10^9 record (`benchmarks/fleet/results/fleet_run_1B.json`)
  holds routed recall against the exact scan, 0.978 at 32 probes and 0.999 at 128, so both are now
  in the table and in Figure 2. The run is final, not interim (the interim point is the 349-server
  one at 10^12). The text says how it differed: a serving path with a coordinator, not batch jobs,
  plus a cold store. The note's suggested 0.982 is the 10^10 value. The 32-probe series is now
  0.979, 0.978, 0.982, 0.983, 0.989, and the text says the 10^8 to 10^9 step is one neighbour in a
  thousand. Also corrected: "the score program has not changed since the 10^9 run".
  `fleet_score10.py` dates from the 10^10 run, and at 10^9 the coordinator (`fleet_coord.py`)
  computed recall with the same function.
- **Section 4.7 title.** The note read the section as 200 queries from four shards. The section is
  200 queries, one from each of 200 shards (row 0 of shards 400000 to 400099, and one shard on each
  of 100 servers). Its opening sentence, about the other two sets, caused the misreading. Retitled
  "Two hundred queries from two hundred shards", and the opening now names the four-shard sets.
- **Cell count.** New paragraph at the end of Section 4.8 and a threats-table row. 2048 was set in
  `benchmarks/fleet/fleet_common.py` at the 10^9 run and never changed. No other count was
  measured, and the repo records no other reason for the number, so the paper gives none. The
  paragraph says what more cells would change (65536 balanced cells: about 2 billion rows at 128
  probes, against 62.5), why recall at another count is unmeasured, and that the claim is about
  fixed cells and width, not latency. The note's "1239 s" is not in Table 10. Routed 128 is 1335 s
  against the scan's 2664 s.
- **M0, M1, M2.** Equations (1) to (3) in Section 4.3. Table 4 is now full width with each
  registration's calibration sizes, curve and fitted law. The fits come from the two committed
  prediction records through `figdata/transfer_fits.py`, run on Atlas, output
  `figdata/transfer_fits.txt`. It asserts that every refit equals the recorded prediction, to 1e-9.
- **Synthetic corpus.** Section 7 has two new paragraphs. The first says why the corpus is seeded,
  what depends on the generator (where neighbours fall, and so recall at a width and its rise with
  the rows), and what does not. The second says what the real pilot shows and does not, and that the
  two are not combined. Corrected: the paper called the pilot "the same index". Its record
  (`real_pilot_15M_wiki1024.json`) has 256 projected dimensions and 4096 cells, against the fleet's
  24 and 2048, and 500 queries. Its 0.949 at 64 probes is now stated too. The threats row and the
  conclusion say the result is on this generator.
- Build: 12 pages including references, 0 undefined references, 0 overfull. Underfull is the same
  profile as before: 11 lines in the sloppy Reproducibility paragraph and 8 column-balance
  warnings. Banned-word, colour, em-dash and eaten-backslash checks clean.
