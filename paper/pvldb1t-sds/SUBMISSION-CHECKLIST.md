# Submission checklist, PVLDB Volume 20, Scalable Data Science

Companion to `paper/pvldb1t` (Experiment, Analysis & Benchmark). Split from one draft on
2026-09-29: this paper is how the trillion-row index was built and measured on a shared cluster,
the companion is what the measurement found. Venue rules and dates are in
`../pvldb1t/SUBMISSION-CHECKLIST.md` and apply to both.

## Rules that bind the pair

- **Monthly cap of two papers per author.** Both go in the same round: abstract by Sun Oct 25,
  paper by Sun Nov 1, 2026, 5:00 PM Pacific.
- **Cross-citation.** Each paper cites the other as "Under submission to PVLDB"
  (`bond2026recall`, `bond2026fleet` in `refs.bib`), and both must be declared as related
  concurrent submissions in CMT.
- **Distinct contributions.** This paper carries the corpus generator, storage, job shape, pool,
  failure classes, cost table and wall-time figure. The companion carries the scale results, the
  reachability identity, the rerank bound and the real-embedding pilot. The fleet diagram
  (Figure 1) and the index paragraph appear in both, because each needs the layout. Consider
  redrawing one of them before upload so no figure is identical across the pair.
- **Rejection rule.** Rejected work cannot return to PVLDB for a year, so each paper must stand on
  its own.
- **Title in CMT**, exactly: `A Trillion Vectors in the Smallest Job Class [Scalable Data Science]`.

## State

- Builds clean: 4 pages including references, body ends on page 4, no undefined references, no
  overfull boxes, all three mandatory first-page blocks present.
- Figure data (wall-time ECDF) is read from `../pvldb1t/figdata`, produced by `fig_data.py`.
- Prose scan clean apart from the clock-time colons in Table 1's caption. House tone grep
  matches only the author email line. The "Pods that stop answering" paragraph was rewritten at
  the split as neutral counts.

## To reach the page budget (8 pages, body now about 3.5)

Material in the record, none written yet:

1. The protocol as a job graph: the four phases, what each reads and writes, and why each is
   idempotent (from `driver1t_post.py` and the companion's Section 3).
2. The submission controller: pacing, the twenty-job cap, when it defers, and how the deferral
   rule interacts with it, with the constants of the pool rules in a table.
3. The memory budget per phase as a table: build, exact scan, routed pass, score, metadata, with
   mean and peak from the pilots and the build logs.
4. The build phase measured: per-server build wall times from the build logs, and the rebuild of
   six servers from seed.
5. What the seed saves, quantified: the bytes a file-based build would have moved to 500
   volumes, against what the seeded build moved.
6. The run on the calendar at more resolution: jobs in flight over time from the driver logs.

## Owner items

1. Read the whole paper for voice before upload.
2. Decide whether to redraw Figure 1 so the pair shares no identical figure.
3. Declare the companion as a related concurrent submission in CMT.
