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

## State (2026-09-29, after the expansion)

- Builds clean: 6 pages including references, body ends on page 5 of the 8 allowed, no undefined
  references, no overfull horizontal boxes, all three mandatory first-page blocks present. The
  8 pages are a limit, not a target, and the paper is not padded to reach them.
- Added from the record, each number traced to its source:
  1. **Phases as a job graph** (Table 1): what each phase reads and writes, idempotency, the
     per-phase state file and job adoption (`driver1t_post.py`).
  2. **The submission controller and the pool's rules** (Table 2): the controller settings as
     deployed on Atlas (`/etc/nats-bursting/config.yaml`, unchanged since 2026-08-06, 20 running,
     5 pending, 100 cluster pending pods, 0.85 node CPU, back-off 30 s to 15 min, 15 attempts) and
     the pool constants in `driver1t_post.py`. Cites nats-bursting and polite-submit (both public).
  3. **Memory per job** (Table 3): exact-scan and routed pilots, and the build row from one of the
     six September rebuilds (cgroup memory at the 2 GiB limit through written pages, 292 MiB
     anonymous, per `fleet_common.drop_page_cache`).
  4. **Jobs in flight over time** (Figure 3), from the driver logs by `fig_data.py`
     (`inflight_*.dat`, `pool_timeline` in `stats.json`): median 19 in flight, 835 own completions
     plus 14 found done in 38.95 h, 501 in 39.43 h. Recycles reconcile with Table 4 (101, 73).
  5. **What the seed saves**, quantified: a file-based build would place and read a 128 TB corpus.
- Corrected at the expansion: the build memory figure "742 to 805 MiB" (from the build note) is
  not what the logs show and was replaced; "page cache is not charged" was wrong and was fixed.
- Dropped: per-server build wall times. The build pool's own log of August and September was not
  kept, so only the six rebuilds are measured, and the text keeps "two and a half to four hours".
- Prose scan clean apart from the clock-time colons in Table 4's caption. House tone grep matches
  only the author email line. CPU use appears as absolute cores, never as a share of a request.

## Owner items

1. Read the whole paper for voice before upload.
2. Decide whether to redraw Figure 1 so the pair shares no identical figure.
3. Declare the companion as a related concurrent submission in CMT.
