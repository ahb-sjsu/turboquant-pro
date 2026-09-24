# Pre-registration: the planner's P0 exit test (issue #172)

**Status: REGISTERED on commit to master, before the planner has run on any campaign arm and
before the registrant has read any result of the RaBitQ public campaign.** The campaign's
results (`benchmarks/RESULTS_rabitq_public.md`, the cell JSON files, anything derived from them)
are sealed for this test: they are read only by the registered scorer, and only after the
planner's choices are committed (section 5). This file is never edited to fit results; changes go
in the amendment log with a date and a reason.

> **The question.** `docs/DESIGN_planner.md` calls the planner "a working control plane, not a
> validated one". Given a sample of a corpus, a query sample and a budget, does it choose a
> configuration the exhaustive grid would also have chosen, at a small fraction of the grid's
> compute, with evidence whose stated confidence is honest?

## 0. What the registrant knows

- The grid's design (`docs/PREREG_rabitq_public.md`, `benchmarks/rabitq_public/grid.py`): six
  public arms, configurations, seeds, stored-byte accounting, and the rr5 protocol.
- The planner's code, and the embedding codecs added for this test (`turboquant_pro/embedding_codecs.py`,
  #172 step 1), tested on synthetic data only.
- **Not known:** any recall, byte count or timing the campaign measured. The registrant has not
  opened its results file or cell outputs. General knowledge that PQ-family codecs and 4-bit
  scalar codes reach high recall after reranking is prior knowledge and is stated as such.

## 1. What the planner is given

For each arm and each budget of section 2, one planner run, `CompressionPlanner(spec).plan(...)`,
with:

| input | value |
|---|---|
| artifact | `N_PLAN = 100,000` corpus rows, drawn with seed `20260924` from the arm's corpus, excluding the grid's evaluation queries (on the real arms those rows were removed from the corpus already) |
| queries | `Q_PLAN = 500` planning queries disjoint from the grid's evaluation queries: on the ann-benchmarks arms `test[2000:2500]`; on the real arms 500 corpus rows drawn with seed `20260925`, removed from the artifact |
| consumer | `topk_inner_product`, `k = 10`, `rerank = 5`, `n_queries = 500`: the campaign's rr5 protocol |
| candidates | `tq_embedding`, `faiss_rabitq`, `faiss_pq`, `faiss_opq`, with `candidate_configs` pinned to exactly the grid's configurations of methods `tq`, `rabitq_flat`, `pq`, `opq` on that arm |
| split, search | `holdout_fraction = 0.3`, `halving_rounds = 3`, `min_sample = 256`, `n_boot = 512`, `seed = 0`, `confidence = 0.95` |

The **reachable** grid is the grid's configurations of `tq`, `rabitq_flat`, `pq` and `opq`. The
grid's `rabitq_ivf`, `rabitqlib_ivf` and `pca_rabitq_ivf` configurations are IVF search structures,
which the planner does not yet offer (P1). They enter only the full-grid regret, where a gap is
attributed to the missing candidates rather than to the planner's choice.

## 2. Budgets

Fixed from the design alone, per arm of dimension `d`:

- **Byte budgets, objective `max_quality`, no floor.** `B_b = ceil(d · b / 8) + 4` bytes per
  vector for `b ∈ {1, 2, 3, 4}`: four runs per arm.
- **Quality floors, objective `min_cost`.** rr5 recall floor `F ∈ {0.90, 0.95, 0.99}` at 95%
  one-sided confidence, no byte budget: three runs per arm.

Seven runs per arm, 42 in all.

## 3. Quantities

Grid values are the campaign scorer's (`rabitq_public.score.load`): per-query rr5 averaged over the
three seeds, on the grid's evaluation queries over the full corpus; a configuration's stored bytes
are the maximum over its seeds.

**Regret, byte budgets.** For budget `B`, `g*(B)` is the reachable grid configuration with the
highest mean rr5 among those storing at most `B`; the planner's choice is `p`. Compare `p` with
`g*(B)` on the grid's per-query rr5 by the campaign's paired rule (`score.paired_ci`,
`score.verdict`: BEATS / TIES / LOSES / INCONCLUSIVE, tie band ±0.01). A run has **no regret** when
`p = g*(B)`, or the comparison is TIES or BEATS. Regret in recall units, `mean(g*) − mean(p)`, is
reported for every run. The same is computed against the full grid (`G*(B)` over every method) and
reported.

**Regret, quality floors.** `c*(F)` is the cheapest reachable grid configuration whose mean rr5 is
at least `F`. The run **meets the floor** when the grid's mean rr5 of `p` is at least `F − 0.005`
(half the tie band), and has **no regret** when it meets the floor and stores at most
`1.05 · bytes(c*(F))` (the campaign's byte window). Excess bytes `bytes(p) / bytes(c*)` are reported.

**Planning cost.** CPU core-seconds of each planner run (process CPU time, recorded by the runner),
summed over the arm's seven runs, against the grid's core-seconds for the reachable configurations
on that arm: the sum over their cells (three seeds each) of `usage.mean_cpu_cores × usage.wall_s`
recorded by `cell.py`. Ground-truth computation is excluded from both. Candidate evaluations and
rows compressed are reported beside it. One asymmetry is stated in advance: the grid's PQ and
OPQ cells ran faiss's polysemous training (`index_factory` enables it for `PQ{m}x8`), which
permutes code ids and changes no score; the planner's adapters skip it (tested identical). The
grid's reachable core-seconds therefore include that training time.

**Calibration.** Two readings of the selected candidate's held-out lower bound `L` (one-sided,
95%):

- **same scale (scored):** the chosen configuration is evaluated once more on a fresh sample,
  disjoint from the artifact and the planning queries, of the held-out split's size (seed
  `20260926`, 30,000 rows, the same 500 planning queries); a violation is a fresh mean below `L`;
- **full scale (reported):** a violation is a grid mean rr5 below `L`. The planner declares no
  scaling law from a 100,000-row sample to the full corpus (`DESIGN_planner.md` §2.4), so this
  reading measures an extrapolation the planner does not claim; it is reported, not scored.

Runs that abstain have no `L` and are counted separately.

## 4. Verdicts (P0 exit test)

| component | PASS | FAIL |
|---|---|---|
| **R, regret** | no regret in ≥ 80% of the 42 runs, and no byte-budget run whose comparison with `g*` is LOSES by more than 0.05 mean | no regret in < 60%, or any LOSES by more than 0.10 |
| **K, planning cost** | the arm's planning core-seconds ≤ 10% of its reachable grid core-seconds on ≥ 5 of 6 arms | > 25% on ≥ 3 arms |
| **C, calibration** | same-scale violations ≤ 5% of non-abstaining runs plus binomial slack: at most `x` where `P(Binomial(n, 0.05) > x) < 0.05` | more than twice that |

Otherwise the component is INCONCLUSIVE. **The planner passes P0 when R, K and C all PASS.**
Abstentions: a floor run may abstain when no reachable configuration's grid mean clears `F`
(correct abstention, counted as no regret); an abstention when one does counts as regret. A
byte-budget run cannot legitimately abstain (no floor), so an abstention there counts as regret.

## 5. Execution and sealing

1. Merge this document (registration).
2. Run the 42 planner runs and their same-scale re-evaluations (`python -m planner_exit.run`, on
   NRP CPU pods through the campaign's pool and data volume, sized by its rules). Each run writes
   its plan record (schema-validated), its CPU seconds and its fresh estimate.
3. Commit every record under `benchmarks/planner_exit/records/` with `MANIFEST.sha256` **before**
   the scorer reads any grid result. The scorer refuses records that do not match the manifest.
4. Run the scorer (`python -m planner_exit.score`), which reads the committed records and the
   grid's cell results, and write `benchmarks/RESULTS_planner_exit.md` whichever way it falls.
5. No run is repeated because of its result; an operational failure is rerun unchanged.

## 6. Consequences (decided now)

| outcome | consequence |
|---|---|
| R, K, C PASS | "not a validated one" is removed from `DESIGN_planner.md`, the changelog and #169, with a pointer to the results; the claim is scoped to the six arms and the four reachable families |
| K FAILS | P0 fails its exit test (`DESIGN_planner.md` §6): successive halving and priors do not keep planning cheap, and the search design changes before any claim |
| R FAILS | the caveat stays, with the regret table; the failing runs are diagnosed by the stage at which `g*` left (prior, halving, frontier, verification) |
| C FAILS | the planner's statistical evidence is not trusted as stated: the bootstrap bound is replaced or widened before any plan quotes one |
| full-scale calibration poor | reported as the need for a declared scaling law (P1), not as a P0 failure |

## 7. Known limitations

- Four codec families, flat search only; IVF and graph indexes are outside P0.
- One planning sample size; the planning-cost ratio depends on it.
- The grid's queries on the real arms are held-out corpus rows (the campaign's limitation 1).

## 8. Amendment log

(none)
