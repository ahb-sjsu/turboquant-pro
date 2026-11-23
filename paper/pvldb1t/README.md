# Routed Recall Does Not Degrade at a Trillion Rows

PVLDB Vol. 20, Experiment, Analysis & Benchmark. This folder holds the paper. The code that
built and measured the index is in `benchmarks/fleet/`, and the record of the run, the logs and
JSON files every table and figure is computed from, is in `benchmarks/fleet/record/`. Paths below
are relative to the repository root.

## Where things are

| what | where |
|---|---|
| Paper source and PDF | `paper/pvldb1t/main.tex`, `refs.bib`, `main.pdf` |
| Figure data and the scripts that write it | `paper/pvldb1t/figdata/` |
| Lean 4 proofs of the two identities | `paper/pvldb1t/lean/` (its own `README.md`) |
| Build, scan, routing, scoring and analysis scripts | `benchmarks/fleet/` |
| Record of the trillion-row runs (corpus-row and non-member queries) | `benchmarks/fleet/record/1t/`, scores and analyses in `record/1t/post/` |
| Record of the regenerated-index run (Section 4.7) | `benchmarks/fleet/record/1trg/` (its own `README.md`) |
| Earlier scale points, 10^8 to 10^11 rows, and the real-data pilot | `benchmarks/fleet/results/` |
| The two scale-transfer registrations | `docs/PREREG_scale_transfer.md`, `docs/PREREG_scale_transfer_small.md` |
| Per-server partial results, the shared volume (bootstrap shard, coarse quantizer, cell census), non-member partials and index checksums | Zenodo, DOI 10.5281/zenodo.23608434 (too large for git) |
| This repository at the submission commit | Zenodo, DOI 10.5281/zenodo.23608470 |

The corpus is not stored anywhere. Every shard is a seeded function of its global index,
`gen_block(g)` in `benchmarks/fleet/fleet_common.py`, and any job can regenerate it.

## Rebuilding the paper

```
cd paper/pvldb1t
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The figure data is already committed. To recompute it from the record, run these from the
repository root (Python 3 and NumPy):

```
python paper/pvldb1t/figdata/fig_data.py
python paper/pvldb1t/figdata/transfer_fits.py benchmarks/fleet/record/1t/post > paper/pvldb1t/figdata/transfer_fits.txt
python paper/pvldb1t/figdata/cost_1trg.py benchmarks/fleet/record/1trg > paper/pvldb1t/figdata/cost_1trg.txt
```

`fig_data.py` reads `record/1t/post/` and writes the `.dat` files the figures plot and
`stats.json`, which holds the statistics the prose quotes.

## Tables and figures and their sources

Record files are in `benchmarks/fleet/record/1t/post/` unless another folder is given. Scripts
are in `benchmarks/fleet/` unless another folder is given. Each log's result is its
`RESULT_JSON`, `ANALYSIS_JSON`, `CELLS_JSON` or `INTERIM_JSON` line.

| item | record | script |
|---|---|---|
| Table 1, phases | `driver1t_post.log`, `driver1t_probe.log` | `fleet_qcache.py` (query cache), `fleet_ref.py` (exact scan), `fleet_ivf.py` (routed passes), `fleet_score10.py` (score), `fleet_cellhist.py` and `fleet_cellmerge.py` (census), driven by `driver1t_post.py` |
| Table 2 and Figure 2, recall by corpus size | `results/fleet_run_100M.json`, `fleet_run_1B.json`, `fleet_run_10B.json`, `fleet_run_100B.json`; `score_1T.log` at 10^12; `interim_score.log` for the triangles | the 10^9 run was scored by its coordinator, `fleet_coord.py`, as it served the queries; the batch runs by `fleet_score10.py` |
| Table 3 and Figure 3, nested series | `nested_1T.log`, `nested_1Tnm.log` | `fleet_nested.py`; plotted from `figdata/nested.dat`, `nested_nm.dat` |
| Table 4, registered predictions | `scale_transfer_predict_1tnm.json`, `scale_transfer_small_predict_1tnm.json`, grades `scale_transfer_grade_1tnm.json`, `scale_transfer_small_grade_1tnm.json` | `scale_transfer.py`, `scale_transfer_small.py`; the fitted laws by `figdata/transfer_fits.py` (output `transfer_fits.txt`) |
| Table 5 and Figure 4, recall from the probe order | `analysis1t_cells.log` (predicted), `score_1T.log` and `probe_1T.log` (measured) | `fleet_cellhist.py`, `fleet_cellmerge.py`; plotted from `figdata/reach.dat`; identity in `lean/Pvldb1t/Reachability.lean` |
| Section 4.4, score agreement and the tenth-to-eleventh margin | `audit_1t.log`, `audit_1tnm.log`, `span_1t.log`, `span_1tnm.log` | `fleet_partials_audit.py`, `fleet_boundary_span.py` |
| Table 6, cell order at equal work | `order_control_1T.log` | `fleet_order_control.py` |
| Section 4.6, query placement and non-member queries | `analysis1t_partials.log`, `analysis1tnm_partials.log`, `score_1Tnm.log` | `fleet_partials_analysis.py`, `fleet_nonmember_analysis.py` |
| Section 4.7, two hundred queries on the regenerated subset | `record/1trg/analysis1trg_final.json`, `untried1trg_final.json` | `fleet_qcache_regen.py`, `fleet_regen_score.py`, `fleet_regen_merge.py` |
| Table 7, work per query | `analysis1t_work.log` | `fleet_work_analysis.py` |
| Table 8, what one job used | `pilot_ref_exempt.log` (exact scan), `pilot_ivf_exempt.log` (routed pass); build wall times from the six rebuild logs `record/1t/tqp-fleet-1t-build-*.log` | the build row's memory is the measured pod noted in `drop_page_cache` in `fleet_common.py` |
| Table 9 and Figure 5, the pool | `driver1t_post.log`, `driver1t_probe.log` (one line per submission, completion and recycle) | `figdata/fig_data.py` (`inflight_*.dat`). The split of the 148 failed jobs by cause is not in these logs |
| Table 10 and Section 6, cost | `score_1T.log`, `probe_1T.log` (per-server search times), `score_1Tnm.log`; `record/1trg/` for the regenerated run | `figdata/fig_data.py` (`stats.json`); `figdata/cost_1trg.py` (`cost_1trg.txt`) |
| Table 12, rerank bound | `rerank_1T.log` | `fleet_rerank_prep.py`, `fleet_rerank_gen.py`, `fleet_rerank_score.py`; bound in `lean/Pvldb1t/RerankBound.lean` |
| Section 7, true recall at 10^9 and the real-data pilot | `results/fleet_run_1B.json` (`true_recall`), `results/real_pilot_15M_wiki1024.json` | `fleet_gt.py`, `real_pilot.py` |
| Section 2, intrinsic dimension and hubness of the corpus | `results/fleet_corpus_geometry.json` | |

Figure 1 is a diagram and Table 11 lists threats to validity. Neither is computed from data.
