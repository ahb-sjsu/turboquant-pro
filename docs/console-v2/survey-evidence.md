# Console redesign survey: evidence, observer, planning, tooling

Read-only survey of turboquant-pro **origin/master @ 24d49ac** (fetched 2026-09-26; extracted with
`git archive` into `scratchpad/om/`, nothing in `C:\source\turboquant-pro` touched). Note:
`C:\source\tqp-master-wt` is checked out at `c8d7d2b` on another line and has no `console/`,
`telemetry/` or `schemas/`, so it is NOT a copy of master.

Requirement ids are from `readscope_req.txt` (Concept v0.1). Status marks: EXISTS = shipped on
master, PARTIAL, NONE = not built. Effort: S (hours, one file), M (a day or two, a few files),
L (new subsystem).

---

## 0. Headline findings (read these first)

1. **Every evidence artifact already exists and is JSON with a `schema` id**, but only 11 of
   roughly 30 artifact kinds have a JSON Schema file in `turboquant_pro/schemas/`. There is no
   `schema id -> schema file` registry. An Artifacts workspace needs one (S).
2. **Almost no artifact records the command that produced it.** Only `embedding-plan` and
   `kv-plan` carry a `reproduction` string, and neither reproduces the artifact itself (the
   embedding plan's string is a follow-up `tqp certify` with a `<recon>.npy` placeholder; the KV
   plan's is a Python call). All 24 CLI emitters go through one function, `cli._emit_doc`
   (cli.py:477), so stamping an `invocation` block (argv, cwd, tool_version, git commit) in one
   place is S effort and unlocks "reproduce with:" on every panel (BM-005).
3. **The console's certificate status is never computed.** `console/server.py:356` passes the
   certificate's issue-time `validity` section (sketches and thresholds, which has no `status`
   key) to the UI, and both `tui.py:343` and `app.js:219` fall back to the string `UNCHECKED`.
   The status shown is technically true (nothing checked it) but the console holds the data to
   check it (index originals, recent queries) and does not call `validity.check_validity`. Also
   the UIs know only three states; the library has four (`VALID / STALE / INCONCLUSIVE /
   UNCHECKED`, validity.py:59-60).
4. **`index-certificate` is a second-class certificate.** `tqp index certify` (cli.py:2814)
   emits no `created_utc`, no input sha256, no observer, validity or environment, so `compose`,
   `capabilities` and `verify` cannot use it (they require `turboquant-pro/rank-certificate`).
5. **The canonical benchmark bundle is missing its results file.**
   `benchmarks/artifacts/embedding_glove_recall/MANIFEST.md` cites `results.json` for every
   number, but the bundle holds only MANIFEST.md, certificate.json, command.sh,
   environment.json and git_commit.txt. It is the only bundle; there is no run registry.
6. **`agent_tools.best_compression_at_recall` emits a reproduce string that does not run**:
   `tqp plan embeddings --target '...'` without the required `--embeddings` (agent_tools.py:154).
7. **Only 2 of 17 claims in `claims.yaml` are executable** (6 experimental, 4 reproducible,
   3 reported, 1 partial, 1 retracted). A claims panel must show status, not imply every
   claim replays.
8. `docs/DESIGN_console.md` cites the requirements as
   `docs/notes/TurboQuantPro_ReadScope_Visual_Interface_Requirements.docx`, which is not in the
   repo, and its header still says the work is on `feat/console-phase0` although it is on master.

---

## 1. Artifact inventory (what the Artifacts workspace must open)

| schema id (`schema` field) | emitted by | JSON Schema file | key fields | records its command? |
|---|---|---|---|---|
| `turboquant-pro/observer-contract` (profile `tqp-observer/1`, file `.tqo`, YAML or JSON) | `tqp observer init/learn`, hand-written | observer_contract | observer, target, source, consumers[metric, config, weight, name], population, requirements (floor, confidence, worst_stratum_minimum), budget, fallback; identity = sha256 of canonical JSON | no (learn writes `source` block: request count, ts span, share) |
| `turboquant-pro/rank-certificate` | `tqp certify`, plan-embeddings preview | rank_certificate | inputs.{original,reconstructed}.{path,shape,dtype,sha256}, params{metric,n_anchors,seed}, certificate{kappa, mu_hat, tau_floor, spearman_floor, n_pairs, max_certifiable_kappa, vacuous}, interpretation, passed; optional task, environment, limitations, observer, reference{provider, operator_sha256, effective_rank, consumer_distortion, reconstruction_distortion}, validity{issued_for, thresholds, operator_sketch, coverage_sketch, strata_sketch} | no, but fully reconstructable from inputs.path + params |
| `turboquant-pro/verification` | `tqp verify` | none | certificate{path,schema}, checks, verified, validity{status, applicable, reason, action, checks{source_artifact, observer_contract, operator_overlap, data_coverage, strata_coverage}}, applicable | no |
| `turboquant-pro/index-certificate` | `tqp index certify` | none | index, metric, n_live, sample, certificate{...}, passed (no timestamp, no hashes) | no |
| `turboquant-pro/composition-certificate` | `tqp compose` | none | stages[certificate path, kappa, tau_floor, original_sha256, reconstructed_sha256], product_kappa, mu_hat, tau_floor, spearman_floor, vacuous, unconditional, weakest_stage, passed, interpretation | no (stage paths recorded) |
| `turboquant-pro/capability-report` | `tqp capabilities` | none | artifact, artifact_sha256, certified[], conditional[], not_certified[] (each name, status, certificate, reason, observer_sha256, detail), certificates_for_other_artifacts | no |
| `turboquant-pro/feasibility-report` | `tqp feasibility` | none | result PASS/INFEASIBLE/ABSTAIN, reason, action, warnings, source{rows,dim,rank}, observer, observable{observable_rank, unread_dimensions, omitted_sensitivity_fraction}, attainable{floor_distortion_fraction, target_fraction, min_bytes, curve[fraction, bytes]}, rank{min_tau, max_certifiable_kappa}, identification | no |
| `turboquant-pro/compression-plan` | `tqp plan run` | compression_plan | workload, artifact{identity}, target, consumer, operator_regime, preflight{flags, spectrum, norm_spread}, candidate_results[codec, config, verdict, left_at, reason, source, tier, declared_bits, bytes_per_vector, quality{mean, bound, ci_low, ci_high, n_items, confidence}, holdout_quality, nominal, false_clear, certificate], selected_codec (or ABSTAIN), selection{rule, reason, beat{codec, margin}, decided_on, not_decided_on}, certificate_requirement{status}, fallback_policy{action, thresholds, triggers}, evidence[kind, claim, measurement, population, method], observer, environment | no, but `workload` holds every spec field |
| `turboquant-pro/compression-plan-replay` | `tqp plan replay` | none | artifact_identity_matches, recorded_codec, replayed_codec, codec_agrees, recorded_quality, replayed_quality, delta_mean, reproduced | no |
| `turboquant-pro/embedding-plan` | `tqp plan embeddings` | embedding_plan | input, constraints, recommended, certificate_preview, alternatives (Pareto), risk_flags, passed | `reproduction` (a follow-up certify, not this plan) |
| `turboquant-pro/kv-plan` | `tqp plan kv` | kv_plan | request, policy, key_zero_point, risk_flags | `reproduction` (Python API) |
| `tqp.weight_plan/1` | `tqp plan weights` | weight_plan | model, predictor, cost_table_hash, budget_bits, stored_bits, cost, dual_bound, duality_gap, lattice_states, bits{matrix: b} | no |
| `tqp.weight_cost_table/1` (input) | produced by the weights campaign | weight_cost_table | model, predictor, matrices, costs[name][bits], provenance | n/a |
| `turboquant-pro/refinement-report` | `tqp plan refine` | none | base, refined, dim, overlap, alone, progressive, flat, separate, tax, tax_threshold, verdict, reason | no |
| `turboquant-pro/compatibility-matrix` | `tqp plan compat` | none | labels, budget_bytes, own, distortion[built][reader], ratio[built][reader], safe[built][reader], overlap, safe_ratio | no |
| `turboquant-pro/workload-summary` | `tqp observer learn --summary` | none | lines, parsed, unparseable, distinct_readers, min_count, retained[metric, config, count, weight], abstained, unregistered, ts span | no |
| `turboquant-pro/replay-report` | `tqp replay` | none | claims[id, command, verdict reproduced/regressed/error/manual/dry_run, measured, checks, drift_class, duration_s, stderr_tail], summary | YES per claim (`command`) |
| `turboquant-pro/adaptive-policy` | library only (`adaptive_rerank.calibrate`) | adaptive_policy | guarantee, k, band, epsilon, target_recall, calibration{n_queries, risk_bound, recall, rows_read, scan_fraction}, index fingerprint | no |
| `turboquant-pro/query-catalog` / `query-plan` / `query-result` | `tqp query ANALYZE / EXPLAIN / SELECT` | none | catalog: geometry (advisory), calibration sweep, planner_basis; plan: statement, plan{rerank, chosen_by, predicted, target_unreachable}, honesty; result: plan, n_queries, mean_latency_us, results | the statement itself is in the doc |
| `turboquant-pro/query-trace` | telemetry tracer (console, library) | query_trace | id, component, input{sha256, shape}, params, index, observer, scan_path, stages[name, ms, candidates, score range], results{approx, final, rerank_agreement, rank movement}, total_ms, sampled | n/a (live) |
| metric reading (no `schema` field) | `telemetry.metrics.reading` | metric_reading | name, unit, aggregation, window_s, source, interval_s, kind measured/estimated/sampled/derived, value, n, as_of, unavailable_reason | n/a |
| `turboquant-pro/console-setup` (`.tqs`) | console key `S` | console_setup | observer_sha256, view, scope{channels, s_per_div, trigger, acquire, decay, masks}, analyzer{...} | n/a |
| `turboquant-pro/console-export` | console key `e`, `/api/export` | none | capabilities, snapshot, readscope, traces[<=200], replays | no |
| `turboquant-pro/index-shards` (manifest.json) | sharded `tqp index create` | none | shard list, shared basis | no |
| `tqp-strata-report/1` | `tqp anatomy --strata`, hubdiff with strata | none | per-area anatomy, ABSTAIN areas | no |
| (no schema field) anatomy | `tqp anatomy` | none | battery, k, count_skew, top_hub_mass_share, robin_hood_index, corr_count_centrality, corr_count_neg_dk, mechanism, prescription, fingerprints | no |
| (no schema field) hubdiff | `tqp hubdiff` | none | recall_at_k, p05 per-query recall, hub_rank_corr, hub_set_jaccard, anti_hub_recall, recall_by_count_decile | no |
| (flat dict, Prometheus names) monitor | `tqp monitor` | none | turboquant_quality_* (mean/min/std cosine, is_healthy, drift, median_tangential_fraction, radial_drift) and turboquant_certificate_{valid, stale, operator_overlap, coverage_divergence, uncovered_fraction} | no |
| probe JSON (no schema field) | `tqp probe --json` | none | consumer, bits, spearman per family, recommendation | no |
| `tqp-kv-identity/1`, `tqp-kv-store-state/2` | vLLM connector | none | identity profile, store state | outside this scope |
| text only | `tqp version`, `plugin list`, `plugin conformance`, `plan consumers`, `plan explain`, `trace` | none | printed tables | n/a |

Schema-less artifacts (anatomy, hubdiff, monitor, probe) must be detected by field shape, which
is fragile. Adding a `schema` field to each is S.

---

## 2. Feature rows

Columns: **what** / **artifact (exists?)** / **live or artifact** / **best visualization** /
**workspace, requirement ids** / **missing to feed it, effort** / **honesty notes**.

### 2.1 Observer (ReadScope core)

**Observer contract** (`observer.py`, `tqp observer validate|show|hash|init`, schema observer_contract)
- What: declares who reads the data (weighted consumer metrics or a `read_operator` provider), the population, requirements, budget, fallback; content-addressed by sha256.
- Artifact: `.tqo` EXISTS.
- Live/artifact: artifact (bound to the live session with `--observer`).
- Visualization: **observer card**: name, target, short sha256 with copy, consumer weight bar (stacked, one segment per consumer, labelled metric+config; primary one marked "the planner reads only this one"), requirements row (floor, confidence, worst-stratum minimum with a "recorded, not enforced" chip), budget, fallback action chip; `validate` problems listed in red beneath. A persistent observer badge in the top bar on every workspace (the requirements' "observer-aware" principle).
- Workspace/ids: ReadScope, RS-001; Overview OV-004 (badge); API-001.
- Missing: nothing to display. Run `ObserverContract.validate()` server-side so the card shows registry problems (S).
- Honesty: Phase 1 planner uses only the primary (largest-weight) consumer; the card must say so. `worst_stratum_minimum` is validated but not enforced (DESIGN_observer_contracts section 3).

**Learned observer from traffic** (`workload.py`, `tqp observer learn TRACE --summary S.json`)
- What: writes the contract the traffic performed; abstains on readers seen < `--min-count`; never maps unknown metrics.
- Artifact: contract + `workload-summary` EXISTS.
- Live/artifact: artifact (could be live if the console fed its own traces in).
- Visualization: **reader census**: horizontal bars of retained readers by count with weight, a separate greyed "abstained (n < min_count)" group with counts, an "unregistered metric" group in amber, and a coverage strip: lines read, unparseable, retained share of traffic, timestamp span ("learned from 3 h 12 min of traffic").
- Workspace/ids: ReadScope RS-001, RS-002.
- Missing: the console does not read JSONL request traces; showing it is S from the summary file.
- Honesty: a contract learned from an hour is a different claim than one from a month; the span must be as prominent as the weights.

**Observer compare** (two `.tqo` files; `spectrum.Analyzer.store_reference` + `drift`)
- What: compare two observers direction by direction.
- Artifact: none directly; spectrum reference is in-memory (live).
- Visualization: side-by-side observer cards with field diff (changed fields highlighted, hashes differ), plus the spectrum analyzer in delta mode (below).
- Workspace/ids: ReadScope RS-003.
- Missing: a contract diff function (S); replay of a trace under another observer is NOT built (DESIGN_console section 4 "Not yet") (M).
- Honesty: spectrum drift ||P - P_ref|| / ||P_ref|| is computed on the reference's directions (Rayleigh quotients); say so.

### 2.2 Certificates and their validity

**Rank certificate** (`rank_certificate.py`, `certify_report.py`, `tqp certify`, schema rank_certificate)
- What: distribution-free floors Kendall tau >= 1 - 2 mu_hat(kappa), Spearman >= 1 - 3 mu_hat, from sampled anchor pairs.
- Artifact: `certificate.json` EXISTS; `--html` report EXISTS.
- Live/artifact: artifact.
- Visualization: **certificate card**: PASS / FAIL / VACUOUS chip; tau floor and Spearman floor as bars on [-1, 1] with the `--min-tau` gate line; kappa vs `max_certifiable_kappa` as a gauge (kappa past the max = "no finite distortion certifies rank, exact rerank required"); robust vs strict regime chip ("conditional: central 95% of pairs" when robust); n_pairs, anchors, seed; input sha256 chain (original -> reconstructed, short hashes, copyable); optional task, limitations list, environment block, observer sha256 (linked to the observer card), reference provider with operator sha256, effective rank, consumer vs reconstruction distortion. Add a **mu(kappa) curve** (from `mu_curve`) with the measured kappa marked, which shows why a certificate is vacuous.
- Workspace/ids: ReadScope RS-005; Artifacts; Overview OV-004.
- Missing: mu curve is not stored in the certificate (recompute needs the data, or add `mu_curve` samples to the doc) (S). Regime (lo/hi percentiles) is not recorded in `params` (S).
- Honesty: the default robust kappa makes the floor conditional; the floors are worst-case and "typically sit far below measured recall" (module docstring). The GloVe bundle's tau floor 0.045 is weak by design; never color a weak-but-positive floor green without the gate context.

**Certificate validity / expiry** (`validity.py`, `tqp verify --data --queries --observer`, `monitor.QualityMonitor(certificate=...)`)
- What: decides whether a certificate still applies: VALID / STALE / INCONCLUSIVE / UNCHECKED, with reason and action REPLAN or RECERTIFY.
- Artifact: `verification` doc EXISTS (`validity` block); monitor exposes `turboquant_certificate_*` gauges EXISTS.
- Live/artifact: both (monitor re-checks every `validity_every` records).
- Visualization: **validity checklist**: five rows (source artifact unchanged, observer contract unchanged, observer read geometry, data within calibration coverage, strata coverage), each ok / FAIL / abstain / not checked with its number against its bar (overlap 0.91 vs min 0.85; divergence vs noise floor vs max; uncovered fraction with Wilson interval vs limit, and which areas the uncovered rows came from and why). Status chip with four states, reason line, action chip. Live: a status timeline strip (VALID/STALE/INCONCLUSIVE over time) and overlap/divergence sparklines.
- Workspace/ids: ReadScope RS-005, RS-007; Overview OV-004, OV-005 (STALE transitions as events).
- Missing: the console never calls `check_validity` (finding 3). Wire it against the workload's originals and recent queries, periodically, like the monitor does (S-M). Add INCONCLUSIVE to the UI vocabulary (S).
- Honesty: STALE is "no longer applicable", not "false"; INCONCLUSIVE means too few rows to decide, not pass. Thresholds are recorded at issue so they cannot be moved after the fact; show them as recorded.

**Verify (reproduction check)** (`tqp verify CERT --original --reconstructed`)
- What: schema/self-consistency always; recompute with recorded params and compare within atol/rtol; hashes re-checked.
- Artifact: `verification` EXISTS.
- Visualization: **two-column recorded vs recomputed table** (kappa, mu_hat, tau_floor, spearman_floor, sha256 original, sha256 reconstructed) with match ticks and the tolerance; schema problems listed.
- Workspace/ids: ReadScope RS-005, RS-006; NFR-009.
- Missing: nothing; the console does not run it (S to add a "verify" action on a loaded certificate when inputs are available).
- Honesty: "verified" and "applicable" are separate verdicts; show both.

**Index certificate** (`tqp index certify`)
- Visualization: same card, degraded, with a banner "no input hashes: cannot be composed or matched to an artifact".
- Workspace/ids: Index & Shards IS-004; ReadScope RS-005.
- Missing: bring it to rank-certificate parity (created_utc, inputs sha256, environment) (S).

**Pipeline composition** (`composition.py`, `tqp compose`)
- What: certifies a chain: checks each stage's reconstructed sha256 equals the next stage's original sha256; floor at the product of kappas on the source distances.
- Artifact: `composition-certificate` EXISTS.
- Visualization: **left-to-right provenance chain**: nodes are arrays (short sha256), edges are stage certificates labelled kappa and tau floor; the weakest stage edge highlighted; a broken link drawn red with the two mismatching hashes; end node shows product kappa, chain tau floor, and a "conditional (robust kappas)" or "unconditional (strict)" chip.
- Workspace/ids: ReadScope RS-002, RS-005, RS-007.
- Missing: nothing to render; composition refuses a broken chain with exit 2 and emits no doc, so the broken-link view needs the error captured as a doc (S).
- Honesty: a robust composition is never unconditional; keep the caveat on the node, not in a tooltip.

**Capabilities** (`capabilities.py`, `tqp capabilities`)
- What: what this artifact is currently certified for: certified / conditional / not certified, with reasons; certificates about other artifacts listed and not counted.
- Artifact: `capability-report` EXISTS.
- Visualization: **three-lane capability board**: green lane (certified), amber lane (conditional, each card with its reason and "what would make it certified"), grey lane (not certified, "certify next"), plus a collapsed "certificates about a different artifact (not counted)" drawer. Artifact sha256 at the top.
- Workspace/ids: ReadScope RS-005, RS-007; Index & Shards IS-004.
- Missing: nothing; the console could compute it live for the loaded index (M, needs the artifact array to hash).
- Honesty: an UNCHECKED certificate is conditional, not certified; only VALID certifies.

**Feasibility** (`feasibility.py`, `tqp feasibility`)
- What: before planning: is the guarantee attainable; observable rank; unread dimensions; omitted sensitivity (omission floor); fewest bytes/vector to reach a distortion.
- Artifact: `feasibility-report` EXISTS.
- Visualization: **verdict banner** (PASS / INFEASIBLE / ABSTAIN with reason and action) over a **rate-distortion curve** (bytes/vector vs distortion fraction from `attainable.curve`, the floor at max width as a horizontal asymptote, target line, min_bytes marked), and a **sensitivity budget bar**: of tr(P), how much is observable, unread, omitted. Rank section: required kappa vs max_certifiable_kappa.
- Workspace/ids: ReadScope RS-004 (the natural "budget" view); Benchmarks BM-004.
- Missing: nothing to render; the per-direction data is not in the doc (only aggregates), so a per-direction chart must come from the spectrum analyzer instead.
- Honesty: a recall target is refused on purpose (no distribution-free conversion); ABSTAIN when the operator is an estimate below full dimension (readscope recovery cliff). The distortion model is the Lloyd-Max table, a prediction.

### 2.3 Planning

**Plan run / control plane** (`planner.py`, `tqp plan run`, schema compression_plan)
- What: enumerate codecs from the plugin registry, prune on byte priors, successive halving on the consumer's metric with bootstrap bounds, keep a frontier, verify once on held-out, emit a record; ABSTAIN is a legal answer.
- Artifact: `compression-plan` EXISTS.
- Visualization: it is a **funnel, not a decision tree**. (a) **Candidate funnel**: columns enumeration -> prior -> halving round 1..n -> frontier -> verification; each candidate is a row that ends at its `left_at` column with its verdict chip (selected / candidate / reject / infeasible / unsupported) and reason on hover. (b) **Frontier scatter**: x = stored bytes/vector, y = consumer metric, each candidate an interval (ci_low..ci_high) with the conservative bound marked, floor as a horizontal line, budget as a vertical line, selected highlighted, runner-up labelled with margin; candidates with a false clear drawn with a warning ring. (c) **Evidence list** grouped by kind (statistical, certificate, measured_cost, diagnostic, comparison) with claim, population, method. (d) Fallback card: action, thresholds, triggers. (e) Preflight flags as amber chips.
- Workspace/ids: new Plans view inside ReadScope or Benchmarks; BM-002, BM-004, RS-008 (explain), OV-006 (fallback mode).
- Missing: nothing to render the record. `measured_cost` covers bytes only, not time (DESIGN_planner "Not yet done").
- Honesty: "decided_on" is the consumer metric; "not_decided_on: reconstruction cosine (diagnostic only)" should be visible. The planner has not passed its P0 exit (regret vs the RaBitQ grid); label "working, not validated" per DESIGN_planner.

**Plan explain** (`CompressionPlan.explain`, `tqp plan explain RECORD`)
- What: text rendering of the record.
- Visualization: the same funnel + scatter above; keep the text as the "plain-language" pane of RS-008, clearly marked as rendered from the record (computed facts), not generated.
- Workspace/ids: RS-008.
- Missing: nothing.

**Plan replay** (`planner.replay_plan`, `tqp plan replay`)
- What: re-runs verification; reports artifact identity match, codec agreement, delta in the held-out mean.
- Artifact: `compression-plan-replay` EXISTS.
- Visualization: **replay verdict row**: three ticks (same bytes, same codec, same quality within 1e-9), recorded vs replayed quality intervals drawn on one axis, delta_mean.
- Workspace/ids: ReadScope RS-006; NFR-009.
- Missing: nothing. Not currently in the console.
- Honesty: a replay on different bytes reports mismatch rather than reproduction.

**Plan consumers** (`consumers.py`, `tqp plan consumers`)
- Visualization: registry table: consumer name, targets, exact vs proxy, evidence kind (certificate / statistical / analytic), description; codecs per target.
- Workspace/ids: System (capabilities), API-001.
- Missing: text output only; needs `--format json` (S).

**Plan embeddings** (`auto_compress.py`, `tqp plan embeddings`)
- What: sweep PCA dim x bits x rotation; Pareto on measured recall@k; certificate preview.
- Artifact: `embedding-plan` EXISTS.
- Visualization: **Pareto plot**: bytes/vector vs recall@k, alternatives as points, recommended highlighted, certificate preview card beside it, risk flags.
- Workspace/ids: Benchmarks BM-004.
- Honesty: cosine is only a labelled diagnostic; a vacuous preview means exact rerank is required.

**Plan KV** (`autoconfig`, `tqp plan kv`)
- Visualization: key/value bit policy table per layer/head geometry, cache size and ratio, risk flags (keys below 4 bits).
- Workspace/ids: Benchmarks or System; low priority for ReadScope.

**Plan weights** (`weight_plan.py`, `tqp plan weights`)
- What: exact multiple-choice knapsack of bits per matrix, with Lagrangian dual bound.
- Artifact: `tqp.weight_plan/1` EXISTS.
- Visualization: **per-matrix bit strip** (matrices in model order, colored by bits) and a budget gauge (stored_bits vs budget_bits), cost vs dual bound with duality gap.
- Workspace/ids: Benchmarks BM-004.
- Honesty: exact solver; any deficit belongs to the cost model (the predictor), which is named in the doc.

**Refinement** (`refinement.py`, `tqp plan refine`)
- What: can a base code for observer A be refined for observer B without paying twice; refinement tax.
- Artifact: `refinement-report` EXISTS.
- Visualization: **bytes comparison bars**: A alone, B alone, progressive (base + layer), flat (one code meeting both), separate (two codes); tax as a percentage vs threshold; operator overlap; verdict.
- Workspace/ids: ReadScope RS-003, RS-004.
- Honesty: a prediction from the Lloyd-Max table; nothing is built; the layered container is phase 2, not built.

**Cross-observer compatibility** (`refinement.compatibility_matrix`, `tqp plan compat`)
- Artifact: `compatibility-matrix` EXISTS.
- Visualization: **heatmap**, rows = observer the code was built for, columns = reader, cell = ratio vs the reader's own code, red where ratio > safe_ratio; diagonal = 1; overlap as a secondary matrix.
- Workspace/ids: ReadScope RS-003, RS-007.
- Honesty: prediction, not measured recall.

**Query planner** (`query.py`, `tqp query ANALYZE / EXPLAIN / SELECT`)
- Visualization: catalog: recall vs rerank calibration curve with the chosen operating point and target line; `target_unreachable` banner; EXPLAIN shows `chosen_by` provenance.
- Workspace/ids: Queries QI-004 / Benchmarks BM-004 (likely overlaps another surveyor).

### 2.4 Diagnostics

**False-clear rate** (`false_clear.py`; embedded per candidate in plan records)
- What: how often cheap cosine cleared a result the consumer rejected (harmful direction) vs conservative miss.
- Artifact: inside `compression-plan.candidate_results[].false_clear`; standalone library only, no CLI.
- Visualization: **2x2 confusion tile** (nominal cleared / flagged x consumer pass / fail) with the false-clear cell emphasized, P(consumer fails | cleared) against warn 0.05 / fail 0.20 lines.
- Workspace/ids: ReadScope; Benchmarks BM-003.
- Honesty: thresholds are conventions, not measurements (module docstring).

**Calibration coverage** (`calibration_coverage.py`)
- What: Jeffreys divergence etc. between calibration set and serving data.
- Artifact: `CoverageReport.to_dict` (no schema, no CLI).
- Visualization: gauge in nats with warn 10 / fail 50, plus mean shift and worst-direction variance ratio.
- Missing: CLI or console hook (S).
- Honesty: thresholds are conventions; the cited 25-50 nat knee is from a different task.

**(A2) probe** (`a2_probe.py`, `tqp probe`)
- Visualization: two bars (Spearman polar vs per_channel), recommendation, median tangential fraction.
- Workspace/ids: ReadScope (consumer fit) or Benchmarks.

**Quality monitor** (`monitor.py`, `tqp monitor`)
- Live: yes (library), artifact via Prometheus text/JSON.
- Visualization: health lamp gated on BOTH cosine floor and (A2) noncollapse, tangential fraction sparkline, radial drift flag, plus the certificate gauges above.
- Workspace/ids: Overview OV-004, OV-005.
- Missing: the console does not host a QualityMonitor (M).
- Honesty: cosine is the base signal here only because it is guarded by the tangential statistics.

**Operator trace** (`operator_trace.py`, `tqp trace`): regime and discipline per tensor. Table/treemap by regime. Likely another surveyor's scope; listed for completeness.

### 2.5 Claims, benchmarks, runs

**Claims ledger** (`claims.yaml`, `tqp replay --list`)
- What: 17 claims, status vocabulary executable / reproducible / needs-local-run / partial / experimental / reported / retracted.
- Visualization: **claims table**: id, track, status chip, dataset, hardware, expected floors, last replay verdict and date, drift class on failure; retracted row kept visible and struck through; filter by status.
- Workspace/ids: Benchmarks BM-001, BM-003, BM-005.
- Missing: last-run results are not persisted anywhere; the table needs a replay-report store (S: write each `replay-report` into a runs directory).
- Honesty: only 2 of 17 are executable; the table must not imply the rest replay.

**Claim replay** (`tqp replay <claim|all>`)
- Artifact: `replay-report` EXISTS.
- Visualization: per-claim result rows: measured value vs expected min/max as a bullet chart, verdict chip, duration, stderr tail on error.
- Workspace/ids: Benchmarks BM-003, BM-005.
- Honesty: `command` runs through the shell; the console must never run replay from the browser (NFR-005). Show the command; run it in the TUI only on explicit confirmation, or not at all.

**Benchmark artifact bundle** (`benchmarks/artifacts/<claim>/`)
- Visualization: bundle card: claim, command.sh, git commit, environment, certificate card, results.
- Missing: `results.json` missing from the only bundle (finding 5); no run registry of any kind (M: a `runs/` index of bundles keyed by claim, commit, hardware).

### 2.6 Registries and tooling

**Version** (`tqp version`): one string. Show in the System header with `telemetry.capabilities()` (API-008). No chart.

**Plugin registry** (`plugins.py`, `tqp plugin list`): name, tier, targets, description (source in-tree vs entry point). Table in System. Missing JSON output (S).

**Plugin conformance** (`plugin_conformance.py`, `tqp plugin conformance`): per plugin x check (roundtrip, packed, affine, csr, serialization, ...) pass / skip-with-reason / FAIL. **Matrix grid** plugins x checks. System / Index IS-004. Missing JSON output (S).

**Read-operator registry + conformance** (`read_operators.py`, `read_operator_conformance.py`): providers (identity, ...), exact flag, requires; conformance incl. `non_degenerate`. Same grid. Library only, no CLI (S to expose).

**Adaptive rerank policy** (`adaptive_rerank.py`, schema adaptive_policy): calibrated epsilon band with a recall guarantee. Visualization: recall vs rows-read curve with the calibrated epsilon marked; per-query stage share (scan-only vs rerank). Library only, no CLI; the console workload does not use it.

**Agent tools** (`agent_tools.py`): three JSON tools (best_compression_at_recall, certify_ranking, recommend_kv_key_quantizer) + `list_tools()` schemas. Visualization: a tool catalog in System (name, params, what it certifies against). No live data. Honesty: reproduce_cli bug (finding 6).

### 2.7 Telemetry (the live contract)

**Metric registry and readings** (`telemetry/metrics.py`): 12 registered metrics: search.qps, search.latency_ms.p50/p95/p99, search.stage_ms.encode/scan/rerank, search.rerank_agreement (derived), index.compression_ratio (derived), index.rows, process.cpu_percent (sampled), process.rss_mb. Each carries unit, aggregation, window, source, kind. Visualization: every KPI tile shows unit, window and a kind glyph (measured / sampled / derived / estimated) and a freshness dot; unavailable shows the reason, never zero. Overview OV-001, OV-002, UX-005, NFR-004. Missing: GPU utilization is declared false in capabilities; no disk/network/queue metrics (OV-002 gap).

**Query trace** (`telemetry/trace.py`): see console section. Queries QI-001, QI-002. Missing: stage spans inside IVFIndex and ShardedIndex (DESIGN_console section 3).

**Capabilities / version** (`telemetry/api.py`): API_VERSION 0.1, schemas list (only query-trace and metric-reading), features flags. System, API-008. Missing: the schemas list should enumerate every artifact schema the console can open (S).

### 2.8 Declared but not built (mark as such in the UI, or omit)

| feature | where declared | status |
|---|---|---|
| Replay under a different configuration or observer; arbitrary two-trace compare | DESIGN_console section 4 | not built (QI-003, QI-004 second half, RS-003) |
| Entity schemas Session, Runtime, Shard, Event, ExperimentRun | DESIGN_console section 3 | not built |
| Value-of-observation planner (#181) | DESIGN_value_of_observation | "Not implemented" by design: needs measured seconds per resource |
| Progressive / layered refinement container (#174 phase 2) | DESIGN_progressive_codes section 3 | not built; `plan refine` is prediction only |
| Weighted mixture of consumers in the planner | DESIGN_observer_contracts section 3 | not built |
| Measured latency/throughput evidence in plans; transforms and search operators as planner stages; planner P0 exit test | DESIGN_planner "Not yet done" | not built |
| KV identity/observability metrics (hit rates, load latency, recompute fallbacks, compatibility misses, probe verdict + age) | ROADMAP_2.0 P1-M3 | designed |
| Break-even admission, region reported | ROADMAP_2.0 P1-M4, POSITIONING Moat 3 | designed |
| Recall contract (stored guarantee with staleness rules), `WITH (RECALL >= r PER AREA)` | ROADMAP_2.0, POSITIONING | designed |
| A2 verdict expiring automatically on config change (KV) | ROADMAP_2.0 P1-M6 | designed |
| Operator actions, auth scopes | DESIGN_console section 6 | out of scope in Phase 0-1 |

---

## 3. What the current console already visualizes (keep all of it)

Three views in the TUI, cycled with `v`; the TUI opens on the **scope**. The web view (`--web`) has the Overview only.

**Overview** (tui.py `frame`, web index.html panels 1-6), quality good:
- 1 System KPIs from registered readings (QPS, latency, CPU, RSS), with freshness and PAUSED state.
- 2 Throughput/latency sparklines (QPS, p95), last 120 s.
- 3 Pipeline: encode / scan / rerank mean ms bars, "top-k from k*r reranked" note.
- 4 ReadScope: observer name, target, sha256, consumers; certificate PASSED + tau floor + validity (always UNCHECKED, finding 3); provenance list of sha256 steps (source, observer, certificate inputs). Explicit text when no observer is loaded.
- 5 Index: kind, rows, dim, metric, bytes/row, kernel AVX2 vs numpy, workload.
- 6 Query stream table (time, trace id, row, scan path, total/encode/scan/rerank ms, agreement), selection and inspect.
- **Inspector overlay**: query row + sha256 + dim, params, scan path, observer, stage list with candidates, approximate -> exact results with rank movement; `r` **replay** (same query row, reports pinned query/index/params, nondeterminism list, diff); `e` export.

**Oscilloscope** (scope.py + scope_view.py), quality high, pure and tested:
- Channels latency, encode, scan, rerank, candidates, agree, move, tau (per-query Kendall tau approx vs exact), err; 1-2-5 scales; 10x8 graticule, braille waveforms.
- Trigger edge / pulse width / logic (including `scan_path == numpy`), holdoff, pre-trigger position; modes auto / normal / single; Run/Stop, Force.
- Acquisition sample / peak detect / average; phosphor persistence (decaying or infinite).
- Measurements with statistics (mean, min, max, pk-pk, sigma, p50/p95/p99); cursors; segmented memory (`h` steps records, Enter inspects the firing query); masks with pass/fail count and stop-on-fail; Autoset.
- `F`: Lomb-Scargle periodogram of a channel (irregular arrivals).
- Certificate tau floor drawn on the tau channel as a **reference, never judged** (correct: different population).

**Spectrum analyzer** (spectrum.py + spectrum_view.py), quality high:
- x = eigendirection of the read operator E[qq'] from the last 256 workload queries, y = dB; traces sens, var, weighted, predicted, noise; water level theta as limit line with pass/fail; max/min hold, averaging, detector; markers, peak, next peak, delta; waterfall (drift); gap realised vs predicted distortion in dB; `R` stores a reference and later sweeps show delta and drift ||P - P_ref||/||P_ref||.

**Setups** (setup.py, `.tqs`): full front-panel save/recall, validated before apply, warns on a different observer sha256.

**Export** (`e`): `console-export` JSON (capabilities, snapshot, readscope, 200 traces, replays).

Gaps in the current console relative to this survey: no plan, composition, capability, feasibility, refinement, compat, claims, or verify views; validity never evaluated; web view lacks scope and spectrum; no artifact opener.

---

## 4. Proposal: an "Artifacts" workspace

1. **Open**: file picker / drag-drop / `tqp console --open FILE...` / a watched directory. Read-only; parse JSON (and YAML for `.tqo`); never execute anything (claims `command` shown, not run).
2. **Identify**: dispatch on `schema` (and `profile` for `.tqo`); for schema-less docs fall back to field signatures (anatomy: `battery` + `count_skew`; hubdiff: `hub_set_jaccard`; monitor: `turboquant_quality_is_healthy`; probe: `spearman` per family). Show "unrecognized" with raw JSON rather than guessing.
3. **Validate**: if a JSON Schema exists, validate and show pass/fail with paths; if none exists say "no schema shipped for this type" (grey chip), not "valid".
4. **Render** with the renderer registered for that schema (section 2), always with a common header: schema id + version, tool_version, created_utc (or "no timestamp recorded"), file sha256, and **reproduce with** (section 5), marked *recorded* or *derived*.
5. **Link**: resolve sha256 references across open artifacts: certificate.inputs -> other certificates (composition chain), certificate.observer.sha256 -> open `.tqo`, capability-report.artifact_sha256 -> certificates, plan.observer -> contract, console-setup.observer_sha256 -> contract. Draw the resulting **provenance graph** (RS-002, RS-007): nodes = arrays / contracts / certificates / plans, edges = "issued over", "issued for", "composed from". Unresolved references shown as dangling hashes.
6. **Compare** (`c`): two artifacts of the same schema -> field diff (certificates: floors and hashes; plans: selected codec, bounds; replays: before/after) (RS-003, BM-002, API-005).
7. **Export** a bundle: selected artifacts + manifest with hashes (UX-008).

Needed plumbing: a schema registry `{schema id -> schema file, renderer, reproduce builder}` in `turboquant_pro/schemas/__init__.py` (S); JSON Schemas for the ~18 kinds without one (M); `schema` field on the four schema-less outputs (S); `invocation` block in `_emit_doc` (S).

---

## 5. "Reproduce with" per artifact

R = recorded in the artifact today; D = derivable from recorded fields (show as "derived"); X = cannot be reconstructed from the artifact alone.

| artifact | command | R/D/X |
|---|---|---|
| observer contract | `tqp observer init ...` or `tqp observer learn TRACE --name N --out F [--min-count M]` | X (learned contracts keep counts and span, not the trace path) |
| rank-certificate | `tqp certify --original {inputs.original.path} --reconstructed {inputs.reconstructed.path} --metric {params.metric} --anchors {params.n_anchors} --seed {params.seed}` (+ `--observer`, `--task`, `--environment`, `--limitation` if those sections exist; `--min-tau` is NOT recorded) | D |
| verification | `tqp verify {certificate.path} [--original --reconstructed] [--data --queries] [--observer]` (data paths not recorded) | D (partial) |
| index-certificate | `tqp index certify {index} --sample {sample}` (`--min-tau` not recorded) | D |
| composition-certificate | `tqp compose --certificate {stages[i].certificate} ... --source ? --metric {metric}` (source path not recorded) | D (partial) |
| capability-report | `tqp capabilities --artifact {artifact} --certificate ?` (certificate list partly recoverable from items) | D (partial) |
| feasibility-report | `tqp feasibility --artifact ? --observer ?` | X/partial |
| compression-plan | `tqp plan run --artifact {artifact path if any} --target {workload.target} --consumer {workload.consumer} --consumer-config {workload.consumer_config} --floor ... --max-bits ... --objective ... --seed ... --n-boot ...` | D (from `workload`) |
| compression-plan-replay | `tqp plan replay RECORD --artifact PATH` | X (record path not stored) |
| embedding-plan | `tqp plan embeddings --embeddings {input.path} --target ... --max-bytes-per-vector ...`; the stored `reproduction` is the follow-up certify | D + R (misleading) |
| kv-plan | `tqp plan kv --model {request.model} --target {request.target} --context {request.context}` | D; stored `reproduction` is a Python call |
| weight-plan | `tqp plan weights --costs ? --budget-bits {budget_bits}` (cost table matched by `cost_table_hash`) | D (partial) |
| refinement-report / compatibility-matrix | `tqp plan refine|compat --artifact ? --observer A --observer B ...` | X/partial |
| replay-report | `tqp replay {claim} [--full]`; each claim row has its shell `command` | R |
| query-catalog / plan / result | `tqp query "{statement}" [--queries ?]` | D |
| console-setup | `tqp console ... --setup FILE` | D |
| console-export | `tqp console ...` (flags not recorded) | X |
| anatomy / hubdiff | `tqp anatomy --npy ? --k {k}` / `tqp hubdiff ... --k {k}` (paths not recorded, fingerprints are) | X/partial |
| monitor | `tqp monitor --original ? --reconstructed ? --floor ?` | X |
| benchmark bundle | `command.sh` | R |

The single fix is `invocation: {argv, cwd, tool_version, git_commit, created_utc}` added in `cli._emit_doc` (S), after which every row becomes R.

---

## 6. Top 10 visualizations by value

1. **Certificate card + four-state validity checklist** (RS-005): the product's core evidence, currently shown as one line with a status that is never computed.
2. **Provenance graph across open artifacts** (RS-002, RS-007): sha256 links between arrays, contracts, certificates, plans; makes "which result depends on which observer" visible.
3. **Plan candidate funnel + frontier scatter with intervals and floor** (BM-002, BM-004, RS-008): shows why a codec won and that ABSTAIN is an answer.
4. **Oscilloscope** (keep): time-domain query stream with trigger, peak detect, segments, masks.
5. **Spectrum analyzer with reference/delta** (keep; RS-003, RS-004): what the observer reads and where the codec is worse than optimal.
6. **Composition chain** (RS-005): left-to-right stages with per-link kappa and tau, weakest link, broken hash link.
7. **Capability three-lane board** (RS-007, IS-004): certified / conditional / not certified for the loaded artifact.
8. **Claims ledger table with status chips and replay bullet charts** (BM-001, BM-003, BM-005).
9. **Feasibility verdict + rate-distortion curve + sensitivity budget bar** (RS-004).
10. **Cross-observer compatibility heatmap** (RS-003).

Close runners-up: query inspector with replay (keep, QI-001/002/004), false-clear 2x2 tile, reader census for learned observers.

## 7. No sensible visualization (show as text/metadata only)

- `tqp version` (a string in the header).
- `tqp plan explain` as text (it is a rendering of the plan; the plan view replaces it).
- `tqp observer hash` / `validate` outputs (fold into the observer card).
- `tqp-kv-store-state/2` and `index-shards` manifests beyond a table (listing, not charting).
- `turboquant-pro/console-export` itself (a container: open its parts, do not chart it).
- Agent-tool schemas (`list_tools()`): a catalog table, nothing to plot.
- Weight cost tables at full size (thousands of cells): summarize by predictor and matrix count; a heatmap of costs[name][bits] is possible but low value.
