# Part V registered runs: how the Tier A grid is executed on NRP

Companion to `docs/PREREG_full_consumer_kv.md` (section 6). This is the execution plan
for the registered cells on Llama-2-7B-chat, Mistral-7B-Instruct-v0.2 and
Qwen2.5-7B-Instruct, on NRP A10 pods. It changes no arm, endpoint, bar or verdict rule.
The waves are defined in `benchmarks/kvquant_matrix/part5_grid.py` (`WAVES`) and the
pods in `part5_nrp.py set-run`.

## Constraints that shape the plan

- **One pod per volume set at a time.** Each model's weights sit on its own set of
  single-attach (RWO) linstor-unl volumes in zone unl. A model's pods therefore run one
  after another, and the three models run in parallel. That makes at most three GPU
  pods at once, inside the programme's cap of four heavy pods. `set-run` refuses to
  submit while any unfinished Job of this project uses the model's set. That check also
  matters because a same-name submit deletes the old Job first.
- **NRP's usage rules.** The shape is 2 CPU / 7 GiB / 30 GiB ephemeral, one A10. It is
  sized from the 7B set-smoke's guard record:
  - mean 0.89 cores, which is 45% of the request;
  - mean 4.64 GiB, which is 66% of the request;
  - peak 5.63 GiB, under the limit.

  pod_guard stops a pod whose windowed GPU mean falls under 40% after warm-up, and
  every pod has a hard deadline.
- **Nothing is lost when a pod stops.** Cells resume per document and per perplexity
  chunk. The next pod of the same wave continues where the last one stopped, whether
  that pod hit its deadline, was stopped by the guard, or failed operationally. A cell
  is never rerun because of its result.
- **Telemetry survives the pod.** pod_guard's lines (`[gpu] util, mean, mem`) and the
  runner's `[run]` lines are teed to `<run root>/<model>/pod_<wave>.log` on volume 0,
  and `fetch-set --what run` brings them back with the cells.

## Waves, in order, per model

Times are estimates for Qwen2.5-7B, scaled from the 7B smoke's per-document and
per-chunk times. The full run is 200 documents per task and 146 perplexity chunks, plus
about 10 minutes per pod to set up and build the verified fp16 copy. Each deadline is
1.5 times the estimate.

| wave | arms | est. h | why here |
|---|---|---|---|
| `gate` | `u3`, `u3_jit`, `u3_rep`, `u3_jit_rep` | 2.6 | G1 is decided before any other cell runs. Its full-length cells also measure the utilization every later wave relies on. |
| `g0` | `g0_native`, `g0_read`, `g0_head`, `g0_head_cal`, `g0_fisher_cal`, `fp16` | 3.4 | The G0 gates. The calibration tables are fitted here, by their G0 arms, before any scored arm reads them. |
| `verdict` | `u3_read`, `u3_head_cal`, `u3_fisher_cal` | 2.7 | References for V1, V3 and V4. |
| `head` | `u3_head` | 3.8 | The arm in V1, V2, V4 and V5. The head weight is the costly path, at about 20 s per document against u3's 3. |
| `reported3` | `g0_key`, `g0_read_cal`, `u3_key`, `u3_read_cal` | 2.2 | Reported. |
| `reported2`, `reported4` | the seven non-head arms at 2 and at 4 bits | 5.6 each | Reported. |
| `head2`, `head4` | `u2_head`, `u4_head` | 3.8 each | Reported. |
| `baselines` | `aatc`, `aatc_native`, `ea`, `ea_u3` | 3.4 | V5's baselines at u3's stored bytes, which the runner reads from the `gate` wave's `u3` cell: AATC's key bits, and Expected Attention's eviction per prompt (`ea.py`). Expected Attention's perplexity takes two passes. |
| `fidelity` | `kivi2`, `aatc2` | 1.8 | AATC's fidelity check, on Qwen2.5-7B only, decided before registration. |

- **Verdict waves:** about 12.5 GPU-hours per model, so about 13 hours of wall time with
  the three models in parallel.
- **Reported waves:** about 21 more GPU-hours per model.
- **Tier A in total:** about 100 A10-hours.

Llama-2's 4k context makes its documents cheaper. Mistral's 32k context makes them
dearer if its qasper prompts run longer than Qwen's. Each `gate` pod's log gives that
model's true rates before its later waves go out.

A model's own estimates live in `MODEL_WAVE_HOURS` (part5_nrp.py). Mistral's `baselines`
is 16 h, a 24 h deadline: its first pod reached the 5.1 h deadline on 2026-10-15
before finishing `aatc`, the wave's first arm (owner, 2026-10-16). Mistral's `head` is
12 h, an 18 h deadline: its pod reached the 5.7 h deadline on 2026-10-19 inside `u3_head`,
the wave's one arm (owner, 2026-10-19).

## Sequence

1. **Registration.** The owner reviews the prereg, and the merge to master registers it.
   Registered commit = C.
2. **Stage C's code** (`part5_nrp.py code --commit C`, CPU, exempt class).
3. **Gate.** Submit `set-run --wave gate` on all three models, then `fetch-set --what run`
   for each, then run `score_part5.py`. It reports G1 per model. Its verdicts are
   INCOMPLETE at this point by construction.
   - G1 FAIL_UNEXPLAINED on any model: stop. Explain the failure under Part II's
     Amendment 5 rule before anything else runs.
   - Otherwise: re-estimate `WAVE_HOURS` from each gate log. This is an execution change,
     recorded in the run log, not an amendment.
4. **Verdict waves** (`g0`, then `verdict`, then `head`) on each model. Each model's
   next wave is submitted when its previous pod has finished. Then fetch and score; the
   verdicts are final when every gate passes.
5. **Reported waves**, in the table's order.
6. **Results** go in `benchmarks/RESULTS_full_consumer_kv.md`, whichever way they fall.

## Watching

- **On Atlas:** the report-only utilization guard (screen `kv5-guard`, selector
  `app=tqp-kv5`) must be beating, or `set-run` refuses to submit. `part5_observe.py`
  reports pending reasons, stalls and finishes.
- **No active Job is ever deleted from Atlas.** A pod that idles is ended from inside, by
  its own pod_guard.

## Not covered here

- **Tier B (Qwen2.5-1.5B, Llama-3.2-3B).** At 1.5B the harness kept an A10 only about
  20% busy, under NRP's floor. The owner moved Tier B to L4 pods (2026-10-13). Each model
  gets a volume set on Fullerton's Ceph (`rook-ceph-block-fullerton`, zone `fullerton`,
  where the open L4 nodes are): `volumes --apply`, `stage-set`, then `set-smoke`. The
  smoke measures the L4's utilization, and the waves and their hours are set from it.
- **AATC's fidelity check.** It runs before registration: `stage --what aatc-text`, then
  `set-run --wave fidelity --smoke` (4 documents per task, a wiring check), then
  `set-run --wave fidelity` on Qwen2.5-7B, then `score_part5.py --fidelity`. The outcome is
  written once, to `benchmarks/kvquant_matrix/part5_aatc_fidelity.json`.
