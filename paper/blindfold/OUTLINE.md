# The Quantizer's Blindfold — outline (v0.1 draft skeleton)

**Working title:** *The Quantizer's Blindfold: Compressed Representations are
Faithful to Themselves and Blind to Their Consumers*

**Status:** skeleton for review. Empirical anchors: (A) the 1B fleet triplet
(measured, committed); (B) the KV attention-softmax bench (running); (C) the
real-embedding pilot (running). Theory is CITED from the companion papers —
Keep the Angle (spectral basis) and Observation Theory (two-observer
successive refinement, the tr(P_C·Σ_δ) mechanism) — not re-derived here.

## Abstract (draft)

Every lossy representation must decide what to keep. The standard acceptance
metrics for compressed vector systems — reconstruction error, cosine to the
original, internal recall — share a hidden assumption: that fidelity to the
*signal* implies fidelity to the *consumer* that reads it. We show, with a
betting-game formalization and measurements from billions-scale retrieval and
LLM KV-cache serving, that this assumption fails in a structured way: a
compressed index reproduces its own exact ranking almost perfectly
(recall 0.999 at one billion rows) while agreeing with the true fp32 ranking
barely more than half the time (0.592) — and a bounded consumer-side exact
rescore restores 0.991. The dissociation is not a defect of one quantizer but
a property of observation: which invariant must be kept is defined by the
consumer's read operator, not by the signal [Observation Theory, Cor./Thm refs].
We give practitioners a decision rule — accept compression only on
consumer-relative rank fidelity, never on reconstruction — and quantify its
price and payoff across three consumers: retrieval ranking, the attention
softmax, and exact rerank.

## Sections

1. **The betting game.** Two players compress the same corpus at equal bits;
   the referee is a *consumer*, not a distance. Payoffs under (i) a
   reconstruction referee, (ii) a self-consistency referee, (iii) the true
   consumer. The game makes "faithful-to-itself vs faithful-to-truth"
   precise and shows the referees disagree by construction when the
   consumer's read operator is misaligned with the signal's energy.
   [Formal machinery cited: read operator P_C, distortion tr(P_C·Σ_δ);
   observer-induced quotient geometry (angle/rank/channel-scale).]

2. **Exhibit A — retrieval at 10⁹.** The fleet triplet: 0.999 self /
   0.592 truth / 0.991 reranked at 24 B/row, with the distributed-build and
   exact-merge protocol as evidence the measurement isn't an infrastructure
   artifact. Include the 10B replication (pending) and the real-embedding
   pilot (pending: does the *structure* — not the numbers — transfer to
   Embed-V3 Wikipedia at 1024-d?).

3. **Exhibit B — the attention softmax as consumer.** Same structure on the
   serving path of a real model (Qwen2.5-1.5B): per-decode-step attention KL
   under KV compression at matched streams (teacher-forced), against
   behavioral agreement and bytes/token. Ties to the in-tree matched-bit
   behavioral results (8-bit statistically indistinguishable; 4-bit 40%
   behavioral disagreement) — reconstruction-style metrics rank these
   configs; the consumer does not agree with the ranking. [Bench running.]

4. **The mechanism, briefly.** Why the blindfold exists: compression
   optimizes an isotropic (or signal-energy-weighted) objective; consumers
   read a low-rank, generally misaligned operator. Cite: radial degeneracy /
   keep-the-angle (spectral consumers); tr(P_C·Σ_δ) across attention /
   retrieval / Hessian consumers; the D4 boundary (when read ≡ energy, the
   blindfold is harmless — the honest negative that scopes the claim).

5. **The decision rule and its price.** Acceptance = consumer-relative rank
   fidelity (recall@k vs the consumer's exact answer, rank certificates),
   never reconstruction. Costs: a reference oracle (scattered exact scan),
   a cold tier for rescue (24 B/row hot + shortlist-bounded fetch), and the
   engineering to keep scores comparable across nodes (shared basis).
   Payoff table from the fleet runs.

6. **Related work.** PQ/OPQ/ADC lineage and refine/rerank practice (the
   *practice* is standard; the *acceptance-metric claim* is the
   contribution); KV quantization lines (KVQuant, KIVI); certified /
   behavioral evaluation; the companion theory papers.

7. **Limitations.** Synthetic corpus for the flagship number (real-corpus
   pilot bounds this); one model family for the KV exhibit so far; QPS not
   competitive by design; consumer families where read≡energy.

## Data/asset map

- `benchmarks/fleet/results/fleet_run_1B.json` (committed) — Exhibit A core
- `benchmarks/fleet/results/fleet_run_10B.json` (pending tonight)
- `/archive/tqp_real/wiki1024/real_pilot_result.json` (pending) — §2 transfer
- `benchmarks/bench_kv_serving_result.json` (pending) — Exhibit B
- `experiments/results_matched_bit/behavioral_*.json` (committed) — §3
- Citations: ahb-sjsu/the-angular-observer (Keep the Angle);
  observation-theory (two-observer successive refinement; tr(P_C·Σ_δ));
  turboquant-pro RESULTS docs.
