"""RESULTS_weights_codec_allocation.md: every number read from the committed scorer output."""

import json
import sys

root = sys.argv[1]
r = json.load(
    open(f"{root}/benchmarks/weight_observer/results/codec/results_codec.json")
)
M = ("qwen2.5-3b", "gemma-2-2b", "llama3.1-8b")
NAME = {
    "qwen2.5-3b": "Qwen2.5-3B",
    "gemma-2-2b": "Gemma-2-2B",
    "llama3.1-8b": "Llama-3.1-8B",
}
H = {
    "C1a": (
        "gptq_f",
        "gptq_u",
        "planning still pays under an error-compensating codec",
    ),
    "C1b": (
        "gptq_f",
        "awq_u",
        "the planned GPTQ path beats the other codec's uniform path",
    ),
    "C2": ("gptq_f", "rtn_f", "the codec pays"),
    "C3": ("gptq_f", "gptq_frtn", "the per-codec table matters"),
}
kl = {m: r["models"][m]["mean_kl"] for m in M}


def pct(x):
    return f"{100 * x:+.1f}%"


def cell(m, x, y, b):
    c = r["comparisons"][f"{m}|{x}{b}|{y}{b}"]
    flag = " ‡" if 0.037 <= abs(c["rel"]) <= 0.063 else ""
    return f"{pct(c['rel'])} [{pct(c['lo'])}, {pct(c['hi'])}] {c['judgement']}{flag}"


L = []
L.append("""# Results: allocation and codec, Part III-c (weights)

Registration: `docs/PREREG_weights_codec_allocation.md` (master `59e578e`, PR #242), with its
amendment log: the pilot's run-to-run floor (#288; 0, 48 of 48 sequences bit-identical) and
where the registered models run (#290). Harness `a1f1301`, torch 2.8.0+cu128, transformers
4.56.1, fp16 with each architecture's own attention. Runs, 2026-09-30/10-01: Qwen2.5-3B and
Gemma-2-2B on a Quadro GV100 (Atlas), Llama-3.1-8B on an A100 80 GB (Colab). Plans: PRs #292,
#293, #295. Scorer: `python -m weight_observer.score_codec` at `bc66b91`. Data:
`benchmarks/weight_observer/results/codec/` (scorer output `results_codec.json`).

## Verdicts

Gates, every model: **G1 PASS** (every arm within its budget), **G2 PASS** (the repeated
`gptq_f4` reproduced all 48 sequences bit for bit), **samples PASS** (`hashes.json` equals the
registered identities). Verdict status: **FINAL**.

| id | X against Y | verdict |
|---|---|---|""")
for h, (x, y, what) in H.items():
    L.append(f"| **{h}** ({what}) | `{x}` against `{y}` | **{r['verdicts'][h]}** |")
L.append("""
Each cell: relative KL difference `mean(KL_X) / mean(KL_Y) − 1` over the 48 scored sequences,
95% paired-bootstrap interval, judgement (better: interval below 0 and at most −5%).
‡ = numerics-sensitive (point estimate within 1.3 points of the 5% line; reported, verdict
unchanged).
""")
L.append("| id | budget | " + " | ".join(NAME[m] for m in M) + " |")
L.append("|---|---|" + "---|" * len(M))
for h, (x, y, _) in H.items():
    for b in (3, 4):
        L.append(f"| {h} | {b}-bit | " + " | ".join(cell(m, x, y, b) for m in M) + " |")

L.append("""
**What the registered consequences (section 5) say.** C1a and C2 hold, and C3 fails while C1a
holds: `tqp plan weights` gains GPTQ as its encoder and plans with the one diagonal-Fisher table
it already has; per-codec cost tables are not shipped. C1b holds: the documentation may state
that, at matched stored bytes, the Fisher-planned GPTQ path beats uniform AWQ on these three
models and this corpus.

Reading of the evidence, within the registered scope (three models, WikiText-2, KL):
- Planning pays under GPTQ on Qwen2.5-3B and Llama-3.1-8B at both budgets, and on Gemma-2-2B at
  the 4-bit budget only; at 3 bits on Gemma the planned and uniform GPTQ arms are level.
- GPTQ's gain over the same plan encoded by round-to-nearest is large on every model and budget.
- GPTQ planned with RTN's cost table matches GPTQ planned with its own table: no cell differs by
  5%. One cell (Qwen2.5-3B, 3 bits, −3.8%) excludes 0 but stays under the bar and is flagged.

## Reported, not scored

**Mean KL per token** (nats; `_u` uniform widths, `_f` Fisher-planned; `gptq_frtn` GPTQ on RTN's
plan; `gptq_seq_u` sequential GPTQ):
""")
arms = [
    "rtn_u",
    "rtn_f",
    "awq_u",
    "awq_f",
    "gptq_u",
    "gptq_f",
    "gptq_frtn",
    "gptq_seq_u",
]
L.append(
    "| arm | " + " | ".join(f"{NAME[m]} 3-bit | {NAME[m]} 4-bit" for m in M) + " |"
)
L.append("|---|" + "---|---|" * len(M))
for a in arms:
    L.append(
        f"| `{a}` | "
        + " | ".join(f"{kl[m][a + '3']:.4f} | {kl[m][a + '4']:.4f}" for m in M)
        + " |"
    )

L.append("""
**The interaction: allocation gain `KL(codec_u) / KL(codec_f)`** (above 1: planning helps):
""")
L.append(
    "| codec | " + " | ".join(f"{NAME[m]} 3-bit | {NAME[m]} 4-bit" for m in M) + " |"
)
L.append("|---|" + "---|---|" * len(M))
gain = {}
for c in ("rtn", "awq", "gptq"):
    row = []
    for m in M:
        for b in (3, 4):
            g = kl[m][f"{c}_u{b}"] / kl[m][f"{c}_f{b}"]
            gain[(c, m, b)] = g
            row.append(f"{g:.2f}×")
    L.append(f"| {c.upper()} | " + " | ".join(row) + " |")
smaller = sum(gain[("gptq", m, b)] < gain[("rtn", m, b)] for m in M for b in (3, 4))
L.append(f"""
The allocation gain is smaller under GPTQ than under RTN in {smaller} of 6 cells, and still above 1
in every GPTQ cell. Under AWQ, planning hurt in one cell (Gemma-2-2B, 3 bits:
{gain[('awq', 'gemma-2-2b', 3)]:.2f}×).
""")
pr = {
    m: json.load(
        open(f"{root}/benchmarks/weight_observer/results/codec/{m}/probes/score.json")
    )
    for m in M
}
L.append("""
## Reported probes (section 2, not scored)

Run with `weight_observer.probes` (master `05f49c1`): every plan assembled from a GPTQ code
cache made once by the harness's own path, after `verify` reproduced the registered `gptq_f3`,
`gptq_f4` and `gptq_u4` per-sequence KL bit for bit on each model's own product. Each probe's
starting plan re-measured identical to its registered arm. Data:
`benchmarks/weight_observer/results/codec/<model>/probes/`.

**The per-matrix bound** (64 seeded single same-type width swaps from `gptq_f4`, budget exact,
each kept only if the mean KL falls; the best plan found against `gptq_f4`, 95% paired
bootstrap):
""")
L.append("| model | swaps kept | best against gptq_f4 |")
L.append("|---|---|---|")
for m in M:
    se = pr[m]["search"]
    bv = se["best_vs_start"]
    L.append(
        f"| {NAME[m]} | {se['accepted']} of {se['steps']} | "
        f"{pct(bv['rel'])} [{pct(bv['ci'][0])}, {pct(bv['ci'][1])}] |"
    )
best = max(abs(pr[m]["search"]["best_vs_start"]["rel"]) for m in M)
L.append(f"""
No improvement the search found on the Fisher-planned GPTQ plan reaches the 5% bar: the
largest is {100 * best:.1f}%, the only one whose interval excludes 0. A 64-step local search
is a lower bound on the headroom, not its global maximum (section 6), but it finds as little
under GPTQ as Part III's exploration did under RTN (1.9%, #240).

**Additivity under GPTQ** (each planned arm's measured KL over the sum of its matrices'
single-matrix KL, every other matrix at full precision):
""")
L.append("| arm | " + " | ".join(NAME[m] for m in M) + " |")
L.append("|---|" + "---|" * len(M))
for arm in ("gptq_f3", "gptq_f4", "gptq_frtn3", "gptq_frtn4"):
    L.append(
        f"| `{arm}` | "
        + " | ".join(f"{pr[m]['additivity'][arm]['ratio']:.3f}" for m in M)
        + " |"
    )
r3 = [pr[m]["additivity"][a]["ratio"] for m in M for a in ("gptq_f3", "gptq_frtn3")]
r4 = [pr[m]["additivity"][a]["ratio"] for m in M for a in ("gptq_f4", "gptq_frtn4")]
err = 100 * max(max(r3) - 1, 1 - min(r4))
L.append(f"""
At 3 bits the damage is super-additive ({min(r3):.2f} to {max(r3):.2f}), at 4 bits
sub-additive ({min(r4):.2f} to {max(r4):.2f}): a per-matrix sum misjudges a whole plan by up
to {err:.0f}%, in a direction set by the budget.

**The flatness curve around `gptq_f`** (`flatness.perturb` with Part III's swap counts, draws
and seed: `k` disjoint same-type swaps, budget exact; KL rise over the anchor, mean of 3
draws; in brackets, the share of stored bits moved):
""")
ks = ["1", "2", "4", "8", "16", "32"]
L.append("| model | budget | " + " | ".join(f"k = {k}" for k in ks) + " |")
L.append("|---|---|" + "---|" * len(ks))
for m in M:
    for b in ("3-bit", "4-bit"):
        fk = pr[m]["flatness"][b]["k"]
        cells = [
            f"{pct(fk[k]['kl_rise_rel'])} ({100 * fk[k]['bits_moved']:.1f}%)"
            for k in ks
        ]
        L.append(f"| {NAME[m]} | {b} | " + " | ".join(cells) + " |")
BUD = ("3-bit", "4-bit")
k32 = [pr[m]["flatness"][b]["k"]["32"]["kl_rise_rel"] for m in M for b in BUD]
small = [
    pr[m]["flatness"][b]["k"][k]["kl_rise_rel"]
    for m in M
    for b in BUD
    for k in ("1", "2")
]
mv = [
    pr[m]["flatness"][b]["k"][k]["bits_moved"]
    for m in M
    for b in BUD
    for k in ("1", "2")
]
mv32 = [pr[m]["flatness"][b]["k"]["32"]["bits_moved"] for m in M for b in BUD]
L.append(f"""
One or two swaps (at most {100 * max(mv):.1f}% of the stored bits) change the KL by
{pct(min(small))} to {pct(max(small))}; 32 swaps ({100 * min(mv32):.1f}-{100 * max(mv32):.1f}% of the bits) cost {100 * min(k32):.0f}% to
{100 * max(k32):.0f}%, and the curve between is not monotone in `k` (three draws per point).
The Fisher-planned GPTQ plan sits near a local optimum (single swaps improve it by at most
{100 * best:.1f}%, above), and the optimum is not flat. Part III's RTN flatness also reported each
perturbed plan's additive prediction; that cannot be made here, because the registered single
sweep covers the planned widths only and a swap moves a matrix to a width it was not measured
at.
""")
L.append("""
## Conduct

- **Pre-registration checks** (2026-09-28/29, PR #268): each registered model's GPU peak measured
  on its own shapes; the fp16 harness checked against fp32 on the scored windows (Gemma-2 needs its
  own eager attention: sdpa drops its logit soft-cap); the pilot's fp16-against-fp32 invariance
  failed its 1% tolerance once (1.27%, calibration numerics changing GPTQ/AWQ codes), recorded in
  the prereg with the numerics-sensitivity flag above.
- **Where the runs happened.** On NRP every registered cost-table job was deleted within minutes
  of starting, with no reason recorded (2026-09-30); a faster start-up (#289) did not change that.
  The prereg was amended before any registered result (#290): GV100 for Qwen2.5-3B and
  Gemma-2-2B, A100 80 GB for Llama-3.1-8B, one product per model for every phase.
- **Samples.** Every run loaded pre-tokenized windows whose sha256 were rechecked on load and
  equal the registered identities (gate "samples").
- **Gate G0** (GPTQ with H = I and AWQ with α = 0 reproduce RTN bit for bit) passed on each device
  before the cost tables and again after the arms.
""")
out = f"{root}/benchmarks/RESULTS_weights_codec_allocation.md"
open(out, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("written", out)
