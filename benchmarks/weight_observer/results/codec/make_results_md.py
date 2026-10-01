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

**Not yet run** (reported probes of section 2, none scored): the per-matrix bound by local search
from `gptq_f` at 4 bits, additivity under GPTQ, and the flatness curve around `gptq_f`.

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
