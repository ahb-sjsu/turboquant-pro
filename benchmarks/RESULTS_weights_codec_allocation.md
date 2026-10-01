# Results: allocation and codec, Part III-c (weights)

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
|---|---|---|
| **C1a** (planning still pays under an error-compensating codec) | `gptq_f` against `gptq_u` | **HOLDS** |
| **C1b** (the planned GPTQ path beats the other codec's uniform path) | `gptq_f` against `awq_u` | **HOLDS** |
| **C2** (the codec pays) | `gptq_f` against `rtn_f` | **HOLDS** |
| **C3** (the per-codec table matters) | `gptq_f` against `gptq_frtn` | **FAILS** |

Each cell: relative KL difference `mean(KL_X) / mean(KL_Y) − 1` over the 48 scored sequences,
95% paired-bootstrap interval, judgement (better: interval below 0 and at most −5%).
‡ = numerics-sensitive (point estimate within 1.3 points of the 5% line; reported, verdict
unchanged).

| id | budget | Qwen2.5-3B | Gemma-2-2B | Llama-3.1-8B |
|---|---|---|---|---|
| C1a | 3-bit | -11.4% [-13.2%, -9.6%] better | -1.0% [-4.1%, +2.7%] neither | -29.9% [-43.7%, -18.1%] better |
| C1a | 4-bit | -15.2% [-17.9%, -12.7%] better | -22.9% [-25.7%, -19.9%] better | -21.8% [-30.8%, -14.6%] better |
| C1b | 3-bit | -31.5% [-34.1%, -28.7%] better | -59.6% [-61.8%, -56.9%] better | -33.6% [-36.1%, -30.9%] better |
| C1b | 4-bit | -35.3% [-37.7%, -32.8%] better | -71.4% [-73.0%, -69.5%] better | -36.3% [-39.5%, -32.9%] better |
| C2 | 3-bit | -66.6% [-70.1%, -63.4%] better | -70.1% [-71.6%, -68.2%] better | -35.6% [-38.2%, -32.7%] better |
| C2 | 4-bit | -51.0% [-53.2%, -48.4%] better | -74.8% [-76.2%, -73.1%] better | -34.6% [-37.6%, -31.7%] better |
| C3 | 3-bit | -3.8% [-4.7%, -3.0%] neither ‡ | -0.4% [-1.5%, +0.7%] neither | -0.4% [-0.6%, -0.1%] neither |
| C3 | 4-bit | -0.6% [-1.6%, +0.3%] neither | -1.2% [-2.4%, +0.0%] neither | +0.8% [-0.3%, +2.0%] neither |

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

| arm | Qwen2.5-3B 3-bit | Qwen2.5-3B 4-bit | Gemma-2-2B 3-bit | Gemma-2-2B 4-bit | Llama-3.1-8B 3-bit | Llama-3.1-8B 4-bit |
|---|---|---|---|---|---|---|
| `rtn_u` | 2.9755 | 0.1839 | 1.8336 | 0.4849 | 0.8429 | 0.1122 |
| `rtn_f` | 0.7035 | 0.0883 | 1.6057 | 0.3440 | 0.3307 | 0.0652 |
| `awq_u` | 0.3426 | 0.0669 | 1.1898 | 0.3029 | 0.3207 | 0.0669 |
| `awq_f` | 0.2643 | 0.0516 | 1.3153 | 0.1872 | 0.2749 | 0.0571 |
| `gptq_u` | 0.2650 | 0.0510 | 0.4856 | 0.1124 | 0.3038 | 0.0545 |
| `gptq_f` | 0.2347 | 0.0433 | 0.4807 | 0.0867 | 0.2129 | 0.0427 |
| `gptq_frtn` | 0.2441 | 0.0435 | 0.4825 | 0.0877 | 0.2137 | 0.0423 |
| `gptq_seq_u` | 0.2696 | 0.0526 | 0.4628 | 0.1141 | 0.2805 | 0.0530 |

**The interaction: allocation gain `KL(codec_u) / KL(codec_f)`** (above 1: planning helps):

| codec | Qwen2.5-3B 3-bit | Qwen2.5-3B 4-bit | Gemma-2-2B 3-bit | Gemma-2-2B 4-bit | Llama-3.1-8B 3-bit | Llama-3.1-8B 4-bit |
|---|---|---|---|---|---|---|
| RTN | 4.23× | 2.08× | 1.14× | 1.41× | 2.55× | 1.72× |
| AWQ | 1.30× | 1.30× | 0.90× | 1.62× | 1.17× | 1.17× |
| GPTQ | 1.13× | 1.18× | 1.01× | 1.30× | 1.43× | 1.28× |

The allocation gain is smaller under GPTQ than under RTN in 6 of 6 cells, and still above 1
in every GPTQ cell. Under AWQ, planning hurt in one cell (Gemma-2-2B, 3 bits:
0.90×).

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

