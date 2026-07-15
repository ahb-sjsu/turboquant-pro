# Heat-taper vs hard truncation (pre-registered theory test)

**Script:** [`heat_taper_experiment.py`](heat_taper_experiment.py) · deterministic, CPU-only, ~1 min.

**Pre-registered prediction** (from Theorem `thm:heat` of the
[companion theory paper](https://github.com/ahb-sjsu/the-angular-observer), v0.8): hard truncation
at `output_dim` is the *plain-filter* move; exponential tail suppression buys
uniformity-in-truncation. At matched total bits, a soft exponential taper (full-precision head +
exponentially decaying tail bits) should beat hard truncation on rank stability across operating
points, most visibly at small budgets.

**Protocol:** synthetic corpus, power-law spectrum λ_j ∝ j^−1.2 (the concentrated-spectrum regime
where PCA-Matryoshka is the right tool), n=5000 corpus / 500 queries / 1024-d; one PCA basis at
width 512; each config is a per-dim bit schedule at **matched total bits**; metrics vs exact
original-space cosine: single-stage recall@10, anchor-pair Spearman, and the shipped
distribution-free certificate (κ, τ floor) from `rank_certificate`.

## Results

| budget (bits/vec) | config | dims | recall@10 | Spearman | κ | τ floor |
|---:|---|---:|---:|---:|---:|---:|
| 512 | hard@4bit | 128 | 0.7496 | 0.9752 | 1.34 | −0.265 |
| 512 | hard@3bit | 170 | 0.5802 | 0.9034 | 1.58 | −0.666 |
| 512 | **taper(4/3/2/1)** | 165 | **0.7660** | 0.9628 | **1.27** | **−0.091** |
| 768 | hard@4bit | 192 | 0.7728 | 0.9757 | 1.28 | −0.135 |
| 768 | **taper(4/3/2/1)** | 243 | **0.7886** | 0.9700 | **1.24** | **+0.004** |
| 1024 | hard@4bit | 256 | 0.7840 | 0.9759 | 1.26 | −0.063 |
| 1024 | **taper(4/3/2/1)** | 322 | **0.7882** | 0.9728 | **1.23** | **+0.043** |
| 1536 | **hard@4bit** | 384 | **0.7934** | 0.9761 | 1.23 | 0.018 |
| 1536 | taper(4/3/2/1) | 487 | 0.7920 | 0.9738 | 1.22 | 0.059 |

(hard@3bit omitted above 512 for brevity — it is uniformly far behind: at this spectrum, bits
beat dims, consistent with `RESULTS_ablation_rank_bits.md`.)

## Verdict — the prediction splits, informatively

- **Recall@10 (the retrieval metric): taper wins exactly where predicted** — at small budgets
  (+1.6 pts at 512 and 768 bits, +0.4 at 1024) — and converges with hard truncation at large
  budget (−0.1 at 1536). The operating-point cliff that makes `suggest_output_dim` sensitive is
  flattened at the constrained end.
- **Certified floor: taper wins at every budget.** Lower measured distortion κ throughout, and
  the distribution-free τ floor turns **non-vacuous at 768 bits for taper vs 1536 for hard** —
  the taper *certifies* at half the budget. (This is the certificate doing its job: it rewards
  bounded worst-case distortion, which is what the tail bits buy.)
- **Raw anchor-pair Spearman marginally favors hard@4** (−0.002…−0.012): concentrating bits on
  fewer dims tracks the *global* pairwise ordering slightly better, while the taper's extra
  coarse dims buy top-of-list fidelity and worst-case control. Not the uniform sweep the
  strongest reading of the prediction hoped for — documented as the boundary it is.

**Reading:** the heat-filter analogy holds for the quantities that matter operationally
(single-stage recall at constrained budgets; certified worst-case) and does not hold for mean
pairwise rank at generous budgets. One synthetic spectrum, one run — treat as L5-adjacent
evidence pending a public-data replication (GloVe/LaBSE) through the canonical harness.

---

# Public-data replication: the prediction fails, and the boundary is quantified

**Script:** [`heat_taper_public.py`](heat_taper_public.py) · corpus: **BGE-M3 (public model)
over WikiText-2 (public text)**, 6000×1024 unnormalized CLS embeddings
(regenerate with [`embed_wikitext_bge.py`](embed_wikitext_bge.py); the npz is not committed).

| budget | config | recall@10 | Spearman | κ | τ floor |
|---:|---|---:|---:|---:|---:|
| 512 | **hard@4bit** | **0.8044** | **0.9307** | 1.25 | −0.800 |
| 512 | taper(4/3/2/1) | 0.7800 | 0.5983 | 1.46 | −0.961 |
| 1024 | **hard@4bit** | **0.9008** | **0.9680** | **1.11** | **−0.170** |
| 1024 | taper(4/3/2/1) | 0.8710 | 0.8071 | 1.23 | −0.762 |
| 1536 | **hard@4bit** | **0.9354** | **0.9781** | 1.07 | **+0.138** |
| 1536 | taper(4/3/2/1) | 0.9264 | 0.9491 | 1.11 | −0.228 |

**Hard truncation wins every metric at every budget** (recall −0.9 to −3.6 pts for taper;
Spearman far worse; κ and the certified floor also favor hard). The synthetic result does
**not** replicate here — the pre-registration said either outcome is informative, and this
outcome located the boundary:

**Why, quantified.** The synthetic power-law (α=1.2) has **effective rank 13.6** — the top-128
dims carry 85 % of variance and the tail is near-noise, so the taper's cheap 3/2/1-bit tail
dims cost almost nothing and catch residual variance. BGE-M3's real spectrum has **effective
rank 136.7** — top-128 dims carry only 68 %, and the mid-spectrum (128–384) still holds real
variance, so coarse tail bits inject real noise into dimensions that matter while the taper's
narrower 4-bit head loses precision where it matters most.

**The rule:** exponential bit-tapering beats hard truncation when the spectrum's effective
rank is far below the bit-budget's head width (steep-spectrum regime); when effective rank is
comparable to the head width (real sentence encoders), uniform precision on fewer dims wins.
Effective rank vs `budget/4` dims is the one-number diagnostic — computable from the same
covariance `suggest_output_dim` already estimates.
