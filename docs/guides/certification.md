# Certification guide — what a certificate means, and what it does not

A TurboQuant Pro **rank certificate** answers one question: *after compression, how
badly can the ranking my consumer depends on be distorted?* It is a floor on
the measured pairs with no distributional assumption, not a vibe.

## What it is

`certificate_from_embeddings(original, reconstructed, metric=...)` samples anchor
pairs, measures the exact vs reconstructed pairwise distances, and returns a
[`RankCertificate`](../CERTIFICATE_SPEC.md):

```python
cert = certificate_from_embeddings(corpus, recon, metric="cosine")
cert.tau_floor        # Kendall-tau floor on the measured pairs (tail penalty included)
cert.spearman_floor   # Spearman rho floor, likewise
cert.kappa            # robust distortion (the measured quantity)
cert.tail_fraction    # share of pairs the kappa percentile band leaves out
cert.tau_floor_central  # the unpenalized value: a diagnostic, not a guarantee
cert.vacuous          # True when the floor is not positive
```

`tqp certify --original o.npy --reconstructed r.npy --min-tau 0.5` emits the
provenance-stamped JSON ([schema](../CERTIFICATE_SPEC.md)) and **exits non-zero** if
the floor is not met — so it gates CI and deploys.

## What it **guarantees**

- **A floor, not an estimate.** `tau_floor >= 0.87` means the Kendall-tau between the
  exact ranking and the compressed ranking is *at least* 0.87 on the certified
  configuration — a worst-case bound over the measured anchor pairs, derived from
  the measured distortion and concentration with no distributional assumption.
  The default `kappa` trims the most distorted ~5% of distances, and the floor
  pays for them: `tau >= 1 − 2·mu_hat − 2·tail_fraction`, a penalty of about
  0.195 at the default trim. Use the strict regime (`lo=0, hi=100`) for no
  penalty, at the price of sensitivity to a single collapsed pair.
- **The metric you name.** Certify under the metric your index actually ranks by
  (`cosine` or `l2`); a certificate under the wrong metric certifies the wrong thing.

## What it does **not** guarantee

- **Not reconstruction quality.** The certificate is about *rank* preservation, which
  is what retrieval consumes. A high reconstruction cosine with a vacuous certificate
  is a *worse* index than a lower cosine with a positive floor — see the
  [operator-aware guide](operator_aware_quantization.md).
- **Not your downstream task score.** It bounds ranking distortion, not answer
  quality; a RAG system can still fail for reasons upstream of retrieval.
- **Not the pairs beyond the anchors.** It is a bound on the sampled anchor
  pairs; sampling error is not included. Use enough anchors (default 200 → ~20k
  pairs).
- **Not top-k recall.** A rank-correlation floor does not by itself certify
  recall@k; that needs a separate argument.

## Reading the verdict

| Result | Meaning | Do |
|---|---|---|
| `tau_floor` high, `vacuous=False` | ranking is safe under compression | ship the compressed index |
| `tau_floor` low but positive | some distortion guaranteed-bounded | tighten bits, or rerank the top-k |
| `vacuous=True` | uncertified at this setting (distances concentrate, or the tail penalty removes the floor) | validate the setting directly, or rerank; the floor is loose, so vacuity does not show retrieval fails |

The runtime policy automates that last column: `TQPRuntimePolicy.evaluate_certificate`
maps a vacuous certificate to `require_exact_rerank`
([production lifecycle](production_lifecycle.md)).

## The coherence rule

Certification is the formal face of the project's one rule: **acceptance is rank
fidelity, never reconstruction cosine.** The certificate exists precisely because the
cheap metric (cosine) can look great while the ranking collapses. If you are gating a
release on cosine, you are gating on the wrong number.
