# Certification guide — what a certificate means, and what it does not

A TurboQuant Pro **rank certificate** answers one question: *after compression, how
badly can the ranking my consumer depends on be distorted?* It is a
distribution-free floor, not a vibe.

## What it is

`certificate_from_embeddings(original, reconstructed, metric=...)` samples anchor
pairs, measures the exact vs reconstructed pairwise distances, and returns a
[`RankCertificate`](../CERTIFICATE_SPEC.md):

```python
cert = certificate_from_embeddings(corpus, recon, metric="cosine")
cert.tau_floor        # guaranteed Kendall-tau between exact and compressed ranking
cert.spearman_floor   # guaranteed Spearman rho
cert.kappa            # robust distortion (the measured quantity)
cert.vacuous          # True when no positive floor can be guaranteed
```

`tqp certify --original o.npy --reconstructed r.npy --min-tau 0.5` emits the
provenance-stamped JSON ([schema](../CERTIFICATE_SPEC.md)) and **exits non-zero** if
the floor is not met — so it gates CI and deploys.

## What it **guarantees**

- **A floor, not an estimate.** `tau_floor >= 0.87` means the Kendall-tau between the
  exact ranking and the compressed ranking is *at least* 0.87 on the certified
  configuration — a worst-case bound derived from the measured distortion
  concentration, with no distributional assumption beyond a light robust trim.
- **The metric you name.** Certify under the metric your index actually ranks by
  (`cosine` or `l2`); a certificate under the wrong metric certifies the wrong thing.

## What it does **not** guarantee

- **Not reconstruction quality.** The certificate is about *rank* preservation, which
  is what retrieval consumes. A high reconstruction cosine with a vacuous certificate
  is a *worse* index than a lower cosine with a positive floor — see the
  [operator-aware guide](operator_aware_quantization.md).
- **Not your downstream task score.** It bounds ranking distortion, not answer
  quality; a RAG system can still fail for reasons upstream of retrieval.
- **Not the tail beyond the anchors.** It is a sampled bound; use enough anchors
  (default 200 → ~20k pairs) for a tight, trustworthy floor.

## Reading the verdict

| Result | Meaning | Do |
|---|---|---|
| `tau_floor` high, `vacuous=False` | ranking is safe under compression | ship the compressed index |
| `tau_floor` low but positive | some distortion guaranteed-bounded | tighten bits, or rerank the top-k |
| `vacuous=True` | **no** positive floor can be guaranteed | require exact rerank, or refuse to certify |

The runtime policy automates that last column: `TQPRuntimePolicy.evaluate_certificate`
maps a vacuous certificate to `require_exact_rerank`
([production lifecycle](production_lifecycle.md)).

## The coherence rule

Certification is the formal face of the project's one rule: **acceptance is rank
fidelity, never reconstruction cosine.** The certificate exists precisely because the
cheap metric (cosine) can look great while the ranking collapses. If you are gating a
release on cosine, you are gating on the wrong number.
