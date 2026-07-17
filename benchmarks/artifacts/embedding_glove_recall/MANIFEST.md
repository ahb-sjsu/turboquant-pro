# Canonical artifact bundle — `embedding_glove_recall` (full 1.18M GloVe)

A durable, self-contained record of the project's flagship Track-1 claim run on
the **real 1.18M-vector glove-100-angular** corpus. Everything needed to trust and
reproduce the number is here: the exact command, the measured results, a
provenance-stamped rank certificate, the hardware/software environment, and the
git commit certified.

## The claim

**~9.6× compression at recall@10 ≈ 0.999** on real GloVe, via full-dimension PCA
rotation + 3-bit TurboQuant + compressed-domain ADC search + exact rerank.

| metric | value | source |
|---|---|---|
| compression ratio | **9.639×** | `results.json` |
| recall@10 (reranked, ×12) | **0.9991** | `results.json` — the acceptance metric |
| recall@10 (single-pass ADC) | 0.7364 | `results.json` |
| reconstruction mean cosine | 0.9813 | `results.json` — *labelled diagnostic only* |
| corpus / queries | 1,183,514 / 1,000 | `results.json` |

## The certificate — and why it is *weak*, on purpose

`certificate.json` is a distribution-free rank certificate of the **single-pass
3-bit reconstruction** (not the reranked result):

| field | value |
|---|---|
| Kendall τ floor | **0.045** (positive but weak) |
| Spearman ρ floor | −0.43 |
| κ (robust distortion) | 1.133 |
| vacuous | false |

**This weak floor is the point, not a failure.** The certificate is the instrument
correctly reporting that an aggressive 3-bit *single-pass* ranking is **not**
guaranteed — which is exactly why the claim's acceptance metric is *reranked*
recall (0.9991), and why the recipe reranks the top candidates against the exact
originals (recall 0.74 → 0.999). Read together, the certificate and the recall
number tell the complete, honest story: **cheap single-pass ranking is not
trustworthy here → rerank → near-perfect recall.** This is the project's coherence
rule made concrete — acceptance is reranked recall, never reconstruction cosine
(which reads a healthy 0.98 while the single-pass rank floor is 0.045).

## Provenance

- **git commit:** `git_commit.txt` (the exact tree state certified).
- **environment:** `environment.json` — turboquant-pro version, Python, NumPy,
  CPU/GPU, host.
- **input hashes:** `certificate.json → inputs.*.sha256` (the certified arrays).
- **exact command:** `command.sh`.

## Reproduce

```bash
export TQP_GLOVE_HDF5=/path/to/glove-100-angular.hdf5   # ann-benchmarks HDF5
pip install "turboquant-pro[yaml]"
tqp replay embedding_glove_recall --full                 # or --small for the hermetic CI subset
```

The hermetic `--small` variant (a tiny bundled real-GloVe subset) is what CI gates
on every push, so this claim cannot silently regress. See
[`docs/guides/claim_replay.md`](../../../docs/guides/claim_replay.md) and
[`docs/guides/certification.md`](../../../docs/guides/certification.md).
