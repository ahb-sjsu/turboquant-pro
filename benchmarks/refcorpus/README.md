# Reference corpus work has moved

The reference-corpus specifications and geometry battery now live in their own
repository:

**https://github.com/ahb-sjsu/openvector-bench**

A benchmark hosted inside the repository of the system it evaluates is
structurally suspect — reviewers discount it, and other index implementations
cannot adopt it without adopting this engine. It is therefore developed and
versioned independently, with its own licence, DOI, and issue tracker.

What moved:

| was here | now |
|---|---|
| `PREREG.md` | `spec/PREREG_RC1.md` |
| `DISTRIBUTION.md` | `spec/DISTRIBUTION.md` |
| `FAMILY.md` | `spec/FAMILY.md` |
| `corpus_geometry.py` | `harness/geometry/corpus_geometry.py` (also `openvector_bench.geometry`) |

Git history for these files remains in this repository up to commit
`bbdc14f`.

**Why it matters to turboquant-pro.** The RC-1 battery measures whether a
generated corpus resembles real embeddings on the properties that govern ANN
search. It is what tells us how far the seeded synthetic corpora behind the
1B/10B fleet runs can be trusted: those runs are valid as *systems*
measurements, and the geometry battery is what bounds any claim about
retrieval difficulty. See `benchmarks/RESULTS_ivf.md` and
`paper/systems/OUTLINE.md` §7.
