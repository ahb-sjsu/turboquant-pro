# Production lifecycle guide — a mutable, compressed, drift-aware index

A benchmark index lives in RAM and is thrown away. A production RAG index must
persist, grow, forget, stay honest about storage, survive format upgrades, prove it
still ranks well, and notice when its basis goes stale. `TQEIndex` + the `tqp index`
command group cover that whole loop over the corruption-checkable **TQIX** container.

```
ingest → search → update → compact → migrate → certify → monitor
```

## The loop

```bash
# ingest: fit the PCA basis once, persist basis + codes
tqp index create --embeddings corpus.npy --out corpus.tqe --bits 3

# search: compressed, with exact rerank for high recall
tqp index search corpus.tqe --queries q.npy --k 10 --rerank 10

# update: append (same basis, no refit) and tombstone by id
tqp index add    corpus.tqe --embeddings new.npy
tqp index delete corpus.tqe --ids 12,88,90

# compact: physically drop tombstoned rows, reclaim bytes (ids are preserved)
tqp index compact corpus.tqe

# migrate: upgrade the on-disk format under a real version bump
tqp index migrate corpus.tqe --to-version 2

# certify: a rank certificate over the live vectors (gates the exit code)
tqp index certify corpus.tqe --min-tau 0.5

# monitor: is the PCA basis still a good fit for recent data?
tqp index drift corpus.tqe --embeddings recent.npy
```

Same surface from Python via `TQEIndex.create/open/add/delete/compact/migrate/search/certify/drift`.

## What makes it trustworthy

- **Corruption is detected, never silent.** Every section of the TQIX container
  carries a CRC32; a flipped byte is a clean `IndexCorruptionError`, not
  wrong-but-plausible vectors. Writes are atomic (temp file + rename). A
  single-byte-flip fuzzer guards the invariant "detected, or byte-identical."
- **Ids are external and stable.** `create`/`add` assign monotonic ids; they survive
  `compact` (rows are dropped, ids are never renumbered), so external references stay
  valid across maintenance.
- **Exact rerank + certify need the originals.** `create` keeps fp32 originals by
  default (`--no-originals` to skip); they power exact rerank and let the index
  certify itself. Without them, rerank degrades to the reconstruction.

## Closing the loop — adaptive and drift-aware

Two signals turn this from a static store into an adaptive one:

```python
from turboquant_pro import TQEIndex, TQPRuntimePolicy
idx = TQEIndex.open("corpus.tqe")
policy = TQPRuntimePolicy()

# adaptive retrieval: single-pass where margins are wide, exact rerank where tied
ids, _ = idx.search(queries, k=10, policy=policy)

# scheduled maintenance: refit/migrate when the encoder distribution drifts
report = idx.drift(recent_embeddings)
if policy.evaluate_index_drift(report).conservative:
    #  action == "refit_or_migrate"  →  rebuild the basis on fresh data
    ...
```

`drift` compares the variance the stored basis retains on new data to what it
retained at fit time (plus a mean shift); a large drop means the basis is stale and
recall will silently erode. Catch it before your users do. See
[certification](certification.md) for the guarantee behind `certify`, and the
[operator-aware guide](operator_aware_quantization.md) for why acceptance is recall /
a rank certificate, never reconstruction cosine.
