# Response to Review 3 — version consistency & PyPI-page positioning

Three remaining issues, all presentation:

**1. Version consistency (PyPI 1.4.0 vs stale v1.1.0 crawl). ✎ (done)**
Replaced the version table with the reviewer's exact top-of-README block, artifact commit filled in:

```
Current package:               turboquant-pro 1.4.0
Current paper artifact:        v1.4.0 / commit 1f39747
Main public benchmark notebook: compatible with 1.4.0
Earlier v1.1.0 docs are retained for release history only.
```

Verified the repo itself is already consistent — README and `CHANGELOG.md` both lead with **v1.4.0**;
the only `v1.1.0` occurrences are the legitimate release-history table row and changelog section. So
the "v1.1.0 in several places" the reviewer saw is **stale search indexing**, not a repo error. The
block makes that explicit and the row now says it's retained for history only.
- *Root cause of the stale PyPI text (incl. "489 tests"): PyPI serves the README of the **last
  published release** (1.4.0), which predates these edits. It only refreshes on the next `twine
  upload`.* → cut **v1.4.1** (see below).

**2. Name collision / positioning. ✎ (done)**
Added **vLLM's own TurboQuant integration** to the "Not to be confused with" list (alongside
`turboquant`, `pyturboquant`, `turboquant-ml`, `turboquant-py`, `turboquant_plus`) and adopted the
reviewer's exact framing: *"turboquant-pro is not just a KV-cache TurboQuant implementation. It is a
PCA-Matryoshka + TurboQuant toolkit for compressed embedding retrieval, with additional KV-cache,
vector-database, and systems integrations."*

**3. Dense high-impact claims paragraph → claims table. ✎ (done)**
Trimmed the README headline paragraph to a short lede; the full claim list (27×, RaBitQ/OPQ, 4–20×
builds, 22% codebooks, KV-cache) now lives **only** in the [`CLAIMS.md`](CLAIMS.md) table with
per-claim dataset / notebook / hardware / reproduction status. Removed the static test count in favour
of the CI badge (already done in Review 1; the PyPI copy will refresh on the next release).

## Needs the author (outside the repo)
- **Cut `v1.4.1`** (or `v1.5.0`) and `twine upload` so the **PyPI page** picks up the slimmed claims
  paragraph, the removed "489 tests", and ships the `notebooks/claims/` reproduction suite in a tagged
  artifact. This is the single action that resolves the externally-visible staleness the reviewer keeps
  seeing. I can prep the release notes + tag on request.
- Refresh the **GitHub "About"/social preview** if it still shows old text.
