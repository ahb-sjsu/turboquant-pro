# Response to Review 2 — positioning & discoverability

Review 2 raised five remaining issues, all about **presentation, versioning, and disambiguation**
(no correctness concerns). All five addressed:

**1. Inconsistent versioning (GitHub crawl shows v1.1.0, PyPI at v1.4.0). ✎ (done)**
Added a **Version map** table near the top of the README: latest PyPI **1.4.0**, GitHub docs/`main`
**1.4.0** (+ post-1.4.0 reproduction notebooks on `master`), benchmark notebooks compatible with
1.4.0, paper/archival artifact = tag `v1.4.0` + Zenodo DOI. Explicit note that the canonical artifact
is v1.4.0 and any v1.1.0 the crawl surfaces is stale.
- *Recommendation (needs author):* cut a **v1.4.1** release so the `notebooks/claims/` reproduction
  suite ships in a tagged artifact (they currently live on `master`, post-v1.4.0).

**2. Broad tool surface → API stability table. ✎ (done)**
Added a compact **API stability** table (Stable / Beta / Experimental) directly in the README,
linking the full [`docs/api-stability.md`](docs/api-stability.md). (The doc existed from Review 1;
Review 2 asked for it to be visible in the README itself — now it is.)

**3. Crowded namespace (turboquant, pyturboquant, turboquant-py, turboquant-ml, turboquant_plus). ✎ (done)**
Added a **"Not to be confused with"** block naming the lookalikes and stating what `turboquant-pro`
**uniquely** does: PCA-Matryoshka embedding compression + production-oriented compressed-domain search
and integrations (FAISS, pgvector, HNSW, ADCIndex), with architecture-aware KV-cache quantization as a
separate module.

**4. Make the claim hierarchy impossible to miss → CLAIMS.md. ✎ (done)**
Added a top-level **[`CLAIMS.md`](CLAIMS.md)** in exactly the requested schema —
*Claim / Public reproduction? / Dataset / Command-or-notebook / Hardware / Status* — split into
Track 1 (central, CPU-reproducible) and Track 2 (GPU, experimental). Linked from the README intro;
`docs/claims.md` now cross-references it as the detailed version.

**5. Tighten paper/repo distinction. ✎ (done)**
The README two-contribution block now states plainly that the **central, most-validated result is
embedding compression + compressed-domain retrieval (Track 1)**, with KV-cache and fused decode
(Track 2) as the engineering-package extras. CLAIMS.md reinforces the same split. This gives a
reviewer one central result to validate rather than a whole platform.

## Still needs the author (outside the repo)
- Update the **GitHub "About"** blurb / repo social preview if it still emphasizes v1.1.0 or an old
  test count.
- Optionally cut **v1.4.1** to tag the reproduction notebooks into a release artifact.
