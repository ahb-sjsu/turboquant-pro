# RC-1 distribution: a corpus as a cryptographically defined object

**Problem.** A trillion-row corpus cannot be distributed as bytes (128 TB at
128-d uint8), and a *procedurally* defined one cannot be trusted without a way
to prove that what you regenerated is what everyone else regenerated. Both
problems have the same solution: define the corpus by a **signed Merkle
manifest**, and treat every byte source — local regeneration, a regional
cache, a public mirror — as an interchangeable way to satisfy a hash.

**Target property.** Delete every materialized file, retain only the signed
root manifest, and reconstruct benchmark shards that are byte-identical to
the canonical ones from a mixture of caches and mirrors — then show the
rebuilt index answers queries identically. Until that experiment passes, this
is a design, not a result.

---

## 1. What is actually being distributed

Not vectors. Three small things:

1. **Generator spec** — source URI(s), pinned toolchain version, parameter
   blob, and the seed→shard mapping. Kilobytes.
2. **Merkle manifest** — per shard, the chunk hashes; per corpus, the root.
   ~32 bytes per chunk, so an 8 MiB chunk size costs ~4 MB of manifest per TB
   of corpus. A 128 TB corpus has a ~0.5 GB manifest — distributable.
3. **Detached signature** over the root (GPG, the same identity that signs
   the repository's commits).

**Licensing consequence, and it is a feature.** Hashes and pointers are
facts about third-party data, not redistribution of it. For corpora whose
terms restrict mirroring (or whose size forbids it), we distribute the
manifest and the locator; the bytes come from the origin. Only material we
are licensed to mirror is cached in our pools.

---

## 2. Manifest schema (v0)

```jsonc
{
  "corpus": "rc1-wiki1024", "version": "0.1.0",
  "metric": "cosine", "dim": 1024, "dtype": "float32",
  "n_rows": 41000000, "rows_per_shard": 1000000,
  "chunk_bytes": 8388608, "hash": "sha256",
  "generator": {                       // absent for pure-mirror corpora
    "impl": "refcorpus.generate:v0.1.0",
    "toolchain": {"python": "3.12", "numpy": "2.3.1"},
    "params_sha256": "…", "seed_scheme": "shard_index"
  },
  "sources": [                         // ordered fallbacks, not preferences
    {"kind": "nrp-s3", "region": "west",
     "url": "https://s3-west.nrp-nautilus.io/rc1/…", "role": "cache"},
    {"kind": "zenodo", "doi": "10.5281/zenodo.…", "role": "durable"},
    {"kind": "origin", "url": "https://dl.fbaipublicfiles.com/…",
     "role": "authoritative", "license": "…"}
  ],
  "shards": [{"i": 0, "rows": 1000000, "root": "…", "chunks": ["…"]}],
  "root": "…"
}
```

The root hashes the shard roots; each shard root hashes its chunk list. A
worker that needs shard *i* verifies only shard *i* — **partial verification
is the point**, since nobody will ever hash 128 TB.

---

## 3. Reconstruction, with graceful degradation

For each shard, in order, stopping at the first that verifies:

1. **Regenerate** from the generator spec (no network; fastest).
2. **Fetch** from the nearest NRP S3 region (cache).
3. **Fetch** from another region, then the durable mirror, then origin.

**Why regeneration is best-effort, not guaranteed.** Bit-exact regeneration
requires a fixed RNG stream *and* fixed floating-point behaviour. NumPy
guarantees `Generator` streams across platforms for a given version, but not
across versions; reductions and BLAS matmuls are threading- and
implementation-dependent; and libm transcendentals can differ by an ULP.
Therefore:

- the generator avoids BLAS reductions and transcendentals in the emission
  path wherever possible, and pins its toolchain in the manifest;
- a regenerated shard is **accepted only if it matches the shard root**;
- a mismatch is not an error but a **cache miss** — the worker falls through
  to a byte source and records the event.

This inverts the usual fragility: determinism becomes an optimization
(skip the download when it works), and correctness is guaranteed by the hash
either way. Regeneration-success rate across toolchains is itself a reportable
measurement.

---

## 4. Phased build-out

**Phase 1 — single pool, signed root, two public fallbacks.** One NRP S3
bucket (west), public-read; SHA-256 chunking; GPG-signed root; fallbacks =
Zenodo DOI + origin. *Testable now at 10⁷ rows;* the reconstruction
experiment (§6) runs entirely within this phase.

**Phase 2 — regions and placement.** Replicate to central and east; an
application-level locator resolves the nearest region first, measured rather
than assumed (per-region fetch latency from pods is itself a result worth
publishing — we already measured 10× node variance on this cluster). Explicit
replica placement so a region loss is a known, not discovered, condition.

**Phase 3 — demand-driven caching and scale.** Fetch-on-miss populates the
nearest region; Merkle chunking allows partial and parallel verification;
independent external mirrors (institutional, HF Hub) remove single-provider
dependence; trillion-scale construction becomes tractable *because no party
ever holds the corpus* — each worker verifies and discards.

**Standing constraint (from NRP's own policy).** NRP is explicitly not
archival storage and unused volumes may be purged after ~6 months. NRP pools
are therefore **caches by definition**, and the durable root must live in
Zenodo/HF/institutional storage. The design must survive the loss of every
NRP replica without loss of the corpus — which, given §3, it does.

---

## 5. Open questions to resolve with NRP

1. S3 quota per namespace, and whether a multi-TB public benchmark bucket is
   acceptable use (their docs do not state a limit).
2. Whether public-read buckets are appropriate for third-party-licensed
   mirrors, or only for material we generate.
3. Egress expectations if external mirrors pull from the NRP pool.

---

## 6. The reconstruction experiment (the claim to be tested)

**Setup.** Publish a corpus at 10⁷ rows under Phase 1. Build an index from
locally materialized shards; record per-query results and the index bytes.

**Procedure.** Delete every materialized source file and every local shard,
retaining only the signed root manifest and the public key. Reconstruct the
shards on a fresh pod from whatever sources resolve — deliberately disabling
some, so the mixture includes regeneration, cache, and mirror.

**Pass criteria (registered before the run):**
1. every shard verifies against its Merkle root;
2. reconstructed shard bytes are identical to the originals;
3. an index rebuilt from them returns **identical** results for a fixed query
   set (not merely similar recall);
4. the signature over the root verifies, and a deliberately corrupted chunk
   is detected and rejected.

**Reported regardless:** regeneration-success rate per toolchain, per-source
fetch latency, and total bytes moved — the last being the number that makes
the case at trillion scale.

A pass licenses the claim: *the corpus is a cryptographically defined
distributed object rather than a pile of files.* It does not license claims
about durability beyond the mirrors that actually exist, and the manifest
records exactly which those are.
