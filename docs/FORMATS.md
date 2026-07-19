# Formats at a glance

The four on-disk / in-contract formats TurboQuant Pro ships, side by side. Each is
**versioned** and **self-describing** — a reader reconstructs the data with no
out-of-band metadata — because a format that drifts is not an industry-standard
tool. Full specs are linked per section.

| Format | Magic | Version field | What it holds | Integrity | Spec |
|---|---|---|---|---|---|
| **TQE1** record | `TQE1` | `uint8` (1, 2) | one compressed embedding/KV vector | length-checked | [FORMAT_SPEC.md](FORMAT_SPEC.md) |
| **TQIX** index | `TQIX` | `uint16` (1, 2, 3) | a whole persisted ADC index | **CRC32 per section** | [index_file.py](../turboquant_pro/index_file.py) |
| **Plugin** container | — (in-memory) | plugin-defined | one quantizer's compressed output | conformance kit | [PLUGINS.md](PLUGINS.md) |
| **Certificate** JSON | `schema` field | `schema_version` int | a distribution-free rank floor | JSON Schema + golden | [CERTIFICATE_SPEC.md](CERTIFICATE_SPEC.md) |

---

## 1. TQE1 — the compressed-vector record

The atom: one embedding (or KV vector) as a self-describing little-endian record.
`version==1` is the 20-byte default (`"qr"` rotation, byte-identical to early
releases); `version==2` inserts a rotation byte.

```
v1 (20-byte header)                       v2 adds one byte:
 off  sz  field                            off sz field
 0    4   magic   b"TQE1"                   16  1  rot  uint8 (0=qr, 1=hadamard)
 4    1   version uint8 (==1)               (codelen + codes shift by 1)
 5    1   bits    uint8 (2|3|4)
 6    2   dim     uint16   (quantized dim)
 8    4   seed    uint32   (reproduces codebook + rotation)
 12   4   norm    float32  (per-vector L2 norm)
 16   4   codelen uint32
 20   ..  codes   codelen bytes (bit-packed indices)
```

- **Self-describing decode:** `bits`, `dim`, `seed`, `norm`, and rotation travel
  *with* the record, so `unpack(buf)` reconstructs it with no side metadata (a wrong
  seed silently decodes to a different vector — so it must never live out-of-band).
- **Batches:** `pack_batch` concatenates records back-to-back; `record_size` walks
  them. No container header — TQIX is the container.
- **Forward compat:** readers reject an unknown `version` and may ignore trailing
  bytes after `codes` within a record.

## 2. TQIX — the persisted index container

The whole `TQEIndex` on disk: a versioned directory of CRC-checked byte sections.
Fixed 12-byte header, then a section table, then payloads.

```
header (12 bytes)                         directory: n_sections × 56 bytes
 off  sz  field                            off sz field
 0    4   magic   b"TQIX"                   0   32 name    utf-8, null-padded
 4    2   version uint16                    32  8  offset  uint64 (from file start)
 6    2   n_sections uint16                 40  8  length  uint64
 8    4   reserved uint32                   48  4  crc32   uint32 (of the bytes)
                                            52  4  flags   uint32
 then: section payloads, back to back
```

- **Corruption is detected, never silent:** every section carries a CRC32;
  `read_container` raises `IndexCorruptionError` on any mismatch or truncation. A
  single-byte-flip fuzzer guards the invariant "detected, or byte-identical."
- **Atomic writes:** written to a temp file and `os.replace`d in — a crash mid-write
  never leaves a torn index.
- **Sections the index layer stores:** `meta` (JSON: format version, tool version,
  PCA/quant params, ids/next_id, dtypes), `pca_mean` / `pca_components` /
  `pca_eigenvalues` / `pca_all_eigenvalues`, the ADC payload `codes` / `cnorm` /
  `vrnorm`, `ids`, `tombstones`, and optional `originals` (for exact rerank +
  certify).
- **Versioning:** v1 = implicit positional ids; v2 = explicit ids + tombstone bitmap;
  v3 (new in 1.9.0) = a **lossless compact re-encoding** of the v2 sections —
  reconstruction and rankings are bit-identical to v2 (asserted by tests, not
  sampled). `tqp index migrate --to-version 2` upgrades in place, as does
  `TQEIndex.migrate(3)`. See the
  [production lifecycle guide](guides/production_lifecycle.md).
- **v3 compaction (1.9.0):** the payload shrinks four ways without touching
  fidelity — (1) sub-byte quantizer codes are **bit-packed at slot granularity**
  (2 codes/byte at 3–4 bits, 4 codes/byte at 2-bit, vs one byte per code in v2);
  (2) `arange`-reconstructible `ids` are **elided from the file entirely** (a
  `meta` field `ids_arange_start` records the start) and empty tombstones are
  dropped; (3) IVF member sidecars shrink to `uint32`. Measured **24.1 B/row**
  all-in (codes 12 + norms 8 + members 4) vs **41 B/row** for the same layout in
  v2, at 2M rows / 4-bit / `--no-originals` (~1.7× smaller).
- **v3 compatibility:** v1/v2 files keep opening unchanged, and a writer can pin
  `format_version=2` to emit for old readers. On a memory-mapped open a
  `PackedCodes` view unpacks only the rows a probe actually gathers — so packing
  also halves code I/O on the storage-bound path — while a RAM open unpacks once.

## 3. Plugin container — a contract, not a byte layout

A quantizer plugin's `compress(x)` returns an **opaque container** of its own
design; the *contract* (not a fixed layout) is what makes it interoperable. The
executable [conformance kit](PLUGINS.md) enforces it:

| Capability | Method(s) | Conformance check |
|---|---|---|
| **Required** round-trip | `compress` / `decompress` | shape, finite, error envelope |
| Bit-packing | `compress(packed=True)`, container `.packed` | packed ≡ unpacked decode, exercised |
| **Affine** (fused decode) | `grid_params` / `codes` / `outlier_csr` | `mu + weight·grid[codes]` (+overlay) ≡ decompress, exactly |
| Serialization | container `to_bytes` / `from_bytes` | byte round-trip ≡ decompress |
| Native dtype | `native_dtype()` | passthrough naming |

A container that exposes the **affine** capability inherits the fused
compute-on-codes decode with zero kernel work (keys enter attention only through
`q·k`, so score equality follows from container equality). See the reference
implementation in
[`tqp-reference-plugin`](https://github.com/ahb-sjsu/tqp-reference-plugin), which
passes roundtrip + packed + affine + serialization.

## 4. Certificate JSON — the durable rank floor

`tqp certify` emits a provenance-stamped, schema-locked JSON. Non-finite
measurements serialize as `null` (never bare `NaN`), so it is always spec-valid.

```jsonc
{
  "schema": "turboquant-pro/rank-certificate",   // format identity
  "schema_version": 1,                            // bumps only on breaking change
  "tool_version": "1.8.0",
  "created_utc": "2026-07-17T…Z",
  "inputs": { "original": {"path","shape","dtype","sha256"},
              "reconstructed": {…} },             // provenance: what was certified
  "params": { "metric": "cosine", "n_anchors": 200, "seed": 0 },
  "certificate": {
    "kappa": 1.02, "mu_hat": 0.06,                // measured distortion + concentration
    "tau_floor": 0.876, "spearman_floor": 0.814,  // GUARANTEED rank floors
    "n_pairs": 2016, "max_certifiable_kappa": 1.19,
    "vacuous": false                              // true => no positive floor
  },
  "interpretation": "…",
  "passed": true
}
```

- **Compatibility promise** ([CERTIFICATE_SPEC.md](CERTIFICATE_SPEC.md)):
  `schema_version` bumps only on a breaking change; fields are additive within v1;
  `null` always means non-finite; provenance (`inputs.*.sha256`, `tool_version`,
  `created_utc`) is guaranteed present.
- **Drift-guarded:** a committed golden fixture is regenerated and compared in CI, so
  the format cannot silently drift.
- **Optional additive sections** (still `schema_version` 1): `task` (declared
  consumer + target), `environment` (tool/python/numpy/platform/git/hardware), and
  `limitations` — plus a `--html` human report. A base certificate carries none of
  them; their presence never bumps the version.
- **What it means:** a *guaranteed* floor on rank preservation, never reconstruction
  quality — see the [certification guide](guides/certification.md).

---

Everything here obeys the project's one rule: acceptance is rank fidelity / a
certificate / the consumer metric — **never reconstruction cosine.** See the
[documentation hub](README.md).
