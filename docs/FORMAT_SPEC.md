# TQE1 — TurboQuant Compressed-Embedding Format (v1 / v2)

A small, versioned, **self-describing** container for one compressed embedding, so a
reader can reconstruct a vector with no out-of-band metadata. Format stability is a
first-class goal: a vector written by any future version of turboquant-pro that
declares a given `version` decodes identically with the algorithm below.

Reference implementation: [`turboquant_pro/format.py`](../turboquant_pro/format.py);
conformance tests: [`tests/test_format.py`](../tests/test_format.py).

## Record layout (little-endian)

**Version 1** (20-byte header) — the default `"qr"` rotation. Byte-identical to
prior releases.

| offset | size | field | type | notes |
|---:|---:|---|---|---|
| 0 | 4 | `magic` | bytes | ASCII `"TQE1"` |
| 4 | 1 | `version` | uint8 | `1`; readers MUST reject unknown versions |
| 5 | 1 | `bits` | uint8 | quantization width: `2`, `3`, or `4` |
| 6 | 2 | `dim` | uint16 | quantized dimension `d'` (number of code indices) |
| 8 | 4 | `seed` | uint32 | rotation seed → reproduces the rotation + Lloyd-Max codebook |
| 12 | 4 | `norm` | float32 | original L2 norm of the vector |
| 16 | 4 | `codelen` | uint32 | length in bytes of the packed code block |
| 20 | `codelen` | `codes` | bytes | bit-packed `dim` indices, `bits` each (see packing) |

Header is **20 bytes**; total record = `20 + codelen`.

**Version 2** (21-byte header) — adds a `rotation` byte after `norm` so a
non-default rotation family (e.g. the opt-in Hadamard rotation) is fully
self-describing. A writer emits v2 **only** when `rotation != "qr"`, so v1 stays the
common on-disk case and existing data/readers are unaffected.

| offset | size | field | type | notes |
|---:|---:|---|---|---|
| 0 | 4 | `magic` | bytes | ASCII `"TQE1"` |
| 4 | 1 | `version` | uint8 | `2` |
| 5 | 1 | `bits` | uint8 | `2`, `3`, or `4` |
| 6 | 2 | `dim` | uint16 | quantized dimension `d'` |
| 8 | 4 | `seed` | uint32 | rotation seed |
| 12 | 4 | `norm` | float32 | original L2 norm |
| 16 | 1 | `rotation` | uint8 | `0` = `qr`, `1` = `hadamard` |
| 17 | 4 | `codelen` | uint32 | length in bytes of the packed code block |
| 21 | `codelen` | `codes` | bytes | bit-packed indices |

Header is **21 bytes**; total record = `21 + codelen`. In both versions
`codelen = ceil(dim * bits / 8)` (padding rules per `bits` below). A batch file is a
back-to-back concatenation of records of either version; `record_size()` reads each
record's `version` byte to advance the cursor correctly.

## Bit-packing
Indices are packed LSB-first into bytes:
- **2-bit:** 4 indices/byte (pad the final group to a multiple of 4).
- **3-bit:** 8 indices into 3 bytes (pad to a multiple of 8).
- **4-bit:** 2 indices/byte.

## Decode algorithm
Given a record and `(bits, dim, seed, norm, codes)`:
1. Unpack `codes` → `dim` integer indices in `[0, 2^bits)`.
2. Look up centroid values `c = codebook(bits)[indices]`, where `codebook(bits)` is
   the fixed Lloyd-Max table for a unit-Gaussian coordinate scaled by `1/sqrt(dim)`.
3. Apply the inverse rotation `R(seed)^T` to obtain the reconstructed unit vector.
   The rotation family is `qr` for v1 records and the `rotation` field for v2:
   - `qr` — full QR for `dim ≤ 4096`, structured sign-flip + permutation otherwise.
   - `hadamard` — randomized Fast Walsh-Hadamard, `R = (1/sqrt(dim)) · H · diag(s)`
     with `s` the seed-derived ±1 sign vector; requires `dim` a power of two.
4. Multiply by `norm`.

For *search*, decode is unnecessary: scores are computed directly on the codes by
asymmetric distance (see `ADCIndex`), which is exact w.r.t. step 1–4.

## Versioning & compatibility policy
- The `magic`+`version` prefix is permanent. A breaking change increments `version`
  (and may change `magic` to `TQE2…`).
- Readers MUST reject records whose `version` they do not implement.
- Within a version, the header layout and decode algorithm are frozen; only
  additive, length-delimited trailers (after `codes`) may be introduced, and readers
  MAY ignore trailing bytes within a record's declared size.
- The codebook and rotation are fully determined by `(bits, dim, seed, rotation)`
  (with `rotation = qr` implied for v1), so records are portable across machines and
  language bindings.

## Conformance
An implementation conforms if, for all `bits ∈ {2,3,4}` and representative `dim`:
round-tripping `pack`→`unpack` preserves `(bits, dim, norm, codes)` and yields
bit-identical reconstruction; bad magic, unknown version, and truncated records raise
errors. These are exercised in `tests/test_format.py`.
