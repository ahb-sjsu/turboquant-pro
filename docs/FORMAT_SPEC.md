# TQE1 — TurboQuant Compressed-Embedding Format (v1 / v2 / v3)

A small, versioned, **self-describing** container for one compressed embedding, so a
reader can reconstruct a vector with no out-of-band metadata. Format stability is a
first-class goal: a vector written by any future version of turboquant-pro that
declares a given `version` decodes identically with the algorithm below.

Reference implementation: [`turboquant_pro/format.py`](../turboquant_pro/format.py);
conformance tests: [`tests/test_format.py`](../tests/test_format.py).

> **Portability proof (2.0 Pillar 2):** a **single-file, dependency-free
> reference reader** — [`contrib/tqe1_reader.py`](../contrib/tqe1_reader.py)
> (stdlib + numpy, no turboquant-pro import; vendor the one file and you can
> read TQE1 forever) — validated against the committed **golden corpus**
> ([`tests/golden/tqe1/`](../tests/golden/tqe1/): pinned `.tqe` bytes +
> expected decoded tensors + sha256 manifest) by
> [`tests/test_tqe1_golden.py`](../tests/test_tqe1_golden.py). A third-party
> implementation conforms iff it reproduces `expected.npz` from the `.tqe`
> bytes alone; an in-tree writer change that alters the golden bytes is a
> format break and fails CI.

> **Scope note — this spec is the per-vector TQE1 record only.** The *persisted
> index* is a separate format (**TQIX**, `turboquant_pro/index_file.py`) with its
> own version field. Its **v3** (new in 1.9.0) adds a lossless compact re-encoding
> — bit-packed codes at slot granularity, elided `arange` ids, dropped empty
> tombstones, `uint32` IVF member sidecars — that is bit-identical to v2 on
> reconstruction and rankings. That is an index-level concern and does **not**
> change the TQE1 record below. See [FORMATS.md § TQIX](FORMATS.md#2-tqix--the-persisted-index-container).

## Record layout (little-endian)

**Version 1** (20-byte header) — the default `"qr"` rotation. Byte-identical to
prior releases.

| offset | size | field | type | notes |
|---:|---:|---|---|---|
| 0 | 4 | `magic` | bytes | ASCII `"TQE1"` |
| 4 | 1 | `version` | uint8 | `1`; readers MUST reject unknown versions |
| 5 | 1 | `bits` | uint8 | quantization width: `2`, `3`, or `4` |
| 6 | 2 | `dim` | uint16 | quantized dimension `d'` (number of code indices) |
| 8 | 4 | `seed` | uint32 | rotation seed → reproduces the rotation |
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

Header is **21 bytes**; total record = `21 + codelen`.

**Version 3** (22-byte header) — adds a `codebook` byte after `rotation` so the
scalar table the indices refer to is self-describing. A writer emits v3 **only**
when `codebook != legacy`, so v1/v2 records are unaffected.

| offset | size | field | type | notes |
|---:|---:|---|---|---|
| 0 | 4 | `magic` | bytes | ASCII `"TQE1"` |
| 4 | 1 | `version` | uint8 | `3` |
| 5 | 1 | `bits` | uint8 | `2`, `3`, or `4` |
| 6 | 2 | `dim` | uint16 | quantized dimension `d'` |
| 8 | 4 | `seed` | uint32 | rotation seed |
| 12 | 4 | `norm` | float32 | original L2 norm |
| 16 | 1 | `rotation` | uint8 | `0` = `qr`, `1` = `hadamard` |
| 17 | 1 | `codebook` | uint8 | `0` = `legacy`, `1` = `lloyd-max` |
| 18 | 4 | `codelen` | uint32 | length in bytes of the packed code block |
| 22 | `codelen` | `codes` | bytes | bit-packed indices |

Header is **22 bytes**; total record = `22 + codelen`. In every version
`codelen = ceil(dim * bits / 8)` (padding rules per `bits` below), and `codelen` is
the last four bytes of the header. A batch file is a back-to-back concatenation of
records of any version; `record_size()` reads each record's `version` byte to
advance the cursor correctly.

## Codebooks
The codebook is normative: the values below, not a recomputation of them. v1 and
v2 records always use `legacy`; v3 records name theirs.

- **`legacy`** — the table every release before v3 wrote. It is **not** the
  Lloyd-Max quantizer for N(0, 1) at 3 and 4 bits (its N(0, 1) mean-square error is
  0.045556 and 0.010803, against 0.034548 and 0.009501 for Lloyd-Max). It is frozen
  because stored data depends on it.
  - 2-bit: `-1.510, -0.453, 0.453, 1.510`
  - 3-bit: `-1.748, -1.050, -0.500, -0.069, 0.069, 0.500, 1.050, 1.748`
  - 4-bit: `-2.401, -1.844, -1.437, -1.099, -0.800, -0.524, -0.262, -0.066,
    0.066, 0.262, 0.524, 0.800, 1.099, 1.437, 1.844, 2.401`
- **`lloyd-max`** — the Lloyd-Max (MSE-optimal) quantizer for N(0, 1), a fixed
  point of Lloyd's iteration rounded to six decimals. Symmetric; positive half:
  - 2-bit: `0.452780, 1.510418`
  - 3-bit: `0.245094, 0.756005, 1.343909, 2.151946`
  - 4-bit: `0.128395, 0.388048, 0.656759, 0.942340, 1.256231, 1.618046,
    2.069017, 2.732590`

Index `i` maps to the `i`-th value in ascending order. Encoders assign each
coordinate to the nearest value (cell boundaries at midpoints).

## Bit-packing
Indices are packed LSB-first into bytes:
- **2-bit:** 4 indices/byte (pad the final group to a multiple of 4).
- **3-bit:** 8 indices into 3 bytes (pad to a multiple of 8).
- **4-bit:** 2 indices/byte.

## Decode algorithm
Given a record and `(bits, dim, seed, norm, codes)`:
1. Unpack `codes` → `dim` integer indices in `[0, 2^bits)`.
2. Look up centroid values `c = codebook(bits)[indices] / sqrt(dim)`, where
   `codebook(bits)` is the table of the record's codebook (see **Codebooks**:
   `legacy` for v1/v2, the `codebook` field for v3).
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
- The codebook and rotation are fully determined by
  `(bits, dim, seed, rotation, codebook)` (with `rotation = qr` implied for v1 and
  `codebook = legacy` implied for v1 and v2), so records are portable across
  machines and language bindings.
- A codebook table is never edited in place. An improved table is a new
  `codebook` value, which only a v3 record can carry, so a reader that predates it
  rejects the record rather than decoding it against the wrong table.

## Conformance
An implementation conforms if, for all `bits ∈ {2,3,4}`, both codebooks, and
representative `dim`:
round-tripping `pack`→`unpack` preserves `(bits, dim, norm, codes)` and yields
bit-identical reconstruction; bad magic, unknown version, and truncated records raise
errors. These are exercised in `tests/test_format.py`.
