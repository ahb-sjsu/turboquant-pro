# TQE1 — TurboQuant Compressed-Embedding Format (v1 / v2 / v3)

A small, versioned, **self-describing** container for one compressed embedding, so a
reader can reconstruct a vector with no out-of-band metadata. Format stability is a
first-class goal: a vector written by any future version of turboquant-pro that
declares a given `version` decodes identically with the algorithm below (for the
dense `qr` rotation, to within the rounding stated under **Conformance**).

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
> bytes alone, within the tolerance under **Conformance**; an in-tree writer
> change that alters the golden bytes is a format break and fails CI.
>
> **Second implementation:** [`rust/tqformat`](../rust/tqformat/) is an independent, dependency-free Rust reader
> for TQE1 and TQIX, checked against the golden corpus and against a corpus the Python package writes in CI.
> What writing it showed about this spec is in [`rust/tqformat/FINDINGS.md`](../rust/tqformat/FINDINGS.md); the
> **Rotation derivation** section and the corrected `codelen` and **Conformance** text below come from it.

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

Header is **22 bytes**; total record = `22 + codelen`. In every version the codes
are packed in whole groups (see **Bit-packing**), so
`codelen = ceil(dim / g) * (g * bits / 8)` with the group size `g = 8 / gcd(8, bits)`:
`ceil(dim / 4)` bytes at 2 bits, `ceil(dim / 8) * 3` at 3 bits and `ceil(dim / 2)` at
4 bits. (At 3 bits this is not `ceil(dim * bits / 8)`: `dim = 100` takes 39 bytes, not
38.) `codelen` is the last four bytes of the header. The rotation is drawn from
`seed` exactly as stored, so a writer refuses a seed outside `0 … 2^32 − 1` rather
than truncating it (a truncated seed would decode against a different rotation). A batch file is a back-to-back concatenation of
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
   `legacy` for v1/v2, the `codebook` field for v3). The table is scaled in double
   precision and rounded to float32 (the reference multiplies by `1 / sqrt(dim)`;
   for every `dim` from 1 to 65535 that gives the same float32 values as dividing).
3. Apply the inverse rotation `R(seed)^T` to obtain the reconstructed unit vector.
   The rotation family is `qr` for v1 records and the `rotation` field for v2 and v3:
   - `qr` — full QR for `dim ≤ 4096`, structured sign-flip + permutation otherwise.
   - `hadamard` — randomized Fast Walsh-Hadamard, `R = (1/sqrt(dim)) · H · diag(s)`
     with `s` the seed-derived ±1 sign vector; requires `dim` a power of two.

   How each rotation is generated from `seed` is normative and given in
   **Rotation derivation**.
4. Multiply by `norm`. Steps 3 and 4 are carried out in float32.

For *search*, decode is unnecessary: scores are computed directly on the codes by
asymmetric distance (see `ADCIndex`), which is exact w.r.t. step 1–4.

## Rotation derivation
The rotation is not stored; every reader regenerates it from `seed`. The random draws
are those of a fixed generator, written out in two executable statements of this
section: [`turboquant_pro/frozen_rng.py`](../turboquant_pro/frozen_rng.py) (with
NumPy's ziggurat tables in
[`turboquant_pro/_ziggurat_tables.py`](../turboquant_pro/_ziggurat_tables.py)) and
[`rust/tqformat/src/rng.rs`](../rust/tqformat/src/rng.rs). Both are pinned by the
same known answers ([`rust/tqformat/src/kat.rs`](../rust/tqformat/src/kat.rs),
[`tests/test_frozen_rng.py`](../tests/test_frozen_rng.py)). The generator draws
exactly what `numpy.random.default_rng(seed)` draws (measured identical in NumPy
1.21.6, 1.26.4, 2.0.2, 2.2.6, 2.4.6 and 2.5.5), which is how every earlier release
generated its rotations, so
existing records decode unchanged. The format does not depend on NumPy, though:
NumPy's policy (NEP 19) allows its streams to change, and a NumPy that changed them
would no longer match this section.

**Generator.** `SeedSequence(seed)`, as in NumPy, mixes the seed's 32-bit words
(lowest first; `0` is one word) into a pool of four words and expands it to four
64-bit words `v0..v3`. A `PCG64` generator starts from them: `inc = (v2·2^64 + v3)·2 + 1`,
`state = inc`, `state += v0·2^64 + v1`, then one step. A step is
`state = state · M + inc (mod 2^128)` with
`M = 2549297995355413924 · 2^64 + 4865540595714422341`. The generator offers three
draws:
- `next64`: step, then the XSL-RR output of the new state, `rotr64(hi64 ^ lo64, state >> 122)`;
- `next32`: if a half is buffered, return it; otherwise take `next64`, return its low
  32 bits and buffer its high 32 bits (`next64` does not touch the buffer);
- `next_double`: `(next64 >> 11) · 2^-53`.

**Draws.**
- `normal()` is NumPy's 256-layer ziggurat (`random_standard_normal`) with NumPy's
  tables `ki`, `wi`, `fi` and constants `r_nor = 3.6541528853610088`,
  `inv_r_nor = 0.27366123732975828` (all in `_ziggurat_tables.py`). Take `r = next64`,
  `idx = r & 0xff`, `rabs = (r >> 9) & (2^52 − 1)`, `x = rabs · wi[idx]` (in double
  precision), negated when bit 8 of `r` is 1. Return `x` if `rabs < ki[idx]`.
  Otherwise, for `idx = 0` draw pairs `xx = −inv_r_nor · log1p(−next_double)`,
  `yy = −log1p(−next_double)` until `yy + yy > xx · xx`, and return `r_nor + xx`,
  negated when bit 8 of `rabs` is 1; for `idx > 0` return `x` if
  `(fi[idx−1] − fi[idx]) · next_double + fi[idx] < exp(−0.5 · x · x)` (each operation
  rounded separately, left to right), and else start again with a new `r`. These two
  tests are the only steps that call the C library (`exp`, `log1p`), as NumPy's do.
- `signs(n)`: `n` values, one `next32` each: `−1.0` when its top bit is 0, `+1.0`
  otherwise (NumPy's `choice([-1.0, 1.0], size=n)`).
- `permutation(n)`: start from `0, 1, ..., n−1`; for `i = n−1` down to `1`, draw
  `j = next32 & mask` with `mask = 2^bitlength(i) − 1` until `j ≤ i`, then swap
  elements `i` and `j` (NumPy's `permutation(n)`).

**Rotations.** Each starts from a fresh generator for `seed`.
- `qr`, `dim ≤ 4096`: `G` = `dim × dim` values of `normal()` filled row by row and
  rounded to float32; `Q` = the Q factor of the Householder QR factorisation `G = QR`
  with LAPACK's sign convention (each reflector maps the column's leading entry
  `alpha` to `beta = −sign(alpha) ·` the column's norm), as `numpy.linalg.qr` gives. The writer rotates with `y = x · Qᵀ`; the reader inverts with `x = y · Q`.
- `qr`, `dim > 4096`: `s = signs(dim)`, then `p = permutation(dim)`. The writer
  computes `y[i] = (x ⊙ s)[p[i]]`; the reader `x[p[i]] = y[i] / s[p[i]]`.
- `hadamard`: `s = signs(dim)` and `c = float32(1 / sqrt(dim))`. The writer computes
  `y = H(x ⊙ s) · c`; the reader `x = (H(y) · c) ⊙ s`. `H` is the unnormalized
  Walsh-Hadamard transform in natural (Sylvester) order, computed in float32 by
  radix-2 butterflies `(a, b) → (a + b, a − b)` over pairs `h` apart, for
  `h = 1, 2, 4, …, dim/2` in that order; any other order rounds differently.

`hadamard`, and `qr` above 4096 dimensions, decode bit for bit in any conforming
reader. Dense `qr` decodes only to within rounding, because `Q` depends on the
floating-point library that factorises `G` (see **Conformance**).

## Versioning & compatibility policy
- The `magic`+`version` prefix is permanent. A breaking change increments `version`
  (and may change `magic` to `TQE2…`).
- Readers MUST reject records whose `version` they do not implement.
- Within a version, the header layout and decode algorithm are frozen; only
  additive, length-delimited trailers (after `codes`) may be introduced, and readers
  MAY ignore trailing bytes within a record's declared size.
- The codebook and rotation are fully determined by
  `(bits, dim, seed, rotation, codebook)` (with `rotation = qr` implied for v1 and
  `codebook = legacy` implied for v1 and v2) through **Rotation derivation**, so
  records are portable across machines and language bindings: exactly for
  `hadamard` and for `qr` above 4096 dimensions, to within rounding for dense `qr`.
- The rotation derivation is frozen with the format. No dependency's random streams
  may change it; a different generator would be a new `rotation` value.
- A codebook table is never edited in place. An improved table is a new
  `codebook` value, which only a v3 record can carry, so a reader that predates it
  rejects the record rather than decoding it against the wrong table.

## Conformance
An implementation conforms if, for all `bits ∈ {2,3,4}`, both codebooks, and
representative `dim`:
- round-tripping `pack`→`unpack` preserves `(version, bits, dim, seed, norm, rotation,
  codebook, codes)` byte for byte;
- bad magic, unknown version, unknown rotation or codebook values, and truncated
  records raise errors;
- decoding matches the reference decoder (the golden corpus's `expected.npz`):
  - `hadamard`, and `qr` with `dim > 4096`: bit-identical float32 values;
  - `qr` with `dim ≤ 4096`: every coordinate within `4 · 2^-23 · norm` of the
    reference, where `2^-23` is float32's machine epsilon. `Q` comes from
    single-precision Householder factorisation, whose last bits depend on the
    BLAS/LAPACK build and the CPU, so two conforming decoders, or the reference on
    two machines, can differ in the last bits. Measured: at most `0.4 · 2^-23 · norm`
    between the Rust reader and NumPy 2.5.5 with OpenBLAS 0.3.34, and `1.22 · 2^-23 ·
    norm` against the golden corpus (written under an unrecorded BLAS).

Writers are held to decoding, not to bytes: on another BLAS, a coordinate within
rounding of a cell boundary can be assigned the neighbouring index, so the same vector
written on two machines may give records that differ in a few codes. Both records are
valid; each decodes as above.

These are exercised in `tests/test_format.py` and `tests/test_codebooks.py` (round
trips and error cases), `tests/test_tqe1_golden.py` (decoding against the corpus)
and the `rust-reader` workflow (a second, independent reader).
