# tqformat: a standalone Rust reader for TQE1 and TQIX

A read-only, dependency-free Rust implementation of turboquant-pro's two persisted formats (issue #166):

- **TQE1** records, versions 1 to 3 (`docs/FORMAT_SPEC.md`): parse, unpack the 2-, 3- and 4-bit codes, and
  reconstruct the vector as the Python decoder does, including the rotation regenerated from the stored seed.
- **TQIX** containers (`docs/FORMATS.md` section 2): verify every section's CRC32, then read a persisted index on
  top of it, its `meta`, typed arrays, unpacked codes, ids and tombstones. `TQPW` (packed weights) containers are
  read at the container level.

Unknown versions are rejected, and corrupt or truncated input is a typed `Error`. Writing is out of scope.

The point of a second implementation is to test the claim that the formats are self-describing and stable. What
that turned up is in [FINDINGS.md](FINDINGS.md). The main points were that the spec did not say how the rotation is
derived from the seed, so this reader reproduces NumPy's random streams exactly, and that the default `qr`
rotation can only be reproduced to rounding, by any implementation, Python included. The spec now states both
(`FORMAT_SPEC.md`, "Rotation derivation" and "Conformance"), and the Python package draws its rotations from a
frozen generator (`turboquant_pro/frozen_rng.py`) that this crate's `rng.rs` matches.

## Exactness

| record | reconstruction | measured against the Python decoder |
|---|---|---|
| `hadamard` rotation | bit-exact | 0 differing values over 39,680 (dims 32 to 8192) |
| `qr` rotation, dim > 4096 (sign flip and permutation) | bit-exact | 0 differing values over 52,800 (dims 4097, 4100, 5003) |
| `qr` rotation, dim ≤ 4096 (dense Q) | to rounding | worst coordinate error 0.4 × f32 epsilon × norm against NumPy 2.5.5 with OpenBLAS 0.3.34 (dims 100 to 4096), 1.22 × against the committed golden corpus (dim 32) |

The golden corpus's expectations were computed when each case was added, and its three oldest `qr` files differ
from this reader by up to 1.22 × epsilon × norm, its two newer ones by at most 0.44 ×. That is consistent with
expectations computed under different BLAS builds (the corpus does not record which), the subject of the next
paragraph.

The dense case is not bit-exact because NumPy's Q comes from LAPACK (`sgeqrf`/`sorgqr` of whichever BLAS NumPy is
built with) and the product `y @ Q` from BLAS, whose last bits depend on the build and the CPU. This reader forms Q
from the same Householder reflectors with LAPACK's sign convention, in double precision, and rounds once. The
tests allow 4 × epsilon × norm. A wrong seed or a wrong draw order gives errors of the order of the norm, so that
bound cannot pass a wrong rotation. `Decoder::exactness` and `tqread info` say which case a record is in.

## Use

```rust
let bytes = std::fs::read("vectors.tqe")?;
let mut decoder = tqformat::decode::Decoder::new(); // caches each regenerated rotation
for rec in tqformat::tqe::parse_batch(&bytes)? {
    let v: Vec<f32> = decoder.decode(&rec)?;
}
let index = tqformat::index::Index::open("index.tqix")?; // every section CRC-checked
let (codes, rows, dim) = index.codes()?;
let pca_mean = index.array("pca_mean")?.to_f32()?;
```

```text
cargo run --release --bin tqread -- info  FILE           # records, or sections and arrays
cargo run --release --bin tqread -- decode FILE.tqe OUT  # every record as little-endian float32
```

Regenerating a dense Q costs O(dim³) once per `(dim, seed)`: about 80 s on one core at dim 4096 (Atlas, Xeon
E5-2690 v3), well under a second at dim 1024. NumPy takes about 7 s at dim 4096 with a 4-thread OpenBLAS. The
`hadamard` and structured rotations cost O(dim log dim) and O(dim).

## Tests

```text
cargo test                                      # unit tests, NumPy known answers, the golden corpus
python conformance/make_corpus.py /tmp/corpus   # from the repository root, with turboquant-pro installed
TQFORMAT_CORPUS=/tmp/corpus cargo test --test conformance -- --nocapture
```

- **Known answers** (`src/kat.rs`, written by `tools/make_kat.py` from NumPy 2.4.6) pin the ported streams:
  SeedSequence mixing, PCG64 output, 2,000,000 ziggurat normals (tail and wedge branches included), the sign
  draws, and the permutation, with the 32-bit buffer that one leaves for the other.
- **The golden corpus** (`tests/golden/tqe1/` of the repository) is decoded and compared with `expected.npz`.
- **Cross-language conformance** (`conformance/make_corpus.py`, `tests/conformance.rs`): Python writes TQE1 batches
  at 12 dimension, rotation, codebook and seed combinations, and four indexes (format versions 2, 3 with implicit
  ids, 3 with deletions, and 4). Rust decodes the records against Python's own decode, and checks every index
  section, array, unpacked code, id and tombstone byte for byte, and that a flipped byte is caught.

CI runs all three (`.github/workflows/rust-reader.yml`), on changes to the reader or the Python format code, and
weekly, so that a NumPy release that changes the streams fails a check instead of changing decodes silently.

## Provenance and licence

MIT, as the rest of turboquant-pro. The NumPy-derived parts (the random-stream port in `src/rng.rs` and the
ziggurat tables in `src/ziggurat.rs`, copied textually from NumPy's `ziggurat_constants.h` by
`tools/make_ziggurat.py`) are under NumPy's BSD-3 licence, reproduced in [NOTICE](NOTICE). The crate is not
published to crates.io (`publish = false`).
