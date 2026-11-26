# What an independent reader found in the TQE1 and TQIX specifications

Issue #166 asks for a reader written against the documented layouts, and says that any discrepancy between the
documents and the bytes is a finding to report rather than a bug to work around. These are the findings, measured
on 2026-11-26 against `docs/FORMAT_SPEC.md`, `docs/FORMATS.md`, the reference implementation at commit `5389c05`
(turboquant-pro 2.0.0a3) and NumPy 1.26.4 to 2.5.5. None of them changes a stored byte. They concern what a reader
must know to decode, and what the spec promises.

## 1. The spec does not say how the rotation is derived from the seed

`FORMAT_SPEC.md` step 3 of the decode algorithm says to apply "the inverse rotation R(seed)^T", with `qr` being
"full QR for dim ≤ 4096, structured sign-flip + permutation otherwise" and `hadamard` using "the seed-derived ±1
sign vector". It names no generator, no sampling method and no draw order, and the versioning section still says
the rotation is "fully determined by (bits, dim, seed, rotation, codebook) ... so records are portable across
machines and language bindings".

From the spec alone, then, a vector cannot be decoded. This reader had to reproduce what the writer does, which is
`np.random.default_rng(seed)`, that is `Generator(PCG64(SeedSequence(seed)))`, followed by:

- `qr`, dim ≤ 4096: `standard_normal((dim, dim))` (NumPy's ziggurat, row-major, one 64-bit draw per value unless
  the ziggurat rejects), cast to float32, then the Q factor of `np.linalg.qr` (Householder reflectors with
  LAPACK's sign convention, `beta = -sign(alpha) ||x||`);
- `qr`, dim > 4096: `choice([-1.0, 1.0], size=dim)` (one 32-bit draw per sign, Lemire's bounded method, -1 when
  the draw's top bit is 0), then `permutation(dim)` (Fisher-Yates from the last element down, each index by
  masked rejection on 32-bit draws), where PCG64's 32-bit draws hand out the low half of a 64-bit output first and
  keep the high half buffered, so an odd `dim` carries half a draw from the signs into the permutation;
- `hadamard`: the same sign draws.

`src/rng.rs` ports each of these from NumPy's source and pins them with known answers.

**Recommendation.** Add a normative "Rotation derivation" section to `FORMAT_SPEC.md` that names these algorithms,
or that names NumPy's implementations as normative and this crate as a second, executable statement of them.

## 2. The default rotation is reproducible only to rounding, in any implementation

For `qr` at dim ≤ 4096, which is the default and the common case, Q is computed by the single-precision LAPACK
routines of whichever BLAS NumPy is linked with, and `y @ Q` by BLAS. Their last bits depend on the build and the
CPU. Two Python installations can therefore decode the same record to vectors that differ in the last bits, and
this reader, which forms Q in double precision from the same reflectors, cannot match either one bit for bit.

Measured, coordinate error over `f32::EPSILON × norm`: at most 0.4 against NumPy 2.5.5 with OpenBLAS 0.3.34
(dims 100 to 4096, where 53% to 91% of values differ in the last bits), and up to 1.22 against the committed
golden corpus, whose three oldest `qr` expectations differ more than its two newer ones (0.37 and 0.44), which is
consistent with their having been computed under a different BLAS (the corpus does not record which).
`tests/golden/tqe1/generate.py` already allows for this ("so adding a case never re-derives an old expectation
under a different BLAS"), but the spec does not.

The first CI run gave a direct comparison. The conformance corpus is written afresh on each machine from the same
seeds, and with the same NumPy 2.5.5 and OpenBLAS 0.3.34 the number of decoded values that differ from this
reader's output changed between Atlas and the GitHub runner: 723 against 721 at dim 256 (Lloyd-Max, 3 bits), 3,431
against 3,438 at dim 1024, 14,885 against 14,883 at dim 4096. So at least one of the two decoders gives last bits
that depend on the machine, and the BLAS-dependent one is the likely one. Both stayed within 0.4 × epsilon × norm.

The writer side has the same exposure. Codes come from rotating with that same Q, so on another BLAS a coordinate
within rounding of a cell boundary can quantize to the neighbouring index. `generate.py` re-encodes every pinned
case and stops with "the writer no longer reproduces the pinned bytes; that is a format break" when the bytes
differ, so run on a machine whose BLAS differs it could report a format break that is not one. (The CI test
`test_golden_files_are_immutable` only hashes the committed files and is not affected.)

The other two rotations are bit-exact here, against the same Python decoder: 0 differing values over 39,680
(`hadamard`, dims 32 to 8192) and 52,800 (structured `qr`, dims 4097 to 5003).

**Recommendation.** Say in the spec that dense `qr` decodes to within a stated tolerance (this reader's tests use
4 × `f32::EPSILON` × norm per coordinate), and correct the "Conformance" section, which asks for "bit-identical
reconstruction" while the repository's own golden tests compare with `rtol=1e-5, atol=1e-6`. Where bit-exact
decoding across machines matters, the `hadamard` rotation provides it.

## 3. Decoding depends on NumPy keeping its random streams, which it does not promise

NEP 19, NumPy's policy for its random module, says that "breaking stream-compatibility in order to introduce new
features or improve performance will be *allowed* with *caution*" in feature releases. The `qr` and `hadamard`
rotations are built from those streams, so a future NumPy could change what every stored `qr` and `hadamard`
record decodes to, while turboquant-pro declares `numpy>=1.21`.

Measured so far, the streams have not moved: the known answers in `src/kat.rs` (from NumPy 2.4.6) are identical
under NumPy 1.26.4, 2.0.2 and 2.2.6, and the conformance corpus written under 2.5.5 decodes with them. NumPy 1.21,
the declared minimum, was not tested (it does not install on the Python 3.12 used).

**Recommendation.** Treat the stream as part of the format. The Rust port now fixes it independently of NumPy, and
the `rust-reader` workflow runs weekly, so a NumPy release that changes the streams fails a check. A stronger fix
is for the Python writer and decoder to stop depending on NumPy's `Generator` for the rotation, with a frozen
generator of their own (this crate's `rng.rs` is one), which would also make `numpy>=1.21` safe as declared.

## 4. `codelen` is not `ceil(dim * bits / 8)` at 3 bits

`FORMAT_SPEC.md` says "In every version `codelen = ceil(dim * bits / 8)` (padding rules per `bits` below)". The
packer pads to whole groups: at 3 bits a group is 8 values in 3 bytes, so `codelen = ceil(dim / 8) * 3`. For
`dim = 100` the writer stores 39 bytes, where the formula gives 38. At 2 and 4 bits the two agree.

**Recommendation.** Replace the formula with `ceil(dim / g) * (g * bits / 8)` for the group size
`g = 8 / gcd(8, bits)`, which is `packed_codes.packed_nbytes`.

## 5. TQIX: what the Python reader does not check

`index_file.read_container` verifies magic, bounds and every CRC32, as documented. Three things it does not do,
which a strict reader might:

- **Duplicate section names.** The writer refuses them, but the reader builds a dictionary, so a file with two
  sections of one name opens with the last one silently winning. This reader rejects such a file
  (`Error::DuplicateSection`).
- **Reserved and flags fields.** They are written as 0 and never checked on read. This reader records them and does
  not reject non-zero values, which leaves them free for a future extension. The spec could say which is intended.
- **Container version.** `read_container` accepts any version. The index layer checks (`TQEIndex.open` supports 1
  to 4), and so does this reader's `Index::open`, but the container document (`FORMATS.md` section 2) does not say
  that the version's meaning belongs to the stored format.

**Recommendation.** Reject duplicate names on read, and state the policy for `reserved`, `flags` and the version in
`FORMATS.md`.

## Checked and cleared: the centroid scaling

The spec writes `codebook(bits)[indices] / sqrt(dim)`, a division, as `contrib/tqe1_reader.py` computes it, while
the reference decoder multiplies by `1 / sqrt(dim)` in double precision before rounding to float32. The two could
in principle differ in the last bit. They do not: over every dimension a TQE1 record can carry (1 to 65,535), both
codebooks and all three widths, 0 of 3,669,960 float32 centroids differ. No change is needed, though the spec
could name the multiplication to say exactly what the reference does.
