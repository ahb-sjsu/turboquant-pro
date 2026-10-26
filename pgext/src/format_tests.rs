//! The `tqvector` encoding, frozen (issue #165).
//!
//! The existing roundtrip tests check that pack-then-unpack agrees with itself
//! *in the current build*. That is a much weaker property than the one a
//! persisted format needs, because both sides of such a test move together: a
//! change that silently re-encodes every vector still passes. Rows already in
//! a table do not get to move with it.
//!
//! Two layers, as the issue asks.
//!
//! **Golden bytes.** A fixed input, run through today's encoder, must produce
//! the exact bytes recorded here. Any change to a codebook constant, to the
//! packing order, or to the rotation generator fails loudly instead of
//! re-encoding in silence. The fixtures span all three bit widths and both
//! rotation regimes: `MAX_QR_DIM` is **512**, so dims at and below it take the
//! Gram-Schmidt path and dims above it take the sign-flip path. (The issue
//! says 1024; the constant in `compress.rs` says 512, and the fixtures follow
//! the code.)
//!
//! **Properties.** Roundtrip behaviour over arbitrary dims, widths and value
//! distributions, including the cases hand-written tests miss: dims that are
//! not a multiple of 8 for 3-bit packing, single-element vectors, all-zero
//! vectors, and values far outside the codebook's range.
//!
//! Regenerating the fixtures is deliberately a manual act. If a change to the
//! encoder is intended, run
//!
//! ```text
//! TQVECTOR_REGENERATE_FIXTURES=1 cargo test --features pg16 print_golden_fixtures -- --nocapture
//! ```
//!
//! and paste the printed table below, in the same commit as the format version
//! bump, so the diff shows the bytes changing and says why.

use crate::compress::{compress, decompress, pack, unpack};
use crate::types::{FORMAT_VERSION, LEGACY_ROTATION_SEED};

/// A frozen encoding: this input, at this width and seed, produces these bytes.
struct Golden {
    dim: usize,
    bits: u8,
    seed: u32,
    /// `f32::to_bits` of the recorded norm, so the comparison is exact rather
    /// than up to a printed decimal.
    norm_bits: u32,
    data_hex: &'static str,
}

/// The fixture input. Fixed here on purpose: if it ever changes, every golden
/// row below fails, which is the intended signal rather than a nuisance.
fn fixture_vector(dim: usize) -> Vec<f32> {
    (0..dim).map(|i| ((i as f32) * 0.37).sin()).collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn print_golden_fixtures() {
    if std::env::var("TQVECTOR_REGENERATE_FIXTURES").is_err() {
        return;
    }
    for &(dim, bits) in DIMS_AND_WIDTHS {
        let v = fixture_vector(dim);
        let tqv = compress(&v, bits, LEGACY_ROTATION_SEED);
        println!(
            "    Golden {{ dim: {dim}, bits: {bits}, seed: {}, norm_bits: 0x{:08x}, data_hex: \"{}\" }},",
            LEGACY_ROTATION_SEED,
            tqv.norm.to_bits(),
            hex(&tqv.data)
        );
    }
}

/// Both rotation regimes (`MAX_QR_DIM` is 512) at every supported width, plus
/// a dim that is not a multiple of 8 so 3-bit packing's tail is covered.
const DIMS_AND_WIDTHS: &[(usize, u8)] = &[
    (64, 2),
    (64, 3),
    (64, 4),
    (100, 3),
    (512, 2),
    (512, 3),
    (512, 4),
    (600, 3),
    (1024, 4),
];

#[test]
fn golden_bytes_are_unchanged() {
    assert!(
        !GOLDEN.is_empty(),
        "no fixtures recorded; see the module docs for how to generate them"
    );
    for g in GOLDEN {
        let v = fixture_vector(g.dim);
        let tqv = compress(&v, g.bits, g.seed);
        assert_eq!(
            hex(&tqv.data),
            g.data_hex,
            "the encoding changed for dim {} at {} bits. If that was intended, \
             bump FORMAT_VERSION and regenerate the fixtures in the same commit; \
             if it was not, a codebook constant, the packing order or the \
             rotation generator moved underneath rows that are already stored",
            g.dim,
            g.bits
        );
        assert_eq!(
            tqv.norm.to_bits(),
            g.norm_bits,
            "the recorded norm changed for dim {} at {} bits",
            g.dim,
            g.bits
        );
        assert_eq!(tqv.dim as usize, g.dim);
        assert_eq!(tqv.bits, g.bits);
        assert_eq!(tqv.seed, g.seed);
        assert_eq!(
            tqv.format_version, FORMAT_VERSION,
            "a fresh vector must declare the current format version"
        );
    }
}

/// Every fixture must also still decode to something close to its input. A
/// golden test alone would keep passing if the encoder and decoder were
/// changed together in a way that preserved bytes but lost meaning.
#[test]
fn golden_fixtures_still_decode() {
    for g in GOLDEN {
        let v = fixture_vector(g.dim);
        let tqv = compress(&v, g.bits, g.seed);
        let back = decompress(&tqv);
        let cos = cosine(&v, &back);
        let floor = match g.bits {
            2 => 0.75,
            3 => 0.90,
            _ => 0.95,
        };
        assert!(
            cos > floor,
            "dim {} at {} bits decoded to cosine {cos}, below {floor}",
            g.dim,
            g.bits
        );
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-20 || nb < 1e-20 {
        return 0.0;
    }
    dot / (na * nb)
}

// ─── The tables the indices mean nothing without ─────────────────

/// A stored vector is a list of indices. What those indices *mean* is the
/// codebook, so the codebook is part of the persisted format even though it
/// never travels with a row.
///
/// This is not redundant with the golden bytes, and the difference was found
/// by running the acceptance check rather than by reasoning: moving
/// `CODEBOOK_2BIT[0]` from -1.510 to -1.511 changed no golden byte at all,
/// because no value crossed a decision boundary, and every byte fixture passed.
/// It would still have silently changed what every stored 2-bit row decodes to.
///
/// The boundaries are frozen for the same reason from the other side: they
/// decide which index a value gets, so moving one re-encodes new rows while
/// leaving old rows to be read against a table that no longer matches how they
/// were written.
#[test]
fn the_codebooks_and_boundaries_are_frozen() {
    use crate::codebook::{BOUNDS_2BIT, BOUNDS_3BIT, CODEBOOK_2BIT, CODEBOOK_3BIT, CODEBOOK_4BIT};

    assert_eq!(
        CODEBOOK_2BIT,
        [-1.510, -0.453, 0.453, 1.510],
        "the 2-bit centroids changed; every stored 2-bit row now decodes differently"
    );
    assert_eq!(
        CODEBOOK_3BIT,
        [-1.748, -1.050, -0.500, -0.069, 0.069, 0.500, 1.050, 1.748],
        "the 3-bit centroids changed; every stored 3-bit row now decodes differently"
    );
    assert_eq!(
        CODEBOOK_4BIT,
        [
            -2.401, -1.844, -1.437, -1.099, -0.800, -0.523, -0.258, 0.000, 0.258, 0.523, 0.800,
            1.099, 1.437, 1.844, 2.401, 2.401,
        ],
        "the 4-bit centroids changed; every stored 4-bit row now decodes differently"
    );

    assert_eq!(
        BOUNDS_2BIT,
        [-0.9815, 0.0, 0.9815],
        "the 2-bit decision boundaries changed; new rows encode differently from old ones"
    );
    assert_eq!(
        BOUNDS_3BIT,
        [-1.399, -0.775, -0.2845, 0.0, 0.2845, 0.775, 1.399],
        "the 3-bit decision boundaries changed; new rows encode differently from old ones"
    );
    // The 4-bit boundaries are computed at compile time as the midpoints of
    // adjacent centroids (BOUNDS_4BIT), so freezing CODEBOOK_4BIT above freezes
    // them as well; codebook.rs checks they equal the run-time midpoints bit for
    // bit and that quantize_4bit matches the former linear scan.
}

// ─── Properties ──────────────────────────────────────────────────

proptest::proptest! {
    // The rotation is a Gram-Schmidt over a dim x dim matrix, so case count
    // times dimension dominates the run time. The dims here stay small on
    // purpose: the large ones are frozen by the golden fixtures above, and a
    // gate slow enough to be resented is a gate that gets bypassed.
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(64))]

    /// Packing indices and unpacking them must be exactly invertible, at every
    /// width, for any length. The interesting lengths are the ones that are not
    /// a multiple of the width's group size: 3-bit packs eight indices into
    /// three bytes, so a tail of one to seven is where an off-by-one lives.
    #[test]
    fn pack_unpack_is_exactly_invertible(
        bits in proptest::sample::select(vec![2u8, 3, 4]),
        len in 1usize..300,
        seed in proptest::num::u64::ANY,
    ) {
        let max = (1u16 << bits) - 1;
        let mut state = seed;
        let indices: Vec<u8> = (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 33) as u16 % (max + 1)) as u8
            })
            .collect();
        let packed = pack(&indices, bits);
        let back = unpack(&packed, len, bits);
        proptest::prop_assert_eq!(back, indices);
    }

    /// A compressed vector decodes to one that still points the same way. The
    /// floor is per width, and deliberately loose: this property is about the
    /// absence of a structural break (a lost tail, a mis-sized buffer, a
    /// wrong-basis read), not about quantization quality, which the cosine
    /// tests in compress.rs pin.
    #[test]
    fn compress_decompress_keeps_direction(
        bits in proptest::sample::select(vec![2u8, 3, 4]),
        dim in 2usize..130,
        scale in 0.001f32..1000.0,
        seed in proptest::num::u32::ANY,
    ) {
        let v: Vec<f32> = (0..dim)
            .map(|i| ((i as f32) * 0.61).sin() * scale)
            .collect();
        let tqv = compress(&v, bits, seed);
        proptest::prop_assert_eq!(tqv.seed, seed);
        proptest::prop_assert_eq!(tqv.dim as usize, dim);
        let back = decompress(&tqv);
        proptest::prop_assert_eq!(back.len(), dim);
        let floor = if bits == 2 { 0.55 } else { 0.75 };
        let cos = cosine(&v, &back);
        proptest::prop_assert!(
            cos > floor,
            "dim {} bits {} scale {} gave cosine {}",
            dim, bits, scale, cos
        );
    }

    /// The norm is what carries scale through the codec, so it must survive
    /// independently of direction, across a wide dynamic range.
    #[test]
    fn the_norm_survives(
        bits in proptest::sample::select(vec![2u8, 3, 4]),
        dim in 2usize..130,
        scale in 0.001f32..1000.0,
    ) {
        let v: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.23).cos() * scale).collect();
        let expected: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let tqv = compress(&v, bits, 42);
        proptest::prop_assert!(
            (tqv.norm - expected).abs() <= expected * 1e-4,
            "norm {} vs expected {}",
            tqv.norm,
            expected
        );
    }
}

// ─── Degenerate inputs, asserted rather than implied ──────────────

#[test]
fn a_zero_vector_has_zero_norm_and_decodes_to_zeros() {
    for bits in [2u8, 3, 4] {
        let tqv = compress(&vec![0.0; 64], bits, 42);
        assert!(tqv.norm < 1e-20, "zero vector got norm {}", tqv.norm);
        let back = decompress(&tqv);
        assert_eq!(back.len(), 64);
        assert!(
            back.iter().all(|x| x.abs() < 1e-20),
            "a zero vector must decode to zeros, not to the codebook's centre"
        );
    }
}

#[test]
fn a_single_element_vector_survives() {
    for bits in [2u8, 3, 4] {
        let tqv = compress(&[3.5], bits, 42);
        assert_eq!(tqv.dim, 1);
        assert!((tqv.norm - 3.5).abs() < 1e-4);
        assert_eq!(decompress(&tqv).len(), 1);
    }
}

#[test]
fn values_far_outside_the_codebook_clamp_rather_than_wrap() {
    // The codebook covers a few standard deviations. A vector dominated by one
    // enormous component must saturate at the extreme index, not wrap round to
    // the opposite sign, which is what an unclamped cast would do.
    for bits in [2u8, 3, 4] {
        let mut v = vec![0.01f32; 32];
        v[7] = 1.0e6;
        let tqv = compress(&v, bits, 42);
        let back = decompress(&tqv);
        assert!(
            back[7] > 0.0,
            "the dominant component changed sign at {bits} bits: {}",
            back[7]
        );
        assert!(cosine(&v, &back) > 0.5, "bits {bits}");
    }
}

/// One explicit case above `MAX_QR_DIM`, so the sign-flip rotation path is
/// exercised by a roundtrip and not only by the golden bytes. Kept as a single
/// case rather than a property, because its cost is the reason the property's
/// dims are capped.
#[test]
fn a_dim_above_the_qr_threshold_roundtrips() {
    for bits in [2u8, 3, 4] {
        let v: Vec<f32> = (0..600).map(|i| ((i as f32) * 0.19).sin()).collect();
        let tqv = compress(&v, bits, 42);
        let back = decompress(&tqv);
        assert_eq!(back.len(), 600);
        let floor = if bits == 2 { 0.70 } else { 0.85 };
        assert!(cosine(&v, &back) > floor, "bits {bits}");
    }
}

#[test]
fn a_dim_that_is_not_a_multiple_of_eight_roundtrips_at_three_bits() {
    // 3-bit packing groups eight indices into three bytes; the tail is where
    // an off-by-one would live, and it is not covered by the power-of-two dims
    // the other tests use.
    for dim in [1usize, 7, 9, 15, 17, 100, 513] {
        let v: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.41).sin()).collect();
        let tqv = compress(&v, 3, 42);
        let back = decompress(&tqv);
        assert_eq!(back.len(), dim, "dim {dim} lost its tail");
    }
}

/// The frozen encodings. Generated by `print_golden_fixtures` against the
/// encoder at the commit that introduced them; regenerating them is a
/// deliberate act, documented at the top of this file.
const GOLDEN: &[Golden] = &[
    Golden {
        dim: 64,
        bits: 2,
        seed: 42,
        norm_bits: 0x40b40c70,
        data_hex: "3489a9a75dbb82e34dc86e8087c9a961",
    },
    Golden {
        dim: 64,
        bits: 3,
        seed: 42,
        norm_bits: 0x40b40c70,
        data_hex: "89a38671eb95fae4be4dead47b06c37c95841f9ae66bab70",
    },
    Golden {
        dim: 64,
        bits: 4,
        seed: 42,
        norm_bits: 0x40b40c70,
        data_hex: "323b9583a4886b79d5579c8c2a933bb9d673a1b1c84933736c8094e3979a2578",
    },
    Golden {
        dim: 100,
        bits: 3,
        seed: 42,
        norm_bits: 0x40e33d85,
        data_hex: "2911956342cdcee8fe91097bae200d25939d5360d18e55298d53a87222ccf23d9478bd6a510600",
    },
    Golden {
        dim: 512,
        bits: 2,
        seed: 42,
        norm_bits: 0x417f8782,
        data_hex: "885c61538208bfd767e5651585243956816139774f551fd8c22b6bf51245dad6fa6198f9ee35862112991652b7267a2b78b616a59d59b020a5992457e81a997788e4ee33565267b19805516b2b1659ae79771abfb69eb5a144195eca9e866e245ba85a91fed354a9a56956ebb94a45b59fa9d7669a9daf5c78b28a91a7e0596a",
    },
    Golden {
        dim: 512,
        bits: 3,
        seed: 42,
        norm_bits: 0x417f8782,
        data_hex: "601a6f0bf74c458a02f7f9c51727d551a7295b8a11a3516d0aa878aa713d36a44dbe00ca0c6c323627d985b045b46cc9a69d74e0acfa75bd191d2a348e328a9c5048ce5911f4f31ae8579d95a0b9721a4a888b380a9ba6987345614f2ab26a7d618cf4bdff3896566c97a5d8a8b8258a647b2fe309aae2b7b2655daef29b96cd8b8badb418b62efce6c6f5ca807c9738b70297f6a684beefe85094b64b995694e6f6ab4b6612269dbf9abad64e39ac2aab360947a9c5b865a8a8569dd4e2e456",
    },
    Golden {
        dim: 512,
        bits: 4,
        seed: 42,
        norm_bits: 0x417f8782,
        data_hex: "8093c37637683d471893a112be7c6cb35f6856ba5469753576a37207862b59663572367a962b5c4cbb527557db1571b438b28b37ab7855bb18247763a7b65ac68acb347981a785dcb9b9772b79a125393a358775680619554b8d5917a84d9c2aa16f697c4906659ac5947465328b313a459894a4723a5d5483d98835a5954c7d73a342c9d9ea3c2b5a7519775e6b26aba08467333554ab7a8f276a15a545ca99a55b5b5d9a25dc7b5aabd88646bc34a97272a636c766aac3baa64770d868433aad34a088aa662574dbdb2cc541449499467a845958659cd8978b78725672548ddf9593aa6bc6473a9895b485bb88c264935b389b897225845ca833ba74569a59",
    },
    Golden {
        dim: 600,
        bits: 3,
        seed: 42,
        norm_bits: 0x418a9ad9,
        data_hex: "6c0ca75b6338a59cc722e53cd61ac3e9d6184e29e7b158398fb5c6703a398653ca792c36ce63ad319c4e8ee194721e8b8de3586b0ca793e338a59cc762e338d61ac3e9e4384e29e7b1d8388eb5c6703a398e53ca792c368e63ad319c4e8ee394721e8b8de3586b0ca793e338a59cc762e338d61ac3e9e4384e29e7b1d8188eb5c6703a398f53ca792c368663ad319c4ecee394721e8b8de1546b0ca793f358a49cc7626338dd1ac3e9e43c1629e7b1d6184eb7c6704a398f45ca79ac3586d3a9319c52ce63b1711e6b8de174720ca794f3586c9cc75a63389d1cc329e53c161bc7",
    },
    Golden {
        dim: 1024,
        bits: 4,
        seed: 42,
        norm_bits: 0x41b501dd,
        data_hex: "97b3c2a3663a2c4b79b5c2b284482b3c5a97c3c2a3663a2c4b79b5c2b284482b3c5a97c3c2a3663a2c4b79b5c2b284482b3c5a97c3c2a3663a2c4b79b5c2b284482b3c5a97c3c2a3663a2c4b79b5c2b284482b3c5a97c3c293663a2c4b79b5c2b285482b3c5a97c3c293663a2c4b79b5c2b285482b3c6a96c3c293573a2c4b88b5c2b275492b3c6a96c3c293573a2c4b88b5c2b275492b3c6aa6c3c293573a2c4b88b4c2b275492b3c6aa6c3c293573a2c4b88b4c2b275492b3c6aa6c3c293573a2c4b88b4c2b275492b3c6aa6c3c293573a2c4b88b4c2b275492b3c6aa6c3c293573a2c4b88b4c2b375392c3c6aa6c3b294572b2c4b88b4c2a375392c3c6aa6c3b294572b2c4b88b4c2a375392c3c6aa6c3b294572b2c4b88b4c2a375392c3c6aa6c3b294572b2c4b88b4c2a375392c3c6aa6c3b294572b2c4b88b4c2a375392c3c69a6c3b294572b2c5b88b4c2a375392c3c69a6c3b284572b2c5b87b4c2a366392c3c79a5c3b284582b2c5b97b4c2a366392c3c79a5c3b284482b2c5b97b4c2a3663a2c3c79a5c3b284482b2c5b97b4c2a3663a2c3c79a5c3b284482b2c5b97b4c2a3663a2c3c79a5c3b284482b2c5b97b4c2a3663a2c3c79a5c3b284482b2c5b97b4c2a3663a2c3c79b5c3b284482b3c5a97c3c2a3663a2c4b79b5c2b284482b3c5a97c3c2a3663a2c4b79b5c2b284482b3c5a97c3c2a3663a2c4b79b5c2",
    },
];
