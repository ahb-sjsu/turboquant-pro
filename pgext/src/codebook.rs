//! Scalar codebooks and quantization boundaries for this extension's format.
//!
//! These tables were described as the Lloyd-Max quantizer for N(0,1). They are
//! not: at 3 and 4 bits their mean-square error is above Lloyd-Max, and the
//! 4-bit table here also differs from the Python package's `legacy` table (it has
//! a 0.000 level and a repeated 2.401). Stored rows depend on them, so they are
//! frozen (see `format_tests.rs`); a better table would be a new format version.
//! The Lloyd-Max values are listed in `docs/FORMAT_SPEC.md` (Codebooks).
//! At runtime, coordinates are scaled by 1/sqrt(dim) before lookup.

/// 2-bit centroids (4 values), frozen format table
pub const CODEBOOK_2BIT: [f32; 4] = [-1.510, -0.453, 0.453, 1.510];

/// 3-bit centroids (8 values), frozen format table
#[rustfmt::skip]
pub const CODEBOOK_3BIT: [f32; 8] = [
    -1.748, -1.050, -0.500, -0.069,
     0.069,  0.500,  1.050,  1.748,
];

/// 4-bit centroids (16 values), frozen format table
#[rustfmt::skip]
pub const CODEBOOK_4BIT: [f32; 16] = [
    -2.401, -1.844, -1.437, -1.099,
    -0.800, -0.523, -0.258,  0.000,
     0.258,  0.523,  0.800,  1.099,
     1.437,  1.844,  2.401,  2.401,
];

/// 2-bit quantization boundaries (3 values)
pub const BOUNDS_2BIT: [f32; 3] = [-0.9815, 0.0, 0.9815];

/// 3-bit quantization boundaries (7 values)
pub const BOUNDS_3BIT: [f32; 7] = [-1.399, -0.775, -0.2845, 0.0, 0.2845, 0.775, 1.399];

/// 4-bit quantization boundaries (15 values): the midpoints between adjacent
/// entries of [`CODEBOOK_4BIT`], computed at compile time with the same f32
/// expression the linear scan used at run time, so every stored row encodes
/// exactly as before. The last boundary equals 2.401, the repeated top entry.
pub const BOUNDS_4BIT: [f32; 15] = midpoints_4bit();

const fn midpoints_4bit() -> [f32; 15] {
    let mut bounds = [0.0f32; 15];
    let mut i = 0;
    while i < 15 {
        bounds[i] = (CODEBOOK_4BIT[i] + CODEBOOK_4BIT[i + 1]) / 2.0;
        i += 1;
    }
    bounds
}

/// Quantize a scalar value to 2-bit index.
#[inline]
pub fn quantize_2bit(val: f32) -> u8 {
    if val < BOUNDS_2BIT[1] {
        if val < BOUNDS_2BIT[0] {
            0
        } else {
            1
        }
    } else if val < BOUNDS_2BIT[2] {
        2
    } else {
        3
    }
}

/// Quantize a scalar value to 3-bit index.
#[inline]
pub fn quantize_3bit(val: f32) -> u8 {
    if val < BOUNDS_3BIT[3] {
        if val < BOUNDS_3BIT[1] {
            if val < BOUNDS_3BIT[0] {
                0
            } else {
                1
            }
        } else if val < BOUNDS_3BIT[2] {
            2
        } else {
            3
        }
    } else if val < BOUNDS_3BIT[5] {
        if val < BOUNDS_3BIT[4] {
            4
        } else {
            5
        }
    } else if val < BOUNDS_3BIT[6] {
        6
    } else {
        7
    }
}

/// Quantize a scalar value to 4-bit index: four comparisons down a balanced
/// tree over [`BOUNDS_4BIT`]. Every `false` branch goes to the higher indices,
/// so NaN lands on 15, as it did with the linear scan.
#[inline]
pub fn quantize_4bit(val: f32) -> u8 {
    let b = &BOUNDS_4BIT;
    if val < b[7] {
        if val < b[3] {
            if val < b[1] {
                if val < b[0] {
                    0
                } else {
                    1
                }
            } else if val < b[2] {
                2
            } else {
                3
            }
        } else if val < b[5] {
            if val < b[4] {
                4
            } else {
                5
            }
        } else if val < b[6] {
            6
        } else {
            7
        }
    } else if val < b[11] {
        if val < b[9] {
            if val < b[8] {
                8
            } else {
                9
            }
        } else if val < b[10] {
            10
        } else {
            11
        }
    } else if val < b[13] {
        if val < b[12] {
            12
        } else {
            13
        }
    } else if val < b[14] {
        14
    } else {
        15
    }
}

/// Get the codebook for a given bit width.
pub fn codebook(bits: u8) -> &'static [f32] {
    match bits {
        2 => &CODEBOOK_2BIT,
        3 => &CODEBOOK_3BIT,
        4 => &CODEBOOK_4BIT,
        _ => &CODEBOOK_3BIT,
    }
}

/// Quantize a value at the given bit width.
#[inline]
pub fn quantize(val: f32, bits: u8) -> u8 {
    match bits {
        2 => quantize_2bit(val),
        3 => quantize_3bit(val),
        4 => quantize_4bit(val),
        _ => quantize_3bit(val),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The linear scan `quantize_4bit` used before the decision tree (issue
    /// #18), kept as the reference the tree must match on every input.
    fn quantize_4bit_linear(val: f32) -> u8 {
        for i in 0..15 {
            let mid = (CODEBOOK_4BIT[i] + CODEBOOK_4BIT[i + 1]) / 2.0;
            if val < mid {
                return i as u8;
            }
        }
        15
    }

    #[test]
    fn bounds_4bit_are_the_runtime_midpoints_bit_for_bit() {
        for (i, b) in BOUNDS_4BIT.iter().enumerate() {
            let mid = (std::hint::black_box(CODEBOOK_4BIT[i]) + CODEBOOK_4BIT[i + 1]) / 2.0;
            assert_eq!(b.to_bits(), mid.to_bits(), "boundary {i}");
        }
        assert!(
            BOUNDS_4BIT.windows(2).all(|w| w[0] <= w[1]),
            "boundaries must be sorted for the tree"
        );
    }

    #[test]
    fn quantize_4bit_tree_matches_the_linear_scan() {
        let mut probes = vec![
            0.0f32,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::NAN,
        ];
        // Every boundary and the floats on either side of it: the only places a
        // wrong comparison or an off-by-one index can show.
        for b in BOUNDS_4BIT {
            probes.extend([b.next_down(), b, b.next_up()]);
        }
        // A dense sweep over the range the scaled coordinates occupy.
        probes.extend((-40_000..=40_000).map(|k| k as f32 * 1e-4));
        for v in probes {
            assert_eq!(
                quantize_4bit(v),
                quantize_4bit_linear(v),
                "value {v:e} ({:#010x})",
                v.to_bits()
            );
        }
    }

    #[test]
    fn quantize_4bit_maps_each_centroid_to_its_own_index() {
        // The top entry is repeated (2.401 twice), so its first copy encodes as
        // 15, the index the linear scan gave it; every other centroid is its own.
        for (i, &c) in CODEBOOK_4BIT.iter().enumerate().take(14) {
            assert_eq!(quantize_4bit(c) as usize, i, "centroid {i} = {c}");
        }
        assert_eq!(quantize_4bit(CODEBOOK_4BIT[14]), 15);
        assert_eq!(quantize_4bit(CODEBOOK_4BIT[15]), 15);
    }
}
