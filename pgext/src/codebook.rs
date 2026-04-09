//! Lloyd-Max codebooks and quantization boundaries.
//!
//! Precomputed for standard normal N(0,1) distribution.
//! At runtime, coordinates are scaled by 1/sqrt(dim) before lookup.

/// 2-bit Lloyd-Max centroids (4 values)
pub const CODEBOOK_2BIT: [f32; 4] = [-1.510, -0.453, 0.453, 1.510];

/// 3-bit Lloyd-Max centroids (8 values)
pub const CODEBOOK_3BIT: [f32; 8] = [
    -1.748, -1.050, -0.500, -0.069,
     0.069,  0.500,  1.050,  1.748,
];

/// 4-bit Lloyd-Max centroids (16 values)
pub const CODEBOOK_4BIT: [f32; 16] = [
    -2.401, -1.844, -1.437, -1.099,
    -0.800, -0.523, -0.258,  0.000,
     0.258,  0.523,  0.800,  1.099,
     1.437,  1.844,  2.401,  2.401,
];

/// 2-bit quantization boundaries (3 values)
pub const BOUNDS_2BIT: [f32; 3] = [-0.9815, 0.0, 0.9815];

/// 3-bit quantization boundaries (7 values)
pub const BOUNDS_3BIT: [f32; 7] = [
    -1.399, -0.775, -0.2845, 0.0,
     0.2845,  0.775,  1.399,
];

/// Quantize a scalar value to 2-bit index.
#[inline]
pub fn quantize_2bit(val: f32) -> u8 {
    if val < BOUNDS_2BIT[1] {
        if val < BOUNDS_2BIT[0] { 0 } else { 1 }
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
            if val < BOUNDS_3BIT[0] { 0 } else { 1 }
        } else if val < BOUNDS_3BIT[2] {
            2
        } else {
            3
        }
    } else if val < BOUNDS_3BIT[5] {
        if val < BOUNDS_3BIT[4] { 4 } else { 5 }
    } else if val < BOUNDS_3BIT[6] {
        6
    } else {
        7
    }
}

/// Quantize a scalar value to 4-bit index.
#[inline]
pub fn quantize_4bit(val: f32) -> u8 {
    for i in 0..15 {
        let mid = (CODEBOOK_4BIT[i] + CODEBOOK_4BIT[i + 1]) / 2.0;
        if val < mid {
            return i as u8;
        }
    }
    15
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
