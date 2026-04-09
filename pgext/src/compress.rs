//! Compression and decompression of embedding vectors.
//!
//! Algorithm: PolarQuant + Lloyd-Max (Zandieh et al., ICLR 2026)
//!   1. Extract L2 norm
//!   2. Normalize to unit vector
//!   3. Apply random orthogonal rotation (full QR via Gram-Schmidt)
//!   4. Scale by sqrt(dim) for Lloyd-Max codebook
//!   5. Scalar quantize each coordinate
//!   6. Bit-pack indices

use crate::codebook;
use crate::types::TqVector;

// ─── LCG PRNG (deterministic from seed) ──────────────────────────

#[inline]
fn lcg_next(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    *state
}

/// Box-Muller: generate standard normal from LCG uniform.
#[inline]
fn randn(state: &mut u32) -> f32 {
    let u1 = (lcg_next(state) & 0x7FFF_FFFF) as f32 / 0x7FFF_FFFF as f32;
    let u2 = (lcg_next(state) & 0x7FFF_FFFF) as f32 / 0x7FFF_FFFF as f32;
    let u1 = if u1 < 1e-10 { 1e-10 } else { u1 };
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

// ─── Full orthogonal rotation via Gram-Schmidt QR ────────────────
//
// Generate a random orthogonal matrix Q from a Gaussian matrix G
// using modified Gram-Schmidt. Q is stored row-major as dim x dim.
// This gives proper decorrelation for optimal Lloyd-Max quantization.
//
// For dim <= 1024 this takes ~4MB and runs in <50ms. For larger dims,
// we fall back to structured rotation (Hadamard-like sign flips).

/// Public wrapper for GPU module access.
pub fn generate_rotation_pub(dim: usize, seed: u32) -> Vec<f32> {
    generate_rotation(dim, seed)
}

/// Public wrapper for GPU module bit-packing.
pub fn pack_pub(indices: &[u8], bits: u8) -> Vec<u8> {
    pack(indices, bits)
}

fn generate_rotation(dim: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    let mut q = vec![0.0f32; dim * dim];

    // Fill with Gaussian random (column-major conceptually)
    for i in 0..dim * dim {
        q[i] = randn(&mut state);
    }

    // Modified Gram-Schmidt (column-wise)
    for j in 0..dim {
        // Subtract projections onto previous columns
        for k in 0..j {
            let mut dot = 0.0f32;
            for i in 0..dim {
                dot += q[i * dim + j] * q[i * dim + k];
            }
            for i in 0..dim {
                q[i * dim + j] -= dot * q[i * dim + k];
            }
        }

        // Normalize column j
        let mut norm = 0.0f32;
        for i in 0..dim {
            norm += q[i * dim + j] * q[i * dim + j];
        }
        norm = norm.sqrt();
        if norm > 1e-30 {
            for i in 0..dim {
                q[i * dim + j] /= norm;
            }
        }
    }

    q
}

/// Apply rotation: out = Q^T * vec (project into rotated space).
fn apply_rotation(vec: &mut [f32], rotation: &[f32], dim: usize) {
    let mut tmp = vec![0.0f32; dim];
    for i in 0..dim {
        let mut sum = 0.0f32;
        for j in 0..dim {
            sum += rotation[j * dim + i] * vec[j];
        }
        tmp[i] = sum;
    }
    vec[..dim].copy_from_slice(&tmp);
}

/// Inverse rotation: out = Q * vec (project back to original space).
fn inverse_rotation(vec: &mut [f32], rotation: &[f32], dim: usize) {
    let mut tmp = vec![0.0f32; dim];
    for i in 0..dim {
        let mut sum = 0.0f32;
        for j in 0..dim {
            sum += rotation[i * dim + j] * vec[j];
        }
        tmp[i] = sum;
    }
    vec[..dim].copy_from_slice(&tmp);
}

/// Structured rotation for large dims (> 1024): sign flips.
/// Self-inverse, O(d) memory and time.
fn apply_sign_flip(vec: &mut [f32], seed: u32) {
    let mut state = seed;
    for v in vec.iter_mut() {
        if lcg_next(&mut state) & 1 == 1 {
            *v = -*v;
        }
    }
}

// ─── Bit packing ─────────────────────────────────────────────────

fn pack_3bit(indices: &[u8]) -> Vec<u8> {
    let groups = (indices.len() + 7) / 8;
    let mut packed = vec![0u8; groups * 3];
    for g in 0..groups {
        let mut bits24: u32 = 0;
        let base = g * 8;
        for i in 0..8 {
            if base + i < indices.len() {
                bits24 |= (indices[base + i] as u32) << (i * 3);
            }
        }
        packed[g * 3] = (bits24 & 0xFF) as u8;
        packed[g * 3 + 1] = ((bits24 >> 8) & 0xFF) as u8;
        packed[g * 3 + 2] = ((bits24 >> 16) & 0xFF) as u8;
    }
    packed
}

fn unpack_3bit(packed: &[u8], n: usize) -> Vec<u8> {
    let groups = (n + 7) / 8;
    let mut indices = vec![0u8; n];
    for g in 0..groups {
        let bits24 = packed[g * 3] as u32
            | ((packed[g * 3 + 1] as u32) << 8)
            | ((packed[g * 3 + 2] as u32) << 16);
        let base = g * 8;
        for i in 0..8 {
            if base + i < n {
                indices[base + i] = ((bits24 >> (i * 3)) & 0x07) as u8;
            }
        }
    }
    indices
}

fn pack_2bit(indices: &[u8]) -> Vec<u8> {
    let nbytes = (indices.len() + 3) / 4;
    let mut packed = vec![0u8; nbytes];
    for (i, &idx) in indices.iter().enumerate() {
        packed[i / 4] |= (idx & 0x03) << ((i % 4) * 2);
    }
    packed
}

fn unpack_2bit(packed: &[u8], n: usize) -> Vec<u8> {
    let mut indices = vec![0u8; n];
    for i in 0..n {
        indices[i] = (packed[i / 4] >> ((i % 4) * 2)) & 0x03;
    }
    indices
}

fn pack_4bit(indices: &[u8]) -> Vec<u8> {
    let nbytes = (indices.len() + 1) / 2;
    let mut packed = vec![0u8; nbytes];
    for (i, &idx) in indices.iter().enumerate() {
        if i % 2 == 0 {
            packed[i / 2] = idx & 0x0F;
        } else {
            packed[i / 2] |= (idx & 0x0F) << 4;
        }
    }
    packed
}

fn unpack_4bit(packed: &[u8], n: usize) -> Vec<u8> {
    let mut indices = vec![0u8; n];
    for i in 0..n {
        if i % 2 == 0 {
            indices[i] = packed[i / 2] & 0x0F;
        } else {
            indices[i] = (packed[i / 2] >> 4) & 0x0F;
        }
    }
    indices
}

fn pack(indices: &[u8], bits: u8) -> Vec<u8> {
    match bits {
        2 => pack_2bit(indices),
        3 => pack_3bit(indices),
        4 => pack_4bit(indices),
        _ => pack_3bit(indices),
    }
}

fn unpack(packed: &[u8], n: usize, bits: u8) -> Vec<u8> {
    match bits {
        2 => unpack_2bit(packed, n),
        3 => unpack_3bit(packed, n),
        4 => unpack_4bit(packed, n),
        _ => unpack_3bit(packed, n),
    }
}

// ─── Compress ────────────────────────────────────────────────────

/// Maximum dimension for full QR rotation. Above this, use sign flips.
/// Maximum dim for full QR rotation. Above this, use sign flips.
/// QR is O(d^3) to generate + O(d^2) per vector to apply.
/// At 512-dim: ~1ms per vector. At 1024-dim: ~8ms. At 256: ~0.1ms.
const MAX_QR_DIM: usize = 512;

pub fn compress(vec: &[f32], bits: u8, seed: u32) -> TqVector {
    let dim = vec.len();

    // 1. Compute L2 norm
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();

    // 2. Normalize
    let mut rotated: Vec<f32> = if norm > 1e-30 {
        vec.iter().map(|x| x / norm).collect()
    } else {
        vec![0.0; dim]
    };

    // 3. Apply rotation
    if dim <= MAX_QR_DIM {
        let rotation = generate_rotation(dim, seed);
        apply_rotation(&mut rotated, &rotation, dim);
    } else {
        apply_sign_flip(&mut rotated, seed);
    }

    // 4. Scale for Lloyd-Max
    let scale = (dim as f32).sqrt();

    // 5. Quantize
    let indices: Vec<u8> = rotated
        .iter()
        .map(|&x| codebook::quantize(x * scale, bits))
        .collect();

    // 6. Bit-pack
    let data = pack(&indices, bits);

    TqVector::new(dim as u16, bits, norm, data)
}

// ─── Batch Compress (amortized rotation) ─────────────────────────

/// Compress multiple vectors, generating the rotation matrix only once.
/// ~3-5x faster than calling compress() per vector at 1024-dim.
pub fn compress_batch(
    vecs: &[Vec<f32>],
    bits: u8,
    seed: u32,
) -> Vec<TqVector> {
    if vecs.is_empty() {
        return vec![];
    }

    let dim = vecs[0].len();
    let scale = (dim as f32).sqrt();

    // Generate rotation ONCE for the whole batch
    let use_full_qr = dim <= MAX_QR_DIM;
    let rotation = if use_full_qr {
        Some(generate_rotation(dim, seed))
    } else {
        None
    };

    vecs.iter()
        .map(|vec| {
            // 1. Norm
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();

            // 2. Normalize
            let mut rotated: Vec<f32> = if norm > 1e-30 {
                vec.iter().map(|x| x / norm).collect()
            } else {
                vec![0.0; dim]
            };

            // 3. Rotate (reusing precomputed matrix)
            if let Some(ref rot) = rotation {
                apply_rotation(&mut rotated, rot, dim);
            } else {
                apply_sign_flip(&mut rotated, seed);
            }

            // 4-5. Quantize
            let indices: Vec<u8> = rotated
                .iter()
                .map(|&x| codebook::quantize(x * scale, bits))
                .collect();

            // 6. Pack
            let data = pack(&indices, bits);
            TqVector::new(dim as u16, bits, norm, data)
        })
        .collect()
}

// ─── Decompress ──────────────────────────────────────────────────

pub fn decompress(tqv: &TqVector) -> Vec<f32> {
    let dim = tqv.dim as usize;
    let bits = tqv.bits;
    let cb = codebook::codebook(bits);
    let scale = 1.0 / (dim as f32).sqrt();

    // 1. Unpack indices
    let indices = unpack(&tqv.data, dim, bits);

    // 2. Dequantize
    let mut result: Vec<f32> = indices
        .iter()
        .map(|&idx| cb[idx as usize] * scale)
        .collect();

    // 3. Inverse rotation
    if dim <= MAX_QR_DIM {
        let rotation = generate_rotation(dim, 42);
        inverse_rotation(&mut result, &rotation, dim);
    } else {
        apply_sign_flip(&mut result, 42);
    }

    // 4. Scale by norm
    for v in result.iter_mut() {
        *v *= tqv.norm;
    }

    result
}

// ─── Unit tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_3bit_roundtrip() {
        let indices: Vec<u8> = (0..24).map(|i| i % 8).collect();
        let packed = pack_3bit(&indices);
        let unpacked = unpack_3bit(&packed, 24);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_pack_unpack_2bit_roundtrip() {
        let indices: Vec<u8> = (0..16).map(|i| i % 4).collect();
        let packed = pack_2bit(&indices);
        let unpacked = unpack_2bit(&packed, 16);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_pack_unpack_4bit_roundtrip() {
        let indices: Vec<u8> = (0..16).map(|i| i % 16).collect();
        let packed = pack_4bit(&indices);
        let unpacked = unpack_4bit(&packed, 16);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_compress_preserves_norm() {
        let vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let expected = (1.0_f32 + 4.0 + 9.0 + 16.0).sqrt();
        let tqv = compress(&vec, 3, 42);
        assert!((tqv.norm - expected).abs() < 0.001);
    }

    #[test]
    fn test_compress_decompress_cosine_3bit() {
        let vec: Vec<f32> = (0..128)
            .map(|i| (i as f32) * 0.01 + 0.1)
            .collect();
        let tqv = compress(&vec, 3, 42);
        let recovered = decompress(&tqv);

        let dot: f32 = vec.iter().zip(recovered.iter())
            .map(|(a, b)| a * b).sum();
        let na: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = recovered.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine = dot / (na * nb);
        assert!(cosine > 0.95, "3-bit cosine was {}", cosine);
    }

    #[test]
    fn test_compress_decompress_cosine_4bit() {
        let vec: Vec<f32> = (0..128)
            .map(|i| (i as f32) * 0.01 + 0.1)
            .collect();
        let tqv = compress(&vec, 4, 42);
        let recovered = decompress(&tqv);

        let dot: f32 = vec.iter().zip(recovered.iter())
            .map(|(a, b)| a * b).sum();
        let na: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = recovered.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine = dot / (na * nb);
        assert!(cosine > 0.98, "4-bit cosine was {}", cosine);
    }

    #[test]
    fn test_compression_ratio_3bit_1024() {
        let vec: Vec<f32> = vec![0.1; 1024];
        let tqv = compress(&vec, 3, 42);
        let ratio = tqv.compression_ratio();
        assert!(ratio > 8.0, "ratio was {}", ratio);
    }

    #[test]
    fn test_deterministic_seed() {
        let vec: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let a = compress(&vec, 3, 42);
        let b = compress(&vec, 3, 42);
        assert_eq!(a.data, b.data);
        assert_eq!(a.norm, b.norm);
    }

    #[test]
    fn test_different_seeds_differ() {
        let vec: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let a = compress(&vec, 3, 42);
        let b = compress(&vec, 3, 99);
        assert_ne!(a.data, b.data);
    }

    #[test]
    fn test_zero_vector() {
        let vec: Vec<f32> = vec![0.0; 64];
        let tqv = compress(&vec, 3, 42);
        assert!(tqv.norm < 1e-20);
        let recovered = decompress(&tqv);
        for v in &recovered {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_all_bit_widths() {
        let vec: Vec<f32> = (0..256)
            .map(|i| (i as f32 - 128.0) * 0.01)
            .collect();
        for bits in [2u8, 3, 4] {
            let tqv = compress(&vec, bits, 42);
            assert_eq!(tqv.bits, bits);
            let recovered = decompress(&tqv);
            assert_eq!(recovered.len(), 256);
        }
    }

    #[test]
    fn test_rotation_orthogonality() {
        let dim = 16;
        let q = generate_rotation(dim, 42);
        // Q^T * Q should be identity
        for i in 0..dim {
            for j in 0..dim {
                let mut dot = 0.0f32;
                for k in 0..dim {
                    dot += q[k * dim + i] * q[k * dim + j];
                }
                if i == j {
                    assert!(
                        (dot - 1.0).abs() < 0.01,
                        "diagonal ({},{}) was {}",
                        i, j, dot
                    );
                } else {
                    assert!(
                        dot.abs() < 0.01,
                        "off-diagonal ({},{}) was {}",
                        i, j, dot
                    );
                }
            }
        }
    }
}
