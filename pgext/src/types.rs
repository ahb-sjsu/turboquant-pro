//! TqVector — the PostgreSQL custom type for compressed vectors.

use pgrx::prelude::*;
use serde::{Deserialize, Serialize};

/// A TurboQuant compressed vector.
///
/// Stored as CBOR in PostgreSQL via serde. Display format shows
/// a human-readable summary: `tqvector(1024-dim, 3-bit, 10.5x)`.
#[derive(
    Clone, Debug,
    Serialize, Deserialize,
    PostgresType,
)]
#[inoutfuncs]
pub struct TqVector {
    /// Original embedding dimension
    pub dim: u16,
    /// Quantization bit width (2, 3, or 4)
    pub bits: u8,
    /// L2 norm of the original vector
    pub norm: f32,
    /// Bit-packed quantization indices
    pub data: Vec<u8>,
}

impl InOutFuncs for TqVector {
    /// Text input — not supported (use tq_compress instead)
    fn input(_input: &core::ffi::CStr) -> Self
    where
        Self: Sized,
    {
        pgrx::error!(
            "tqvector text input not supported. \
             Use tq_compress(embedding, bits) instead."
        );
    }

    /// Text output — human-readable summary
    fn output(&self, buffer: &mut pgrx::StringInfo) {
        let original = self.dim as f32 * 4.0;
        let compressed = self.data.len() as f32 + 4.0;
        let ratio = original / compressed;
        buffer.push_str(&format!(
            "tqvector({}-dim, {}-bit, norm={:.4}, {:.1}x)",
            self.dim, self.bits, self.norm, ratio
        ));
    }
}

impl TqVector {
    /// Create a new TqVector from components.
    pub fn new(dim: u16, bits: u8, norm: f32, data: Vec<u8>) -> Self {
        Self {
            dim,
            bits,
            norm,
            data,
        }
    }

    /// Calculate packed data size for given dim and bits.
    pub fn packed_size(dim: usize, bits: u8) -> usize {
        match bits {
            2 => (dim + 3) / 4,
            3 => ((dim + 7) / 8) * 3,
            4 => (dim + 1) / 2,
            _ => dim,
        }
    }

    /// Theoretical compression ratio vs float32.
    pub fn compression_ratio(&self) -> f32 {
        let original = self.dim as f32 * 4.0;
        let compressed = self.data.len() as f32 + 4.0;
        original / compressed
    }
}
