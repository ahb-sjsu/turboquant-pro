//! TQE1 decoding: the spec's four steps, as `TurboQuantPGVector.decompress_embedding` performs them.
//!
//! 1. unpack `dim` indices of `bits` each;
//! 2. look each up in the record's codebook scaled by `1/sqrt(dim)` (see [`crate::codebook::centroids`]);
//! 3. invert the rotation regenerated from `(rotation, dim, seed)` (see [`crate::rotation`]);
//! 4. multiply by `norm`, in single precision.

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use crate::bits::unpack_bits;
use crate::codebook::{centroids, Codebook};
use crate::error::Result;
use crate::rotation::{exactness_of, Exactness, RotationData};
use crate::tqe::{Record, Rotation};

/// Decodes records, keeping each regenerated rotation and centroid table for the next record that shares it (a
/// batch file usually holds one quantizer's records, so the rotation is built once).
#[derive(Debug, Default)]
pub struct Decoder {
    rotations: HashMap<(Rotation, u16, u32), RotationData>,
    tables: HashMap<(u8, Codebook, u16), Vec<f32>>,
}

impl Decoder {
    pub fn new() -> Decoder {
        Decoder::default()
    }

    /// The reconstructed float32 vector of `rec`.
    pub fn decode(&mut self, rec: &Record<'_>) -> Result<Vec<f32>> {
        let dim = rec.dim as usize;
        let idx = unpack_bits(rec.codes, dim, rec.bits)?;
        let key = (rec.bits, rec.codebook, rec.dim);
        if let Entry::Vacant(e) = self.tables.entry(key) {
            e.insert(centroids(rec.bits, rec.codebook, dim)?);
        }
        let table = &self.tables[&key];
        let y: Vec<f32> = idx.iter().map(|&i| table[i as usize]).collect();
        let rot = self.rotation(rec)?;
        let mut x = rot.unrotate(&y);
        for v in &mut x {
            *v *= rec.norm;
        }
        Ok(x)
    }

    /// Whether decoding `rec` is bit-exact against the reference decoder (without building its rotation).
    pub fn exactness(&self, rec: &Record<'_>) -> Result<Exactness> {
        Ok(exactness_of(rec.rotation, rec.dim as usize))
    }

    fn rotation(&mut self, rec: &Record<'_>) -> Result<&RotationData> {
        let key = (rec.rotation, rec.dim, rec.seed);
        if let Entry::Vacant(e) = self.rotations.entry(key) {
            e.insert(RotationData::build(rec.rotation, rec.dim as usize, rec.seed)?);
        }
        Ok(&self.rotations[&key])
    }
}

/// Decode one record without keeping its rotation.
pub fn decode(rec: &Record<'_>) -> Result<Vec<f32>> {
    Decoder::new().decode(rec)
}
