//! TQE1 records (`turboquant_pro/format.py`, `docs/FORMAT_SPEC.md`).
//!
//! Three header versions, all little-endian, all starting `magic b"TQE1", version u8, bits u8, dim u16, seed u32,
//! norm f32`: version 1 then has `codelen u32` (20 bytes, the default `qr` rotation and `legacy` codebook),
//! version 2 a `rotation u8` before `codelen` (21 bytes), version 3 `rotation u8, codebook u8` before `codelen`
//! (22 bytes). The packed codes follow the header. A batch file is records back to back.

use crate::codebook::Codebook;
use crate::error::{Error, Result};

pub const MAGIC: [u8; 4] = *b"TQE1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rotation {
    /// Code 0. Full QR of a seeded Gaussian matrix for `dim <= 4096`, a seeded sign flip and permutation above.
    Qr,
    /// Code 1. Randomized fast Walsh-Hadamard transform; needs a power-of-two `dim`.
    Hadamard,
}

impl Rotation {
    pub fn from_code(c: u8) -> Result<Rotation> {
        match c {
            0 => Ok(Rotation::Qr),
            1 => Ok(Rotation::Hadamard),
            _ => Err(Error::UnknownRotation(c)),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Rotation::Qr => "qr",
            Rotation::Hadamard => "hadamard",
        }
    }
}

/// One parsed record. `codes` borrows from the input.
#[derive(Debug, Clone, PartialEq)]
pub struct Record<'a> {
    pub version: u8,
    pub bits: u8,
    /// The quantized dimension, the number of code indices.
    pub dim: u16,
    pub seed: u32,
    pub norm: f32,
    pub rotation: Rotation,
    pub codebook: Codebook,
    pub codes: &'a [u8],
    /// Header plus `codelen`: where the next record of a batch starts.
    pub size: usize,
}

/// Header size of a record version; unknown versions are rejected, as the spec requires.
pub fn header_size(version: u8) -> Result<usize> {
    match version {
        1 => Ok(20),
        2 => Ok(21),
        3 => Ok(22),
        v => Err(Error::UnsupportedVersion { format: "TQE1", version: v as u32 }),
    }
}

/// Parse the record that starts at `buf[0]`.
pub fn parse(buf: &[u8]) -> Result<Record<'_>> {
    if buf.len() < 20 {
        return Err(Error::Truncated { what: "buffer too small for a TQE1 header".into() });
    }
    if buf[0..4] != MAGIC {
        return Err(Error::BadMagic { expected: MAGIC, found: buf[0..4].to_vec() });
    }
    let version = buf[4];
    let hsize = header_size(version)?;
    if buf.len() < hsize {
        return Err(Error::Truncated { what: format!("buffer too small for a TQE1 v{version} header") });
    }
    let bits = buf[5];
    if !matches!(bits, 2..=4) {
        return Err(Error::UnsupportedBits(bits));
    }
    let dim = u16::from_le_bytes([buf[6], buf[7]]);
    let seed = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    let norm = f32::from_le_bytes(buf[12..16].try_into().unwrap());
    let (rotation, codebook) = match version {
        1 => (Rotation::Qr, Codebook::Legacy),
        2 => (Rotation::from_code(buf[16])?, Codebook::Legacy),
        _ => (Rotation::from_code(buf[16])?, Codebook::from_code(buf[17])?),
    };
    let codelen = u32::from_le_bytes(buf[hsize - 4..hsize].try_into().unwrap()) as usize;
    let end = hsize + codelen;
    if buf.len() < end {
        return Err(Error::Truncated { what: "codes shorter than codelen".into() });
    }
    Ok(Record { version, bits, dim, seed, norm, rotation, codebook, codes: &buf[hsize..end], size: end })
}

/// Parse a batch: records back to back, each of any version, to the end of `buf`.
pub fn parse_batch(buf: &[u8]) -> Result<Vec<Record<'_>>> {
    let mut out = Vec::new();
    let mut off = 0;
    while off < buf.len() {
        let r = parse(&buf[off..])?;
        off += r.size;
        out.push(r);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(version: u8, bits: u8, extra: &[u8], codes: &[u8]) -> Vec<u8> {
        let mut b = MAGIC.to_vec();
        b.push(version);
        b.push(bits);
        b.extend_from_slice(&8u16.to_le_bytes());
        b.extend_from_slice(&42u32.to_le_bytes());
        b.extend_from_slice(&1.5f32.to_le_bytes());
        b.extend_from_slice(extra);
        b.extend_from_slice(&(codes.len() as u32).to_le_bytes());
        b.extend_from_slice(codes);
        b
    }

    #[test]
    fn versions_and_errors() {
        let v1 = header(1, 3, &[], &[1, 2, 3]);
        let r = parse(&v1).unwrap();
        assert_eq!((r.version, r.bits, r.dim, r.seed, r.norm, r.size), (1, 3, 8, 42, 1.5, 23));
        assert_eq!((r.rotation, r.codebook, r.codes), (Rotation::Qr, Codebook::Legacy, &[1u8, 2, 3][..]));
        let v2 = header(2, 2, &[1], &[9, 9]);
        assert_eq!(parse(&v2).unwrap().rotation, Rotation::Hadamard);
        let v3 = header(3, 4, &[0, 1], &[7; 4]);
        let r3 = parse(&v3).unwrap();
        assert_eq!((r3.rotation, r3.codebook, r3.size), (Rotation::Qr, Codebook::LloydMax, 26));

        let mut batch = v1.clone();
        batch.extend_from_slice(&v2);
        batch.extend_from_slice(&v3);
        assert_eq!(parse_batch(&batch).unwrap().len(), 3);

        assert!(matches!(parse(&v1[..19]), Err(Error::Truncated { .. })));
        assert!(matches!(parse(&v1[..22]), Err(Error::Truncated { .. })));
        let mut bad = v1.clone();
        bad[0] = b'X';
        assert!(matches!(parse(&bad), Err(Error::BadMagic { .. })));
        assert_eq!(
            parse(&header(4, 3, &[0, 0, 0], &[])).unwrap_err(),
            Error::UnsupportedVersion { format: "TQE1", version: 4 }
        );
        assert_eq!(parse(&header(1, 5, &[], &[])).unwrap_err(), Error::UnsupportedBits(5));
        assert_eq!(parse(&header(2, 3, &[2], &[])).unwrap_err(), Error::UnknownRotation(2));
        assert_eq!(parse(&header(3, 3, &[0, 2], &[])).unwrap_err(), Error::UnknownCodebook(2));
    }
}
