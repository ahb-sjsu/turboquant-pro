//! The scalar codebooks (`turboquant_pro/codebooks.py`, `docs/FORMAT_SPEC.md` "Codebooks"). The tables are
//! normative: these literal values, never a recomputation.

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Codebook {
    /// Code 0: the table every release before TQE1 v3 wrote. Not the Lloyd-Max quantizer at 3 and 4 bits.
    Legacy,
    /// Code 1: the Lloyd-Max quantizer for N(0, 1), rounded to six decimals.
    LloydMax,
}

impl Codebook {
    pub fn from_code(c: u8) -> Result<Codebook> {
        match c {
            0 => Ok(Codebook::Legacy),
            1 => Ok(Codebook::LloydMax),
            _ => Err(Error::UnknownCodebook(c)),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Codebook::Legacy => "legacy",
            Codebook::LloydMax => "lloyd-max",
        }
    }
}

static LEGACY_2: [f64; 4] = [-1.510, -0.453, 0.453, 1.510];
static LEGACY_3: [f64; 8] = [-1.748, -1.050, -0.500, -0.069, 0.069, 0.500, 1.050, 1.748];
static LEGACY_4: [f64; 16] = [
    -2.401, -1.844, -1.437, -1.099, -0.800, -0.524, -0.262, -0.066, 0.066, 0.262, 0.524, 0.800, 1.099, 1.437, 1.844,
    2.401,
];
static LLOYD_2: [f64; 4] = [-1.510418, -0.452780, 0.452780, 1.510418];
static LLOYD_3: [f64; 8] = [-2.151946, -1.343909, -0.756005, -0.245094, 0.245094, 0.756005, 1.343909, 2.151946];
static LLOYD_4: [f64; 16] = [
    -2.732590, -2.069017, -1.618046, -1.256231, -0.942340, -0.656759, -0.388048, -0.128395, 0.128395, 0.388048,
    0.656759, 0.942340, 1.256231, 1.618046, 2.069017, 2.732590,
];

/// The unit-scale table, ascending, for `bits` in {2, 3, 4}.
pub fn table(bits: u8, codebook: Codebook) -> Result<&'static [f64]> {
    Ok(match (codebook, bits) {
        (Codebook::Legacy, 2) => &LEGACY_2,
        (Codebook::Legacy, 3) => &LEGACY_3,
        (Codebook::Legacy, 4) => &LEGACY_4,
        (Codebook::LloydMax, 2) => &LLOYD_2,
        (Codebook::LloydMax, 3) => &LLOYD_3,
        (Codebook::LloydMax, 4) => &LLOYD_4,
        _ => return Err(Error::UnsupportedBits(bits)),
    })
}

/// The centroids at dimension `dim`, as the reference decoder computes them: each table value times
/// `1 / sqrt(dim)` in double precision, then rounded to single precision
/// (`(raw * (1.0 / math.sqrt(dim))).astype(np.float32)` in `turboquant_pro/pgvector.py`).
///
/// `docs/FORMAT_SPEC.md` writes `codebook(bits)[indices] / sqrt(dim)`, a division, and so does
/// `contrib/tqe1_reader.py`. The two can differ in the last bit of the double; this reader follows the reference
/// implementation, which wrote every stored record.
pub fn centroids(bits: u8, codebook: Codebook, dim: usize) -> Result<Vec<f32>> {
    let scale = 1.0 / (dim as f64).sqrt();
    Ok(table(bits, codebook)?.iter().map(|&v| (v * scale) as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_are_ascending_and_symmetric() {
        for cb in [Codebook::Legacy, Codebook::LloydMax] {
            for bits in [2u8, 3, 4] {
                let t = table(bits, cb).unwrap();
                assert_eq!(t.len(), 1 << bits);
                assert!(t.windows(2).all(|w| w[0] < w[1]));
                assert!((0..t.len()).all(|i| t[i] == -t[t.len() - 1 - i]));
            }
        }
        assert!(table(5, Codebook::Legacy).is_err());
        assert!(Codebook::from_code(2).is_err());
    }
}
