//! A standalone, read-only reader for turboquant-pro's two persisted formats, with no dependencies.
//!
//! - **TQE1** ([`tqe`], [`decode`]): one compressed embedding per record, versions 1 to 3. Records are parsed,
//!   their codes unpacked, and vectors reconstructed as the Python reference decoder reconstructs them: bit for bit
//!   for the `hadamard` rotation and for `qr` above 4096 dimensions, to a stated tolerance for `qr` up to 4096
//!   dimensions (see [`rotation`] for why).
//! - **TQIX** ([`tqix`], [`index`]): the section container of a persisted index, every section CRC32-checked, and
//!   the index layer on top of it (`meta`, typed arrays, unpacked codes, ids).
//!
//! Unknown versions are rejected and malformed input is an [`Error`], never a guess. Writing is out of scope.
//!
//! ```no_run
//! let bytes = std::fs::read("vectors.tqe")?;
//! let mut decoder = tqformat::decode::Decoder::new();
//! for rec in tqformat::tqe::parse_batch(&bytes)? {
//!     let v: Vec<f32> = decoder.decode(&rec)?;
//!     println!("{} dims, norm {}", v.len(), rec.norm);
//! }
//! let index = tqformat::index::Index::open("index.tqix")?;
//! let (codes, rows, dim) = index.codes()?;
//! # Ok::<(), tqformat::Error>(())
//! ```

pub mod bits;
pub mod codebook;
pub mod crc32;
pub mod decode;
pub mod error;
pub mod index;
pub mod json;
pub mod rng;
pub mod rotation;
pub mod tqe;
pub mod tqix;
// Generated files (tools/make_ziggurat.py, tools/make_kat.py), left as their generators write them so that a
// regeneration can be compared byte for byte.
#[rustfmt::skip]
mod ziggurat;

#[cfg(test)]
#[rustfmt::skip]
mod kat;

pub use error::{Error, Result};
