//! TQIX: the section container of a persisted index (and, with magic `TQPW`, of packed weights).
//!
//! Layout, little-endian (`turboquant_pro/index_file.py`, `docs/FORMATS.md` section 2):
//!
//! ```text
//! offset size field
//! 0      4    magic       b"TQIX"
//! 4      2    version     u16
//! 6      2    n_sections  u16
//! 8      4    reserved    u32 (written as 0)
//! 12     ..   directory   n_sections entries of 56 bytes:
//!               32 name   UTF-8, NUL-padded
//!               8  offset u64, from the start of the file
//!               8  length u64
//!               4  crc32  u32 of the section bytes (zlib CRC-32)
//!               4  flags  u32 (written as 0)
//! ..     ..   section payloads
//! ```
//!
//! The container is generic: what a version number means is up to the format stored in it (see [`crate::index`]),
//! so [`Container::parse`] accepts any version and leaves the check to the layer that knows the format.

use crate::crc32::crc32;
use crate::error::{Error, Result};

pub const MAGIC_TQIX: [u8; 4] = *b"TQIX";
pub const MAGIC_TQPW: [u8; 4] = *b"TQPW";
pub const FIXED_HEADER: usize = 12;
pub const ENTRY_SIZE: usize = 56;
pub const NAME_LEN: usize = 32;

/// One directory entry: where a section lives, without its bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionRef {
    pub name: String,
    pub offset: u64,
    pub length: u64,
    pub crc32: u32,
    pub flags: u32,
}

/// The fixed header and the directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Directory {
    pub magic: [u8; 4],
    pub version: u16,
    pub reserved: u32,
    pub sections: Vec<SectionRef>,
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

/// Parse the fixed header and the directory from the start of `bytes`, without touching payloads.
pub fn parse_directory(bytes: &[u8], magic: [u8; 4]) -> Result<Directory> {
    if bytes.len() < FIXED_HEADER {
        return Err(Error::Truncated { what: "file too small for the 12-byte header".into() });
    }
    if bytes[0..4] != magic {
        return Err(Error::BadMagic { expected: magic, found: bytes[0..4].to_vec() });
    }
    let version = u16_at(bytes, 4);
    let n = u16_at(bytes, 6) as usize;
    let reserved = u32_at(bytes, 8);
    let dir_end = FIXED_HEADER + n * ENTRY_SIZE;
    if bytes.len() < dir_end {
        return Err(Error::Truncated { what: format!("directory of {n} entries") });
    }
    let mut sections = Vec::with_capacity(n);
    for i in 0..n {
        let e = FIXED_HEADER + i * ENTRY_SIZE;
        let raw = &bytes[e..e + NAME_LEN];
        let end = raw.iter().rposition(|&c| c != 0).map_or(0, |p| p + 1);
        let name = std::str::from_utf8(&raw[..end]).map_err(|_| Error::BadSectionName { index: i })?.to_string();
        if sections.iter().any(|s: &SectionRef| s.name == name) {
            return Err(Error::DuplicateSection(name));
        }
        sections.push(SectionRef {
            name,
            offset: u64_at(bytes, e + 32),
            length: u64_at(bytes, e + 40),
            crc32: u32_at(bytes, e + 48),
            flags: u32_at(bytes, e + 52),
        });
    }
    Ok(Directory { magic, version, reserved, sections })
}

/// A whole container in memory, every section's CRC32 verified.
#[derive(Debug, Clone)]
pub struct Container {
    pub directory: Directory,
    bytes: Vec<u8>,
}

impl Container {
    /// Parse a container and verify every section: in bounds and matching its CRC32.
    pub fn parse(bytes: Vec<u8>, magic: [u8; 4]) -> Result<Container> {
        let directory = parse_directory(&bytes, magic)?;
        let size = bytes.len() as u64;
        for s in &directory.sections {
            let end = s.offset.saturating_add(s.length);
            if end > size {
                return Err(Error::SectionOutOfBounds { section: s.name.clone(), end, size });
            }
            let computed = crc32(&bytes[s.offset as usize..end as usize]);
            if computed != s.crc32 {
                return Err(Error::CrcMismatch { section: s.name.clone(), stored: s.crc32, computed });
            }
        }
        Ok(Container { directory, bytes })
    }

    /// Read and parse a container file.
    pub fn open(path: impl AsRef<std::path::Path>, magic: [u8; 4]) -> Result<Container> {
        Container::parse(std::fs::read(path)?, magic)
    }

    pub fn version(&self) -> u16 {
        self.directory.version
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.directory.sections.iter().map(|s| s.name.as_str())
    }

    /// The bytes of section `name`, if present.
    pub fn section(&self, name: &str) -> Option<&[u8]> {
        self.directory
            .sections
            .iter()
            .find(|s| s.name == name)
            .map(|s| &self.bytes[s.offset as usize..(s.offset + s.length) as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a container the way `index_file.write_container` does.
    pub(crate) fn write(version: u16, sections: &[(&str, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC_TQIX);
        out.extend_from_slice(&version.to_le_bytes());
        out.extend_from_slice(&(sections.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        let mut offset = (FIXED_HEADER + sections.len() * ENTRY_SIZE) as u64;
        for (name, data) in sections {
            let mut n = [0u8; NAME_LEN];
            n[..name.len()].copy_from_slice(name.as_bytes());
            out.extend_from_slice(&n);
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(&(data.len() as u64).to_le_bytes());
            out.extend_from_slice(&crc32(data).to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            offset += data.len() as u64;
        }
        for (_, data) in sections {
            out.extend_from_slice(data);
        }
        out
    }

    #[test]
    fn round_trip_and_every_error() {
        let file = write(3, &[("meta", b"{}"), ("codes", &[1, 2, 3])]);
        let c = Container::parse(file.clone(), MAGIC_TQIX).unwrap();
        assert_eq!(c.version(), 3);
        assert_eq!(c.section("codes"), Some(&[1u8, 2, 3][..]));
        assert_eq!(c.names().collect::<Vec<_>>(), ["meta", "codes"]);

        assert!(matches!(Container::parse(file.clone(), MAGIC_TQPW), Err(Error::BadMagic { .. })));
        assert!(matches!(Container::parse(file[..8].to_vec(), MAGIC_TQIX), Err(Error::Truncated { .. })));
        assert!(matches!(Container::parse(file[..40].to_vec(), MAGIC_TQIX), Err(Error::Truncated { .. })));
        let short = file[..file.len() - 1].to_vec();
        assert!(matches!(Container::parse(short, MAGIC_TQIX), Err(Error::SectionOutOfBounds { .. })));
        // flip every payload bit in turn: each is a CRC mismatch, never silent
        let payload_start = FIXED_HEADER + 2 * ENTRY_SIZE;
        for byte in payload_start..file.len() {
            for bit in 0..8 {
                let mut bad = file.clone();
                bad[byte] ^= 1 << bit;
                assert!(matches!(Container::parse(bad, MAGIC_TQIX), Err(Error::CrcMismatch { .. })));
            }
        }
        let dup = write(3, &[("meta", b"{}"), ("meta", b"[]")]);
        assert_eq!(Container::parse(dup, MAGIC_TQIX).unwrap_err(), Error::DuplicateSection("meta".into()));
    }
}
