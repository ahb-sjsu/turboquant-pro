//! Errors. Every malformed input is a typed error; nothing is silently accepted or repaired.

use std::fmt;

/// Everything that can go wrong reading a TQE1 record or a TQIX container.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The first four bytes are not the expected magic.
    BadMagic { expected: [u8; 4], found: Vec<u8> },
    /// A format version this reader does not implement. Readers must reject these.
    UnsupportedVersion { format: &'static str, version: u32 },
    /// The input ends before a field or payload it declares.
    Truncated { what: String },
    /// A TQIX section's bytes do not match its stored CRC32.
    CrcMismatch { section: String, stored: u32, computed: u32 },
    /// A TQIX section's declared range runs past the end of the file.
    SectionOutOfBounds { section: String, end: u64, size: u64 },
    /// Two TQIX directory entries share a name (the writer forbids this).
    DuplicateSection(String),
    /// A TQIX section name is not valid UTF-8.
    BadSectionName { index: usize },
    /// A required TQIX section is absent.
    MissingSection(String),
    /// A TQE1 `bits` value other than 2, 3 or 4.
    UnsupportedBits(u8),
    /// A TQE1 rotation code other than 0 (qr) or 1 (hadamard).
    UnknownRotation(u8),
    /// A TQE1 codebook code other than 0 (legacy) or 1 (lloyd-max).
    UnknownCodebook(u8),
    /// The code block is shorter than `dim` codes of `bits` each need.
    CodesTooShort { needed: usize, got: usize },
    /// The Hadamard rotation needs a power-of-two dimension.
    HadamardDimension(usize),
    /// The index `meta` section is not valid JSON.
    Json(String),
    /// The index `meta` section is valid JSON but does not describe the file.
    Meta(String),
    /// An I/O failure reading a file.
    Io(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadMagic { expected, found } => write!(
                f,
                "bad magic {:?}; expected {:?}",
                String::from_utf8_lossy(found),
                String::from_utf8_lossy(expected)
            ),
            Error::UnsupportedVersion { format, version } => {
                write!(f, "unsupported {format} version {version}")
            }
            Error::Truncated { what } => write!(f, "truncated: {what}"),
            Error::CrcMismatch { section, stored, computed } => {
                write!(f, "CRC mismatch in section {section:?} (stored {stored:#010x}, computed {computed:#010x})")
            }
            Error::SectionOutOfBounds { section, end, size } => {
                write!(f, "section {section:?} runs past the end of the file ({end} > {size})")
            }
            Error::DuplicateSection(name) => write!(f, "duplicate section name {name:?}"),
            Error::BadSectionName { index } => write!(f, "section {index} has a name that is not UTF-8"),
            Error::MissingSection(name) => write!(f, "missing section {name:?}"),
            Error::UnsupportedBits(b) => write!(f, "unsupported bits {b}; expected 2, 3 or 4"),
            Error::UnknownRotation(r) => write!(f, "unknown rotation code {r}"),
            Error::UnknownCodebook(c) => write!(f, "unknown codebook code {c}"),
            Error::CodesTooShort { needed, got } => {
                write!(f, "code block has {got} bytes; {needed} are needed")
            }
            Error::HadamardDimension(d) => {
                write!(f, "the hadamard rotation needs a power-of-two dim; got {d}")
            }
            Error::Json(m) => write!(f, "meta is not valid JSON: {m}"),
            Error::Meta(m) => write!(f, "meta: {m}"),
            Error::Io(m) => write!(f, "I/O: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
