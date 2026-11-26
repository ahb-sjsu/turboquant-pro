//! `tqread`: inspect and decode turboquant-pro files without Python.
//!
//! ```text
//! tqread info FILE            header, records or sections, and the CRC check
//! tqread decode FILE [OUT]    decode every TQE1 record to little-endian float32 rows (OUT, or stdout)
//! ```
//!
//! Exit status 0 on success, 1 on a malformed file (the error is printed), 2 on bad usage.

use std::io::Write;
use std::process::ExitCode;

use tqformat::decode::Decoder;
use tqformat::index::Index;
use tqformat::tqix::{Container, MAGIC_TQIX, MAGIC_TQPW};
use tqformat::{tqe, Error};

fn info(path: &str) -> Result<(), Error> {
    let bytes = std::fs::read(path)?;
    let magic = bytes.get(0..4).unwrap_or(&[]);
    if magic == tqe::MAGIC {
        let recs = tqe::parse_batch(&bytes)?;
        println!("TQE1 batch: {} record(s)", recs.len());
        let dec = Decoder::new();
        for (i, r) in recs.iter().enumerate().take(20) {
            println!(
                "  {i}: v{} bits={} dim={} seed={} norm={} rotation={} codebook={} codes={}B decode={:?}",
                r.version,
                r.bits,
                r.dim,
                r.seed,
                r.norm,
                r.rotation.name(),
                r.codebook.name(),
                r.codes.len(),
                dec.exactness(r)?
            );
        }
        if recs.len() > 20 {
            println!("  ... {} more", recs.len() - 20);
        }
    } else if magic == MAGIC_TQIX {
        let idx = Index::open(path)?;
        println!("TQIX index, format version {}, all section CRCs verified", idx.format_version());
        for s in &idx.container.directory.sections {
            println!("  {:<24} offset {:>10} length {:>10} crc32 {:#010x}", s.name, s.offset, s.length, s.crc32);
        }
        let (_, rows, dim) = idx.codes()?;
        println!("  {rows} rows of {dim} codes; arrays: {}", idx.array_names().join(", "));
    } else if magic == MAGIC_TQPW {
        let c = Container::open(path, MAGIC_TQPW)?;
        println!("TQPW container, version {}, all section CRCs verified", c.version());
        for s in &c.directory.sections {
            println!("  {:<24} offset {:>10} length {:>10} crc32 {:#010x}", s.name, s.offset, s.length, s.crc32);
        }
    } else {
        return Err(Error::BadMagic { expected: tqe::MAGIC, found: magic.to_vec() });
    }
    Ok(())
}

fn decode(path: &str, out: Option<&str>) -> Result<(), Error> {
    let bytes = std::fs::read(path)?;
    let mut dec = Decoder::new();
    let mut buf = Vec::new();
    for r in tqe::parse_batch(&bytes)? {
        for v in dec.decode(&r)? {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    match out {
        Some(p) => std::fs::write(p, &buf)?,
        None => std::io::stdout().write_all(&buf)?,
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let res = match args.get(1).map(String::as_str) {
        Some("info") if args.len() == 3 => info(&args[2]),
        Some("decode") if args.len() == 3 || args.len() == 4 => decode(&args[2], args.get(3).map(String::as_str)),
        _ => {
            eprintln!("usage: tqread info FILE | tqread decode FILE [OUT]");
            return ExitCode::from(2);
        }
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tqread: {e}");
            ExitCode::from(1)
        }
    }
}
