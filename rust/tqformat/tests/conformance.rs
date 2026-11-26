//! Cross-language conformance: Python writes a corpus (`conformance/make_corpus.py`), this test reads it.
//!
//! Set `TQFORMAT_CORPUS` to the corpus directory; without it the test only says it was skipped.
//!
//! Records decode exactly where the arithmetic is deterministic (`hadamard`, structured `qr`) and to a tolerance
//! for dense `qr` (Q comes from LAPACK). The tolerance is `4 * f32::EPSILON * norm` on every coordinate: measured
//! against NumPy 2.5.5 with OpenBLAS 0.3.34 the worst case was 0.3 at 100 to 1024 dimensions and 0.43 at 4096, so
//! the bound leaves room for another BLAS without admitting a wrong rotation (a wrong seed or draw order gives
//! errors of the order of the norm itself). Indexes must match section by section and array by array, byte for
//! byte (compared by CRC32).

/// Allowed dense-`qr` error per coordinate, in units of `f32::EPSILON * norm`.
const QR_TOLERANCE: f64 = 4.0;

mod common;

use std::path::{Path, PathBuf};

use tqformat::decode::Decoder;
use tqformat::index::Index;
use tqformat::json::Json;
use tqformat::rotation::Exactness;
use tqformat::{crc32::crc32, tqe, Error};

fn corpus() -> Option<PathBuf> {
    match std::env::var_os("TQFORMAT_CORPUS") {
        Some(p) => Some(PathBuf::from(p)),
        None => {
            eprintln!("TQFORMAT_CORPUS is not set: cross-language conformance skipped");
            None
        }
    }
}

fn manifest(dir: &Path) -> Json {
    Json::parse(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap()
}

fn u(j: &Json, k: &str) -> u64 {
    j.get(k).unwrap_or_else(|| panic!("no {k}")).as_u64().unwrap()
}

#[test]
fn records_decode_like_python() {
    let Some(dir) = corpus() else { return };
    let m = manifest(&dir);
    eprintln!("corpus from NumPy {:?}, turboquant-pro {:?}", m.get("numpy").unwrap(), m.get("turboquant_pro").unwrap());
    for (name, c) in m.get("records").unwrap().as_object().unwrap() {
        let bytes = std::fs::read(dir.join(format!("{name}.tqe"))).unwrap();
        let recs = tqe::parse_batch(&bytes).unwrap();
        let (dim, n) = (u(c, "dim") as usize, u(c, "n") as usize);
        assert_eq!(recs.len(), n, "{name}");
        assert_eq!(recs[0].codes.len() as u64, u(c, "codelen"), "{name}");
        assert_eq!(recs[0].codes.len(), tqformat::bits::packed_nbytes(dim, recs[0].bits), "{name}");
        let want: Vec<f32> = std::fs::read(dir.join(format!("{name}.f32")))
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let mut dec = Decoder::new();
        let exact = dec.exactness(&recs[0]).unwrap();
        assert_eq!(exact, tqformat::rotation::exactness_of(recs[0].rotation, dim), "{name}");
        let mut worst_scaled = 0.0f64;
        let mut differ = 0;
        for (i, r) in recs.iter().enumerate() {
            let got = dec.decode(r).unwrap();
            let w = &want[i * dim..(i + 1) * dim];
            let (max, d, _) = common::compare(&got, w, 0.0, 0.0);
            differ += d;
            // max coordinate error over the vector's norm, in units of float32 epsilon
            let scale = (r.norm as f64).max(f64::MIN_POSITIVE) * f32::EPSILON as f64;
            worst_scaled = worst_scaled.max(max / scale);
        }
        eprintln!(
            "{name}: {exact:?}, {differ} of {} values not bit-identical, worst error {worst_scaled:.1} x eps x norm",
            n * dim
        );
        match exact {
            Exactness::BitExact => assert_eq!(differ, 0, "{name}: expected a bit-exact decode"),
            Exactness::Tolerance => assert!(worst_scaled <= QR_TOLERANCE, "{name}: error {worst_scaled} x eps x norm"),
        }
    }
}

#[test]
fn indexes_read_like_python() {
    let Some(dir) = corpus() else { return };
    let m = manifest(&dir);
    for (name, want) in m.get("indexes").unwrap().as_object().unwrap() {
        let path = dir.join(format!("{name}.tqix"));
        let idx = Index::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(idx.format_version() as u64, u(want, "version"), "{name}");
        let sections: Vec<(String, u64, u64)> = want
            .get("sections")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|s| {
                let s = s.as_array().unwrap();
                (s[0].as_str().unwrap().to_string(), s[1].as_u64().unwrap(), s[2].as_u64().unwrap())
            })
            .collect();
        let got: Vec<(String, u64, u64)> =
            idx.container.directory.sections.iter().map(|s| (s.name.clone(), s.length, s.crc32 as u64)).collect();
        assert_eq!(got, sections, "{name}");
        assert_eq!(crc32(idx.container.section("meta").unwrap()) as u64, u(want, "meta_crc32"), "{name}");
        for (arr, spec) in want.get("arrays").unwrap().as_object().unwrap() {
            let a = idx.array(arr).unwrap_or_else(|e| panic!("{name}/{arr}: {e}"));
            let shape: Vec<usize> =
                spec.get("shape").unwrap().as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect();
            assert_eq!(a.shape, shape, "{name}/{arr}");
            assert_eq!(crc32(a.bytes) as u64, u(spec, "crc32"), "{name}/{arr}");
        }
        let (codes, rows, dim) = idx.codes().unwrap();
        let wc = want.get("codes").unwrap();
        assert_eq!(
            (rows as u64, dim as u64, crc32(&codes) as u64),
            (u(wc, "rows"), u(wc, "dim"), u(wc, "crc32")),
            "{name}"
        );
        let ids = idx.ids().unwrap();
        assert_eq!(ids.len() as u64, u(want, "n_ids"), "{name}");
        let id_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(crc32(&id_bytes) as u64, u(want, "ids_crc32"), "{name}");
        assert_eq!(crc32(&idx.tombstones().unwrap()) as u64, u(want, "tombstones_crc32"), "{name}");
        eprintln!("{name}: v{}, {} sections, {rows} rows of {dim} codes: match", idx.format_version(), got.len());

        // corruption anywhere in a payload is an error, never silent
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        assert!(matches!(Index::parse(bytes), Err(Error::CrcMismatch { .. })), "{name}");
    }
}
