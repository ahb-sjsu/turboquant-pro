//! The committed TQE1 golden corpus (`tests/golden/tqe1/` of the repository): parse every `.tqe` file, check its
//! headers against `manifest.json`, decode it, and compare with `expected.npz`, the in-tree decoder's output.
//!
//! The comparison is exact where the arithmetic is (the `hadamard` rotation) and to a tolerance where it is not
//! (`qr`, whose Q comes from LAPACK; see `src/rotation.rs`). The tolerance is the one `tests/test_tqe1_golden.py`
//! holds the in-tree decoder to: `rtol = 1e-5, atol = 1e-6`.

mod common;

use std::path::PathBuf;

use tqformat::decode::Decoder;
use tqformat::json::Json;
use tqformat::rotation::Exactness;
use tqformat::tqe;

fn golden() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/tqe1")
}

#[test]
fn golden_corpus_decodes() {
    let dir = golden();
    let manifest = Json::parse(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    let n = manifest.get("n").unwrap().as_u64().unwrap() as usize;
    let dim = manifest.get("dim").unwrap().as_u64().unwrap() as usize;
    let expected = common::zip_stored(&std::fs::read(dir.join("expected.npz")).unwrap());
    let files = manifest.get("files").unwrap().as_object().unwrap();
    assert!(!files.is_empty());
    for (name, meta) in files {
        let bytes = std::fs::read(dir.join(format!("{name}.tqe"))).unwrap();
        assert_eq!(bytes.len() as u64, meta.get("bytes").unwrap().as_u64().unwrap(), "{name}");
        let recs = tqe::parse_batch(&bytes).unwrap();
        assert_eq!(recs.len(), n, "{name}");
        let mut dec = Decoder::new();
        let mut got = Vec::new();
        for r in &recs {
            assert_eq!(r.version as u64, meta.get("version").unwrap().as_u64().unwrap(), "{name}");
            assert_eq!(r.bits as u64, meta.get("bits").unwrap().as_u64().unwrap(), "{name}");
            assert_eq!(r.dim as usize, dim, "{name}");
            assert_eq!(r.seed as u64, meta.get("seed").unwrap().as_u64().unwrap(), "{name}");
            assert_eq!(r.rotation.name(), meta.get("rotation").unwrap().as_str().unwrap(), "{name}");
            assert_eq!(r.codebook.name(), meta.get("codebook").and_then(Json::as_str).unwrap_or("legacy"), "{name}");
            got.extend(dec.decode(r).unwrap());
        }
        let (shape, want) = common::npy_f32(&expected[&format!("{name}.npy")]);
        assert_eq!(shape, vec![n, dim], "{name}");
        let (max, differ, ok) = common::compare(&got, &want, 1e-5, 1e-6);
        let exact = dec.exactness(&recs[0]).unwrap();
        // the same measure as tests/conformance.rs: worst coordinate error over f32 epsilon times the norm
        let scaled = recs
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let (m, _, _) = common::compare(&got[i * dim..(i + 1) * dim], &want[i * dim..(i + 1) * dim], 0.0, 0.0);
                m / (r.norm as f64 * f32::EPSILON as f64)
            })
            .fold(0.0f64, f64::max);
        eprintln!(
            "{name}: {exact:?}, max |diff| {max:.3e} ({scaled:.2} x eps x norm), {differ} of {} values not bit-identical",
            want.len()
        );
        match exact {
            Exactness::BitExact => assert_eq!(differ, 0, "{name}: expected a bit-exact decode"),
            Exactness::Tolerance => assert!(ok, "{name}: outside rtol 1e-5, atol 1e-6 (max {max:e})"),
        }
    }
}

#[test]
fn malformed_records_are_errors() {
    let bytes = std::fs::read(golden().join("qr_b3.tqe")).unwrap();
    assert!(tqe::parse_batch(&bytes[..bytes.len() - 1]).is_err());
    let mut bad = bytes.clone();
    bad[4] = 9;
    assert!(matches!(tqe::parse(&bad), Err(tqformat::Error::UnsupportedVersion { .. })));
    bad[4] = 1;
    bad[0] = b'X';
    assert!(matches!(tqe::parse(&bad), Err(tqformat::Error::BadMagic { .. })));
}
