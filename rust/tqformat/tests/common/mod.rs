//! Test helpers: a minimal reader for NumPy's `.npy` and uncompressed `.npz` files (what `np.savez` writes), and
//! the comparison the conformance tests use.

#![allow(dead_code)]

use std::collections::HashMap;

/// A float32 array from a `.npy` blob: `(shape, values)`. Only `'<f4'`, C order.
pub fn npy_f32(blob: &[u8]) -> (Vec<usize>, Vec<f32>) {
    assert_eq!(&blob[..6], b"\x93NUMPY", "not a .npy");
    let major = blob[6];
    let (hlen, start) = if major == 1 {
        (u16::from_le_bytes([blob[8], blob[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes(blob[8..12].try_into().unwrap()) as usize, 12)
    };
    let header = std::str::from_utf8(&blob[start..start + hlen]).unwrap();
    assert!(header.contains("'descr': '<f4'"), "{header}");
    assert!(header.contains("'fortran_order': False"), "{header}");
    let s = header.split("'shape': (").nth(1).unwrap().split(')').next().unwrap();
    let shape: Vec<usize> = s.split(',').map(str::trim).filter(|t| !t.is_empty()).map(|t| t.parse().unwrap()).collect();
    let data = &blob[start + hlen..];
    let vals = data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>();
    assert_eq!(vals.len(), shape.iter().product::<usize>());
    (shape, vals)
}

/// The members of an uncompressed (`ZIP_STORED`) zip archive: name -> bytes.
pub fn zip_stored(blob: &[u8]) -> HashMap<String, Vec<u8>> {
    let mut out = HashMap::new();
    let mut o = 0;
    while o + 30 <= blob.len() && blob[o..o + 4] == [0x50, 0x4b, 0x03, 0x04] {
        let method = u16::from_le_bytes([blob[o + 8], blob[o + 9]]);
        assert_eq!(method, 0, "only ZIP_STORED members are supported");
        let mut size = u32::from_le_bytes(blob[o + 18..o + 22].try_into().unwrap()) as u64;
        let nlen = u16::from_le_bytes([blob[o + 26], blob[o + 27]]) as usize;
        let elen = u16::from_le_bytes([blob[o + 28], blob[o + 29]]) as usize;
        let name = String::from_utf8(blob[o + 30..o + 30 + nlen].to_vec()).unwrap();
        if size == 0xFFFF_FFFF {
            // np.savez writes members with force_zip64=True: the sizes live in the ZIP64 extra field (id 0x0001),
            // uncompressed size then compressed size, as u64
            let extra = &blob[o + 30 + nlen..o + 30 + nlen + elen];
            let mut e = 0;
            while e + 4 <= extra.len() {
                let id = u16::from_le_bytes([extra[e], extra[e + 1]]);
                let len = u16::from_le_bytes([extra[e + 2], extra[e + 3]]) as usize;
                if id == 0x0001 {
                    size = u64::from_le_bytes(extra[e + 12..e + 20].try_into().unwrap());
                }
                e += 4 + len;
            }
        }
        let size = size as usize;
        let data_start = o + 30 + nlen + elen;
        out.insert(name, blob[data_start..data_start + size].to_vec());
        o = data_start + size;
    }
    out
}

/// How far `got` is from `want`: `(max |got - want|, number of values not bit-identical)`, and whether every value
/// satisfies `|got - want| <= atol + rtol * |want|` (NumPy's `assert_allclose`).
pub fn compare(got: &[f32], want: &[f32], rtol: f64, atol: f64) -> (f64, usize, bool) {
    assert_eq!(got.len(), want.len());
    let mut max = 0.0f64;
    let mut differ = 0;
    let mut ok = true;
    for (&g, &w) in got.iter().zip(want) {
        if g.to_bits() != w.to_bits() {
            differ += 1;
        }
        let d = (g as f64 - w as f64).abs();
        max = max.max(d);
        if d > atol + rtol * (w as f64).abs() {
            ok = false;
        }
    }
    (max, differ, ok)
}
