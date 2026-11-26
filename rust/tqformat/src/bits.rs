//! Bit-unpacking of quantizer level indices (`turboquant_pro/packed_codes.py`).
//!
//! The stream layout is LSB-first: value `j` occupies stream bits `[j*bits, (j+1)*bits)`, and stream bit `p`
//! lives in byte `p / 8` at bit position `p % 8`. A ragged tail is zero-padded to a whole group, the fewest values
//! whose bits fill whole bytes (4 values in 1 byte at 2 bits, 8 values in 3 bytes at 3 bits, 2 values in 1 byte at
//! 4 bits).

use crate::error::{Error, Result};

const fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Values per padding group: the fewest whose `bits` fill whole bytes.
pub fn stream_group(bits: u8) -> usize {
    8 / gcd(8, bits as usize)
}

/// Bytes the packer emits for `n` codes of `bits` each.
///
/// Not `ceil(n * bits / 8)` in general: at 3 bits the tail pads to a group of 8 values (3 bytes), so 10 codes take
/// 6 bytes, not 4.
pub fn packed_nbytes(n: usize, bits: u8) -> usize {
    let group = stream_group(bits);
    (n + group - 1) / group * (group * bits as usize / 8)
}

/// The first `n` codes of `bits` each from the LSB-first stream `packed`.
pub fn unpack_bits(packed: &[u8], n: usize, bits: u8) -> Result<Vec<u8>> {
    if !(1..=8).contains(&bits) {
        return Err(Error::UnsupportedBits(bits));
    }
    let needed = packed_nbytes(n, bits);
    if packed.len() < needed {
        return Err(Error::CodesTooShort { needed, got: packed.len() });
    }
    let mask = (1u16 << bits) - 1;
    let b = bits as usize;
    Ok((0..n)
        .map(|j| {
            let p = j * b;
            let byte = p / 8;
            let mut w = packed[byte] as u16;
            if byte + 1 < packed.len() {
                w |= (packed[byte + 1] as u16) << 8;
            }
            ((w >> (p % 8)) & mask) as u8
        })
        .collect())
}

/// Unpack the byte-aligned rows of an index v3 `codes` section: `n_rows` rows of `cols` bytes, each holding
/// `8 / slot_bits` codes LSB-first, of which the first `dim` per row are real.
pub fn unpack_rows(packed: &[u8], n_rows: usize, cols: usize, dim: usize, slot_bits: u8) -> Result<Vec<u8>> {
    if slot_bits >= 8 {
        if packed.len() != n_rows * dim {
            return Err(Error::Meta(format!("codes section has {} bytes, expected {}", packed.len(), n_rows * dim)));
        }
        return Ok(packed.to_vec());
    }
    if !matches!(slot_bits, 1 | 2 | 4) {
        return Err(Error::Meta(format!("slot_bits must be 1, 2 or 4 (or 8 or more), got {slot_bits}")));
    }
    let per = 8 / slot_bits as usize;
    if cols * per < dim || packed.len() != n_rows * cols {
        return Err(Error::Meta(format!(
            "codes section of {} bytes does not hold {n_rows} rows of {dim}",
            packed.len()
        )));
    }
    let mask = (1u8 << slot_bits) - 1;
    let mut out = Vec::with_capacity(n_rows * dim);
    for r in 0..n_rows {
        let row = &packed[r * cols..(r + 1) * cols];
        for k in 0..dim {
            out.push((row[k / per] >> ((k % per) * slot_bits as usize)) & mask);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack the way `packed_codes.pack_bits` does, for round trips.
    fn pack(values: &[u8], bits: u8) -> Vec<u8> {
        let mut out = vec![0u8; packed_nbytes(values.len(), bits)];
        for (j, &v) in values.iter().enumerate() {
            let p = j * bits as usize;
            let w = (v as u16) << (p % 8);
            out[p / 8] |= w as u8;
            if p / 8 + 1 < out.len() {
                out[p / 8 + 1] |= (w >> 8) as u8;
            }
        }
        out
    }

    #[test]
    fn sizes() {
        assert_eq!(packed_nbytes(10, 3), 6); // two 8-value groups of 3 bytes
        assert_eq!(packed_nbytes(10, 2), 3);
        assert_eq!(packed_nbytes(10, 4), 5);
        assert_eq!(packed_nbytes(32, 3), 12);
    }

    #[test]
    fn round_trip_every_width_and_length() {
        for bits in [2u8, 3, 4] {
            for n in 0..40usize {
                let v: Vec<u8> = (0..n).map(|i| ((i * 7 + 3) % (1 << bits)) as u8).collect();
                assert_eq!(unpack_bits(&pack(&v, bits), n, bits).unwrap(), v, "bits {bits} n {n}");
            }
        }
        assert!(matches!(unpack_bits(&[0u8; 2], 10, 3), Err(Error::CodesTooShort { needed: 6, got: 2 })));
    }

    #[test]
    fn known_bytes() {
        // packed_codes.pack_bits(np.arange(8) % 8, 3) == bytes([0x88, 0xC6, 0xFA])
        assert_eq!(pack(&[0, 1, 2, 3, 4, 5, 6, 7], 3), vec![0x88, 0xC6, 0xFA]);
        assert_eq!(unpack_bits(&[0x88, 0xC6, 0xFA], 8, 3).unwrap(), vec![0, 1, 2, 3, 4, 5, 6, 7]);
        // pack_bits([1, 2, 3, 0], 2) == bytes([0b00_11_10_01])
        assert_eq!(unpack_bits(&[0b0011_1001], 4, 2).unwrap(), vec![1, 2, 3, 0]);
    }

    #[test]
    fn rows() {
        // two rows of dim 3 at 4-bit slots: [1, 2, 3], [4, 5, 6] -> bytes [0x21, 0x03], [0x54, 0x06]
        let packed = [0x21, 0x03, 0x54, 0x06];
        assert_eq!(unpack_rows(&packed, 2, 2, 3, 4).unwrap(), vec![1, 2, 3, 4, 5, 6]);
    }
}
