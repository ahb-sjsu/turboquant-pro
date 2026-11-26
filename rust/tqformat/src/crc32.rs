//! CRC-32 as computed by zlib's `crc32` (reflected polynomial 0xEDB88320, initial value and final XOR 0xFFFFFFFF),
//! which is what `turboquant_pro.index_file` stores for every TQIX section.

const fn make_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

static TABLE: [u32; 256] = make_table();

/// The CRC-32 of `data`, equal to Python's `zlib.crc32(data) & 0xFFFFFFFF`.
pub fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c = TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::crc32;

    #[test]
    fn known_values() {
        // zlib.crc32(b"") == 0, zlib.crc32(b"123456789") == 0xCBF43926 (the CRC-32 check value)
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        // zlib.crc32(b"The quick brown fox jumps over the lazy dog") == 0x414FA339
        assert_eq!(crc32(b"The quick brown fox jumps over the lazy dog"), 0x414F_A339);
    }
}
