//! The parts of NumPy's random module that the TQE1 rotations draw from, ported exactly.
//!
//! The writer builds its rotation with `np.random.default_rng(seed)`, which is `Generator(PCG64(SeedSequence(seed)))`,
//! then draws, depending on the rotation, `standard_normal((dim, dim))`, `choice([-1.0, 1.0], size=dim)` and
//! `permutation(dim)`. Decoding a record needs those draws again, bit for bit. Each function here names the NumPy
//! routine it reproduces (NumPy v2.4.6 sources: `bit_generator.pyx`, `src/pcg64/pcg64.h`, `_pcg64.pyx`,
//! `src/distributions/distributions.c`, `_generator.pyx`, `_bounded_integers.pyx.in`). See NOTICE for NumPy's
//! licence.
//!
//! The format's spec does not name these algorithms; they are what the reference implementation does, and NumPy
//! does not promise that `Generator` streams stay the same across its releases. That is the main finding recorded
//! in `FINDINGS.md`.

use crate::ziggurat::{FI_DOUBLE, KI_DOUBLE, WI_DOUBLE, ZIGGURAT_NOR_INV_R, ZIGGURAT_NOR_R};

// ------------------------------------------------------------------ SeedSequence (bit_generator.pyx)

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^ (v >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^ (r >> XSHIFT)
}

/// `numpy.random.SeedSequence(entropy)` for a non-negative integer entropy and no spawn key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeedSequence {
    pub pool: [u32; POOL_SIZE],
}

impl SeedSequence {
    pub fn new(entropy: u128) -> SeedSequence {
        // _int_to_uint32_array: 32-bit words, lowest first; zero is one word
        let mut words = Vec::new();
        let mut n = entropy;
        if n == 0 {
            words.push(0);
        }
        while n > 0 {
            words.push(n as u32);
            n >>= 32;
        }
        // mix_entropy
        let mut pool = [0u32; POOL_SIZE];
        let mut hc = INIT_A;
        for (i, slot) in pool.iter_mut().enumerate() {
            *slot = hashmix(words.get(i).copied().unwrap_or(0), &mut hc);
        }
        for src in 0..POOL_SIZE {
            for dst in 0..POOL_SIZE {
                if src != dst {
                    pool[dst] = mix(pool[dst], hashmix(pool[src], &mut hc));
                }
            }
        }
        for &w in words.iter().skip(POOL_SIZE) {
            for slot in pool.iter_mut() {
                *slot = mix(*slot, hashmix(w, &mut hc));
            }
        }
        SeedSequence { pool }
    }

    /// `generate_state(n_words, np.uint32)`.
    pub fn generate_state_u32(&self, n_words: usize) -> Vec<u32> {
        let mut hc = INIT_B;
        (0..n_words)
            .map(|i| {
                let mut d = self.pool[i % POOL_SIZE] ^ hc;
                hc = hc.wrapping_mul(MULT_B);
                d = d.wrapping_mul(hc);
                d ^ (d >> XSHIFT)
            })
            .collect()
    }

    /// `generate_state(n_words, np.uint64)`: twice as many 32-bit words, paired little-endian.
    pub fn generate_state_u64(&self, n_words: usize) -> Vec<u64> {
        let w = self.generate_state_u32(2 * n_words);
        (0..n_words).map(|k| w[2 * k] as u64 | ((w[2 * k + 1] as u64) << 32)).collect()
    }
}

// ------------------------------------------------------------------ PCG64 (pcg64.h, _pcg64.pyx)

const PCG_MULT: u128 = (2549297995355413924u128 << 64) | 4865540595714422341u128;

/// NumPy's `PCG64` bit generator: the 128-bit LCG with the XSL-RR output, and the 32-bit buffer that hands out
/// the low half of a 64-bit draw first and keeps the high half for the next 32-bit request.
#[derive(Debug, Clone)]
pub struct Pcg64 {
    state: u128,
    inc: u128,
    has_uint32: bool,
    uinteger: u32,
}

impl Pcg64 {
    /// `PCG64(seed_seq)`: seed from `generate_state(4, np.uint64)`, words 0-1 the state, words 2-3 the stream.
    pub fn from_seed_sequence(ss: &SeedSequence) -> Pcg64 {
        let v = ss.generate_state_u64(4);
        let initstate = ((v[0] as u128) << 64) | v[1] as u128;
        let initseq = ((v[2] as u128) << 64) | v[3] as u128;
        // pcg_setseq_128_srandom_r
        let mut g = Pcg64 { state: 0, inc: (initseq << 1) | 1, has_uint32: false, uinteger: 0 };
        g.step();
        g.state = g.state.wrapping_add(initstate);
        g.step();
        g
    }

    /// `np.random.default_rng(seed)`'s bit generator.
    pub fn default_rng(seed: u128) -> Pcg64 {
        Pcg64::from_seed_sequence(&SeedSequence::new(seed))
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    /// `pcg64_next64`: step, then XSL-RR output of the new state.
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let s = self.state;
        ((s >> 64) as u64 ^ s as u64).rotate_right((s >> 122) as u32)
    }

    /// `pcg64_next32`.
    pub fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            return self.uinteger;
        }
        let next = self.next_u64();
        self.has_uint32 = true;
        self.uinteger = (next >> 32) as u32;
        next as u32
    }

    /// `pcg64_double`: the top 53 bits as a double in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
}

// ------------------------------------------------------------------ distributions (distributions.c)

/// `random_standard_normal`: NumPy's ziggurat for the standard normal, one double.
///
/// The tail and wedge branches call `log1p` and `exp`; they are the only steps that rely on the C library, and
/// they decide whether a candidate is accepted in under 1% of draws.
pub fn standard_normal(rng: &mut Pcg64) -> f64 {
    loop {
        let mut r = rng.next_u64();
        let idx = (r & 0xff) as usize;
        r >>= 8;
        let sign = r & 0x1;
        let rabs = (r >> 1) & 0x000f_ffff_ffff_ffff;
        let mut x = rabs as f64 * WI_DOUBLE[idx];
        if sign & 0x1 != 0 {
            x = -x;
        }
        if rabs < KI_DOUBLE[idx] {
            return x;
        }
        if idx == 0 {
            loop {
                let xx = -ZIGGURAT_NOR_INV_R * (-rng.next_f64()).ln_1p();
                let yy = -(-rng.next_f64()).ln_1p();
                if yy + yy > xx * xx {
                    return if (rabs >> 8) & 0x1 != 0 { -(ZIGGURAT_NOR_R + xx) } else { ZIGGURAT_NOR_R + xx };
                }
            }
        } else if (FI_DOUBLE[idx - 1] - FI_DOUBLE[idx]) * rng.next_f64() + FI_DOUBLE[idx] < (-0.5 * x * x).exp() {
            return x;
        }
    }
}

/// `buffered_bounded_lemire_uint32`: a uniform integer in `[0, rng]` (inclusive), `rng < u32::MAX`.
fn bounded_lemire_u32(g: &mut Pcg64, rng: u32) -> u32 {
    let rng_excl = rng + 1;
    let mut m = g.next_u32() as u64 * rng_excl as u64;
    let mut leftover = m as u32;
    if leftover < rng_excl {
        let threshold = (u32::MAX - rng) % rng_excl;
        while leftover < threshold {
            m = g.next_u32() as u64 * rng_excl as u64;
            leftover = m as u32;
        }
    }
    (m >> 32) as u32
}

/// `random_interval`: a uniform integer in `[0, max]` by masked rejection, the draw `Generator.shuffle` uses for a
/// one-dimensional array.
fn random_interval(g: &mut Pcg64, max: u64) -> u64 {
    if max == 0 {
        return 0;
    }
    let mut mask = max;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;
    if max <= 0xffff_ffff {
        loop {
            let v = g.next_u32() as u64 & mask;
            if v <= max {
                return v;
            }
        }
    } else {
        loop {
            let v = g.next_u64() & mask;
            if v <= max {
                return v;
            }
        }
    }
}

// ------------------------------------------------------------------ Generator methods (_generator.pyx)

/// `Generator.standard_normal((rows, cols))`, row-major.
pub fn standard_normal_matrix(g: &mut Pcg64, rows: usize, cols: usize) -> Vec<f64> {
    (0..rows * cols).map(|_| standard_normal(g)).collect()
}

/// `Generator.choice([-1.0, 1.0], size=n)`: `integers(0, 2, size=n, dtype=np.int64)` picks the index, which is the
/// Lemire draw over `[0, 1]`, one 32-bit draw per value (the rejection step cannot fire for a range of 2).
pub fn choice_signs(g: &mut Pcg64, n: usize) -> Vec<f64> {
    (0..n).map(|_| if bounded_lemire_u32(g, 1) == 0 { -1.0 } else { 1.0 }).collect()
}

/// `Generator.permutation(n)`: `shuffle(np.arange(n))`, Fisher-Yates from the last element down, each index drawn
/// by [`random_interval`].
pub fn permutation(g: &mut Pcg64, n: usize) -> Vec<usize> {
    let mut a: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = random_interval(g, i as u64) as usize;
        a.swap(i, j);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    // Known answers, from NumPy 2.4.6 on Atlas (see tools/make_kat.py):
    //   SeedSequence(0).pool, SeedSequence(42).generate_state(4, np.uint64), PCG64(42).random_raw(3)

    #[test]
    fn seed_sequence_and_pcg64() {
        let ss = SeedSequence::new(0);
        assert_eq!(ss.pool, crate::kat::SEEDSEQ_0_POOL);
        assert_eq!(SeedSequence::new(42).generate_state_u64(4), crate::kat::SEEDSEQ_42_STATE_U64);
        let mut g = Pcg64::default_rng(42);
        let raw: Vec<u64> = (0..3).map(|_| g.next_u64()).collect();
        assert_eq!(raw, crate::kat::PCG64_42_RAW);
    }

    #[test]
    fn generator_methods() {
        let mut g = Pcg64::default_rng(42);
        let z: Vec<u64> = (0..6).map(|_| standard_normal(&mut g).to_bits()).collect();
        assert_eq!(z, crate::kat::NORMAL_42_BITS);
        let mut g = Pcg64::default_rng(7);
        assert_eq!(choice_signs(&mut g, 9), crate::kat::CHOICE_7_9);
        let mut g = Pcg64::default_rng(7);
        assert_eq!(permutation(&mut g, 10), crate::kat::PERM_7_10);
        // the uint32 buffer carries over from choice into permutation (odd count leaves a half buffered)
        let mut g = Pcg64::default_rng(11);
        let s = choice_signs(&mut g, 5);
        let p = permutation(&mut g, 12);
        assert_eq!((s, p), (crate::kat::CHOICE_11_5.to_vec(), crate::kat::PERM_11_12_AFTER.to_vec()));
    }

    #[test]
    fn many_normals_match_including_the_tail() {
        // 2,000,000 draws reach the idx == 0 tail and the wedge branches many times
        let mut g = Pcg64::default_rng(2026);
        let mut xor = 0u64;
        let mut sum = 0.0f64;
        for _ in 0..2_000_000 {
            let v = standard_normal(&mut g);
            xor ^= v.to_bits();
            sum += v;
        }
        assert_eq!((xor, sum.to_bits()), crate::kat::NORMAL_2026_2M_XOR_SUM);
    }
}
