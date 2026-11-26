//! The inverse rotations of TQE1 decoding, regenerated from `(rotation, dim, seed)` as
//! `turboquant_pro.pgvector.TurboQuantPGVector` builds them.
//!
//! | rotation, dim | what the writer drew from `default_rng(seed)` | inverse | this reader |
//! |---|---|---|---|
//! | `hadamard` (power-of-two dim) | `choice([-1, 1], dim)` | `fwht(y) * (1/sqrt(dim)) * s` | bit-exact |
//! | `qr`, dim > 4096 | `choice([-1, 1], dim)`, then `permutation(dim)` | `y[inv_perm] / s` | bit-exact |
//! | `qr`, dim <= 4096 | `standard_normal((dim, dim))` as float32, then LAPACK QR | `y @ Q` | to a tolerance |
//!
//! The dense case cannot be bit-exact against NumPy in general: `np.linalg.qr` calls the single-precision LAPACK
//! routines `sgeqrf`/`sorgqr` of whatever BLAS NumPy is linked to, whose last bits depend on the build and the CPU,
//! and `y @ Q` is a BLAS product with its own summation order. This reader forms Q from the same Householder
//! reflectors, with LAPACK's sign convention, in double precision, and rounds once.

use crate::error::{Error, Result};
use crate::rng::{choice_signs, permutation, standard_normal_matrix, Pcg64};
use crate::tqe::Rotation;

/// The largest dimension the `qr` rotation materializes as a dense matrix.
pub const MAX_DENSE_QR_DIM: usize = 4096;

/// A regenerated rotation, ready to invert.
#[derive(Debug, Clone)]
pub enum RotationData {
    Hadamard {
        signs: Vec<f32>,
        scale: f32,
    },
    /// `qr` above [`MAX_DENSE_QR_DIM`].
    Structured {
        signs: Vec<f32>,
        inv_perm: Vec<usize>,
    },
    /// `qr` up to [`MAX_DENSE_QR_DIM`]: Q, row-major, `dim x dim`.
    Dense {
        q: Vec<f32>,
        dim: usize,
    },
}

/// Whether a decode through this rotation reproduces the reference decoder bit for bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exactness {
    BitExact,
    /// Agrees to rounding only; see the module documentation.
    Tolerance,
}

/// Whether decoding through `rotation` at `dim` is bit-exact, without building the rotation (a dense Q at 4096
/// dimensions takes about a minute and a half to regenerate on one core).
pub fn exactness_of(rotation: Rotation, dim: usize) -> Exactness {
    match rotation {
        Rotation::Qr if dim <= MAX_DENSE_QR_DIM => Exactness::Tolerance,
        _ => Exactness::BitExact,
    }
}

impl RotationData {
    /// Regenerate the rotation of a record.
    pub fn build(rotation: Rotation, dim: usize, seed: u32) -> Result<RotationData> {
        let mut g = Pcg64::default_rng(seed as u128);
        match rotation {
            Rotation::Hadamard => {
                if dim == 0 || !dim.is_power_of_two() {
                    return Err(Error::HadamardDimension(dim));
                }
                let signs = choice_signs(&mut g, dim).into_iter().map(|s| s as f32).collect();
                Ok(RotationData::Hadamard { signs, scale: (1.0 / (dim as f64).sqrt()) as f32 })
            }
            Rotation::Qr if dim > MAX_DENSE_QR_DIM => {
                let signs: Vec<f32> = choice_signs(&mut g, dim).into_iter().map(|s| s as f32).collect();
                let perm = permutation(&mut g, dim);
                let mut inv_perm = vec![0usize; dim];
                for (i, &p) in perm.iter().enumerate() {
                    inv_perm[p] = i;
                }
                Ok(RotationData::Structured { signs, inv_perm })
            }
            Rotation::Qr => {
                let gauss: Vec<f32> = standard_normal_matrix(&mut g, dim, dim).into_iter().map(|v| v as f32).collect();
                Ok(RotationData::Dense { q: householder_q(&gauss, dim), dim })
            }
        }
    }

    pub fn exactness(&self) -> Exactness {
        match self {
            RotationData::Dense { .. } => Exactness::Tolerance,
            _ => Exactness::BitExact,
        }
    }

    /// The inverse rotation of `y` (length `dim`).
    pub fn unrotate(&self, y: &[f32]) -> Vec<f32> {
        match self {
            RotationData::Hadamard { signs, scale } => {
                let mut v = y.to_vec();
                fwht(&mut v);
                // `_fwht(y) * self._hadamard_scale * self._sign_flip`, left to right, in float32
                v.iter().zip(signs).map(|(&x, &s)| x * scale * s).collect()
            }
            RotationData::Structured { signs, inv_perm } => {
                // `y[..., self._inv_perm] / self._sign_flip`
                (0..y.len()).map(|i| y[inv_perm[i]] / signs[i]).collect()
            }
            RotationData::Dense { q, dim } => {
                // `y @ self._Pi`
                let d = *dim;
                let mut acc = vec![0.0f64; d];
                for (i, &yi) in y.iter().enumerate() {
                    let row = &q[i * d..(i + 1) * d];
                    let yi = yi as f64;
                    for (a, &qij) in acc.iter_mut().zip(row) {
                        *a += yi * qij as f64;
                    }
                }
                acc.into_iter().map(|a| a as f32).collect()
            }
        }
    }
}

/// The unnormalized fast Walsh-Hadamard transform in place, in the butterfly order of
/// `turboquant_pro.pgvector._fwht`: for `h = 1, 2, 4, ...`, each block of `2h` becomes `(x + y, x - y)` of its two
/// halves. Single-precision arithmetic throughout, so the result matches NumPy's bit for bit.
pub fn fwht(v: &mut [f32]) {
    let d = v.len();
    let mut h = 1;
    while h < d {
        let mut block = 0;
        while block < d {
            for j in block..block + h {
                let (x, y) = (v[j], v[j + h]);
                v[j] = x + y;
                v[j + h] = x - y;
            }
            block += 2 * h;
        }
        h *= 2;
    }
}

/// Q of the QR decomposition of `g` (`dim x dim`, row-major, float32), as `np.linalg.qr` returns it: the product
/// of LAPACK-convention Householder reflectors `H_k = I - tau_k v_k v_k^T` with `beta_k = -sign(alpha_k) ||x_k||`.
/// Computed in double precision; returned row-major in float32.
pub fn householder_q(g: &[f32], dim: usize) -> Vec<f32> {
    let d = dim;
    // column-major working copy
    let mut a = vec![0.0f64; d * d];
    for r in 0..d {
        for c in 0..d {
            a[c * d + r] = g[r * d + c] as f64;
        }
    }
    let mut tau = vec![0.0f64; d];
    for k in 0..d {
        let alpha = a[k * d + k];
        let xnorm = a[k * d + k + 1..(k + 1) * d].iter().map(|x| x * x).sum::<f64>().sqrt();
        if xnorm == 0.0 {
            tau[k] = 0.0; // H_k = I (dlarfg)
            continue;
        }
        let beta = -alpha.hypot(xnorm).copysign(alpha);
        tau[k] = (beta - alpha) / beta;
        let scal = 1.0 / (alpha - beta);
        for x in &mut a[k * d + k + 1..(k + 1) * d] {
            *x *= scal;
        }
        a[k * d + k] = beta;
        // apply H_k to the remaining columns
        let (head, tail) = a.split_at_mut((k + 1) * d);
        let v = &head[k * d + k + 1..(k + 1) * d];
        for col in tail.chunks_exact_mut(d) {
            let mut w = col[k];
            for (x, &vr) in col[k + 1..].iter().zip(v) {
                w += vr * x;
            }
            w *= tau[k];
            col[k] -= w;
            for (x, &vr) in col[k + 1..].iter_mut().zip(v) {
                *x -= w * vr;
            }
        }
    }
    // Q = H_0 H_1 ... H_{d-1} applied to the identity, accumulated backwards (dorg2r)
    let mut q = vec![0.0f64; d * d];
    for i in 0..d {
        q[i * d + i] = 1.0;
    }
    for k in (0..d).rev() {
        if tau[k] == 0.0 {
            continue;
        }
        let v = &a[k * d + k + 1..(k + 1) * d];
        for col in q[k * d..].chunks_exact_mut(d) {
            let mut w = col[k];
            for (x, &vr) in col[k + 1..].iter().zip(v) {
                w += vr * x;
            }
            w *= tau[k];
            col[k] -= w;
            for (x, &vr) in col[k + 1..].iter_mut().zip(v) {
                *x -= w * vr;
            }
        }
    }
    // row-major float32
    let mut out = vec![0.0f32; d * d];
    for r in 0..d {
        for c in 0..d {
            out[r * d + c] = q[c * d + r] as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht_is_orthogonal_up_to_dim() {
        let x: Vec<f32> = (0..16).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut v = x.clone();
        fwht(&mut v);
        fwht(&mut v);
        for (a, b) in v.iter().zip(&x) {
            assert!((a / 16.0 - b).abs() < 1e-6);
        }
    }

    #[test]
    fn q_is_orthogonal_and_reproduces_g() {
        let d = 24;
        let g: Vec<f32> = (0..d * d).map(|i| ((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0).collect();
        let q = householder_q(&g, d);
        for i in 0..d {
            for j in 0..d {
                let dot: f64 = (0..d).map(|k| q[k * d + i] as f64 * q[k * d + j] as f64).sum();
                assert!((dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-5, "{i} {j} {dot}");
            }
        }
        // R = Q^T G is upper triangular
        for i in 0..d {
            for j in 0..i {
                let r: f64 = (0..d).map(|k| q[k * d + i] as f64 * g[k * d + j] as f64).sum();
                assert!(r.abs() < 1e-4, "R[{i}][{j}] = {r}");
            }
        }
    }

    #[test]
    fn hadamard_needs_a_power_of_two() {
        assert_eq!(RotationData::build(Rotation::Hadamard, 24, 1).unwrap_err(), Error::HadamardDimension(24));
    }
}
