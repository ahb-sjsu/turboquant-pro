//! GPU-accelerated compression via CUDA (cudarc).
//!
//! Moves the expensive rotation (matrix multiply) and quantization
//! to the GPU. On a GV100 (14.9 TFLOPS FP32), a 1024x1024 matmul
//! takes ~0.1ms vs ~8ms on CPU.

use crate::types::TqVector;

#[cfg(feature = "gpu")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
#[cfg(feature = "gpu")]
use cudarc::nvrtc::compile_ptx;

/// CUDA kernel source for batch rotation + quantization.
#[cfg(feature = "gpu")]
const COMPRESS_KERNEL: &str = r#"
extern "C" __global__ void tq_compress_kernel(
    const float* __restrict__ vecs,
    const float* __restrict__ rotation,
    const float* __restrict__ bounds,
    unsigned char* __restrict__ indices,
    float* __restrict__ norms,
    int n_vecs,
    int dim,
    int n_bounds
) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= n_vecs) return;

    const float* v = vecs + vec_idx * dim;
    unsigned char* out = indices + vec_idx * dim;

    // Compute L2 norm
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) norm += v[i] * v[i];
    norm = sqrtf(norm);
    norms[vec_idx] = norm;

    float inv_norm = (norm > 1e-30f) ? (1.0f / norm) : 0.0f;
    float scale = sqrtf((float)dim);

    // Normalize + rotate + scale + quantize
    for (int i = 0; i < dim; i++) {
        float rotated = 0.0f;
        for (int j = 0; j < dim; j++) {
            rotated += rotation[j * dim + i] * v[j] * inv_norm;
        }
        float scaled = rotated * scale;

        // Quantize via boundary search
        unsigned char idx = 0;
        for (int b = 0; b < n_bounds; b++) {
            if (scaled >= bounds[b]) idx = (unsigned char)(b + 1);
        }
        out[i] = idx;
    }
}
"#;

/// GPU compressor — holds rotation matrix and compiled kernel on device.
#[cfg(feature = "gpu")]
pub struct GpuCompressor {
    dev: std::sync::Arc<CudaDevice>,
    d_rotation: CudaSlice<f32>,
    d_bounds: CudaSlice<f32>,
    dim: usize,
    bits: u8,
    n_bounds: usize,
}

#[cfg(feature = "gpu")]
impl GpuCompressor {
    /// Initialize GPU compressor. Compiles kernel, uploads rotation matrix.
    pub fn new(dim: usize, bits: u8, seed: u32) -> Result<Self, String> {
        let ptx = compile_ptx(COMPRESS_KERNEL)
            .map_err(|e| format!("NVRTC: {e}"))?;

        let dev = CudaDevice::new(0)
            .map_err(|e| format!("CUDA init: {e}"))?;

        dev.load_ptx(ptx, "tq", &["tq_compress_kernel"])
            .map_err(|e| format!("Load PTX: {e}"))?;

        // Generate and upload rotation matrix
        let rotation = crate::compress::generate_rotation_pub(dim, seed);
        let d_rotation = dev.htod_copy(rotation)
            .map_err(|e| format!("Upload rotation: {e}"))?;

        // Upload quantization boundaries
        let bounds: Vec<f32> = match bits {
            2 => crate::codebook::BOUNDS_2BIT.to_vec(),
            3 => crate::codebook::BOUNDS_3BIT.to_vec(),
            _ => {
                let cb = crate::codebook::CODEBOOK_4BIT;
                (0..15).map(|i| (cb[i] + cb[i + 1]) / 2.0).collect()
            }
        };
        let n_bounds = bounds.len();
        let d_bounds = dev.htod_copy(bounds)
            .map_err(|e| format!("Upload bounds: {e}"))?;

        Ok(Self { dev, d_rotation, d_bounds, dim, bits, n_bounds })
    }

    /// Compress a batch of vectors on GPU.
    ///
    /// `vecs`: flat `[n_vecs * dim]` f32 array, row-major.
    pub fn compress_batch(
        &self,
        vecs: &[f32],
        n_vecs: usize,
    ) -> Result<Vec<TqVector>, String> {
        let dim = self.dim;
        assert_eq!(vecs.len(), n_vecs * dim);

        let d_vecs = self.dev.htod_copy(vecs.to_vec())
            .map_err(|e| format!("Upload: {e}"))?;
        let mut d_indices = self.dev.alloc_zeros::<u8>(n_vecs * dim)
            .map_err(|e| format!("Alloc idx: {e}"))?;
        let mut d_norms = self.dev.alloc_zeros::<f32>(n_vecs)
            .map_err(|e| format!("Alloc norms: {e}"))?;

        let func = self.dev
            .get_func("tq", "tq_compress_kernel")
            .ok_or("Kernel not found")?;

        let block = 256usize;
        let grid = (n_vecs + block - 1) / block;
        let cfg = LaunchConfig {
            grid_dim: (grid as u32, 1, 1),
            block_dim: (block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (
                &d_vecs,
                &self.d_rotation,
                &self.d_bounds,
                &mut d_indices,
                &mut d_norms,
                n_vecs as i32,
                dim as i32,
                self.n_bounds as i32,
            )).map_err(|e| format!("Launch: {e}"))?;
        }

        let h_indices = self.dev.dtoh_sync_copy(&d_indices)
            .map_err(|e| format!("Download idx: {e}"))?;
        let h_norms = self.dev.dtoh_sync_copy(&d_norms)
            .map_err(|e| format!("Download norms: {e}"))?;

        // Bit-pack on CPU
        let mut results = Vec::with_capacity(n_vecs);
        for i in 0..n_vecs {
            let idx_slice = &h_indices[i * dim..(i + 1) * dim];
            let data = crate::compress::pack_pub(idx_slice, self.bits);
            results.push(TqVector::new(dim as u16, self.bits, h_norms[i], data));
        }

        Ok(results)
    }
}

/// Placeholder for non-GPU builds.
#[cfg(not(feature = "gpu"))]
pub struct GpuCompressor;

#[cfg(not(feature = "gpu"))]
impl GpuCompressor {
    pub fn new(_dim: usize, _bits: u8, _seed: u32) -> Result<Self, String> {
        Err("GPU not compiled. Rebuild with --features gpu".into())
    }
}
