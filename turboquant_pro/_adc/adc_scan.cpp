// turboquant-pro M1: CPU SIMD batched ADC fast-scan for tq-pro per-dim codes.
//
// Computes, for each query, score[n] = norm[n] * sum_j LUT[j][code[n][j]] over a
// corpus of per-dim 4-bit codes, and returns top-k. The AVX2 path uses the
// faiss-style uint8-LUT pshufb trick (16-entry table lookup, 32 db vectors per
// step, uint16 accumulation). A scalar reference path validates correctness.
//
// Build (Atlas): g++ -O3 -march=native -fopenmp -shared -fPIC -std=c++17 \
//   $(python -m pybind11 --includes) adc_scan.cpp -o adc_scan$(python3-config --extension-suffix)

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace py = pybind11;
static constexpr int BLK = 32;  // db vectors per SIMD step

// Repack (N, d) row-major 4-bit codes (stored 1 byte each, 0..15) into blocked
// layout [nblk][d][BLK] so each (block, dim) holds BLK contiguous codes.
static std::vector<uint8_t> repack(const uint8_t* codes, int64_t N, int d,
                                   int64_t& nblk_out) {
  int64_t nblk = (N + BLK - 1) / BLK;
  nblk_out = nblk;
  std::vector<uint8_t> out((size_t)nblk * d * BLK, 0);
  for (int64_t b = 0; b < nblk; ++b)
    for (int j = 0; j < d; ++j) {
      uint8_t* dst = &out[((size_t)b * d + j) * BLK];
      for (int t = 0; t < BLK; ++t) {
        int64_t n = b * BLK + t;
        if (n < N) dst[t] = codes[(size_t)n * d + j];
      }
    }
  return out;
}

// Per-query top-k over scores[0..N) (float), descending. Simple partial select.
static void topk(const float* sc, int64_t N, int k, int64_t* out_idx,
                 float* out_sc) {
  std::vector<int64_t> idx(N);
  for (int64_t i = 0; i < N; ++i) idx[i] = i;
  int kk = (int)std::min<int64_t>(k, N);
  std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                    [&](int64_t a, int64_t b) { return sc[a] > sc[b]; });
  for (int i = 0; i < kk; ++i) {
    out_idx[i] = idx[i];
    out_sc[i] = sc[idx[i]];
  }
  for (int i = kk; i < k; ++i) {
    out_idx[i] = -1;
    out_sc[i] = -1e30f;
  }
}

// Build float LUT[j][s] = q[j]*cent[s]; quantize to uint8 (global scale, per-dim
// bias). Returns scale and total bias so score = scale*accum + bias.
static void build_lut(const float* q, const float* cent, int d, int S,
                      std::vector<uint8_t>& lut_u8, std::vector<float>& lut_f,
                      float& scale, float& bias) {
  std::vector<float> dmin(d);
  float rmax = 1e-20f;
  for (int j = 0; j < d; ++j) {
    float qj = q[j], lo = 1e30f, hi = -1e30f;
    for (int s = 0; s < S; ++s) {
      float v = qj * cent[s];
      lut_f[j * S + s] = v;
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
    dmin[j] = lo;
    rmax = std::max(rmax, hi - lo);
  }
  scale = rmax / 255.0f;
  bias = 0.0f;
  // uint8 LUT uses a fixed stride of 16 (pshufb needs a 16-entry table); for
  // 3-bit codes (S=8) entries 8..15 are unused padding.
  for (int j = 0; j < d; ++j) {
    bias += dmin[j];
    for (int s = 0; s < 16; ++s) lut_u8[j * 16 + s] = 0;
    for (int s = 0; s < S; ++s) {
      int u = (int)((lut_f[j * S + s] - dmin[j]) / scale + 0.5f);
      lut_u8[j * 16 + s] = (uint8_t)std::min(255, std::max(0, u));
    }
  }
}

// Scalar reference: exact float-LUT ADC for one block of BLK vectors.
static void scan_ref(const uint8_t* blk, const float* lut_f, int d, int S,
                     float* acc) {
  for (int t = 0; t < BLK; ++t) acc[t] = 0.f;
  for (int j = 0; j < d; ++j) {
    const uint8_t* cj = &blk[(size_t)j * BLK];
    const float* lj = &lut_f[(size_t)j * S];
    for (int t = 0; t < BLK; ++t) acc[t] += lj[cj[t]];
  }
}

#if defined(__AVX2__)
// AVX2 pshufb fast-scan for one block (BLK=32), 4-bit codes, uint8 LUT.
static void scan_simd(const uint8_t* blk, const uint8_t* lut_u8, int d,
                      uint16_t* acc16) {
  __m256i a0 = _mm256_setzero_si256();  // 16 x uint16 (vectors 0..15)
  __m256i a1 = _mm256_setzero_si256();  // 16 x uint16 (vectors 16..31)
  for (int j = 0; j < d; ++j) {
    __m128i lut128 = _mm_loadu_si128((const __m128i*)&lut_u8[(size_t)j * 16]);
    __m256i lut = _mm256_broadcastsi128_si256(lut128);     // dup to both lanes
    __m256i codes = _mm256_loadu_si256((const __m256i*)&blk[(size_t)j * BLK]);
    __m256i looked = _mm256_shuffle_epi8(lut, codes);      // 32 x uint8 lookups
    a0 = _mm256_add_epi16(a0, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(looked)));
    a1 = _mm256_add_epi16(a1, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(looked, 1)));
  }
  _mm256_storeu_si256((__m256i*)&acc16[0], a0);
  _mm256_storeu_si256((__m256i*)&acc16[16], a1);
}
#endif

// Score model (general cosine over 768-d reconstruction):
//   score[n] = (qbias[qi] + vnorm[n] * adc[n]) * vrnorm[n]
// where adc[n] = sum_j q_rot[j]*cent[code[j]]. For the plain 256-d cosine set
// qbias=0, vnorm=1, vrnorm=1/||cent[codes]||.
py::tuple search(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> codes,
                 py::array_t<float, py::array::c_style | py::array::forcecast> queries,
                 py::array_t<float, py::array::c_style | py::array::forcecast> cent,
                 py::array_t<float, py::array::c_style | py::array::forcecast> vnorm,
                 py::array_t<float, py::array::c_style | py::array::forcecast> vrnorm,
                 py::array_t<float, py::array::c_style | py::array::forcecast> qbias,
                 int k, bool use_simd) {
  auto cb = codes.unchecked<2>();
  auto qb = queries.unchecked<2>();
  int64_t N = cb.shape(0);
  int d = (int)cb.shape(1);
  int Q = (int)qb.shape(0);
  int S = (int)cent.shape(0);
  const float* cptr = cent.data();
  const float* vn = vnorm.data();
  const float* vr = vrnorm.data();
  const float* qbi = qbias.data();
  int64_t nblk;
  std::vector<uint8_t> blocked = repack(codes.data(), N, d, nblk);

  auto out_idx = py::array_t<int64_t>({(py::ssize_t)Q, (py::ssize_t)k});
  auto out_sc = py::array_t<float>({(py::ssize_t)Q, (py::ssize_t)k});
  int64_t* oi = out_idx.mutable_data();
  float* os = out_sc.mutable_data();

#pragma omp parallel
  {
    std::vector<float> lut_f((size_t)d * S), scores(N);
    std::vector<uint8_t> lut_u8((size_t)d * 16);  // fixed 16-stride for pshufb
    std::vector<uint16_t> acc16(BLK);
    std::vector<float> accf(BLK);
#pragma omp for schedule(dynamic)
    for (int qi = 0; qi < Q; ++qi) {
      float scale, bias;
      build_lut(&qb(qi, 0), cptr, d, S, lut_u8, lut_f, scale, bias);
      float qb_bias = qbi[qi];
      for (int64_t b = 0; b < nblk; ++b) {
        const uint8_t* blk = &blocked[(size_t)b * d * BLK];
        int64_t base = b * BLK;
#if defined(__AVX2__)
        if (use_simd) {
          scan_simd(blk, lut_u8.data(), d, acc16.data());
          for (int t = 0; t < BLK; ++t) {
            int64_t n = base + t;
            if (n < N)
              scores[n] = (qb_bias + vn[n] * (scale * (float)acc16[t] + bias)) * vr[n];
          }
          continue;
        }
#endif
        scan_ref(blk, lut_f.data(), d, S, accf.data());
        for (int t = 0; t < BLK; ++t) {
          int64_t n = base + t;
          if (n < N) scores[n] = (qb_bias + vn[n] * accf[t]) * vr[n];
        }
      }
      topk(scores.data(), N, k, &oi[(size_t)qi * k], &os[(size_t)qi * k]);
    }
  }
  return py::make_tuple(out_idx, out_sc);
}

PYBIND11_MODULE(adc_scan, m) {
  m.doc() = "tq-pro M1 CPU SIMD batched ADC fast-scan";
  m.def("search", &search, py::arg("codes"), py::arg("queries"), py::arg("cent"),
        py::arg("vnorm"), py::arg("vrnorm"), py::arg("qbias"), py::arg("k") = 10,
        py::arg("use_simd") = true);
}
