// turboquant-pro M1: CPU SIMD batched ADC fast-scan for tq-pro per-dim codes.
//
// Computes, for each query, score[n] over a corpus of per-dim codes (<= 4 bits) and
// returns top-k. The AVX2 path uses the faiss-style uint8-LUT pshufb trick (16-entry
// table lookup, 32 db vectors per step). A scalar reference path computes the exact
// float-LUT score.
//
// Layout and arithmetic (v3, 2026-09-17):
// - The index stores its codes in this kernel's blocked layout, packed once at add time
//   by ``turboquant_pro.packed_codes.pack_blocks`` and never repacked at search: blocks
//   of 32 vectors x d dims, two 4-bit codes per byte, 16 bytes per (block, dim) strip;
//   slot t of a block is the low nibble of byte t/2 when t is even, the high nibble
//   when odd. Rows are scanned in **chunks** (one packed array each, with a row offset
//   into the flat per-row arrays); a flat index is one chunk per add() batch and an
//   IVF cell is one chunk with a centroid, so one entry point serves both.
// - Each dim has its own symbol table (``tables[d][16]``, ``nsym[d]`` entries used),
//   so dims quantized at different bit widths scan together. Dims are grouped into
//   contiguous **segments** with a per-row weight each (a segmented pipeline stores
//   the energy fraction of each segment); a uniform index has one segment and no
//   weights. The score of row n for a query with constant ``bias`` on this chunk is
//     (bias + vnorm[n] * sum_seg w[n][seg] * (scale * acc_seg[n] + lutbias_seg)) * vrnorm[n]
//   where acc_seg is the uint8 lookup sum over the segment's dims, ``scale`` the query's
//   global LUT scale and ``lutbias_seg`` the sum of the per-dim table minima. With one
//   segment this is exactly the v2 arithmetic.
// - The SIMD path accumulates uint8 lookups in uint16 for at most ACC_FLUSH dims, then
//   folds into uint32. v1 accumulated all d dims in uint16, which wraps past 65535 once
//   d > 257: the wrap hit exactly the highest-scoring vectors (1536-d, 2-bit,
//   text-embedding-3-large: recall@10 0.848 vs 0.905 exact).
// - Top-k is streaming: a size-k min-heap and a threshold per query, across every
//   chunk the query probes.
//
// Build: python -m turboquant_pro._adc   (g++ -O3 -march=native -fopenmp ...)

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace py = pybind11;
static constexpr int BLK = 32;         // db vectors per SIMD step
static constexpr int TAB = 16;         // symbol-table stride (pshufb needs 16 entries)
static constexpr int ACC_FLUSH = 256;  // dims per uint16 accumulation (255*256 < 65536)

// Repack (N, d) row-major codes (1 byte each, 0..15) into [nblk][d][16 bytes]. The same
// layout ``packed_codes.pack_blocks`` produces in numpy; this one is for the legacy
// ``search`` wrapper and for callers who want the C++ speed.
static std::vector<uint8_t> repack4(const uint8_t* codes, int64_t N, int d,
                                    int64_t& nblk_out) {
  int64_t nblk = (N + BLK - 1) / BLK;
  nblk_out = nblk;
  std::vector<uint8_t> out((size_t)nblk * d * (BLK / 2), 0);
  for (int64_t b = 0; b < nblk; ++b)
    for (int j = 0; j < d; ++j) {
      uint8_t* dst = &out[((size_t)b * d + j) * (BLK / 2)];
      for (int t = 0; t < BLK; ++t) {
        int64_t n = b * BLK + t;
        uint8_t c = n < N ? (uint8_t)(codes[(size_t)n * d + j] & 0x0F) : 0;
        dst[t / 2] |= (t % 2 == 0) ? c : (uint8_t)(c << 4);
      }
    }
  return out;
}

// Streaming top-k: min-heap of (score, index) holding the best k seen so far.
struct TopK {
  int k;
  std::vector<std::pair<float, int64_t>> heap;  // min-heap on score
  explicit TopK(int k_) : k(k_) { heap.reserve((size_t)k_); }
  void reset() { heap.clear(); }
  float threshold() const {
    return (int)heap.size() < k ? -std::numeric_limits<float>::infinity() : heap.front().first;
  }
  static bool cmp(const std::pair<float, int64_t>& a, const std::pair<float, int64_t>& b) {
    return a.first > b.first;  // makes std::heap a min-heap on score
  }
  void push(float s, int64_t i) {
    if ((int)heap.size() < k) {
      heap.emplace_back(s, i);
      std::push_heap(heap.begin(), heap.end(), cmp);
    } else if (s > heap.front().first) {
      std::pop_heap(heap.begin(), heap.end(), cmp);
      heap.back() = {s, i};
      std::push_heap(heap.begin(), heap.end(), cmp);
    }
  }
  void emit(int64_t* out_idx, float* out_sc) {
    std::sort(heap.begin(), heap.end(),
              [](const std::pair<float, int64_t>& a, const std::pair<float, int64_t>& b) {
                return a.first > b.first || (a.first == b.first && a.second < b.second);
              });
    int m = (int)heap.size();
    for (int i = 0; i < m; ++i) {
      out_idx[i] = heap[i].second;
      out_sc[i] = heap[i].first;
    }
    for (int i = m; i < k; ++i) {
      out_idx[i] = -1;
      out_sc[i] = -1e30f;
    }
  }
};

// The per-dim symbol tables and segment layout a scan runs against.
struct Tables {
  const float* tab;   // (d, TAB) float32
  const int32_t* nsym;  // (d,)
  const int32_t* segs;  // (nseg + 1,) dim offsets
  int d;
  int nseg;
};

// Build LUT[j][s] = q[j]*tab[j][s]; quantize to uint8 (global scale, per-dim bias).
// score contribution of a segment = scale*accum_seg + segbias[seg].
static void build_lut(const float* q, const Tables& T, std::vector<uint8_t>& lut_u8,
                      std::vector<float>& lut_f, float& scale, std::vector<float>& segbias) {
  const int d = T.d;
  std::vector<float> dmin(d);
  float rmax = 1e-20f;
  for (int j = 0; j < d; ++j) {
    const float qj = q[j];
    const int S = T.nsym[j];
    float lo = 1e30f, hi = -1e30f;
    for (int s = 0; s < TAB; ++s) lut_f[(size_t)j * TAB + s] = 0.f;
    for (int s = 0; s < S; ++s) {
      float v = qj * T.tab[(size_t)j * TAB + s];
      lut_f[(size_t)j * TAB + s] = v;
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
    dmin[j] = lo;
    rmax = std::max(rmax, hi - lo);
  }
  scale = rmax / 255.0f;
  for (int g = 0; g < T.nseg; ++g) {
    float b = 0.0f;
    for (int j = T.segs[g]; j < T.segs[g + 1]; ++j) b += dmin[j];
    segbias[g] = b;
  }
  for (int j = 0; j < d; ++j) {
    const int S = T.nsym[j];
    for (int s = 0; s < TAB; ++s) lut_u8[(size_t)j * TAB + s] = 0;
    for (int s = 0; s < S; ++s) {
      int u = (int)((lut_f[(size_t)j * TAB + s] - dmin[j]) / scale + 0.5f);
      lut_u8[(size_t)j * TAB + s] = (uint8_t)std::min(255, std::max(0, u));
    }
  }
}

// Scalar reference: exact float-LUT ADC over dims [jbeg, jend) for one block.
static void scan_ref_range(const uint8_t* blk, const float* lut_f, int jbeg, int jend,
                           float* acc) {
  for (int t = 0; t < BLK; ++t) acc[t] = 0.f;
  for (int j = jbeg; j < jend; ++j) {
    const uint8_t* cj = &blk[(size_t)j * (BLK / 2)];
    const float* lj = &lut_f[(size_t)j * TAB];
    for (int t = 0; t < BLK; ++t) {
      uint8_t c = (t % 2 == 0) ? (cj[t / 2] & 0x0F) : (cj[t / 2] >> 4);
      acc[t] += lj[c];
    }
  }
}

#if defined(__AVX2__)
// AVX2 pshufb fast-scan over dims [jbeg, jend) for one block (BLK=32), nibble-packed
// codes, uint8 LUT. uint16 lanes are folded into uint32 every ACC_FLUSH dims, so no
// sum can wrap.
static void scan_simd_range(const uint8_t* blk, const uint8_t* lut_u8, int jbeg, int jend,
                            uint32_t* acc32) {
  const __m128i low4 = _mm_set1_epi8(0x0F);
  for (int t = 0; t < BLK; ++t) acc32[t] = 0;
  uint16_t tmp[BLK];
  for (int j0 = jbeg; j0 < jend; j0 += ACC_FLUSH) {
    int j1 = std::min(jend, j0 + ACC_FLUSH);
    __m256i a0 = _mm256_setzero_si256();  // 16 x uint16 (vectors 0..15)
    __m256i a1 = _mm256_setzero_si256();  // 16 x uint16 (vectors 16..31)
    for (int j = j0; j < j1; ++j) {
      __m128i packed = _mm_loadu_si128((const __m128i*)&blk[(size_t)j * (BLK / 2)]);
      __m128i lo = _mm_and_si128(packed, low4);
      __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), low4);
      // codes in vector order: lo0 hi0 lo1 hi1 ... (bytes 0..7), then bytes 8..15
      __m256i codes = _mm256_inserti128_si256(
          _mm256_castsi128_si256(_mm_unpacklo_epi8(lo, hi)), _mm_unpackhi_epi8(lo, hi), 1);
      __m128i lut128 = _mm_loadu_si128((const __m128i*)&lut_u8[(size_t)j * TAB]);
      __m256i lut = _mm256_broadcastsi128_si256(lut128);  // same table in both lanes
      __m256i looked = _mm256_shuffle_epi8(lut, codes);     // 32 x uint8 lookups
      a0 = _mm256_add_epi16(a0, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(looked)));
      a1 = _mm256_add_epi16(a1, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(looked, 1)));
    }
    _mm256_storeu_si256((__m256i*)&tmp[0], a0);
    _mm256_storeu_si256((__m256i*)&tmp[16], a1);
    for (int t = 0; t < BLK; ++t) acc32[t] += tmp[t];
  }
}
#endif

// Scalar uint8-LUT sum over dims [jbeg, jend) for one vector (slot t of a block): the same
// integer arithmetic as scan_simd_range, used for sparse survivors and without AVX2.
static inline uint32_t sum_u8_range(const uint8_t* blk, const uint8_t* lut_u8, int jbeg,
                                    int jend, int t) {
  uint32_t acc = 0;
  const size_t half = (size_t)(t / 2);
  const int shift = (t % 2) ? 4 : 0;
  for (int j = jbeg; j < jend; ++j) {
    uint8_t c = (uint8_t)((blk[(size_t)j * (BLK / 2) + half] >> shift) & 0x0F);
    acc += lut_u8[(size_t)j * TAB + c];
  }
  return acc;
}

// One chunk of blocked codes: ``nblk`` blocks holding ``n`` rows, which are rows
// ``off .. off + n`` of the flat per-row arrays.
struct Chunk {
  const uint8_t* blk;
  int64_t n;
  int64_t nblk;
  int64_t off;
};

struct RowTerms {
  const float* vnorm;   // (N,)
  const float* vrnorm;  // (N,)
  const float* segw;    // (N, nseg) or nullptr when nseg == 1 and unweighted
  int nseg;
};

// Score of row n (global) from its per-segment lookup sums.
static inline float row_score(const RowTerms& R, int64_t n, float bias, float scale,
                              const float* segbias, const uint32_t* acc, int stride,
                              int t) {
  float inner = 0.f;
  if (R.segw == nullptr) {
    inner = scale * (float)acc[t] + segbias[0];
  } else {
    const float* w = &R.segw[(size_t)n * R.nseg];
    for (int g = 0; g < R.nseg; ++g)
      inner += w[g] * (scale * (float)acc[(size_t)g * stride + t] + segbias[g]);
  }
  return (bias + R.vnorm[n] * inner) * R.vrnorm[n];
}

static inline float row_score_f(const RowTerms& R, int64_t n, float bias, const float* acc,
                                int stride, int t) {
  float inner = 0.f;
  if (R.segw == nullptr) {
    inner = acc[t];
  } else {
    const float* w = &R.segw[(size_t)n * R.nseg];
    for (int g = 0; g < R.nseg; ++g) inner += w[g] * acc[(size_t)g * stride + t];
  }
  return (bias + R.vnorm[n] * inner) * R.vrnorm[n];
}

// The scan: every query against the chunks it probes, one streaming top-k per query.
static void scan_core(const std::vector<Chunk>& chunks, const float* queries, int Q,
                      const Tables& T, const RowTerms& R, const int32_t* probes,
                      const float* biases, int P, int k, bool use_simd, int64_t* oi,
                      float* os) {
  const int d = T.d;
  const int nseg = T.nseg;
  const size_t strip = (size_t)d * (BLK / 2);
#pragma omp parallel
  {
    std::vector<float> lut_f((size_t)d * TAB);
    std::vector<uint8_t> lut_u8((size_t)d * TAB);
    std::vector<float> segbias((size_t)nseg);
    std::vector<uint32_t> acc32((size_t)nseg * BLK);
    std::vector<float> accf((size_t)nseg * BLK);
    TopK top(k);
#pragma omp for schedule(dynamic)
    for (int qi = 0; qi < Q; ++qi) {
      float scale;
      build_lut(&queries[(size_t)qi * d], T, lut_u8, lut_f, scale, segbias);
      top.reset();
      for (int p = 0; p < P; ++p) {
        const int32_t c = probes[(size_t)qi * P + p];
        if (c < 0) continue;
        const float bias = biases[(size_t)qi * P + p];
        const Chunk& ch = chunks[(size_t)c];
        for (int64_t b = 0; b < ch.nblk; ++b) {
          const uint8_t* blk = &ch.blk[(size_t)b * strip];
          const int64_t base = b * BLK;
          const int tmax = (int)std::min<int64_t>(BLK, ch.n - base);
          float thr = top.threshold();
#if defined(__AVX2__)
          if (use_simd) {
            for (int g = 0; g < nseg; ++g)
              scan_simd_range(blk, lut_u8.data(), T.segs[g], T.segs[g + 1],
                              &acc32[(size_t)g * BLK]);
            for (int t = 0; t < tmax; ++t) {
              const int64_t n = ch.off + base + t;
              float s = row_score(R, n, bias, scale, segbias.data(), acc32.data(), BLK, t);
              if (s > thr) {
                top.push(s, n);
                thr = top.threshold();
              }
            }
            continue;
          }
#endif
          for (int g = 0; g < nseg; ++g)
            scan_ref_range(blk, lut_f.data(), T.segs[g], T.segs[g + 1], &accf[(size_t)g * BLK]);
          for (int t = 0; t < tmax; ++t) {
            const int64_t n = ch.off + base + t;
            float s = row_score_f(R, n, bias, accf.data(), BLK, t);
            if (s > thr) {
              top.push(s, n);
              thr = top.threshold();
            }
          }
        }
      }
      top.emit(&oi[(size_t)qi * k], &os[(size_t)qi * k]);
    }
  }
}

template <typename T>
using arr = py::array_t<T, py::array::c_style | py::array::forcecast>;

static Tables tables_of(const arr<float>& tables, const arr<int32_t>& nsym,
                        const arr<int32_t>& segs) {
  if (tables.ndim() != 2 || tables.shape(1) != TAB)
    throw std::invalid_argument("adc_scan: tables must have shape (d, 16)");
  Tables T;
  T.d = (int)tables.shape(0);
  T.tab = tables.data();
  if (nsym.ndim() != 1 || nsym.shape(0) != T.d)
    throw std::invalid_argument("adc_scan: nsym must have shape (d,)");
  T.nsym = nsym.data();
  for (int j = 0; j < T.d; ++j)
    if (T.nsym[j] < 1 || T.nsym[j] > TAB)
      throw std::invalid_argument("adc_scan: every dim needs 1..16 symbols (<= 4 bits)");
  if (segs.ndim() != 1 || segs.shape(0) < 2)
    throw std::invalid_argument("adc_scan: segs must be (nseg + 1,) dim offsets");
  T.nseg = (int)segs.shape(0) - 1;
  T.segs = segs.data();
  if (T.segs[0] != 0 || T.segs[T.nseg] != T.d)
    throw std::invalid_argument("adc_scan: segs must start at 0 and end at d");
  for (int g = 0; g < T.nseg; ++g)
    if (T.segs[g + 1] <= T.segs[g]) throw std::invalid_argument("adc_scan: empty segment");
  return T;
}

// search_chunks: the v3 entry point. ``chunks`` is a list of blocked uint8 arrays,
// ``ns`` their row counts, ``offsets`` their row offsets into the flat per-row arrays.
py::tuple search_chunks(py::list chunks, arr<int64_t> ns, arr<int64_t> offsets,
                        arr<float> queries, arr<float> tables, arr<int32_t> nsym,
                        arr<int32_t> segs, arr<float> vnorm, arr<float> vrnorm,
                        py::object segw, arr<int32_t> probes, arr<float> biases, int k,
                        bool use_simd) {
  const Tables T = tables_of(tables, nsym, segs);
  const int d = T.d;
  if (queries.ndim() != 2 || queries.shape(1) != d)
    throw std::invalid_argument("adc_scan: queries must have shape (Q, d)");
  const int Q = (int)queries.shape(0);
  const int nc = (int)chunks.size();
  if (ns.shape(0) != nc || offsets.shape(0) != nc)
    throw std::invalid_argument("adc_scan: ns and offsets must have one entry per chunk");
  const size_t strip = (size_t)d * (BLK / 2);
  std::vector<arr<uint8_t>> keep;  // hold the buffers while the GIL is released
  std::vector<Chunk> cv(nc);
  int64_t N = 0;
  for (int c = 0; c < nc; ++c) {
    keep.push_back(chunks[c].cast<arr<uint8_t>>());
    const arr<uint8_t>& a = keep.back();
    Chunk ch;
    ch.n = ns.data()[c];
    ch.off = offsets.data()[c];
    ch.nblk = (ch.n + BLK - 1) / BLK;
    if ((int64_t)a.size() != ch.nblk * (int64_t)strip)
      throw std::invalid_argument("adc_scan: chunk bytes do not match nblk * d * 16");
    ch.blk = a.data();
    cv[c] = ch;
    N = std::max(N, ch.off + ch.n);
  }
  if (vnorm.shape(0) < N || vrnorm.shape(0) < N)
    throw std::invalid_argument("adc_scan: vnorm/vrnorm shorter than the rows the chunks cover");
  RowTerms R;
  R.vnorm = vnorm.data();
  R.vrnorm = vrnorm.data();
  R.nseg = T.nseg;
  R.segw = nullptr;
  arr<float> segw_a;
  if (!segw.is_none()) {
    segw_a = segw.cast<arr<float>>();
    if (segw_a.ndim() != 2 || segw_a.shape(1) != T.nseg || segw_a.shape(0) < N)
      throw std::invalid_argument("adc_scan: segw must have shape (N, nseg)");
    R.segw = segw_a.data();
  } else if (T.nseg != 1) {
    throw std::invalid_argument("adc_scan: several segments need per-row weights (segw)");
  }
  if (probes.ndim() != 2 || probes.shape(0) != Q || biases.ndim() != 2 ||
      biases.shape(0) != Q || biases.shape(1) != probes.shape(1))
    throw std::invalid_argument("adc_scan: probes and biases must have shape (Q, P)");
  const int P = (int)probes.shape(1);
  for (py::ssize_t i = 0; i < probes.size(); ++i)
    if (probes.data()[i] >= nc) throw std::invalid_argument("adc_scan: probe names no chunk");

  auto out_idx = py::array_t<int64_t>({(py::ssize_t)Q, (py::ssize_t)k});
  auto out_sc = py::array_t<float>({(py::ssize_t)Q, (py::ssize_t)k});
  int64_t* oi = out_idx.mutable_data();
  float* os = out_sc.mutable_data();
  {
    py::gil_scoped_release release;
    scan_core(cv, queries.data(), Q, T, R, probes.data(), biases.data(), P, k, use_simd, oi,
              os);
  }
  return py::make_tuple(out_idx, out_sc);
}

// pack: (N, d) uint8 codes -> blocked bytes, the C++ twin of packed_codes.pack_blocks.
py::array_t<uint8_t> pack(arr<uint8_t> codes) {
  if (codes.ndim() != 2) throw std::invalid_argument("adc_scan: codes must be (N, d)");
  int64_t nblk;
  std::vector<uint8_t> blocked =
      repack4(codes.data(), codes.shape(0), (int)codes.shape(1), nblk);
  auto out = py::array_t<uint8_t>((py::ssize_t)blocked.size());
  std::copy(blocked.begin(), blocked.end(), out.mutable_data());
  return out;
}

// search: the v1/v2 signature, kept for callers who hold plain (N, d) codes and one
// symbol table. Packs on the fly (so it costs an O(N d) pass per call); the index
// itself uses search_chunks on codes packed once.
py::tuple search(arr<uint8_t> codes, arr<float> queries, arr<float> cent, arr<float> vnorm,
                 arr<float> vrnorm, arr<float> qbias, int k, bool use_simd) {
  auto cb = codes.unchecked<2>();
  const int64_t N = cb.shape(0);
  const int d = (int)cb.shape(1);
  const int Q = (int)queries.shape(0);
  const int S = (int)cent.shape(0);
  if (S > TAB) throw std::invalid_argument("adc_scan: codes must be <= 4 bits (S <= 16)");
  std::vector<float> tab((size_t)d * TAB, 0.f);
  std::vector<int32_t> nsym((size_t)d, S);
  for (int j = 0; j < d; ++j)
    for (int s = 0; s < S; ++s) tab[(size_t)j * TAB + s] = cent.data()[s];
  int32_t segs[2] = {0, d};
  Tables T;
  T.tab = tab.data();
  T.nsym = nsym.data();
  T.segs = segs;
  T.d = d;
  T.nseg = 1;
  int64_t nblk;
  std::vector<uint8_t> blocked = repack4(codes.data(), N, d, nblk);
  std::vector<Chunk> cv(1);
  cv[0].blk = blocked.data();
  cv[0].n = N;
  cv[0].nblk = nblk;
  cv[0].off = 0;
  RowTerms R;
  R.vnorm = vnorm.data();
  R.vrnorm = vrnorm.data();
  R.segw = nullptr;
  R.nseg = 1;
  std::vector<int32_t> probes((size_t)Q, 0);
  auto out_idx = py::array_t<int64_t>({(py::ssize_t)Q, (py::ssize_t)k});
  auto out_sc = py::array_t<float>({(py::ssize_t)Q, (py::ssize_t)k});
  int64_t* oi = out_idx.mutable_data();
  float* os = out_sc.mutable_data();
  {
    py::gil_scoped_release release;
    scan_core(cv, queries.data(), Q, T, R, probes.data(), qbias.data(), 1, k, use_simd, oi,
              os);
  }
  return py::make_tuple(out_idx, out_sc);
}

// Two-pass pruned scan (experiment, docs/PREREG_pruned_scan.md), on one blocked chunk
// with one segment.
//
// The score is increasing in the uint8 lookup sum acc = sum_j u[j][code_j]. Pass 1 sums the
// first m dims of every vector and extrapolates the rest from the vector's own centred
// prefix (the ADSampling-style estimate)
//   est = acc_m + mu_rest + (acc_m - mu_pre) * (d - m) / m,
// where mu_j and var_j are the mean and variance of u[j][.] under the corpus code
// frequencies and sd = sqrt(var_rest + var_pre * ((d - m) / m)^2). The threshold for pass 2
// is the k-th largest lower score at est - z*sd; pass 2 finishes only the vectors whose upper
// score at est + z*sd reaches it, with their exact uint8 score, so every returned score
// equals the unpruned kernel's. A block with at least SIMD_MIN survivors finishes in SIMD;
// sparser survivors are summed scalar. Returns (idx, scores, survivors per query).
static constexpr int SIMD_MIN = 6;

py::tuple search_pruned(arr<uint8_t> blocked, int64_t N, arr<float> queries, arr<float> tables,
                        arr<int32_t> nsym, arr<float> vnorm, arr<float> vrnorm,
                        arr<float> qbias, arr<float> freq, int k, int m, float z) {
  const int d = (int)tables.shape(0);
  int32_t segs_arr[2] = {0, d};
  arr<int32_t> segs(2, segs_arr);
  const Tables T = tables_of(tables, nsym, segs);
  auto fq = freq.unchecked<2>();
  const int Q = (int)queries.shape(0);
  if (fq.shape(0) != d || fq.shape(1) != TAB)
    throw std::invalid_argument("adc_scan: freq must have shape (d, 16)");
  if (m < 1 || m >= d) throw std::invalid_argument("adc_scan: need 1 <= m < d");
  const int64_t nblk = (N + BLK - 1) / BLK;
  const size_t strip = (size_t)d * (BLK / 2);
  if ((int64_t)blocked.size() != nblk * (int64_t)strip)
    throw std::invalid_argument("adc_scan: blocked bytes do not match nblk * d * 16");
  const uint8_t* blocks = blocked.data();
  const float* vn = vnorm.data();
  const float* vr = vrnorm.data();
  const float* qbi = qbias.data();

  auto out_idx = py::array_t<int64_t>({(py::ssize_t)Q, (py::ssize_t)k});
  auto out_sc = py::array_t<float>({(py::ssize_t)Q, (py::ssize_t)k});
  auto out_surv = py::array_t<int64_t>({(py::ssize_t)Q});
  int64_t* oi = out_idx.mutable_data();
  float* os = out_sc.mutable_data();
  int64_t* osv = out_surv.mutable_data();
  const double slope = (double)(d - m) / (double)m;

  {
    py::gil_scoped_release release;
#pragma omp parallel
    {
      std::vector<float> lut_f((size_t)d * TAB);
      std::vector<uint8_t> lut_u8((size_t)d * TAB);
      std::vector<float> segbias(1);
      std::vector<uint32_t> accm((size_t)N);
      uint32_t acc32[BLK];
      int surv_t[BLK];
      TopK lower(k), top(k);
#pragma omp for schedule(dynamic)
      for (int qi = 0; qi < Q; ++qi) {
        float scale;
        build_lut(&queries.data()[(size_t)qi * d], T, lut_u8, lut_f, scale, segbias);
        const float bias = segbias[0];
        const double qb_bias = qbi[qi];
        double mu_pre = 0, mu_rest = 0, var_pre = 0, var_rest = 0;
        for (int j = 0; j < d; ++j) {
          double mu = 0, m2 = 0;
          for (int s = 0; s < TAB; ++s) {
            double u = lut_u8[(size_t)j * TAB + s];
            mu += fq(j, s) * u;
            m2 += fq(j, s) * u * u;
          }
          double var = std::max(0.0, m2 - mu * mu);
          if (j < m) {
            mu_pre += mu;
            var_pre += var;
          } else {
            mu_rest += mu;
            var_rest += var;
          }
        }
        const double sd = std::sqrt(var_rest + var_pre * slope * slope);
        // the kernel's own score arithmetic (float32), as a function of the lookup sum
        auto exact = [&](int64_t n, uint32_t acc) {
          return ((float)qb_bias + vn[n] * (scale * (float)acc + bias)) * vr[n];
        };
        auto bound = [&](int64_t n, double acc) {
          return (float)((qb_bias + vn[n] * (scale * acc + bias)) * vr[n]);
        };

        // pass 1: prefix sums and the lower-bound threshold
        lower.reset();
        for (int64_t b = 0; b < nblk; ++b) {
          const uint8_t* blk = &blocks[(size_t)b * strip];
          int64_t base = b * BLK;
          int tmax = (int)std::min<int64_t>(BLK, N - base);
#if defined(__AVX2__)
          scan_simd_range(blk, lut_u8.data(), 0, m, acc32);
#else
          for (int t = 0; t < tmax; ++t) acc32[t] = sum_u8_range(blk, lut_u8.data(), 0, m, t);
#endif
          float lthr = lower.threshold();
          for (int t = 0; t < tmax; ++t) {
            int64_t n = base + t;
            accm[(size_t)n] = acc32[t];
            double est = acc32[t] + mu_rest + (acc32[t] - mu_pre) * slope;
            float lo = bound(n, est - z * sd);
            if (lo > lthr) {
              lower.push(lo, n);
              lthr = lower.threshold();
            }
          }
        }
        const float thr = lower.threshold();  // -inf when N < k: nothing is pruned

        // pass 2: finish the survivors exactly
        top.reset();
        int64_t survivors = 0;
        for (int64_t b = 0; b < nblk; ++b) {
          const uint8_t* blk = &blocks[(size_t)b * strip];
          int64_t base = b * BLK;
          int tmax = (int)std::min<int64_t>(BLK, N - base);
          int ns = 0;
          for (int t = 0; t < tmax; ++t) {
            int64_t n = base + t;
            double a = accm[(size_t)n];
            double est = a + mu_rest + (a - mu_pre) * slope;
            if (bound(n, est + z * sd) >= thr) surv_t[ns++] = t;
          }
          if (ns == 0) continue;
          survivors += ns;
          float fthr = top.threshold();
#if defined(__AVX2__)
          if (ns >= SIMD_MIN) {
            scan_simd_range(blk, lut_u8.data(), m, d, acc32);
            for (int i = 0; i < ns; ++i) {
              int64_t n = base + surv_t[i];
              float s = exact(n, accm[(size_t)n] + acc32[surv_t[i]]);
              if (s > fthr) {
                top.push(s, n);
                fthr = top.threshold();
              }
            }
            continue;
          }
#endif
          for (int i = 0; i < ns; ++i) {
            int64_t n = base + surv_t[i];
            uint32_t rest = sum_u8_range(blk, lut_u8.data(), m, d, surv_t[i]);
            float s = exact(n, accm[(size_t)n] + rest);
            if (s > fthr) {
              top.push(s, n);
              fthr = top.threshold();
            }
          }
        }
        top.emit(&oi[(size_t)qi * k], &os[(size_t)qi * k]);
        osv[qi] = survivors;
      }
    }
  }
  return py::make_tuple(out_idx, out_sc, out_surv);
}

PYBIND11_MODULE(adc_scan, m) {
  m.doc() =
      "tq-pro M1 CPU SIMD batched ADC fast-scan (v3: blocked codes packed once, chunks, "
      "per-dim symbol tables, segments, streaming top-k, no uint16 wrap)";
  // v4: reports whether the AVX2 uint8-LUT path was compiled in, so a result can name
  // the scorer that made it (turboquant_pro.scorer). No change to any scan.
  m.attr("VERSION") = 4;
#if defined(__AVX2__)
  m.attr("SIMD") = true;
#else
  m.attr("SIMD") = false;
#endif
  m.def("pack", &pack, py::arg("codes"));
  m.def("search_chunks", &search_chunks, py::arg("chunks"), py::arg("ns"), py::arg("offsets"),
        py::arg("queries"), py::arg("tables"), py::arg("nsym"), py::arg("segs"),
        py::arg("vnorm"), py::arg("vrnorm"), py::arg("segw"), py::arg("probes"),
        py::arg("biases"), py::arg("k") = 10, py::arg("use_simd") = true);
  m.def("search", &search, py::arg("codes"), py::arg("queries"), py::arg("cent"),
        py::arg("vnorm"), py::arg("vrnorm"), py::arg("qbias"), py::arg("k") = 10,
        py::arg("use_simd") = true);
  m.def("search_pruned", &search_pruned, py::arg("blocked"), py::arg("n"), py::arg("queries"),
        py::arg("tables"), py::arg("nsym"), py::arg("vnorm"), py::arg("vrnorm"),
        py::arg("qbias"), py::arg("freq"), py::arg("k"), py::arg("m"), py::arg("z"));
}
