/**
 * @Description  : Prefill-only INT8 VNNI mirror path for the MXFP4 MoE operator.
 *
 * Dual-format scheme (KT_CPU_INT8_PREFILL=1): keep the checkpoint-native MXFP4
 * buffers untouched for decode/verify, and additionally materialize an INT8
 * mirror (u8-biased codes, VPDPBUSD u8x i8 layout + per-16-lane 2^(e-1) scale
 * vectors) used only when the batch is large (prefill).  Weight conversion is
 * exact (E2M1 grid = k/2 * 2^e, code = 2*value in [-12,12], u8 = code+128);
 * the only numerical change is per-32-group dynamic INT8 activation
 * quantization on the prefill path (measured ~0.07% mean output error).
 *
 * Everything here is AVX512_VNNI gated; without it the operator stays on the
 * FP4 path (env is ignored with a one-line notice).
 */
#ifndef CPUINFER_OPERATOR_AMX_INT8_PREFILL_H
#define CPUINFER_OPERATOR_AMX_INT8_PREFILL_H

#include <immintrin.h>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "la/amx_buffers.hpp"  // BufferBInt4KGroupImpl
#include "la/amx_raw_buffers.hpp"  // BufferABF16Impl / BufferCReduceImpl

namespace amx {

// ----------------------------------------------------------------------------
// INT8 mirror of one weight matrix: VNNI layout per (16-n-lane tile, k-group).
//   w8[tile][kg][512B]: byte (k/4)*64 + 4*lane + (k%4)  = code(n_lane, k)+128
//   wexp[tile][kg][16 floats]                           = 2^(e-1) per lane
// Position order inside a group follows PERMUTE_ACT ([even|odd]) so it pairs
// directly with the permuted BufferA rows.
// ----------------------------------------------------------------------------
struct Int8W8Buffer {
  uint8_t* w8 = nullptr;
  float* wexp = nullptr;
  int n = 0, k = 0, kgs = 0, tiles = 0;  // kgs = k/32 groups
  size_t w8_bytes = 0;
  bool owning = true;

  static constexpr int PERM(int p) { return p < 16 ? 2 * p : 2 * (p - 16) + 1; }

  static size_t required_bytes(int n, int k) { return (size_t)n * k + (size_t)n * (k / 32) * 4; }

  void alloc(int n_, int k_) {
    n = n_; k = k_; kgs = k / 32; tiles = n / 16;
    w8_bytes = (size_t)tiles * kgs * 512;
    w8 = (uint8_t*)std::aligned_alloc(64, w8_bytes);
    wexp = (float*)std::aligned_alloc(64, (size_t)tiles * kgs * 16 * sizeof(float));
    if (!w8 || !wexp) throw std::bad_alloc();
  }
  // Non-owning view over externally owned storage (persistent hugepage arena).
  void view(int n_, int k_, uint8_t* w8p, float* wexp_) {
    n = n_; k = k_; kgs = k / 32; tiles = n / 16;
    w8_bytes = (size_t)tiles * kgs * 512;
    w8 = w8p; wexp = wexp_; owning = false;
  }
  ~Int8W8Buffer() {
    if (owning) {
      if (w8) std::free(w8);
      if (wexp) std::free(wexp);
    }
  }

  // Build from the MXFP4 BufferB (packed nibbles, row-major, and se fold
  // addends).  Works on both cold-load and hugepage-REUSED buffers because the
  // packed region is byte-identical to the checkpoint.
  template <typename K>
  void build_from_fp4(BufferBInt4KGroupImpl<K>& bb, int n_start, int n_end) {
    const size_t row_bytes = (size_t)bb.k / 2;
    static const int code_lut[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    for (int n_pos = n_start; n_pos < n_end; n_pos++) {
      const int tile = n_pos / 16, lane = n_pos % 16;
      const uint8_t* src = reinterpret_cast<const uint8_t*>(bb.b) + (size_t)n_pos * row_bytes;
      const int16_t* se_row = bb.get_scale16(n_pos);
      for (int g = 0; g < kgs; g++) {
        const int e = se_row[g] >> 7;  // recover exponent from fold addend
        uint8_t* dst = w8 + ((size_t)tile * kgs + g) * 512;
        for (int p = 0; p < 32; p++) {          // permuted position order
          const int k = PERM(p);
          const int nib = (k & 1) ? (src[(size_t)g * 16 + (k >> 1)] >> 4) : (src[(size_t)g * 16 + (k >> 1)] & 0xF);
          dst[(p >> 2) * 64 + 4 * lane + (p & 3)] = (uint8_t)(code_lut[nib] + 128);
        }
        const float wsc = e == 0 ? 1.0f : exp2f((float)(e - 1));
        wexp[((size_t)tile * kgs + g) * 16 + lane] = wsc;
      }
    }
  }
};

// ----------------------------------------------------------------------------
// Sidecar state.  NOTE: the TP_MOE<AMX_MOE_BASE<T, Derived>> specialization
// instantiates TP_MOE_Common WITHOUT passing Concrete=Derived, so tps[i] is
// allocated as the BASE class and CRTP derived()-> accesses to derived DATA
// MEMBERS are out-of-bounds.  Every AMX/AVX MoE class historically added no
// data members, which is why this latent bug never fired.  All INT8-prefill
// state therefore lives here, keyed by (layer_idx, tp_part_idx), and the
// operator classes must stay member-free.
// ----------------------------------------------------------------------------
struct Int8PrefillSidecar {
  std::vector<std::unique_ptr<Int8W8Buffer>> gate, up, down;
  std::atomic<int> state{0};  // 0=none, 1=building, 2=built
  bool enabled = false;
  int min_m = 64;

  static Int8PrefillSidecar& get(int layer_idx, int tp_idx) {
    static std::mutex mu;
    static std::map<std::pair<int, int>, Int8PrefillSidecar*> registry;
    std::lock_guard<std::mutex> lk(mu);
    auto key = std::make_pair(layer_idx, tp_idx);
    auto it = registry.find(key);
    if (it != registry.end()) return *it->second;
    auto* sc = new Int8PrefillSidecar();
    registry.emplace(key, sc);
    return *sc;
  }
};

// ----------------------------------------------------------------------------
// Per-thread staging: quantize permuted bf16 rows -> s8 codes + fp32 group
// scales + int32 group sums (zero-point correction needs 128*sumA).
// ----------------------------------------------------------------------------
struct Int8ActStaging {
  std::vector<int8_t> codes;   // [row][k]   (permuted position order)
  std::vector<float> scale;    // [row][k/32]
  std::vector<int32_t> suma;   // [row][k/32]
  int rows = 0, k = 0;

  void ensure(int m, int k_) {
    if (m > rows || k_ != k) {
      rows = std::max(m, rows); k = k_;
      codes.resize((size_t)rows * k);
      scale.resize((size_t)rows * (k / 32));
      suma.resize((size_t)rows * (k / 32));
    }
  }

  // Quantize one row (permuted positions; group = 32 consecutive positions).
  inline void quant_row(const ggml_bf16_t* src, int row) {
    const int kgs = k / 32;
    for (int g = 0; g < kgs; g++) {
      __m512 f0, f1;
      avx512_32xbf16_to_32xfp32((__m512i*)(src + (size_t)g * 32), &f0, &f1);
      const __m512 af0 = _mm512_abs_ps(f0), af1 = _mm512_abs_ps(f1);
      float amax = _mm512_reduce_max_ps(_mm512_max_ps(af0, af1));
      if (amax < 1e-12f) amax = 1e-12f;
      const float s = amax / 127.0f;
      scale[(size_t)row * kgs + g] = s;
      const __m512 inv = _mm512_set1_ps(1.0f / s);
      __m512i q0 = _mm512_cvtps_epi32(_mm512_mul_ps(f0, inv));
      __m512i q1 = _mm512_cvtps_epi32(_mm512_mul_ps(f1, inv));
      q0 = _mm512_min_epi32(_mm512_max_epi32(q0, _mm512_set1_epi32(-127)), _mm512_set1_epi32(127));
      q1 = _mm512_min_epi32(_mm512_max_epi32(q1, _mm512_set1_epi32(-127)), _mm512_set1_epi32(127));
      const __m256i packed = _mm256_set_m128i(_mm512_cvtepi32_epi8(q1), _mm512_cvtepi32_epi8(q0));
      _mm256_storeu_si256((__m256i*)(codes.data() + (size_t)row * k + (size_t)g * 32), packed);  // vector heap ptr: 16B-aligned only
      suma[(size_t)row * kgs + g] = _mm512_reduce_add_epi32(q0) + _mm512_reduce_add_epi32(q1);
    }
  }
};

// ----------------------------------------------------------------------------
// GEMM: [m x k] bf16 activations x [n x k] INT8 mirror -> [m x n] fp32.
// ith/nth split over 16-lane n-tiles.  m rows staged per thread.
// ----------------------------------------------------------------------------
template <typename KA, typename KC>
void int8_mat_mul_kgroup(int m, int n, int k, BufferABF16Impl<KA>* ba, Int8W8Buffer* wb,
                         BufferCReduceImpl<KC>* bc, int ith, int nth) {
  const int tiles = wb->tiles, kgs = wb->kgs;
  int t_per = (tiles + nth - 1) / nth;
  const int t_start = std::min(ith * t_per, tiles);
  const int t_end = std::min(t_start + t_per, tiles);
  if (t_start >= t_end) return;

  static thread_local Int8ActStaging stage;
  stage.ensure(m, k);
  for (int r = 0; r < m; r++)
    stage.quant_row((const ggml_bf16_t*)ba->get_submat(m, k, r, 0), r);

  for (int r = 0; r < m; r++) {
    const int8_t* arow = stage.codes.data() + (size_t)r * k;
    const float* ascale = stage.scale.data() + (size_t)r * kgs;
    const int32_t* asum = stage.suma.data() + (size_t)r * kgs;
    for (int t = t_start; t < t_end; t++) {
      const uint8_t* wbase = wb->w8 + (size_t)t * kgs * 512;
      const float* wexp = wb->wexp + (size_t)t * kgs * 16;
      __m512 accf = _mm512_setzero_ps();
      for (int g = 0; g < kgs; g++) {
        __m512i acc = _mm512_setzero_si512();
        const int8_t* ag = arow + (size_t)g * 32;
        for (int i = 0; i < 8; i++) {
          __m512i w = _mm512_load_si512((const __m512i*)(wbase + (size_t)g * 512 + i * 64));
          __m512i a = _mm512_set1_epi32(*(const int*)(ag + i * 4));
          acc = _mm512_dpbusd_epi32(acc, w, a);
        }
        __m512 f = _mm512_cvtepi32_ps(_mm512_sub_epi32(acc, _mm512_set1_epi32(128 * asum[g])));
        accf = _mm512_fmadd_ps(f, _mm512_mul_ps(_mm512_load_ps(wexp + (size_t)g * 16),
                                                _mm512_set1_ps(ascale[g])), accf);
      }
      _mm512_storeu_ps(bc->get_submat(m, n, r, t * 16), accf);
    }
  }
}

}  // namespace amx
#endif
