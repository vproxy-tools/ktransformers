// Standalone MXFP4 GEMM kernel testbed v2 — DRAM-sized weight arena,
// prefetch-distance sweep, m-loop over 4-row tiles.
// Build: g++ -O3 -march=native -std=c++20 -o /tmp/kern_test /tmp/kern_test.cpp
// Run:   taskset -c 0 /tmp/kern_test <m> <iters> <nexp>
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

alignas(64) static constexpr uint8_t LUT0[64] = {
    0x00, 0x00, 0x00, 0x3F, 0x80, 0x3F, 0xC0, 0x3F, 0x00, 0x40, 0x40, 0x40,
    0x80, 0x40, 0xC0, 0x40, 0x00, 0x80, 0x00, 0xBF, 0x80, 0xBF, 0xC0, 0xBF,
    0x00, 0xC0, 0x40, 0xC0, 0x80, 0xC0, 0xC0, 0xC0,
    0x00, 0x00, 0x00, 0x3F, 0x80, 0x3F, 0xC0, 0x3F, 0x00, 0x40, 0x40, 0x40,
    0x80, 0x40, 0xC0, 0x40, 0x00, 0x80, 0x00, 0xBF, 0x80, 0xBF, 0xC0, 0xBF,
    0x00, 0xC0, 0x40, 0xC0, 0x80, 0xC0, 0xC0, 0xC0};
alignas(64) static constexpr uint8_t LUTNZ[64] = {
    0x00, 0x10, 0x00, 0x3F, 0x80, 0x3F, 0xC0, 0x3F, 0x00, 0x40, 0x40, 0x40,
    0x80, 0x40, 0xC0, 0x40, 0x00, 0x90, 0x00, 0xBF, 0x80, 0xBF, 0xC0, 0xBF,
    0x00, 0xC0, 0x40, 0xC0, 0x80, 0xC0, 0xC0, 0xC0,
    0x00, 0x10, 0x00, 0x3F, 0x80, 0x3F, 0xC0, 0x3F, 0x00, 0x40, 0x40, 0x40,
    0x80, 0x40, 0xC0, 0x40, 0x00, 0x90, 0x00, 0xBF, 0x80, 0xBF, 0xC0, 0xBF,
    0x00, 0xC0, 0x40, 0xC0, 0x80, 0xC0, 0xC0, 0xC0};
alignas(32) static constexpr uint8_t DUPIDX[32] = {
    0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14,
    0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14};
alignas(64) static constexpr uint8_t SEQ01[64] = {
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
alignas(64) static constexpr uint16_t ACTPERM[32] = {
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};

static inline __m512i dequant_lut(__m128i packed, const uint8_t* LUT) {
  const __m256i b16 = _mm256_cvtepu8_epi16(packed);
  const __m256i m = _mm256_set1_epi16(0x0F);
  const __m256i lo16 = _mm256_and_si256(b16, m);
  const __m256i hi16 = _mm256_and_si256(_mm256_srli_epi16(b16, 4), m);
  const __m256i dup = _mm256_load_si256((const __m256i*)DUPIDX);
  const __m256i lo_dup = _mm256_shuffle_epi8(lo16, dup);
  const __m256i hi_dup = _mm256_shuffle_epi8(hi16, dup);
  const __m512i nib = _mm512_inserti64x4(_mm512_castsi256_si512(lo_dup), hi_dup, 1);
  const __m512i idx = _mm512_add_epi8(_mm512_slli_epi16(nib, 1),
                                      _mm512_load_si512((const void*)SEQ01));
  return _mm512_permutexvar_epi8(idx, _mm512_load_si512((const void*)LUT));
}

// legacy: per-group act permute + fp32 scale fmadd (original production path)
static void kern_legacy(const uint8_t* w, const float* s, const __m512bh* const* a, float* c,
                        int n, int kg, int rows) {
  for (int n_pos = 0; n_pos + 4 <= n; n_pos += 4) {
    const __m128i* wr[4];
    const float* sr[4];
    for (int j = 0; j < 4; j++) {
      wr[j] = (const __m128i*)(w + (size_t)(n_pos + j) * kg * 16);
      sr[j] = s + (size_t)(n_pos + j) * kg;
    }
    __m512 acc[4][4];
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) acc[i][j] = _mm512_setzero_ps();
    for (int g = 0; g < kg; g++) {
      if ((g & 3) == 0) {
        for (int j = 0; j < 4; j++) _mm_prefetch((const char*)(wr[j] + g + 16), _MM_HINT_T0);
      }
      const __m512bh d0 = (__m512bh)dequant_lut(wr[0][g], LUT0);
      const __m512bh d1 = (__m512bh)dequant_lut(wr[1][g], LUT0);
      const __m512bh d2 = (__m512bh)dequant_lut(wr[2][g], LUT0);
      const __m512bh d3 = (__m512bh)dequant_lut(wr[3][g], LUT0);
      const __m512 sv0 = _mm512_set1_ps(sr[0][g]);
      const __m512 sv1 = _mm512_set1_ps(sr[1][g]);
      const __m512 sv2 = _mm512_set1_ps(sr[2][g]);
      const __m512 sv3 = _mm512_set1_ps(sr[3][g]);
      for (int i = 0; i < rows; i++) {
        const __m512bh ap = (__m512bh)_mm512_permutexvar_epi16(
            _mm512_load_si512((const void*)ACTPERM), (__m512i)a[i][g]);
        acc[i][0] = _mm512_fmadd_ps(sv0, _mm512_dpbf16_ps(_mm512_setzero_ps(), ap, d0), acc[i][0]);
        acc[i][1] = _mm512_fmadd_ps(sv1, _mm512_dpbf16_ps(_mm512_setzero_ps(), ap, d1), acc[i][1]);
        acc[i][2] = _mm512_fmadd_ps(sv2, _mm512_dpbf16_ps(_mm512_setzero_ps(), ap, d2), acc[i][2]);
        acc[i][3] = _mm512_fmadd_ps(sv3, _mm512_dpbf16_ps(_mm512_setzero_ps(), ap, d3), acc[i][3]);
      }
    }
    for (int i = 0; i < rows; i++) {
      float* cr = c + (size_t)i * n;
      cr[n_pos + 0] = _mm512_reduce_add_ps(acc[i][0]);
      cr[n_pos + 1] = _mm512_reduce_add_ps(acc[i][1]);
      cr[n_pos + 2] = _mm512_reduce_add_ps(acc[i][2]);
      cr[n_pos + 3] = _mm512_reduce_add_ps(acc[i][3]);
    }
  }
}

// fold: pre-permuted acts, scale exponent folded into dequant
template <int MB, int NB, int PFD>
static void kern_fold(const uint8_t* w, const int16_t* se, const __m512bh* const* a, float* c,
                      int n, int kg, int rows) {
  for (int n_pos = 0; n_pos + NB <= n; n_pos += NB) {
    const __m128i* wr[NB];
    const int16_t* ser[NB];
    for (int j = 0; j < NB; j++) {
      wr[j] = (const __m128i*)(w + (size_t)(n_pos + j) * kg * 16);
      ser[j] = se + (size_t)(n_pos + j) * kg;
    }
    __m512 acc[MB][NB];
    for (int i = 0; i < MB; i++)
      for (int j = 0; j < NB; j++) acc[i][j] = _mm512_setzero_ps();
    for (int g = 0; g < kg; g++) {
      if constexpr (PFD > 0) {
        if ((g & 3) == 0) {
          for (int j = 0; j < NB; j++) {
            _mm_prefetch((const char*)(wr[j] + g + PFD), _MM_HINT_T0);
          }
        }
      }
      __m512bh d[NB];
      for (int j = 0; j < NB; j++) {
        const __m512i dv = dequant_lut(wr[j][g], LUTNZ);
        d[j] = (__m512bh)_mm512_add_epi16(dv, _mm512_set1_epi16(ser[j][g]));
      }
      for (int i = 0; i < MB; i++) {
        const __m512bh av = a[i][g];
        for (int j = 0; j < NB; j++) {
          acc[i][j] = _mm512_dpbf16_ps(acc[i][j], av, d[j]);
        }
      }
    }
    for (int i = 0; i < MB; i++) {
      float* cr = c + (size_t)i * n;
      for (int j = 0; j < NB; j++) cr[n_pos + j] = _mm512_reduce_add_ps(acc[i][j]);
    }
  }
}

int main(int argc, char** argv) {
  const int m = argc > 1 ? atoi(argv[1]) : 12;
  const int iters = argc > 2 ? atoi(argv[2]) : 3;
  const int nexp = argc > 3 ? atoi(argv[3]) : 100;  // experts in arena
  const int kg = argc > 4 ? atoi(argv[4]) : 128;    // groups (k = kg*32)
  const int n = kg == 128 ? 2048 : 4096;            // down shape when short-k

  std::vector<std::vector<uint8_t>> packed(n);
  std::vector<std::vector<float>> scale(n);
  for (int r = 0; r < n; r++) {
    packed[r].resize(kg * 16);
    scale[r].resize(kg);
    for (int i = 0; i < kg * 16; i++) packed[r][i] = rand();
    for (int g = 0; g < kg; g++) {
      uint32_t bits = (uint32_t)(-8 + (rand() & 3) + 127) << 23;
      memcpy(&scale[r][g], &bits, 4);
    }
  }
  std::vector<uint8_t> wbuf((size_t)nexp * n * kg * 16);
  std::vector<int16_t> sebuf((size_t)nexp * n * kg);
  std::vector<float> sbuf((size_t)nexp * n * kg);
  for (int e = 0; e < nexp; e++) {
    for (int r = 0; r < n; r++) {
      size_t off = (size_t)e * n + r;
      memcpy(wbuf.data() + off * kg * 16, packed[r % n].data(), kg * 16);
      memcpy(sbuf.data() + off * kg, scale[r % n].data(), kg * 4);
      for (int g = 0; g < kg; g++) {
        uint32_t b;
        memcpy(&b, &scale[r % n][g], 4);
        sebuf[off * kg + g] = (int16_t)(((b >> 16) & 0xFF80u) - 0x3F80u);
      }
    }
  }
  void* amem = aligned_alloc(64, (size_t)16 * kg * 64);
  __m512bh* abuf = (__m512bh*)amem;
  for (size_t i = 0; i < (size_t)16 * kg; i++) abuf[i] = (__m512bh)_mm512_set1_epi32(rand());
  std::vector<const __m512bh*> a(16);
  for (int i = 0; i < 16; i++) a[i] = abuf + (size_t)i * kg;
  std::vector<float> c((size_t)16 * n);

  auto run_pass = [&](auto&& kern_fn, const char* name) {
    auto t0 = std::chrono::high_resolution_clock::now();
    double macs = 0;
    for (int it = 0; it < iters; it++) {
      for (int e = 0; e < nexp; e++) {
        const uint8_t* w = wbuf.data() + (size_t)e * n * kg * 16;
        const int16_t* se = sebuf.data() + (size_t)e * n * kg;
        const float* s = sbuf.data() + (size_t)e * n * kg;
        for (int mp = 0; mp < m; mp += 4) {
          const int rows_t = std::min(4, m - mp);
          macs += (double)rows_t * n * 4096;
          kern_fn(w, s, se, a.data() + mp, c.data(), n, kg, rows_t);
        }
      }
    }
    double dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    printf("%-24s %9.2f ms  %8.1f GMAC/s  (%.1f GB weight/s)\n", name, dt * 1e3, macs / dt / 1e9,
           (double)nexp * n * kg * 16 * iters / dt / 1e9);
  };

  run_pass([](const uint8_t* w, const float* s, const int16_t* se, const __m512bh* const* a,
              float* c, int n_, int kg_, int rows) { kern_legacy(w, s, a, c, n_, kg_, rows); },
           "legacy 4x4 pf16");
  run_pass([](const uint8_t* w, const float* s, const int16_t* se, const __m512bh* const* a,
              float* c, int n_, int kg_, int rows) { kern_fold<4, 4, 64>(w, se, a, c, n_, kg_, rows); },
           "fold 4x4 pf64");
  run_pass([](const uint8_t* w, const float* s, const int16_t* se, const __m512bh* const* a,
              float* c, int n_, int kg_, int rows) { kern_fold<8, 2, 64>(w, se, a, c, n_, kg_, rows); },
           "fold 8x2 pf64");
  run_pass([](const uint8_t* w, const float* s, const int16_t* se, const __m512bh* const* a,
              float* c, int n_, int kg_, int rows) { kern_fold<2, 8, 64>(w, se, a, c, n_, kg_, rows); },
           "fold 2x8 pf64");
  run_pass([](const uint8_t* w, const float* s, const int16_t* se, const __m512bh* const* a,
              float* c, int n_, int kg_, int rows) { kern_fold<4, 8, 64>(w, se, a, c, n_, kg_, rows); },
           "fold 4x8x2 pf64");
  run_pass([](const uint8_t* w, const float* s, const int16_t* se, const __m512bh* const* a,
              float* c, int n_, int kg_, int rows) { kern_fold<4, 4, 64>(w, se, a, c, n_, kg_, rows); },
           "fold 4x4 pf64");
  volatile float sink = 0;
  for (size_t i = 0; i < c.size(); i += 512) sink += c[i];
  (void)sink;
  free(amem);
  return 0;
}
