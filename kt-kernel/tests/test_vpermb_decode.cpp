// 自包含单测: 对比 vpermb 版与 PSHUFB 版 FP4->BF16 解码 (位级一致)。
// 实现从 operators/amx/fp4-moe.hpp 原样复制; 修改内核后需同步此处。
#include <cstdio>
#include <cstdint>
#include <immintrin.h>

// ---- LUTs (fp4-moe.hpp) ----
alignas(16) static constexpr uint8_t fp4_bf16_lo[16] = {
    0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0,
    0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0};
alignas(16) static constexpr uint8_t fp4_bf16_hi[16] = {
    0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40,
    0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0};

alignas(64) static constexpr uint8_t fp4_bf16_vpermb_lut[64] = {
    0x00, 0x00, 0x00, 0x3F, 0x80, 0x3F, 0xC0, 0x3F, 0x00, 0x40, 0x40, 0x40,
    0x80, 0x40, 0xC0, 0x40, 0x00, 0x80, 0x00, 0xBF, 0x80, 0xBF, 0xC0, 0xBF,
    0x00, 0xC0, 0x40, 0xC0, 0x80, 0xC0, 0xC0, 0xC0,
    0x00, 0x00, 0x00, 0x3F, 0x80, 0x3F, 0xC0, 0x3F, 0x00, 0x40, 0x40, 0x40,
    0x80, 0x40, 0xC0, 0x40, 0x00, 0x80, 0x00, 0xBF, 0x80, 0xBF, 0xC0, 0xBF,
    0x00, 0xC0, 0x40, 0xC0, 0x80, 0xC0, 0xC0, 0xC0};
alignas(32) static constexpr uint8_t fp4_dup_lo_idx[32] = {
    0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14,
    0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14};

alignas(64) static constexpr uint8_t fp4_seq01[64] = {
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};

// ---- vpermb 版 (fp4-moe.hpp, __AVX512VBMI__ && __AVX512BF16__) ----
static inline __m512i vpermb_decode(__m128i packed) {
  const __m256i b16 = _mm256_cvtepu8_epi16(packed);
  const __m256i m = _mm256_set1_epi16(0x0F);
  const __m256i lo16 = _mm256_and_si256(b16, m);
  const __m256i hi16 = _mm256_and_si256(_mm256_srli_epi16(b16, 4), m);
  const __m256i dup_idx = _mm256_load_si256((const __m256i*)fp4_dup_lo_idx);
  const __m256i lo_dup = _mm256_shuffle_epi8(lo16, dup_idx);
  const __m256i hi_dup = _mm256_shuffle_epi8(hi16, dup_idx);
  const __m512i nib = _mm512_inserti64x4(_mm512_castsi256_si512(lo_dup), hi_dup, 1);
  const __m512i idx = _mm512_add_epi8(_mm512_slli_epi16(nib, 1), _mm512_load_si512((const void*)fp4_seq01));
  return _mm512_permutexvar_epi8(idx, _mm512_load_si512((const void*)fp4_bf16_vpermb_lut));
}

// ---- PSHUFB 原版 (fp4-moe.hpp) ----
static inline __m512i pshufb_decode(__m128i packed) {
  __m128i lo_mask = _mm_set1_epi8(0x0F);
  __m128i lo = _mm_and_si128(packed, lo_mask);
  __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), lo_mask);
  __m128i lut_lo = _mm_load_si128((__m128i*)fp4_bf16_lo);
  __m128i lut_hi = _mm_load_si128((__m128i*)fp4_bf16_hi);
  __m128i l_lo = _mm_shuffle_epi8(lut_lo, lo);
  __m128i l_hi = _mm_shuffle_epi8(lut_hi, lo);
  __m128i lo_bf16_0 = _mm_unpacklo_epi8(l_lo, l_hi);
  __m128i lo_bf16_1 = _mm_unpackhi_epi8(l_lo, l_hi);
  __m128i h_lo = _mm_shuffle_epi8(lut_lo, hi);
  __m128i h_hi = _mm_shuffle_epi8(lut_hi, hi);
  __m128i hi_bf16_0 = _mm_unpacklo_epi8(h_lo, h_hi);
  __m128i hi_bf16_1 = _mm_unpackhi_epi8(h_lo, h_hi);
  __m128i p0 = _mm_unpacklo_epi16(lo_bf16_0, hi_bf16_0);
  __m128i p1 = _mm_unpackhi_epi16(lo_bf16_0, hi_bf16_0);
  __m128i p2 = _mm_unpacklo_epi16(lo_bf16_1, hi_bf16_1);
  __m128i p3 = _mm_unpackhi_epi16(lo_bf16_1, hi_bf16_1);
  __m256i q0 = _mm256_inserti128_si256(_mm256_castsi128_si256(p0), p1, 1);
  __m256i q1 = _mm256_inserti128_si256(_mm256_castsi128_si256(p2), p3, 1);
  return _mm512_inserti64x4(_mm512_castsi256_si512(q0), q1, 1);
}

// 参考: bf16 word 列表, 交错顺序 [lo0,hi0,lo1,hi1,...]
static void ref_decode(const uint8_t* in, uint16_t* out) {
  for (int j = 0; j < 16; j++) {
    uint8_t lo = in[j] & 0x0F, hi = (in[j] >> 4) & 0x0F;
    out[2 * j] = (uint16_t)(fp4_bf16_hi[lo] << 8 | fp4_bf16_lo[lo]);
    out[2 * j + 1] = (uint16_t)(fp4_bf16_hi[hi] << 8 | fp4_bf16_lo[hi]);
  }
}

int main() {
  int fails = 0, tested = 0;
  uint8_t in[16];
  uint16_t ref[32], got_p[32], got_v[32];
  for (uint32_t v = 0; v < (1u << 20); v += 123457u) {
    for (int j = 0; j < 16; j++) in[j] = (uint8_t)((v * 2654435761u) >> ((j % 4) * 8) ^ (j * 97));
    ref_decode(in, ref);
    _mm512_storeu_si512((__m512i*)got_p, pshufb_decode(_mm_loadu_si128((__m128i*)in)));
    _mm512_storeu_si512((__m512i*)got_v, vpermb_decode(_mm_loadu_si128((__m128i*)in)));
    tested++;
    for (int w = 0; w < 32; w++) {
      if (got_p[w] != ref[w]) { if (fails++ < 5) std::printf("pshufb != ref @w%d\n", w); }
      // vpermb 输出顺序: [lo0..lo15 | hi0..hi15]
      uint16_t exp = (w < 16) ? ref[2 * w] : ref[2 * (w - 16) + 1];
      if (got_v[w] != exp) {
        if (fails++ < 5)
          std::printf("vpermb != ref @w%d nib=%d got=%04x exp=%04x\n",
                      w, (w < 16) ? (in[w / 2] & 0xF) : ((in[(w - 16) / 2] >> 4) & 0xF),
                      got_v[w], exp);
      }
    }
  }
  // 全 nibble 对枚举
  for (int lo = 0; lo < 16; lo++)
    for (int hi = 0; hi < 16; hi++) {
      for (int j = 0; j < 16; j++) in[j] = (uint8_t)((lo | (hi << 4)) + j * 0);
      in[0] = (uint8_t)(lo | (hi << 4));
      for (int j = 1; j < 16; j++) in[j] = in[0];
      ref_decode(in, ref);
      _mm512_storeu_si512((__m512i*)got_v, vpermb_decode(_mm_loadu_si128((__m128i*)in)));
      tested++;
      for (int w = 0; w < 32; w++) {
        uint16_t exp = (w < 16) ? ref[2 * w] : ref[2 * (w - 16) + 1];
        if (got_v[w] != exp) fails++;
      }
    }
  if (fails == 0) {
    std::printf("PASS: %d vectors, vpermb & pshufb both match reference\n", tested);
    return 0;
  }
  std::printf("FAIL: %d mismatches over %d vectors\n", fails, tested);
  return 1;
}
