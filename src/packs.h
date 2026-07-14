// bonsai-turbo — Bonsai 27B weight pack formats.
//
// These mirror the vendor fork's ggml definitions bit-for-bit
// (ggml-common.h + ggml-quants.c at the pinned SHA in scripts/build_vendor_fork.sh).
// The reference dequant functions below are the loader's correctness contract:
// tests compare our re-tiled GPU layouts against these, element by element.
#pragma once

#include <cstdint>
#include <cstddef>

namespace bt {

// IEEE 754 binary16 stored as raw bits.
using half_bits = uint16_t;

float half_to_float(half_bits h);

// ---------------------------------------------------------------------------
// Q1_0_g128 ("1-bit"): 128 weights per block, one FP16 scale.
//   bit j of qs[j/8] (LSB-first) selects  w = bit ? +d : -d      (1.125 bpw)
// ---------------------------------------------------------------------------
inline constexpr int kQ1GroupSize = 128;

struct BlockQ1_0 {
    half_bits d;
    uint8_t   qs[kQ1GroupSize / 8];
};
static_assert(sizeof(BlockQ1_0) == 18, "Q1_0 block must be 18 bytes");

// ---------------------------------------------------------------------------
// Q2_0_g128 ("ternary"): 128 weights per block, one FP16 scale.
//   code j = (qs[j/4] >> (2*(j%4))) & 3  (LSB-first);  w = (code - 1) * d
//   codes: 0 -> -d, 1 -> 0, 2 -> +d, 3 -> +2d.
//   The vendor quantizer never emits code 3 (verified in their CUDA dot:
//   "code 3 is unreachable from the reference quantizer"); the loader asserts
//   this on real weights so kernels may specialize for {-1, 0, +1}. (2.125 bpw)
// ---------------------------------------------------------------------------
inline constexpr int kQ2GroupSize = 128;

struct BlockQ2_0 {
    half_bits d;
    uint8_t   qs[kQ2GroupSize / 4];
};
static_assert(sizeof(BlockQ2_0) == 34, "Q2_0 block must be 34 bytes");

// Reference dequant, transcribed from the fork's dequantize_row_q1_0 /
// dequantize_row_q2_0. `k` is the number of weights (multiple of group size).
void dequant_q1_0_ref(const BlockQ1_0* blocks, float* out, int64_t k);
void dequant_q2_0_ref(const BlockQ2_0* blocks, float* out, int64_t k);

// Scans real weights for the nominally-unreachable Q2_0 code 3.
// Returns the number of code-3 elements found (0 on healthy vendor packs).
int64_t count_q2_0_code3(const BlockQ2_0* blocks, int64_t k);

}  // namespace bt
