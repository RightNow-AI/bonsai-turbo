// bonsai-turbo — weight re-tiling for coalesced GEMV.
//
// GGUF stores Q2_0/Q1_0 rows as a stream of (fp16 scale + code bytes) blocks:
// 34 B per 128 weights (Q2) / 18 B (Q1). Two problems for a GPU GEMV:
//   1. 34/18-byte blocks are never 16-byte aligned, so 128-bit loads straddle.
//   2. dp4a wants each 32-bit code word to expand into byte lanes that match
//      *sequential* activation bytes; the natural bit order does not.
//
// The re-tiler fixes both once at load time:
//   - codes plane:  [row][K/4 bytes] (Q2) or [row][K/8 bytes] (Q1), 16B-aligned
//     rows, with codes permuted inside each 32-bit word (see permute map below)
//   - scales plane: [row][K/128] fp16
//
// Decode contract (what the CUDA kernel does with a permuted word w):
//   Q2: dp4a #i (i=0..3) uses bytes ((w >> 2i) & 0x03030303) and must see
//       original codes {4i, 4i+1, 4i+2, 4i+3}. So original code j = 4i+b goes
//       to bits [8b+2i, 8b+2i+1] of w.
//   Q1: dp4a #i (i=0..7) uses bytes ((w >> i) & 0x01010101) and must see
//       original codes {4i .. 4i+3}. So original code j = 4i+b goes to bit
//       [8b+i] of w.
// decode_permuted_* below implement exactly the kernel's read order on the
// CPU; tests pit them against the reference dequant to prove the permutation.
#pragma once

#include <cstdint>
#include <vector>

#include "packs.h"

namespace bt {

struct RetiledTensor {
    int64_t rows = 0;
    int64_t cols = 0;                 // K; multiple of 128
    std::vector<uint8_t>   codes;     // rows * cols/4 (Q2) or cols/8 (Q1)
    std::vector<half_bits> scales;    // rows * cols/128
};

// src points at `rows` consecutive GGUF rows, each cols/128 blocks.
RetiledTensor retile_q2_0(const BlockQ2_0* src, int64_t rows, int64_t cols);
RetiledTensor retile_q1_0(const BlockQ1_0* src, int64_t rows, int64_t cols);

// CPU mirror of the GPU decode order; out gets rows*cols floats.
void decode_permuted_q2_0(const RetiledTensor& t, float* out);
void decode_permuted_q1_0(const RetiledTensor& t, float* out);

}  // namespace bt
