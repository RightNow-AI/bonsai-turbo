// bonsai-turbo — batch-1 GEMV over re-tiled Q2_0/Q1_0 weights.
#pragma once

#include <cstdint>

#include <cuda_fp16.h>

namespace bt {

// x[K] fp16 -> int8 activations, one fp32 scale per 128-group, and int32
// HALF-group sums (K/64 entries) for the deferred sign correction.
void quant_acts_launch(const __half* x, int K, int8_t* a8, float* a_scale,
                       int32_t* a_gsum64, cudaStream_t stream);

// y[M] = W[M,K] @ x[K]; W given as permuted codes + fp16 group scales
// (see retile.h for the layout contract). nbits selects Q2 (2) or Q1 (1).
// Needs K + K/32 + K/16 bytes of dynamic shared memory per block.
void gemv_launch(int nbits, const uint8_t* codes, const __half* w_scale,
                 const int8_t* a8, const float* a_scale, const int32_t* a_gsum64,
                 float* y, int M, int K, cudaStream_t stream);

// dequantize one row of a re-tiled tensor to f16 (embedding lookup)
void dequant_row_launch(int nbits, const uint8_t* codes, const __half* w_scale,
                        int row, int K, __half* out, cudaStream_t stream);

}  // namespace bt
