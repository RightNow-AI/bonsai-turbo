// bonsai-turbo — batch-1 GEMV over re-tiled Q2_0/Q1_0 weights.
#pragma once

#include <cstdint>

#include <cuda_fp16.h>

namespace bt {

// x[K] fp16 -> per-128-group int8 activations + scale + group sum.
// Grid-stride safe; launch with quant_acts_launch.
void quant_acts_launch(const __half* x, int K, int8_t* a8, float* a_scale,
                       int32_t* a_gsum, cudaStream_t stream);

// y[M] = W[M,K] @ x[K]; W given as permuted codes + fp16 group scales
// (see retile.h for the layout contract). nbits selects Q2 (2) or Q1 (1).
void gemv_launch(int nbits, const uint8_t* codes, const __half* w_scale,
                 const int8_t* a8, const float* a_scale, const int32_t* a_gsum,
                 float* y, int M, int K, cudaStream_t stream);

}  // namespace bt
