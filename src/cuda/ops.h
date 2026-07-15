// bonsai-turbo — decode-step elementwise/norm ops (batch 1).
// Correctness-first v1 kernels; each disappears into the fused step later.
#pragma once

#include <cstdint>

#include <cuda_fp16.h>

namespace bt {

// y[i] = x[i] * w[i] / rms(x), one vector of length n
void rmsnorm_launch(const __half* x, const __half* w, __half* y, int n,
                    float eps, cudaStream_t stream);

// per-head RMS norm: h vectors of length d, same weight w[d] for every head
void rmsnorm_heads_launch(const __half* x, const __half* w, __half* y, int h,
                          int d, float eps, cudaStream_t stream);

// per-head L2 norm: x / sqrt(sum(x^2) + eps), h vectors of length d
void l2norm_heads_launch(const __half* x, __half* y, int h, int d, float eps,
                         cudaStream_t stream);

// y[i] += x[i]
void add_inplace_launch(__half* y, const __half* x, int n, cudaStream_t stream);

// y[i] = silu(a[i]) * b[i]
void silu_mul_launch(const __half* a, const __half* b, __half* y, int n,
                     cudaStream_t stream);

// y[i] = x[i] * sigmoid(g[i])
void sigmoid_mul_launch(const __half* x, const __half* g, __half* y, int n,
                        cudaStream_t stream);

// f32 -> f16 cast (GEMV epilogue glue in v1)
void f32_to_f16_launch(const float* x, __half* y, int n, cudaStream_t stream);

// NeoX-style RoPE applied in place to h heads of dim d over the first `rot`
// dims (pairs (i, i+rot/2)), position pos. Matches ggml_rope_multi for pure
// text (all M-RoPE sections see the same position; verified against the fork
// at parity-gate time).
void rope_neox_launch(__half* x, int h, int d, int rot, int pos,
                      float freq_base, cudaStream_t stream);

// out[i] = table[token*n + i] (embedding row fetch, f16 table)
void embed_lookup_launch(const __half* table, int token, int n, __half* out,
                         cudaStream_t stream);

// single-vector argmax (greedy sampling); out[0] = index
void argmax_launch(const float* x, int n, int32_t* out, cudaStream_t stream);

}  // namespace bt
