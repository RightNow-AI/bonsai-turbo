// bonsai-turbo — decode-step elementwise/norm ops (batch 1).
// Correctness-first v1 kernels; each disappears into the fused step later.
#pragma once

#include <cstdint>

#include <cuda_fp16.h>

namespace bt {

// y[i] = x[i] * w[i] / rms(x), one vector of length n
void rmsnorm_launch(const __half* x, const __half* w, __half* y, int n,
                    float eps, cudaStream_t stream);

// fp32-residual-stream variants: hidden state x stays fp32 end to end
void rmsnorm_f32_launch(const float* x, const __half* w, __half* y, int n,
                        float eps, cudaStream_t stream);
// x[i] += d[i] (residual add straight from a GEMV's fp32 output)
void add_f32_launch(float* x, const float* d, int n, cudaStream_t stream);

// fused rmsnorm + int8 activation quantization (one launch instead of two):
// y (f16 normalized copy) may be null when only the quantized form is needed
void rmsnorm_quant_f32_launch(const float* x, const __half* w, int n, __half* y,
                              int8_t* a8, float* a_scale, int32_t* a_gsum64,
                              float eps, cudaStream_t stream);

// fused elementwise-gate + int8 quantization: op 0: silu(a)*b, op 1: a*sigmoid(b)
// y (f16 result copy) may be null
void gate_mul_quant_launch(int op, const __half* a, const __half* b, int n,
                           __half* y, int8_t* a8, float* a_scale,
                           int32_t* a_gsum64, cudaStream_t stream);

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

// ---- CUDA-graph-capturable variants: control state lives on device ----

// like rope_neox_launch but position read from *d_pos
void rope_neox_dev_launch(__half* x, int h, int d, int rot, const int32_t* d_pos,
                          float freq_base, cudaStream_t stream);

// like embed_lookup_launch but token read from *d_tok
void embed_lookup_dev_launch(const __half* table, const int32_t* d_tok, int n,
                             __half* out, cudaStream_t stream);

// append k,v ([H_kv*D] each, f16) into caches at position *d_pos
void kv_append_dev_launch(const __half* k, const __half* v, __half* k_cache,
                          __half* v_cache, int row_elems, const int32_t* d_pos,
                          cudaStream_t stream);

// ring[*d_step] = *d_tok; ++*d_step; ++*d_pos  (end-of-step bump)
void step_bump_launch(int32_t* d_pos, int32_t* d_step, int32_t* ring, int cap,
                      const int32_t* d_tok, cudaStream_t stream);

// single-vector argmax (greedy sampling); out[0] = index
void argmax_launch(const float* x, int n, int32_t* out, cudaStream_t stream);

// dst[h*D + d] = src[h*stride + offset + d] for h in [0,H): pulls the q or
// gate half out of an interleaved [q|gate] per-head projection.
void gather_heads_launch(const __half* src, __half* dst, int H, int D,
                         int stride, int offset, cudaStream_t stream);

// y[M] = W[M,K] @ x[K] for f16 weights (fallback for non-quantized matrices)
void gemv_f16_launch(const __half* W, const __half* x, float* y, int M, int K,
                     cudaStream_t stream);

// causal conv1d decode step over C channels, kernel width k (matches
// ggml_ssm_conv + trailing SiLU): hist = [state[c][0..k-2], x[c]];
// y[c] = silu(sum_i hist[i] * w[c*k+i]); state shifts left by one.
// w layout: [C][k] contiguous per channel (= GGUF ssm_conv1d, ne[0]=k), f16.
void conv1d_step_launch(const __half* x, const __half* w, float* conv_state,
                        __half* y, int C, int k, cudaStream_t stream);

}  // namespace bt
