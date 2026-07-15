// bonsai-turbo — fused gated delta-net decode step (batch 1, one token).
//
// Recurrence (matches the vendor fork's GGML_OP_GATED_DELTA_NET, scalar-gate
// path, at the pinned SHA):
//   g       = exp(-exp(A_log[h]) * softplus(alpha[h] + dt_bias[h]))
//   beta    = sigmoid(beta_raw[h])
//   kv[c]   = sum_i S[i][c] * k[i]
//   delta[c]= (v[c] - g * kv[c]) * beta
//   S[i][c] = g * S[i][c] + k[i] * delta[c]
//   out[c]  = scale * sum_i S[i][c] * q[i]
// State is stored transposed per head (column c contiguous), fp32.
// The gate/beta scalar math is fused in (saves 4 elementwise launches).
#pragma once

#include <cuda_fp16.h>

namespace bt {

// q,k: [H_k][S] f16 (head h uses h % H_k); v: [H_v][S] f16
// alpha_raw, beta_raw: [H_v] f16 (straight from the GEMV outputs)
// A_log, dt_bias: [H_v] f32 model constants
// state: [H_v][S][S] f32, transposed layout; out: [H_v][S] f16
void gdn_decode_launch(const __half* q, const __half* k, const __half* v,
                       const __half* alpha_raw, const __half* beta_raw,
                       const float* A_log, const float* dt_bias, float* state,
                       __half* out, int H_v, int H_k, int S, float scale,
                       cudaStream_t stream);

}  // namespace bt
