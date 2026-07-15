#include "gdn.h"

#include <cstdio>
#include <cstdlib>

namespace bt {

namespace {

__device__ __forceinline__ float softplus_f(float x) {
    // log1p(exp(x)) with overflow guard, matching ggml_softplus
    return x > 20.f ? x : log1pf(expf(x));
}

// One block per value head. Threads: 8 warps; each warp owns columns
// {warp, warp+8, ...}. State column c is contiguous (transposed layout), so
// lanes stream it coalesced; k and q live in shared memory.
template <int S>
__global__ void gdn_decode_kernel(const __half* __restrict__ q,
                                  const __half* __restrict__ k,
                                  const __half* __restrict__ v,
                                  const __half* __restrict__ alpha_raw,
                                  const __half* __restrict__ beta_raw,
                                  const float* __restrict__ A_log,
                                  const float* __restrict__ dt_bias,
                                  float* __restrict__ state,
                                  __half* __restrict__ out, int H_k,
                                  float scale) {
    const int h = blockIdx.x;
    const int hk = h % H_k;

    __shared__ float s_k[S], s_q[S];
    for (int i = threadIdx.x; i < S; i += blockDim.x) {
        s_k[i] = __half2float(k[(size_t)hk * S + i]);
        s_q[i] = __half2float(q[(size_t)hk * S + i]);
    }
    __syncthreads();

    const float g = expf(-expf(A_log[h]) *
                         softplus_f(__half2float(alpha_raw[h]) + dt_bias[h]));
    const float beta = 1.f / (1.f + expf(-__half2float(beta_raw[h])));

    float* S_h = state + (size_t)h * S * S;
    const __half* v_h = v + (size_t)h * S;
    __half* out_h = out + (size_t)h * S;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    constexpr int kRowsPerLane = S / 32;

    for (int c = warp; c < S; c += blockDim.x / 32) {
        float* col = S_h + (size_t)c * S;  // transposed: column c contiguous

        float s_reg[kRowsPerLane];
        float kv = 0.f;
#pragma unroll
        for (int r = 0; r < kRowsPerLane; ++r) {
            const int i = r * 32 + lane;
            s_reg[r] = col[i];
            kv += s_reg[r] * s_k[i];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) kv += __shfl_xor_sync(0xFFFFFFFFu, kv, off);

        const float delta = (__half2float(v_h[c]) - g * kv) * beta;

        float attn = 0.f;
#pragma unroll
        for (int r = 0; r < kRowsPerLane; ++r) {
            const int i = r * 32 + lane;
            s_reg[r] = g * s_reg[r] + s_k[i] * delta;
            col[i] = s_reg[r];
            attn += s_reg[r] * s_q[i];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) attn += __shfl_xor_sync(0xFFFFFFFFu, attn, off);

        if (lane == 0) out_h[c] = __float2half(attn * scale);
    }
}

}  // namespace

void gdn_decode_launch(const __half* q, const __half* k, const __half* v,
                       const __half* alpha_raw, const __half* beta_raw,
                       const float* A_log, const float* dt_bias, float* state,
                       __half* out, int H_v, int H_k, int S, float scale,
                       cudaStream_t stream) {
    const dim3 grid(H_v);
    switch (S) {
        case 64:
            gdn_decode_kernel<64><<<grid, 256, 0, stream>>>(
                q, k, v, alpha_raw, beta_raw, A_log, dt_bias, state, out, H_k, scale);
            break;
        case 128:
            gdn_decode_kernel<128><<<grid, 256, 0, stream>>>(
                q, k, v, alpha_raw, beta_raw, A_log, dt_bias, state, out, H_k, scale);
            break;
        case 256:
            gdn_decode_kernel<256><<<grid, 256, 0, stream>>>(
                q, k, v, alpha_raw, beta_raw, A_log, dt_bias, state, out, H_k, scale);
            break;
        default:
            std::fprintf(stderr, "gdn_decode: unsupported head dim %d\n", S);
            std::abort();
    }
}

}  // namespace bt
