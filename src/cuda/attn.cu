#include "attn.h"

#include <cstdio>
#include <cstdlib>

namespace bt {

namespace {

// One block (8 warps) per query head. Warps stride context positions keeping
// per-warp online-softmax state; a final block pass merges the 8 partials.
// POS_PTR: context length = *d_pos + 1 read on device (CUDA-graph capturable)
template <int D, bool POS_PTR = false>
__global__ void attn_decode_kernel(const __half* __restrict__ q,
                                   const __half* __restrict__ k_cache,
                                   const __half* __restrict__ v_cache,
                                   __half* __restrict__ out, int H, int H_kv,
                                   int ctx_len, const int32_t* __restrict__ d_pos,
                                   float scale) {
    constexpr int kWarps = 8;
    constexpr int kVecPerLane = D / 32;

    if (POS_PTR) ctx_len = *d_pos + 1;
    const int h = blockIdx.x;
    const int hkv = h / (H / H_kv);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const size_t kv_row = (size_t)H_kv * D;

    __shared__ float s_q[D];
    __shared__ float s_m[kWarps], s_l[kWarps];
    __shared__ float s_acc[kWarps][D];

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        s_q[i] = __half2float(q[(size_t)h * D + i]);
    }
    __syncthreads();

    float m = -INFINITY, l = 0.f;
    float acc[kVecPerLane];
#pragma unroll
    for (int r = 0; r < kVecPerLane; ++r) acc[r] = 0.f;

    for (int pos = warp; pos < ctx_len; pos += kWarps) {
        const __half* k_pos = k_cache + (size_t)pos * kv_row + (size_t)hkv * D;
        float dot = 0.f;
#pragma unroll
        for (int r = 0; r < kVecPerLane; ++r) {
            const int d = r * 32 + lane;
            dot += s_q[d] * __half2float(k_pos[d]);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
        const float score = dot * scale;

        const float m_new = fmaxf(m, score);
        const float rescale = expf(m - m_new);
        const float p = expf(score - m_new);
        const __half* v_pos = v_cache + (size_t)pos * kv_row + (size_t)hkv * D;
#pragma unroll
        for (int r = 0; r < kVecPerLane; ++r) {
            const int d = r * 32 + lane;
            acc[r] = acc[r] * rescale + p * __half2float(v_pos[d]);
        }
        l = l * rescale + p;
        m = m_new;
    }

    if (lane == 0) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
#pragma unroll
    for (int r = 0; r < kVecPerLane; ++r) s_acc[warp][r * 32 + lane] = acc[r];
    __syncthreads();

    if (warp == 0) {
        float m_star = -INFINITY;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) m_star = fmaxf(m_star, s_m[w]);
        float l_star = 0.f;
        float o[kVecPerLane];
#pragma unroll
        for (int r = 0; r < kVecPerLane; ++r) o[r] = 0.f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            // empty warps (ctx < kWarps) have m = -inf -> factor 0
            const float factor = s_m[w] == -INFINITY ? 0.f : expf(s_m[w] - m_star);
            l_star += factor * s_l[w];
#pragma unroll
            for (int r = 0; r < kVecPerLane; ++r) {
                o[r] += factor * s_acc[w][r * 32 + lane];
            }
        }
        const float inv_l = 1.f / l_star;
#pragma unroll
        for (int r = 0; r < kVecPerLane; ++r) {
            out[(size_t)h * D + r * 32 + lane] = __float2half(o[r] * inv_l);
        }
    }
}

}  // namespace

void attn_decode_launch(const __half* q, const __half* k_cache,
                        const __half* v_cache, __half* out, int H, int H_kv,
                        int D, int ctx_len, float scale, cudaStream_t stream) {
    switch (D) {
        case 64:
            attn_decode_kernel<64><<<H, 256, 0, stream>>>(
                q, k_cache, v_cache, out, H, H_kv, ctx_len, nullptr, scale);
            break;
        case 128:
            attn_decode_kernel<128><<<H, 256, 0, stream>>>(
                q, k_cache, v_cache, out, H, H_kv, ctx_len, nullptr, scale);
            break;
        case 256:
            attn_decode_kernel<256><<<H, 256, 0, stream>>>(
                q, k_cache, v_cache, out, H, H_kv, ctx_len, nullptr, scale);
            break;
        default:
            std::fprintf(stderr, "attn_decode: unsupported head dim %d\n", D);
            std::abort();
    }
}

void attn_decode_dev_launch(const __half* q, const __half* k_cache,
                            const __half* v_cache, __half* out, int H, int H_kv,
                            int D, const int32_t* d_pos, float scale,
                            cudaStream_t stream) {
    switch (D) {
        case 64:
            attn_decode_kernel<64, true><<<H, 256, 0, stream>>>(
                q, k_cache, v_cache, out, H, H_kv, 0, d_pos, scale);
            break;
        case 128:
            attn_decode_kernel<128, true><<<H, 256, 0, stream>>>(
                q, k_cache, v_cache, out, H, H_kv, 0, d_pos, scale);
            break;
        case 256:
            attn_decode_kernel<256, true><<<H, 256, 0, stream>>>(
                q, k_cache, v_cache, out, H, H_kv, 0, d_pos, scale);
            break;
        default:
            std::fprintf(stderr, "attn_decode: unsupported head dim %d\n", D);
            std::abort();
    }
}

}  // namespace bt
