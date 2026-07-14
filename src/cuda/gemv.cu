#include "gemv.h"

namespace bt {

namespace {

// u8 codes x s8 activations dot product; no mixed-sign __dp4a overload exists.
__device__ __forceinline__ int dp4a_u8s8(uint32_t codes, uint32_t acts, int acc) {
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(acc) : "r"(codes), "r"(acts), "r"(acc));
    return acc;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    }
    return v;
}

// One warp per 128-group: absmax -> int8 quant -> group sum.
__global__ void quant_acts_kernel(const __half* __restrict__ x, int K,
                                  int8_t* __restrict__ a8,
                                  float* __restrict__ a_scale,
                                  int32_t* __restrict__ a_gsum) {
    const int group = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (group >= K / 128) return;
    const int lane = threadIdx.x & 31;

    float vals[4];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vals[i] = __half2float(x[group * 128 + lane * 4 + i]);
        amax = fmaxf(amax, fabsf(vals[i]));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
    }
    const float s = fmaxf(amax, 1e-8f) / 127.f;
    const float inv_s = 1.f / s;

    int gsum = 0;
    char4 q;
    int8_t* qp = reinterpret_cast<int8_t*>(&q);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int v = __float2int_rn(vals[i] * inv_s);
        v = max(-127, min(127, v));
        qp[i] = (int8_t)v;
        gsum += v;
    }
    reinterpret_cast<char4*>(a8)[group * 32 + lane] = q;

    gsum += __shfl_down_sync(0xFFFFFFFFu, gsum, 16);
    gsum += __shfl_down_sync(0xFFFFFFFFu, gsum, 8);
    gsum += __shfl_down_sync(0xFFFFFFFFu, gsum, 4);
    gsum += __shfl_down_sync(0xFFFFFFFFu, gsum, 2);
    gsum += __shfl_down_sync(0xFFFFFFFFu, gsum, 1);
    if (lane == 0) {
        a_scale[group] = s;
        a_gsum[group] = gsum;
    }
}

// One warp per output row; lanes stride the row's 128-weight groups.
// NBITS=2: codes {0,1,2,(3)}, w = (q-1)*d  -> dot' = dot - gsum
// NBITS=1: codes {0,1},       w = (2q-1)*d -> dot' = 2*dot - gsum
template <int NBITS>
__global__ void gemv_kernel(const uint8_t* __restrict__ codes,
                            const __half* __restrict__ w_scale,
                            const int8_t* __restrict__ a8,
                            const float* __restrict__ a_scale,
                            const int32_t* __restrict__ a_gsum,
                            float* __restrict__ y, int M, int K) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= M) return;
    const int lane = threadIdx.x & 31;

    const int groups = K >> 7;
    const int code_row_bytes = NBITS == 2 ? (K >> 2) : (K >> 3);
    const uint8_t* row_codes = codes + (size_t)row * code_row_bytes;
    const __half* row_scale = w_scale + (size_t)row * groups;

    float acc = 0.f;
    for (int g = lane; g < groups; g += 32) {
        const uint4* aw = reinterpret_cast<const uint4*>(a8 + g * 128);
        int dot = 0;

        if (NBITS == 2) {
            const uint4* cw = reinterpret_cast<const uint4*>(row_codes + g * 32);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint4 c = cw[h];
                const uint32_t cws[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
                for (int wnum = 0; wnum < 4; ++wnum) {
                    const uint4 av = aw[h * 4 + wnum];
                    const uint32_t w = cws[wnum];
                    dot = dp4a_u8s8(w & 0x03030303u, av.x, dot);
                    dot = dp4a_u8s8((w >> 2) & 0x03030303u, av.y, dot);
                    dot = dp4a_u8s8((w >> 4) & 0x03030303u, av.z, dot);
                    dot = dp4a_u8s8((w >> 6) & 0x03030303u, av.w, dot);
                }
            }
        } else {
            const uint4 c = *reinterpret_cast<const uint4*>(row_codes + g * 16);
            const uint32_t cws[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
            for (int wnum = 0; wnum < 4; ++wnum) {
                const uint32_t w = cws[wnum];
                // word wnum covers codes [32*wnum, 32*wnum+32) = activation
                // u32s [8*wnum, 8*wnum+8) = uint4s {2*wnum, 2*wnum+1}
                const uint4 av0 = aw[wnum * 2];
                const uint4 av1 = aw[wnum * 2 + 1];
                const uint32_t avs[8] = {av0.x, av0.y, av0.z, av0.w,
                                         av1.x, av1.y, av1.z, av1.w};
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    dot = dp4a_u8s8((w >> i) & 0x01010101u, avs[i], dot);
                }
            }
        }

        const float wd = __half2float(row_scale[g]);
        const int corrected = (NBITS == 1 ? 2 * dot : dot) - a_gsum[g];
        acc += wd * a_scale[g] * (float)corrected;
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) y[row] = acc;
}

}  // namespace

void quant_acts_launch(const __half* x, int K, int8_t* a8, float* a_scale,
                       int32_t* a_gsum, cudaStream_t stream) {
    const int groups = K / 128;
    const int warps_per_block = 8;
    const int blocks = (groups + warps_per_block - 1) / warps_per_block;
    quant_acts_kernel<<<blocks, warps_per_block * 32, 0, stream>>>(x, K, a8, a_scale, a_gsum);
}

void gemv_launch(int nbits, const uint8_t* codes, const __half* w_scale,
                 const int8_t* a8, const float* a_scale, const int32_t* a_gsum,
                 float* y, int M, int K, cudaStream_t stream) {
    const int warps_per_block = 8;
    const int blocks = (M + warps_per_block - 1) / warps_per_block;
    if (nbits == 2) {
        gemv_kernel<2><<<blocks, warps_per_block * 32, 0, stream>>>(
            codes, w_scale, a8, a_scale, a_gsum, y, M, K);
    } else {
        gemv_kernel<1><<<blocks, warps_per_block * 32, 0, stream>>>(
            codes, w_scale, a8, a_scale, a_gsum, y, M, K);
    }
}

}  // namespace bt
