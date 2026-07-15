#include "gemv.h"

namespace bt {

namespace {

// u8 codes x s8 activations dot product; no mixed-sign __dp4a overload exists.
__device__ __forceinline__ int dp4a_u8s8(uint32_t codes, uint32_t acts, int acc) {
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(acc) : "r"(codes), "r"(acts), "r"(acc));
    return acc;
}

// One warp per 128-group: absmax -> int8 quant -> HALF-group (64) sums.
// gsum64 granularity lets the GEMV apply the (code-1)/(2b-1) correction per
// 16-byte code chunk (64 codes for Q2, 128 for Q1) without crossing chunks.
__global__ void quant_acts_kernel(const __half* __restrict__ x, int K,
                                  int8_t* __restrict__ a8,
                                  float* __restrict__ a_scale,
                                  int32_t* __restrict__ a_gsum64) {
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

    int lsum = 0;
    char4 q;
    int8_t* qp = reinterpret_cast<int8_t*>(&q);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int v = __float2int_rn(vals[i] * inv_s);
        v = max(-127, min(127, v));
        qp[i] = (int8_t)v;
        lsum += v;
    }
    reinterpret_cast<char4*>(a8)[group * 32 + lane] = q;

    // sum within each half-warp: lanes 0-15 hold codes [0,64), 16-31 [64,128)
    const uint32_t half_mask = (lane < 16) ? 0x0000FFFFu : 0xFFFF0000u;
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        lsum += __shfl_down_sync(half_mask, lsum, off);
    }
    if ((lane & 15) == 0) {
        a_gsum64[group * 2 + lane / 16] = lsum;
    }
    if (lane == 0) a_scale[group] = s;
}

// 32 rows per block (8 warps x 4 rows), activations staged in shared memory.
// Each row is covered by 8 lanes streaming contiguous 16B code chunks:
//   Q2: chunk = 64 codes  = half a group  -> correction gsum64[chunk]
//   Q1: chunk = 128 codes = a full group  -> correction gsum64[2c] + gsum64[2c+1]
template <int NBITS>
__global__ void gemv_kernel(const uint8_t* __restrict__ codes,
                            const __half* __restrict__ w_scale,
                            const int8_t* __restrict__ a8,
                            const float* __restrict__ a_scale,
                            const int32_t* __restrict__ a_gsum64,
                            float* __restrict__ y, int M, int K) {
    extern __shared__ uint8_t smem[];
    int8_t* s_a8 = reinterpret_cast<int8_t*>(smem);                       // K bytes
    float* s_scale = reinterpret_cast<float*>(smem + K);                  // K/128 floats
    int32_t* s_gsum = reinterpret_cast<int32_t*>(smem + K + (K / 128) * 4);  // K/64 ints

    // cooperative stage
    for (int i = threadIdx.x; i < K / 16; i += blockDim.x) {
        reinterpret_cast<uint4*>(s_a8)[i] = reinterpret_cast<const uint4*>(a8)[i];
    }
    for (int i = threadIdx.x; i < K / 128; i += blockDim.x) s_scale[i] = a_scale[i];
    for (int i = threadIdx.x; i < K / 64; i += blockDim.x) s_gsum[i] = a_gsum64[i];
    __syncthreads();

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int sub = lane / 8;   // row within the warp's quad
    const int li = lane & 7;    // lane within the row's 8-lane team

    const int row = blockIdx.x * 32 + warp * 4 + sub;
    if (row >= M) return;

    const int code_row_bytes = NBITS == 2 ? (K >> 2) : (K >> 3);
    const int chunks = code_row_bytes / 16;  // 16B chunks per row
    const uint4* row_codes = reinterpret_cast<const uint4*>(codes + (size_t)row * code_row_bytes);
    const __half* row_scale = w_scale + (size_t)row * (K >> 7);

    float acc = 0.f;
    for (int c = li; c < chunks; c += 8) {
        const uint4 cw = row_codes[c];
        const uint32_t cws[4] = {cw.x, cw.y, cw.z, cw.w};
        int dot = 0;

        if (NBITS == 2) {
            // chunk c covers codes [64c, 64c+64) = activation bytes [64c, 64c+64)
            const uint4* aw = reinterpret_cast<const uint4*>(s_a8 + c * 64);
#pragma unroll
            for (int wnum = 0; wnum < 4; ++wnum) {
                const uint4 av = aw[wnum];
                const uint32_t w = cws[wnum];
                dot = dp4a_u8s8(w & 0x03030303u, av.x, dot);
                dot = dp4a_u8s8((w >> 2) & 0x03030303u, av.y, dot);
                dot = dp4a_u8s8((w >> 4) & 0x03030303u, av.z, dot);
                dot = dp4a_u8s8((w >> 6) & 0x03030303u, av.w, dot);
            }
            const int g = c >> 1;  // 2 chunks per 128-group
            const float wd = __half2float(row_scale[g]);
            acc += wd * s_scale[g] * (float)(dot - s_gsum[c]);
        } else {
            // chunk c covers codes [128c, 128c+128) = one full group
            const uint4* aw = reinterpret_cast<const uint4*>(s_a8 + c * 128);
#pragma unroll
            for (int wnum = 0; wnum < 4; ++wnum) {
                const uint32_t w = cws[wnum];
                const uint4 av0 = aw[wnum * 2];
                const uint4 av1 = aw[wnum * 2 + 1];
                const uint32_t avs[8] = {av0.x, av0.y, av0.z, av0.w,
                                         av1.x, av1.y, av1.z, av1.w};
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    dot = dp4a_u8s8((w >> i) & 0x01010101u, avs[i], dot);
                }
            }
            const int g = c;
            const float wd = __half2float(row_scale[g]);
            const int gsum = s_gsum[2 * c] + s_gsum[2 * c + 1];
            acc += wd * s_scale[g] * (float)(2 * dot - gsum);
        }
    }

    // reduce across the row's 8-lane team
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    }
    if (li == 0) y[row] = acc;
}

}  // namespace

void quant_acts_launch(const __half* x, int K, int8_t* a8, float* a_scale,
                       int32_t* a_gsum64, cudaStream_t stream) {
    const int groups = K / 128;
    const int warps_per_block = 8;
    const int blocks = (groups + warps_per_block - 1) / warps_per_block;
    quant_acts_kernel<<<blocks, warps_per_block * 32, 0, stream>>>(x, K, a8, a_scale, a_gsum64);
}

void gemv_launch(int nbits, const uint8_t* codes, const __half* w_scale,
                 const int8_t* a8, const float* a_scale, const int32_t* a_gsum64,
                 float* y, int M, int K, cudaStream_t stream) {
    const int blocks = (M + 31) / 32;
    const int smem = K + (K / 128) * 4 + (K / 64) * 4;
    if (nbits == 2) {
        gemv_kernel<2><<<blocks, 256, smem, stream>>>(codes, w_scale, a8, a_scale, a_gsum64, y, M, K);
    } else {
        gemv_kernel<1><<<blocks, 256, smem, stream>>>(codes, w_scale, a8, a_scale, a_gsum64, y, M, K);
    }
}

}  // namespace bt
