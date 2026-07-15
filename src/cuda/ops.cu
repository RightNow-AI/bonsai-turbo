#include "ops.h"

namespace bt {

namespace {

__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    const int n_warps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < n_warps) ? warp_sums[threadIdx.x] : 0.f;
    if (warp == 0) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
        if (lane == 0) warp_sums[0] = v;
    }
    __syncthreads();
    return warp_sums[0];
}

__global__ void rmsnorm_kernel(const __half* __restrict__ x, const __half* __restrict__ w,
                               __half* __restrict__ y, int n, float eps) {
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __half2float(x[i]);
        ss += v * v;
    }
    ss = block_reduce_sum(ss);
    const float inv = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        y[i] = __float2half(__half2float(x[i]) * inv * __half2float(w[i]));
    }
}

// one block per head
__global__ void rmsnorm_heads_kernel(const __half* __restrict__ x, const __half* __restrict__ w,
                                     __half* __restrict__ y, int d, float eps) {
    const __half* xh = x + (size_t)blockIdx.x * d;
    __half* yh = y + (size_t)blockIdx.x * d;
    float ss = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float v = __half2float(xh[i]);
        ss += v * v;
    }
    ss = block_reduce_sum(ss);
    const float inv = rsqrtf(ss / d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        yh[i] = __float2half(__half2float(xh[i]) * inv * __half2float(w[i]));
    }
}

// one block per head; matches ggml_l2_norm (no 1/d)
__global__ void l2norm_heads_kernel(const __half* __restrict__ x, __half* __restrict__ y,
                                    int d, float eps) {
    const __half* xh = x + (size_t)blockIdx.x * d;
    __half* yh = y + (size_t)blockIdx.x * d;
    float ss = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float v = __half2float(xh[i]);
        ss += v * v;
    }
    ss = block_reduce_sum(ss);
    const float inv = rsqrtf(fmaxf(ss, eps));
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        yh[i] = __float2half(__half2float(xh[i]) * inv);
    }
}

__global__ void add_inplace_kernel(__half* __restrict__ y, const __half* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(__half2float(y[i]) + __half2float(x[i]));
}

__global__ void silu_mul_kernel(const __half* __restrict__ a, const __half* __restrict__ b,
                                __half* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = __half2float(a[i]);
        const float s = v / (1.f + expf(-v));
        y[i] = __float2half(s * __half2float(b[i]));
    }
}

__global__ void sigmoid_mul_kernel(const __half* __restrict__ x, const __half* __restrict__ g,
                                   __half* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float s = 1.f / (1.f + expf(-__half2float(g[i])));
        y[i] = __float2half(__half2float(x[i]) * s);
    }
}

__global__ void f32_to_f16_kernel(const float* __restrict__ x, __half* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

// NeoX pairing: dims (i, i + rot/2) rotate by pos * base^(-2i/rot)
__global__ void rope_neox_kernel(__half* __restrict__ x, int d, int rot, int pos,
                                 float freq_base) {
    __half* xh = x + (size_t)blockIdx.x * d;
    const int half_rot = rot / 2;
    for (int i = threadIdx.x; i < half_rot; i += blockDim.x) {
        const float theta = pos * powf(freq_base, -2.f * i / rot);
        float c, s;
        sincosf(theta, &s, &c);  // accurate variant: logit parity beats speed here
        const float x0 = __half2float(xh[i]);
        const float x1 = __half2float(xh[i + half_rot]);
        xh[i] = __float2half(x0 * c - x1 * s);
        xh[i + half_rot] = __float2half(x0 * s + x1 * c);
    }
}

__global__ void embed_lookup_kernel(const __half* __restrict__ table, int token, int n,
                                    __half* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = table[(size_t)token * n + i];
}

__global__ void argmax_kernel(const float* __restrict__ x, int n, int32_t* __restrict__ out) {
    __shared__ float best_v[32];
    __shared__ int best_i[32];
    float bv = -INFINITY;
    int bi = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (x[i] > bv) {
            bv = x[i];
            bi = i;
        }
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xFFFFFFFFu, bv, off);
        const int oi = __shfl_down_sync(0xFFFFFFFFu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) {
        best_v[warp] = bv;
        best_i[warp] = bi;
    }
    __syncthreads();
    if (warp == 0) {
        const int n_warps = (blockDim.x + 31) >> 5;
        bv = lane < n_warps ? best_v[lane] : -INFINITY;
        bi = lane < n_warps ? best_i[lane] : 0;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xFFFFFFFFu, bv, off);
            const int oi = __shfl_down_sync(0xFFFFFFFFu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) *out = bi;
    }
}

__global__ void conv1d_step_kernel(const __half* __restrict__ x, const __half* __restrict__ w,
                                   float* __restrict__ conv_state, __half* __restrict__ y,
                                   int C, int k) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* st = conv_state + (size_t)c * (k - 1);
    const __half* wc = w + (size_t)c * k;
    const float xn = __half2float(x[c]);

    float acc = xn * __half2float(wc[k - 1]);
    for (int i = 0; i < k - 1; ++i) acc += st[i] * __half2float(wc[i]);
    for (int i = 0; i < k - 2; ++i) st[i] = st[i + 1];
    st[k - 2] = xn;

    const float s = acc / (1.f + expf(-acc));  // SiLU
    y[c] = __float2half(s);
}

constexpr int kThreads = 256;
inline int blocks_for(int n) { return (n + kThreads - 1) / kThreads; }

}  // namespace

void rmsnorm_launch(const __half* x, const __half* w, __half* y, int n, float eps,
                    cudaStream_t stream) {
    rmsnorm_kernel<<<1, 1024, 0, stream>>>(x, w, y, n, eps);
}

void rmsnorm_heads_launch(const __half* x, const __half* w, __half* y, int h, int d,
                          float eps, cudaStream_t stream) {
    rmsnorm_heads_kernel<<<h, 128, 0, stream>>>(x, w, y, d, eps);
}

void l2norm_heads_launch(const __half* x, __half* y, int h, int d, float eps,
                         cudaStream_t stream) {
    l2norm_heads_kernel<<<h, 128, 0, stream>>>(x, y, d, eps);
}

void add_inplace_launch(__half* y, const __half* x, int n, cudaStream_t stream) {
    add_inplace_kernel<<<blocks_for(n), kThreads, 0, stream>>>(y, x, n);
}

void silu_mul_launch(const __half* a, const __half* b, __half* y, int n,
                     cudaStream_t stream) {
    silu_mul_kernel<<<blocks_for(n), kThreads, 0, stream>>>(a, b, y, n);
}

void sigmoid_mul_launch(const __half* x, const __half* g, __half* y, int n,
                        cudaStream_t stream) {
    sigmoid_mul_kernel<<<blocks_for(n), kThreads, 0, stream>>>(x, g, y, n);
}

void f32_to_f16_launch(const float* x, __half* y, int n, cudaStream_t stream) {
    f32_to_f16_kernel<<<blocks_for(n), kThreads, 0, stream>>>(x, y, n);
}

void rope_neox_launch(__half* x, int h, int d, int rot, int pos, float freq_base,
                      cudaStream_t stream) {
    rope_neox_kernel<<<h, 128, 0, stream>>>(x, d, rot, pos, freq_base);
}

void embed_lookup_launch(const __half* table, int token, int n, __half* out,
                         cudaStream_t stream) {
    embed_lookup_kernel<<<blocks_for(n), kThreads, 0, stream>>>(table, token, n, out);
}

void argmax_launch(const float* x, int n, int32_t* out, cudaStream_t stream) {
    argmax_kernel<<<1, 1024, 0, stream>>>(x, n, out);
}

void conv1d_step_launch(const __half* x, const __half* w, float* conv_state,
                        __half* y, int C, int k, cudaStream_t stream) {
    conv1d_step_kernel<<<blocks_for(C), kThreads, 0, stream>>>(x, w, conv_state, y, C, k);
}

}  // namespace bt
