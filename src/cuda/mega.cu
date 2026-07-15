#include "mega.h"

#include <cooperative_groups.h>

#include <algorithm>
#include <cstdio>

namespace cg = cooperative_groups;

namespace bt {

namespace {

__device__ __forceinline__ int dp4a_u8s8(uint32_t codes, uint32_t acts, int acc) {
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(acc) : "r"(codes), "r"(acts), "r"(acc));
    return acc;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    return v;
}

__device__ __forceinline__ float softplus_f(float x) {
    return x > 20.f ? x : log1pf(expf(x));  // matches ggml_softplus
}

// ---- int8 group quantization (identical rounding to quant_acts_kernel) ----
// one warp quantizes one 128-group whose values come from `val(j)`
template <typename F>
__device__ void quant_group(int g, F val, int8_t* a8, float* a_scale,
                            int32_t* a_gsum64) {
    const int lane = threadIdx.x & 31;
    float vals[4];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vals[i] = val(g * 128 + lane * 4 + i);
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
    reinterpret_cast<char4*>(a8)[g * 32 + lane] = q;
    const uint32_t half_mask = (lane < 16) ? 0x0000FFFFu : 0xFFFF0000u;
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) lsum += __shfl_down_sync(half_mask, lsum, off);
    if ((lane & 15) == 0) a_gsum64[g * 2 + lane / 16] = lsum;
    if (lane == 0) a_scale[g] = s;
}

// ---- ops --------------------------------------------------------------

// embed: dequant row *d_tok into x; also zero this token's reduction cells
__device__ void op_embed(const MegaParams& p, int gtid, int gsize) {
    const int row = *p.d_tok;
    const MegaMat& m = p.tok_embd;
    for (int j = gtid; j < p.n_embd; j += gsize) {
        const float d = __half2float(m.scales[(size_t)row * (m.K >> 7) + (j >> 7)]);
        if (m.nbits == 2) {
            const uint8_t* rc = m.codes + (size_t)row * (m.K >> 2);
            const int word = j >> 4, r = j & 15, i = r >> 2, b = r & 3;
            uint32_t w;
            memcpy(&w, rc + 4 * word, 4);
            const int q = (int)((w >> (8 * b + 2 * i)) & 3u);
            p.x[j] = (float)(q - 1) * d;
        } else {
            const uint8_t* rc = m.codes + (size_t)row * (m.K >> 3);
            const int word = j >> 5, r = j & 31, i = r >> 2, b = r & 3;
            uint32_t w;
            memcpy(&w, rc + 4 * word, 4);
            const int bit = (int)((w >> (8 * b + i)) & 1u);
            p.x[j] = bit ? d : -d;
        }
    }
    for (int j = gtid; j < 2 * p.n_layer + 4; j += gsize) p.red_scratch[j] = 0.f;
}

// rmsnorm phase A: accumulate sum(x^2) into red_scratch[slot]
__device__ void op_norm_partial(const MegaParams& p, int slot, int bid, int nblocks) {
    float ss = 0.f;
    for (int i = bid * (int)blockDim.x + threadIdx.x; i < p.n_embd;
         i += nblocks * (int)blockDim.x) {
        ss += p.x[i] * p.x[i];
    }
    __shared__ float warp_acc[8];
    const float w = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) warp_acc[threadIdx.x >> 5] = w;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < (int)blockDim.x / 32; ++i) t += warp_acc[i];
        atomicAdd(&p.red_scratch[slot], t);
    }
}

// rmsnorm phase B: normalize+scale, quantize per 128-group (warp per group)
__device__ void op_norm_quant(const MegaParams& p, const __half* w, int slot,
                              int bid, int nblocks) {
    const float inv = rsqrtf(p.red_scratch[slot] / p.n_embd + p.rms_eps);
    const int warp = (bid * (int)blockDim.x + threadIdx.x) >> 5;
    const int warps_total = nblocks * (int)blockDim.x / 32;
    for (int g = warp; g < p.n_embd / 128; g += warps_total) {
        quant_group(g,
                    [&](int j) { return p.x[j] * inv * __half2float(w[j]); },
                    p.a8, p.a_scale, p.a_gsum);
    }
}

// GEMV over re-tiled Q2 codes: v5 inner loop, grid-strided 32-row tiles.
// EPI 0: y32 store, 1: f16 store (dst16), 2: += into f32 (dst32)
template <int EPI>
__device__ void op_gemv(const MegaParams& p, const MegaMat& m, float* dst32,
                        __half* dst16, uint8_t* smem, int bid, int nblocks) {
    const int K = m.K;
    int8_t* s_a8 = reinterpret_cast<int8_t*>(smem);
    float* s_scale = reinterpret_cast<float*>(smem + K);
    int32_t* s_gsum = reinterpret_cast<int32_t*>(smem + K + (K / 128) * 4);

    for (int i = threadIdx.x; i < K / 16; i += blockDim.x) {
        reinterpret_cast<uint4*>(s_a8)[i] = reinterpret_cast<const uint4*>(p.a8)[i];
    }
    for (int i = threadIdx.x; i < K / 128; i += blockDim.x) s_scale[i] = p.a_scale[i];
    for (int i = threadIdx.x; i < K / 64; i += blockDim.x) s_gsum[i] = p.a_gsum[i];
    __syncthreads();

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int chunks = (K / 4) / 16;
    const int tiles = (m.M + 31) / 32;

    for (int tile = bid; tile < tiles; tile += nblocks) {
        const int row0 = tile * 32 + warp * 4;
        if (row0 >= m.M) continue;
        const int nrows = min(4, m.M - row0);
        const uint4* rc[4];
        const __half* rs[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = row0 + (r < nrows ? r : 0);
            rc[r] = reinterpret_cast<const uint4*>(m.codes + (size_t)row * (K >> 2));
            rs[r] = m.scales + (size_t)row * (K >> 7);
        }
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int c = lane; c < chunks; c += 32) {
            const uint4* aw = reinterpret_cast<const uint4*>(s_a8 + c * 64);
            const uint4 av[4] = {aw[0], aw[1], aw[2], aw[3]};
            const int g = c >> 1;
            const float as = s_scale[g];
            const int gs = s_gsum[c];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const uint4 cw = rc[r][c];
                const uint32_t cws[4] = {cw.x, cw.y, cw.z, cw.w};
                int dot = 0;
#pragma unroll
                for (int wnum = 0; wnum < 4; ++wnum) {
                    const uint32_t wv = cws[wnum];
                    dot = dp4a_u8s8(wv & 0x03030303u, av[wnum].x, dot);
                    dot = dp4a_u8s8((wv >> 2) & 0x03030303u, av[wnum].y, dot);
                    dot = dp4a_u8s8((wv >> 4) & 0x03030303u, av[wnum].z, dot);
                    dot = dp4a_u8s8((wv >> 6) & 0x03030303u, av[wnum].w, dot);
                }
                acc[r] += __half2float(rs[r][g]) * as * (float)(dot - gs);
            }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                acc[r] += __shfl_down_sync(0xFFFFFFFFu, acc[r], off);
            }
            if (lane == 0 && r < nrows) {
                if (EPI == 1) {
                    dst16[row0 + r] = __float2half(acc[r]);
                } else if (EPI == 2) {
                    dst32[row0 + r] += acc[r];
                } else {
                    dst32[row0 + r] = acc[r];
                }
            }
        }
        __syncthreads();  // s_a8 reused across tiles; keep warps in step
    }
}

// conv1d + SiLU over all channels, then L2-norm for q/k heads.
// tile = 128 channels; q/k region = first 2*Sk*Hk channels (heads of 128)
__device__ void op_conv_l2(const MegaParams& p, const MegaLayer& l, int bid,
                           int nblocks) {
    const int C = p.conv_channels;
    const int k = p.ssm_conv;
    const int qk_ch = 2 * p.ssm_state * p.ssm_groups;
    __half* buf = p.big_a;
    __shared__ float s_v[128];

    for (int tile = bid; tile < C / 128; tile += nblocks) {
        const int c0 = tile * 128;
        // 128 threads compute their channel; 256-thread block: half idle here
        if (threadIdx.x < 128) {
            const int c = c0 + threadIdx.x;
            float* st = l.conv_state + (size_t)c * (k - 1);
            const __half* wc = l.conv_w + (size_t)c * k;
            const float xnv = __half2float(buf[c]);
            float acc = xnv * __half2float(wc[k - 1]);
            for (int i = 0; i < k - 1; ++i) acc += st[i] * __half2float(wc[i]);
            for (int i = 0; i < k - 2; ++i) st[i] = st[i + 1];
            st[k - 2] = xnv;
            const float sv = acc / (1.f + expf(-acc));
            s_v[threadIdx.x] = sv;
        }
        __syncthreads();
        if (threadIdx.x < 128 && c0 < qk_ch) {
            // l2 normalize this 128-wide head (matches l2norm_heads_kernel)
            float ss = s_v[threadIdx.x] * s_v[threadIdx.x];
            // block-level reduce over the first 4 warps
            __shared__ float wsum[4];
            float wsr = warp_sum(ss);
            if ((threadIdx.x & 31) == 0) wsum[threadIdx.x >> 5] = wsr;
            __syncthreads();
            const float total = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float inv = rsqrtf(fmaxf(total, p.rms_eps));
            buf[c0 + threadIdx.x] = __float2half(s_v[threadIdx.x] * inv);
        } else if (threadIdx.x < 128) {
            buf[c0 + threadIdx.x] = __float2half(s_v[threadIdx.x]);
        }
        __syncthreads();
    }
}

// gated delta net step: one block per value head, grid-strided (port of
// gdn_decode_kernel with S=128)
__device__ void op_gdn(const MegaParams& p, const MegaLayer& l, int bid,
                       int nblocks) {
    constexpr int S = 128;
    const int Hv = p.ssm_dt_rank;
    const int Hk = p.ssm_groups;
    const float scale = rsqrtf((float)S);
    const __half* qkv = p.big_a;
    const __half* alpha_raw = p.big_a + l.in_rows + l.gate_rows;
    const __half* beta_raw = alpha_raw + Hv;
    __shared__ float s_k[S], s_q[S];

    for (int h = bid; h < Hv; h += nblocks) {
        const int hk = h % Hk;
        for (int i = threadIdx.x; i < S; i += blockDim.x) {
            s_q[i] = __half2float(qkv[(size_t)hk * S + i]);
            s_k[i] = __half2float(qkv[(size_t)(Hk + hk) * S + i]);
        }
        __syncthreads();

        // ssm_a already stores -exp(A_log); same math as gdn_decode_kernel
        const float g = expf(l.ssm_a[h] *
                             softplus_f(__half2float(alpha_raw[h]) + l.ssm_dt[h]));
        const float beta = 1.f / (1.f + expf(-__half2float(beta_raw[h])));

        float* S_h = l.gdn_state + (size_t)h * S * S;
        const __half* v_h = qkv + (size_t)2 * p.ssm_state * Hk + (size_t)h * S;
        __half* out_h = p.attn_out + (size_t)h * S;

        const int warp = threadIdx.x / 32;
        const int lane = threadIdx.x & 31;
        for (int c = warp; c < S; c += (int)blockDim.x / 32) {
            float* col = S_h + (size_t)c * S;
            float s_reg[4];
            float kv = 0.f;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int i = r * 32 + lane;
                s_reg[r] = col[i];
                kv += s_reg[r] * s_k[i];
            }
            kv = warp_sum(kv);
            const float delta = (__half2float(v_h[c]) - g * kv) * beta;
            float attn = 0.f;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int i = r * 32 + lane;
                s_reg[r] = g * s_reg[r] + s_k[i] * delta;
                col[i] = s_reg[r];
                attn += s_reg[r] * s_q[i];
            }
            attn = warp_sum(attn);
            if (lane == 0) out_h[c] = __float2half(attn * scale);
        }
        __syncthreads();
    }
}

// gated RMS norm over each 128-wide head + silu(z) gate + int8 quantization;
// head == quant group, one warp each
__device__ void op_gdn_gate_quant(const MegaParams& p, const MegaLayer& l,
                                  int bid, int nblocks) {
    const int Hv = p.ssm_dt_rank;
    const __half* z = p.big_a + l.in_rows;
    const int warp = (bid * (int)blockDim.x + threadIdx.x) >> 5;
    const int warps_total = nblocks * (int)blockDim.x / 32;
    const int lane = threadIdx.x & 31;

    for (int h = warp; h < Hv; h += warps_total) {
        const __half* oh = p.attn_out + (size_t)h * 128;
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float v = __half2float(oh[lane * 4 + i]);
            ss += v * v;
        }
        ss = warp_sum(ss);
        const float inv = rsqrtf(ss / 128 + p.rms_eps);
        quant_group(h,
                    [&](int j) {
                        const int d = j - h * 128;
                        const float normed = __half2float(oh[d]) * inv *
                                             __half2float(l.ssm_norm[d]);
                        const float zv = __half2float(z[h * 128 + d]);
                        return zv / (1.f + expf(-zv)) * normed;
                    },
                    p.a8, p.a_scale, p.a_gsum);
    }
}

// attention prep: per q-head norm+rope into q_buf; per kv-head norm+rope k and
// raw v appended straight into the caches at *d_pos
__device__ void op_attn_prep(const MegaParams& p, const MegaLayer& l, int bid,
                             int nblocks) {
    const int D = p.head_dim;
    const int H = p.n_head, Hkv = p.n_head_kv;
    const int pos = *p.d_pos;
    const __half* qg = p.big_a;                       // [H][2D]
    const __half* kk = p.big_a + l.in_rows;           // [Hkv][D]
    const __half* vv = kk + Hkv * D;
    __shared__ float s_d[256];

    for (int unit = bid; unit < H + 2 * Hkv; unit += nblocks) {
        const bool is_q = unit < H;
        const bool is_k = !is_q && unit < H + Hkv;
        const int h = is_q ? unit : (is_k ? unit - H : unit - H - Hkv);
        if (!is_q && !is_k) {
            // v: raw copy into cache
            __half* vdst = l.v_cache + (size_t)pos * Hkv * D + (size_t)h * D;
            for (int i = threadIdx.x; i < D; i += blockDim.x) {
                vdst[i] = vv[(size_t)h * D + i];
            }
            continue;
        }
        const __half* src = is_q ? qg + (size_t)h * 2 * D : kk + (size_t)h * D;
        const __half* w = is_q ? l.q_norm : l.k_norm;
        // rms over D
        float ss = 0.f;
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            const float v = __half2float(src[i]);
            s_d[i] = v;
            ss += v * v;
        }
        __shared__ float wsum[8];
        float wsr = warp_sum(ss);
        if ((threadIdx.x & 31) == 0) wsum[threadIdx.x >> 5] = wsr;
        __syncthreads();
        float total = 0.f;
        for (int i = 0; i < (int)blockDim.x / 32; ++i) total += wsum[i];
        const float inv = rsqrtf(total / D + p.rms_eps);
        __syncthreads();
        // normalize + rope (pairs (i, i+rot/2) over first n_rot dims)
        __half* dst = is_q ? p.q_buf + (size_t)h * D
                           : l.k_cache + (size_t)pos * Hkv * D + (size_t)h * D;
        const int half_rot = p.n_rot / 2;
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            float v = s_d[i] * inv * __half2float(w[i]);
            if (i < p.n_rot) {
                const int pair = i < half_rot ? i : i - half_rot;
                const float theta = pos * powf(p.rope_base, -2.f * pair / p.n_rot);
                float c, s;
                sincosf(theta, &s, &c);
                const float x0 = s_d[pair] * inv * __half2float(w[pair]);
                const float x1 = s_d[pair + half_rot] * inv * __half2float(w[pair + half_rot]);
                v = i < half_rot ? x0 * c - x1 * s : x0 * s + x1 * c;
            }
            dst[i] = __float2half(v);
        }
        __syncthreads();
    }
}

// softmax attention decode (port of attn_decode_kernel, block per head)
__device__ void op_attn(const MegaParams& p, const MegaLayer& l, int bid,
                        int nblocks, uint8_t* smem) {
    constexpr int kWarps = 8;
    const int D = p.head_dim;
    const int kVecPerLane = D / 32;
    const int H = p.n_head, Hkv = p.n_head_kv;
    const int ctx_len = *p.d_pos + 1;
    const size_t kv_row = (size_t)Hkv * D;

    float* s_q = reinterpret_cast<float*>(smem);              // D
    float* s_m = s_q + D;                                     // kWarps
    float* s_l = s_m + kWarps;                                // kWarps
    float* s_acc = s_l + kWarps;                              // kWarps*D

    for (int h = bid; h < H; h += nblocks) {
        const int hkv = h / (H / Hkv);
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            s_q[i] = __half2float(p.q_buf[(size_t)h * D + i]);
        }
        __syncthreads();

        const int warp = threadIdx.x / 32;
        const int lane = threadIdx.x & 31;
        float m = -INFINITY, lsum = 0.f;
        float acc[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) acc[r] = 0.f;

        for (int pos = warp; pos < ctx_len; pos += kWarps) {
            const __half* k_pos = l.k_cache + (size_t)pos * kv_row + (size_t)hkv * D;
            float dot = 0.f;
            for (int r = 0; r < kVecPerLane; ++r) {
                const int d = r * 32 + lane;
                dot += s_q[d] * __half2float(k_pos[d]);
            }
            dot = warp_sum(dot);
            const float score = dot * rsqrtf((float)D);
            const float m_new = fmaxf(m, score);
            const float rescale = expf(m - m_new);
            const float pw = expf(score - m_new);
            const __half* v_pos = l.v_cache + (size_t)pos * kv_row + (size_t)hkv * D;
            for (int r = 0; r < kVecPerLane; ++r) {
                const int d = r * 32 + lane;
                acc[r] = acc[r] * rescale + pw * __half2float(v_pos[d]);
            }
            lsum = lsum * rescale + pw;
            m = m_new;
        }
        if (lane == 0) {
            s_m[warp] = m;
            s_l[warp] = lsum;
        }
        for (int r = 0; r < kVecPerLane; ++r) s_acc[warp * D + r * 32 + lane] = acc[r];
        __syncthreads();

        if (warp == 0) {
            float m_star = -INFINITY;
            for (int w = 0; w < kWarps; ++w) m_star = fmaxf(m_star, s_m[w]);
            float l_star = 0.f;
            float o[8];
            for (int r = 0; r < kVecPerLane; ++r) o[r] = 0.f;
            for (int w = 0; w < kWarps; ++w) {
                const float factor = s_m[w] == -INFINITY ? 0.f : expf(s_m[w] - m_star);
                l_star += factor * s_l[w];
                for (int r = 0; r < kVecPerLane; ++r) {
                    o[r] += factor * s_acc[w * D + r * 32 + lane];
                }
            }
            const float inv_l = 1.f / l_star;
            for (int r = 0; r < kVecPerLane; ++r) {
                p.attn_out[(size_t)h * D + r * 32 + lane] = __float2half(o[r] * inv_l);
            }
        }
        __syncthreads();
    }
}

// attention output gate (sigmoid, gate lives interleaved in qg) + quantize
__device__ void op_attn_gate_quant(const MegaParams& p, const MegaLayer& l,
                                   int bid, int nblocks) {
    const int D = p.head_dim;
    const int n = p.n_head * D;
    const __half* qg = p.big_a;
    const int warp = (bid * (int)blockDim.x + threadIdx.x) >> 5;
    const int warps_total = nblocks * (int)blockDim.x / 32;
    for (int g = warp; g < n / 128; g += warps_total) {
        quant_group(g,
                    [&](int j) {
                        const int h = j / D, d = j % D;
                        const float gate = __half2float(qg[(size_t)h * 2 * D + D + d]);
                        return __half2float(p.attn_out[j]) *
                               (1.f / (1.f + expf(-gate)));
                    },
                    p.a8, p.a_scale, p.a_gsum);
    }
}

// swiglu over fused [gate|up] + quantize (group == 128 wide)
__device__ void op_swiglu_quant(const MegaParams& p, const MegaLayer& l, int bid,
                                int nblocks) {
    const int warp = (bid * (int)blockDim.x + threadIdx.x) >> 5;
    const int warps_total = nblocks * (int)blockDim.x / 32;
    for (int g = warp; g < p.n_ff / 128; g += warps_total) {
        quant_group(g,
                    [&](int j) {
                        const float gv = __half2float(p.big_a[j]);
                        const float uv = __half2float(p.big_a[l.mlp_gate_rows + j]);
                        return gv / (1.f + expf(-gv)) * uv;
                    },
                    p.a8, p.a_scale, p.a_gsum);
    }
}

// argmax over y32 (partials per block, then block 0 finishes + bumps state)
__device__ void op_argmax_partial(const MegaParams& p, int bid, int nblocks) {
    float best = -INFINITY;
    int besti = 0;
    for (int i = bid * (int)blockDim.x + threadIdx.x; i < p.vocab;
         i += nblocks * (int)blockDim.x) {
        if (p.y32[i] > best || (p.y32[i] == best && i < besti)) {
            best = p.y32[i];
            besti = i;
        }
    }
    __shared__ float bv[8];
    __shared__ int bi[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xFFFFFFFFu, best, off);
        const int oi = __shfl_down_sync(0xFFFFFFFFu, besti, off);
        if (ov > best || (ov == best && oi < besti)) {
            best = ov;
            besti = oi;
        }
    }
    if (lane == 0) {
        bv[warp] = best;
        bi[warp] = besti;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < (int)blockDim.x / 32; ++i) {
            if (bv[i] > bv[0] || (bv[i] == bv[0] && bi[i] < bi[0])) {
                bv[0] = bv[i];
                bi[0] = bi[i];
            }
        }
        p.amax_v[bid] = bv[0];
        p.amax_i[bid] = bi[0];
    }
}

__device__ void op_argmax_final_bump(const MegaParams& p, int nblocks) {
    if (blockIdx.x != 0) return;
    __shared__ float bv;
    __shared__ int bi;
    if (threadIdx.x == 0) {
        bv = -INFINITY;
        bi = 0;
        for (int i = 0; i < nblocks; ++i) {
            if (p.amax_v[i] > bv || (p.amax_v[i] == bv && p.amax_i[i] < bi)) {
                bv = p.amax_v[i];
                bi = p.amax_i[i];
            }
        }
        *p.d_tok = bi;
        p.d_ring[*p.d_step % p.ring_cap] = bi;
        ++*p.d_step;
        ++*p.d_pos;
    }
}

// ---- the megakernel: one launch = one decoded token --------------------

__global__ void __launch_bounds__(256)
mega_decode_kernel(MegaParams p) {
    cg::grid_group grid = cg::this_grid();
    extern __shared__ uint8_t smem[];
    const int bid = blockIdx.x;
    const int nb = gridDim.x;
    const int gtid = bid * (int)blockDim.x + threadIdx.x;
    const int gsz = nb * (int)blockDim.x;

    op_embed(p, gtid, gsz);
    grid.sync();

    for (int il = 0; il < p.n_layer; ++il) {
        const MegaLayer& l = p.layers[il];
        op_norm_partial(p, 2 * il, bid, nb);
        grid.sync();
        op_norm_quant(p, l.attn_norm, 2 * il, bid, nb);
        grid.sync();
        op_gemv<1>(p, l.proj, nullptr, p.big_a, smem, bid, nb);
        grid.sync();
        if (l.recurrent) {
            op_conv_l2(p, l, bid, nb);
            grid.sync();
            op_gdn(p, l, bid, nb);
            grid.sync();
            op_gdn_gate_quant(p, l, bid, nb);
        } else {
            op_attn_prep(p, l, bid, nb);
            grid.sync();
            op_attn(p, l, bid, nb, smem);
            grid.sync();
            op_attn_gate_quant(p, l, bid, nb);
        }
        grid.sync();
        op_gemv<2>(p, l.out_proj, p.x, nullptr, smem, bid, nb);
        grid.sync();
        op_norm_partial(p, 2 * il + 1, bid, nb);
        grid.sync();
        op_norm_quant(p, l.post_norm, 2 * il + 1, bid, nb);
        grid.sync();
        op_gemv<1>(p, l.gate_up, nullptr, p.big_a, smem, bid, nb);
        grid.sync();
        op_swiglu_quant(p, l, bid, nb);
        grid.sync();
        op_gemv<2>(p, l.down, p.x, nullptr, smem, bid, nb);
        grid.sync();
    }

    op_norm_partial(p, 2 * p.n_layer, bid, nb);
    grid.sync();
    op_norm_quant(p, p.output_norm, 2 * p.n_layer, bid, nb);
    grid.sync();
    op_gemv<0>(p, p.lm_head, p.y32, nullptr, smem, bid, nb);
    grid.sync();
    op_argmax_partial(p, bid, nb);
    grid.sync();
    op_argmax_final_bump(p, nb);
}

}  // namespace

bool mega_decode_launch(const MegaParams& p, cudaStream_t stream) {
    static int grid_size = 0;
    static int smem_bytes = 0;
    if (grid_size == 0) {
        // largest GEMV K decides the activation-staging pool
        int max_k = std::max({p.n_embd, p.n_ff, p.ssm_dt_rank * p.head_v_dim,
                              p.n_head * p.head_dim});
        smem_bytes = max_k + (max_k / 128) * 4 + (max_k / 64) * 4;

        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, dev);
        int per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, (const void*)mega_decode_kernel, 256, smem_bytes);
        if (per_sm < 1 || !prop.cooperativeLaunch) {
            std::fprintf(stderr, "mega: cooperative launch unavailable (per_sm=%d)\n",
                         per_sm);
            return false;
        }
        grid_size = prop.multiProcessorCount * per_sm;
        std::fprintf(stderr, "mega: grid %d blocks (%d/SM), smem %d B\n", grid_size,
                     per_sm, smem_bytes);
    }
    MegaParams p_copy = p;
    void* args[] = {&p_copy};
    const cudaError_t err = cudaLaunchCooperativeKernel(
        (const void*)mega_decode_kernel, dim3((unsigned)grid_size), dim3(256), args,
        (size_t)smem_bytes, stream);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "mega launch: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

}  // namespace bt
