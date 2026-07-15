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

// idle-block trick: blocks with no work in a narrow phase pull the NEXT
// GEMV's weight codes into L2 — budget-bounded so the grid barrier never
// waits on a prefetching block
__device__ void prefetch_l2(const MegaParams& p, const MegaMat& m, int rel_bid,
                            int nb_idle, int budget_words_per_block) {
    if (nb_idle <= 0) return;
    const size_t words = ((size_t)m.M * (m.nbits == 2 ? m.K / 4 : m.K / 8)) / 16;
    const uint4* src = reinterpret_cast<const uint4*>(m.codes);
    const size_t start = (size_t)rel_bid * budget_words_per_block;
    const size_t end = min(words, start + (size_t)budget_words_per_block);
    unsigned acc = 0;
    for (size_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        acc += __ldg(&src[i]).x;
    }
    if (acc == 0x9E3779B9u && threadIdx.x == 0) {
        atomicAdd(&p.red_scratch[0], 0.f);  // defeat DCE, no observable effect
    }
}

// ---- fused activation staging -----------------------------------------
//
// Every GEMV block quantizes its own activation copy straight into shared
// memory, computing the producer op (norm / gates / swiglu) on the fly.
// Rounding is identical to the standalone quant_acts kernel, so results stay
// bit-compatible with the graph-mode engine.

enum StageSrc {
    SRC_NORM_X = 0,    // rmsnorm(x) * w        (inv precomputed from red_scratch)
    SRC_GDN_GATE = 1,  // silu(z) * headnorm(gdn_out) * ssm_norm
    SRC_ATTN_GATE = 2, // attn_out * sigmoid(gate from interleaved qg)
    SRC_SWIGLU = 3,    // silu(gate) * up from fused [gate|up]
    SRC_FROM_A8 = 4,   // copy the globally pre-quantized activation buffers
};

// quantize one 128-group of vals held per-lane (4 each) into block-local smem
__device__ void quantize_vals(const float* vals, int g, int8_t* s_a8,
                              float* s_scale, int32_t* s_gsum) {
    const int lane = threadIdx.x & 31;
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) amax = fmaxf(amax, fabsf(vals[i]));
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
    reinterpret_cast<char4*>(s_a8)[g * 32 + lane] = q;
    const uint32_t half_mask = (lane < 16) ? 0x0000FFFFu : 0xFFFF0000u;
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) lsum += __shfl_down_sync(half_mask, lsum, off);
    if ((lane & 15) == 0) s_gsum[g * 2 + lane / 16] = lsum;
    if (lane == 0) s_scale[g] = s;
}

// compute this lane's 4 activation values of group g for the given source
template <int SRC>
__device__ void stage_vals(const MegaParams& p, const MegaLayer* l, float inv_norm,
                           const __half* norm_w, int g, float* vals) {
    const int lane = threadIdx.x & 31;
    const int j0 = g * 128 + lane * 4;
    if (SRC == SRC_NORM_X) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int j = j0 + i;
            vals[i] = p.x[j] * inv_norm * __half2float(norm_w[j]);
        }
    } else if (SRC == SRC_SWIGLU) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int j = j0 + i;
            const float gv = __half2float(p.big_a[j]);
            const float uv = __half2float(p.big_a[l->mlp_gate_rows + j]);
            vals[i] = gv / (1.f + expf(-gv)) * uv;
        }
    } else if (SRC == SRC_ATTN_GATE) {
        const int D = p.head_dim;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int j = j0 + i;
            const int h = j / D, d = j % D;
            const float gate = __half2float(p.big_a[(size_t)h * 2 * D + D + d]);
            vals[i] = __half2float(p.attn_out[j]) * (1.f / (1.f + expf(-gate)));
        }
    } else {  // SRC_GDN_GATE: group == one 128-wide value head
        const __half* oh = p.attn_out + (size_t)g * 128;
        const __half* z = p.big_a + l->in_rows;
        float ss = 0.f;
        float ov[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            ov[i] = __half2float(oh[lane * 4 + i]);
            ss += ov[i] * ov[i];
        }
        ss = warp_sum(ss);
        const float inv = rsqrtf(ss / 128 + p.rms_eps);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d = lane * 4 + i;
            const float normed = ov[i] * inv * __half2float(l->ssm_norm[d]);
            const float zv = __half2float(z[(size_t)g * 128 + d]);
            vals[i] = zv / (1.f + expf(-zv)) * normed;
        }
    }
}

// ---- the GEMV phase ----------------------------------------------------
//
// EPI 0: store f32 into dst32 (logits)
// EPI 1: store f16 into dst16 (projections)
// EPI 2: accumulate into dst32 (fp32 residual); when norm_slot >= 0 the
//        epilogue also accumulates sum(x_new^2) into red_scratch[norm_slot],
//        replacing the next norm's partial-reduction phase entirely.
template <int EPI, int SRC>
__device__ void op_gemv(const MegaParams& p, const MegaLayer* l, const MegaMat& m,
                        float* dst32, __half* dst16, int norm_slot,
                        float inv_norm, const __half* norm_w, uint8_t* smem,
                        int bid, int nblocks) {
    const int K = m.K;
    int8_t* s_a8 = reinterpret_cast<int8_t*>(smem);
    float* s_scale = reinterpret_cast<float*>(smem + K);
    int32_t* s_gsum = reinterpret_cast<int32_t*>(smem + K + (K / 128) * 4);

    // staging: raw copy of pre-quantized buffers, or fused producer+quant
    if (SRC == SRC_FROM_A8) {
        for (int i = threadIdx.x; i < K / 16; i += blockDim.x) {
            reinterpret_cast<uint4*>(s_a8)[i] = reinterpret_cast<const uint4*>(p.a8)[i];
        }
        for (int i = threadIdx.x; i < K / 128; i += blockDim.x) s_scale[i] = p.a_scale[i];
        for (int i = threadIdx.x; i < K / 64; i += blockDim.x) s_gsum[i] = p.a_gsum[i];
    } else {
        const int warp = threadIdx.x / 32;
        const int warps = (int)blockDim.x / 32;
        float vals[4];
        for (int g = warp; g < K / 128; g += warps) {
            stage_vals<SRC>(p, l, inv_norm, norm_w, g, vals);
            quantize_vals(vals, g, s_a8, s_scale, s_gsum);
        }
    }
    __syncthreads();

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int chunks = (K / 4) / 16;
    const int tiles = (m.M + 31) / 32;
    float block_ss = 0.f;  // fused norm partial (EPI 2 with norm_slot >= 0)

    for (int tile = bid; tile < tiles; tile += nblocks) {
        const int row0 = tile * 32 + warp * 4;
        // no early continue: barriers below must stay block-uniform
        const int nrows = row0 < m.M ? min(4, m.M - row0) : 0;
        const uint4* rc[4];
        const __half* rs[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = min(row0 + (r < nrows ? r : 0), m.M - 1);
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
                    const float xn = dst32[row0 + r] + acc[r];
                    dst32[row0 + r] = xn;
                    if (norm_slot >= 0) block_ss += xn * xn;
                } else {
                    dst32[row0 + r] = acc[r];
                }
            }
        }
        __syncthreads();  // s_a8 reused across tiles; keep warps in step
    }

    if (EPI == 2 && norm_slot >= 0) {
        // fold this block's sum(x_new^2) contribution into the next norm
        __shared__ float warp_acc[8];
        const float w = warp_sum(block_ss);
        if ((threadIdx.x & 31) == 0) warp_acc[threadIdx.x >> 5] = w;
        __syncthreads();
        if (threadIdx.x == 0) {
            float t = 0.f;
            for (int i = 0; i < (int)blockDim.x / 32; ++i) t += warp_acc[i];
            atomicAdd(&p.red_scratch[norm_slot], t);
        }
    }
}

// one-shot global quantization for a producer op: each 128-group handled by
// exactly one warp grid-wide (cheap ops recompute in gemv staging instead)
template <int SRC>
__device__ void op_quant_global(const MegaParams& p, const MegaLayer& l, int n,
                                const MegaMat& next, int bid, int nblocks) {
    const int warps_per_block = (int)blockDim.x / 32;
    const int busy = (n / 128 + warps_per_block - 1) / warps_per_block;
    if (bid >= busy) {
        prefetch_l2(p, next, bid - busy, nblocks - busy, 1024);
        return;
    }
    const int warp = (bid * (int)blockDim.x + threadIdx.x) >> 5;
    const int warps_total = min(nblocks, busy) * warps_per_block;
    float vals[4];
    for (int g = warp; g < n / 128; g += warps_total) {
        stage_vals<SRC>(p, &l, 0.f, nullptr, g, vals);
        quantize_vals(vals, g, p.a8, p.a_scale, p.a_gsum);
    }
}

// ---- non-GEMV ops -------------------------------------------------------

// embed: dequant row *d_tok into x, accumulate sum(x^2) into this token's
// first norm slot, and zero the OTHER parity's slot bank for the next token
__device__ void op_embed(const MegaParams& p, int slot0, int other_base,
                         int slots_per_token, int bid, int nblocks) {
    const int row = *p.d_tok;
    const MegaMat& m = p.tok_embd;
    const int gtid = bid * (int)blockDim.x + threadIdx.x;
    const int gsize = nblocks * (int)blockDim.x;
    float ss = 0.f;
    for (int j = gtid; j < p.n_embd; j += gsize) {
        const float d = __half2float(m.scales[(size_t)row * (m.K >> 7) + (j >> 7)]);
        float v;
        if (m.nbits == 2) {
            const uint8_t* rc = m.codes + (size_t)row * (m.K >> 2);
            const int word = j >> 4, r = j & 15, i = r >> 2, b = r & 3;
            uint32_t w;
            memcpy(&w, rc + 4 * word, 4);
            v = (float)((int)((w >> (8 * b + 2 * i)) & 3u) - 1) * d;
        } else {
            const uint8_t* rc = m.codes + (size_t)row * (m.K >> 3);
            const int word = j >> 5, r = j & 31, i = r >> 2, b = r & 3;
            uint32_t w;
            memcpy(&w, rc + 4 * word, 4);
            v = ((w >> (8 * b + i)) & 1u) ? d : -d;
        }
        p.x[j] = v;
        ss += v * v;
    }
    __shared__ float warp_acc[8];
    const float w = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) warp_acc[threadIdx.x >> 5] = w;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < (int)blockDim.x / 32; ++i) t += warp_acc[i];
        atomicAdd(&p.red_scratch[slot0], t);
    }
    for (int j = gtid; j < slots_per_token; j += gsize) {
        p.red_scratch[other_base + j] = 0.f;
    }
}

// conv1d + SiLU over all channels, then L2-norm for q/k heads
__device__ void op_conv_l2(const MegaParams& p, const MegaLayer& l, int bid,
                           int nblocks) {
    const int C = p.conv_channels;
    const int k = p.ssm_conv;
    const int qk_ch = 2 * p.ssm_state * p.ssm_groups;
    __half* buf = p.big_a;
    __shared__ float s_v[128];
    __shared__ float wsum[4];

    for (int tile = bid; tile < C / 128; tile += nblocks) {
        const int c0 = tile * 128;
        float sv = 0.f;
        if (threadIdx.x < 128) {
            const int c = c0 + threadIdx.x;
            float* st = l.conv_state + (size_t)c * (k - 1);
            const __half* wc = l.conv_w + (size_t)c * k;
            const float xnv = __half2float(buf[c]);
            float acc = xnv * __half2float(wc[k - 1]);
            for (int i = 0; i < k - 1; ++i) acc += st[i] * __half2float(wc[i]);
            for (int i = 0; i < k - 2; ++i) st[i] = st[i + 1];
            st[k - 2] = xnv;
            sv = acc / (1.f + expf(-acc));
            s_v[threadIdx.x] = sv;
        }
        __syncthreads();
        const float ss = threadIdx.x < 128 ? sv * sv : 0.f;
        const float wsr = warp_sum(ss);
        if ((threadIdx.x & 31) == 0 && threadIdx.x < 128) {
            wsum[threadIdx.x >> 5] = wsr;
        }
        __syncthreads();
        if (threadIdx.x < 128) {
            if (c0 < qk_ch) {
                const float total = wsum[0] + wsum[1] + wsum[2] + wsum[3];
                const float inv = rsqrtf(fmaxf(total, p.rms_eps));
                buf[c0 + threadIdx.x] = __float2half(sv * inv);
            } else {
                buf[c0 + threadIdx.x] = __float2half(sv);
            }
        }
        __syncthreads();
    }
}

// gated delta net step (block per value head); idle blocks prefetch out_proj
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

    if (bid >= Hv) {
        prefetch_l2(p, l.out_proj, bid - Hv, nblocks - Hv, 3072);
        return;
    }
    for (int h = bid; h < Hv; h += nblocks) {
        const int hk = h % Hk;
        for (int i = threadIdx.x; i < S; i += blockDim.x) {
            s_q[i] = __half2float(qkv[(size_t)hk * S + i]);
            s_k[i] = __half2float(qkv[(size_t)(Hk + hk) * S + i]);
        }
        __syncthreads();

        // ssm_a already stores -exp(A_log)
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

// attention prep: q/k norm+rope, k/v appended into caches at *d_pos
__device__ void op_attn_prep(const MegaParams& p, const MegaLayer& l, int bid,
                             int nblocks) {
    const int D = p.head_dim;
    const int H = p.n_head, Hkv = p.n_head_kv;
    const int pos = *p.d_pos;
    const __half* qg = p.big_a;
    const __half* kk = p.big_a + l.in_rows;
    const __half* vv = kk + Hkv * D;
    __shared__ float s_d[256];
    __shared__ float wsum[8];

    for (int unit = bid; unit < H + 2 * Hkv; unit += nblocks) {
        const bool is_q = unit < H;
        const bool is_k = !is_q && unit < H + Hkv;
        const int h = is_q ? unit : (is_k ? unit - H : unit - H - Hkv);
        if (!is_q && !is_k) {
            __half* vdst = l.v_cache + (size_t)pos * Hkv * D + (size_t)h * D;
            for (int i = threadIdx.x; i < D; i += blockDim.x) {
                vdst[i] = vv[(size_t)h * D + i];
            }
            continue;
        }
        const __half* src = is_q ? qg + (size_t)h * 2 * D : kk + (size_t)h * D;
        const __half* w = is_q ? l.q_norm : l.k_norm;
        float ss = 0.f;
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            const float v = __half2float(src[i]);
            s_d[i] = v;
            ss += v * v;
        }
        const float wsr = warp_sum(ss);
        if ((threadIdx.x & 31) == 0) wsum[threadIdx.x >> 5] = wsr;
        __syncthreads();
        float total = 0.f;
        for (int i = 0; i < (int)blockDim.x / 32; ++i) total += wsum[i];
        const float inv = rsqrtf(total / D + p.rms_eps);
        __syncthreads();
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
                const float x1 =
                    s_d[pair + half_rot] * inv * __half2float(w[pair + half_rot]);
                v = i < half_rot ? x0 * c - x1 * s : x0 * s + x1 * c;
            }
            dst[i] = __float2half(v);
        }
        __syncthreads();
    }
}

// softmax attention decode; idle blocks prefetch out_proj
__device__ void op_attn(const MegaParams& p, const MegaLayer& l, int bid,
                        int nblocks, uint8_t* smem) {
    constexpr int kWarps = 8;
    const int D = p.head_dim;
    const int kVecPerLane = D / 32;
    const int H = p.n_head, Hkv = p.n_head_kv;
    const int ctx_len = *p.d_pos + 1;
    const size_t kv_row = (size_t)Hkv * D;

    float* s_q = reinterpret_cast<float*>(smem);
    float* s_m = s_q + D;
    float* s_l = s_m + kWarps;
    float* s_acc = s_l + kWarps;

    if (bid >= H) {
        prefetch_l2(p, l.out_proj, bid - H, nblocks - H, 3072);
        return;
    }
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
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float bv = -INFINITY;
    int bi = 0;
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

// ---- the megakernel: one launch = one decoded token ---------------------

__device__ __forceinline__ void stamp(const MegaParams& p) {
    if (p.ts && blockIdx.x == 0 && threadIdx.x == 0) {
        p.ts[atomicAdd(p.ts_count, 1)] = (unsigned long long)clock64();
    }
}

__global__ void __launch_bounds__(256)
mega_decode_kernel(MegaParams p) {
    cg::grid_group grid = cg::this_grid();
    extern __shared__ uint8_t smem[];
    const int bid = blockIdx.x;
    const int nb = gridDim.x;

    // double-buffered norm slots keyed by token parity: this token uses its
    // own bank while embed zeroes the other bank for the next token
    const int spt = 2 * p.n_layer + 2;  // slots per token
    const int parity = (int)(*p.d_step & 1);
    const int base = parity * spt;
    const int other = (1 - parity) * spt;
    float* rs = p.red_scratch;

    stamp(p);
    op_embed(p, base, other, spt, bid, nb);
    grid.sync();
    stamp(p);

    for (int il = 0; il < p.n_layer; ++il) {
        const MegaLayer& l = p.layers[il];
        const float inv1 = rsqrtf(rs[base + 2 * il] / p.n_embd + p.rms_eps);
        op_gemv<1, SRC_NORM_X>(p, &l, l.proj, nullptr, p.big_a, -1, inv1,
                               l.attn_norm, smem, bid, nb);
        grid.sync();
        stamp(p);
        if (l.recurrent) {
            op_conv_l2(p, l, bid, nb);
            grid.sync();
            stamp(p);
            op_gdn(p, l, bid, nb);
            grid.sync();
            stamp(p);
            op_quant_global<SRC_GDN_GATE>(p, l, p.ssm_dt_rank * p.head_v_dim,
                                          l.out_proj, bid, nb);
        } else {
            op_attn_prep(p, l, bid, nb);
            grid.sync();
            stamp(p);
            op_attn(p, l, bid, nb, smem);
            grid.sync();
            stamp(p);
            op_quant_global<SRC_ATTN_GATE>(p, l, p.n_head * p.head_dim,
                                           l.out_proj, bid, nb);
        }
        grid.sync();
        stamp(p);
        op_gemv<2, SRC_FROM_A8>(p, &l, l.out_proj, p.x, nullptr,
                                base + 2 * il + 1, 0.f, nullptr, smem, bid, nb);
        grid.sync();
        stamp(p);
        const float inv2 = rsqrtf(rs[base + 2 * il + 1] / p.n_embd + p.rms_eps);
        op_gemv<1, SRC_NORM_X>(p, &l, l.gate_up, nullptr, p.big_a, -1, inv2,
                               l.post_norm, smem, bid, nb);
        grid.sync();
        stamp(p);
        op_quant_global<SRC_SWIGLU>(p, l, p.n_ff, l.down, bid, nb);
        grid.sync();
        stamp(p);
        const int next_slot =
            il + 1 < p.n_layer ? base + 2 * (il + 1) : base + 2 * p.n_layer;
        op_gemv<2, SRC_FROM_A8>(p, &l, l.down, p.x, nullptr, next_slot, 0.f,
                                nullptr, smem, bid, nb);
        grid.sync();
        stamp(p);
    }

    const float inv_h = rsqrtf(rs[base + 2 * p.n_layer] / p.n_embd + p.rms_eps);
    op_gemv<0, SRC_NORM_X>(p, nullptr, p.lm_head, p.y32, nullptr, -1, inv_h,
                           p.output_norm, smem, bid, nb);
    grid.sync();
    stamp(p);
    op_argmax_partial(p, bid, nb);
    grid.sync();
    op_argmax_final_bump(p, nb);
    stamp(p);
}

}  // namespace

bool mega_decode_launch(const MegaParams& p, cudaStream_t stream) {
    static int grid_size = 0;
    static int smem_bytes = 0;
    if (grid_size == 0) {
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
