// bt-run — bonsai-turbo decode engine.
//
// Modes:
//   eager  (default): one kernel per op, host-driven control flow
//   --graph         : the whole decode step (embed -> 64 layers -> logits ->
//                     argmax -> state bump) is captured once as a CUDA graph
//                     and replayed with a single launch per token
//
// Usage:
//   bt-run --model X.gguf (--ids 1,2,3 | --ids-file f) [--n 64] [--graph]
//          [--logits-out prefix] [--bench] [--eos ID]
//
// Token ids in, token ids out (tokenize with any GGUF-compatible tokenizer;
// the repro harness uses the vendor fork's llama-tokenize). Greedy only.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../src/cuda/attn.h"
#include "../src/cuda/gdn.h"
#include "../src/cuda/gemv.h"
#include "../src/cuda/mega.h"
#include "../src/cuda/ops.h"
#include "../src/model.h"

using namespace bt;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t err_ = (expr);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                         cudaGetErrorString(err_), __FILE__, __LINE__);         \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

namespace {

constexpr int kRingCap = 8192;

struct Runtime {
    Model m;
    int max_ctx = 8192;
    int eos = -1;
    bool graph_mode = false;
    bool mega_mode = false;
    cudaStream_t st = nullptr;

    // activation buffers (f16 unless noted)
    float* x;  // fp32 residual stream
    __half *xn, *big_a, *big_b, *q, *k, *v, *attn_out, *gdn_out;
    float* y32;
    int8_t* a8;
    float* a_scale;
    int32_t* a_gsum;
    int quant_k = 0;

    // device-resident control state (graph/mega modes)
    int32_t *d_pos, *d_step, *d_tok, *d_ring;
    // megakernel plumbing
    float* red_scratch = nullptr;
    float* amax_v = nullptr;
    int32_t* amax_i = nullptr;
    MegaLayer* d_mlayers = nullptr;
    MegaParams mp{};

    // per-layer persistent state
    std::vector<float*> gdn_state;
    std::vector<float*> conv_state;
    std::vector<__half*> k_cache, v_cache;
    int pos = 0;  // host mirror (eager mode)

    cudaGraphExec_t graph_exec = nullptr;

    int qkv_dim() const {
        return 2 * m.hp.ssm_state * m.hp.ssm_groups + m.hp.ssm_inner;
    }

    void alloc() {
        const HParams& hp = m.hp;
        CUDA_CHECK(cudaStreamCreate(&st));
        const int big = std::max({2 * hp.n_ff, qkv_dim() + hp.ssm_inner + 256,
                                  2 * hp.head_dim * hp.n_head + hp.n_head_kv * hp.head_dim * 2,
                                  hp.n_embd});
        const int max_k = std::max({hp.n_embd, hp.ssm_inner, hp.n_ff,
                                    hp.head_dim * hp.n_head});
        CUDA_CHECK(cudaMalloc(&x, hp.n_embd * 4));
        CUDA_CHECK(cudaMalloc(&xn, hp.n_embd * 2));
        CUDA_CHECK(cudaMalloc(&big_a, (size_t)big * 2));
        CUDA_CHECK(cudaMalloc(&big_b, (size_t)big * 2));
        CUDA_CHECK(cudaMalloc(&q, (size_t)hp.head_dim * hp.n_head * 2));
        CUDA_CHECK(cudaMalloc(&k, (size_t)hp.head_dim * hp.n_head * 2));
        CUDA_CHECK(cudaMalloc(&v, (size_t)hp.head_dim * hp.n_head * 2));
        const int attn_out_elems = std::max(hp.head_dim * hp.n_head, hp.ssm_inner);
        CUDA_CHECK(cudaMalloc(&attn_out, (size_t)attn_out_elems * 2));
        CUDA_CHECK(cudaMalloc(&gdn_out, (size_t)std::max(hp.ssm_inner, 256) * 2));
        CUDA_CHECK(cudaMalloc(&y32, (size_t)hp.vocab * 4));
        CUDA_CHECK(cudaMalloc(&a8, (size_t)max_k));
        CUDA_CHECK(cudaMalloc(&a_scale, (size_t)(max_k / 128) * 4));
        CUDA_CHECK(cudaMalloc(&a_gsum, (size_t)(max_k / 64) * 4));
        CUDA_CHECK(cudaMalloc(&d_pos, 4));
        CUDA_CHECK(cudaMalloc(&d_step, 4));
        CUDA_CHECK(cudaMalloc(&d_tok, 4));
        CUDA_CHECK(cudaMalloc(&d_ring, kRingCap * 4));
        CUDA_CHECK(cudaMalloc(&red_scratch, (size_t)(2 * hp.n_layer + 4) * 4));
        CUDA_CHECK(cudaMalloc(&amax_v, 4096 * 4));
        CUDA_CHECK(cudaMalloc(&amax_i, 4096 * 4));

        for (int il = 0; il < hp.n_layer; ++il) {
            if (m.layers[(size_t)il].recurrent) {
                float* s;
                const size_t state_elems =
                    (size_t)hp.ssm_dt_rank * hp.head_v_dim * hp.head_v_dim;
                CUDA_CHECK(cudaMalloc(&s, state_elems * 4));
                gdn_state.push_back(s);
                float* c;
                CUDA_CHECK(cudaMalloc(&c, (size_t)hp.conv_channels * (hp.ssm_conv - 1) * 4));
                conv_state.push_back(c);
                k_cache.push_back(nullptr);
                v_cache.push_back(nullptr);
            } else {
                gdn_state.push_back(nullptr);
                conv_state.push_back(nullptr);
                __half *kc, *vc;
                const size_t kv = (size_t)max_ctx * hp.n_head_kv * hp.head_dim * 2;
                CUDA_CHECK(cudaMalloc(&kc, kv));
                CUDA_CHECK(cudaMalloc(&vc, kv));
                k_cache.push_back(kc);
                v_cache.push_back(vc);
            }
        }
        reset_sequence();
        build_mega();
    }

    static MegaMat as_mega(const Mat& m) {
        return MegaMat{m.codes, m.scales, m.M, m.K, m.nbits};
    }

    void build_mega() {
        const HParams& hp = m.hp;
        std::vector<MegaLayer> host((size_t)hp.n_layer);
        for (int il = 0; il < hp.n_layer; ++il) {
            const Layer& l = m.layers[(size_t)il];
            MegaLayer& d = host[(size_t)il];
            d.recurrent = l.recurrent ? 1 : 0;
            d.attn_norm = l.attn_norm;
            d.post_norm = l.attn_post_norm;
            d.gate_up = as_mega(l.gate_up);
            d.down = as_mega(l.down);
            d.mlp_gate_rows = l.gate_rows;
            if (l.recurrent) {
                d.proj = as_mega(l.gdn_fused);
                d.out_proj = as_mega(l.ssm_out);
                d.in_rows = l.ssm_in_rows;
                d.gate_rows = l.ssm_gate_rows;
                d.ssm_norm = l.ssm_norm;
                d.ssm_a = l.ssm_a;
                d.ssm_dt = l.ssm_dt;
                d.conv_w = l.conv_w;
                d.gdn_state = gdn_state[(size_t)il];
                d.conv_state = conv_state[(size_t)il];
            } else {
                d.proj = as_mega(l.qkv_fused);
                d.out_proj = as_mega(l.wo);
                d.in_rows = l.wq_rows;
                d.gate_rows = 0;
                d.q_norm = l.q_norm;
                d.k_norm = l.k_norm;
                d.k_cache = k_cache[(size_t)il];
                d.v_cache = v_cache[(size_t)il];
            }
        }
        CUDA_CHECK(cudaMalloc(&d_mlayers, host.size() * sizeof(MegaLayer)));
        CUDA_CHECK(cudaMemcpy(d_mlayers, host.data(), host.size() * sizeof(MegaLayer),
                              cudaMemcpyHostToDevice));

        mp.layers = d_mlayers;
        mp.tok_embd = as_mega(m.tok_embd);
        mp.lm_head = as_mega(m.lm_head);
        mp.output_norm = m.output_norm;
        mp.n_layer = hp.n_layer;
        mp.n_embd = hp.n_embd;
        mp.n_ff = hp.n_ff;
        mp.vocab = hp.vocab;
        mp.n_head = hp.n_head;
        mp.n_head_kv = hp.n_head_kv;
        mp.head_dim = hp.head_dim;
        mp.n_rot = hp.n_rot;
        mp.ssm_state = hp.ssm_state;
        mp.ssm_groups = hp.ssm_groups;
        mp.ssm_dt_rank = hp.ssm_dt_rank;
        mp.head_v_dim = hp.head_v_dim;
        mp.ssm_conv = hp.ssm_conv;
        mp.conv_channels = hp.conv_channels;
        mp.rms_eps = hp.rms_eps;
        mp.rope_base = hp.rope_base;
        mp.x = x;
        mp.xn = xn;
        mp.big_a = big_a;
        mp.big_b = big_b;
        mp.q_buf = q;
        mp.attn_out = attn_out;
        mp.y32 = y32;
        mp.a8 = a8;
        mp.a_scale = a_scale;
        mp.a_gsum = a_gsum;
        mp.red_scratch = red_scratch;
        mp.amax_v = amax_v;
        mp.amax_i = amax_i;
        mp.d_pos = d_pos;
        mp.d_step = d_step;
        mp.d_tok = d_tok;
        mp.d_ring = d_ring;
        mp.ring_cap = kRingCap;
    }

    void reset_sequence() {
        const HParams& hp = m.hp;
        for (int il = 0; il < hp.n_layer; ++il) {
            if (m.layers[(size_t)il].recurrent) {
                CUDA_CHECK(cudaMemset(gdn_state[(size_t)il], 0,
                                      (size_t)hp.ssm_dt_rank * hp.head_v_dim *
                                          hp.head_v_dim * 4));
                CUDA_CHECK(cudaMemset(conv_state[(size_t)il], 0,
                                      (size_t)hp.conv_channels * (hp.ssm_conv - 1) * 4));
            }
        }
        CUDA_CHECK(cudaMemset(d_pos, 0, 4));
        CUDA_CHECK(cudaMemset(d_step, 0, 4));
        pos = 0;  // attn caches need no clear: reads cover [0, pos) only
    }

    void quant(const __half* src, int K) {
        quant_acts_launch(src, K, a8, a_scale, a_gsum, st);
        quant_k = K;
    }

    void mv16(const Mat& mat, const __half* src, float* out32) {
        if (mat.nbits == 16) {
            gemv_f16_launch(mat.dense, src, out32, mat.M, mat.K, st);
        } else {
            if (quant_k != mat.K) {
                std::fprintf(stderr, "quant buffer K mismatch %d vs %d\n", quant_k, mat.K);
                std::exit(1);
            }
            gemv_launch(mat.nbits, mat.codes, mat.scales, a8, a_scale, a_gsum, out32,
                        mat.M, mat.K, st);
        }
    }

    void check_quant(const Mat& mat) {
        if (mat.nbits == 16 || quant_k != mat.K) {
            std::fprintf(stderr, "fused epilogue needs quant mat, K %d vs %d\n",
                         quant_k, mat.K);
            std::exit(1);
        }
    }

    // projection: GEMV writing f16 directly (cast fused into the epilogue)
    void mv_f16(const Mat& mat, __half* y16) {
        check_quant(mat);
        gemv_f16out_launch(mat.nbits, mat.codes, mat.scales, a8, a_scale, a_gsum,
                           y16, mat.M, mat.K, st);
    }

    // output projection: GEMV accumulating into the fp32 residual stream
    void mv_add(const Mat& mat, float* xdst) {
        check_quant(mat);
        gemv_addinto_launch(mat.nbits, mat.codes, mat.scales, a8, a_scale, a_gsum,
                            xdst, mat.M, mat.K, st);
    }
};

void probe_vec(Runtime& rt, const char* name, int il, const __half* ptr, int n);
void probe(Runtime& rt, const char* tag, int il);

void gdn_layer(Runtime& rt, int il) {
    const HParams& hp = rt.m.hp;
    const Layer& l = rt.m.layers[(size_t)il];
    const int Sk = hp.ssm_state, Hk = hp.ssm_groups;
    const int Sv = hp.head_v_dim, Hv = hp.ssm_dt_rank;

    rmsnorm_quant_f32_launch(rt.x, l.attn_norm, hp.n_embd, rt.xn, rt.a8,
                             rt.a_scale, rt.a_gsum, hp.rms_eps, rt.st);
    rt.quant_k = hp.n_embd;
    probe_vec(rt, "attn_norm", il, rt.xn, hp.n_embd);

    // fused [qkv_mixed | z | alpha | beta] projection, f16 epilogue
    rt.mv_f16(l.gdn_fused, rt.big_a);
    __half* qkv_mixed = rt.big_a;
    __half* z = rt.big_a + l.ssm_in_rows;
    __half* alpha_raw = z + l.ssm_gate_rows;
    __half* beta_raw = alpha_raw + Hv;
    probe_vec(rt, "qkv_mixed", il, qkv_mixed, rt.qkv_dim());
    probe_vec(rt, "z", il, z, hp.ssm_inner);

    conv1d_step_launch(qkv_mixed, l.conv_w, rt.conv_state[(size_t)il], qkv_mixed,
                       rt.qkv_dim(), hp.ssm_conv, rt.st);
    probe_vec(rt, "conv_output_silu", il, qkv_mixed, rt.qkv_dim());

    __half* qc = qkv_mixed;
    __half* kc = qkv_mixed + (size_t)Sk * Hk;
    __half* vc = qkv_mixed + (size_t)2 * Sk * Hk;
    l2norm_heads_launch(qc, qc, Hk, Sk, hp.rms_eps, rt.st);
    l2norm_heads_launch(kc, kc, Hk, Sk, hp.rms_eps, rt.st);
    probe_vec(rt, "q_conv_l2", il, qc, Sk * Hk);

    // the fork's GDN op scales its output by 1/sqrt(S_v) internally
    gdn_decode_launch(qc, kc, vc, alpha_raw, beta_raw, l.ssm_a, l.ssm_dt,
                      rt.gdn_state[(size_t)il], rt.attn_out, Hv, Hk, Sv,
                      1.f / sqrtf((float)Sv), rt.st);
    probe_vec(rt, "gdn_core_out", il, rt.attn_out, hp.ssm_inner);

    rmsnorm_heads_launch(rt.attn_out, l.ssm_norm, rt.attn_out, Hv, Sv, hp.rms_eps, rt.st);
    gate_mul_quant_launch(0, z, rt.attn_out, hp.ssm_inner, rt.attn_out, rt.a8,
                          rt.a_scale, rt.a_gsum, rt.st);
    rt.quant_k = hp.ssm_inner;
    probe_vec(rt, "final_output", il, rt.attn_out, hp.ssm_inner);

    rt.mv_add(l.ssm_out, rt.x);
}

void attn_layer(Runtime& rt, int il) {
    const HParams& hp = rt.m.hp;
    const Layer& l = rt.m.layers[(size_t)il];
    const int D = hp.head_dim, H = hp.n_head, Hkv = hp.n_head_kv;

    rmsnorm_quant_f32_launch(rt.x, l.attn_norm, hp.n_embd, rt.xn, rt.a8,
                             rt.a_scale, rt.a_gsum, hp.rms_eps, rt.st);
    rt.quant_k = hp.n_embd;

    // fused [q+gate | k | v] projection, f16 epilogue
    rt.mv_f16(l.qkv_fused, rt.big_a);
    __half* qg = rt.big_a;                       // [H][2D] interleaved q|gate
    __half* kk = rt.big_a + l.wq_rows;
    __half* vv = kk + l.wk_rows;

    gather_heads_launch(qg, rt.q, H, D, 2 * D, 0, rt.st);
    gather_heads_launch(qg, rt.big_b, H, D, 2 * D, D, rt.st);  // gate

    rmsnorm_heads_launch(rt.q, l.q_norm, rt.q, H, D, hp.rms_eps, rt.st);
    rmsnorm_heads_launch(kk, l.k_norm, kk, Hkv, D, hp.rms_eps, rt.st);

    const int kv_elems = Hkv * D;
    if (rt.graph_mode) {
        rope_neox_dev_launch(rt.q, H, D, hp.n_rot, rt.d_pos, hp.rope_base, rt.st);
        rope_neox_dev_launch(kk, Hkv, D, hp.n_rot, rt.d_pos, hp.rope_base, rt.st);
        kv_append_dev_launch(kk, vv, rt.k_cache[(size_t)il], rt.v_cache[(size_t)il],
                             kv_elems, rt.d_pos, rt.st);
        attn_decode_dev_launch(rt.q, rt.k_cache[(size_t)il], rt.v_cache[(size_t)il],
                               rt.attn_out, H, Hkv, D, rt.d_pos,
                               1.f / sqrtf((float)D), rt.st);
    } else {
        rope_neox_launch(rt.q, H, D, hp.n_rot, rt.pos, hp.rope_base, rt.st);
        rope_neox_launch(kk, Hkv, D, hp.n_rot, rt.pos, hp.rope_base, rt.st);
        CUDA_CHECK(cudaMemcpyAsync(rt.k_cache[(size_t)il] + (size_t)rt.pos * kv_elems,
                                   kk, (size_t)kv_elems * 2,
                                   cudaMemcpyDeviceToDevice, rt.st));
        CUDA_CHECK(cudaMemcpyAsync(rt.v_cache[(size_t)il] + (size_t)rt.pos * kv_elems,
                                   vv, (size_t)kv_elems * 2,
                                   cudaMemcpyDeviceToDevice, rt.st));
        attn_decode_launch(rt.q, rt.k_cache[(size_t)il], rt.v_cache[(size_t)il],
                           rt.attn_out, H, Hkv, D, rt.pos + 1,
                           1.f / sqrtf((float)D), rt.st);
    }

    gate_mul_quant_launch(1, rt.attn_out, rt.big_b, H * D, rt.attn_out, rt.a8,
                          rt.a_scale, rt.a_gsum, rt.st);
    rt.quant_k = H * D;
    rt.mv_add(l.wo, rt.x);
}

void mlp(Runtime& rt, int il) {
    const HParams& hp = rt.m.hp;
    const Layer& l = rt.m.layers[(size_t)il];
    rmsnorm_quant_f32_launch(rt.x, l.attn_post_norm, hp.n_embd, rt.xn, rt.a8,
                             rt.a_scale, rt.a_gsum, hp.rms_eps, rt.st);
    rt.quant_k = hp.n_embd;
    rt.mv_f16(l.gate_up, rt.big_a);
    gate_mul_quant_launch(0, rt.big_a, rt.big_a + l.gate_rows, hp.n_ff, nullptr,
                          rt.a8, rt.a_scale, rt.a_gsum, rt.st);
    rt.quant_k = hp.n_ff;
    rt.mv_add(l.down, rt.x);
}

// BT_PROBE=1: print ||x|| after embed and each layer (first decode step only)
// BT_PROBE=2: additionally dump sub-layer tensor sums for layer 0 (matches the
// vendor eval-callback's cb() names for direct comparison)
int probe_level() {
    static const int lvl = getenv("BT_PROBE") ? atoi(getenv("BT_PROBE")) : 0;
    return lvl;
}
static int g_probe_done = 0;

void probe_vec(Runtime& rt, const char* name, int il, const __half* ptr, int n) {
    if (probe_level() < 2 || g_probe_done || il != 0) return;
    std::vector<__half> h((size_t)n);
    CUDA_CHECK(cudaStreamSynchronize(rt.st));
    CUDA_CHECK(cudaMemcpy(h.data(), ptr, (size_t)n * 2, cudaMemcpyDeviceToHost));
    double sum = 0;
    for (__half v : h) sum += (double)__half2float(v);
    std::fprintf(stderr, "probe2 %-20s sum = %+.6f  [%+.5f %+.5f %+.5f]\n", name,
                 sum, __half2float(h[0]), __half2float(h[1]), __half2float(h[2]));
}

void probe(Runtime& rt, const char* tag, int il) {
    if (!probe_level() || g_probe_done) return;
    if (il < 0 && std::strcmp(tag, "done") == 0) {
        g_probe_done++;
        return;
    }
    std::vector<float> h((size_t)rt.m.hp.n_embd);
    CUDA_CHECK(cudaStreamSynchronize(rt.st));
    CUDA_CHECK(cudaMemcpy(h.data(), rt.x, h.size() * 4, cudaMemcpyDeviceToHost));
    double ss = 0, sum = 0;
    for (float v : h) {
        const double f = v;
        ss += f * f;
        sum += f;
    }
    std::fprintf(stderr, "probe %-8s %3d ||x|| = %.6f sum = %+.6f\n", tag, il,
                 std::sqrt(ss), sum);
}

// layer stack + head; embed comes from host `token` (eager) or *d_tok (graph)
void decode_body(Runtime& rt, int token) {
    const HParams& hp = rt.m.hp;
    // fp32 residual stream: quantized embeddings only (true for all Bonsai packs)
    if (rt.graph_mode) {
        dequant_row_f32_dev_launch(rt.m.tok_embd.nbits, rt.m.tok_embd.codes,
                                   rt.m.tok_embd.scales, rt.d_tok, hp.n_embd, rt.x, rt.st);
    } else {
        dequant_row_f32_launch(rt.m.tok_embd.nbits, rt.m.tok_embd.codes,
                               rt.m.tok_embd.scales, token, hp.n_embd, rt.x, rt.st);
    }
    probe(rt, "embed", -1);
    for (int il = 0; il < hp.n_layer; ++il) {
        if (rt.m.layers[(size_t)il].recurrent) {
            gdn_layer(rt, il);
            probe(rt, "gdn", il);
        } else {
            attn_layer(rt, il);
            probe(rt, "attn", il);
        }
        mlp(rt, il);
        probe(rt, "mlp", il);
    }
    probe(rt, "done", -1);
    rmsnorm_quant_f32_launch(rt.x, rt.m.output_norm, hp.n_embd, rt.xn, rt.a8,
                             rt.a_scale, rt.a_gsum, hp.rms_eps, rt.st);
    rt.quant_k = hp.n_embd;
    rt.mv16(rt.m.lm_head, rt.xn, rt.y32);
}

void decode_eager(Runtime& rt, int token) {
    decode_body(rt, token);
    rt.pos++;
}

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> ids;
    const char* p = s.c_str();
    while (*p && *p != '\n') {
        ids.push_back((int)std::strtol(p, (char**)&p, 10));
        if (*p == ',') ++p;
    }
    return ids;
}

// one prompt: feed ids, generate n_gen greedy tokens, optionally dump logits
void run_prompt(Runtime& rt, const std::vector<int>& prompt, int n_gen,
                const std::string& logits_path, bool bench) {
    const HParams& hp = rt.m.hp;
    rt.reset_sequence();
    FILE* lf = logits_path.empty() ? nullptr : std::fopen(logits_path.c_str(), "wb");
    std::vector<float> logits((size_t)hp.vocab);

    // prompt phase: mega mode feeds the megakernel token by token; otherwise
    // eager (graph capture only covers steady-state decode)
    if (rt.mega_mode) {
        for (int id : prompt) {
            CUDA_CHECK(cudaMemcpyAsync(rt.d_tok, &id, 4, cudaMemcpyHostToDevice, rt.st));
            if (!mega_decode_launch(rt.mp, rt.st)) std::exit(1);
        }
        // bump advanced d_step during the prompt; generation records from 0
        CUDA_CHECK(cudaStreamSynchronize(rt.st));
        CUDA_CHECK(cudaMemset(rt.d_step, 0, 4));
        rt.pos = (int)prompt.size();
    } else {
        const bool want_graph = rt.graph_mode;
        rt.graph_mode = false;
        for (int id : prompt) decode_eager(rt, id);
        rt.graph_mode = want_graph;
    }

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    std::printf("generated:");
    int n_done = 0;

    if (rt.mega_mode) {
        int32_t first = 0;
        CUDA_CHECK(cudaMemcpy(&first, rt.d_tok, 4, cudaMemcpyDeviceToHost));
        std::printf(" %d", first);
        n_done = 1;

        CUDA_CHECK(cudaEventRecord(t0, rt.st));
        std::vector<int32_t> ring(kRingCap);
        int launched = 0;
        bool done = first == rt.eos;
        while (n_done < n_gen && !done) {
            const int todo = std::min(16, n_gen - n_done);
            for (int i = 0; i < todo; ++i) {
                if (!mega_decode_launch(rt.mp, rt.st)) std::exit(1);
                ++launched;
            }
            CUDA_CHECK(cudaStreamSynchronize(rt.st));
            CUDA_CHECK(cudaMemcpy(ring.data(), rt.d_ring, (size_t)launched * 4,
                                  cudaMemcpyDeviceToHost));
            while (n_done - 1 < launched && n_done < n_gen) {
                const int32_t tok = ring[(size_t)(n_done - 1)];
                std::printf(" %d", tok);
                ++n_done;
                if (tok == rt.eos) {
                    done = true;
                    break;
                }
            }
        }
        CUDA_CHECK(cudaEventRecord(t1, rt.st));
        rt.pos += launched;
    } else if (!rt.graph_mode) {
        CUDA_CHECK(cudaEventRecord(t0, rt.st));
        int32_t tok = 0;
        for (int i = 0; i < n_gen; ++i) {
            if (lf) {
                CUDA_CHECK(cudaMemcpy(logits.data(), rt.y32, (size_t)hp.vocab * 4,
                                      cudaMemcpyDeviceToHost));
                std::fwrite(logits.data(), 4, (size_t)hp.vocab, lf);
            }
            argmax_launch(rt.y32, hp.vocab, rt.d_tok, rt.st);
            CUDA_CHECK(cudaMemcpy(&tok, rt.d_tok, 4, cudaMemcpyDeviceToHost));
            std::printf(" %d", tok);
            ++n_done;
            if (tok == rt.eos) break;
            decode_eager(rt, tok);
        }
        CUDA_CHECK(cudaEventRecord(t1, rt.st));
    } else {
        // seed device state: pos = prompt length, token = argmax(logits)
        CUDA_CHECK(cudaMemcpy(rt.d_pos, &rt.pos, 4, cudaMemcpyHostToDevice));
        argmax_launch(rt.y32, hp.vocab, rt.d_tok, rt.st);

        if (!rt.graph_exec) {
            cudaGraph_t graph;
            CUDA_CHECK(cudaStreamBeginCapture(rt.st, cudaStreamCaptureModeGlobal));
            decode_body(rt, /*token unused*/ 0);
            argmax_launch(rt.y32, hp.vocab, rt.d_tok, rt.st);
            step_bump_launch(rt.d_pos, rt.d_step, rt.d_ring, kRingCap, rt.d_tok, rt.st);
            CUDA_CHECK(cudaStreamEndCapture(rt.st, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&rt.graph_exec, graph, nullptr, nullptr, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
            std::fprintf(stderr, "captured decode graph\n");
        }

        // token 0 comes from the prompt logits, mirroring eager mode; each
        // graph launch then consumes *d_tok and appends the next token to ring
        int32_t first = 0;
        CUDA_CHECK(cudaStreamSynchronize(rt.st));
        CUDA_CHECK(cudaMemcpy(&first, rt.d_tok, 4, cudaMemcpyDeviceToHost));
        std::printf(" %d", first);
        n_done = 1;

        CUDA_CHECK(cudaEventRecord(t0, rt.st));
        std::vector<int32_t> ring(kRingCap);
        int launched = 0;
        bool done = first == rt.eos;
        while (n_done < n_gen && !done) {
            const int todo = std::min(16, n_gen - n_done);
            for (int i = 0; i < todo; ++i) {
                CUDA_CHECK(cudaGraphLaunch(rt.graph_exec, rt.st));
                ++launched;
            }
            CUDA_CHECK(cudaStreamSynchronize(rt.st));
            CUDA_CHECK(cudaMemcpy(ring.data(), rt.d_ring, (size_t)launched * 4,
                                  cudaMemcpyDeviceToHost));
            while (n_done - 1 < launched && n_done < n_gen) {
                const int32_t tok = ring[(size_t)(n_done - 1)];
                std::printf(" %d", tok);
                ++n_done;
                if (tok == rt.eos) {
                    done = true;
                    break;
                }
            }
        }
        CUDA_CHECK(cudaEventRecord(t1, rt.st));
        rt.pos += launched;
    }

    CUDA_CHECK(cudaEventSynchronize(t1));
    std::printf("\n");
    if (lf) std::fclose(lf);

    if (bench) {
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        std::printf("decode: %d tokens in %.1f ms = %.2f tok/s%s\n", n_done, ms,
                    1000.f * n_done / ms,
                    logits_path.empty() ? "" : " (WARNING: includes logit dumps)");
    }
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, ids_str, ids_file, logits_out;
    int n_gen = 32;
    int eos = -1;
    int ctx = 8192;
    bool bench = false, graph = false, mega = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--ids" && i + 1 < argc) ids_str = argv[++i];
        else if (a == "--ids-file" && i + 1 < argc) ids_file = argv[++i];
        else if (a == "--n" && i + 1 < argc) n_gen = std::atoi(argv[++i]);
        else if (a == "--logits-out" && i + 1 < argc) logits_out = argv[++i];
        else if (a == "--bench") bench = true;
        else if (a == "--graph") graph = true;
        else if (a == "--mega") mega = true;
        else if (a == "--eos" && i + 1 < argc) eos = std::atoi(argv[++i]);
        else if (a == "--ctx" && i + 1 < argc) ctx = std::atoi(argv[++i]);
    }
    if (model_path.empty() || (ids_str.empty() && ids_file.empty())) {
        std::fprintf(stderr,
                     "usage: bt-run --model X.gguf (--ids 1,2,3 | --ids-file f) "
                     "[--n 64] [--graph] [--logits-out prefix] [--bench] [--eos ID]\n");
        return 2;
    }

    Runtime rt;
    rt.max_ctx = ctx;
    std::fprintf(stderr, "loading %s ...\n", model_path.c_str());
    rt.m.load(model_path);
    const HParams& hp = rt.m.hp;
    std::fprintf(stderr,
                 "arch ok: %d layers (%d attn), n_embd %d, heads %d/%d x %d, "
                 "ff %d, vocab %d, gdn: Hk %d Sk %d Hv %d Sv %d conv %d(k)x%d(C)\n",
                 hp.n_layer,
                 (int)std::count(hp.recurrent.begin(), hp.recurrent.end(), (uint8_t)0),
                 hp.n_embd, hp.n_head, hp.n_head_kv, hp.head_dim, hp.n_ff, hp.vocab,
                 hp.ssm_groups, hp.ssm_state, hp.ssm_dt_rank, hp.head_v_dim,
                 hp.ssm_conv, hp.conv_channels);
    rt.alloc();
    rt.eos = eos;
    rt.graph_mode = graph;
    rt.mega_mode = mega;

    if (graph && !logits_out.empty()) {
        std::fprintf(stderr, "--graph and --logits-out are mutually exclusive\n");
        return 2;
    }

    if (!ids_str.empty()) {
        run_prompt(rt, parse_ids(ids_str), n_gen, logits_out, bench);
        return 0;
    }

    FILE* f = std::fopen(ids_file.c_str(), "r");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", ids_file.c_str());
        return 1;
    }
    char line[65536];
    int idx = 0;
    while (std::fgets(line, sizeof(line), f)) {
        const std::vector<int> ids = parse_ids(line);
        if (ids.empty()) continue;
        const std::string lp =
            logits_out.empty() ? "" : logits_out + "." + std::to_string(idx) + ".bin";
        run_prompt(rt, ids, n_gen, lp, bench);
        ++idx;
    }
    std::fclose(f);
    return 0;
}
