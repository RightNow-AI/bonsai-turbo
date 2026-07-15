// bt-run — bonsai-turbo decode engine (v1: one kernel per op; fusion follows).
//
// Usage:
//   bt-run --model X.gguf --ids 1,2,3 --n 64 [--logits-out logits.bin] [--bench]
//
// Token ids in, token ids out (tokenize with any GGUF-compatible tokenizer;
// the repro harness uses the vendor fork's llama-tokenize). Greedy sampling
// only — this engine exists to measure decode speed and logit parity.
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

struct Runtime {
    Model m;
    int max_ctx = 4096;

    // activation buffers (f16 unless noted)
    __half *x, *xn, *big_a, *big_b, *q, *k, *v, *attn_out, *gdn_out;
    float* y32;  // widest GEMV output (vocab)
    int8_t* a8;
    float* a_scale;
    int32_t* a_gsum;
    int quant_k = 0;  // K of the currently quantized activation

    // per-layer persistent state
    std::vector<float*> gdn_state;   // [H_v][S][S]
    std::vector<float*> conv_state;  // [C][k-1]
    std::vector<__half*> k_cache, v_cache;
    int pos = 0;

    int qkv_dim() const {
        return 2 * m.hp.ssm_state * m.hp.ssm_groups + m.hp.ssm_inner;
    }

    void alloc() {
        const HParams& hp = m.hp;
        const int big = std::max({hp.n_ff, qkv_dim(), 2 * hp.head_dim * hp.n_head,
                                  hp.ssm_inner, hp.n_embd});
        const int max_k = std::max({hp.n_embd, hp.ssm_inner, hp.n_ff,
                                    hp.head_dim * hp.n_head});
        CUDA_CHECK(cudaMalloc(&x, hp.n_embd * 2));
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

        for (int il = 0; il < hp.n_layer; ++il) {
            if (m.layers[(size_t)il].recurrent) {
                float* s;
                const size_t state_elems =
                    (size_t)hp.ssm_dt_rank * hp.head_v_dim * hp.head_v_dim;
                CUDA_CHECK(cudaMalloc(&s, state_elems * 4));
                CUDA_CHECK(cudaMemset(s, 0, state_elems * 4));
                gdn_state.push_back(s);
                float* c;
                const size_t conv_elems = (size_t)hp.conv_channels * (hp.ssm_conv - 1);
                CUDA_CHECK(cudaMalloc(&c, conv_elems * 4));
                CUDA_CHECK(cudaMemset(c, 0, conv_elems * 4));
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
    }

    void quant(const __half* src, int K) {
        quant_acts_launch(src, K, a8, a_scale, a_gsum, 0);
        quant_k = K;
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
        pos = 0;  // attn caches need no clear: reads cover [0, pos) only
    }

    // y32 <- mat @ (currently quantized activation); f16 result optionally
    void mv(const Mat& mat, float* out32) {
        if (mat.nbits == 16) {
            std::fprintf(stderr, "f16 matrix hit quant path\n");
            std::exit(1);
        }
        gemv_launch(mat.nbits, mat.codes, mat.scales, a8, a_scale, a_gsum, out32,
                    mat.M, mat.K, 0);
    }

    void mv16(const Mat& mat, const __half* src, float* out32) {
        if (mat.nbits == 16) {
            gemv_f16_launch(mat.dense, src, out32, mat.M, mat.K, 0);
        } else {
            if (quant_k != mat.K) {
                std::fprintf(stderr, "quant buffer K mismatch %d vs %d\n", quant_k, mat.K);
                std::exit(1);
            }
            mv(mat, out32);
        }
    }

    // src must equal the currently quantized activation for quant mats
    void mv_f16out(const Mat& mat, const __half* src, float* scratch32, __half* out16) {
        mv16(mat, src, scratch32);
        f32_to_f16_launch(scratch32, out16, mat.M, 0);
    }
};

void gdn_layer(Runtime& rt, int il) {
    const HParams& hp = rt.m.hp;
    const Layer& l = rt.m.layers[(size_t)il];
    const int Sk = hp.ssm_state, Hk = hp.ssm_groups;
    const int Sv = hp.head_v_dim, Hv = hp.ssm_dt_rank;

    rmsnorm_launch(rt.x, l.attn_norm, rt.xn, hp.n_embd, hp.rms_eps, 0);
    rt.quant(rt.xn, hp.n_embd);

    // qkv_mixed -> big_a (f16), z -> big_b (f16)
    rt.mv_f16out(l.ssm_in, rt.xn, rt.y32, rt.big_a);
    rt.mv_f16out(l.ssm_gate, rt.xn, rt.y32, rt.big_b);
    // alpha and beta raw projections (H_v each) — gdn_out doubles as scratch
    __half* alpha_raw = rt.gdn_out;
    __half* beta_raw = rt.gdn_out + Hv;
    rt.mv_f16out(l.ssm_alpha, rt.xn, rt.y32, alpha_raw);
    rt.mv_f16out(l.ssm_beta, rt.xn, rt.y32, beta_raw);

    // conv over qkv_mixed (in place into big_a)
    conv1d_step_launch(rt.big_a, l.conv_w, rt.conv_state[(size_t)il], rt.big_a,
                       rt.qkv_dim(), hp.ssm_conv, 0);

    __half* qc = rt.big_a;
    __half* kc = rt.big_a + (size_t)Sk * Hk;
    __half* vc = rt.big_a + (size_t)2 * Sk * Hk;
    l2norm_heads_launch(qc, qc, Hk, Sk, hp.rms_eps, 0);
    l2norm_heads_launch(kc, kc, Hk, Sk, hp.rms_eps, 0);

    gdn_decode_launch(qc, kc, vc, alpha_raw, beta_raw, l.ssm_a, l.ssm_dt,
                      rt.gdn_state[(size_t)il], rt.attn_out /*[Hv][Sv]*/, Hv, Hk,
                      Sv, 1.0f, 0);

    // gated norm: rmsnorm per head, then * silu(z)
    rmsnorm_heads_launch(rt.attn_out, l.ssm_norm, rt.attn_out, Hv, Sv, hp.rms_eps, 0);
    silu_mul_launch(rt.big_b, rt.attn_out, rt.attn_out, hp.ssm_inner, 0);

    rt.quant(rt.attn_out, hp.ssm_inner);
    rt.mv_f16out(l.ssm_out, rt.attn_out, rt.y32, rt.xn);
    add_inplace_launch(rt.x, rt.xn, hp.n_embd, 0);
}

void attn_layer(Runtime& rt, int il) {
    const HParams& hp = rt.m.hp;
    const Layer& l = rt.m.layers[(size_t)il];
    const int D = hp.head_dim, H = hp.n_head, Hkv = hp.n_head_kv;

    rmsnorm_launch(rt.x, l.attn_norm, rt.xn, hp.n_embd, hp.rms_eps, 0);
    rt.quant(rt.xn, hp.n_embd);

    rt.mv_f16out(l.wq, rt.xn, rt.y32, rt.big_a);  // [H][2D] interleaved q|gate
    rt.mv_f16out(l.wk, rt.xn, rt.y32, rt.k);
    rt.mv_f16out(l.wv, rt.xn, rt.y32, rt.v);

    gather_heads_launch(rt.big_a, rt.q, H, D, 2 * D, 0, 0);
    gather_heads_launch(rt.big_a, rt.big_b, H, D, 2 * D, D, 0);  // gate

    rmsnorm_heads_launch(rt.q, l.q_norm, rt.q, H, D, hp.rms_eps, 0);
    rmsnorm_heads_launch(rt.k, l.k_norm, rt.k, Hkv, D, hp.rms_eps, 0);
    rope_neox_launch(rt.q, H, D, hp.n_rot, rt.pos, hp.rope_base, 0);
    rope_neox_launch(rt.k, Hkv, D, hp.n_rot, rt.pos, hp.rope_base, 0);

    const size_t kv_row = (size_t)Hkv * D * 2;
    CUDA_CHECK(cudaMemcpyAsync(rt.k_cache[(size_t)il] + rt.pos * (kv_row / 2), rt.k,
                               kv_row, cudaMemcpyDeviceToDevice, 0));
    CUDA_CHECK(cudaMemcpyAsync(rt.v_cache[(size_t)il] + rt.pos * (kv_row / 2), rt.v,
                               kv_row, cudaMemcpyDeviceToDevice, 0));

    attn_decode_launch(rt.q, rt.k_cache[(size_t)il], rt.v_cache[(size_t)il],
                       rt.attn_out, H, Hkv, D, rt.pos + 1,
                       1.f / sqrtf((float)D), 0);

    sigmoid_mul_launch(rt.attn_out, rt.big_b, rt.attn_out, H * D, 0);

    rt.quant(rt.attn_out, H * D);
    rt.mv_f16out(l.wo, rt.attn_out, rt.y32, rt.xn);
    add_inplace_launch(rt.x, rt.xn, hp.n_embd, 0);
}

void mlp(Runtime& rt, int il) {
    const HParams& hp = rt.m.hp;
    const Layer& l = rt.m.layers[(size_t)il];
    rmsnorm_launch(rt.x, l.attn_post_norm, rt.xn, hp.n_embd, hp.rms_eps, 0);
    rt.quant(rt.xn, hp.n_embd);
    rt.mv_f16out(l.gate, rt.xn, rt.y32, rt.big_a);
    rt.mv_f16out(l.up, rt.xn, rt.y32, rt.big_b);
    silu_mul_launch(rt.big_a, rt.big_b, rt.big_a, hp.n_ff, 0);
    rt.quant(rt.big_a, hp.n_ff);
    rt.mv_f16out(l.down, rt.big_a, rt.y32, rt.xn);
    add_inplace_launch(rt.x, rt.xn, hp.n_embd, 0);
}

// full decode step: token in, logits (device y32) out
void decode(Runtime& rt, int token) {
    const HParams& hp = rt.m.hp;
    if (rt.m.tok_embd.nbits == 16) {
        embed_lookup_launch(rt.m.tok_embd.dense, token, hp.n_embd, rt.x, 0);
    } else {
        dequant_row_launch(rt.m.tok_embd.nbits, rt.m.tok_embd.codes,
                           rt.m.tok_embd.scales, token, hp.n_embd, rt.x, 0);
    }
    for (int il = 0; il < hp.n_layer; ++il) {
        if (rt.m.layers[(size_t)il].recurrent) {
            gdn_layer(rt, il);
        } else {
            attn_layer(rt, il);
        }
        mlp(rt, il);
    }
    rmsnorm_launch(rt.x, rt.m.output_norm, rt.xn, hp.n_embd, hp.rms_eps, 0);
    rt.quant(rt.xn, hp.n_embd);
    rt.mv16(rt.m.lm_head, rt.xn, rt.y32);
    rt.pos++;
}

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> ids;
    const char* p = s.c_str();
    while (*p) {
        ids.push_back((int)std::strtol(p, (char**)&p, 10));
        if (*p == ',') ++p;
    }
    return ids;
}

}  // namespace

// one prompt: feed ids, generate n_gen greedy tokens, optionally dump logits
void run_prompt(Runtime& rt, const std::vector<int>& prompt, int n_gen,
                const std::string& logits_path, bool bench) {
    const HParams& hp = rt.m.hp;
    rt.reset_sequence();
    FILE* lf = logits_path.empty() ? nullptr : std::fopen(logits_path.c_str(), "wb");
    std::vector<float> logits((size_t)hp.vocab);

    for (int id : prompt) decode(rt, id);

    static int32_t* d_tok = nullptr;
    if (!d_tok) CUDA_CHECK(cudaMalloc(&d_tok, 4));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    std::printf("generated:");
    int32_t tok = 0;
    for (int i = 0; i < n_gen; ++i) {
        if (lf) {
            CUDA_CHECK(cudaMemcpy(logits.data(), rt.y32, (size_t)hp.vocab * 4,
                                  cudaMemcpyDeviceToHost));
            std::fwrite(logits.data(), 4, (size_t)hp.vocab, lf);
        }
        argmax_launch(rt.y32, hp.vocab, d_tok, 0);
        CUDA_CHECK(cudaMemcpy(&tok, d_tok, 4, cudaMemcpyDeviceToHost));
        std::printf(" %d", tok);
        decode(rt, tok);
    }
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    std::printf("\n");
    if (lf) std::fclose(lf);

    if (bench) {
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        std::printf("decode: %d tokens in %.1f ms = %.2f tok/s%s\n", n_gen, ms,
                    1000.f * n_gen / ms,
                    logits_path.empty() ? "" : " (WARNING: includes logit dumps)");
    }
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
}

int main(int argc, char** argv) {
    std::string model_path, ids_str, ids_file, logits_out;
    int n_gen = 32;
    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--ids" && i + 1 < argc) ids_str = argv[++i];
        else if (a == "--ids-file" && i + 1 < argc) ids_file = argv[++i];
        else if (a == "--n" && i + 1 < argc) n_gen = std::atoi(argv[++i]);
        else if (a == "--logits-out" && i + 1 < argc) logits_out = argv[++i];
        else if (a == "--bench") bench = true;
    }
    if (model_path.empty() || (ids_str.empty() && ids_file.empty())) {
        std::fprintf(stderr,
                     "usage: bt-run --model X.gguf (--ids 1,2,3 | --ids-file f) "
                     "[--n 64] [--logits-out prefix] [--bench]\n");
        return 2;
    }

    Runtime rt;
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
