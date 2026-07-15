// bonsai-turbo — Bonsai 27B model: hparams, device weights, loading.
#pragma once

#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "gguf.h"

namespace bt {

struct HParams {
    int n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0;
    int n_ff = 0, vocab = 0, n_rot = 0;
    float rms_eps = 1e-6f, rope_base = 10000.f;
    // gated delta net
    int ssm_state = 0;    // head_k_dim (= key head size)
    int ssm_groups = 0;   // H_k (k/q heads)
    int ssm_dt_rank = 0;  // H_v (value heads)
    int ssm_inner = 0;    // d_inner = H_v * head_v_dim
    int ssm_conv = 0;     // conv kernel width
    int head_v_dim = 0;   // ssm_inner / ssm_dt_rank
    int conv_channels = 0;
    std::vector<uint8_t> recurrent;  // per layer: 1 = gated delta net
};

// device matrix: quantized (retiled) or f16
struct Mat {
    int nbits = 0;  // 2 / 1 = quant packs, 16 = f16, 0 = absent
    int M = 0, K = 0;
    uint8_t* codes = nullptr;  // quant
    __half* scales = nullptr;  // quant
    __half* dense = nullptr;   // f16
    bool valid() const { return nbits != 0; }
};

struct Layer {
    bool recurrent = false;
    __half *attn_norm = nullptr, *attn_post_norm = nullptr;
    // full attention
    Mat wq, wk, wv, wo;
    __half *q_norm = nullptr, *k_norm = nullptr;
    // gated delta net
    Mat ssm_in, ssm_gate, ssm_beta, ssm_alpha, ssm_out;
    __half* ssm_norm = nullptr;  // [head_v_dim]
    float *ssm_a = nullptr, *ssm_dt = nullptr;  // [H_v]
    __half* conv_w = nullptr;                   // [C][k]
    // mlp
    Mat up, gate, down;
};

struct Model {
    HParams hp;
    std::vector<Layer> layers;
    Mat tok_embd;  // quant or f16
    __half* output_norm = nullptr;
    Mat lm_head;

    // throws std::runtime_error with diagnostics on any missing tensor
    void load(const std::string& path);
};

}  // namespace bt
