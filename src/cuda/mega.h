// bonsai-turbo — persistent single-launch decode step ("megakernel").
//
// The entire token — embed, all 64 layers, head, argmax, state bump — runs as
// ONE cooperative kernel launch. Kernel boundaries become grid.sync()s, so
// per-op ramp-up/tail overhead and the graph scheduler disappear. All op
// implementations mirror the standalone kernels bit-for-bit (same quantization
// rounding, same dp4a layout), so logit parity carries over.
#pragma once

#include <cstdint>

#include <cuda_fp16.h>

namespace bt {

struct MegaMat {
    const uint8_t* codes;
    const __half* scales;
    int M, K, nbits;
};

struct MegaLayer {
    int recurrent;
    const __half *attn_norm, *post_norm;
    // recurrent: fused [qkv|z|alpha|beta] + out; attention: fused [q+gate|k|v] + out
    MegaMat proj, out_proj, gate_up, down;
    int in_rows, gate_rows;      // proj sub-block rows (qkv / z)
    int mlp_gate_rows;           // rows of the MLP gate half in gate_up
    // gated delta net
    const __half* ssm_norm;
    const float *ssm_a, *ssm_dt;
    const __half* conv_w;
    float *gdn_state, *conv_state;
    // attention
    const __half *q_norm, *k_norm;
    __half *k_cache, *v_cache;
};

struct MegaParams {
    // model
    const MegaLayer* layers;     // device array [n_layer]
    MegaMat tok_embd, lm_head;
    const __half* output_norm;
    // dims
    int n_layer, n_embd, n_ff, vocab;
    int n_head, n_head_kv, head_dim, n_rot;
    int ssm_state, ssm_groups, ssm_dt_rank, head_v_dim, ssm_conv, conv_channels;
    float rms_eps, rope_base;
    // workspace
    float* x;
    __half *xn, *big_a, *big_b, *q_buf, *attn_out;
    float* y32;
    int8_t* a8;
    float* a_scale;
    int32_t* a_gsum;
    float* red_scratch;          // [4] cross-block reduction cells
    float* amax_v;               // [grid] argmax partials
    int32_t* amax_i;
    // control state (device-resident)
    int32_t *d_pos, *d_step, *d_tok, *d_ring;
    int ring_cap;
    // phase telemetry: block 0 stamps clock64() at every barrier when non-null
    unsigned long long* ts;
    int* ts_count;
};

// One launch decodes one token: embed(*d_tok) ... logits, argmax -> *d_tok,
// ring[*d_step] = token, ++pos/step. Requires cooperative-launch support.
// Returns false if the device cannot co-schedule the grid.
bool mega_decode_launch(const MegaParams& p, cudaStream_t stream);

}  // namespace bt
