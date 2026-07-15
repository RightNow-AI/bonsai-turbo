#include "model.h"

#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

#include "packs.h"
#include "retile.h"

namespace bt {

namespace {

#define BT_CUDA(expr)                                                          \
    do {                                                                       \
        cudaError_t err_ = (expr);                                             \
        if (err_ != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("cuda: ") +                   \
                                     cudaGetErrorString(err_));                \
        }                                                                      \
    } while (0)

float bf16_to_float(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

uint16_t float_to_half_bits(float f) {
    // round-to-nearest-even fp32 -> fp16 (host-side, load path only)
    uint32_t x;
    std::memcpy(&x, &f, sizeof(f));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF) return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0));
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);  // overflow -> inf
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        const int shift = 14 - exp;
        uint32_t half_man = man >> shift;
        const uint32_t rem = man & ((1u << shift) - 1);
        const uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half_man & 1))) half_man++;
        return (uint16_t)(sign | half_man);
    }
    uint32_t half_man = man >> 13;
    const uint32_t rem = man & 0x1FFFu;
    uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | half_man);
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) h++;
    return h;
}

// rows = product of dims 1..n; cols = dim 0
void tensor_rows_cols(const TensorInfo& t, int64_t& rows, int64_t& cols) {
    cols = t.shape.empty() ? 0 : t.shape[0];
    rows = 1;
    for (size_t i = 1; i < t.shape.size(); ++i) rows *= t.shape[i];
}

struct Loader {
    const GGUFFile& f;

    std::string missing_report(const std::string& wanted) const {
        std::string msg = "tensor not found: " + wanted + "\n  available blk.0/global names:\n";
        for (const auto& t : f.tensors()) {
            if (t.name.rfind("blk.", 0) != 0 || t.name.rfind("blk.0.", 0) == 0) {
                msg += "    " + t.name + " (" + ggml_type_name(t.type) + ")\n";
            }
        }
        return msg;
    }

    const TensorInfo* find_opt(const std::vector<std::string>& names) const {
        for (const auto& n : names) {
            if (const TensorInfo* t = f.find(n)) return t;
        }
        return nullptr;
    }

    const TensorInfo& need(const std::vector<std::string>& names) const {
        const TensorInfo* t = find_opt(names);
        if (!t) throw std::runtime_error(missing_report(names.front()));
        return *t;
    }

    // upload any of F16/F32/BF16 as f16
    __half* f16(const TensorInfo& t) const {
        const int64_t n = t.n_elements();
        std::vector<uint16_t> host((size_t)n);
        const uint8_t* src = f.tensor_data(t);
        if (t.type == GGMLType::F16) {
            std::memcpy(host.data(), src, (size_t)n * 2);
        } else if (t.type == GGMLType::F32) {
            const float* s = reinterpret_cast<const float*>(src);
            for (int64_t i = 0; i < n; ++i) host[(size_t)i] = float_to_half_bits(s[i]);
        } else if (t.type == GGMLType::BF16) {
            const uint16_t* s = reinterpret_cast<const uint16_t*>(src);
            for (int64_t i = 0; i < n; ++i) {
                host[(size_t)i] = float_to_half_bits(bf16_to_float(s[i]));
            }
        } else {
            throw std::runtime_error("expected float tensor: " + t.name);
        }
        __half* dev;
        BT_CUDA(cudaMalloc(&dev, (size_t)n * 2));
        BT_CUDA(cudaMemcpy(dev, host.data(), (size_t)n * 2, cudaMemcpyHostToDevice));
        return dev;
    }

    float* f32(const TensorInfo& t) const {
        const int64_t n = t.n_elements();
        std::vector<float> host((size_t)n);
        const uint8_t* src = f.tensor_data(t);
        if (t.type == GGMLType::F32) {
            std::memcpy(host.data(), src, (size_t)n * 4);
        } else if (t.type == GGMLType::F16) {
            const uint16_t* s = reinterpret_cast<const uint16_t*>(src);
            for (int64_t i = 0; i < n; ++i) host[(size_t)i] = half_to_float(s[i]);
        } else if (t.type == GGMLType::BF16) {
            const uint16_t* s = reinterpret_cast<const uint16_t*>(src);
            for (int64_t i = 0; i < n; ++i) host[(size_t)i] = bf16_to_float(s[i]);
        } else {
            throw std::runtime_error("expected float tensor: " + t.name);
        }
        float* dev;
        BT_CUDA(cudaMalloc(&dev, (size_t)n * 4));
        BT_CUDA(cudaMemcpy(dev, host.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
        return dev;
    }

    Mat mat(const TensorInfo& t) const {
        Mat m;
        int64_t rows, cols;
        tensor_rows_cols(t, rows, cols);
        m.M = (int)rows;
        m.K = (int)cols;
        if (t.type == GGMLType::Q2_0 || t.type == GGMLType::Q1_0) {
            m.nbits = t.type == GGMLType::Q2_0 ? 2 : 1;
            RetiledTensor rt;
            if (m.nbits == 2) {
                const auto* blocks = reinterpret_cast<const BlockQ2_0*>(f.tensor_data(t));
                const int64_t c3 = count_q2_0_code3(blocks, rows * cols);
                if (c3 != 0) {
                    throw std::runtime_error(t.name + ": " + std::to_string(c3) +
                                             " code-3 (+2d) elements; pack not ternary");
                }
                rt = retile_q2_0(blocks, rows, cols);
            } else {
                rt = retile_q1_0(reinterpret_cast<const BlockQ1_0*>(f.tensor_data(t)), rows, cols);
            }
            BT_CUDA(cudaMalloc(&m.codes, rt.codes.size()));
            BT_CUDA(cudaMemcpy(m.codes, rt.codes.data(), rt.codes.size(), cudaMemcpyHostToDevice));
            BT_CUDA(cudaMalloc(&m.scales, rt.scales.size() * 2));
            BT_CUDA(cudaMemcpy(m.scales, rt.scales.data(), rt.scales.size() * 2,
                               cudaMemcpyHostToDevice));
        } else {
            m.nbits = 16;
            m.dense = f16(t);
        }
        return m;
    }
};

std::string blk(int il, const char* suffix) {
    return "blk." + std::to_string(il) + "." + suffix;
}

}  // namespace

void Model::load(const std::string& path) {
    static GGUFFile f;  // keeps the mmap alive for the process lifetime
    f.open(path);
    Loader L{f};

    const std::string arch = f.value("general.architecture").as_str();
    auto key = [&](const char* k) { return arch + "." + k; };

    hp.n_layer = (int)f.value(key("block_count")).as_int();
    hp.n_embd = (int)f.value(key("embedding_length")).as_int();
    hp.n_head = (int)f.value(key("attention.head_count")).as_int();
    hp.n_head_kv = (int)f.get_int(key("attention.head_count_kv"), hp.n_head);
    hp.head_dim = (int)f.get_int(key("attention.key_length"), hp.n_embd / hp.n_head);
    hp.n_ff = (int)f.get_int(key("feed_forward_length"), 0);
    hp.rms_eps = f.has(key("attention.layer_norm_rms_epsilon"))
                     ? (float)f.value(key("attention.layer_norm_rms_epsilon")).as_f64()
                     : 1e-6f;
    hp.rope_base = f.has(key("rope.freq_base"))
                       ? (float)f.value(key("rope.freq_base")).as_f64()
                       : 10000.f;
    hp.n_rot = (int)f.get_int(key("rope.dimension_count"), hp.head_dim);
    hp.ssm_state = (int)f.get_int(key("ssm.state_size"), 0);
    hp.ssm_groups = (int)f.get_int(key("ssm.group_count"), 0);
    hp.ssm_dt_rank = (int)f.get_int(key("ssm.time_step_rank"), 0);
    hp.ssm_inner = (int)f.get_int(key("ssm.inner_size"), 0);
    hp.ssm_conv = (int)f.get_int(key("ssm.conv_kernel"), 4);

    // recurrent layout: explicit per-layer array, else (i+1) % interval != 0
    hp.recurrent.assign((size_t)hp.n_layer, 1);
    if (f.has(key("attention.recurrent_layers"))) {
        const auto& arr = f.value(key("attention.recurrent_layers")).as_ints();
        for (int i = 0; i < hp.n_layer && i < (int)arr.size(); ++i) {
            hp.recurrent[(size_t)i] = (uint8_t)(arr[(size_t)i] != 0);
        }
    } else {
        const int interval = (int)f.get_int(key("full_attention_interval"), 4);
        for (int i = 0; i < hp.n_layer; ++i) {
            hp.recurrent[(size_t)i] = (uint8_t)((i + 1) % interval != 0);
        }
    }

    tok_embd = L.mat(L.need({"token_embd.weight"}));
    hp.vocab = tok_embd.M;
    output_norm = L.f16(L.need({"output_norm.weight"}));
    if (const TensorInfo* out = L.find_opt({"output.weight"})) {
        lm_head = L.mat(*out);
    } else {
        lm_head = tok_embd;  // tied embeddings
    }

    layers.resize((size_t)hp.n_layer);
    for (int il = 0; il < hp.n_layer; ++il) {
        Layer& l = layers[(size_t)il];
        l.recurrent = hp.recurrent[(size_t)il] != 0;
        l.attn_norm = L.f16(L.need({blk(il, "attn_norm.weight")}));
        l.attn_post_norm = L.f16(L.need({blk(il, "post_attention_norm.weight"),
                                         blk(il, "ffn_norm.weight"),
                                         blk(il, "attn_post_norm.weight")}));
        l.up = L.mat(L.need({blk(il, "ffn_up.weight")}));
        l.gate = L.mat(L.need({blk(il, "ffn_gate.weight")}));
        l.down = L.mat(L.need({blk(il, "ffn_down.weight")}));

        if (l.recurrent) {
            // Bonsai 27B names these attn_qkv/attn_gate even on delta-net layers
            l.ssm_in = L.mat(L.need({blk(il, "attn_qkv.weight"), blk(il, "ssm_in.weight")}));
            l.ssm_gate = L.mat(L.need({blk(il, "attn_gate.weight"),
                                       blk(il, "ssm_in_gate.weight"),
                                       blk(il, "ssm_gate.weight")}));
            l.ssm_beta = L.mat(L.need({blk(il, "ssm_beta.weight")}));
            l.ssm_alpha = L.mat(L.need({blk(il, "ssm_alpha.weight")}));
            l.ssm_out = L.mat(L.need({blk(il, "ssm_out.weight")}));
            l.ssm_norm = L.f16(L.need({blk(il, "ssm_norm.weight")}));
            l.ssm_a = L.f32(L.need({blk(il, "ssm_a")}));
            l.ssm_dt = L.f32(L.need({blk(il, "ssm_dt.bias"), blk(il, "ssm_dt")}));
            const TensorInfo& conv = L.need({blk(il, "ssm_conv1d.weight")});
            l.conv_w = L.f16(conv);
            int64_t rows, cols;
            tensor_rows_cols(conv, rows, cols);
            hp.conv_channels = (int)rows;  // ne[0] = kernel width, rows = channels
            if (hp.ssm_conv != (int)cols) hp.ssm_conv = (int)cols;
        } else {
            l.wq = L.mat(L.need({blk(il, "attn_q.weight")}));
            l.wk = L.mat(L.need({blk(il, "attn_k.weight")}));
            l.wv = L.mat(L.need({blk(il, "attn_v.weight")}));
            l.wo = L.mat(L.need({blk(il, "attn_output.weight")}));
            l.q_norm = L.f16(L.need({blk(il, "attn_q_norm.weight")}));
            l.k_norm = L.f16(L.need({blk(il, "attn_k_norm.weight")}));
        }
    }

    if (hp.ssm_inner == 0) {
        for (const Layer& l : layers) {
            if (l.recurrent) {
                hp.ssm_inner = l.ssm_out.K;
                break;
            }
        }
    }
    for (const Layer& l : layers) {
        if (l.recurrent) {
            if (hp.ssm_dt_rank == 0) hp.ssm_dt_rank = l.ssm_beta.M;
            break;
        }
    }
    hp.head_v_dim = hp.ssm_dt_rank ? hp.ssm_inner / hp.ssm_dt_rank : 0;
    if (hp.ssm_groups == 0 && hp.ssm_state != 0) {
        // conv channels = 2*S_k*H_k + d_inner
        hp.ssm_groups = (hp.conv_channels - hp.ssm_inner) / (2 * hp.ssm_state);
    }
}

}  // namespace bt
