// vendor_probe — print layer-0 tensor sums from the vendor fork for parity
// bisection (mirrors bt-run's BT_PROBE=2 output).
//
// Build (harness-time, against the fork's libllama; never shipped):
//   g++ -O2 -I$FORK/include -I$FORK/ggml/include tools/vendor_probe.cpp \
//       -L$FORK/build/bin -lllama -lggml -lggml-base \
//       -Wl,-rpath,$FORK/build/bin -o vendor-probe
//
// Usage: vendor-probe model.gguf "9419"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

static const char* kWatch[] = {
    "attn_norm-0", "linear_attn_qkv_mixed-0", "z-0", "beta-0", "a_softplus-0",
    "gate-0", "conv_output_raw-0", "conv_output_silu-0", "q_conv_predelta-0",
    "k_conv_predelta-0", "v_conv_predelta-0", "final_output-0",
    "linear_attn_out-0", "attn_residual-0", "attn_post_norm-0", "l_out-0",
};

static bool cb(struct ggml_tensor* t, bool ask, void* /*user*/) {
    bool want = false;
    for (const char* name : kWatch) {
        if (std::strcmp(t->name, name) == 0) want = true;
    }
    if (ask) return want;  // request data only for watched tensors
    if (!want) return true;

    const int64_t n = ggml_nelements(t);
    std::vector<float> host((size_t)n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, host.data(), 0, (size_t)n * 4);
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<uint16_t> raw((size_t)n);
        ggml_backend_tensor_get(t, raw.data(), 0, (size_t)n * 2);
        for (int64_t i = 0; i < n; ++i) host[(size_t)i] = ggml_fp16_to_fp32(raw[(size_t)i]);
    } else {
        std::printf("vprobe %-24s (unsupported type)\n", t->name);
        return true;
    }
    double sum = 0;
    for (float v : host) sum += v;
    std::printf("vprobe %-24s n=%-6lld sum = %+.6f  [%+.5f %+.5f %+.5f]\n", t->name,
                (long long)n, sum, host[0], host[1], host[2]);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: vendor-probe model.gguf ids\n");
        return 2;
    }
    std::vector<llama_token> prompt;
    for (const char* p = argv[2]; *p;) {
        prompt.push_back((llama_token)std::strtol(p, (char**)&p, 10));
        if (*p == ',') ++p;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.cb_eval = cb;
    cp.cb_eval_user_data = nullptr;
    llama_context* ctx = llama_init_from_model(model, cp);

    for (llama_token id : prompt) {
        llama_batch step = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx, step)) {
            std::fprintf(stderr, "decode failed\n");
            return 1;
        }
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
