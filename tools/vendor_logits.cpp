// vendor_logits — dump the vendor fork's per-step greedy logits for parity.
//
// Compiled at harness time against the fork's own libllama (never shipped):
//   g++ -O2 -I$FORK/include -I$FORK/ggml/include tools/vendor_logits.cpp \
//       -L$FORK/build/bin -lllama -Wl,-rpath,$FORK/build/bin -o vendor-logits
//
// Usage: vendor-logits model.gguf ids.txt N_GEN out_prefix
//   ids.txt: one comma-separated token-id list per line (one prompt each)
// Writes out_prefix.<i>.bin (N_GEN * n_vocab float32) + greedy ids on stdout,
// exactly mirroring `bt-run --ids-file --logits-out`.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: vendor-logits model.gguf ids.txt n_gen out_prefix\n");
        return 2;
    }
    const char* model_path = argv[1];
    FILE* idsf = std::fopen(argv[2], "r");
    if (!idsf) {
        std::fprintf(stderr, "cannot open %s\n", argv[2]);
        return 1;
    }
    const int n_gen = std::atoi(argv[3]);
    const std::string out_prefix = argv[4];

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        std::fprintf(stderr, "model load failed\n");
        return 1;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 512;
    llama_context* ctx = llama_init_from_model(model, cp);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    char line[65536];
    int idx = 0;
    while (std::fgets(line, sizeof(line), idsf)) {
        std::vector<llama_token> prompt;
        for (const char* p = line; *p && *p != '\n';) {
            prompt.push_back((llama_token)std::strtol(p, (char**)&p, 10));
            if (*p == ',') ++p;
        }
        if (prompt.empty()) continue;

        llama_memory_clear(llama_get_memory(ctx), true);
        FILE* out = std::fopen((out_prefix + "." + std::to_string(idx) + ".bin").c_str(), "wb");

        // one token at a time: same decode regime as bt-run, so batched-prefill
        // numeric differences can't blur the comparison
        bool failed = false;
        for (llama_token id : prompt) {
            llama_batch step = llama_batch_get_one(&id, 1);
            if (llama_decode(ctx, step)) {
                std::fprintf(stderr, "prompt decode failed (prompt %d)\n", idx);
                failed = true;
                break;
            }
        }
        if (failed) return 1;

        std::printf("generated:");
        llama_token tok = 0;
        for (int i = 0; i < n_gen; ++i) {
            const float* logits = llama_get_logits(ctx);
            std::fwrite(logits, 4, (size_t)n_vocab, out);
            tok = 0;
            for (int v = 1; v < n_vocab; ++v) {
                if (logits[v] > logits[tok]) tok = v;
            }
            std::printf(" %d", tok);
            llama_batch step = llama_batch_get_one(&tok, 1);
            if (llama_decode(ctx, step)) {
                std::fprintf(stderr, "decode failed at step %d (prompt %d)\n", i, idx);
                return 1;
            }
        }
        std::printf("\n");
        std::fclose(out);
        ++idx;
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
