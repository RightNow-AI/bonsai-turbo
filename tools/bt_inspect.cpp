// bt-inspect — dump GGUF metadata, tensor table, and pack sanity checks.
//
// Usage: bt-inspect model.gguf [--scan-code3]
//
// --scan-code3 walks every Q2_0 tensor and counts the nominally-unreachable
// code 3 (+2d) so kernels can safely specialize for {-1, 0, +1}.
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>

#include "../src/gguf.h"
#include "../src/packs.h"

using namespace bt;

static void print_value(const std::string& key, const GGUFValue& v) {
    std::printf("  %-52s ", key.c_str());
    switch (v.type) {
        case GGUFType::F32:
        case GGUFType::F64:
            std::printf("%g\n", v.as_f64());
            break;
        case GGUFType::Bool:
            std::printf("%s\n", v.as_bool() ? "true" : "false");
            break;
        case GGUFType::String: {
            std::string s = v.as_str();
            if (s.size() > 80) s = s.substr(0, 77) + "...";
            for (auto& ch : s) {
                if (ch == '\n') ch = ' ';
            }
            std::printf("\"%s\"\n", s.c_str());
            break;
        }
        case GGUFType::Array: {
            if (std::holds_alternative<std::vector<int64_t>>(v.v)) {
                const auto& a = v.as_ints();
                std::printf("[%zu ints]", a.size());
                const size_t show = a.size() <= 80 ? a.size() : 16;
                for (size_t i = 0; i < show; ++i) std::printf(" %" PRId64, a[i]);
                if (show < a.size()) std::printf(" ...");
                std::printf("\n");
            } else if (std::holds_alternative<std::vector<std::string>>(v.v)) {
                std::printf("[%zu strings]\n", v.as_strs().size());
            } else {
                std::printf("[float array]\n");
            }
            break;
        }
        default:
            std::printf("%" PRId64 "\n", v.as_int());
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: bt-inspect model.gguf [--scan-code3]\n");
        return 2;
    }
    const bool scan = argc > 2 && std::strcmp(argv[2], "--scan-code3") == 0;

    GGUFFile f;
    try {
        f.open(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    std::printf("== metadata (%zu keys)\n", f.metadata().size());
    for (const auto& [key, v] : f.metadata()) print_value(key, v);

    std::printf("\n== tensors (%zu)\n", f.tensors().size());
    uint64_t total = 0;
    for (const auto& t : f.tensors()) {
        total += f.tensor_nbytes(t);
        std::printf("  %-56s %-5s [", t.name.c_str(), ggml_type_name(t.type));
        for (size_t i = 0; i < t.shape.size(); ++i) {
            std::printf("%s%" PRId64, i ? ", " : "", t.shape[i]);
        }
        std::printf("]\n");
    }
    std::printf("  total tensor bytes: %.3f GB\n", total / 1e9);

    if (scan) {
        std::printf("\n== Q2_0 code-3 scan\n");
        int64_t worst = 0;
        for (const auto& t : f.tensors()) {
            if (t.type != GGMLType::Q2_0) continue;
            const int64_t n = t.n_elements();
            const int64_t c3 = count_q2_0_code3(
                reinterpret_cast<const BlockQ2_0*>(f.tensor_data(t)), n);
            if (c3 > 0) {
                std::printf("  %-56s %" PRId64 " code-3 elements!\n", t.name.c_str(), c3);
            }
            worst += c3;
        }
        std::printf("  total code-3 elements across all Q2_0 tensors: %" PRId64 "\n", worst);
        std::printf("  %s\n", worst == 0
            ? "OK: pack is strictly ternary; kernels may specialize {-1,0,+1}"
            : "WARNING: +2d level in use; kernels must implement all 4 codes");
    }
    return 0;
}
