// bt-microbench — correctness + achieved-bandwidth test for the GEMV family.
//
// Synthesizes random packs (or loads a real GGUF tensor), re-tiles, then times
// the decode GEMV. Correctness is checked against a double-precision CPU
// emulation of the exact same int8 pipeline (using the GPU's own quantized
// activations), so the only allowed difference is fp32 summation order.
//
// Usage:
//   bt-microbench                                   # default Bonsai-ish shapes
//   bt-microbench 4096x4096,151936x5120             # custom MxK list
//   bt-microbench --gguf model.gguf tensor_name     # one real tensor
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../src/cuda/gemv.h"
#include "../src/gguf.h"
#include "../src/packs.h"
#include "../src/retile.h"

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

static float measured_copy_peak_gbs() {
    const size_t n = 1ull << 30;  // 1 GiB each way
    void *a, *b;
    CUDA_CHECK(cudaMalloc(&a, n));
    CUDA_CHECK(cudaMalloc(&b, n));
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaMemcpy(b, a, n, cudaMemcpyDeviceToDevice));  // warm
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < 8; ++i) CUDA_CHECK(cudaMemcpy(b, a, n, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    // a D2D copy reads AND writes every byte
    return (float)(8.0 * 2.0 * n / (ms / 1e3) / 1e9);
}

struct BenchResult {
    double ms;
    double gbs;
    double rel_err_int8;   // vs double emulation of the same int8 pipeline
    double rel_err_fp;     // vs full-precision reference (int8 act quant error)
};

template <typename Block>
BenchResult run_case(int nbits, const Block* blocks, int64_t M, int64_t K) {
    const int64_t groups = K / 128;

    RetiledTensor t;
    if constexpr (std::is_same_v<Block, BlockQ2_0>) {
        t = retile_q2_0(reinterpret_cast<const BlockQ2_0*>(blocks), M, K);
    } else {
        t = retile_q1_0(reinterpret_cast<const BlockQ1_0*>(blocks), M, K);
    }

    // activations
    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<__half> x_h((size_t)K);
    for (auto& v : x_h) v = __float2half(dist(rng));

    // device buffers
    uint8_t* d_codes;
    __half *d_scales, *d_x;
    int8_t* d_a8;
    float *d_ascale, *d_y;
    int32_t* d_gsum;
    CUDA_CHECK(cudaMalloc(&d_codes, t.codes.size()));
    CUDA_CHECK(cudaMalloc(&d_scales, t.scales.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_x, (size_t)K * 2));
    CUDA_CHECK(cudaMalloc(&d_a8, (size_t)K));
    CUDA_CHECK(cudaMalloc(&d_ascale, groups * 4));
    CUDA_CHECK(cudaMalloc(&d_gsum, groups * 4));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)M * 4));
    CUDA_CHECK(cudaMemcpy(d_codes, t.codes.data(), t.codes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, t.scales.data(), t.scales.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x_h.data(), (size_t)K * 2, cudaMemcpyHostToDevice));

    quant_acts_launch(d_x, (int)K, d_a8, d_ascale, d_gsum, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // pull the GPU's own quantization for exact-pipeline emulation
    std::vector<int8_t> a8((size_t)K);
    std::vector<float> ascale((size_t)groups);
    std::vector<int32_t> gsum((size_t)groups);
    CUDA_CHECK(cudaMemcpy(a8.data(), d_a8, (size_t)K, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ascale.data(), d_ascale, groups * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gsum.data(), d_gsum, groups * 4, cudaMemcpyDeviceToHost));

    gemv_launch(nbits, d_codes, d_scales, d_a8, d_ascale, d_gsum, d_y, (int)M, (int)K, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> y((size_t)M);
    CUDA_CHECK(cudaMemcpy(y.data(), d_y, (size_t)M * 4, cudaMemcpyDeviceToHost));

    // reference dequant of the ORIGINAL blocks (double accumulation).
    // Capped at 1024 rows so the CPU reference stays fast on lm_head-sized
    // shapes; rows are checked from the front, timing always covers all M.
    const int64_t m_check = M < 1024 ? M : 1024;
    std::vector<float> w_row((size_t)K);
    double err_int8 = 0, norm_int8 = 0, err_fp = 0, norm_fp = 0;
    for (int64_t r = 0; r < m_check; ++r) {
        if constexpr (std::is_same_v<Block, BlockQ2_0>) {
            dequant_q2_0_ref(reinterpret_cast<const BlockQ2_0*>(blocks) + r * groups, w_row.data(), K);
        } else {
            dequant_q1_0_ref(reinterpret_cast<const BlockQ1_0*>(blocks) + r * groups, w_row.data(), K);
        }
        double ref_int8 = 0, ref_fp = 0;
        for (int64_t g = 0; g < groups; ++g) {
            long long dot = 0;
            double wd = 0;
            for (int j = 0; j < 128; ++j) {
                const int64_t k = g * 128 + j;
                const double w = w_row[(size_t)k];
                ref_fp += w * (double)__half2float(x_h[(size_t)k]);
                // int levels: Q2 codes-1 in {-1..2}; Q1 2*bit-1 in {-1,1}
                const double d_scale = __half2float(
                    (std::is_same_v<Block, BlockQ2_0>)
                        ? reinterpret_cast<const BlockQ2_0*>(blocks)[r * groups + g].d
                        : reinterpret_cast<const BlockQ1_0*>(blocks)[r * groups + g].d);
                if (j == 0) wd = d_scale;
                const int level = (int)std::lround(d_scale == 0 ? 0 : w / d_scale);
                dot += (long long)level * a8[(size_t)k];
            }
            ref_int8 += wd * (double)ascale[(size_t)g] * (double)dot;
        }
        err_int8 += (y[(size_t)r] - ref_int8) * (y[(size_t)r] - ref_int8);
        norm_int8 += ref_int8 * ref_int8;
        err_fp += (y[(size_t)r] - ref_fp) * (y[(size_t)r] - ref_fp);
        norm_fp += ref_fp * ref_fp;
    }

    // timing
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    for (int i = 0; i < 20; ++i) {
        gemv_launch(nbits, d_codes, d_scales, d_a8, d_ascale, d_gsum, d_y, (int)M, (int)K, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const int iters = 100;
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) {
        gemv_launch(nbits, d_codes, d_scales, d_a8, d_ascale, d_gsum, d_y, (int)M, (int)K, nullptr);
    }
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    BenchResult res;
    res.ms = ms / iters;
    const double bytes = (double)t.codes.size() + 2.0 * t.scales.size() + K + M * 4.0;
    res.gbs = bytes / (res.ms / 1e3) / 1e9;
    res.rel_err_int8 = std::sqrt(err_int8 / (norm_int8 + 1e-30));
    res.rel_err_fp = std::sqrt(err_fp / (norm_fp + 1e-30));

    CUDA_CHECK(cudaFree(d_codes));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_a8));
    CUDA_CHECK(cudaFree(d_ascale));
    CUDA_CHECK(cudaFree(d_gsum));
    CUDA_CHECK(cudaFree(d_y));
    return res;
}

template <typename Block>
std::vector<Block> synth_blocks(int nbits, int64_t M, int64_t K, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<Block> blocks((size_t)(M * K / 128));
    for (auto& b : blocks) {
        b.d = (half_bits)(0x2C00 + (rng() & 0x07FF));  // small positive scales
        for (auto& q : b.qs) {
            if (nbits == 2) {
                // codes 0..2 only, matching the vendor quantizer's range
                q = (uint8_t)((rng() % 3) | ((rng() % 3) << 2) | ((rng() % 3) << 4) |
                              ((rng() % 3) << 6));
            } else {
                q = (uint8_t)(rng() & 0xFF);
            }
        }
    }
    return blocks;
}

int main(int argc, char** argv) {
    const float peak = measured_copy_peak_gbs();
    std::printf("device copy peak (read+write): %.0f GB/s\n\n", peak);
    std::printf("%-6s %-14s %10s %10s %8s %12s %12s\n", "pack", "shape (MxK)", "ms/call",
                "GB/s", "%peak", "err(int8)", "err(fp16)");

    std::vector<std::pair<int64_t, int64_t>> shapes;
    GGUFFile gguf;
    if (argc >= 4 && std::strcmp(argv[1], "--gguf") == 0) {
        gguf.open(argv[2]);
        const TensorInfo* ti = gguf.find(argv[3]);
        if (!ti) {
            std::fprintf(stderr, "tensor %s not found\n", argv[3]);
            return 1;
        }
        int64_t rows = 1;
        for (size_t i = 1; i < ti->shape.size(); ++i) rows *= ti->shape[i];
        const int nbits = ti->type == GGMLType::Q2_0 ? 2 : 1;
        BenchResult r;
        if (nbits == 2) {
            r = run_case<BlockQ2_0>(2, reinterpret_cast<const BlockQ2_0*>(gguf.tensor_data(*ti)),
                                    rows, ti->shape[0]);
        } else {
            r = run_case<BlockQ1_0>(1, reinterpret_cast<const BlockQ1_0*>(gguf.tensor_data(*ti)),
                                    rows, ti->shape[0]);
        }
        std::printf("q%d_0   %-14s %10.4f %10.0f %7.1f%% %12.2e %12.2e\n", nbits,
                    (std::to_string(rows) + "x" + std::to_string(ti->shape[0])).c_str(),
                    r.ms, r.gbs, 100.0 * r.gbs / peak, r.rel_err_int8, r.rel_err_fp);
        return 0;
    }

    const char* spec = argc > 1 ? argv[1] : "5120x5120,13824x5120,5120x13824,151936x5120";
    for (const char* p = spec; *p;) {
        int64_t m = std::strtoll(p, (char**)&p, 10);
        ++p;  // 'x'
        int64_t k = std::strtoll(p, (char**)&p, 10);
        if (*p == ',') ++p;
        shapes.push_back({m, k});
    }

    for (auto [M, K] : shapes) {
        const std::string shape = std::to_string(M) + "x" + std::to_string(K);
        {
            auto blocks = synth_blocks<BlockQ2_0>(2, M, K, 1);
            BenchResult r = run_case<BlockQ2_0>(2, blocks.data(), M, K);
            std::printf("q2_0   %-14s %10.4f %10.0f %7.1f%% %12.2e %12.2e\n", shape.c_str(),
                        r.ms, r.gbs, 100.0 * r.gbs / peak, r.rel_err_int8, r.rel_err_fp);
        }
        {
            auto blocks = synth_blocks<BlockQ1_0>(1, M, K, 2);
            BenchResult r = run_case<BlockQ1_0>(1, blocks.data(), M, K);
            std::printf("q1_0   %-14s %10.4f %10.0f %7.1f%% %12.2e %12.2e\n", shape.c_str(),
                        r.ms, r.gbs, 100.0 * r.gbs / peak, r.rel_err_int8, r.rel_err_fp);
        }
    }
    return 0;
}
