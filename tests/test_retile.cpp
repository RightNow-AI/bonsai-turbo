// Proves the dp4a permutation is a bijection that preserves values:
// random blocks -> retile -> decode in the exact GPU read order -> must equal
// the vendor-reference dequant element for element.
#include <cstdio>
#include <random>
#include <vector>

#include "../src/packs.h"
#include "../src/retile.h"

using namespace bt;

static int failures = 0;

static void check_eq(const std::vector<float>& a, const std::vector<float>& b, const char* tag) {
    if (a.size() != b.size()) {
        std::printf("FAIL %s: size %zu vs %zu\n", tag, a.size(), b.size());
        ++failures;
        return;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            std::printf("FAIL %s: elem %zu got %g want %g\n", tag, i, a[i], b[i]);
            ++failures;
            return;
        }
    }
}

int main() {
    std::mt19937 rng(42);
    const int64_t rows = 7, cols = 512;  // deliberately not powers-of-two rows

    {
        std::vector<BlockQ2_0> blocks((size_t)(rows * cols / kQ2GroupSize));
        for (auto& blk : blocks) {
            blk.d = (half_bits)(0x3400 + (rng() & 0x0FFF));  // positive-ish scales
            for (auto& q : blk.qs) q = (uint8_t)(rng() & 0xFF);
        }
        std::vector<float> ref((size_t)(rows * cols)), got((size_t)(rows * cols));
        dequant_q2_0_ref(blocks.data(), ref.data(), rows * cols);
        RetiledTensor t = retile_q2_0(blocks.data(), rows, cols);
        decode_permuted_q2_0(t, got.data());
        check_eq(got, ref, "q2_0 retile round-trip");
    }

    {
        std::vector<BlockQ1_0> blocks((size_t)(rows * cols / kQ1GroupSize));
        for (auto& blk : blocks) {
            blk.d = (half_bits)(0x3400 + (rng() & 0x0FFF));
            for (auto& q : blk.qs) q = (uint8_t)(rng() & 0xFF);
        }
        std::vector<float> ref((size_t)(rows * cols)), got((size_t)(rows * cols));
        dequant_q1_0_ref(blocks.data(), ref.data(), rows * cols);
        RetiledTensor t = retile_q1_0(blocks.data(), rows, cols);
        decode_permuted_q1_0(t, got.data());
        check_eq(got, ref, "q1_0 retile round-trip");
    }

    if (failures) {
        std::printf("%d FAILURES\n", failures);
        return 1;
    }
    std::printf("all retile tests passed\n");
    return 0;
}
