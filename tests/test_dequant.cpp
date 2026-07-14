// Bit-exactness tests for the pack reference dequant (no framework, exit code is the verdict).
//
// The expected values are computed straight from the format definition
// (LSB-first packing, (code-1)*d), which was transcribed from the vendor
// fork's dequantize_row_q1_0/q2_0 at the pinned SHA. Any layout drift in our
// structs or bit math fails these element-for-element checks.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "../src/packs.h"

using namespace bt;

static int failures = 0;

#define CHECK(cond, ...)                                    \
    do {                                                    \
        if (!(cond)) {                                      \
            std::printf("FAIL %s:%d ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                       \
            std::printf("\n");                              \
            ++failures;                                     \
        }                                                   \
    } while (0)

// fp16 1.0 = 0x3C00, 2.0 = 0x4000, 0.5 = 0x3800, -1.5 = 0xBE00
static void test_half_to_float() {
    CHECK(half_to_float(0x3C00) == 1.0f, "h2f(1.0)");
    CHECK(half_to_float(0x4000) == 2.0f, "h2f(2.0)");
    CHECK(half_to_float(0x3800) == 0.5f, "h2f(0.5)");
    CHECK(half_to_float(0xBE00) == -1.5f, "h2f(-1.5)");
    CHECK(half_to_float(0x0000) == 0.0f, "h2f(+0)");
    CHECK(half_to_float(0x0001) == std::ldexp(1.0f, -24), "h2f(min subnormal)");
    CHECK(half_to_float(0x7C00) == INFINITY, "h2f(inf)");
}

static void test_q1_0_known_pattern() {
    BlockQ1_0 b{};
    b.d = 0x4000;  // 2.0
    // byte 0 = 0b10110001: bits (LSB-first) 1,0,0,0,1,1,0,1
    b.qs[0] = 0xB1;
    float out[kQ1GroupSize];
    dequant_q1_0_ref(&b, out, kQ1GroupSize);

    const float expect[8] = {2, -2, -2, -2, 2, 2, -2, 2};
    for (int j = 0; j < 8; ++j) {
        CHECK(out[j] == expect[j], "q1_0 elem %d: got %g want %g", j, out[j], expect[j]);
    }
    for (int j = 8; j < kQ1GroupSize; ++j) {
        CHECK(out[j] == -2.0f, "q1_0 zero-bit elem %d: got %g", j, out[j]);
    }
}

static void test_q2_0_known_pattern() {
    BlockQ2_0 b{};
    b.d = 0x3800;  // 0.5
    // byte 0 = 0b11100100: codes (LSB-first) 0,1,2,3 -> -d, 0, +d, +2d
    b.qs[0] = 0xE4;
    float out[kQ2GroupSize];
    dequant_q2_0_ref(&b, out, kQ2GroupSize);

    CHECK(out[0] == -0.5f, "q2_0 code0: got %g", out[0]);
    CHECK(out[1] == 0.0f, "q2_0 code1: got %g", out[1]);
    CHECK(out[2] == 0.5f, "q2_0 code2: got %g", out[2]);
    CHECK(out[3] == 1.0f, "q2_0 code3: got %g", out[3]);
    for (int j = 4; j < kQ2GroupSize; ++j) {
        CHECK(out[j] == -0.5f, "q2_0 code0 elem %d: got %g", j, out[j]);
    }
}

static void test_q2_0_code3_counter() {
    std::vector<BlockQ2_0> blocks(4);
    std::mt19937 rng(7);
    for (auto& b : blocks) {
        b.d = 0x3C00;
        for (auto& q : b.qs) {
            // codes 0..2 only
            q = (uint8_t)((rng() % 3) | ((rng() % 3) << 2) | ((rng() % 3) << 4) | ((rng() % 3) << 6));
        }
    }
    CHECK(count_q2_0_code3(blocks.data(), 4 * kQ2GroupSize) == 0, "clean blocks report code3");

    blocks[2].qs[5] |= 0x03;  // plant one code 3 in the low position
    CHECK(count_q2_0_code3(blocks.data(), 4 * kQ2GroupSize) == 1, "planted code3 not found");
}

static void test_round_trip_energy() {
    // Randomized structural check: dequant of random blocks stays on the
    // {-d, 0, +d, +2d} lattice and block boundaries don't leak.
    std::mt19937 rng(1234);
    std::vector<BlockQ2_0> blocks(16);
    for (auto& b : blocks) {
        b.d = 0x3C00;  // 1.0
        for (auto& q : b.qs) q = (uint8_t)(rng() & 0xFF);
    }
    std::vector<float> out(16 * kQ2GroupSize);
    dequant_q2_0_ref(blocks.data(), out.data(), (int64_t)out.size());
    for (float v : out) {
        CHECK(v == -1.0f || v == 0.0f || v == 1.0f || v == 2.0f, "off-lattice value %g", v);
    }
}

int main() {
    test_half_to_float();
    test_q1_0_known_pattern();
    test_q2_0_known_pattern();
    test_q2_0_code3_counter();
    test_round_trip_energy();
    if (failures) {
        std::printf("%d FAILURES\n", failures);
        return 1;
    }
    std::printf("all dequant tests passed\n");
    return 0;
}
