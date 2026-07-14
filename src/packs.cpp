#include "packs.h"

#include <cstring>

namespace bt {

float half_to_float(half_bits h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1;
    const uint32_t exp  = (uint32_t)(h >> 10) & 0x1F;
    const uint32_t man  = (uint32_t)h & 0x3FF;

    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign << 31;  // +/- 0
        } else {
            // subnormal: normalize
            uint32_t e = 127 - 15 + 1;
            uint32_t m = man;
            while ((m & 0x400) == 0) {
                m <<= 1;
                e--;
            }
            bits = (sign << 31) | (e << 23) | ((m & 0x3FF) << 13);
        }
    } else if (exp == 0x1F) {
        bits = (sign << 31) | (0xFFu << 23) | (man << 13);  // inf / nan
    } else {
        bits = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

void dequant_q1_0_ref(const BlockQ1_0* blocks, float* out, int64_t k) {
    const int64_t nb = k / kQ1GroupSize;
    for (int64_t i = 0; i < nb; ++i) {
        const float d = half_to_float(blocks[i].d);
        for (int j = 0; j < kQ1GroupSize; ++j) {
            const uint8_t bit = (blocks[i].qs[j / 8] >> (j % 8)) & 1;
            out[i * kQ1GroupSize + j] = bit ? d : -d;
        }
    }
}

void dequant_q2_0_ref(const BlockQ2_0* blocks, float* out, int64_t k) {
    const int64_t nb = k / kQ2GroupSize;
    for (int64_t i = 0; i < nb; ++i) {
        const float d = half_to_float(blocks[i].d);
        for (int j = 0; j < kQ2GroupSize; ++j) {
            const uint8_t q = (blocks[i].qs[j / 4] >> ((j % 4) * 2)) & 0x03;
            out[i * kQ2GroupSize + j] = (float)((int)q - 1) * d;
        }
    }
}

int64_t count_q2_0_code3(const BlockQ2_0* blocks, int64_t k) {
    const int64_t nb = k / kQ2GroupSize;
    int64_t code3 = 0;
    for (int64_t i = 0; i < nb; ++i) {
        for (int j = 0; j < kQ2GroupSize / 4; ++j) {
            uint8_t byte = blocks[i].qs[j];
            for (int p = 0; p < 4; ++p) {
                code3 += ((byte >> (2 * p)) & 3) == 3;
            }
        }
    }
    return code3;
}

}  // namespace bt
