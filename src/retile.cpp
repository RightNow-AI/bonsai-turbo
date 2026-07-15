#include "retile.h"

#include <cassert>
#include <cstring>
#include <stdexcept>

namespace bt {

namespace {

// Original Q2 code j (0..15 within a 32-bit word) -> destination bit offset.
// j = 4i + b lands at bits [8b + 2i, 8b + 2i + 1].
inline int q2_dst_bit(int j) {
    const int i = j / 4;
    const int b = j % 4;
    return 8 * b + 2 * i;
}

// Original Q1 code j (0..31 within a 32-bit word) -> destination bit offset.
// j = 4i + b lands at bit [8b + i].
inline int q1_dst_bit(int j) {
    const int i = j / 4;
    const int b = j % 4;
    return 8 * b + i;
}

}  // namespace

RetiledTensor retile_q2_0(const BlockQ2_0* src, int64_t rows, int64_t cols) {
    if (cols % kQ2GroupSize != 0) throw std::runtime_error("retile: cols not a multiple of 128");
    const int64_t blocks_per_row = cols / kQ2GroupSize;

    RetiledTensor t;
    t.rows = rows;
    t.cols = cols;
    t.codes.resize((size_t)(rows * cols / 4));
    t.scales.resize((size_t)(rows * blocks_per_row));

#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const BlockQ2_0* row = src + r * blocks_per_row;
        uint8_t* out_codes = t.codes.data() + r * (cols / 4);
        for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
            t.scales[r * blocks_per_row + blk] = row[blk].d;
            // 128 codes = 8 32-bit words of 16 codes each
            for (int w = 0; w < 8; ++w) {
                uint32_t src_word;
                std::memcpy(&src_word, row[blk].qs + 4 * w, 4);
                uint32_t dst_word = 0;
                for (int j = 0; j < 16; ++j) {
                    const uint32_t code = (src_word >> (2 * j)) & 0x3u;
                    dst_word |= code << q2_dst_bit(j);
                }
                std::memcpy(out_codes + blk * 32 + 4 * w, &dst_word, 4);
            }
        }
    }
    return t;
}

RetiledTensor retile_q1_0(const BlockQ1_0* src, int64_t rows, int64_t cols) {
    if (cols % kQ1GroupSize != 0) throw std::runtime_error("retile: cols not a multiple of 128");
    const int64_t blocks_per_row = cols / kQ1GroupSize;

    RetiledTensor t;
    t.rows = rows;
    t.cols = cols;
    t.codes.resize((size_t)(rows * cols / 8));
    t.scales.resize((size_t)(rows * blocks_per_row));

#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const BlockQ1_0* row = src + r * blocks_per_row;
        uint8_t* out_codes = t.codes.data() + r * (cols / 8);
        for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
            t.scales[r * blocks_per_row + blk] = row[blk].d;
            // 128 codes = 4 32-bit words of 32 codes each
            for (int w = 0; w < 4; ++w) {
                uint32_t src_word;
                std::memcpy(&src_word, row[blk].qs + 4 * w, 4);
                uint32_t dst_word = 0;
                for (int j = 0; j < 32; ++j) {
                    const uint32_t bit = (src_word >> j) & 0x1u;
                    dst_word |= bit << q1_dst_bit(j);
                }
                std::memcpy(out_codes + blk * 16 + 4 * w, &dst_word, 4);
            }
        }
    }
    return t;
}

void decode_permuted_q2_0(const RetiledTensor& t, float* out) {
    const int64_t blocks_per_row = t.cols / kQ2GroupSize;
    for (int64_t r = 0; r < t.rows; ++r) {
        const uint8_t* codes = t.codes.data() + r * (t.cols / 4);
        for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
            const float d = half_to_float(t.scales[r * blocks_per_row + blk]);
            for (int w = 0; w < 8; ++w) {
                uint32_t word;
                std::memcpy(&word, codes + blk * 32 + 4 * w, 4);
                // exactly the kernel's dp4a view: shift i, byte b -> code 4i+b
                for (int i = 0; i < 4; ++i) {
                    const uint32_t lanes = (word >> (2 * i)) & 0x03030303u;
                    for (int b = 0; b < 4; ++b) {
                        const int code = (int)((lanes >> (8 * b)) & 0xFF);
                        const int64_t j = blk * 128 + w * 16 + 4 * i + b;
                        out[r * t.cols + j] = (float)(code - 1) * d;
                    }
                }
            }
        }
    }
}

void decode_permuted_q1_0(const RetiledTensor& t, float* out) {
    const int64_t blocks_per_row = t.cols / kQ1GroupSize;
    for (int64_t r = 0; r < t.rows; ++r) {
        const uint8_t* codes = t.codes.data() + r * (t.cols / 8);
        for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
            const float d = half_to_float(t.scales[r * blocks_per_row + blk]);
            for (int w = 0; w < 4; ++w) {
                uint32_t word;
                std::memcpy(&word, codes + blk * 16 + 4 * w, 4);
                for (int i = 0; i < 8; ++i) {
                    const uint32_t lanes = (word >> i) & 0x01010101u;
                    for (int b = 0; b < 4; ++b) {
                        const int bit = (int)((lanes >> (8 * b)) & 0xFF);
                        const int64_t j = blk * 128 + w * 32 + 4 * i + b;
                        out[r * t.cols + j] = bit ? d : -d;
                    }
                }
            }
        }
    }
}

}  // namespace bt
