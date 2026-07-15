// bonsai-turbo — batch-1 softmax attention decode (flash-decode style).
// v1 keeps the KV cache fp16 for parity; the q4 in-attend dequant variant
// replaces it once logit parity holds.
#pragma once

#include <cuda_fp16.h>

namespace bt {

// q: [H][D] (already QK-normed + roped). Cache layout: [pos][H_kv][D].
// GQA: head h reads kv head h / (H / H_kv). out: [H][D].
// ctx_len includes the current position (append k/v before calling).
void attn_decode_launch(const __half* q, const __half* k_cache,
                        const __half* v_cache, __half* out, int H, int H_kv,
                        int D, int ctx_len, float scale, cudaStream_t stream);

}  // namespace bt
