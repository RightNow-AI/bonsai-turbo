# bonsai-turbo

An open-source decode engine for PrismML's Bonsai 27B (ternary and 1-bit GGUF packs)
on NVIDIA GPUs. Goal: 3-5x faster batch-1 decode than the vendor's llama.cpp fork by
fusing the whole per-token pass into as few kernel launches as possible.

**Status: work in progress. No performance claims yet — numbers appear here only
after they are measured and after logit-parity and MATH-500 correctness gates pass.**

## Why this should work (roofline)

A dense 27B model reads every weight once per decoded token. The ternary pack
(`Q2_0_g128`, 2.125 bits/weight deployed) is 7.17 GB on disk, so at H100 SXM's
3.35 TB/s HBM3 bandwidth the memory-bound ceiling is roughly:

```
3350 GB/s / 7.17 GB ≈ 467 tok/s   (ternary, batch 1, weights-only traffic)
```

The vendor's own model card reports 98.0 tok/s (ternary) and 104.8 tok/s (1-bit)
on H100 with their llama.cpp fork, and attributes the gap to kernel-launch and
synchronization latency at batch 1 — not bandwidth. That is ~4.7x of documented
headroom. This repo goes after it with:

1. Interleaved weight re-tiling for coalesced 128-bit GEMV loads
2. A templated dequant-in-registers GEMV family for both packs
3. Fused gated delta-net state update (state stays on-chip)
4. Fused softmax attention that dequantizes the 4-bit KV cache in-kernel
5. CUDA Graphs, then a persistent single-launch decode step (cooperative grid sync)

## Scope

- Decode only, batch 1 only, NVIDIA only (H100 primary, RTX 5090 secondary)
- No prefill work, no vision tower / mmproj, no Metal or Apple targets

## Model facts (verified against sources, 2026-07-15)

| Item | Value | Source |
|---|---|---|
| Ternary pack | `Q2_0_g128`: FP16 scale + 32 B codes per 128 weights = 2.125 bpw, 7.17 GB | HF model card + fork `ggml-common.h` |
| 1-bit pack | `Q1_0_g128`: FP16 scale + 16 B codes per 128 weights = 1.125 bpw, 3.8 GB | HF model card + fork `ggml-common.h` |
| Parity reference | F16 GGUF, 53.8 GB | HF repo tree |
| Backbone | Qwen3.6-27B hybrid: 64 layers, gated delta-net linear attention except full attention every 4th layer (16/64), GQA + QK-norm + gated Q, dense-only MTP head appended | fork `src/models/qwen35.cpp` |
| Context | 262K; 4-bit KV cache on the 16 full-attention layers | model card + fork KV options |
| Vendor H100 baseline | 98.0 tok/s ternary, 104.8 tok/s 1-bit (tg128) | HF model cards |
| Vendor RTX 5090 baseline | 134 tok/s ternary, 163 tok/s 1-bit | vendor launch announcement |
| MATH-500 | F16 99.40, ternary 99.20, 1-bit 98.00 | HF model cards |
| DSpark drafter | 6-layer block-parallel, 1.34x (ternary) / 1.37x (1-bit) on H100 | HF model cards |

## License

Apache 2.0. This repo contains no vendor code; the PrismML llama.cpp fork is
cloned at build time for baseline comparison only.
