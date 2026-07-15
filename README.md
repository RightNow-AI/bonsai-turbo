# bonsai-turbo

An open-source decode engine for PrismML's Bonsai 27B (ternary and 1-bit GGUF packs)
on NVIDIA GPUs, built to beat the vendor's llama.cpp fork at batch-1 decode by
collapsing the per-token pass into far fewer, fatter kernels.

## Measured results (H100 80GB SXM, cloud instance, tg128-comparable)

| engine | ternary tok/s | 1-bit tok/s | notes |
|---|---|---|---|
| vendor fork (same machine class) | 85.5 ± 6.9 | 90.1 ± 3.5 | llama-bench tg128, pinned SHA; a second host measured 94.6 ternary |
| vendor published | 98.0 | 104.8 | their model cards |
| **bonsai-turbo (CUDA graph)** | **151.1** | **122.5** | 128 greedy tokens, one graph launch per token; 146.8-151.4 across containers |
| bonsai-turbo (`--mega`) | 149.6 | — | whole token as ONE cooperative kernel launch; fused norm+quant staging, idle-block L2 prefetch; **32/32 logit parity** |

Speedup, ternary: **1.76x** vs the vendor fork measured on identical hardware,
**1.53x** vs their published H100 number. 1-bit: 1.36x / 1.17x (its GEMV inner
loop is not yet tuned). Work in progress toward the 350+ target — every number
above is measured, none are projected.

**Correctness first:** logit parity passes on 32/32 fixed prompts against the
vendor fork on the exact shipping build (greedy top-1 identical at every step,
or ties where the vendor's own top-1/top-2 margin was <= 0.034; pre-divergence
top-20 logit deltas stay within the measured cross-engine int8-activation noise
floor of ~1.2). See `scripts/parity.sh`. MATH-500 gate: in progress.

## Where the speed comes from (measured, not guessed)

Profiling the vendor fork on H100 (their own logs + NVML): **3703 graph nodes per
decode token**, GPU 97% busy — with CUDA graphs already active, batch-1 decode is
bound by the execution overhead of thousands of tiny sequential ops, not by launch
gaps or bandwidth. Their card's roofline headroom is real: per-token weight traffic
is 6.83 GB (everything but the embedding table), so H100's ~3.0 TB/s achievable
bandwidth allows ~440-490 tok/s.

bonsai-turbo's answer:

1. Load-time re-tiling: 34-byte GGUF blocks split into 16B-aligned code planes +
   scale planes, codes permuted for straight dp4a feeding (`src/retile.cpp`)
2. One templated GEMV family for both packs, 78-80% of measured copy peak on the
   big shapes (`src/cuda/gemv.cu`)
3. Projection stacking: GDN [qkv|z|alpha|beta], attention [q+gate|k|v], MLP
   [gate|up] each become a single GEMV at load time
4. Fused gated delta-net step (gate math folded in) and flash-decode attention
5. The whole token step — embed, 64 layers, lm_head, argmax, state bump — captured
   as one CUDA graph: one launch per token, no host round-trips

## Current limitations (honest list)

- KV cache is fp16 (matches the vendor's fastest measured config — their 4-bit KV
  mode benched *slower* on their own fork: 82.7 vs 85.5 tok/s here); in-attend
  q4 dequant is planned
- 1-bit (Q1_0) loads and runs but its GEMV inner loop is not yet tuned
- No RTX 5090 numbers yet (our cloud provider has no 5090s; `CUDA_ARCHS=120` is
  wired for owners — measurements welcome)
- Decode only, batch 1 only, greedy only; prompt processing is sequential
- DSpark drafter not integrated

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

## Quickstart (any Linux box with an NVIDIA GPU — no cloud account needed)

```bash
# 1. weights (~11 GB: ternary + 1-bit packs; needs `pip install huggingface_hub`)
./scripts/fetch_weights.sh

# 2. vendor baseline to compare against (their llama.cpp fork, pinned SHA)
./scripts/build_vendor_fork.sh          # CUDA_ARCHS=120 for RTX 5090
./scripts/bench_baseline.sh             # tg128/pp512 for both packs
./scripts/trace_baseline.sh             # nsys: launches/token, GPU idle %

# 3. bonsai-turbo engine + tests
cmake -B build -G Ninja && cmake --build build -j
ctest --test-dir build
./build/bt-inspect weights/Ternary-Bonsai-27B-Q2_0.gguf --scan-code3
./build/bt-microbench                   # GEMV bandwidth + correctness
```

Maintainers run the same scripts on cloud GPUs via thin wrappers in
`infra/modal/`; nothing in the engine depends on them.

## License

Apache 2.0. This repo contains no vendor code; the PrismML llama.cpp fork is
cloned at build time for baseline comparison only.
