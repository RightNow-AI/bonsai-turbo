# Measured results

All numbers below were measured by the maintainers on 2026-07-15 on a cloud
H100 80GB SXM (driver 580.95.05, CUDA 12.8). Raw outputs are not tracked;
regenerate them with `scripts/repro.sh` on your own machine.

## Decode speed (tg128-comparable: short prompt, 128 greedy tokens, batch 1)

| engine | ternary tok/s | 1-bit tok/s |
|---|---|---|
| vendor llama.cpp fork @ 62061f91, llama-bench tg128, 3 reps | 85.5 +/- 6.9 | 90.1 +/- 3.5 |
| bonsai-turbo, CUDA graph mode | 151.1 | 133.0 |
| bonsai-turbo, megakernel | 149.6 | 158.7 |

Run-to-run spread for bonsai-turbo across containers was 146.8 to 151.4
(ternary, graph mode). A second vendor-fork host measured 94.6 ternary. The
1-bit megakernel (158.7) is the fastest engine measured; the 1-bit pack reads
half the weight bytes per token. Both 1-bit engines pass 32/32 logit parity.

## Correctness

- Logit parity: 32/32 prompts pass (`scripts/parity.sh`, 64 steps each).
  Greedy top-1 identical at every step, or the flip was a tie where the
  vendor's own top-1/top-2 margin was at most 0.034. Pre-divergence top-20
  logit deltas stayed under 1.2 (the measured noise floor between the two
  engines' int8 activation quantization group sizes).
- MATH-500 subset: in progress. Best so far 66/100 with greedy decode at an
  8k token budget; of the problems whose generation completed, 64 of 66
  graded correct. Sampled runs with the model's recommended settings are in
  flight.

## Vendor fork decode profile (basis for the design)

From the fork's own logs and NVML sampling (`scripts/trace_baseline.sh`),
ternary pack, tg512: 3703 graph nodes per decode token, 2 graph splits,
10.6 ms per token, NVML busy 97% during decode.
