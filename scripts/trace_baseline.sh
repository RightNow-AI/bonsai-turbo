#!/usr/bin/env bash
# Per-token decode profile of the vendor fork: kernel executions per token and
# GPU busy fraction, via an in-process CUPTI shim (no profiler daemon — works
# in unprivileged containers where nsys stalls).
#
# Method: run the identical prompt with -n 8 and -n 40 generated tokens; every
# counter is differenced between runs, so load/warmup/prompt costs cancel and
# what remains is steady-state per-token decode cost.
# Works on any Linux box with an NVIDIA GPU + CUDA toolkit; no cloud dependency.
#
# Env overrides: FORK_DIR, WEIGHTS_DIR, OUT_DIR, TRACE_MODEL, TRACE_N_HI
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK_DIR="${FORK_DIR:-$ROOT/third_party/llama.cpp-prismml}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$ROOT/weights}"
OUT_DIR="${OUT_DIR:-$ROOT/results/raw}"
CLI="$FORK_DIR/build/bin/llama-cli"
MODEL="${TRACE_MODEL:-$WEIGHTS_DIR/Ternary-Bonsai-27B-Q2_0.gguf}"
PROMPT="The roofline model of GPU performance says"
mkdir -p "$OUT_DIR"

# network/FUSE-mounted weights are slow to load repeatedly; stage locally
if [ "${TRACE_STAGE_LOCAL:-0}" = "1" ]; then
    echo "== staging model to local disk"
    cp -f "$MODEL" /tmp/trace-model.gguf
    MODEL=/tmp/trace-model.gguf
fi

# GPU-side profiling (nsys/CUPTI) is administratively blocked on most cloud
# runners and ggml's static CUDA runtime defeats LD_PRELOAD interposition, so
# the honest observables are:
#   - ops per decode graph, from the fork's own "graph nodes = N" log line
#     (>= kernel launches per token; CUDA graphs amortize the CPU-side cost)
#   - NVML utilization.gpu sampled during a long tg run = fraction of time a
#     kernel was resident = 1 - idle-between-kernels
BENCH="$FORK_DIR/build/bin/llama-bench"
N="${TRACE_N:-512}"
echo "== profiled run: tg $N with NVML sampling"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
    --loop-ms=100 > "$OUT_DIR/smi_util.csv" &
SMI_PID=$!
timeout 900 "$BENCH" -m "$MODEL" -p 0 -n "$N" -r 1 -o json -v \
    > "$OUT_DIR/bench_trace.json" 2> "$OUT_DIR/bench_trace.stderr"
kill "$SMI_PID" 2>/dev/null || true

python3 "$ROOT/scripts/analyze_trace.py" "$OUT_DIR" | tee "$OUT_DIR/trace_summary.json"
