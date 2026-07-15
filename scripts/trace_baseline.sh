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

SHIM="$OUT_DIR/launch_counter.so"
echo "== building launch counter"
gcc -shared -fPIC "$ROOT/tools/launch_counter.c" -ldl -o "$SHIM"

# network/FUSE-mounted weights are slow to load repeatedly; stage locally
if [ "${TRACE_STAGE_LOCAL:-0}" = "1" ]; then
    echo "== staging model to local disk"
    cp -f "$MODEL" /tmp/trace-model.gguf
    MODEL=/tmp/trace-model.gguf
fi

N_LO=8
N_HI="${TRACE_N_HI:-40}"
BENCH="$FORK_DIR/build/bin/llama-bench"
for N in "$N_LO" "$N_HI"; do
    echo "== profiled run: n=$N decode tokens (llama-bench, tg only)"
    if [ "$N" = "$N_HI" ]; then
        # sample SM utilization during the long run: NVML "utilization.gpu" is
        # the fraction of time a kernel was resident = 1 - idle-between-launches
        nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
            --loop-ms=100 > "$OUT_DIR/smi_util.csv" &
        SMI_PID=$!
    fi
    LAUNCH_COUNTER_OUT="$OUT_DIR/shim_n$N.json" LD_PRELOAD="$SHIM" \
        timeout 900 "$BENCH" -m "$MODEL" -p 0 -n "$N" -r 1 -o json \
        > "$OUT_DIR/bench_trace_n$N.json" 2> "$OUT_DIR/bench_trace_n$N.stderr"
    if [ "$N" = "$N_HI" ]; then
        kill "$SMI_PID" 2>/dev/null || true
    fi
done

python3 "$ROOT/scripts/analyze_trace.py" "$OUT_DIR" "$N_LO" "$N_HI" | tee "$OUT_DIR/trace_summary.json"
