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

CUPTI_INC="${CUPTI_INC:-}"
CUPTI_LIB="${CUPTI_LIB:-}"
if [ -z "$CUPTI_INC" ]; then
    for d in /usr/local/cuda/extras/CUPTI/include /usr/local/cuda/include /usr/include; do
        [ -f "$d/cupti.h" ] && CUPTI_INC="$d" && break
    done
fi
if [ -z "$CUPTI_LIB" ]; then
    for d in /usr/local/cuda/extras/CUPTI/lib64 /usr/local/cuda/lib64 \
             /usr/lib/x86_64-linux-gnu; do
        ls "$d"/libcupti.so* >/dev/null 2>&1 && CUPTI_LIB="$d" && break
    done
fi
if [ -z "$CUPTI_INC" ] || [ -z "$CUPTI_LIB" ]; then
    echo "!! CUPTI not found (install cuda-cupti-dev); skipping trace" >&2
    exit 1
fi
SHIM="$OUT_DIR/cupti_shim.so"
echo "== building CUPTI shim"
gcc -shared -fPIC "$ROOT/tools/cupti_shim.c" -I"$CUPTI_INC" -L"$CUPTI_LIB" \
    -lcupti -Wl,-rpath,"$CUPTI_LIB" -o "$SHIM"

# network/FUSE-mounted weights are slow to load repeatedly; stage locally
if [ "${TRACE_STAGE_LOCAL:-0}" = "1" ]; then
    echo "== staging model to local disk"
    cp -f "$MODEL" /tmp/trace-model.gguf
    MODEL=/tmp/trace-model.gguf
fi

N_LO=8
N_HI="${TRACE_N_HI:-40}"
for N in "$N_LO" "$N_HI"; do
    echo "== profiled run: n=$N decode tokens"
    CUPTI_SHIM_OUT="$OUT_DIR/shim_n$N.json" LD_PRELOAD="$SHIM" \
        timeout 900 "$CLI" -m "$MODEL" -p "$PROMPT" -n "$N" -ngl 99 -no-cnv \
        --temp 0 --seed 1 \
        > "$OUT_DIR/cli_n$N.stdout" 2> "$OUT_DIR/cli_n$N.stderr"
done

python3 "$ROOT/scripts/analyze_trace.py" "$OUT_DIR" "$N_LO" "$N_HI" | tee "$OUT_DIR/trace_summary.json"
