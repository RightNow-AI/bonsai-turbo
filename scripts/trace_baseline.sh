#!/usr/bin/env bash
# Capture nsys traces of the vendor fork decoding, and derive per-token launch
# counts and GPU idle fraction. Method: run the identical prompt twice with
# -n 8 and -n 72 generated tokens; every count is differenced between the two
# runs, so prompt processing, warmup, and load noise cancel and what remains is
# the steady-state per-token decode cost over 64 tokens.
# Works on any Linux box with an NVIDIA GPU + nsight-systems; no cloud dependency.
#
# Env overrides: FORK_DIR, WEIGHTS_DIR, OUT_DIR, TRACE_MODEL
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK_DIR="${FORK_DIR:-$ROOT/third_party/llama.cpp-prismml}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$ROOT/weights}"
OUT_DIR="${OUT_DIR:-$ROOT/results/raw}"
CLI="$FORK_DIR/build/bin/llama-cli"
MODEL="${TRACE_MODEL:-$WEIGHTS_DIR/Ternary-Bonsai-27B-Q2_0.gguf}"
PROMPT="The roofline model of GPU performance says"
mkdir -p "$OUT_DIR"

if ! command -v nsys >/dev/null 2>&1; then
    echo "!! nsys not found; install nsight-systems to capture traces" >&2
    exit 1
fi

for N in 8 72; do
    echo "== nsys profile: n=$N decode tokens"
    nsys profile -t cuda --force-overwrite true -o "$OUT_DIR/trace_n$N" \
        "$CLI" -m "$MODEL" -p "$PROMPT" -n "$N" -ngl 99 -no-cnv --temp 0 --seed 1 \
        > "$OUT_DIR/cli_n$N.stdout" 2> "$OUT_DIR/cli_n$N.stderr"
    nsys stats --report cuda_api_sum --report cuda_gpu_kern_sum --format csv \
        --output "$OUT_DIR/trace_n$N" --force-export true "$OUT_DIR/trace_n$N.nsys-rep" >/dev/null
done

python3 "$ROOT/scripts/analyze_trace.py" "$OUT_DIR" 8 72 | tee "$OUT_DIR/trace_summary.json"
