#!/usr/bin/env bash
# Bench bonsai-turbo decode (tg128-comparable): fixed short prompt, 128 greedy
# tokens, eager and CUDA-graph modes, both packs when present.
# Works on any Linux box with an NVIDIA GPU; no cloud dependency.
#
# Env overrides: WEIGHTS_DIR, OUT_DIR, BT_BUILD, BENCH_N
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${WEIGHTS_DIR:-$ROOT/weights}"
OUT_DIR="${OUT_DIR:-$ROOT/results/raw}"
BT_BUILD="${BT_BUILD:-$ROOT/build}"
N="${BENCH_N:-128}"
IDS="785,9426,1614,315,22670,5068,2727,429"
mkdir -p "$OUT_DIR"

nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem \
    --format=csv,noheader | tee "$OUT_DIR/gpu_info_ours.csv"

run_one () {
    local model="$1" tag="$2"; shift 2
    if [ ! -f "$model" ]; then
        echo "-- skip $tag (missing $(basename "$model"))"
        return 0
    fi
    echo "== bt-run: $tag $*"
    if ! "$BT_BUILD/bt-run" --model "$model" --ids "$IDS" --n "$N" --bench "$@" \
        > "$OUT_DIR/ours_${tag}.txt" 2> "$OUT_DIR/ours_${tag}.stderr"; then
        echo "!! $tag FAILED; stderr tail:"
        tail -5 "$OUT_DIR/ours_${tag}.stderr"
        return 0
    fi
    grep "^decode:" "$OUT_DIR/ours_${tag}.txt"
}

run_one "$WEIGHTS_DIR/Ternary-Bonsai-27B-Q2_0.gguf" ternary_eager
run_one "$WEIGHTS_DIR/Ternary-Bonsai-27B-Q2_0.gguf" ternary_graph --graph
run_one "$WEIGHTS_DIR/Bonsai-27B-Q1_0.gguf" onebit_eager
run_one "$WEIGHTS_DIR/Bonsai-27B-Q1_0.gguf" onebit_graph --graph

echo "== raw results in $OUT_DIR"
