#!/usr/bin/env bash
# Reproduce the vendor's batch-1 decode baselines with their own llama.cpp fork.
# Runs llama-bench (pp512 + tg128, 3 reps) for each pack, with default KV and
# with the 4-bit quantized KV cache the model card describes.
# Works on any Linux box with an NVIDIA GPU; no cloud dependency.
#
# Env overrides: FORK_DIR, WEIGHTS_DIR, OUT_DIR, BENCH_REPS
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK_DIR="${FORK_DIR:-$ROOT/third_party/llama.cpp-prismml}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$ROOT/weights}"
OUT_DIR="${OUT_DIR:-$ROOT/results/raw}"
BENCH="$FORK_DIR/build/bin/llama-bench"
mkdir -p "$OUT_DIR"

TERNARY="$WEIGHTS_DIR/Ternary-Bonsai-27B-Q2_0.gguf"
ONEBIT="$WEIGHTS_DIR/Bonsai-27B-Q1_0.gguf"

echo "== GPU under test"
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.sm,clocks.mem,power.limit \
    --format=csv | tee "$OUT_DIR/gpu_info.csv"

run_bench () {
    local model="$1" tag="$2"; shift 2
    if [ ! -f "$model" ]; then
        echo "-- skip $tag (missing $(basename "$model"))"
        return 0
    fi
    echo "== llama-bench: $tag $*"
    if ! "$BENCH" -m "$model" -p 512 -n 128 -r "${BENCH_REPS:-3}" -o json "$@" \
        > "$OUT_DIR/bench_${tag}.json" 2> "$OUT_DIR/bench_${tag}.stderr"; then
        echo "!! bench $tag FAILED; stderr tail:"
        tail -5 "$OUT_DIR/bench_${tag}.stderr"
        return 0
    fi
    python3 - "$OUT_DIR/bench_${tag}.json" "$tag" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
for r in rows:
    kind = f"pp{r['n_prompt']}" if r.get("n_prompt") else f"tg{r['n_gen']}"
    print(f"   {sys.argv[2]:24s} {kind:8s} {r['avg_ts']:8.2f} +/- {r.get('stddev_ts', 0):.2f} tok/s")
PY
}

# Default KV (F16) — what plain `llama-bench -m model.gguf` measures
run_bench "$TERNARY" ternary_default
run_bench "$ONEBIT"  onebit_default

# 4-bit quantized KV cache, as described on the model card
run_bench "$TERNARY" ternary_kvq4 -ctk q4_0 -ctv q4_0
run_bench "$ONEBIT"  onebit_kvq4 -ctk q4_0 -ctv q4_0

echo "== raw results in $OUT_DIR"
