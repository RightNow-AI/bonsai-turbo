#!/usr/bin/env bash
# Logit-parity gate: bonsai-turbo vs the vendor fork, greedy decode, per-step
# logits compared over every prompt in parity_prompts.txt.
# Works on any Linux box with an NVIDIA GPU; no cloud dependency.
#
# Env: FORK_DIR, WEIGHTS_DIR, OUT_DIR, BT_BUILD (our cmake build dir),
#      PARITY_MODEL (default ternary), N_GEN (default 64)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK_DIR="${FORK_DIR:-$ROOT/third_party/llama.cpp-prismml}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$ROOT/weights}"
OUT_DIR="${OUT_DIR:-$ROOT/results/raw}/parity"
BT_BUILD="${BT_BUILD:-$ROOT/build}"
MODEL="${PARITY_MODEL:-$WEIGHTS_DIR/Ternary-Bonsai-27B-Q2_0.gguf}"
N_GEN="${N_GEN:-64}"
mkdir -p "$OUT_DIR"

echo "== building vendor-logits against the fork's libllama"
g++ -O2 -I"$FORK_DIR/include" -I"$FORK_DIR/ggml/include" \
    "$ROOT/tools/vendor_logits.cpp" \
    -L"$FORK_DIR/build/bin" -lllama -Wl,-rpath,"$FORK_DIR/build/bin" \
    -o "$OUT_DIR/vendor-logits"

echo "== tokenizing prompts with the vendor tokenizer"
: > "$OUT_DIR/ids.txt"
while IFS= read -r prompt; do
    [ -z "$prompt" ] && continue
    "$FORK_DIR/build/bin/llama-tokenize" -m "$MODEL" -p "$prompt" --ids \
        | tr -d '[] ' >> "$OUT_DIR/ids.txt"
done < "$ROOT/scripts/parity_prompts.txt"
N_PROMPTS=$(wc -l < "$OUT_DIR/ids.txt")
echo "   $N_PROMPTS prompts"

echo "== vendor pass"
"$OUT_DIR/vendor-logits" "$MODEL" "$OUT_DIR/ids.txt" "$N_GEN" "$OUT_DIR/vendor" \
    | tee "$OUT_DIR/vendor_tokens.txt"

echo "== bonsai-turbo pass"
"$BT_BUILD/bt-run" --model "$MODEL" --ids-file "$OUT_DIR/ids.txt" --n "$N_GEN" \
    ${BT_RUN_FLAGS:-} --logits-out "$OUT_DIR/ours" | tee "$OUT_DIR/ours_tokens.txt"

echo "== comparing"
PASS=0
FAIL=0
for i in $(seq 0 $((N_PROMPTS - 1))); do
    if python3 "$ROOT/scripts/compare_logits.py" \
        "$OUT_DIR/vendor.$i.bin" "$OUT_DIR/ours.$i.bin" "$N_GEN"; then
        PASS=$((PASS + 1))
    else
        echo "   prompt $i FAILED"
        FAIL=$((FAIL + 1))
    fi
done

echo "parity: $PASS/$((PASS + FAIL)) prompts pass"
[ "$FAIL" -eq 0 ]
