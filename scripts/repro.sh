#!/usr/bin/env bash
# One-command repro: fetch weights, build the vendor fork and bonsai-turbo,
# verify logit parity, bench both engines, print the comparison table.
# Requirements: Linux, NVIDIA GPU + CUDA toolkit, cmake, git, python3
# (pip install huggingface_hub for the weight fetch). No cloud account needed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==== [1/6] weights"
./scripts/fetch_weights.sh

echo "==== [2/6] vendor fork (pinned SHA)"
./scripts/build_vendor_fork.sh

echo "==== [3/6] bonsai-turbo"
cmake -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j "$(nproc)"
ctest --test-dir build --output-on-failure

echo "==== [4/6] parity gate (correctness before any speed claim)"
./scripts/parity.sh

echo "==== [5/6] benches"
./scripts/bench_baseline.sh
./scripts/bench_ours.sh
./scripts/trace_baseline.sh || echo "(trace optional: nsys missing or failed)"

echo "==== [6/6] table"
python3 ./scripts/make_table.py "${OUT_DIR:-$ROOT/results/raw}"
