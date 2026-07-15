#!/usr/bin/env bash
# Clone and build the vendor's llama.cpp fork (baseline to beat) at a pinned SHA.
# Works on any Linux box with CUDA toolkit installed; no cloud dependency.
#
# Env overrides:
#   FORK_DIR    where to clone/build   (default: <repo>/third_party/llama.cpp-prismml)
#   CUDA_ARCHS  CMAKE_CUDA_ARCHITECTURES (default: "90" = H100; use "120" for RTX 5090)
set -euo pipefail

FORK_REPO="https://github.com/PrismML-Eng/llama.cpp"
# Pinned 2026-07-14: "ci(release): build examples so llama-speculative-simple ships..."
FORK_SHA="62061f91088281e65071cc38c5f69ee95c39f14e"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK_DIR="${FORK_DIR:-$ROOT/third_party/llama.cpp-prismml}"
CUDA_ARCHS="${CUDA_ARCHS:-90}"

if [ ! -d "$FORK_DIR/.git" ]; then
    mkdir -p "$FORK_DIR"
    git -C "$FORK_DIR" init -q
    git -C "$FORK_DIR" remote add origin "$FORK_REPO"
fi
git -C "$FORK_DIR" fetch --depth 1 origin "$FORK_SHA" || git -C "$FORK_DIR" fetch origin
git -C "$FORK_DIR" checkout -q "$FORK_SHA"

# Driverless build environments (CI containers, cloud builders) have the CUDA
# toolkit but no libcuda; link against the toolkit's stubs — the real driver
# binds at runtime. ld also chases libggml-cuda.so's NEEDED libcuda.so.1 via
# -rpath-link, and the stubs dir only ships libcuda.so, so provide a .so.1
# symlink in a scratch dir. Harmless when a driver is present (link-time only).
for STUBS in "${CUDA_HOME:-/usr/local/cuda}/lib64/stubs" /usr/local/cuda/lib64/stubs; do
    if [ -d "$STUBS" ] && [ ! -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
        STUBLINK="$(mktemp -d)"
        ln -sf "$STUBS/libcuda.so" "$STUBLINK/libcuda.so.1"
        export LIBRARY_PATH="$STUBS:${LIBRARY_PATH:-}"
        export LDFLAGS="-L$STUBS -Wl,-rpath-link,$STUBLINK ${LDFLAGS:-}"
        echo "== linking against CUDA driver stubs at $STUBS (rpath-link $STUBLINK)"
        break
    fi
done

GEN=()
if command -v ninja >/dev/null 2>&1; then GEN=(-G Ninja); fi

cmake -S "$FORK_DIR" -B "$FORK_DIR/build" "${GEN[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DLLAMA_CURL=OFF

cmake --build "$FORK_DIR/build" -j "$(nproc)" --target llama-bench llama-cli llama-tokenize

echo "== vendor fork built at $FORK_SHA"
"$FORK_DIR/build/bin/llama-bench" --version 2>&1 || true
