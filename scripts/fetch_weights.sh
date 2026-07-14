#!/usr/bin/env bash
# Fetch Bonsai 27B GGUF packs from Hugging Face into WEIGHTS_DIR.
# Requires: python3 with `pip install "huggingface_hub[hf_transfer]"`.
# No cloud account needed; the models are public (Apache 2.0).
#
# Usage:
#   ./scripts/fetch_weights.sh            # ternary + 1-bit packs (~11 GB)
#   FETCH_F16=1 ./scripts/fetch_weights.sh    # also the 53.8 GB F16 parity reference
#   FETCH_DSPARK=1 ./scripts/fetch_weights.sh # also the DSpark drafter (1.95 GB)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${WEIGHTS_DIR:-$ROOT/weights}"
mkdir -p "$WEIGHTS_DIR"

fetch () {
    local repo="$1" file="$2"
    echo "== fetching $repo :: $file"
    python3 - "$repo" "$file" "$WEIGHTS_DIR" <<'PY'
import sys
from huggingface_hub import hf_hub_download
repo, file, out = sys.argv[1], sys.argv[2], sys.argv[3]
path = hf_hub_download(repo_id=repo, filename=file, local_dir=out)
print(f"   -> {path}")
PY
}

fetch prism-ml/Ternary-Bonsai-27B-gguf Ternary-Bonsai-27B-Q2_0.gguf
fetch prism-ml/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf

if [ "${FETCH_DSPARK:-0}" = "1" ]; then
    fetch prism-ml/Ternary-Bonsai-27B-gguf Ternary-Bonsai-27B-dspark-Q4_1.gguf
fi

if [ "${FETCH_F16:-0}" = "1" ]; then
    fetch prism-ml/Ternary-Bonsai-27B-gguf Ternary-Bonsai-27B-F16.gguf
fi

echo "== done. weights in $WEIGHTS_DIR"
