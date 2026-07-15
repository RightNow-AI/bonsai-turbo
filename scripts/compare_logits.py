#!/usr/bin/env python3
"""Compare two per-step logit dumps (vendor fork vs bonsai-turbo).

Both files are n_steps x n_vocab float32, written step-major by
vendor-logits and `bt-run --logits-out`. Reports per-run stats and exits
non-zero when the parity gate fails.

Gate (documented tolerance):
  - greedy top-1 token must agree on every step
  - max |delta| over each step's top-20 vendor logits must stay < 0.75
    (both engines quantize activations to int8 per-group; bit-identical
    logits are not expected, argmax stability is)
"""
import struct
import sys
from pathlib import Path

TOP1_REQUIRED = True
TOPK = 20
MAX_ABS_TOL = 0.75


def read_dump(path: Path, n_steps: int) -> list[list[float]]:
    raw = path.read_bytes()
    n = len(raw) // 4
    n_vocab = n // n_steps
    if n_steps * n_vocab != n:
        raise SystemExit(f"{path}: size {n} floats not divisible by steps {n_steps}")
    vals = struct.unpack(f"<{n}f", raw)
    return [list(vals[s * n_vocab:(s + 1) * n_vocab]) for s in range(n_steps)]


def main() -> None:
    vendor_path, ours_path, n_steps = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    vendor, ours = read_dump(vendor_path, n_steps), read_dump(ours_path, n_steps)
    if len(vendor[0]) != len(ours[0]):
        raise SystemExit(f"vocab mismatch: {len(vendor[0])} vs {len(ours[0])}")
    n_vocab = len(vendor[0])
    steps = min(len(vendor), len(ours))

    worst_abs = 0.0
    top1_miss = []
    for s in range(steps):
        vs, os_ = vendor[s], ours[s]
        v_top1 = max(range(n_vocab), key=lambda i: vs[i])
        o_top1 = max(range(n_vocab), key=lambda i: os_[i])
        if v_top1 != o_top1:
            top1_miss.append((s, v_top1, o_top1))
        topk = sorted(range(n_vocab), key=lambda i: vs[i], reverse=True)[:TOPK]
        step_abs = max(abs(vs[i] - os_[i]) for i in topk)
        worst_abs = max(worst_abs, step_abs)

    print(f"steps={steps} top1_mismatches={len(top1_miss)} "
          f"worst_top{TOPK}_abs_delta={worst_abs:.4f}")
    for s, v, o in top1_miss[:5]:
        print(f"  step {s}: vendor argmax {v} vs ours {o}")

    ok = worst_abs < MAX_ABS_TOL and (not TOP1_REQUIRED or not top1_miss)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
