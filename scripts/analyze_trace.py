#!/usr/bin/env python3
"""Derive per-token decode cost from two CUPTI-shim profiled runs.

trace_baseline.sh records the vendor CLI generating N_LO and N_HI tokens.
Everything outside decode (model load, warmup, prompt processing) is identical
between the runs, so differencing cancels it:

    kernels_per_token = (kernels_hi - kernels_lo) / (n_hi - n_lo)
    gpu_busy_fraction = (kernel_ns_hi - kernel_ns_lo) / (eval_wall_hi - eval_wall_lo)

Inputs per run: shim_n{N}.json (cupti_shim output) and cli_n{N}.stderr
(llama-cli perf print for the eval wall time).
"""
import json
import re
import sys
from pathlib import Path


def read_shim(out_dir: Path, n: int) -> dict:
    return json.loads((out_dir / f"shim_n{n}.json").read_text())


def eval_wall_ms(out_dir: Path, n: int) -> tuple[float, int]:
    text = (out_dir / f"cli_n{n}.stderr").read_text(errors="replace")
    m = re.search(r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs", text)
    if not m:
        raise ValueError(f"no eval-time line in cli_n{n}.stderr")
    return float(m.group(1)), int(m.group(2))


def main() -> None:
    out_dir, n_lo, n_hi = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    lo, hi = read_shim(out_dir, n_lo), read_shim(out_dir, n_hi)
    (wall_lo, runs_lo), (wall_hi, runs_hi) = eval_wall_ms(out_dir, n_lo), eval_wall_ms(out_dir, n_hi)

    d_tok = runs_hi - runs_lo
    d_wall_ms = wall_hi - wall_lo
    d_kern_ms = (hi["kernel_ns"] - lo["kernel_ns"]) / 1e6
    d_mem_ms = (hi["memop_ns"] - lo["memop_ns"]) / 1e6
    busy = (d_kern_ms + d_mem_ms) / d_wall_ms if d_wall_ms > 0 else float("nan")

    print(json.dumps({
        "decode_tokens_differenced": d_tok,
        "kernels_per_token": round((hi["kernels"] - lo["kernels"]) / d_tok, 1),
        "memops_per_token": round((hi["memops"] - lo["memops"]) / d_tok, 1),
        "decode_wall_ms_per_token": round(d_wall_ms / d_tok, 3),
        "gpu_kernel_ms_per_token": round(d_kern_ms / d_tok, 3),
        "gpu_busy_fraction_decode": round(busy, 3),
        "gpu_idle_pct_decode": round(100 * (1 - busy), 1),
        "implied_tok_s_from_wall": round(1000 * d_tok / d_wall_ms, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
