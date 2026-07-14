#!/usr/bin/env python3
"""Derive per-token decode cost from two nsys traces of the same prompt.

trace_baseline.sh records the vendor CLI generating N_LO and N_HI tokens.
Everything outside decode (model load, warmup, prompt processing) is identical
between the runs, so differencing cancels it:

    launches_per_token = (api_launches_hi - api_launches_lo) / (n_hi - n_lo)
    kernels_per_token  = (gpu_kernels_hi  - gpu_kernels_lo)  / (n_hi - n_lo)
    gpu_busy_fraction  = (kernel_time_hi - kernel_time_lo) / (eval_wall_hi - eval_wall_lo)

Inputs per run, produced by `nsys stats --format csv`:
    trace_n{N}_cuda_api_sum.csv       host-side API calls (launch count)
    trace_n{N}_cuda_gpu_kern_sum.csv  device-side kernel executions + durations
    cli_n{N}.stderr                   llama-cli perf print (eval wall time)
"""
import csv
import json
import re
import sys
from pathlib import Path

LAUNCH_APIS = {
    "cudaLaunchKernel",
    "cudaLaunchKernelExC",
    "cuLaunchKernel",
    "cuLaunchKernelEx",
    "cudaGraphLaunch",
    "cuGraphLaunch",
}


def find_csv(out_dir: Path, n: int, report: str) -> Path:
    """nsys names exports like <output>_<report>.csv, with minor version drift."""
    candidates = sorted(out_dir.glob(f"trace_n{n}*{report}*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no {report} csv for n={n} in {out_dir}")
    return candidates[0]


def read_rows(path: Path) -> list[dict]:
    # nsys sometimes emits comment/blank lines before the header
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    start = next(i for i, l in enumerate(lines) if "," in l and ("Name" in l or "name" in l))
    return list(csv.DictReader(lines[start:]))


def col(row: dict, *needles: str):
    for key in row:
        k = key.lower()
        if all(n.lower() in k for n in needles):
            return row[key]
    raise KeyError(f"no column matching {needles} in {list(row)}")


def api_launches(out_dir: Path, n: int) -> int:
    total = 0
    for row in read_rows(find_csv(out_dir, n, "cuda_api_sum")):
        name = col(row, "name").strip()
        if name in LAUNCH_APIS:
            total += int(col(row, "num", "calls").replace(",", ""))
    return total


def gpu_kernels(out_dir: Path, n: int) -> tuple[int, int]:
    count, time_ns = 0, 0
    for row in read_rows(find_csv(out_dir, n, "cuda_gpu_kern_sum")):
        count += int(col(row, "instances").replace(",", ""))
        time_ns += int(col(row, "total time").replace(",", ""))
    return count, time_ns


def eval_wall_ms(out_dir: Path, n: int) -> tuple[float, int]:
    """Parse `eval time = X ms / Y runs` from llama-cli perf output."""
    text = (out_dir / f"cli_n{n}.stderr").read_text(errors="replace")
    m = re.search(r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs", text)
    if not m:
        raise ValueError(f"no eval-time line in cli_n{n}.stderr")
    return float(m.group(1)), int(m.group(2))


def main() -> None:
    out_dir, n_lo, n_hi = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

    api_lo, api_hi = api_launches(out_dir, n_lo), api_launches(out_dir, n_hi)
    (kern_lo, ns_lo), (kern_hi, ns_hi) = gpu_kernels(out_dir, n_lo), gpu_kernels(out_dir, n_hi)
    (wall_lo, runs_lo), (wall_hi, runs_hi) = eval_wall_ms(out_dir, n_lo), eval_wall_ms(out_dir, n_hi)

    d_tok = runs_hi - runs_lo
    d_wall_ms = wall_hi - wall_lo
    d_kern_ms = (ns_hi - ns_lo) / 1e6
    busy = d_kern_ms / d_wall_ms if d_wall_ms > 0 else float("nan")

    print(json.dumps({
        "decode_tokens_differenced": d_tok,
        "api_launches_per_token": round((api_hi - api_lo) / d_tok, 1),
        "gpu_kernels_per_token": round((kern_hi - kern_lo) / d_tok, 1),
        "decode_wall_ms_per_token": round(d_wall_ms / d_tok, 3),
        "gpu_kernel_ms_per_token": round(d_kern_ms / d_tok, 3),
        "gpu_busy_fraction_decode": round(busy, 3),
        "gpu_idle_pct_decode": round(100 * (1 - busy), 1),
        "implied_tok_s_from_wall": round(1000 * d_tok / d_wall_ms, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
