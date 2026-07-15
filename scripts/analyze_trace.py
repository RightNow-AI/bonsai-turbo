#!/usr/bin/env python3
"""Derive per-token decode cost from two launch-counted runs.

trace_baseline.sh runs llama-bench tg-only with N_LO and N_HI tokens under an
LD_PRELOAD launch counter. Everything outside decode (load, warmup) is
identical between runs, so differencing cancels it:

    launches_per_token = (launches_hi - launches_lo) / (n_hi - n_lo)

GPU busy fraction comes from NVML utilization sampled during the long run
(fraction of time a kernel was resident = 1 - idle between launches). The
top-quartile median is reported to exclude load/idle phases of the sample
window.
"""
import json
import statistics
import sys
from pathlib import Path


def read_shim(out_dir: Path, n: int) -> dict:
    return json.loads((out_dir / f"shim_n{n}.json").read_text())


def eval_wall_ms(out_dir: Path, n: int) -> tuple[float, int]:
    rows = json.loads((out_dir / f"bench_trace_n{n}.json").read_text())
    for row in rows:
        if row.get("n_gen"):
            return 1000.0 * row["n_gen"] / row["avg_ts"], int(row["n_gen"])
    raise ValueError(f"no tg row in bench_trace_n{n}.json")


def main() -> None:
    out_dir, n_lo, n_hi = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    lo, hi = read_shim(out_dir, n_lo), read_shim(out_dir, n_hi)
    (wall_lo, runs_lo), (wall_hi, runs_hi) = eval_wall_ms(out_dir, n_lo), eval_wall_ms(out_dir, n_hi)

    d_tok = runs_hi - runs_lo
    d_wall_ms = wall_hi - wall_lo

    busy_pct = None
    smi = out_dir / "smi_util.csv"
    if smi.exists():
        samples = sorted(int(l) for l in smi.read_text().split() if l.strip().isdigit())
        if samples:
            top = samples[3 * len(samples) // 4:]  # decode-active samples
            busy_pct = statistics.median(top) if top else None

    print(json.dumps({
        "decode_tokens_differenced": d_tok,
        "api_launches_per_token": round((hi["launches"] - lo["launches"]) / d_tok, 1),
        "graph_launches_per_token": round((hi["graph_launches"] - lo["graph_launches"]) / d_tok, 2),
        "memops_per_token": round((hi["memops"] - lo["memops"]) / d_tok, 1),
        "decode_wall_ms_per_token": round(d_wall_ms / d_tok, 3),
        "gpu_busy_pct_nvml": busy_pct,
        "gpu_idle_pct_nvml": None if busy_pct is None else round(100 - busy_pct, 1),
        "implied_tok_s_from_wall": round(1000 * d_tok / d_wall_ms, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
