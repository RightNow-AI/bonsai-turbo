#!/usr/bin/env python3
"""Summarize the vendor fork's per-token decode profile.

Inputs (from trace_baseline.sh):
  bench_trace.stderr  fork log with "graph nodes = N" (ops per decode graph)
  bench_trace.json    llama-bench tg row (wall time per token)
  smi_util.csv        NVML utilization.gpu samples (100 ms) during the run

graph nodes/token >= kernel launches/token (some nodes are views/reshapes with
no kernel; CUDA graphs amortize CPU launch cost but each node still executes).
NVML utilization = fraction of time any kernel was resident on the GPU, i.e.
1 - idle-between-kernels, sampled coarsely.
"""
import json
import re
import statistics
import sys
from pathlib import Path


def main() -> None:
    out_dir = Path(sys.argv[1])
    log = (out_dir / "bench_trace.stderr").read_text(errors="replace")
    nodes = [int(m) for m in re.findall(r"graph nodes\s*=\s*(\d+)", log)]
    splits = [int(m) for m in re.findall(r"graph splits\s*=\s*(\d+)", log)]

    rows = json.loads((out_dir / "bench_trace.json").read_text())
    tg = next(r for r in rows if r.get("n_gen"))
    ms_per_tok = 1000.0 / tg["avg_ts"]

    busy = None
    smi = out_dir / "smi_util.csv"
    if smi.exists():
        samples = sorted(int(l) for l in smi.read_text().split() if l.strip().isdigit())
        if samples:
            top = samples[3 * len(samples) // 4:]  # decode-active window
            busy = statistics.median(top) if top else None

    print(json.dumps({
        "graph_nodes_per_token": max(nodes) if nodes else None,
        "graph_splits": max(splits) if splits else None,
        "decode_wall_ms_per_token": round(ms_per_tok, 3),
        "measured_tok_s": round(tg["avg_ts"], 2),
        "gpu_busy_pct_nvml": busy,
        "gpu_idle_pct_nvml": None if busy is None else round(100 - busy, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
