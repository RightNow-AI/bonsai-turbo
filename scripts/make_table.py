#!/usr/bin/env python3
"""Assemble the measured-results table from raw bench outputs.

Reads (whatever exists):
  bench_<tag>.json    vendor llama-bench json (tg128/pp512)
  ours_<tag>.txt      bt-run --bench output ("decode: N tokens in X ms = Y tok/s")
Writes results/summary.md with only measured numbers; vendor-published figures
are labeled as such and never mixed with measurements.
"""
import json
import re
import sys
from pathlib import Path

VENDOR_PUBLISHED = {"ternary": 98.0, "onebit": 104.8}  # H100 tg128, model cards


def vendor_tg(raw: Path, tag: str):
    f = raw / f"bench_{tag}.json"
    if not f.exists():
        return None
    for row in json.loads(f.read_text()):
        if row.get("n_gen"):
            return row["avg_ts"], row.get("stddev_ts", 0.0)
    return None


def ours_tg(raw: Path, tag: str):
    f = raw / f"ours_{tag}.txt"
    if not f.exists():
        return None
    m = re.search(r"= ([\d.]+) tok/s", f.read_text())
    return float(m.group(1)) if m else None


def main():
    raw = Path(sys.argv[1] if len(sys.argv) > 1 else "results/raw")
    out = ["# Measured results (tg128, batch 1, greedy)", ""]
    gpu = raw / "gpu_info.csv"
    if gpu.exists():
        out += ["GPU under test: " + gpu.read_text().strip().splitlines()[-1], ""]
    out += ["| pack | vendor fork (measured) | eager | CUDA graph | megakernel | best speedup | vendor published |",
            "|---|---|---|---|---|---|---|"]
    for pack in ("ternary", "onebit"):
        v = vendor_tg(raw, f"{pack}_default")
        oe = ours_tg(raw, f"{pack}_eager")
        og = ours_tg(raw, f"{pack}_graph")
        om = ours_tg(raw, f"{pack}_mega")
        best = max([x for x in (oe, og, om) if x], default=None)
        speedup = f"{best / v[0]:.2f}x" if v and best else "-"
        out.append(
            f"| {pack} | {f'{v[0]:.1f} +/- {v[1]:.1f}' if v else '-'} "
            f"| {f'{oe:.1f}' if oe else '-'} | {f'{og:.1f}' if og else '-'} "
            f"| {f'{om:.1f}' if om else '-'} "
            f"| {speedup} | {VENDOR_PUBLISHED[pack]:.1f} |")
    trace = raw / "trace_summary.json"
    if trace.exists():
        out += ["", "## Vendor fork per-token decode profile (nsys)", "```",
                trace.read_text().strip(), "```"]
    text = "\n".join(out) + "\n"
    (raw.parent / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
