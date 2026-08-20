#!/usr/bin/env python
"""Aggregate an sglang torch-profiler trace: GPU kernel/memcpy time by name.

Used during the prefill/decode optimization rounds (DSv4F-Opt.md §5) to
break down where GPU time goes — e.g. locating the 762us/layer
matmul_ogs kernel at decode-sized M, or measuring GPU idle gaps.

Usage:
    python3 tests/analyze_trace.py <trace.json.gz> [--window 0.3]
      --window F   only aggregate events after fraction F of the trace
                   span (skip prefill warm-up / capture noise)

Output: total GPU busy time, per-kernel aggregate table (top 20).
"""
import argparse
import gzip
import json
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--window", type=float, default=0.0,
                    help="skip the first F fraction of the trace span")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    with gzip.open(args.trace, "rt") as f:
        data = json.load(f)
    ev = data["traceEvents"]
    gpu = sorted(
        (e["ts"], e["ts"] + e["dur"], e.get("cat", ""), e["name"])
        for e in ev
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    )
    if not gpu:
        print("no GPU events found")
        return
    t0, t1 = gpu[0][0], gpu[-1][1]
    lo = t0 + (t1 - t0) * args.window
    sel = [g for g in gpu if g[0] >= lo]
    span = (sel[-1][1] - sel[0][0]) / 1e6

    agg = defaultdict(float)
    cnt = defaultdict(int)
    for s, t, c, n in sel:
        agg[n[:90]] += (t - s) / 1e3
        cnt[n[:90]] += 1
    busy = sum(agg.values())
    print(f"window {span:.2f}s, {len(sel)} events, GPU busy {busy/1e3:.2f}s "
          f"({busy/1e3/span*100:.0f}%), idle {(span - busy/1e3):.2f}s")
    for n, ms in sorted(agg.items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"{ms:9.1f} ms  x{cnt[n]:<6} {n}")


if __name__ == "__main__":
    main()
