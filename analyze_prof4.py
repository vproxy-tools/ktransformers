#!/usr/bin/env python
"""打印一个 decode step 内的 GPU kernel 时间顺序（带 grid/block）。"""
import gzip
import json
import sys

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)
events = data["traceEvents"]

launches = sorted((e["ts"], e["ts"] + e["dur"]) for e in events
                  if e.get("ph") == "X" and e.get("name") == "cudaGraphLaunch"
                  and e.get("cat") == "cuda_runtime")

gpu_events = []
for e in events:
    if e.get("ph") == "X" and e.get("cat", "") in ("kernel", "gpu_memcpy", "gpu_memset"):
        args = e.get("args", {})
        grid = args.get("grid", [])
        block = args.get("block", [])
        regs = args.get("registers per thread", "")
        gpu_events.append((e["ts"], e["dur"], e["name"], grid, block))
gpu_events.sort()

# 取第 2~3 个 step 之间
lo = launches[1][1] if len(launches) > 2 else launches[0][1]
hi = launches[2][0] if len(launches) > 2 else launches[-1][0]
step = [g for g in gpu_events if g[0] >= lo and g[1] <= hi]

print(f"step kernels: {len(step)}, span {(hi-lo)/1e3:.2f} ms")
t0 = step[0][0] if step else 0
for ts, dur, name, grid, block in step:
    short = name.replace("std::enable_if<!(false), void>::type internal::", "") \
                .replace("void ", "").replace("(anonymous namespace)::", "")[:64]
    g = f"{grid}" if grid else ""
    print(f"  +{(ts-t0)/1e3:>8.3f}ms {dur:>6.1f}us  {short}  {g}")
