#!/usr/bin/env python
"""把 GPU kernel 关联回发起它的 CPU op / python 栈。"""
import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
target = sys.argv[2] if len(sys.argv) > 2 else "gemvx"
with gzip.open(path, "rt") as f:
    data = json.load(f)
events = data["traceEvents"]

# runtime 事件: correlation -> (ts, name)
runtime = {}
for e in events:
    if e.get("ph") == "X" and e.get("cat") == "cuda_runtime":
        corr = e.get("args", {}).get("correlation")
        if corr is not None:
            runtime[corr] = (e["ts"], e["ts"] + e["dur"], e["name"])

# kernel 事件: 找目标
kernels = []
for e in events:
    if e.get("ph") == "X" and e.get("cat") in ("kernel",):
        corr = e.get("args", {}).get("correlation")
        if corr is not None and target in e["name"]:
            kernels.append((e["ts"], corr))
kernels.sort()

# cpu 侧事件栈: 按 ts 排序的 (ts, te, name, id)
cpu_stack = []
for e in events:
    if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "python_function", "user_annotation"):
        cpu_stack.append((e["ts"], e["ts"] + e["dur"], e.get("name", "?")))
cpu_stack.sort()

import bisect
starts = [c[0] for c in cpu_stack]

def enclosing(rt_ts):
    # 找 ts <= rt_ts 且 te >= rt_ts 的最内层(最短)事件
    i = bisect.bisect_right(starts, rt_ts) - 1
    best = None
    while i >= 0 and i > len(cpu_stack) - 20000:
        s, t, n = cpu_stack[i]
        if t >= rt_ts and s <= rt_ts:
            dur = t - s
            if best is None or dur < best[0]:
                best = (dur, n)
        i -= 1
    return best[1] if best else "?"

seen = defaultdict(int)
samples = kernels[:: max(len(kernels) // 200, 1)]
for ts, corr in samples:
    if corr in runtime:
        op = enclosing(runtime[corr][0])
        seen[op[:90]] += 1
print(f"target={target}, kernels={len(kernels)}, sampled={len(samples)}")
for op, c in sorted(seen.items(), key=lambda kv: -kv[1])[:15]:
    print(f"  x{c:<4} {op}")
