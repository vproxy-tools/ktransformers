#!/usr/bin/env python
"""按 decode step 分桶统计 GPU kernel 时间（剔除 prefill 段）。"""
import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)
events = data["traceEvents"]

launches = sorted((e["ts"], e["ts"] + e["dur"]) for e in events
                  if e.get("ph") == "X" and e.get("name") == "cudaGraphLaunch"
                  and e.get("cat") == "cuda_runtime")
# 用第一个 graph launch 的结束时间作为 decode 段起点，避开 prefill
if not launches:
    print("no graph launches found")
    sys.exit(0)

gpu_events = [(e["ts"], e["ts"] + e["dur"], e.get("cat", ""), e["name"])
              for e in events
              if e.get("ph") == "X" and e.get("cat", "") in ("kernel", "gpu_memcpy", "gpu_memset")]
gpu_events.sort()

# 只取落在 [launch0.end, launchN.start] 区间内的事件
lo = launches[0][1]
hi = launches[-1][0]
dec = [g for g in gpu_events if g[0] >= lo and g[1] <= hi]
nsteps = len(launches) - 1
span_ms = (hi - lo) / 1e3
busy = sum(te - ts for ts, te, *_ in dec)
print(f"decode-only: {nsteps} steps, span {span_ms:.1f} ms, "
      f"step={span_ms/max(nsteps,1):.2f} ms, GPU busy {busy/1e3:.1f} ms "
      f"({busy/(hi-lo)*100:.0f}%), GPU idle/step={(span_ms-busy/1e3)/nsteps:.2f} ms")

agg = defaultdict(float)
cnt = defaultdict(int)
for ts, te, cat, name in dec:
    agg[name[:70]] += te - ts
    cnt[name[:70]] += 1
print(f"\nPer-step breakdown (us/step, calls/step):")
for name, t in sorted(agg.items(), key=lambda kv: -kv[1])[:28]:
    print(f"  {t/nsteps:>8.1f} us/step  x{cnt[name]/nsteps:>5.1f}  {name}")
