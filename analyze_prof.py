#!/usr/bin/env python
"""解析 sglang torch profiler trace：decode 每步耗时分解。

统计:
  - 每 decode step 的墙钟时间(通过 cudaGraphLaunch 间隔)
  - GPU kernel 执行时间按名称聚合(前 N)
  - Memcpy 时间(D2H/H2D)
  - host 节点/cuda_runtime 关键调用耗时
"""
import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)

events = data["traceEvents"]

# --- GPU stream 时间线: kernel / memcpy 事件 ---
gpu_events = []
for e in events:
    if e.get("ph") != "X":
        continue
    cat = e.get("cat", "")
    if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
        gpu_events.append((e["ts"], e["ts"] + e["dur"], cat, e["name"], e.get("args", {}).get("stream", -1)))

gpu_events.sort()
total_gpu = defaultdict(float)
count_gpu = defaultdict(int)
for ts, te, cat, name, stream in gpu_events:
    key = name[:80]
    total_gpu[key] += te - ts
    count_gpu[key] += 1

span = (gpu_events[0][0], gpu_events[-1][1]) if gpu_events else (0, 0)
busy = sum(te - ts for ts, te, *_ in gpu_events)
print(f"GPU events span: {(span[1]-span[0])/1e3:.1f} ms, busy {busy/1e3:.1f} ms "
      f"({busy/(span[1]-span[0])*100:.0f}%), idle {(span[1]-span[0]-busy)/1e3:.1f} ms")
print("\nTop GPU kernels by total time (us):")
for name, t in sorted(total_gpu.items(), key=lambda kv: -kv[1])[:25]:
    print(f"  {t:>9.0f} us  x{count_gpu[name]:<5} {name}")

# --- cudaGraphLaunch 调用(每 decode step 一次) ---
launches = sorted(e["ts"] for e in events
                  if e.get("ph") == "X" and e.get("name") == "cudaGraphLaunch")
if len(launches) > 2:
    gaps = [(launches[i+1] - launches[i]) / 1e3 for i in range(len(launches)-1)]
    gaps = [g for g in gaps if 5 < g < 200]  # 过滤异常
    if gaps:
        print(f"\ncudaGraphLaunch count={len(launches)}, median step gap={sorted(gaps)[len(gaps)//2]:.2f} ms")

# --- 关键 cuda_runtime 调用耗时 ---
rt = defaultdict(float)
rtc = defaultdict(int)
for e in events:
    if e.get("ph") == "X" and e.get("cat") == "cuda_runtime":
        rt[e["name"]] += e["dur"]
        rtc[e["name"]] += 1
print("\nTop cuda_runtime calls (total us / count):")
for name, t in sorted(rt.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {t:>9.0f} us  x{rtc[name]:<5} {name}")

# --- CPU 侧 user_annotation(调度器每步的阶段) ---
ann = defaultdict(float)
annc = defaultdict(int)
for e in events:
    if e.get("ph") == "X" and e.get("cat") in ("user_annotation", "cpu_op"):
        ann[e["name"][:70]] += e["dur"]
        annc[e["name"][:70]] += 1
print("\nTop user annotations / cpu_op (total us):")
for name, t in sorted(ann.items(), key=lambda kv: -kv[1])[:15]:
    print(f"  {t:>9.0f} us  x{annc[name]:<4} {name}")
