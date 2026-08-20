#!/usr/bin/env python
"""采样 decode 期间 scheduler 进程各线程的 CPU 占用与亲和性。"""
import os
import sys
import time

PID = int(sys.argv[1]) if len(sys.argv) > 1 else None
if PID is None:
    out = os.popen("pidof -x sglang::scheduler").read().split()
    PID = int(out[0])
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0

def snapshot():
    info = {}
    for tid in os.listdir(f"/proc/{PID}/task"):
        try:
            with open(f"/proc/{PID}/task/{tid}/stat") as f:
                parts = f.read().split()
            utime, stime = int(parts[13]), int(parts[14])
            try:
                with open(f"/proc/{PID}/task/{tid}/comm") as f:
                    comm = f.read().strip()
            except OSError:
                comm = parts[1]
            info[int(tid)] = [utime + stime, comm, 0]
        except (OSError, ValueError, IndexError):
            pass
    return info

a = snapshot()
time.sleep(DUR)
b = snapshot()

CLK = os.sysconf("SC_CLK_TCK")
rows = []
for tid, (tot, comm, _) in b.items():
    if tid in a:
        dt = tot - a[tid][0]
        if dt > 0:
            rows.append((dt / DUR / CLK, tid, comm))
rows.sort(reverse=True)
print(f"PID {PID}, threads with usage over {DUR}s:")
for util, tid, comm in rows[:60]:
    print(f"  {util * 100:6.1f}%  tid={tid:<7} {comm}")
total = sum(r[0] for r in rows)
print(f"total thread-seconds/sec = {total:.1f}  ({len(rows)} active threads)")
