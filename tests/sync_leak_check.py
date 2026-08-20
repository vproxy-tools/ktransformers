#!/usr/bin/env python3
"""kt-kernel SyncArgs 泄漏 / cuda-graph 回放 UAF 回归测试（CPU+单卡，~2 分钟）。

背景（DSv4Flash.md 9.5）：CPUInfer::sync_with_cuda_stream 的 SyncArgs 曾
只 new 不 delete（~32B/次）；而图捕获期间发出的 host 节点会用同一 args 指针
在每次回放时重跑回调——所以不能无条件 delete（首版修复因此引发 use-after-free
崩溃循环）。正确行为由本脚本双路验证：

  [eager] 100 万次非捕获调用 → malloc_trim 后 RSS 增长应 ≈ 0（泄漏已修）；
  [graph] 捕获含 4 个 sync host 节点的 cuda graph → 5000 次回放不崩溃、
          RSS 增长 ≈ 0（捕获型 args 永生，回放零分配）。

用法（用 dspark venv 的 python，须有 GPU；与生产服务可共存，显存占用 <1.5GB）:
  <仓库>/.venv/bin/python tests/sync_leak_check.py
退出码 0 = 通过；任何一路数值超标或崩溃 = 不通过。
"""
import gc
import os
import sys
import time

os.environ.setdefault("KT_HUGEPAGE_WEIGHTS", "0")  # 测试进程绝不触碰持久巨页 arena

import ctypes

import torch

from kt_kernel import kt_kernel_ext

EAGER_ITERS = 1_000_000
REPLAY_ITERS = 5_000
REPLAY_NODES = 4
EAGER_B_PER_CALL_LIMIT = 8.0  # trim 后仍 >8B/次视为泄漏
GRAPH_MB_LIMIT = 64.0

libc = ctypes.CDLL("libc.so.6")
libc.malloc_trim.argtypes = [ctypes.c_size_t]


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return 0.0


def settle_trim():
    gc.collect()
    time.sleep(1)
    for _ in range(3):
        libc.malloc_trim(0)
        time.sleep(1)


def main():
    ci = kt_kernel_ext.CPUInfer(4)
    s = torch.cuda.Stream()

    # --- eager 路径：一次性回调应自删，trim 后零增长 ---
    for _ in range(5000):
        ci.sync_with_cuda_stream(s.cuda_stream, 0)
    torch.cuda.synchronize()
    settle_trim()
    r0 = rss_mb()
    for _ in range(EAGER_ITERS):
        ci.sync_with_cuda_stream(s.cuda_stream, 0)
    torch.cuda.synchronize()
    settle_trim()
    r1 = rss_mb()
    eager_bpc = (r1 - r0) * 1e6 / EAGER_ITERS
    print(f"[eager] {EAGER_ITERS} 次，trim 后 RSS {r0:.1f}->{r1:.1f}MB "
          f"({eager_bpc:.2f} B/次)", flush=True)

    # --- graph 路径：捕获型 args 永生，回放不得崩溃/增长 ---
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        ci.sync_with_cuda_stream(st.cuda_stream, 0)  # 图外预热（建 stash 等）
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=st):
        for _ in range(REPLAY_NODES):
            ci.sync_with_cuda_stream(st.cuda_stream, 0)  # 录成 host 节点
    torch.cuda.synchronize()
    settle_trim()
    g0 = rss_mb()
    for _ in range(REPLAY_ITERS):
        g.replay()
    torch.cuda.synchronize()
    settle_trim()
    g1 = rss_mb()
    print(f"[graph] {REPLAY_ITERS} 次回放 ×{REPLAY_NODES} 节点，trim 后 RSS "
          f"{g0:.1f}->{g1:.1f}MB（无崩溃 = UAF/double-free 不存在）", flush=True)

    ok = eager_bpc <= EAGER_B_PER_CALL_LIMIT and (g1 - g0) <= GRAPH_MB_LIMIT
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
