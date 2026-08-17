#!/usr/bin/env python
# coding=utf-8
"""冷缓存版 MXFP4 MoE 基准：每次迭代换一组随机 top-k 专家。

真实 decode 中每个 token 从 256 个专家里随机激活 6 个，L3 无法驻留全部
146GB 权重，每层都必须从内存重新读。bench_fp4_moe.py 固定一组专家会命中
L3，结果偏乐观；本脚本按 token 轮换预生成的随机路由，模拟真实访存。

用法:
    KT_HUGEPAGE_WEIGHTS=0 python bench/bench_fp4_moe_cold.py \
        [--threads-per-numa 22] [--iters 300] [--mode cold|hot]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
from kt_kernel import kt_kernel_ext  # noqa: E402

HIDDEN = 4096
INTER = 2048
EXPERT_NUM = 256
TOP_K = 6
K_GROUP_SIZE = 32

E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def quantize_mxfp4_tensor(weights: torch.Tensor, group_size: int):
    w = weights.to(torch.float32)
    e, rows, cols = w.shape
    reshaped = w.view(e, rows, cols // group_size, group_size)
    max_abs = reshaped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scales = (max_abs / 6.0).squeeze(-1)
    normalized = reshaped / scales.unsqueeze(-1)
    distances = torch.abs(normalized.unsqueeze(-1) - E2M1_VALUES.view(1, 1, 1, 1, 16))
    nibbles = distances.argmin(dim=-1).to(torch.uint8).view(e, rows, cols // 2, 2)
    packed = ((nibbles[..., 1] << 4) | nibbles[..., 0]).contiguous()
    scales = scales.to(torch.bfloat16).contiguous()
    return packed, scales


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threads-per-numa", type=int, default=22)
    p.add_argument("--numa", type=int, default=2)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--mode", choices=["cold", "hot"], default="cold")
    args = p.parse_args()

    wp = kt_kernel_ext.WorkerPoolConfig()
    wp.subpool_count = args.numa
    wp.subpool_numa_map = list(range(args.numa))
    wp.subpool_thread_count = [args.threads_per_numa] * args.numa
    cpu_infer = kt_kernel_ext.CPUInfer(wp)

    print("[cold-bench] synthesizing weights …", flush=True)
    torch.manual_seed(0)
    packs = []
    for name in ("gate", "up"):
        w = torch.randn((EXPERT_NUM, INTER, HIDDEN), dtype=torch.float32) / 100
        packs.append(quantize_mxfp4_tensor(w, K_GROUP_SIZE))
    dw = torch.randn((EXPERT_NUM, HIDDEN, INTER), dtype=torch.float32) / 100
    packs.append(quantize_mxfp4_tensor(dw, K_GROUP_SIZE))

    cfg = kt_kernel_ext.moe.MOEConfig(EXPERT_NUM, TOP_K, HIDDEN, INTER, 0)
    cfg.max_len = 8
    cfg.pool = cpu_infer.backend_
    cfg.quant_config.bits = 4
    cfg.quant_config.group_size = K_GROUP_SIZE
    cfg.quant_config.zero_point = False
    cfg.gate_projs = [[t.data_ptr() for t in packs[0][0]]]
    cfg.up_projs = [[t.data_ptr() for t in packs[1][0]]]
    cfg.down_projs = [[t.data_ptr() for t in packs[2][0]]]
    cfg.gate_scales = [[t.data_ptr() for t in packs[0][1]]]
    cfg.up_scales = [[t.data_ptr() for t in packs[1][1]]]
    cfg.down_scales = [[t.data_ptr() for t in packs[2][1]]]
    moe = kt_kernel_ext.moe.AMXFP4_KGroup_MOE(cfg)
    p2l = torch.arange(EXPERT_NUM, dtype=torch.int64).contiguous()
    cpu_infer.submit(moe.load_weights_task(p2l.data_ptr()))
    cpu_infer.sync()
    print("[cold-bench] weights loaded", flush=True)

    # 预生成 iters 组随机路由（cold）或一组复用（hot）
    n_routes = args.iters + args.warmup
    if args.mode == "cold":
        gen = torch.Generator().manual_seed(42)
        ids_all = torch.stack(
            [torch.randperm(EXPERT_NUM, generator=gen)[:TOP_K] for _ in range(n_routes)]
        ).to(torch.int64).contiguous()
    else:
        hot = torch.randperm(EXPERT_NUM)[:TOP_K].to(torch.int64)
        ids_all = hot.unsqueeze(0).expand(n_routes, TOP_K).contiguous()

    bsz = torch.tensor([1], dtype=torch.int32)
    routing_w = torch.rand((1, TOP_K), dtype=torch.float32).contiguous()
    x = (torch.randn((1, HIDDEN), dtype=torch.bfloat16) / 100).contiguous()
    y = torch.empty((1, HIDDEN), dtype=torch.bfloat16).contiguous()

    def run(i: int) -> None:
        cpu_infer.submit(moe.forward_task(
            bsz.data_ptr(), TOP_K, ids_all[i].data_ptr(),
            routing_w.data_ptr(), x.data_ptr(), y.data_ptr(), False))
        cpu_infer.sync()

    for i in range(args.warmup):
        run(i)
    start = time.perf_counter()
    for i in range(args.iters):
        run(args.warmup + i)
    total = time.perf_counter() - start
    us = total / args.iters * 1e6
    print(f"RESULT: mode={args.mode} threads/numa={args.threads_per_numa} "
          f"per-iter={us:.1f} us  ({us / 1000:.3f} ms/layer, x40层={us * 40 / 1000:.1f} ms/token)")


if __name__ == "__main__":
    main()
