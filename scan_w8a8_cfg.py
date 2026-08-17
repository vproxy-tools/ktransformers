#!/usr/bin/env python
"""扫描 w8a8_block_fp8_matmul 的 M=1 Triton 配置（V4-Flash decode 形状）。

通过 monkeypatch get_w8a8_block_fp8_configs 注入候选配置，
L2 冷却用 8 层权重轮转。
"""
import sys

import torch

sys.path.insert(0, "/home/wkgcass/ktransformers/.venv/lib/python3.12/site-packages")
torch.cuda.set_device(0)

import sglang.srt.layers.quantization.fp8_kernel as fk

SHAPES = [
    (32768, 1024, "wq_b"),
    (4096, 8192, "wo_b"),
    (4096, 4096, "shared_gate_up"),
    (4096, 2048, "shared_down"),
    (1536, 4096, "wqkv_a_fused"),
    (1024, 4096, "wq_a"),
    (8192, 1024, "indexer_wq_b"),
]

CANDIDATES = [
    None,  # 当前 JSON 条目（BM=16,BN=128,BK=128,G=1,w4,s2）
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 3},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 3},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3},
]

_orig = fk.get_w8a8_block_fp8_configs
INJECT = [None]


def patched(N, K, block_n, block_k):
    if INJECT[0] is not None:
        return {1: INJECT[0]}
    return _orig(N, K, block_n, block_k)


fk.get_w8a8_block_fp8_configs = patched

NL = 8


def bench_shape(N, K):
    Wq = torch.randint(-60, 60, (NL, N, K), device="cuda", dtype=torch.int8).view(torch.float8_e4m3fn)
    Ws = torch.rand(NL, N // 128, K // 128, device="cuda") + 0.5
    x = torch.randint(-60, 60, (1, K), device="cuda", dtype=torch.int8).view(torch.float8_e4m3fn)
    xs = torch.rand(1, K // 128, device="cuda") + 0.5

    def rot():
        for i in range(NL):
            fk.w8a8_block_fp8_matmul_triton(x, Wq[i], xs, Ws[i], [128, 128])

    for _ in range(10):
        rot()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(40):
        rot()
    e.record()
    torch.cuda.synchronize()
    us = s.elapsed_time(e) / 40 / NL * 1000
    del Wq, Ws
    torch.cuda.empty_cache()
    return us


def main():
    for N, K, name in SHAPES:
        results = []
        for cfg in CANDIDATES:
            INJECT[0] = cfg
            try:
                us = bench_shape(N, K)
            except Exception as ex:
                us = float("nan")
            label = "json-default" if cfg is None else f"BN{cfg['BLOCK_SIZE_N']}_BK{cfg['BLOCK_SIZE_K']}_w{cfg['num_warps']}s{cfg['num_stages']}"
            results.append((us, label))
        best = min(r for r, _ in results if r == r)
        line = f"{name:16} N={N:<6} K={K:<5}: "
        line += "  ".join(f"{l}={u:.1f}us" for u, l in results if u == u)
        print(line, flush=True)


if __name__ == "__main__":
    main()
