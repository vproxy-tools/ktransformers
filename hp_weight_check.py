#!/usr/bin/env python3
"""CPU-only 验证：NativeMoE MXFP4 巨页权重缓存（不碰 GPU，可与生产服务共存）。

用真实模型某一层的权重走一遍 NativeMoEWrapper.load_weights（与服务器加载
路径完全一致，层 key/stamp 相同，产物可直接被服务器复用）：
  第 1 遍（冷，独立进程）: safetensors 读取 + 转换写入持久 arena + commit；
  第 2 遍（热，独立进程）: python check_reusable 命中 → 跳过 safetensors，
                C++ mmap 已驻留大页（日志 REUSED from persistent hugepages）。

用法: 连跑两遍 python3 hp_weight_check.py [layer_idx]
前置: /dev/hugepages/kt_weights 已由 root 创建并 chown 给当前用户。
"""
import os
import sys
import time

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MODEL = "/var/deepseek-v4-flash/0731"
HIDDEN, INTER, EXPERTS, TOPK = 4096, 2048, 256, 6

import torch
from kt_kernel import KTMoEWrapper
from kt_kernel.utils.hugepage_cache import check_reusable, ensure_model_stamp

mask = torch.zeros(EXPERTS, dtype=torch.bool)  # 全部专家在 CPU
w = KTMoEWrapper(
    layer_idx=LAYER,
    num_experts=EXPERTS,
    num_experts_per_tok=TOPK,
    hidden_size=HIDDEN,
    moe_intermediate_size=INTER,
    gpu_experts_mask=mask,
    cpuinfer_threads=32,
    threadpool_count=2,  # 与服务器一致 → 层 key 的 I1024 一致
    weight_path=MODEL,
    chunked_prefill_size=64,
    method="MXFP4",
)
ensure_model_stamp(MODEL)
p2l = torch.arange(EXPERTS, dtype=torch.int64)
print(f"[pre-check] 本次加载前 check_reusable = "
      f"{check_reusable(LAYER, EXPERTS, HIDDEN, INTER // 2, 32, 4, 'AMXFP4_KGroup_MOE', p2l)}")
t0 = time.time()
w.load_weights(p2l)
print(f"load_weights 总耗时 {(time.time() - t0) * 1000:.0f}ms "
      f"（冷≈分钟级：转换+commit；热≈毫秒级：大页直接 REUSED）")
