#!/usr/bin/env python3
"""Offline unit tests for the dsv4-vision tower SDPA path (_vision_sdpa).

Regression context: the vision tower passes [heads, n, d] (3-D) tensors.
Torch's fused SDPA kernels (flash/cuDNN/mem-efficient) require 4-D
[batch, heads, n, d]; with 3-D inputs they disable themselves and the
dispatcher silently falls back to the math path, whose O(n^2) attention
matrix OOMed production (2026-09-01). The fix adds a singleton batch dim
so fused kernels engage; on OOM it falls back to query-chunked SDPA.

Covers:
  1. Fused engagement: _vision_sdpa must succeed with the kernel list
     restricted to FLASH_ATTENTION (a 3-D call would raise "No available
     kernel") — this is the regression guard for the batch-dim fix.
  2. Correctness vs an fp32 CPU math reference.
  3. FORCE_CHUNK=1 fallback matches the unchunked result (bf16 tolerance).

Run:  .venv/bin/python tests/vision_sdpa_test.py
(needs a CUDA GPU; ~200 MiB)
"""

import os
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.srt.models.deepseek_v4_vision import _vision_sdpa

PASS, FAIL = 0, 0

H, D = 16, 64  # vision_n_heads, head_dim (vision_dim 1024 / 16)


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def make_qkv(n, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # Same layout the tower produces: RoPE output [n, h, d] -> transpose(0,1).
    out = []
    for _ in range(3):
        x = torch.randn(n, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
        out.append(x.transpose(0, 1))  # [h, n, d], non-contiguous
    return out


def fp32_ref(q, k, v):
    return (
        F.scaled_dot_product_attention(
            q.float().cpu(), k.float().cpu(), v.float().cpu()
        )
        .cuda()
        .to(q.dtype)
    )


def close(a, b, tol=2e-2):
    diff = (a.float() - b.float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    )
    return diff.max().item() < tol and cos.item() > 0.999, (
        f"max={diff.max().item():.4g} cos={cos.item():.6f}"
    )


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return

    # 1. Fused-kernel engagement (regression guard for the 3-D bug).
    q, k, v = make_qkv(1024)
    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            o = _vision_sdpa(q, k, v)
        torch.cuda.synchronize()
        check("flash-only dispatch works (4-D conversion)", True)
    except Exception as e:
        check("flash-only dispatch works (4-D conversion)", False, f"{e!r:.160}")
        return

    # 2. Correctness vs fp32 CPU reference.
    ok, detail = close(o, fp32_ref(q, k, v))
    check("flash result matches fp32 math reference", ok, detail)

    # 2b. Default dispatch also matches (whichever backend it picks).
    o_def = _vision_sdpa(q, k, v)
    ok, detail = close(o_def, fp32_ref(q, k, v))
    check("default dispatch matches fp32 math reference", ok, detail)

    # 3. Chunked fallback (forced) matches the unchunked result.
    os.environ["SGLANG_DSV4_VISION_SDPA_FORCE_CHUNK"] = "1"
    try:
        n = 1024 + 17  # not a multiple of the 512 chunk: covers the tail
        q, k, v = make_qkv(n, seed=1)
        o_chunk = _vision_sdpa(q, k, v)
        torch.cuda.synchronize()
    finally:
        del os.environ["SGLANG_DSV4_VISION_SDPA_FORCE_CHUNK"]
    check(
        "chunked fallback output shape",
        tuple(o_chunk.shape) == (H, n, D),
        f"got {tuple(o_chunk.shape)}",
    )
    ok, detail = close(o_chunk, fp32_ref(q, k, v))
    check("chunked fallback matches fp32 math reference", ok, detail)


if __name__ == "__main__":
    main()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
