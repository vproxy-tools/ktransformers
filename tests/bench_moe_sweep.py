#!/usr/bin/env python
# coding=utf-8
"""Quick thread-count sweep + numerics check for the MXFP4 MoE kernel.

Derived from kt-kernel/bench/bench_fp4_moe.py with fewer iterations, a
--threads-per-numa loop and E8M0 (pow2) group scales so the kernel's
scale-fold fast path engages (matches the real V4-Flash F8_E8M0 checkpoint
format; non-pow2 scales make the kernel fall back to the legacy path).

Usage: .venv/bin/python tests/bench_moe_sweep.py [--tpn 24] [--m 512,1024] [--check]
"""
import argparse
import os
os.environ.setdefault("KT_HUGEPAGE_WEIGHTS", "0")
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kt-kernel", "build"))
from kt_kernel import kt_kernel_ext  # noqa: E402

HIDDEN, INTER, EXPERT_NUM, TOP_K, GS = 4096, 2048, 256, 6, 32

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def quant(w):
    """[E, N, K] → packed nibbles + E8M0 (pow2, round-up) scales like the checkpoint."""
    e, r, c = w.shape
    g = w.view(e, r, c // GS, GS)
    mx = g.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    s = torch.exp2(torch.ceil(torch.log2(mx / 6.0))).clamp(min=2.0**-127).squeeze(-1)
    n = g / s.unsqueeze(-1)
    nib = (n.unsqueeze(-1) - E2M1.view(1, 1, 1, 1, 16)).abs().argmin(-1)
    nib = nib.to(torch.uint8).view(e, r, c // 2, 2)
    packed = ((nib[..., 1] << 4) | nib[..., 0]).contiguous()
    return packed, s.to(torch.bfloat16).contiguous(), nib


def reference_forward(x, ids, rw, ws, nibs):
    """fp32 reference MoE using the SAME dequantized weights as the kernel."""
    M = x.shape[0]
    y = torch.zeros(M, HIDDEN, dtype=torch.float32)
    for k in range(TOP_K):
        for e in range(EXPERT_NUM):
            tok = (ids[:, k] == e).nonzero(as_tuple=True)[0]
            if tok.numel() == 0:
                continue
            xe = x[tok].to(torch.float32)
            gw = dequant(ws["gate_s"][e], nibs["gate"][e], INTER, HIDDEN)
            uw = dequant(ws["up_s"][e], nibs["up"][e], INTER, HIDDEN)
            dw = dequant(ws["down_s"][e], nibs["down"][e], HIDDEN, INTER)
            g = (xe @ gw.T)
            u = (xe @ uw.T)
            import torch.nn.functional as F
            act = F.silu(g) * u  # swiglu_limit=0 → plain silu (alpha=1)
            y[tok] += (act @ dw.T) * rw[tok, k].unsqueeze(1)
    return y


def dequant(s_bf16, nib, n, k):
    vals = E2M1.to(torch.float32)
    lo = vals[nib[..., 0].long()]
    hi = vals[nib[..., 1].long()]
    w = torch.stack([lo, hi], dim=-1).view(n, k // GS, GS)
    return (w * s_bf16.to(torch.float32).unsqueeze(-1)).view(n, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpn", default="24")
    ap.add_argument("--m", default="512,1024")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--numa", type=int, default=2)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    ws, nibs = {}, {}
    for name, shape in (("gate", (EXPERT_NUM, INTER, HIDDEN)),
                        ("up", (EXPERT_NUM, INTER, HIDDEN)),
                        ("down", (EXPERT_NUM, HIDDEN, INTER))):
        w, s, nib = quant(torch.randn(shape) / 100)
        ws[f"{name}_w"], ws[f"{name}_s"] = w, s
        nibs[name] = nib

    tpn = int(args.tpn.split(",")[0])
    wp = kt_kernel_ext.WorkerPoolConfig()
    wp.subpool_count = args.numa
    wp.subpool_numa_map = list(range(args.numa))
    wp.subpool_thread_count = [tpn] * args.numa
    ci = kt_kernel_ext.CPUInfer(wp)
    cfg = kt_kernel_ext.moe.MOEConfig(EXPERT_NUM, TOP_K, HIDDEN, INTER, 0)
    cfg.max_len = 1024
    cfg.pool = ci.backend_
    cfg.quant_config.bits = 4
    cfg.quant_config.group_size = GS
    cfg.quant_config.zero_point = False
    for name in ("gate", "up", "down"):
        setattr(cfg, f"{name}_projs", [[t.data_ptr() for t in ws[f"{name}_w"]]])
        setattr(cfg, f"{name}_scales", [[t.data_ptr() for t in ws[f"{name}_s"]]])
    moe = kt_kernel_ext.moe.AMXFP4_KGroup_MOE(cfg)
    p2l = torch.arange(EXPERT_NUM, dtype=torch.int64).contiguous()
    ci.submit(moe.load_weights_task(p2l.data_ptr()))
    ci.sync()

    if args.check:
        M = 64
        bsz = torch.tensor([M], dtype=torch.int32)
        ids = torch.stack([torch.randperm(EXPERT_NUM)[:TOP_K]
                           for _ in range(M)]).to(torch.int64).contiguous()
        rw = torch.rand((M, TOP_K)).contiguous()
        x = (torch.randn((M, HIDDEN), dtype=torch.bfloat16) / 100).contiguous()
        y = torch.zeros((M, HIDDEN), dtype=torch.bfloat16).contiguous()
        ci.submit(moe.forward_task(bsz.data_ptr(), TOP_K, ids.data_ptr(),
                                   rw.data_ptr(), x.data_ptr(), y.data_ptr(), False))
        ci.sync()
        ref = reference_forward(x, ids, rw, ws, nibs)
        got = y.to(torch.float32)
        rel = ((got - ref).abs() / ref.abs().clamp(min=1e-3)).max().item()
        print(f"[check] M={M} max_rel_err={rel:.2e}  "
              f"{'PASS' if rel < 5e-2 else 'FAIL'}")
        return

    for M in [int(x) for x in args.m.split(",")]:
        bsz = torch.tensor([M], dtype=torch.int32)
        ids = torch.stack([torch.randperm(EXPERT_NUM)[:TOP_K]
                           for _ in range(M)]).to(torch.int64).contiguous()
        rw = torch.rand((M, TOP_K)).contiguous()
        x = (torch.randn((M, HIDDEN), dtype=torch.bfloat16) / 100).contiguous()
        y = torch.empty((M, HIDDEN), dtype=torch.bfloat16).contiguous()
        for _ in range(30):
            ci.submit(moe.forward_task(bsz.data_ptr(), TOP_K, ids.data_ptr(),
                                       rw.data_ptr(), x.data_ptr(), y.data_ptr(), False))
            ci.sync()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            ci.submit(moe.forward_task(bsz.data_ptr(), TOP_K, ids.data_ptr(),
                                       rw.data_ptr(), x.data_ptr(), y.data_ptr(), False))
            ci.sync()
        dt = time.perf_counter() - t0
        us = dt / args.iters * 1e6
        print(f"tpn={tpn:>2} M={M:>5}  per-iter={us:>9.1f} us  "
              f"tok/s/layer={M/us*1e6:>9.1f}  e2e-if-43layers={M/(us*43)*1e6:>7.1f}",
              flush=True)


if __name__ == "__main__":
    main()
