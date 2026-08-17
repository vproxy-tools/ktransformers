#!/usr/bin/env python
"""L2 冷缓存版微基准：43 层权重轮转 + w8a8 M=1 配置扫描。"""
import torch
import triton
import triton.language as tl
import sys
sys.path.insert(0, "/home/wkgcass/ktransformers/.venv/lib/python3.12/site-packages")
from sglang.srt.layers.quantization.fp8_kernel import w8a8_block_fp8_matmul

torch.cuda.set_device(0)
G, R, D = 8, 1024, 4096
NLAYER = 16


def bench(fn, iters=129, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000


@triton.jit
def _bf16_gemv_kernel(O, W, OUT, R: tl.constexpr, D: tl.constexpr,
                      BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_g = tl.program_id(0)
    pid_r = tl.program_id(1)
    o_ptr = O + pid_g * D
    acc = tl.zeros((BLOCK_R,), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        ov = tl.load(o_ptr + offs_d).to(tl.float32)
        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        w_ptrs = W + pid_g * R * D + offs_r[:, None] * D + offs_d[None, :]
        wv = tl.load(w_ptrs).to(tl.float32)
        acc += tl.sum(wv * ov[None, :], axis=1)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    tl.store(OUT + pid_g * R + offs_r, acc.to(tl.bfloat16))


def part1_woa():
    print("=== wo_a einsum, 43-layer rotation (L2 cold) ===")
    o = torch.randn(1, G, D, dtype=torch.bfloat16, device="cuda") / 50
    Ws = torch.randn(NLAYER, G, R, D, dtype=torch.bfloat16, device="cuda") / 50
    ref = torch.einsum("tgd,grd->tgr", o, Ws[0]).float()

    def cublas_rot():
        for i in range(NLAYER):
            torch.einsum("tgd,grd->tgr", o, Ws[i])

    out = torch.empty(1, G, R, dtype=torch.bfloat16, device="cuda")

    def triton_rot(br, bd, wp):
        for i in range(NLAYER):
            _bf16_gemv_kernel[(G, R // br)](o, Ws[i], out, R=R, D=D,
                                            BLOCK_R=br, BLOCK_D=bd, num_warps=wp)
    chk = torch.empty_like(out)
    _bf16_gemv_kernel[(G, R // 128)](o, Ws[0], chk, R=R, D=D, BLOCK_R=128, BLOCK_D=128, num_warps=8)
    err = (chk.float() - ref).abs().max().item()
    t = bench(cublas_rot) / NLAYER
    print(f"cublas einsum:        {t:8.1f} us/layer")
    for br, bd, wp in [(128, 128, 8), (64, 128, 4), (64, 256, 8), (128, 256, 8)]:
        t = bench(lambda: triton_rot(br, bd, wp)) / NLAYER
        print(f"triton BR={br} BD={bd} w={wp}: {t:8.1f} us/layer  (chk err {err:.4f})")


def part2_w8a8():
    print("\n=== w8a8_block_fp8_matmul M=1 (16-layer rotation) ===")
    shapes = [(32768, 1024), (4096, 8192), (4096, 4096), (4096, 2048), (1024, 4096), (8192, 1024)]
    for N, K in shapes:
        As = torch.randn(NLAYER, N, K, dtype=torch.float32, device="cuda")
        As = (As / As.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-6) * 400).to(torch.float8_e4m3fn)
        Bs = torch.randn(NLAYER, 1, K, dtype=torch.bfloat16, device="cuda") / 50
        As_s = torch.rand(NLAYER, N // 128, K // 128, dtype=torch.float32, device="cuda") + 0.5
        Bs_s = torch.rand(NLAYER, 1, (K + 127) // 128, dtype=torch.float32, device="cuda") + 0.5
        bl = [128, 128]

        def rot():
            for i in range(NLAYER):
                w8a8_block_fp8_matmul(As[i], As_s[i], Bs[i], Bs_s[i], bl)

        t = bench(rot) / NLAYER
        bw = N * K / 1e9 / (t / 1e6)
        print(f"N={N:<6} K={K:<5} current: {t:7.1f} us  ({bw:,.0f} GB/s)")
        del As, Bs, As_s, Bs_s
        torch.cuda.empty_cache()


if __name__ == "__main__":
    part1_woa()
    part2_w8a8()
