#!/usr/bin/env python
"""GPU 微基准：wo_a einsum 各种实现 + w8a8 M=1 配置扫描。

形状 (V4-Flash decode bs=1):
  wo_a einsum: o[1,8,4096] bf16 × W[8,1024,4096] -> [1,8,1024]
  wq_b:  [1,1024]x[32768,1024] fp8 block128
  wo_b:  [1,8192]x[4096,8192] fp8 block128
  shared gate_up: [1,4096]x[4096,4096] fp8
  shared down:    [1,2048]x[4096,2048] fp8
"""
import torch
import triton
import triton.language as tl

torch.cuda.set_device(0)

G, R, D = 8, 1024, 4096


def bench(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # us


# ---------- wo_a 候选实现 ----------
def make_bf16():
    o = torch.randn(1, G, D, dtype=torch.bfloat16, device="cuda") / 50
    w = torch.randn(G, R, D, dtype=torch.bfloat16, device="cuda") / 50
    return o, w


def einsum_bf16(o, w):
    return torch.einsum("tgd,grd->tgr", o, w)


def bmm_bf16(o, w):
    return torch.bmm(o.transpose(0, 1), w.transpose(1, 2)).transpose(0, 1)


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


def triton_bf16_gemv(o, w, BLOCK_R=128, BLOCK_D=128, warps=8):
    out = torch.empty(1, G, R, dtype=torch.bfloat16, device="cuda")
    _bf16_gemv_kernel[(G, R // BLOCK_R)](
        o, w, out, R=R, D=D, BLOCK_R=BLOCK_R, BLOCK_D=BLOCK_D, num_warps=warps)
    return out


@triton.jit
def _fp8_gemv_kernel(O_Q, O_S, W_Q, W_S, OUT,
                     R: tl.constexpr, D: tl.constexpr,
                     BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_g = tl.program_id(0)
    pid_r = tl.program_id(1)
    oq_ptr = O_Q + pid_g * D
    os_ptr = O_S + pid_g * (D // 128)
    acc = tl.zeros((BLOCK_R,), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        offs_k = d0 // 128
        oq = tl.load(oq_ptr + offs_d).to(tl.float32)
        os = tl.load(os_ptr + offs_k)
        ov = oq * os
        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        w_ptrs = W_Q + pid_g * R * D + offs_r[:, None] * D + offs_d[None, :]
        wq = tl.load(w_ptrs).to(tl.float32)
        # weight block scale: 行块 (g*R+r)//128, 列块 d0//128
        row_blk = (pid_g * R + offs_r) // 128
        ws = tl.load(W_S + row_blk * (D // 128) + offs_k)
        acc += tl.sum(wq * ws[:, None] * ov[None, :], axis=1)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    tl.store(OUT + pid_g * R + offs_r, acc.to(tl.bfloat16))


def triton_fp8_gemv(oq, os_, wq, ws, BLOCK_R=128, BLOCK_D=128, warps=8):
    out = torch.empty(1, G, R, dtype=torch.bfloat16, device="cuda")
    _fp8_gemv_kernel[(G, R // BLOCK_R)](
        oq, os_, wq, ws, out, R=R, D=D, BLOCK_R=BLOCK_R, BLOCK_D=BLOCK_D,
        num_warps=warps)
    return out


def main():
    print("=== wo_a einsum [1,8,4096]x[8,1024,4096] ===")
    o, w = make_bf16()
    ref = einsum_bf16(o, w).float()
    t = bench(lambda: einsum_bf16(o, w))
    print(f"einsum bf16 (cublas):     {t:8.1f} us")
    t = bench(lambda: bmm_bf16(o, w))
    print(f"bmm bf16:                 {t:8.1f} us")
    for br, bd, wp in [(64, 128, 4), (128, 128, 8), (64, 256, 8), (128, 256, 8),
                       (32, 512, 8), (64, 512, 8)]:
        out = triton_bf16_gemv(o, w, br, bd, wp)
        err = (out.float() - ref).abs().max().item()
        t = bench(lambda: triton_bf16_gemv(o, w, br, bd, wp))
        print(f"triton bf16 BR={br} BD={bd} w={wp}: {t:8.1f} us  maxerr={err:.4f}")

    # fp8 版本
    wq = (w.float() / 1.2).to(torch.float8_e4m3fn)
    ws = torch.full((G * R // 128, D // 128), 1.2, dtype=torch.float32, device="cuda")
    oq_f = o.reshape(G, D)
    oscale = torch.ones(G, D // 128, dtype=torch.float32, device="cuda")
    oq = oq_f.to(torch.float8_e4m3fn)
    for br, bd, wp in [(64, 128, 4), (128, 128, 8), (64, 256, 8), (128, 256, 8)]:
        out = triton_fp8_gemv(oq, oscale, wq, ws, br, bd, wp)
        err = (out.float() - ref).abs().max().item() / ref.abs().max().item()
        t = bench(lambda: triton_fp8_gemv(oq, oscale, wq, ws, br, bd, wp))
        print(f"triton fp8 BR={br} BD={bd} w={wp}: {t:8.1f} us  relerr={err:.4f}")


if __name__ == "__main__":
    main()
