#!/usr/bin/env python3
"""Offline validation: sglang-side DSV4 vision tower vs the checkpoint's
reference implementation (inference/vision.py + image_processor.py).

Runs the same image through both and compares aligner outputs, plus the
image-visible window index logic (get_window_topk_idxs_visible vs the
backend's span extension math). Also times each mode.

Usage:
  .venv/bin/python tests/vision_tower_check.py [image_path]
"""

import os
import sys
import time

import torch

MODEL_DIR = "/var/deepseek-v4-flash/vision-exp"
INFERENCE_DIR = os.path.join(MODEL_DIR, "inference")
KT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SGLANG_PATH = os.path.join(KT_ROOT, "third_party/sglang/python")

sys.path.insert(0, INFERENCE_DIR)
sys.path.insert(0, SGLANG_PATH)

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/vision-sample.png")


def load_vision_state():
    from safetensors.torch import safe_open

    state = {}
    with safe_open(os.path.join(MODEL_DIR, "model-00001-of-00048.safetensors"), "pt") as f:
        for k in f.keys():
            if k.startswith(("vision.", "aligner.")) or k.startswith("image_"):
                state[k] = f.get_tensor(k)
    return state


def build_args():
    from image_processor import load_image

    import json

    class A:
        pass

    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    a = A()
    a.vision_patch_size = cfg["vision_patch_size"]
    a.vision_downsample_ratio = cfg["vision_downsample_ratio"]
    a.vision_max_n_token = cfg["vision_max_n_token"]
    a.vision_min_pixels = cfg["vision_min_pixels"]
    a.vision_max_wh_ratio = cfg["vision_max_wh_ratio"]
    a.dim = cfg["hidden_size"]
    a.vision_dim = cfg["vision_dim"]
    a.vision_n_heads = cfg["vision_n_heads"]
    a.vision_inter_dim = cfg["vision_inter_dim"]
    a.vision_n_layers = cfg["vision_n_layers"]
    a.vision_rope_theta = cfg["vision_rope_theta"]
    return a, cfg


def main():
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(32)
    state = load_vision_state()
    args, cfg = build_args()

    # ---------------- reference ----------------
    from image_processor import build_image_block, load_image
    from vision import Aligner, ViT

    ref_vit = ViT(args).cuda()
    ref_aligner = Aligner(args).cuda()
    ref_missing = ref_vit.load_state_dict(
        {k[len("vision."):]: v.cuda() for k, v in state.items() if k.startswith("vision.")},
        strict=True,
    )
    ref_aligner.load_state_dict(
        {k[len("aligner."):]: v.cuda() for k, v in state.items() if k.startswith("aligner.")},
        strict=True,
    )
    assert not ref_missing.missing_keys, ref_missing

    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image({"url": IMAGE_PATH}, args)
    types, perm = build_image_block(n_llm_h, n_llm_w, 0)
    print(
        f"image: {IMAGE_PATH}\n  n_vit_h x n_vit_w = {n_vit_h}x{n_vit_w} "
        f"({n_vit_h*n_vit_w} ViT tokens), n_llm {n_llm_h}x{n_llm_w}, "
        f"span tokens={types.numel()}"
    )
    patches_gpu = patches.cuda()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.set_default_device("cuda")  # reference builds rope tables lazily
    with torch.inference_mode():
        ref_out = ref_aligner(ref_vit(patches_gpu, n_vit_h, n_vit_w), n_vit_h, n_vit_w)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(3):
            ref_out2 = ref_aligner(ref_vit(patches_gpu, n_vit_h, n_vit_w), n_vit_h, n_vit_w)
    torch.cuda.synchronize()
    t_ref = (time.perf_counter() - t0) / 3
    # Nondeterminism baseline: cuBLAS/SDPA kernel selection varies between
    # calls of even the reference itself in bf16.
    self_diff = (ref_out2.float() - ref_out.float()).abs().max().item()
    self_cos = torch.nn.functional.cosine_similarity(
        ref_out2.float().flatten(), ref_out.float().flatten(), dim=0
    ).item()
    print(
        f"reference (pure GPU bf16): {t_ref*1000:.1f} ms steady, out {tuple(ref_out.shape)}, "
        f"self-nondeterminism max_abs={self_diff:.4f} cos={self_cos:.6f}"
    )
    torch.set_default_device("cpu")  # back to CPU default for our hybrid path

    # ---------------- ours ----------------
    from sglang.srt.models.deepseek_v4_vision import (
        IMAGE,
        Dsv4Aligner,
        Dsv4ViT,
    )

    vc = {
        "vision_n_layers": cfg["vision_n_layers"],
        "vision_dim": cfg["vision_dim"],
        "vision_n_heads": cfg["vision_n_heads"],
        "vision_inter_dim": cfg["vision_inter_dim"],
        "vision_patch_size": cfg["vision_patch_size"],
        "vision_rope_theta": cfg["vision_rope_theta"],
        "vision_downsample_ratio": cfg["vision_downsample_ratio"],
        "dim": cfg["hidden_size"],
    }

    def run_ours(mode):
        vit = Dsv4ViT(vc, mode).cuda()
        aligner = Dsv4Aligner(vc).cuda()
        if mode == "gpu":
            vit.load_state_dict(
                {k[len("vision."):]: v for k, v in state.items() if k.startswith("vision.")},
                strict=True,
            )
        else:
            attn_sd = {
                k[len("vision."):]: v
                for k, v in state.items()
                if k.startswith("vision.") and ".mlp.w" not in k
            }
            vit.load_state_dict(attn_sd, strict=False)
            n = vc["vision_n_layers"]
            vit.load_mlp_int8(
                [state[f"vision.blocks.{i}.mlp.w1.weight"] for i in range(n)],
                [state[f"vision.blocks.{i}.mlp.w2.weight"] for i in range(n)],
            )
        aligner.load_state_dict(
            {k[len("aligner."):]: v for k, v in state.items() if k.startswith("aligner.")},
            strict=True,
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            vit_out = vit(patches_gpu, n_vit_h, n_vit_w)
            out = aligner(vit_out, n_vit_h, n_vit_w)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return out, vit_out, dt

    # warmup both modes once (JIT/cuBLAS autotune)
    run_ours("gpu"); run_ours("hybrid")

    # probe: reference ViT output recomputed for a layer-level diff
    torch.set_default_device("cuda")
    with torch.inference_mode():
        ref_vit_out = ref_vit(patches_gpu, n_vit_h, n_vit_w)
    torch.set_default_device("cpu")
    for mode in ("gpu", "hybrid"):
        best = None
        out = None
        for _ in range(3):
            out, vit_out, dt = run_ours(mode)
            best = dt if best is None else min(best, dt)
        dt = best
        vit_diff = (vit_out.float() - ref_vit_out.float()).abs().max().item()
        print(f"  [probe] ours[{mode}] ViT-vs-refViT max_abs={vit_diff:.5f}")
        diff = (out.float() - ref_out.float()).abs()
        rel = diff.max() / ref_out.float().abs().max()
        cos = torch.nn.functional.cosine_similarity(
            out.float().flatten(), ref_out.float().flatten(), dim=0
        ).item()
        print(
            f"ours[{mode}]: {dt*1000:.1f} ms  max_abs={diff.max().item():.4f} "
            f"rel={rel.item():.4%} cos={cos:.6f}"
        )
        if mode == "gpu":
            # Near-exact port of the reference tower. Not bit-exact: our
            # _vision_sdpa adds a batch dim so the flash kernel engages,
            # while the reference's 3-D SDPA call silently falls back to
            # the math path — bf16 kernel-rounding drift only. Must stay
            # far above the hybrid INT8 floor below.
            assert cos > 0.9999, f"gpu mode drifted from the reference (cos={cos})"
        else:
            # INT8 (2-way K-split) noise floor: same trade the MoE INT8
            # prefill makes; validated end-to-end by the benchmark-table QA.
            assert cos > 0.995, f"hybrid int8 noise too large (cos={cos})"
    print("vision tower parity: OK")

    # ---------------- window logic ----------------
    from model import get_image_visible, get_window_topk_idxs_visible
    from image_processor import IMAGE_END, IMAGE_START

    vocab = cfg["vocab_size"]
    max_img = cfg["vision_max_n_token"]
    win = cfg["sliding_window"]
    seqlen = types.numel() + 12
    ids = torch.randint(0, vocab, (1, seqlen))
    s = 4
    e = s + types.numel() - 1
    ids[0, s] = vocab + IMAGE_START
    ids[0, e] = vocab + IMAGE_END
    ids[0, s + 1 : e] = vocab + 2  # IMAGE

    left, right = get_image_visible(ids, vocab, max_img)
    ref_idx = get_window_topk_idxs_visible(win, seqlen, left, right, max_img)[0]

    # our math (mirrors _extend_swa_windows_for_images): treat index i as
    # "swa slot of position i" so we can compare index sets directly.
    ar = torch.arange(seqlen)
    k_past = torch.maximum(
        torch.clamp(ar, max=win - 1) + 1, left[0].long() + 1
    )
    row_len = k_past + right[0].long()
    W = ((win + 2 * max_img + 63) // 64) * 64
    k = torch.arange(W).unsqueeze(0)
    is_past = k < k_past.unsqueeze(1)
    col = torch.where(is_past, ar.unsqueeze(1) - k, ar.unsqueeze(1) + 1 + (k - k_past.unsqueeze(1)))
    valid = k < row_len.unsqueeze(1)
    our = torch.where(valid, col, torch.full_like(col, -1))
    our_set = set(our[our >= 0].tolist())
    ref_set = set(ref_idx[ref_idx >= 0].tolist())
    assert our_set == ref_set, (
        f"window mismatch: only-ours={sorted(our_set - ref_set)[:10]} "
        f"only-ref={sorted(ref_set - our_set)[:10]}"
    )
    for q in (s, s + 5, (s + e) // 2, e - 1, e, e + 3):
        a1 = set(ref_idx[q][ref_idx[q] >= 0].tolist())
        a2 = set(our[q].tolist()) - {-1}
        assert a1 == a2, f"q={q}: ref={sorted(a1)[:12]} ours={sorted(a2)[:12]}"
    print(
        f"image-visible window parity: OK (span [{s},{e}], "
        f"max row len {int(row_len.max())})"
    )


if __name__ == "__main__":
    main()
