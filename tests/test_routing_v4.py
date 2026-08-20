#!/usr/bin/env python
"""Unit test: V4 GPU-expert routing construction + CUDA-graph capture safety.

Verifies the SparseMatrix-based _make_routing_data_v4 (the triton_kernels
0.1.0 lineage ships no `routing` module — see DSv4F-Opt.md §5.3):
  1. routing structures build on GPU for the V4 shape (256 experts, top-6
     padded to pow2 slots);
  2. the whole construction is CUDA-graph capturable (decode graphs bake
     it in; a torch fallback with argsort/histc is NOT capturable).

Usage: .venv/bin/python tests/test_routing_v4.py
"""
import sys

import torch

sys.path.insert(0, "/home/wkgcass/ktransformers/third_party/sglang/python")
from sglang.srt.layers.quantization.v4_triton_kernels_moe import (  # noqa: E402
    _make_routing_data_v4,
)


def main():
    M, topk, E = 512, 6, 256
    ids = torch.stack([torch.randperm(E)[:topk] for _ in range(M)]).cuda().int()
    w = torch.rand(M, topk).cuda().bfloat16()
    rd, gi, si = _make_routing_data_v4(ids, w, E)
    assert rd.gate_scal.shape == (M * 8,), rd.gate_scal.shape  # pow2 padding 6->8
    assert gi.src_indx.shape == (M * 8,)
    assert rd.n_expts_tot == E and rd.n_expts_act == 8
    print("routing ok:", tuple(rd.gate_scal.shape), "act/tot", rd.n_expts_act, rd.n_expts_tot)

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _make_routing_data_v4(ids, w, E)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        rd2, gi2, si2 = _make_routing_data_v4(ids, w, E)
    g.replay()
    print("graph capture OK, gate_scal", tuple(rd2.gate_scal.shape))
    print("PASS")


if __name__ == "__main__":
    main()
