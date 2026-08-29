#!/usr/bin/env python3
"""Unit tests for the decode-phase reduced top-k experiment.

Covers:
1. --kt-decode-topk-layers / SGLANG_KT_DECODE_TOPK_LAYERS spec parsing
   (deepseek_v2.DeepseekV2MoE._decode_topk_layer_cfg).
2. KExpertsCPUBuffer (batch, k) composite-key caching: decode-width and
   prefill-width buffers for the same batch size must be distinct and
   correctly shaped.

Run: $KT_ROOT/.venv/bin/python tests/test_decode_topk.py
(the sglang import needs the repo venv; no GPU required).
"""

import os
import sys
import unittest
from unittest import mock

KT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _FakeMoe:
    def __init__(self, layers=None, k=None):
        self.kt_decode_topk_layers = layers
        self.kt_decode_topk_k = k


class _FakeExec:
    def __init__(self, moe):
        self.moe = moe


class DecodeTopkCfgTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, os.path.join(KT_ROOT, "third_party", "sglang", "python"))
        import sglang.srt.models.deepseek_v2 as dsv2

        self.dsv2 = dsv2
        self._old_env_layers = os.environ.pop("SGLANG_KT_DECODE_TOPK_LAYERS", None)
        self._old_env_k = os.environ.pop("SGLANG_KT_DECODE_TOPK_K", None)

    def tearDown(self):
        if self._old_env_layers is not None:
            os.environ["SGLANG_KT_DECODE_TOPK_LAYERS"] = self._old_env_layers
        if self._old_env_k is not None:
            os.environ["SGLANG_KT_DECODE_TOPK_K"] = self._old_env_k

    def _cfg(self, layers=None, k=None, env_layers=None, env_k=None):
        # Reset the per-process cache and run one fresh resolution with a
        # stubbed exec config / environment.
        self.dsv2._DECODE_TOPK_LAYER_CFG_CACHE = (
            self.dsv2._DECODE_TOPK_LAYER_CFG_UNSET
        )
        os.environ.pop("SGLANG_KT_DECODE_TOPK_LAYERS", None)
        os.environ.pop("SGLANG_KT_DECODE_TOPK_K", None)
        if env_layers is not None:
            os.environ["SGLANG_KT_DECODE_TOPK_LAYERS"] = env_layers
        if env_k is not None:
            os.environ["SGLANG_KT_DECODE_TOPK_K"] = env_k
        with mock.patch.object(
            self.dsv2, "get_exec", return_value=_FakeExec(_FakeMoe(layers, k))
        ):
            return self.dsv2.DeepseekV2MoE._decode_topk_layer_cfg()

    def test_disabled_by_default(self):
        cfg = self._cfg(layers=None, k=None, env_layers=None)
        self.assertIsNone(cfg)

    def test_range_parse(self):
        cfg = self._cfg(layers="3-20")
        self.assertEqual(cfg, {L: 4 for L in range(3, 21)})

    def test_single_layer_parse(self):
        cfg = self._cfg(layers="7")
        self.assertEqual(cfg, {7: 4})

    def test_range_with_k(self):
        cfg = self._cfg(layers="3-20=3")
        self.assertEqual(cfg, {L: 3 for L in range(3, 21)})

    def test_per_layer_map(self):
        cfg = self._cfg(layers="0=6,3-20=4,21=6,42=6")
        self.assertEqual(cfg[0], 6)
        self.assertEqual(cfg[3], 4)
        self.assertEqual(cfg[20], 4)
        self.assertEqual(cfg[21], 6)
        self.assertEqual(cfg[42], 6)
        self.assertNotIn(1, cfg)  # layer 1 unspecified -> native behavior

    def test_later_item_wins_on_overlap(self):
        cfg = self._cfg(layers="3-10=4,7=3")
        self.assertEqual(cfg[7], 3)
        self.assertEqual(cfg[6], 4)

    def test_overlap_later_item_wins(self):
        cfg = self._cfg(layers="3-10=4,7=3")
        self.assertEqual(cfg[7], 3)
        self.assertEqual(cfg[6], 4)
        self.assertEqual(cfg[8], 4)

    def test_summary_compression(self):
        cfg = self._cfg(layers="3-20=4,25=3")
        self.assertEqual(
            self.dsv2.DeepseekV2MoE._decode_topk_cfg_summary(cfg), "3-20=4,25=3"
        )

    def test_env_fallback(self):
        cfg = self._cfg(layers=None, env_layers="5-9=2")
        self.assertEqual(cfg, {L: 2 for L in range(5, 10)})

    def test_arg_overrides_env(self):
        cfg = self._cfg(layers="3-4=5", env_layers="10-12=6")
        self.assertEqual(cfg, {3: 5, 4: 5})

    def test_default_k(self):
        cfg = self._cfg(layers="3-4", env_k=None)
        self.assertEqual(cfg, {3: 4, 4: 4})

    def test_skip_parse(self):
        # k=0 = skip the layer's whole MoE in decode; must be explicit.
        cfg = self._cfg(layers="3-18=4,19-22=0")
        self.assertEqual(cfg[3], 4)
        self.assertEqual(cfg[18], 4)
        self.assertEqual({L: cfg[L] for L in range(19, 23)}, {L: 0 for L in range(19, 23)})
        self.assertNotIn(23, cfg)  # layer 23 unspecified -> native behavior

    def test_skip_mixed_summary(self):
        cfg = self._cfg(layers="3-18=4,19-22=0")
        self.assertEqual(
            self.dsv2.DeepseekV2MoE._decode_topk_cfg_summary(cfg), "3-18=4,19-22=0"
        )

    def test_rejects_garbage(self):
        for bad in ["3-20-7", "a-b", "20-3", "-3", "3..20", "3=x", "3-20=4,5=x"]:
            with self.assertRaises(ValueError, msg=bad):
                self._cfg(layers=bad)
        with self.assertRaises(ValueError):
            self._cfg(layers="3-20", k=0)
        with self.assertRaises(ValueError):
            self._cfg(layers="3-20", env_k="0")


class KExpertsCPUBufferKeyTests(unittest.TestCase):
    """Tests against the INSTALLED kt_kernel (reinstall kt-kernel first:
    cd kt-kernel && pip install . --no-deps --no-build-isolation)."""

    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        from kt_kernel.experts_base import BaseMoEWrapper, KExpertsCPUBuffer

        cls.buf_cls = KExpertsCPUBuffer
        cls.wrapper_cls = BaseMoEWrapper

    def setUp(self):
        # Fresh state per test.
        self.buf_cls.capture_bs = []
        self.buf_cls.capture_buffers = {}
        self.buf_cls.temp_key = None
        self.buf_cls.temp_buffer = tuple()

    def _hidden(self, bs, hidden=8):
        return self.torch.zeros(bs, hidden)

    def test_same_batch_different_k_distinct(self):
        b6 = self.buf_cls.get_buffer(self._hidden(8), 6)
        b4 = self.buf_cls.get_buffer(self._hidden(8), 4)
        self.assertIsNot(b6, b4)
        self.assertEqual(tuple(b6[1][0].shape), (8, 6))
        self.assertEqual(tuple(b4[1][0].shape), (8, 4))
        self.assertEqual(tuple(b4[3][0].shape), (8, 4))

    def test_temp_slot_reuse_same_key(self):
        b1 = self.buf_cls.get_buffer(self._hidden(8), 6)
        self.buf_cls.get_buffer(self._hidden(4), 6)  # evicts temp slot
        b2 = self.buf_cls.get_buffer(self._hidden(8), 6)
        self.assertIsNot(b1, b2)  # evicted -> fresh allocation, correct shape
        self.assertEqual(tuple(b2[1][0].shape), (8, 6))

    def test_capture_promotion_per_key(self):
        self.buf_cls.capture_bs = [8]
        b6 = self.buf_cls.get_buffer(self._hidden(8), 6)
        self.buf_cls.get_buffer(self._hidden(4), 6)  # temp eviction attempt
        b6_again = self.buf_cls.get_buffer(self._hidden(8), 6)
        self.assertIs(b6, b6_again)  # capture_buffers keeps (8, 6) alive
        b4 = self.buf_cls.get_buffer(self._hidden(8), 4)
        b4_again = self.buf_cls.get_buffer(self._hidden(8), 4)
        self.assertIs(b4, b4_again)
        self.assertIsNot(b4, b6)

    def test_clear_buffer_cache(self):
        self.buf_cls.capture_bs = [8]
        self.buf_cls.get_buffer(self._hidden(8), 6)
        self.wrapper_cls.clear_buffer_cache()
        self.assertEqual(self.buf_cls.capture_buffers, {})
        self.assertIsNone(self.buf_cls.temp_key)


class GpuCapHalfLogicTests(unittest.TestCase):
    """Pure-tensor check of the SGLANG_KT_DECODE_GPU_CAP_HALF slot split:
    keep = resident & (cumcount(resident) <= N//2); CPU ids blank kept slots
    with -1, GPU ids blank everything else."""

    def setUp(self):
        import torch

        self.torch = torch
        torch.manual_seed(0)

    def _split(self, ids, resident_mask, cap_on):
        torch = self.torch
        if cap_on:
            cap = ids.shape[-1] // 2
            rank = torch.cumsum(resident_mask.to(torch.int32), dim=1)
            keep = resident_mask & (rank <= cap)
        else:
            keep = resident_mask
        cpu_ids = torch.where(keep, torch.full_like(ids, -1), ids)
        return keep, cpu_ids

    def test_full_resident_layer_3F(self):
        # 3F layer decode: every slot resident, N=6 -> GPU keeps 3, CPU gets 3.
        ids = self.torch.arange(24).reshape(4, 6) % 256
        resident = self.torch.ones_like(ids, dtype=self.torch.bool)
        keep, cpu_ids = self._split(ids, resident, cap_on=True)
        self.assertEqual(int(keep.sum(dim=1).min()), 3)
        self.assertEqual(int(keep.sum(dim=1).max()), 3)
        self.assertEqual(
            int((cpu_ids == -1).sum(dim=1).min()), 3
        )  # CPU computes exactly the other 3
        # positional order: first 3 slots kept, last 3 handed back
        self.assertTrue(bool(keep[:, :3].all()))
        self.assertTrue(bool(~keep[:, 3:].any()))

    def test_sparse_resident_layer(self):
        # 27U layer decode, N=4, occasional resident hits: cap=2 almost never
        # binds; handed-back count = max(resident_hits - 2, 0).
        ids = self.torch.tensor(
            [[5, 7, 200, 9], [5, 7, 8, 9], [300, 301, 302, 303], [1, 2, 3, 4]]
        )
        resident_ids = {5, 7, 300}
        resident = self.torch.zeros_like(ids, dtype=self.torch.bool)
        for t in range(ids.shape[0]):
            for j in range(ids.shape[1]):
                resident[t, j] = ids[t, j].item() in resident_ids
        keep, cpu_ids = self._split(ids, resident, cap_on=True)
        # row0: 2 resident hits, both kept; row1: 2 hits capped to 2 (fine);
        # row2: 1 hit kept; row3: 0 hits
        self.assertEqual(int(keep[0].sum()), 2)
        self.assertEqual(int(keep[1].sum()), 2)
        self.assertEqual(int(keep[2].sum()), 1)
        self.assertEqual(int(keep[3].sum()), 0)
        # row with 3 hits would bind: add one
        ids2 = self.torch.tensor([[5, 7, 300, 9]])
        res2 = self.torch.tensor([[True, True, True, False]])
        keep2, cpu2 = self._split(ids2, res2, cap_on=True)
        self.assertEqual(int(keep2.sum()), 2)  # capped at N//2 = 2
        self.assertEqual(int((cpu2 == -1).sum()), 2)
        self.assertEqual(int(cpu2[0, 2]), 300)  # 3rd hit handed to CPU

    def test_prefill_keeps_all_resident(self):
        ids = self.torch.arange(12).reshape(2, 6) % 256
        resident = self.torch.ones_like(ids, dtype=self.torch.bool)
        keep, cpu_ids = self._split(ids, resident, cap_on=False)
        self.assertTrue(bool(keep.all()))
        self.assertTrue(bool((cpu_ids == -1).all()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
