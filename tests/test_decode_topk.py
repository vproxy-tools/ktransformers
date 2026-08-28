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
        layers, k = self._cfg(layers="3-20")
        self.assertEqual(layers, frozenset(range(3, 21)))
        self.assertEqual(k, 4)

    def test_single_layer_parse(self):
        layers, k = self._cfg(layers="7")
        self.assertEqual(layers, frozenset({7}))
        self.assertEqual(k, 4)

    def test_env_fallback(self):
        layers, k = self._cfg(layers=None, env_layers="5-9", env_k="2")
        self.assertEqual(layers, frozenset(range(5, 10)))
        self.assertEqual(k, 2)

    def test_arg_overrides_env(self):
        layers, k = self._cfg(layers="3-4", env_layers="10-12")
        self.assertEqual(layers, frozenset({3, 4}))

    def test_default_k(self):
        _, k = self._cfg(layers="3-4", env_k=None)
        self.assertEqual(k, 4)

    def test_rejects_garbage(self):
        for bad in ["3-20-7", "a-b", "20-3", "-3", "3..20"]:
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
