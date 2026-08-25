"""Unit tests for the SGLANG_DSPARK_MAX_CTX gate (no GPU required).

Covers the pure threshold predicate used by
DSparkWorkerV2._forward_decode and the env registration defaults.
Run: .venv/bin/python tests/test_dspark_max_ctx.py
"""

import os

from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    dspark_ctx_over_limit,
)


def test_gate_disabled():
    # 0 (and negative) disables the gate entirely, regardless of lengths.
    assert not dspark_ctx_over_limit([], 0)
    assert not dspark_ctx_over_limit([10**9], 0)
    assert not dspark_ctx_over_limit([262144, 10**9], -1)


def test_threshold_semantics():
    # Below threshold -> speculation keeps running.
    assert not dspark_ctx_over_limit([262143], 262144)
    assert not dspark_ctx_over_limit([100, 262143], 262144)
    # "达到该阈值后退化": seq_len == threshold already degrades.
    assert dspark_ctx_over_limit([262144], 262144)
    assert dspark_ctx_over_limit([262145], 262144)
    # Batch-level: ANY request over the line degrades the whole batch.
    assert dspark_ctx_over_limit([100, 262144], 262144)
    # Empty batch never triggers (defensive; idle path returns earlier).
    assert not dspark_ctx_over_limit([], 262144)


def test_accepts_tensor_like_values():
    class FakeScalar:
        def __init__(self, v):
            self.v = v

        def __int__(self):
            return self.v

    assert dspark_ctx_over_limit([FakeScalar(262144)], 262144)
    assert not dspark_ctx_over_limit([FakeScalar(5)], 262144)


def test_env_registration():
    from sglang.srt.environ import envs

    saved = os.environ.pop("SGLANG_DSPARK_MAX_CTX", None)
    try:
        # Default is 256K (K = 1024).
        assert envs.SGLANG_DSPARK_MAX_CTX.get() == 262144
        # Env override is honored (read per .get() call).
        os.environ["SGLANG_DSPARK_MAX_CTX"] = "4096"
        assert envs.SGLANG_DSPARK_MAX_CTX.get() == 4096
        os.environ["SGLANG_DSPARK_MAX_CTX"] = "0"
        assert envs.SGLANG_DSPARK_MAX_CTX.get() == 0
    finally:
        if saved is None:
            os.environ.pop("SGLANG_DSPARK_MAX_CTX", None)
        else:
            os.environ["SGLANG_DSPARK_MAX_CTX"] = saved


if __name__ == "__main__":
    test_gate_disabled()
    test_threshold_semantics()
    test_accepts_tensor_like_values()
    test_env_registration()
    print("PASS: test_dspark_max_ctx")
