#!/usr/bin/env python3
"""Concurrency regression test: text request in DSpark verify while a
concurrent image request is still in chunked prefill.

Crash being pinned (2026-09-01, production): ForwardBatch.init_new computed
compute_mm_token_mask for TARGET_VERIFY batches too; with a second request
still mid-prefill, the schedule batch's multimodal_inputs no longer aligned
with the verify batch's extend lens -> zip(strict=True) ValueError -> whole
server down. Fix: the mask is only computed for extend-mode batches.

Sequence: fire a long-decoding text request A and an image+long-text
request B concurrently; A must decode (verify) while B is still chunking
through prefill. PASS = both requests answer correctly and the server is
still healthy afterwards.

Usage: .venv/bin/python tests/vision_concurrent_verify_test.py
"""

import base64
import json
import threading
import time
import urllib.request

PORT = 30000
MODEL = "/var/deepseek-v4-flash/vision-exp"
IMAGE1 = "/home/wkgcass/vision-sample.png"

FILLER = (
    "这一段是并发场景的填充文本，用于把图片请求的分块预填充拖长到足以覆盖"
    "另一个文本请求的投机解码窗口。文本本身不承载关键信息。"
)  # ~55 chars


def chat(blocks_or_text, max_tokens):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": blocks_or_text}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"thinking": False},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=1200))


def health_ok():
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}/health", timeout=5
        ) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    with open(IMAGE1, "rb") as f:
        img1 = base64.b64encode(f.read()).decode()

    filler = "\n".join(f"第{i}段：{FILLER}" for i in range(130))  # ~9K tokens
    blocks_b = [
        {"type": "text", "text": "开头。\n\n" + filler},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img1}},
        {"type": "text", "text": "结尾提问：图片表格里 Terminal Bench 2.1 一行 "
            "DeepSeek V4-Flash-0731 的分数是多少？只回答数字。"},
    ]

    results = {}

    def run_a():
        results["a"] = chat(
            "写一篇 400 字说明文介绍自来水处理流程，文末单独一行给出水的化学式。",
            600,
        )

    def run_b():
        results["b"] = chat(blocks_b, 100)

    ta = threading.Thread(target=run_a)
    tb = threading.Thread(target=run_b)
    t0 = time.time()
    ta.start()
    time.sleep(0.5)  # A enters decode first; B then chunks through prefill
    tb.start()
    ta.join()
    tb.join()
    dt = time.time() - t0

    ok = True
    a_txt = results["a"]["choices"][0]["message"]["content"]
    b_txt = results["b"]["choices"][0]["message"]["content"]
    print(f"A ({dt:.0f}s window): {a_txt[:80]!r}")
    print(f"B: {b_txt[:80]!r}")
    if "H2O" not in a_txt.replace(" ", "").replace("₂", "2"):
        print("FAIL: A answer wrong")
        ok = False
    if "82.7" not in b_txt:
        print("FAIL: B did not read the table (82.7)")
        ok = False
    time.sleep(2)
    if not health_ok():
        print("FAIL: server not healthy after concurrent mm+verify")
        ok = False
    assert ok, "FAILED"
    print("PASS")


if __name__ == "__main__":
    main()
