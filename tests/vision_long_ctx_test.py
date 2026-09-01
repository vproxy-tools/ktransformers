#!/usr/bin/env python3
"""700K-token prefill WITH an image through /v1/chat/completions.

Builds one user turn: [image][~700K tokens of filler with a secret code near
the start][question], then verifies the model both recalls the code and reads
the table. Reports prefill progression from the server log.

Usage: .venv/bin/python tests/vision_long_ctx_test.py [target_tokens]
"""

import base64
import json
import re
import subprocess
import sys
import time
import urllib.request

PORT = 30000
MODEL = "/var/deepseek-v4-flash/vision-exp"
IMAGE = "/home/wkgcass/vision-sample.png"
TARGET_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 700_000
SECRET = "蓝色风筝7429"

# ~1.4 tokens per Chinese char for this tokenizer; aim slightly under target
# and let the final count come from the response usage/log.
FILLER_PARA = (
    "这是一段用于测试长上下文预填充的填充文本。模型需要在不丢失前文信息的情况下"
    "处理完整的上下文。每一段内容彼此独立，不携带关键信息，只有开头的暗号和结尾的"
    "任务是真正需要关注的。持续的长文本会考验键值缓存与稀疏注意力的正确性。"
)  # ~85 chars ≈ ~118 tokens


def main():
    with open(IMAGE, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    # measured ~67 tokens per paragraph line for this tokenizer
    n_para = int(TARGET_TOKENS / 67)
    filler = f"暗号是：{SECRET}。请记住它。\n\n" + "\n".join(
        f"第{i}段：{FILLER_PARA}" for i in range(n_para)
    )
    question = (
        "\n\n最后请回答两个问题：1) 开头的暗号是什么？2) 图片表格里 Terminal Bench 2.1 "
        "一行 DeepSeek V4-Flash-0731 的分数是多少？请按 '暗号=xxx; 分数=yyy' 格式回答。"
    )
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                    {"type": "text", "text": filler + question},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 200,
        "chat_template_kwargs": {"thinking": False},
    }
    print(f"sending: ~{n_para} paragraphs, payload {len(json.dumps(payload))//1024//1024} MiB")
    t0 = time.time()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.load(urllib.request.urlopen(req, timeout=7200))
    dt = time.time() - t0
    content = resp["choices"][0]["message"]["content"]
    print(f"e2e {dt/60:.1f} min ({dt:.0f}s)")
    print("ANSWER:", content[:200].replace("\n", " | "))

    ok_secret = SECRET in content
    ok_score = "82.7" in content
    print(f"secret recalled: {ok_secret}")
    print(f"table score (82.7) read: {ok_score}")

    log = subprocess.run(
        ["journalctl", "-u", "ds4f", "--since", f"-{int(dt)+60}s", "--no-pager"],
        capture_output=True, text=True,
    ).stdout
    prefills = re.findall(r"Prefill batch.*?#new-token: (\d+), #cached-token: (\d+)", log)
    news = [int(n) for n, _ in prefills]
    cached = [int(c) for _, c in prefills]
    expanded = [l for l in log.splitlines() if "expanded 1 image" in l]
    print(f"prefill batches: {len(news)}, total new {sum(news)}, total cached {sum(cached)}")
    print(f"max single batch: {max(news) if news else 0}  (chunked => should stay ~1024-ish)")
    for l in expanded[-2:]:
        print(l.split("ds4f")[-1][:150])
    assert ok_secret and ok_score, "FAILED"
    print("PASS")


if __name__ == "__main__":
    main()
