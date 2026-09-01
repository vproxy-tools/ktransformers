#!/usr/bin/env python3
"""Interleaved text/image/text (~100K) correctness + prefill profiling.

Prompt layout (one user turn, content blocks):
  [text ~1/3 with 暗号1] [image1 = benchmark table] [text ~1/3 with 暗号2]
  [image2 = generated code image] [text ~1/3] [final question]

Verifies: both secrets recalled, both images read. Reports per-batch prefill
stats from the server log (batch sizes, tok/s, any span-extended batches).

Usage: .venv/bin/python tests/vision_interleave_test.py [target_tokens]
"""

import base64
import io
import json
import re
import subprocess
import sys
import time
import urllib.request

PORT = 30000
MODEL = "/var/deepseek-v4-flash/vision-exp"
IMAGE1 = "/home/wkgcass/vision-sample.png"
SECRET1 = "绯红砂岩3311"
SECRET2 = "墨绿罗盘6604"
CODE2 = "OMEGA-881"
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000

FILLER = (
    "这一段是交叉场景的填充文本，用于把图片分隔到长上下文的不同位置。文本本身不"
    "承载关键信息，模型只需在最后的提问里回忆两处暗号并分别读取两张图片的内容。"
)  # ~58 chars


def make_code_image() -> str:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (760, 420), (24, 108, 52))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 110
        )
    except OSError:
        font = ImageFont.load_default(size=110)
    d.text((60, 150), CODE2, fill=(255, 224, 64), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    with open(IMAGE1, "rb") as f:
        img1 = base64.b64encode(f.read()).decode()
    img2 = make_code_image()

    # ~67 tokens per filler paragraph line (measured for this tokenizer)
    per_seg = max(int(TARGET / 3 / 67), 1)

    def seg(tag, n):
        return "\n".join(f"{tag}第{i}段：{FILLER}" for i in range(n))

    blocks = [
        {"type": "text", "text": f"第一段文本开始。暗号一是：{SECRET1}。\n\n" + seg("甲", per_seg)},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img1}},
        {"type": "text", "text": f"第二段文本开始。暗号二是：{SECRET2}。\n\n" + seg("乙", per_seg)},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img2}},
        {"type": "text", "text": seg("丙", per_seg) + "\n\n最后的提问：请依次回答——"
            "1) 暗号一是什么？2) 暗号二是什么？3) 第一张图片表格里 Terminal Bench 2.1 "
            "一行 DeepSeek V4-Flash-0731 的分数？4) 第二张图片里的代码是什么？"
            "请按行作答，不要多余内容。"},
    ]
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": blocks}],
        "temperature": 0.0,
        "max_tokens": 200,
        "chat_template_kwargs": {"thinking": False},
    }
    print(f"sending interleaved ~{TARGET}-token request "
          f"({len(json.dumps(payload))//1024//1024} MiB payload)")
    t0 = time.time()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.load(urllib.request.urlopen(req, timeout=3600))
    dt = time.time() - t0
    content = resp["choices"][0]["message"]["content"]
    print(f"e2e {dt/60:.1f} min ({dt:.0f}s)")
    print("ANSWER:")
    print(content[:400])

    checks = {
        "secret1": SECRET1 in content,
        "secret2": SECRET2 in content,
        "table(82.7)": "82.7" in content,
        "code2": CODE2 in content,
    }
    for k, v in checks.items():
        print(f"  {k}: {v}")

    log = subprocess.run(
        ["journalctl", "-u", "ds4f", "--since", f"-{int(dt)+60}s", "--no-pager"],
        capture_output=True, text=True,
    ).stdout
    rows = re.findall(
        r"Prefill batch.*?#new-token: (\d+), #cached-token: (\d+).*?throughput \(token/s\): ([\d.]+)",
        log,
    )
    news = [int(n) for n, _, _ in rows]
    tps = [float(t) for *_, t in rows]
    mine = [i for i, n in enumerate(news) if 0 < n]
    expanded = [l for l in log.splitlines() if "expanded" in l and "image" in l]
    big = [n for n in news if n > 1024]
    print(f"prefill batches={len(news)} total_new={sum(news)} "
          f"cached={sum(int(c) for _,c,_ in rows)}")
    print(f"batch sizes: max={max(news) if news else 0} >1024:{len(big)} {big[:6]}")
    print(f"reported tok/s: first={tps[0] if tps else 0:.0f} "
          f"last={tps[-1] if tps else 0:.0f}")
    for l in expanded[-2:]:
        print(l.split("ds4f")[-1][:130])
    assert all(checks.values()), f"FAILED: {checks}"
    print("PASS")


if __name__ == "__main__":
    main()
