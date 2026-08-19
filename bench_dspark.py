#!/usr/bin/env python3
"""DSpark 正确性 + 吞吐基准（5 prompt，贪心，与 DSv4F-Opt.md §7 同一套）。

用法: python3 bench_dspark.py [port]
"""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 30001
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

PROMPTS = [
    # 1. 算术（可精确校验）
    {
        "prompt": "一个长方形的长是12厘米，宽是3厘米。请计算它的周长和面积，只输出两个数字，用逗号分隔。",
        "check": lambda t: "30" in t and "36" in t,
        "max_tokens": 600,
    },
    # 2. 翻译
    {
        "prompt": "把这句话翻译成英文：人工智能正在改变世界，但我们需要谨慎地使用它。",
        "check": lambda t: "artificial intelligence" in t.lower()
        or "AI is" in t
        or "changing the world" in t,
        "max_tokens": 600,
    },
    # 3. 事实问答
    {
        "prompt": "光在真空中的传播速度大约是多少？请给出数值和单位。",
        "check": lambda t: "300" in t or "299" in t or "3×10" in t or "3 x 10" in t,
        "max_tokens": 600,
    },
    # 4. 作文（连贯性，人工抽查）
    {
        "prompt": "写一段150字左右的短文，主题是“城市里的自行车”。",
        "check": lambda t: len(t) > 100,
        "max_tokens": 1200,
    },
    # 5. 代码
    {
        "prompt": "写一个Python函数，判断一个字符串是否是回文，忽略大小写和空格。只给代码，不要解释。",
        "check": lambda t: "def " in t and ("[::-1]" in t or "reversed" in t or "reverse" in t.lower()),
        "max_tokens": 800,
    },
]


def run_one(p):
    body = json.dumps(
        {
            "model": "ds4f",
            "messages": [{"role": "user", "content": p["prompt"]}],
            "max_tokens": p["max_tokens"],
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    dt = time.time() - t0
    msg = out["choices"][0]["message"]
    text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    comp = out["usage"]["completion_tokens"]
    return text, comp, dt


total_tokens = 0
total_time = 0.0
all_ok = True
for i, p in enumerate(PROMPTS, 1):
    text, comp, dt = run_one(p)
    ok = p["check"](text)
    all_ok &= ok
    total_tokens += comp
    total_time += dt
    tps = comp / dt
    print(f"[{i}] {'PASS' if ok else 'FAIL'} {comp} tok / {dt:.1f}s = {tps:.2f} tok/s")
    print(f"    head: {text[:80]!r}")

print(f"\nTOTAL: {total_tokens} tok / {total_time:.1f}s = {total_tokens/total_time:.2f} tok/s")
print("ALL PASS" if all_ok else "SOME FAILED")
