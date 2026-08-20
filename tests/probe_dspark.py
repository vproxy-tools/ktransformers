#!/usr/bin/env python3
"""DSpark 损坏探针：3 个短生成，判 accept/思考解析/重复词/数学正确性。

用法: python3 probe_dspark.py [port]   退出码 0=干净, 1=损坏
"""
import json
import re
import sys
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 30001
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"


def gen(prompt, mt):
    body = json.dumps(
        {
            "model": "ds4f",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": mt,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    m = out["choices"][0]["message"]
    return (m.get("reasoning_content") or ""), (m.get("content") or "")


def dup_score(text):
    """连续 8+ 字符的 chunk 在邻近位置重复出现的次数（中文重复短语探测）"""
    hits = 0
    for i in range(0, len(text) - 16, 8):
        chunk = text[i : i + 10]
        if len(chunk) >= 8 and chunk in text[i + 10 : i + 60]:
            hits += 1
    return hits


fails = []

rc, c = gen("12乘以3等于多少？只回答算式和结果。", 150)
if "36" not in rc + c or "12" not in rc + c:
    fails.append(f"math wrong: {(rc+c)[:80]!r}")

rc, c = gen("把这句话翻译成英文：人工智能正在改变世界。", 200)
en = (c or "").strip()
if not en or not re.search(r"[A-Za-z]{4,}\s+[A-Za-z]{3,}", (c or "")):
    fails.append(f"translate bad: {(rc+c)[:80]!r}")
if len(rc) > 30 and dup_score(rc) > 2:
    fails.append(f"reasoning dup ({dup_score(rc)}): {rc[:100]!r}")

rc, c = gen("写一段100字左右的短文，主题是海洋。", 350)
body = rc + c
if len(body) < 80:
    fails.append(f"essay too short: {body[:80]!r}")
if dup_score(body) > 4:
    fails.append(f"essay dup ({dup_score(body)}): {body[:120]!r}")

if fails:
    print("CORRUPT:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("CLEAN")
