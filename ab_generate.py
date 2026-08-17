#!/usr/bin/env python
"""A/B 生成: 对固定提示词集做贪心生成, 存到目录。

用法: python ab_generate.py <outdir> [port]
"""
import json
import sys
import time
import urllib.request

outdir = sys.argv[1]
port = sys.argv[2] if len(sys.argv) > 2 else 30001
prompts = [l.rstrip("\n") for l in open("/tmp/ab_prompts.txt") if l.strip()]

for i, p in enumerate(prompts):
    payload = json.dumps({
        "text": p + "\nAssistant:",
        "sampling_params": {"max_new_tokens": 256, "temperature": 0.0},
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate", data=payload,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=300))
    with open(f"{outdir}/{i:02d}.txt", "w") as f:
        f.write(r["text"])
    print(f"[{i}] {len(r['text'])} chars in {time.time()-t0:.1f}s", flush=True)
