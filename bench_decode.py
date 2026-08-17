#!/usr/bin/env python
"""DSv4-Flash decoding 基准：单流 decode，测稳态 ITL/吞吐。

用法: python bench_decode.py [max_tokens] [n_runs]
排除前 16 个 token（含首 token 延迟与预热），报告稳态吞吐。
"""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
URL = f"http://127.0.0.1:{PORT}/generate"
WARMUP_SKIP = 16


def run_once(max_tokens: int) -> tuple[float, int]:
    payload = json.dumps({
        "text": "User: 写一篇关于人工智能的短文。\nAssistant:",
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        },
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        URL, data=payload, headers={"Content-Type": "application/json"})

    stamps: list[float] = []
    with urllib.request.urlopen(req, timeout=600) as resp:
        buf = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk
            now = time.perf_counter()
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if line.startswith(b"data:"):
                    line = line[5:].strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "")
                if text:  # 只有带增量文本的帧才算一个 token
                    stamps.append(now)
    if len(stamps) <= WARMUP_SKIP + 1:
        return 0.0, len(stamps)
    steady = stamps[WARMUP_SKIP:]
    dt = steady[-1] - steady[0]
    n = len(steady) - 1
    return n / dt, len(stamps)


def main() -> None:
    max_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    rates = []
    for i in range(n_runs):
        tps, ntok = run_once(max_tokens)
        rates.append(tps)
        print(f"run {i + 1}: {tps:.2f} tok/s  ({ntok} tokens streamed)", flush=True)
    steady = [r for r in rates if r > 0]
    if steady:
        best = max(steady)
        avg = sum(steady) / len(steady)
        print(f"RESULT: avg={avg:.2f} best={best:.2f} tok/s (max_tokens={max_tokens}, skip={WARMUP_SKIP})")


if __name__ == "__main__":
    main()
