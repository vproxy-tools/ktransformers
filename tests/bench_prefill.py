#!/usr/bin/env python3
"""Prefill throughput benchmark for the DSv4-Flash kt server.

Sends long-prompt generation requests (max_new_tokens=1) and derives prefill
tok/s from server-reported TTFT / prompt length. Works against any sglang
OpenAI-compatible endpoint (default port 30001).

Usage:
    python3 tests/bench_prefill.py [port] [--tokens 8192] [--iters 3]

Output: one line per request `prefill= N tok / T s = X tok/s` plus TOTAL.
Judgment: compare across configs; correctness is NOT checked here
(use tests/probe_dspark.py for that).
"""
import argparse
import json
import time
import urllib.request


def build_prompt(n_tokens: int) -> str:
    # ~1 token per 3-4 chars of mixed CJK/latin prose; the server reports the
    # exact prompt_tokens so the estimate only sizes the request.
    words = []
    seed = 42
    for i in range(n_tokens):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        words.append(f"tok{i}-{seed % 9973}")
    return "Count the distinct markers in the following list:\n" + " ".join(words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", type=int, default=30001)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=1)
    args = ap.parse_args()

    prompt = build_prompt(args.tokens)
    url = f"http://127.0.0.1:{args.port}/generate"
    totals = []
    for it in range(args.iters):
        body = json.dumps({
            "text": prompt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": args.max_new},
        }).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=1800) as r:
            resp = json.loads(r.read())
        wall = time.perf_counter() - t0
        meta = resp.get("meta_info", {})
        ptoks = meta.get("prompt_tokens")
        ttft = meta.get("e2e_latency")  # max_new=1 ⇒ e2e ≈ prefill + 1 decode
        if ptoks and ttft:
            tps = ptoks / ttft
            totals.append(tps)
            print(f"[{it}] prompt={ptoks} tok  ttft={ttft:.2f}s  "
                  f"prefill={tps:.1f} tok/s  (wall {wall:.2f}s)")
        else:
            print(f"[{it}] meta missing: {meta}")
    if totals:
        print(f"TOTAL: {sum(totals)/len(totals):.1f} tok/s "
              f"(best {max(totals):.1f}, n={len(totals)})")


if __name__ == "__main__":
    main()
