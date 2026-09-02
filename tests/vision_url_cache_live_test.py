#!/usr/bin/env python3
"""Real-environment test for the dsv4-vision URL image disk cache.

Prereqs: ds4f running on :30000 (with SGLANG_DSV4_IMAGE_CACHE_DIR set),
a local HTTP server serving the test image, e.g.:
    cd /tmp && python3 -m http.server 18099 --bind 127.0.0.1

Checks:
  1. URL image request succeeds and the model actually sees the image.
  2. Image bytes land in the per-pid disk cache dir; repeat fetches of the
     same URL do NOT re-hit the network (server logs "cached ... bytes"
     exactly once across all requests).
  3. Repeating the same image hits the radix prefix cache (pad_value path)
     — verified via the scheduler's "Prefill batch ... #cached-token: N"
     journal lines (ground truth; note the OpenAI usage field
     prompt_tokens_details.cached_tokens under-reports for image spans).

Run:  .venv/bin/python tests/vision_url_cache_live_test.py
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request

PORT = 30000
IMG_URL = "http://127.0.0.1:18099/dsv4-url-test.png"
CACHE_BASE = os.environ.get(
    "SGLANG_DSV4_IMAGE_CACHE_DIR", "/var/hicache/dsv4-image-cache")

PASS, FAIL = 0, 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def chat(question, max_tokens=1024):
    body = {
        "model": "dsv4",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": IMG_URL}},
                {"type": "text", "text": question},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def cache_files():
    if not os.path.isdir(CACHE_BASE):
        return []
    return [os.path.join(CACHE_BASE, f) for f in os.listdir(CACHE_BASE)
            if os.path.isfile(os.path.join(CACHE_BASE, f))
            and ".tmp." not in f]


def journal(since):
    out = subprocess.run(
        ["journalctl", "-u", "ds4f", "--since", since, "--no-pager"],
        capture_output=True, text=True).stdout
    return out


def main():
    t_start = time.strftime("%Y-%m-%d %H:%M:%S")
    pre_existing = len(cache_files()) > 0
    q1 = "What shapes and colors do you see? List every number visible in the image. Answer briefly."
    t0 = time.time()
    r1 = chat(q1)
    dt1 = time.time() - t0
    text1 = r1["choices"][0]["message"]["content"] or ""
    reasoning1 = r1["choices"][0]["message"].get("reasoning_content") or ""
    both1 = (text1 + reasoning1).lower()
    print(f"  [req1] {dt1:.1f}s usage={r1.get('usage')}")
    print(f"  [req1 answer] {text1[:300]}")

    check("req1 sees red", "red" in both1)
    check("req1 sees blue", "blue" in both1)
    check("req1 sees green", "green" in both1 or "circle" in both1)
    check("req1 reads number 12345", "12345" in both1)
    check("req1 reads number 67890", "67890" in both1)

    files1 = cache_files()
    check("cache file exists after req1", len(files1) >= 1, repr(files1))

    # identical request: bytes from disk + radix prefix hit
    t0 = time.time()
    r2 = chat(q1)
    dt2 = time.time() - t0
    print(f"  [req2] {dt2:.1f}s usage={r2.get('usage')}")
    files2 = cache_files()
    check("still exactly one cached file", len(files2) == 1, repr(files2))

    # same image, different question: image span prefix should still match
    q2 = "How many rectangles are in the image? Answer with one word."
    r3 = chat(q2)
    text3 = (r3["choices"][0]["message"]["content"] or "")
    reasoning3 = r3["choices"][0]["message"].get("reasoning_content") or ""
    print(f"  [req3] usage={r3.get('usage')}")
    print(f"  [req3 answer] {text3[:200]}")
    check("req3 answers two rectangles",
          "two" in (text3 + reasoning3).lower() or "2" in text3)

    # journal ground truth: at most one network fetch (0 if the cache file
    # survived from an earlier run in the same process), prefix hits after
    log = journal(t_start)
    n_fetch = len(re.findall(r"\[dsv4-vision\] cached \d+ bytes from", log))
    expected_fetch = 0 if pre_existing else 1
    check(f"network fetch count == {expected_fetch}",
          n_fetch == expected_fetch, f"got {n_fetch}")
    cached_tokens = [int(m) for m in re.findall(r"#cached-token: (\d+)", log)]
    check("scheduler logged prefix hits (>=2 batches with cached-token>0)",
          sum(1 for c in cached_tokens if c > 0) >= 2,
          f"cached-token series={cached_tokens}")

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
