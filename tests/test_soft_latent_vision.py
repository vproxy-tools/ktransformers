"""Integration test: vision + soft latent replay through the radix cache.

Runs against a live vision-exp server with SGLANG_SOFT_THINKING=1. NOTE
(2026-09-10): production (ds4f.service, port 30000) NO LONGER enables soft
latent — to run this test, start a dedicated latent-enabled vision instance,
e.g. temporarily add the SGLANG_SOFT_* envs to the unit (cache-hit assertions
read journald -u ds4f "Prefill batch ... #cached-token: N" lines), or point
--port at an experimental instance started from run_dspark.sh with the
vision-exp model (journal source must be adjusted accordingly).

Proves that:
1. an image + thinking request emits <|latent|> blocks (baseline);
2. replaying the conversation (image + assistant history with latent blocks
   + follow-up) is ACCEPTED (no more 400), the tokenizer manager remaps the
   latent positions past the image sentinel-block expansion
   ("remapped N latent position(s) past 1 image expansion(s)" in the log),
   and the restore path rebuilds the soft embeddings
   ("restored N latent position(s)");
3. the radix prefix match crosses BOTH the image span and the latent spans:
   an identical second replay hits the cache deep past the latent span,
   while a replay with one tampered latent weight only matches up to the
   first latent position (cached_tokens drops by ~the latent span length).

Usage: .venv/bin/python tests/test_soft_latent_vision.py [--port 30000]
"""

import argparse
import base64
import json
import re
import subprocess
import sys
import time

import requests

PASS = []


def check(name, cond, detail=""):
    PASS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


def chat(base, messages, max_tokens, temperature=0.0):
    r = requests.post(
        f"{base}/v1/chat/completions",
        json={
            "model": "dsv4",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        },
        timeout=1800,
    )
    if r.status_code != 200:
        print(f"[ERROR] {r.status_code}: {r.text[:2000]}")
    r.raise_for_status()
    return r.json()


def journal_prefill_stats(since: str):
    """Sum #new-token / #cached-token over Prefill batch lines since `since`."""
    out = subprocess.run(
        ["journalctl", "-u", "ds4f", "--since", since, "--output=cat",
         "--no-pager"],
        capture_output=True, text=True, timeout=60,
    ).stdout
    new_tok = cached = 0
    remapped = restored = 0
    for line in out.splitlines():
        m = re.search(r"#new-token: (\d+).*#cached-token: (\d+)", line)
        if m:
            new_tok += int(m.group(1))
            cached += int(m.group(2))
        m = re.search(r"remapped (\d+) latent position", line)
        if m:
            remapped += int(m.group(1))
        m = re.search(r"restored (\d+) latent position", line)
        if m:
            restored += int(m.group(1))
    return new_tok, cached, remapped, restored


def now_stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


LATENT_PREFIX = "<|latent|>"
LATENT_SUFFIX = "<|/latent|>"


def tamper_first_latent(reasoning: str) -> str:
    """Perturb one weight in the first <|latent|> block (same argmax)."""
    sys.path.insert(
        0, __file__.rsplit("/", 2)[0] + "/third_party/sglang/python"
    )
    from sglang.srt.layers.soft_latent import decode_position, encode_position

    start = reasoning.index(LATENT_PREFIX) + len(LATENT_PREFIX)
    end = reasoning.index(LATENT_SUFFIX)
    cands = decode_position(reasoning[start:end])
    i = 1 if len(cands) > 1 else 0
    cands[i] = (cands[i][0], cands[i][1] * 0.5 + 0.01)
    new_block = LATENT_PREFIX + encode_position(cands) + LATENT_SUFFIX
    return (
        reasoning[: start - len(LATENT_PREFIX)]
        + new_block
        + reasoning[end + len(LATENT_SUFFIX):]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--image", default="/home/wkgcass/vision-sample.png")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    with open(args.image, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    image_part = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
    }
    question = "这张图片里有哪些文字内容？请简要概括。"
    user_msg = {"role": "user", "content": [image_part,
                                            {"type": "text", "text": question}]}

    # --- Step 1: fresh image + thinking request ---------------------------
    # temperature 0.8: since Part A, greedy (temp=0) positions are all
    # single-candidate and emit as real tokens — latent blocks only appear
    # at multi-candidate positions.
    t0 = now_stamp()
    time.sleep(1)
    a = chat(base, [user_msg], max_tokens=3000, temperature=0.8)
    msg_a = a["choices"][0]["message"]
    reasoning = msg_a.get("reasoning_content") or ""
    content = msg_a.get("content") or ""
    n_latent = reasoning.count(LATENT_PREFIX)
    check(
        "step1 image request emits latent blocks",
        n_latent > 0,
        f"latent_blocks={n_latent} reasoning_chars={len(reasoning)}",
    )
    if n_latent == 0:
        print("no latent blocks; aborting")
        sys.exit(1)
    print(f"[INFO] step1 answer: {content[:120]!r}")

    # --- Step 2: replay with image + latent history (twice) ----------------
    filler = "（填充文字，与问题无关，请忽略）" * 60
    followup = f"{filler}用一句话复述你的最终答案。"
    replay_msgs = [
        user_msg,
        {"role": "assistant", "reasoning_content": reasoning,
         "content": content},
        {"role": "user", "content": followup},
    ]

    time.sleep(1)
    t1 = now_stamp()
    time.sleep(1)
    b1 = chat(base, replay_msgs, max_tokens=1024)
    new1, cached1, remapped1, restored1 = journal_prefill_stats(t1)
    check(
        "step2 replay accepted (no 400) + latent positions remapped past image",
        remapped1 >= n_latent,
        f"remapped={remapped1} n_latent={n_latent}",
    )
    check(
        "step2 soft embeddings restored at prefill",
        restored1 >= n_latent,
        f"restored={restored1}",
    )
    print(f"[INFO] step2 first replay: new={new1} cached={cached1}")

    time.sleep(1)
    t2 = now_stamp()
    time.sleep(1)
    b2 = chat(base, replay_msgs, max_tokens=1024)
    new2, cached2, remapped2, restored2 = journal_prefill_stats(t2)
    # Prefix match must cross the image span (~300 tokens) AND the whole
    # latent span plus the assistant content: cached should cover most of
    # the replayed prompt (prompt minus follow-up and page alignment).
    check(
        "step2 second replay hits radix cache through image + latent spans",
        cached2 >= n_latent + 256,
        f"cached={cached2} n_latent={n_latent} new={new2}",
    )
    ans2 = b2["choices"][0]["message"].get("content") or ""
    check("step2 replay answer non-empty", len(ans2) > 0, f"{ans2[:80]!r}")
    # The restore log fires at encode time on every replay (extraction
    # happens before the cache lookup), so it must appear again.
    check(
        "step2 second replay also extracts latent blocks",
        restored2 >= n_latent,
        f"restored={restored2}",
    )

    # --- Step 3: tampered latent block breaks the match at the latent span -
    tampered = tamper_first_latent(reasoning)
    tamper_msgs = [
        user_msg,
        {"role": "assistant", "reasoning_content": tampered,
         "content": content},
        {"role": "user", "content": followup},
    ]
    time.sleep(1)
    t3 = now_stamp()
    time.sleep(1)
    chat(base, tamper_msgs, max_tokens=256)
    _new3, cached3, _r, _s = journal_prefill_stats(t3)
    # The match must break at the first latent position: cached covers the
    # system+user turns (incl. the image span) and the assistant header,
    # but NOT the latent span + content that the full replay matched.
    check(
        "step3 tampered latent breaks match (cached drops by ~latent span)",
        cached2 - cached3 >= n_latent,
        f"full={cached2} tampered={cached3} n_latent={n_latent}",
    )
    check(
        "step3 tampered replay still matches prefix up to latent span",
        cached3 > 0,
        f"cached={cached3} (image span + first turn shared)",
    )

    n_fail = sum(1 for _, ok in PASS if not ok)
    print(
        f"\n{'ALL PASS' if n_fail == 0 else f'{n_fail} FAILED'} "
        f"({len(PASS)} checks)"
    )
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
