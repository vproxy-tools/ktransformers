"""Integration test: latent history replay with SGLANG_SOFT_THINKING OFF.

Runs against a live server WITHOUT soft-latent generation (e.g. production
ds4f, port 30000, since the 2026-09-10 revert). Restoring/parsing historical
<|latent|> blocks must work regardless of the switch: a history recorded on
a latent-enabled instance must replay faithfully (extraction, soft-embedding
rebuild, content-derived cache keys) — only the GENERATION of new latent
blocks is gated by the env.

Proves that:
1. a conversation whose assistant history carries synthetic <|latent|>
   blocks is ACCEPTED (200), the server restores them
   ("[soft-latent] restored N latent position(s)" in journald), and the
   reply contains NO new latent blocks (generation stays off);
2. an identical replay prefix-matches THROUGH the latent span (with
   page_size=256 the span sits well past page 0, so a cached-token hit
   >= 2 pages is only possible via content-derived latent keys);
3. a tampered latent weight breaks the match at the first latent position
   (cached-token drops to 0 pages).

Usage: .venv/bin/python tests/test_soft_latent_restore_off.py [--port 30000]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../third_party/sglang/python"))

from sglang.srt.layers.soft_latent import (  # noqa: E402
    LATENT_PREFIX,
    encode_position,
    wrap_payload,
)

MODEL_DIR = os.environ.get("SOFT_LATENT_TEST_MODEL_DIR", "/var/deepseek-v4-flash/0731")

PASS = []


def check(name, cond, detail=""):
    PASS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


def journal_restored(since: str):
    out = subprocess.run(
        ["journalctl", "-u", "ds4f", "--since", since, "--output=cat",
         "--no-pager"],
        capture_output=True, text=True, timeout=60,
    ).stdout
    m = re.findall(r"\[soft-latent\] restored (\d+) latent position\(s\)", out)
    return [int(x) for x in m]


def journal_cached_tokens(since: str):
    out = subprocess.run(
        ["journalctl", "-u", "ds4f", "--since", since, "--output=cat",
         "--no-pager"],
        capture_output=True, text=True, timeout=60,
    ).stdout
    vals = []
    for line in out.splitlines():
        m = re.search(r"Prefill batch.*#cached-token: (\d+)", line)
        if m:
            vals.append(int(m.group(1)))
    return vals


def now_stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def build_history(n_blocks):
    """Synthetic reasoning_content: n_blocks latent positions built from real
    vocab token strings (weights varied per position so cache keys differ)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    vocab = tok.get_vocab()
    # Plain-text tokens only (skip special/marker tokens), stable order.
    strs = sorted(
        s for s in vocab
        if 0 < len(s) <= 4 and "<" not in s and "�" not in s
    )
    assert len(strs) >= 8, "not enough plain vocab tokens for the test"
    parts = []
    for i in range(n_blocks):
        a = strs[i % len(strs)]
        b = strs[(i + 3) % len(strs)]
        w = 0.55 + (i % 17) * 0.02
        cands = [(a, w), (b, round(1.0 - w, 4))]
        parts.append(wrap_payload(encode_position(cands)))
    return "".join(parts)


def chat(base, messages, max_tokens):
    r = requests.post(
        f"{base}/v1/chat/completions",
        json={
            "model": "dsv4",
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
        },
        timeout=1800,
    )
    if r.status_code != 200:
        print(f"[ERROR] {r.status_code}: {r.text[:2000]}")
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--blocks", type=int, default=500,
                    help="latent positions in the synthetic history "
                         "(>= ~500 keeps the prompt past 2 radix pages)")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    reasoning = build_history(args.blocks)
    msgs = [
        {"role": "user", "content": "请开始分析。"},
        {
            "role": "assistant",
            "reasoning_content": reasoning,
            "content": "初步分析完成。",
        },
        {"role": "user", "content": "基于以上思考，1+1等于几？只回答数字。"},
    ]

    # --- Step 1: history with latent blocks is accepted & restored ---------
    t0 = now_stamp()
    time.sleep(1)
    a = chat(base, msgs, max_tokens=48)
    msg_a = a["choices"][0]["message"]
    rc_a = msg_a.get("reasoning_content") or ""
    c_a = msg_a.get("content") or ""
    restored = journal_restored(t0)
    check("step1 accepted (200)", True)
    check(
        "step1 server restored the history span",
        bool(restored) and max(restored) == args.blocks,
        f"restored={restored} (expect [{args.blocks}])",
    )
    check(
        "step1 reply has NO new latent blocks (generation off)",
        LATENT_PREFIX not in rc_a and LATENT_PREFIX not in c_a,
        f"rc_len={len(rc_a)} content={c_a[:40]!r}",
    )

    # --- Step 2: identical replay hits the cache through the latent span ---
    t1 = now_stamp()
    time.sleep(1)
    chat(base, msgs, max_tokens=48)
    cached = journal_cached_tokens(t1)
    deep = [v for v in cached if v >= 512]
    check(
        "step2 identical replay matches through the latent span",
        bool(deep),
        f"cached_tokens={cached} (need one >= 512 = 2 pages past the span)",
    )

    # --- Step 3: tampered weight breaks the match at the first block -------
    # Perturb one candidate weight of the FIRST latent position; the
    # content-derived key changes, so prefix matching diverges before the
    # radix page that contains the first latent position.
    from sglang.srt.layers.soft_latent import LATENT_SUFFIX, decode_position

    first_end = reasoning.index(LATENT_SUFFIX)
    payload = reasoning[len(LATENT_PREFIX):first_end]
    rest = reasoning[first_end + len(LATENT_SUFFIX):]  # "  \n" + other blocks
    cands = decode_position(payload)
    cands[0] = (cands[0][0], round(cands[0][1] * 0.5, 4))
    tampered = LATENT_PREFIX + encode_position(cands) + LATENT_SUFFIX + rest
    msgs_t = list(msgs)
    msgs_t[1] = dict(msgs[1], reasoning_content=tampered)
    t2 = now_stamp()
    time.sleep(1)
    chat(base, msgs_t, max_tokens=48)
    cached_t = journal_cached_tokens(t2)
    check(
        "step3 tampered weight breaks the match at the first latent page",
        all(v < 256 for v in cached_t),
        f"cached_tokens={cached_t} (all must be < 1 page)",
    )

    n_pass = sum(1 for _, ok in PASS if ok)
    print(f"\n{n_pass}/{len(PASS)} PASS" + ("" if n_pass == len(PASS) else "  <-- FAIL"))
    sys.exit(0 if n_pass == len(PASS) else 1)


if __name__ == "__main__":
    main()
