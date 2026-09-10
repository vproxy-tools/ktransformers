"""Integration test: latent-aware radix prefix cache (SGLANG_SOFT_THINKING).

Runs against a live experimental server started with:

    SOFT=1 ./run_dspark.sh          # port 30001, radix cache ENABLED

Proves that:
1. a generated latent thinking trace is inserted into the radix tree keyed
   by content-derived synthetic ids;
2. replaying a conversation with identical <|latent|> blocks prefix-matches
   THROUGH the latent span — with page_size=256 the first latent position is
   at ~17, so any hit >= 256 tokens is only possible if hundreds of latent
   positions matched by content;
3. a tampered latent block (one weight perturbed) breaks the match
   immediately (cached == 0), i.e. the keys discriminate distributions;
4. cache reuse does not change greedy output (flush_cache + replay gives a
   byte-identical completion).

Usage: .venv/bin/python tests/test_soft_latent_cache.py [--port 30001]
"""

import argparse
import json
import os
import sys

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../third_party/sglang/python"))

from sglang.srt.layers.soft_latent import (  # noqa: E402
    LATENT_PREFIX,
    LATENT_SUFFIX,
    decode_position,
    encode_position,
)

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
    r.raise_for_status()
    return r.json()


def flush_cache(base):
    """Flush L1/L2 (tree/device/host). L3 storage on disk is intentionally
    left untouched by the server. Retries: the server refuses to flush while
    any request is still in flight, and decode tail/backup writes race with
    the caller observing the response."""
    import time

    for _ in range(30):
        r = requests.post(f"{base}/flush_cache", timeout=60)
        if r.status_code == 200:
            return
        time.sleep(1)
    r.raise_for_status()


def hicache_enabled(base):
    r = requests.get(f"{base}/get_server_info", timeout=30)
    r.raise_for_status()
    return bool(r.json().get("enable_hierarchical_cache"))


def cached_tokens(resp):
    return (
        (resp.get("usage") or {}).get("prompt_tokens_details") or {}
    ).get("cached_tokens", 0)


def tamper_first_latent(reasoning: str) -> str:
    """Perturb one weight in the first <|latent|> block (same argmax)."""
    start = reasoning.index(LATENT_PREFIX) + len(LATENT_PREFIX)
    end = reasoning.index(LATENT_SUFFIX)
    b64 = reasoning[start:end]
    cands = decode_position(b64)
    # Greedy steps may carry a single candidate; perturb whichever weight
    # exists — any float32-bit change must change the cache key.
    i = 1 if len(cands) > 1 else 0
    cands[i] = (cands[i][0], cands[i][1] * 0.5 + 0.01)
    new_block = LATENT_PREFIX + encode_position(cands) + LATENT_SUFFIX
    return reasoning[: start - len(LATENT_PREFIX)] + new_block + reasoning[end + len(LATENT_SUFFIX):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30001)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    question = "9.11 和 9.9 哪个大？"
    # Padding makes the replayed prompt long enough that page alignment
    # (page_size=256) and the SWA re-prefill tail (128) do not cap the
    # match below the end of the latent span.
    filler = "（填充文字，与问题无关，请忽略）" * 60
    followup = f"{filler}用一句话复述你的最终答案。"
    # Negative controls must have NO self-history in L3 (which persists across
    # runs): a run-unique nonce in the followup guarantees that any cache hit
    # can only come from cross-matching A/B's sequence — i.e. a real key
    # collision — so cached==0 is a sharp discrimination check.
    import time as _time

    nonce = int(_time.time() * 1000)
    followup_ctrl = f"{filler}用一句话复述你的最终答案。（随机标记{nonce}）"

    # --- Step 1: fresh generation; must produce latent blocks -------------
    # temperature 0.8: since Part A, greedy (temp=0) positions are all
    # single-candidate and emit as real tokens — latent blocks only appear
    # at multi-candidate positions.
    flush_cache(base)
    a = chat(base, [{"role": "user", "content": question}], max_tokens=4000,
             temperature=0.8)
    msg_a = a["choices"][0]["message"]
    reasoning = msg_a.get("reasoning_content") or ""
    content = msg_a.get("content") or ""
    n_latent = reasoning.count(LATENT_PREFIX)
    check(
        "step1 latent blocks emitted",
        n_latent > 0,
        f"latent_blocks={n_latent} reasoning_chars={len(reasoning)}",
    )
    if n_latent == 0:
        print("server is not in SGLANG_SOFT_THINKING mode; aborting")
        sys.exit(1)
    reasoning_tokens = (a.get("usage") or {}).get("reasoning_tokens", 0)
    check(
        "step1 reasoning_tokens covers latent + real thinking tokens",
        reasoning_tokens >= n_latent + 1,
        f"{reasoning_tokens} vs {n_latent}+1",
    )

    # --- Step 2: replay the same conversation + follow-up ------------------
    replay_msgs = [
        {"role": "user", "content": question},
        {"role": "assistant", "reasoning_content": reasoning, "content": content},
        {"role": "user", "content": followup},
    ]
    b1 = chat(base, replay_msgs, max_tokens=256)
    cached_b1 = cached_tokens(b1)
    prompt_b1 = (b1.get("usage") or {}).get("prompt_tokens", 0)
    check(
        "step2 replay prefix cache hit through latent span (>=256)",
        cached_b1 >= 256,
        f"cached={cached_b1}/{prompt_b1} latent_span=17..{17 + n_latent}",
    )

    # Control: same conversation but a *different* assistant reasoning (plain
    # text, no latent blocks) — the common prefix ends at the assistant turn.
    ctrl_msgs = [
        {"role": "user", "content": question},
        {"role": "assistant", "reasoning_content": "好的。", "content": content},
        {"role": "user", "content": followup_ctrl},
    ]
    c = chat(base, ctrl_msgs, max_tokens=256)
    cached_ctrl = cached_tokens(c)
    check(
        "step2 control (different thinking) does not hit",
        cached_ctrl == 0,
        f"cached={cached_ctrl}",
    )

    # Negative control: identical replay except one perturbed weight in the
    # first latent block — the match must break at the first latent position.
    tampered = tamper_first_latent(reasoning)
    tampered_msgs = [
        {"role": "user", "content": question},
        {"role": "assistant", "reasoning_content": tampered, "content": content},
        {"role": "user", "content": followup_ctrl},
    ]
    d = chat(base, tampered_msgs, max_tokens=256)
    cached_d = cached_tokens(d)
    check(
        "step2 tampered latent block does not hit",
        cached_d == 0,
        f"cached={cached_d}",
    )

    # --- Step 3: cache reuse must not change greedy output -----------------
    # Bitwise equality requires like-for-like KV provenance: step2's replay
    # (B1) reused A's *decode-mode* KV, while an uncached replay recomputes
    # the latent span with *prefill-mode* kernels — sub-1e-3 drift can flip a
    # near-tie greedy pick (same as any prefix cache, not latent-specific).
    # So the strict check compares two replays of the same latent history
    # whose cached copy traces to prefill-computed KV:
    #   flush -> replay(X) [inserts] -> replay(Y) [hit] -> replay(Y) again.
    # With HiCache, flush leaves L3 storage intact: the first replay(Y) after
    # flush is served from L3 (B1's write-through backup of the shared
    # assistant history) — a positive L3-restore proof through latent keys.
    hicache = hicache_enabled(base)
    followup_y = f"{filler}{filler}请只回答最终答案里较大的那个数。"
    replay_msgs_y = [
        {"role": "user", "content": question},
        {"role": "assistant", "reasoning_content": reasoning, "content": content},
        {"role": "user", "content": followup_y},
    ]
    flush_cache(base)
    b2 = chat(base, replay_msgs, max_tokens=256)  # replay X after flush
    if hicache:
        # B1's write-through backup persists in L3 and shares the assistant
        # history prefix, so replay X hits L3 through the latent span.
        check(
            "step3 replay X hits L3 storage (hicache)",
            cached_tokens(b2) >= 256,
            f"cached={cached_tokens(b2)}",
        )
    else:
        check(
            "step3 uncached replay X (baseline insert)",
            cached_tokens(b2) == 0,
            f"cached={cached_tokens(b2)}",
        )
    y1 = chat(base, replay_msgs_y, max_tokens=256)
    cached_y1 = cached_tokens(y1)
    check(
        "step3 replay Y hits replay X's KV (L2 or L3)",
        cached_y1 >= 256,
        f"cached={cached_y1} hicache={hicache}",
    )
    flush_cache(base)
    y2 = chat(base, replay_msgs_y, max_tokens=256)
    cached_y2 = cached_tokens(y2)
    if hicache:
        # L2 flushed again; L3 (Y1's own backup) must serve the whole match.
        check(
            "step3 L3 restore after L2 flush (hicache)",
            cached_y2 >= 256,
            f"cached={cached_y2}",
        )
    else:
        check("step3 flush clears cache", cached_y2 == 0, f"cached={cached_y2}")
    content_y1 = y1["choices"][0]["message"].get("content") or ""
    content_y2 = y2["choices"][0]["message"].get("content") or ""
    check(
        "step3 cached vs reloaded output identical (greedy)",
        content_y1 == content_y2 and len(content_y1) > 0,
        f"len={len(content_y1)}",
    )
    if content_y1 != content_y2:
        print("  cached  :", json.dumps(content_y1[:200], ensure_ascii=False))
        print("  reloaded:", json.dumps(content_y2[:200], ensure_ascii=False))

    # Informational: generation-KV replay vs fresh prefill replay may drift
    # numerically (decode vs prefill kernels); both must stay semantically
    # correct.
    content_b1 = b1["choices"][0]["message"].get("content") or ""
    content_b2 = b2["choices"][0]["message"].get("content") or ""
    print(
        f"[INFO] generation-KV replay vs fresh replay: "
        f"identical={content_b1 == content_b2}; "
        f"b1_correct={'9.9' in content_b1} b2_correct={'9.9' in content_b2}"
    )

    n_fail = sum(1 for _, ok in PASS if not ok)
    print(f"\n{'ALL PASS' if n_fail == 0 else f'{n_fail} FAILED'} ({len(PASS)} checks)")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
