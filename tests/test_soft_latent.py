"""Unit tests for Soft Token Latent Reasoning (SGLANG_SOFT_THINKING).

CPU-only. Covers the pure-logic module sglang.srt.layers.soft_latent:
protobuf wire roundtrip, prompt latent-block extraction/assembly, token
string<->id vocab mapping, candidate filtering vs a reference implementation,
and weighted-embedding math.

Run: .venv/bin/python tests/test_soft_latent.py
Requires the DSv4 tokenizer at /var/deepseek-v4-flash/0731 (or set
SOFT_LATENT_TEST_MODEL_DIR).
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../third_party/sglang/python"))

from sglang.srt.layers import soft_latent
from sglang.srt.layers.soft_latent import (
    LATENT_KEY_BASE,
    LATENT_KEY_SPACE,
    SENTINEL_ID,
    apply_latent_keys,
    assemble_prompt_ids,
    decode_position,
    encode_position,
    extract_latent_blocks,
    filter_candidates,
    latent_cache_key,
    resolve_ids,
    weighted_embed,
    wrap_payload,
)

MODEL_DIR = os.environ.get("SOFT_LATENT_TEST_MODEL_DIR", "/var/deepseek-v4-flash/0731")

PASS = []


def check(name, cond, detail=""):
    PASS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)


def test_protobuf_roundtrip():
    cands = [("Ġtherefore", 0.42), ("Ġhowever", 0.31), ("Ġso", 0.18), ("Ġthus", 0.09)]
    b64 = encode_position(cands)
    back = decode_position(b64)
    ok = len(back) == len(cands) and all(
        a[0] == b[0] and abs(a[1] - b[1]) < 1e-6 for a, b in zip(cands, back)
    )
    check("protobuf roundtrip", ok, f"n={len(back)}")

    # unicode + special-token strings survive
    cands2 = [("<think>", 0.5), ("Ġ你好", 0.3), ("</think>", 0.2)]
    back2 = decode_position(encode_position(cands2))
    check(
        "protobuf roundtrip unicode/special",
        [t for t, _ in back2] == [t for t, _ in cands2],
    )

    # empty candidate list
    check("protobuf empty", decode_position(encode_position([])) == [])


def test_extract():
    p1 = wrap_payload(encode_position([("Ġa", 0.7), ("Ġb", 0.3)]))
    p2 = wrap_payload(encode_position([("Ġc", 1.0)]))
    # genuine wire form: adjacent blocks, real text only before/after
    # (the tail after the last block begins with the real </think> token)
    text = f"pre {p1}{p2}</think> post"
    segments, latents = extract_latent_blocks(text)
    check(
        "extract segments",
        segments == ["pre ", "", "</think> post"] and len(latents) == 2,
        f"{segments!r}",
    )
    check(
        "extract candidates",
        latents[0][0][0] == "Ġa" and abs(latents[1][0][1] - 1.0) < 1e-6,
    )
    # no latent blocks
    s2, l2 = extract_latent_blocks("plain text")
    check("extract noop", s2 == ["plain text"] and l2 == [])
    # trailing latent
    s3, l3 = extract_latent_blocks(f"x {p1}")
    check("extract tail", s3[-1] == "" and len(l3) == 1, f"{s3!r}")


def test_display_newline():
    """wrap_payload appends display "  \\n"; extract strips display whitespace."""
    p1 = wrap_payload(encode_position([("Ġa", 0.7), ("Ġb", 0.3)]))
    check("wrap appends display whitespace", p1.endswith("<|/latent|>  \n"))
    # genuine form: adjacent blocks (pure-whitespace gap) + </think> after
    segments, latents = extract_latent_blocks(f"A {p1}{p1}</think>C")
    check(
        "extract strips display whitespace",
        segments == ["A ", "", "</think>C"] and len(latents) == 2,
        f"{segments!r}",
    )
    # legacy formats (no padding / "\n" / "\n\n") parse identically
    p1_no_pad = p1[:-3]
    for p_old in (p1_no_pad, p1_no_pad + "\n", p1_no_pad + "\n\n"):
        segments_old, latents_old = extract_latent_blocks(f"A {p_old}{p_old}")
        check(
            "extract legacy format compat",
            segments_old == ["A ", "", ""] and latents_old == latents,
            f"{segments_old!r}",
        )
    # any mix/count of pure gap whitespace is dropped entirely
    segments2, _ = extract_latent_blocks(
        f"A {p1_no_pad} \n  \n\n \n{p1_no_pad}"
    )
    check("extract strips any whitespace mix", segments2 == ["A ", "", ""])
    # </think> continuation: whitespace stripped, real text kept
    segments2b, _ = extract_latent_blocks(f"A {p1_no_pad}  \n</think>B")
    check(
        "extract strips whitespace before </think>",
        segments2b == ["A ", "</think>B"],
        f"{segments2b!r}",
    )
    # real interleaved text (single-candidate steps emit real tokens):
    # exactly the fixed padding prefix is stripped, the text kept as-is
    segments2d, _ = extract_latent_blocks(f"A {p1}hello {p1}</think>B")
    check(
        "extract keeps interleaved real text",
        segments2d == ["A ", "hello ", "</think>B"],
        f"{segments2d!r}",
    )
    # ... even when the real text itself starts with the padding characters
    segments2e, _ = extract_latent_blocks(f"A {p1}  \nhello")
    check(
        "extract strips exactly one padding prefix",
        segments2e == ["A ", "  \nhello"],
        f"{segments2e!r}",
    )
    # precise scope (legacy-looking or client-mangled): whitespace before
    # arbitrary text without the exact padding prefix is NOT stripped
    segments2c, _ = extract_latent_blocks(f"A {p1_no_pad}\nB")
    check(
        "extract keeps whitespace before other text",
        segments2c == ["A ", "\nB"],
        f"{segments2c!r}",
    )
    # trailing block: padding then end-of-text -> empty tail (tail_is_latent)
    segments3, _ = extract_latent_blocks(f"A {p1}")
    check("extract display-whitespace tail", segments3 == ["A ", ""])


def test_assemble(tok):
    # parity with plain encode when no latents
    text = "<think> hello world 你好 </think> answer"
    segments, latents = extract_latent_blocks(text)
    ids_plain = tok.encode(text)
    ids2, spans, tail = assemble_prompt_ids(tok, segments, latents)
    check("assemble noop parity", list(ids_plain) == list(ids2) and spans == [])

    # with latents: structure + positions + candidate resolution
    # (genuine wire form: blocks are adjacent, only display newlines between)
    p1 = wrap_payload(encode_position([("Ġtherefore", 0.5), ("Ġso", 0.5)]))
    text2 = f"<think> think {p1}{p1}"
    segments2, latents2 = extract_latent_blocks(text2)
    ids3, spans3, tail3 = assemble_prompt_ids(tok, segments2, latents2)
    expected_ids = sorted(
        [tok.convert_tokens_to_ids("Ġtherefore"), tok.convert_tokens_to_ids("Ġso")]
    )
    got_ids = sorted(spans3[0][1])
    check("assemble candidate id resolution", got_ids == expected_ids)
    # placeholder = dominant candidate id (drives hash-MoE routing at prefill)
    best_id = tok.convert_tokens_to_ids("Ġtherefore")  # weight 0.5 == 0.5, first wins
    placeholder_positions = [i for i, t in enumerate(ids3) if t == best_id]
    check(
        "assemble placeholder positions",
        [s[0] for s in spans3] == placeholder_positions
        and len(spans3) == 2
        and all(ids3[s[0]] == best_id for s in spans3),
        f"{[s[0] for s in spans3]} vs {placeholder_positions}",
    )
    check("assemble tail_is_latent", tail3 is True)
    # text roundtrip parity of the non-latent region
    pre = tok.encode("<think> think", add_special_tokens=False)
    check(
        "assemble prefix parity",
        list(ids3[: len(pre)]) == list(pre),
    )
    # old wire format (no display padding) assembles identically
    p1_old = p1[:-3]  # strip the display "  \n"
    text2_old = f"<think> think {p1_old}{p1_old}"
    segments2o, latents2o = extract_latent_blocks(text2_old)
    ids3o, spans3o, tail3o = assemble_prompt_ids(tok, segments2o, latents2o)
    check(
        "assemble old/new wire format parity",
        list(ids3o) == list(ids3) and spans3o == spans3 and tail3o == tail3,
    )


def test_cache_keys(tok):
    cands = [("Ġtherefore", 0.42), ("Ġhowever", 0.31), ("Ġso", 0.18), ("Ġthus", 0.09)]

    # determinism
    check(
        "cache key deterministic",
        latent_cache_key(cands) == latent_cache_key(list(cands)),
    )
    # discrimination: weights, order, strings
    k0 = latent_cache_key(cands)
    cands_w = [(s, w + 0.01) for s, w in cands]
    cands_rev = list(reversed(cands))
    cands_s = [(s + "x", w) for s, w in cands]
    check(
        "cache key discriminates",
        len({k0, latent_cache_key(cands_w), latent_cache_key(cands_rev), latent_cache_key(cands_s)}) == 4,
    )
    # range: far above vocab ids, fits int64
    vocab = tok.vocab_size
    check(
        "cache key range",
        all(
            LATENT_KEY_BASE <= latent_cache_key(c) < LATENT_KEY_BASE + LATENT_KEY_SPACE < 2**63
            for c in (cands, cands_w, cands_rev)
        )
        and LATENT_KEY_BASE > vocab + 8,
        f"base=2**{LATENT_KEY_BASE.bit_length() - 1} vocab={vocab}",
    )
    # float32 wire roundtrip preserves the key (proto float field is float32)
    rt = decode_position(encode_position(cands))
    check("cache key proto roundtrip stable", latent_cache_key(rt) == k0)
    # float64 noise below float32 resolution must not change the key
    import struct

    cands_f32 = [
        (s, struct.unpack("<f", struct.pack("<f", w))[0]) for s, w in cands
    ]
    check("cache key float32 quantization", latent_cache_key(cands_f32) == k0)

    # assemble spans carry the key; same dominant candidate but different
    # tail => different keys (the radix collision this feature fixes)
    p1 = wrap_payload(encode_position([("Ġtherefore", 0.5), ("Ġso", 0.5)]))
    p2 = wrap_payload(encode_position([("Ġtherefore", 0.9), ("Ġso", 0.1)]))
    segments, latents = extract_latent_blocks(f"<think> {p1} x")
    _, spans_a, _ = assemble_prompt_ids(tok, segments, latents)
    segments, latents = extract_latent_blocks(f"<think> {p2} x")
    _, spans_b, _ = assemble_prompt_ids(tok, segments, latents)
    check("assemble span count", len(spans_a) == 1 and len(spans_b) == 1)
    check("assemble span arity 4", len(spans_a[0]) == 4)
    same_dominant = spans_a[0][1][0] == spans_b[0][1][0]  # same argmax
    check(
        "assemble keys distinguish same-argmax distributions",
        same_dominant and spans_a[0][3] != spans_b[0][3],
    )
    check(
        "assemble span key matches direct hash",
        spans_a[0][3]
        == latent_cache_key([("Ġtherefore", 0.5), ("Ġso", 0.5)]),
    )

    # cross-path consistency: generation-side payload (strings, weights)
    # -> proto -> base64 -> restore-side decode -> identical key
    wire = wrap_payload(encode_position(cands))
    segments, latents = extract_latent_blocks(f"pre {wire} post")
    _, spans, _ = assemble_prompt_ids(tok, segments, latents)
    check("cache key gen/restore consistent", spans[0][3] == k0)

    # apply_latent_keys: substitution, range guard, noop
    from array import array

    ids = array("q", [10, 11, 12, 13])
    out = apply_latent_keys(ids, {1: LATENT_KEY_BASE + 7, 99: 5})
    check(
        "apply_latent_keys substitution",
        list(out) == [10, LATENT_KEY_BASE + 7, 12, 13] and list(ids) == [10, 11, 12, 13],
    )
    same = apply_latent_keys(ids, {})
    check("apply_latent_keys noop", same is ids)


def test_native_hash_u64():
    """HiCache 页哈希的 u64 支持（hash_binding.cpp 向后兼容标记编码）。"""
    import hashlib
    import struct
    from array import array

    from sglang.srt.mem_cache.cpp_utils.native_hash import get_native_hash

    def ref_small(ids):
        return hashlib.sha256(
            b"".join(struct.pack("<I", i) for i in ids)
        ).hexdigest()

    def ref_mixed(ids):
        buf = b""
        for v in ids:
            if v > 0xFFFFFFFF:
                buf += struct.pack("<III", 0xFFFFFFFF, v & 0xFFFFFFFF, v >> 32)
            else:
                buf += struct.pack("<I", v)
        return hashlib.sha256(buf).hexdigest()

    small = [0, 1, 128799, 128821, 128822, 129279]
    big = LATENT_KEY_BASE | 12345

    # 向后兼容：全小 id 的 u64 输入与历史 u32 布局逐字节一致
    h32 = get_native_hash(array("I", small), None)
    h64 = get_native_hash(array("q", small), None)
    check(
        "native hash u64 backward-compatible",
        h32 == h64 == ref_small(small),
    )
    # 宽 id：标记编码，可区分，不抛异常
    mixed = small[:3] + [big] + small[3:]
    hm = get_native_hash(array("q", mixed), None)
    check("native hash wide-id marker encoding", hm == ref_mixed(mixed))
    hm2 = get_native_hash(array("q", small[:3] + [big + 1] + small[3:]), None)
    check("native hash discriminates wide ids", hm != hm2)
    # paged（链式）路径
    pages = get_native_hash(array("q", mixed * 100), None, page_size=256)
    check(
        "native hash paged with wide ids",
        isinstance(pages, list) and len(pages) == 3,
        f"pages={len(pages)}",
    )
    # plain-list 兜底（OverflowError -> array('q')）与 array('q') 一致
    check("native hash list fallback", get_native_hash(mixed, None) == hm)


def test_token_string_roundtrip(tok):
    # Vocab-level mapping (never decode->encode) must be exact for the ids we
    # can encounter as candidates.
    sample_ids = [0, 1, 100, 50000, 128799, 128821, 128822, 129279]
    sample_ids += list(range(1000, 2000, 37))
    bad = []
    for i in sample_ids:
        s = tok.convert_ids_to_tokens(i)
        back = tok.convert_tokens_to_ids(s)
        if back != i:
            bad.append((i, s, back))
    check("token string roundtrip", not bad, f"bad={bad[:5]}")

    ids = resolve_ids(tok)
    check(
        "resolve_ids",
        (ids.think_start_id, ids.think_end_id, ids.sentinel_id) == (128821, 128822, SENTINEL_ID),
        f"think=({ids.think_start_id},{ids.think_end_id})",
    )


def _reference_filter(probs_row, top_k, top_p, min_p, cap):
    """Slow reference: returns (ids, renormalized_probs) after filtering."""
    vals, idx = torch.sort(probs_row, descending=True)
    vals, idx = vals[:cap], idx[:cap]
    keep = []
    cum = 0.0
    for r, (v, i) in enumerate(zip(vals.tolist(), idx.tolist())):
        if top_k > 0 and r >= top_k:
            break
        if cum > top_p:
            break
        if min_p > 0 and v < vals[0].item() * min_p:
            break
        keep.append((i, v))
        cum += v
    total = sum(v for _, v in keep)
    return [i for i, _ in keep], [v / total for _, v in keep]


def test_filter_candidates():
    torch.manual_seed(0)
    vocab = 5000
    probs = torch.softmax(torch.randn(4, vocab), dim=-1)
    top_ks = torch.tensor([3, -1, 10, 1], dtype=torch.int32)
    top_ps = torch.tensor([1.0, 0.5, 0.9, 1.0])
    min_ps = torch.tensor([0.0, 0.05, 0.0, 0.0])
    cap = 256
    ids, kept, keep_mask = filter_candidates(probs, top_ks, top_ps, min_ps, cap)
    ok = True
    for r in range(4):
        ref_ids, ref_probs = _reference_filter(
            probs[r], int(top_ks[r]), float(top_ps[r]), float(min_ps[r]), cap
        )
        got_ids = ids[r][keep_mask[r]].tolist()
        got_probs = kept[r][keep_mask[r]].tolist()
        if got_ids != ref_ids:
            ok = False
            print(f"  row {r}: id mismatch {got_ids[:6]} vs {ref_ids[:6]}")
        if any(abs(a - b) > 1e-5 for a, b in zip(got_probs, ref_probs)) or abs(
            sum(got_probs) - 1.0
        ) > 1e-4:
            ok = False
            print(f"  row {r}: prob mismatch sum={sum(got_probs)}")
    check("filter_candidates vs reference", ok)

    # top-1 candidate always survives, even with extreme top_p
    ids2, kept2, keep2 = filter_candidates(
        probs, top_ks, torch.zeros(4), None, cap
    )
    check("filter top_p=0 keeps argmax", bool(keep2[:, 0].all()))
    check(
        "filter argmax==top1",
        bool((ids2[:, 0] == probs.argmax(dim=-1)).all()),
    )


def test_filter_candidates_min_prob():
    """Absolute probability floor: drop < floor; all-below rows fall back
    to the pipeline-filtered top-cap set."""
    # row 0: peaked (0.70 / 0.20 / 0.06 / 0.04) -> floor 0.05 keeps 3
    # row 1: flat-ish (0.30/0.28/0.22/0.20), all >= 0.05 -> all 4 kept
    # row 2: very flat (0.04*25) -> none >= 0.05 -> fallback keeps top cap=4
    # row 3: one-hot -> single candidate at 1.0 survives
    probs = torch.zeros(4, 100)
    probs[0, :4] = torch.tensor([0.70, 0.20, 0.06, 0.04])
    probs[1, :4] = torch.tensor([0.30, 0.28, 0.22, 0.20])
    probs[2, :25] = 0.04
    probs[3, 0] = 1.0
    top_ks = torch.full((4,), -1, dtype=torch.int32)
    top_ps = torch.ones(4)
    ids, kept, keep_mask = filter_candidates(probs, top_ks, top_ps, None, 4, 0.05)

    counts = keep_mask.sum(dim=-1).tolist()
    check("min_prob kept counts", counts == [3, 4, 4, 1], f"{counts}")
    # row 0: the 0.04 candidate dropped, rest renormalized to sum 1
    p0 = kept[0][keep_mask[0]].tolist()
    check(
        "min_prob row0 renormalized",
        abs(sum(p0) - 1.0) < 1e-5
        and abs(p0[0] - 0.70 / 0.96) < 1e-4
        and abs(p0[2] - 0.06 / 0.96) < 1e-4,
        f"{p0}",
    )
    # row 2 fallback: top-4 by probability (ties -> first four), renorm 0.25
    p2 = kept[2][keep_mask[2]].tolist()
    check(
        "min_prob fallback renormalized",
        len(p2) == 4 and all(abs(p - 0.25) < 1e-5 for p in p2),
        f"{p2}",
    )
    # min_prob=0 disables the floor entirely
    _, _, keep_off = filter_candidates(probs, top_ks, top_ps, None, 4, 0.0)
    check(
        "min_prob=0 disables",
        keep_off.sum(dim=-1).tolist() == [4, 4, 4, 4],
    )


def test_weighted_embed():
    torch.manual_seed(1)
    weight = torch.randn(100, 16, dtype=torch.bfloat16)
    cand_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    cand_probs = torch.tensor([[0.2, 0.5, 0.3], [0.9, 0.05, 0.05]])
    out = weighted_embed(weight, cand_ids, cand_probs)
    ref0 = (
        0.2 * weight[1].float() + 0.5 * weight[2].float() + 0.3 * weight[3].float()
    )
    err = (out[0].float() - ref0).abs().max().item()
    check("weighted_embed math", err < 1e-2, f"max_err={err}")
    check("weighted_embed dtype", out.dtype == torch.bfloat16)
    # single candidate = the embedding itself (greedy degenerate case)
    out1 = weighted_embed(weight, torch.tensor([[7]]), torch.tensor([[1.0]]))
    check(
        "weighted_embed single candidate",
        torch.allclose(out1[0].float(), weight[7].float(), atol=1e-2),
    )


def test_assemble_with_image_expansion(tok):
    """Simulate the mm expansion remap: assemble latent spans around a fake
    image placeholder, expand it, shift span positions, verify alignment."""
    from sglang.srt.layers.soft_latent import shift_positions_for_expansions

    # SENTINEL_ID is a special token that never appears in the text below,
    # and special-token strings re-encode to their id.
    fake_image_id = SENTINEL_ID
    p1 = wrap_payload(encode_position([("Ġtherefore", 0.6), ("Ġso", 0.4)]))
    p2 = wrap_payload(encode_position([("Ġyes", 0.9), ("Ġno", 0.1)]))
    # image placeholder sits BETWEEN the two latent blocks
    text = f"<think> {p1} mid </think> <img> <think> {p2}"
    text = text.replace("<img>", tok.decode([fake_image_id]))
    segments, latents = extract_latent_blocks(text)
    ids, spans, _tail = assemble_prompt_ids(tok, segments, latents)
    image_pre_pos = list(ids).index(fake_image_id)
    # both latent spans must bracket the placeholder
    assert spans[0][0] < image_pre_pos < spans[1][0], (
        spans,
        image_pre_pos,
    )

    # expand: placeholder -> 284 sentinel tokens (mimics dsv4 vision block)
    block_len = 284
    expanded = (
        list(ids[:image_pre_pos])
        + [10**9] * block_len
        + list(ids[image_pre_pos + 1 :])
    )
    shifted = shift_positions_for_expansions(
        [s[0] for s in spans], [(image_pre_pos, block_len)]
    )
    check(
        "expansion remap positions",
        shifted == [spans[0][0], spans[1][0] + block_len - 1],
        f"{[s[0] for s in spans]} -> {shifted} (image at {image_pre_pos})",
    )
    # the dominant-candidate placeholder id must sit at the shifted position
    for span, new_pos in zip(spans, shifted):
        best = max(range(len(span[2])), key=lambda j: span[2][j])
        check(
            f"expanded placeholder id at {new_pos}",
            expanded[new_pos] == span[1][best],
        )


def test_shift_positions_for_expansions():
    from sglang.srt.layers.soft_latent import shift_positions_for_expansions

    shift = shift_positions_for_expansions
    # no expansions -> identity
    check("shift noop", shift([3, 10], []) == [3, 10])
    # one expansion: token at 5 -> 284 tokens; positions <= 5 stay, after shift +283
    check(
        "shift single",
        shift([0, 5, 6, 100], [(5, 284)]) == [0, 5, 289, 383],
    )
    # two expansions: cumulative shifts (3->10 tokens: +9; 8->284 tokens: +283)
    check(
        "shift cumulative",
        shift([4, 7, 20], [(3, 10), (8, 284)]) == [13, 16, 312],
    )
    # image AFTER the latent span: latent positions unchanged
    check(
        "shift after-only",
        shift([10, 20], [(30, 284)]) == [10, 20],
    )


def main():
    test_protobuf_roundtrip()
    test_extract()
    test_display_newline()
    test_filter_candidates()
    test_filter_candidates_min_prob()
    test_weighted_embed()
    test_shift_positions_for_expansions()

    tok = get_tokenizer()
    test_assemble(tok)
    test_assemble_with_image_expansion(tok)
    test_token_string_roundtrip(tok)
    test_cache_keys(tok)
    test_native_hash_u64()

    n_fail = sum(1 for _, ok in PASS if not ok)
    print(f"\n{'ALL PASS' if n_fail == 0 else f'{n_fail} FAILED'} ({len(PASS)} checks)")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
