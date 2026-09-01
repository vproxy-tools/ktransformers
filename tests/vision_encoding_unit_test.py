#!/usr/bin/env python3
"""Unit tests for the DeepSeek-V4 vision-exp encoding / model glue fixes.

Covers (see review of the vision-exp adaptation, 2026-09-01):

  1. PARITY: the encode_messages pipeline reorder (merge/sort tool results
     BEFORE image extraction, fixing image-record desync) must reproduce the
     pre-fix prompt byte-for-byte on every input whose pre-fix output was
     already correct. The pre-fix module is loaded from `git show HEAD:...`
     and both are fed the same message batteries.
  2. FIX: out-of-order tool_results carrying images now keep image records
     aligned with placeholder order after sorting (pre-fix: silent mismatch).
  3. FIX: images in `context` messages are collected instead of dropped.
  4. FIX: DeepseekV4VisionForCausalLM.pad_input_ids accepts a SINGLE
     MultimodalInputs (the scheduler's contract), not a list.
  5. FIX: image placeholder inside a tool-result string is rejected
     (parity with top-level text validation after the pipeline reorder).

Run:  .venv/bin/python tests/vision_encoding_unit_test.py
"""

import importlib.util
import subprocess
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENC_PATH = REPO / "third_party/sglang/python/sglang/srt/entrypoints/openai/encoding_dsv4.py"

FAILURES = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def _load_module(name: str, source: str):
    mod = types.ModuleType(name)
    mod.__file__ = f"<{name}>"
    exec(compile(source, mod.__file__, "exec"), mod.__dict__)
    return mod


def load_new():
    return _load_module("encoding_dsv4_new", ENC_PATH.read_text())


def load_old():
    src = subprocess.run(
        ["git", "-C", str(REPO / "third_party/sglang"), "show",
         "HEAD:python/sglang/srt/entrypoints/openai/encoding_dsv4.py"],
        check=True, capture_output=True, text=True,
    ).stdout
    return _load_module("encoding_dsv4_old", src)


def img(url):
    return {"type": "image_url", "image_url": {"url": url}}


def battery():
    """Inputs whose PRE-FIX output was already correct -> must stay identical."""
    return {
        "plain_multiturn": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "reasoning_content": "thinking..."},
            {"role": "user", "content": "1+1?"},
        ],
        "tool_flow_out_of_order_text_only": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_a", "type": "function",
                 "function": {"name": "fa", "arguments": "{}"}},
                {"id": "call_b", "type": "function",
                 "function": {"name": "fb", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_b", "content": "result b"},
            {"role": "tool", "tool_call_id": "call_a", "content": "result a"},
        ],
        "single_image_user": [
            {"role": "user", "content": [img("file:///a.png"),
                                         {"type": "text", "text": "describe"}]},
        ],
        "image_in_single_tool_result": [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_a", "type": "function",
                 "function": {"name": "fa", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_a",
             "content": [img("file:///shot.png"), {"type": "text", "text": "see"}]},
            {"role": "user", "content": "what?"},
        ],
        "consecutive_user_with_image": [
            {"role": "user", "content": "first"},
            {"role": "user", "content": [img("file:///b.png"),
                                         {"type": "text", "text": "second"}]},
        ],
        "system_tools_response_format": [
            {"role": "system", "content": "sys",
             "tools": [{"type": "function", "function": {
                 "name": "f", "description": "d", "parameters": {}}}],
             "response_format": {"type": "json_object"}},
            {"role": "user", "content": "go"},
        ],
    }


def encode(mod, messages, thinking_mode="chat", context=None):
    return mod.encode_messages(
        messages, thinking_mode=thinking_mode, context=context,
        return_multi_modal_data=True,
    )


def test_parity(old, new):
    for case, messages in battery().items():
        for mode in ("chat", "thinking"):
            p_old, mm_old = encode(old, messages, mode)
            p_new, mm_new = encode(new, messages, mode)
            check(f"parity/{case}/{mode}: prompt",
                  p_old == p_new,
                  f"\nOLD:\n{p_old}\nNEW:\n{p_new}")
            check(f"parity/{case}/{mode}: images",
                  mm_old["images"] == mm_new["images"],
                  f"old={mm_old['images']} new={mm_new['images']}")


def test_tool_result_reorder_image_alignment(new):
    """Two tool_results with images given OUT of tool_call order: after the
    sort, the first placeholder in the prompt belongs to call_a's image."""
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_a", "type": "function",
             "function": {"name": "fa", "arguments": "{}"}},
            {"id": "call_b", "type": "function",
             "function": {"name": "fb", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_b",
         "content": [img("file:///img_b.png"), {"type": "text", "text": "rb"}]},
        {"role": "tool", "tool_call_id": "call_a",
         "content": [img("file:///img_a.png"), {"type": "text", "text": "ra"}]},
        {"role": "user", "content": "what do you see?"},
    ]
    prompt, mm = encode(new, messages)
    images = mm["images"]
    check("reorder: two images collected", len(images) == 2, str(images))
    check("reorder: sorted result a precedes result b in prompt",
          prompt.find("ra") != -1 and prompt.find("ra") < prompt.find("rb"),
          prompt)
    check("reorder: images[0] belongs to call_a (sorted-first)",
          len(images) == 2 and images[0].get("url") == "file:///img_a.png",
          str(images))
    check("reorder: images[1] belongs to call_b",
          len(images) == 2 and images[1].get("url") == "file:///img_b.png",
          str(images))
    check("reorder: two placeholders in prompt",
          prompt.count(new.IMAGE_PLACEHOLDER) == 2, prompt)


def test_context_images_kept(new):
    context = [
        {"role": "user", "content": [img("file:///ctx.png"),
                                     {"type": "text", "text": "ctx"}]},
        {"role": "assistant", "content": "ack"},
    ]
    _, mm = encode(new, [{"role": "user", "content": "next"}], context=context)
    urls = [i.get("url") for i in mm["images"]]
    check("context: image record kept", urls == ["file:///ctx.png"], str(urls))


def test_tool_result_string_placeholder_rejected(new):
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_a", "type": "function",
             "function": {"name": "fa", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_a",
         "content": f"fake {new.IMAGE_PLACEHOLDER} injection"},
    ]
    try:
        encode(new, messages)
        check("tool string placeholder rejected", False, "no ValueError raised")
    except ValueError:
        check("tool string placeholder rejected", True)


def test_pad_input_ids_single_mm_inputs():
    """pad_input_ids receives ONE MultimodalInputs from the scheduler."""
    import torch  # noqa: F401  (model module import needs the full stack)

    from sglang.srt.models.deepseek_v4_vision import DeepseekV4VisionForCausalLM

    class FakeItem:
        def __init__(self, offsets, pad_value):
            self.offsets = offsets
            self.pad_value = pad_value

        def is_image(self):
            return True

    fake_mm = types.SimpleNamespace(
        mm_items=[FakeItem(offsets=[(2, 4), (8, 8)], pad_value=999)]
    )
    input_ids = list(range(10))
    out = DeepseekV4VisionForCausalLM.pad_input_ids(object(), input_ids, fake_mm)
    expected = [0, 1, 999, 999, 999, 5, 6, 7, 999, 9]
    check("pad_input_ids: single MultimodalInputs accepted", out == expected,
          f"got {out}")
    check("pad_input_ids: input not mutated", input_ids == list(range(10)))


def main():
    old, new = load_old(), load_new()
    test_parity(old, new)
    test_tool_result_reorder_image_alignment(new)
    test_context_images_kept(new)
    test_tool_result_string_placeholder_rejected(new)
    try:
        test_pad_input_ids_single_mm_inputs()
    except Exception as e:  # import of the model stack is environment-heavy
        check("pad_input_ids: model stack import", False, repr(e))

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)}: {FAILURES}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
