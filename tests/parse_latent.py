"""解析 <|latent|> 块，输出人类可读的 latent 思维链。

输入：任何含 <|latent|>base64<|/latent|> 块的文本（如保存的
reasoning_content、会话历史、日志），来自文件参数或 stdin。
（wire 格式每个块后带 "  \n" 展示空白（markdown 硬换行）；解析时自动剥除。）

输出分两部分：
1. 通读文本：普通文本段原样保留，每个 latent 位置用概率最高的候选
   token 还原，直接读通整段语句；latent 还原的部分用〔〕标出，
   便于区分"真实 token"与"latent 主候选"。
2. 逐位置明细：每个 latent 位置的候选 token 及概率（百分比，
   按概率降序），主候选以 * 标记。

用法：
    .venv/bin/python tests/parse_latent.py response.txt
    .venv/bin/python tests/parse_latent.py response.txt --top 5
    jq -r '.choices[0].message.reasoning_content' r.json | \\
        .venv/bin/python tests/parse_latent.py
    .venv/bin/python tests/parse_latent.py r.txt --json   # 机器可读输出

token 字符串是 tokenizer 词表形式（如 "Ġtherefore"）；默认加载
/var/deepseek-v4-flash/0731 的 tokenizer 做 detokenize（可用
--tokenizer 或 SOFT_LATENT_TEST_MODEL_DIR 覆盖；加载失败时退化为
简单的 Ġ→空格替换）。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../third_party/sglang/python"))

from sglang.srt.layers import soft_latent  # noqa: E402
from sglang.srt.layers.soft_latent import (  # noqa: E402
    decode_position,
    extract_latent_blocks,
)

DEFAULT_TOKENIZER = os.environ.get(
    "SOFT_LATENT_TEST_MODEL_DIR", "/var/deepseek-v4-flash/0731"
)


def load_detokenizer(path):
    """返回 token_string -> 可读文本 的函数；tokenizer 不可用则退化。"""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        def detok(token_strs):
            ids = []
            for s in token_strs:
                tid = tok.convert_tokens_to_ids(s)
                if not isinstance(tid, int) or tid is None or tid < 0:
                    ids = None
                    break
                ids.append(tid)
            if ids is not None:
                return tok.decode(ids, skip_special_tokens=False)
            # 某个候选不是单个词表项：逐 token 退化拼接
            return "".join(s.replace("Ġ", " ").replace("Ċ", "\n")
                           for s in token_strs)

        # 预热，确认可用
        detok(["Ġtest"])
        return detok
    except Exception as e:  # noqa: BLE001
        print(f"[warn] tokenizer 加载失败（{e}），退化为简单替换", file=sys.stderr)
        return lambda ss: "".join(s.replace("Ġ", " ").replace("Ċ", "\n")
                                  for s in ss)


def render_fluent(segments, latents, detok):
    """普通文本 + 每位置主候选，拼成可通读的文本（latent 部分加〔〕）。"""
    out = []
    for i, seg in enumerate(segments):
        out.append(seg)
        if i < len(latents):
            cands = latents[i]
            best = max(cands, key=lambda c: c[1])[0] if cands else "?"
            out.append("〔" + detok([best]) + "〕")
    return "".join(out)


def format_detail(idx, cands, detok, top):
    lines = []
    cands_sorted = sorted(cands, key=lambda c: -c[1])
    for rank, (token_str, w) in enumerate(cands_sorted[:top]):
        mark = "*" if rank == 0 else " "
        shown = detok([token_str])
        shown = shown.replace("\n", "\\n")
        lines.append(f"  {mark} {shown!r:24} {w * 100:6.2f}%")
    hidden = len(cands_sorted) - top
    if hidden > 0:
        rest_w = sum(w for _, w in cands_sorted[top:])
        lines.append(f"    ... 其余 {hidden} 个候选，合计 {rest_w * 100:.2f}%")
    return [f"[latent #{idx}] {len(cands)} 个候选"] + lines


def main():
    ap = argparse.ArgumentParser(description="解析 <|latent|> 块为人类可读格式")
    ap.add_argument("file", nargs="?", help="输入文件（缺省读 stdin）")
    ap.add_argument("--top", type=int, default=8, help="每位置显示的候选数（默认 8）")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                    help="tokenizer 路径（detokenize 用）")
    ap.add_argument("--no-fluent", action="store_true", help="不输出通读文本")
    ap.add_argument("--no-detail", action="store_true", help="不输出逐位置明细")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="输出 JSON（机器可读）")
    args = ap.parse_args()

    text = open(args.file).read() if args.file else sys.stdin.read()
    segments, latents = extract_latent_blocks(text)
    if not latents:
        print("输入中没有 <|latent|> 块。", file=sys.stderr)
        sys.exit(1)

    if args.as_json:
        print(json.dumps({
            "n_latent": len(latents),
            "segments": segments,
            "positions": [
                {"index": i,
                 "candidates": [{"token": t, "text": t.replace("Ġ", " "),
                                 "weight": w}
                                for t, w in sorted(c, key=lambda x: -x[1])]}
                for i, c in enumerate(latents)
            ],
        }, ensure_ascii=False, indent=2))
        return

    detok = load_detokenizer(args.tokenizer)
    n = len(latents)
    avg_cands = sum(len(c) for c in latents) / n
    print(f"=== latent 思维链：{n} 个 latent 位置，平均每位置 {avg_cands:.1f} 个候选 ===")

    if not args.no_fluent:
        print("\n【通读文本】（〔〕内为 latent 位置的概率最高候选，其余为真实文本）\n")
        print(render_fluent(segments, latents, detok))

    if not args.no_detail:
        print(f"\n【逐位置明细】（* 为主候选，每位置最多显示 {args.top} 个）\n")
        for i, cands in enumerate(latents):
            for line in format_detail(i, cands, detok, args.top):
                print(line)


if __name__ == "__main__":
    main()
