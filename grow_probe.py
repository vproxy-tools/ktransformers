#!/usr/bin/env python3
"""上下文增长探针：ctx=131072 服务器上单会话逐级加长，每级做三件事
  1. 远程暗号回忆（第 1 级埋下，检验长程注意力是否存活）
  2. 新数学题（检验当步生成是否退化）
  3. 重复度检测（dup_score，损坏时文本复读）
判定各级 PASS/FAIL，定位损坏出现的“实际序列长度”区间。

用法: python3 grow_probe.py [port] [--stages 8,64,104,112,120] (单位 K token)
"""
import json
import random
import re
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 30001
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

stages_arg = [a for a in sys.argv[1:] if a.startswith("--stages")]
if stages_arg:
    STAGES = [int(x) for x in stages_arg[0].split("=")[1].split(",")]
else:
    STAGES = [20, 96, 112, 120]

CODEWORD = "XK-42Q7"

WORDS = [
    "灯塔", "海雾", "齿轮", "苔藓", "陨石", "季风", "运河", "蜻蜓", "琥珀", "沙丘",
    "矿井", "帆船", "苔原", "温泉", "果园", "苔藓虫", "火山灰", "玄武岩", "枫叶", "潮汐",
    "萤火虫", "苔湖", "冰川", "峡谷", "稻田", "茶园", "盐湖", "珊瑚礁", "红树林", "针叶林",
]
PLACES = ["北港", "南湾", "西原", "东麓", "中洲", "环礁", "旧城", "新埠", "高地", "洼地"]
ACTS = ["测绘", "采样", "勘测", "记录", "观测", "巡检", "标定", "试验", "分析", "归档"]


def make_filler(target_tokens: int, seed: int) -> str:
    """生成 ~target_tokens 的伪资料文本（去重、带编号，避免天然重复）。
    实测 DSv4 分词器下每行约 39 token（2026-08-19 校准）。"""
    rng = random.Random(seed)
    lines = []
    n = max(1, target_tokens // 39)
    for i in range(n):
        w = rng.sample(WORDS, 4)
        lines.append(
            f"记录{seed:02d}-{i:05d}：{w[0]}与{w[1]}在{rng.choice(PLACES)}"
            f"{rng.choice(ACTS)}时，于{rng.randint(1, 12)}月{rng.randint(1, 28)}日"
            f"观测到{w[2]}，样本编号{rng.randint(1000, 9999)}，"
            f"结论为{w[3]}的影响属于{rng.choice(['显著', '轻微', '可忽略'])}。"
        )
    return "\n".join(lines)


def chat(messages, mt):
    body = json.dumps(
        {
            "model": "ds4f",
            "messages": messages,
            "max_tokens": mt,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        out = json.load(r)
    m = out["choices"][0]["message"]
    usage = out.get("usage", {})
    return (
        (m.get("reasoning_content") or ""),
        (m.get("content") or ""),
        int(usage.get("prompt_tokens", 0)),
        int(usage.get("completion_tokens", 0)),
        time.time() - t0,
    )


def dup_score(text):
    hits = 0
    for i in range(0, len(text) - 16, 8):
        chunk = text[i : i + 10]
        if len(chunk) >= 8 and chunk in text[i + 10 : i + 60]:
            hits += 1
    return hits


history = []
print(f"[stage0] 埋暗号 + 8K 起始上下文", flush=True)
seed_msg = (
    f"重要：请记住暗号 {CODEWORD}，后续我会随时抽查。然后阅读以下资料：\n"
    + make_filler(19500, 1)
    + "\n阅读完毕只回复：已记录。"
)
history.append({"role": "user", "content": seed_msg})
rc, c, pt, ct, dt = chat(history, 20)
history.append({"role": "assistant", "content": (rc + c)[:200]})
print(f"  prompt={pt} completion={ct} {dt:.0f}s", flush=True)


def check(stage_k, pt):
    fails = []
    rc, c, p1, _, _ = chat(
        history + [{"role": "user", "content": "只输出我最初给你的暗号本身，不要其它内容。"}],
        150,
    )
    ok_code = CODEWORD in (rc + c)
    if not ok_code:
        fails.append(f"codeword lost: {(rc + c)[:60]!r}")
    a, b = random.Random(stage_k * 7).sample(range(11, 59), 2)
    rc2, c2, _, _, _ = chat(
        history + [{"role": "user", "content": f"{a}乘以{b}等于多少？只回答算式和结果。"}],
        200,
    )
    ok_math = str(a * b) in (rc2 + c2)
    if not ok_math:
        fails.append(f"math {a}x{b}={a*b} wrong: {(rc2 + c2)[:60]!r}")
    rc3, c3, _, _, _ = chat(
        history
        + [{"role": "user", "content": f"写一段80字左右的短文，主题：{random.Random(stage_k).choice(WORDS)}。"}],
        400,
    )
    body = rc3 + c3
    d = dup_score(body)
    if len(body) < 60:
        fails.append(f"essay short: {body[:60]!r}")
    elif d > 4:
        fails.append(f"essay dup({d}): {body[:100]!r}")
    status = "FAIL" if fails else "PASS"
    print(f"[stage {stage_k}K] prompt~{pt} => {status} codeword={ok_code} math={ok_math} dup={d}", flush=True)
    for f in fails:
        print(f"    - {f}", flush=True)
    return not fails


# 第一级上下文已经在 ~8K：直接检查
check(STAGES[0] if STAGES[0] <= 8 else 8, pt)

prev = 8
for k in STAGES[1:]:
    add = (k - prev) * 1024
    msg = (
        f"继续阅读第{k}批资料（不要总结，读完只回复：已记录第{k}批）：\n"
        + make_filler(add - 400, 100 + k)
    )
    history.append({"role": "user", "content": msg})
    rc, c, pt, ct, dt = chat(history, 20)
    history.append({"role": "assistant", "content": (rc + c)[:200]})
    print(f"[stage{k}K] prefill done prompt={pt} ({dt:.0f}s)", flush=True)
    check(k, pt)
    prev = k

print("grow_probe done", flush=True)
