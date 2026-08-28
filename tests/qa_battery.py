#!/usr/bin/env python3
"""通用答题正确性电池：数学/常识/逻辑，贪心解码，硬校验答案。

用法: python3 tests/qa_battery.py [port]   退出码 0=全对, 1=有错
默认走 chat completions，请求级 thinking=false（要快；思考路径由
probe_dspark.py / bench_dspark.py 覆盖）。
"""
import json
import re
import sys
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

# (类别, 问题, [可接受的答案子串列表——全部出现才算对], max_tokens, thinking)
CASES = [
    ("math", "357 乘以 89 等于多少？只回答数字。", ["31773"], 200, False),
    ("math", "计算：(17 + 25) × 13 − 48。只回答数字。", ["498"], 800, True),
    ("math", "2 的 20 次方等于多少？只回答数字。", ["1048576"], 200, False),
    ("math", "999 × 999 等于多少？只回答数字。", ["998001"], 200, False),
    ("math", "1000 以内（含 1000）有多少个能被 7 整除的正整数？只回答数字。", ["142"], 300, False),
    ("math", "鸡兔同笼，共有 35 个头、94 条腿。鸡和兔各有多少只？只回答数量。", ["23", "12"], 400, False),
    ("logic", "所有猫都是动物。咪咪是一只猫。咪咪是动物吗？只回答是或不是并给一句理由。", ["是"], 200, False),
    ("logic", "甲说：乙在说谎。乙说：丙在说谎。丙说：甲和乙都在说谎。"
              "三人中有且只有一人说了真话。谁说了真话？只回答甲、乙或丙。", ["乙"], 1500, True),
    ("logic", "数列 2, 3, 5, 9, 17, ... 的下一个数是多少？只回答数字。", ["33"], 300, False),
    ("logic", "如果所有的 A 都是 B，所有的 B 都是 C，那么所有的 A 都是 C 吗？只回答是或不是。",
     ["是"], 200, False),
    ("logic", "一个班里有 30 名学生，其中 18 人会游泳，15 人会骑车，5 人两样都不会。"
              "两样都会的有几人？只回答数字。", ["8"], 400, False),
    ("common", "标准大气压下，纯水的沸点是多少摄氏度？只回答数字。", ["100"], 200, False),
    ("common", "《红楼梦》的作者是谁？只回答姓名。", ["曹雪芹"], 200, False),
    ("common", "光在真空中的传播速度约为每秒多少公里？只回答数字（万公里数取整）。",
     ["30"], 200, False),
    ("common", "水的化学式是什么？只回答化学式。", ["H2O"], 150, False),
    ("common", "Which is heavier, a pound of feathers or a pound of bricks? "
               "Answer in one short sentence.",
     ["same", "equal"], 200, False),
    ("common", "一年有四季。如果今天是星期三，那么 100 天后是星期几？只回答星期几。",
     ["星期五"], 400, False),
    # 长 prompt 阅读理解：专门跨过 INT8 prefill 的 qlen>=64 触发线，
    # 校验 INT8 路径对 prompt 内容的保真（答案是文中嵌入的事实）
    ("int8rc", "阅读以下记录并回答问题。2026 年 8 月 27 日的维护日志："
               "值班员林澈在 03 时 17 分巡检了编号为 KX-7482 的冷却机组，"
               "记录入口温度 41.6 摄氏度、出口温度 37.2 摄氏度，"
               "更换了两枚规格为 M8×30 的不锈钢螺栓，扭矩设定为 24 牛米。"
               "同日 04 时 02 分，同事周慕复核了全部读数并签字确认。"
               "问题：这次维护更换的螺栓扭矩设定为多少牛米？只回答数字。",
     ["24"], 300, False),
    ("int8rc", "阅读以下会议纪要并回答问题。季度评审会于周二上午举行，"
               "应到 14 人，实到 12 人。会议决定：项目 Nimbus 的预算从 85 万元"
               "上调至 97 万元，截止日期从 9 月 15 日顺延到 10 月 9 日；"
               "项目 Halcyon 维持原预算 62 万元不变，但负责人由赵岑更换为钱蔚。"
               "两个项目均需在下次例会提交周报模板。后勤组负责会议室预订，"
               "下次例会仍定在第一会议室。问题：项目 Nimbus 调整后的预算是"
               "多少万元？只回答数字。", ["97"], 300, False),
]


def gen(prompt, mt, thinking):
    body = json.dumps({
        "model": "ds4f",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": mt,
        "temperature": 0,
        "chat_template_kwargs": {"thinking": thinking},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    m = out["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + "\n" + (m.get("content") or "")


def norm(s):
    # 全角/半角与常见写法归一
    return (s.replace("Ｈ２Ｏ", "H2O").replace("H₂O", "H2O").replace("H₂o", "H2O")
             .replace(",", "").replace("，", "").replace(" ", ""))


fails = []
for cat, q, expects, mt, th in CASES:
    try:
        text = norm(gen(q, mt, th))
    except Exception as e:
        fails.append(f"[{cat}] request error: {e}")
        print(f"[{cat:6s}] ERROR {e}")
        continue
    ok = all(norm(e) in text for e in expects)
    # 同义答案放宽
    if not ok and "星期五" in expects and ("星期五" in text or "周五" in text or "Friday" in text):
        ok = True
    if not ok and "same" in expects and ("same" in text or "equal" in text):
        ok = True
    print(f"[{cat:6s}] {'PASS' if ok else 'FAIL'}  {q[:24]}... => {text.strip()[:60]!r}")
    if not ok:
        fails.append(f"[{cat}] {q[:30]}... expects {expects}, got {text.strip()[:120]!r}")

print("=" * 60)
if fails:
    print(f"FAIL {len(fails)}/{len(CASES)}")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print(f"ALL {len(CASES)} PASS")
sys.exit(0)
