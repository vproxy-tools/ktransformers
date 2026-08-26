#!/usr/bin/env python3
"""1M 上下文 U 档位阶梯测试（ctx_ladder）。

目标：为 ctx=1048576 找到能通过完整阶梯的最大 --kt-num-gpu-experts（U）。
每个档位的成败与失败点记录在 --dir 下的结果文件，供 DSv4F-Opt.md §5.17 引用。

工作流（配合 ds4f.service 的 --hicache-manual-mode）：
  1. 开始时 POST /flush_cache 清空在树 KV；若磁盘快照存在则
     POST /hicache/snapshot/load 灌回（load 内部会再 flush 一次）。
  2. 会话为确定性构造：填充文本按 rung 种子生成、助手回复固定为
     "已记录。"（不采用模型实际输出）——保证跨重启/跨 U 档字节稳定，
     前缀 KV 可完整复用；每档只 prefill 增量部分。
  3. 每个 rung：增量填充 → 三项检查（暗号回忆/新数学题/重复度）→
     通过后 POST /hicache/snapshot/save 落盘快照、更新进度文件。
  4. 已通过的 rung 记录在 progress.json，换 U 档重启后从快照+进度
     续测，不重复已测内容。
  5. 失败（abort/OOM 崩溃/检查不过）时记录分类与 journal 尾部，
     不保存快照，退出码非 0。

日志全程追加写入 --dir/u{U}.log（不做截断）。
用法: python3 tests/ctx_ladder.py [port] --u 28 [--stages 20,100,...]
退出码: 0=本档通过 2=rung失败 3=载入快照后复查失败 4=快照缺失 5=锁占用
"""
import argparse
import fcntl
import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request

LADDER = [20, 100, 200, 300, 400, 500, 575, 650, 700, 750, 800, 850, 900, 950, 1000]
CODEWORD = "XK-42Q7"
FIXED_REPLY = "已记录。"
# 每 rung 的模板/固定回复开销（token），填充目标按此预留
RUNG_OVERHEAD = 48
FILL_TIMEOUT = 1800
CHECK_TIMEOUT = 900
IDLE_RETRY = 24          # flush/save/load 等空闲的重试次数 × 5s

WORDS = [
    "灯塔", "海雾", "齿轮", "苔藓", "陨石", "季风", "运河", "蜻蜓", "琥珀", "沙丘",
    "矿井", "帆船", "苔原", "温泉", "果园", "苔藓虫", "火山灰", "玄武岩", "枫叶", "潮汐",
    "萤火虫", "苔湖", "冰川", "峡谷", "稻田", "茶园", "盐湖", "珊瑚礁", "红树林", "针叶林",
]
PLACES = ["北港", "南湾", "西原", "东麓", "中洲", "环礁", "旧城", "新埠", "高地", "洼地"]
ACTS = ["测绘", "采样", "勘测", "记录", "观测", "巡检", "标定", "试验", "分析", "归档"]


class StageFailure(Exception):
    """一个 rung 的失败（已分类），携带 detail 供结果文件记录。"""

    def __init__(self, kind, detail=""):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


def make_filler_lines(n_lines: int, seed: int) -> str:
    """确定性伪资料文本（同 grow_probe 校准，每行 ~39 token）。"""
    rng = random.Random(seed)
    lines = []
    for i in range(max(1, n_lines)):
        w = rng.sample(WORDS, 4)
        lines.append(
            f"记录{seed:02d}-{i:05d}：{w[0]}与{w[1]}在{rng.choice(PLACES)}"
            f"{rng.choice(ACTS)}时，于{rng.randint(1, 12)}月{rng.randint(1, 28)}日"
            f"观测到{w[2]}，样本编号{rng.randint(1000, 9999)}，"
            f"结论为{w[3]}的影响属于{rng.choice(['显著', '轻微', '可忽略'])}。"
        )
    return "\n".join(lines)


def dup_score(text):
    hits = 0
    for i in range(0, len(text) - 16, 8):
        chunk = text[i : i + 10]
        if len(chunk) >= 8 and chunk in text[i + 10 : i + 60]:
            hits += 1
    return hits


class Client:
    def __init__(self, port, log, recovery_wait):
        self.base = f"http://127.0.0.1:{port}"
        self.log = log
        self.recovery_wait = recovery_wait

    def _post(self, path, payload, timeout):
        data = json.dumps(payload).encode() if payload is not None else b""
        req = urllib.request.Request(
            self.base + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode()

    def _get(self, path, timeout):
        with urllib.request.urlopen(self.base + path, timeout=timeout) as r:
            return r.read().decode()

    def _classify_conn_error(self, detail):
        """连接层错误：轮询 /health 判断服务器是否崩溃后自愈。"""
        deadline = time.time() + self.recovery_wait
        recovered_at = None
        while time.time() < deadline:
            try:
                self._get("/health", 5)
                recovered_at = time.time()
                break
            except Exception:
                time.sleep(5)
        if recovered_at is not None:
            self.log.warning("server unreachable then RECOVERED (crash-restart suspected)")
            raise StageFailure("server_recovered", detail)
        raise StageFailure("server_down", detail)

    def post(self, path, payload=None, timeout=300):
        """普通 POST；HTTP 错误抛 StageFailure，连接错误做崩溃判别。"""
        try:
            return self._post(path, payload, timeout)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:500]
            except Exception:
                pass
            raise StageFailure(f"http_{e.code}", f"{path}: {body}")
        except urllib.error.URLError as e:
            self._classify_conn_error(str(e))
        except TimeoutError as e:
            self._classify_conn_error(f"timeout: {e}")
        except OSError as e:
            self._classify_conn_error(str(e))

    def post_idle(self, path, payload=None):
        """flush/save/load：服务器未完全空闲会 400，等待重试。"""
        last = ""
        for i in range(IDLE_RETRY):
            try:
                return self._post(path, payload, 600)
            except urllib.error.HTTPError as e:
                try:
                    last = e.read().decode()[:200]
                except Exception:
                    last = str(e)
                # 只有“未空闲”才值得等；其它 400 直接抛
                if e.code == 400 and ("idle" in last or "running" in last or "queue" in last):
                    if i == 0:
                        self.log.info(f"{path}: server busy, waiting for idle...")
                    time.sleep(5)
                    continue
                raise StageFailure(f"http_{e.code}", f"{path}: {last}")
            except urllib.error.URLError as e:
                self._classify_conn_error(str(e))
        raise StageFailure("idle_timeout", f"{path}: still busy, last={last}")

    def chat(self, messages, max_tokens, timeout):
        body = json.dumps(
            {
                "model": "ds4f",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0,
            }
        ).encode()
        req = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.load(r)
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode()[:500]
            except Exception:
                detail = str(e)
            raise StageFailure(f"http_{e.code}", f"chat: {detail}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            self._classify_conn_error(f"chat: {e}")
        m = out["choices"][0]["message"]
        usage = out.get("usage", {})
        return {
            "text": (m.get("reasoning_content") or "") + (m.get("content") or ""),
            "finish": out["choices"][0].get("finish_reason"),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "latency": time.time() - t0,
        }

    def tokenize_count(self, text):
        out = json.loads(self._post("/tokenize", {"prompt": text}, 300))
        return int(out["count"])


def vram():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
        used, free = r.stdout.strip().split(",")[0:2]
        return f"used={int(used)}MiB free={int(free)}MiB"
    except Exception:
        return "vram=?"


def journal_tail():
    try:
        r = subprocess.run(
            ["journalctl", "-u", "ds4f", "--since", "-15 min", "--no-pager"],
            capture_output=True, text=True, timeout=60,
        )
        lines = r.stdout.strip().splitlines()
        return "\n".join(lines[-60:])
    except Exception as e:
        return f"journalctl failed: {e}"


def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def save_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def calibrate_lines(client, target_tokens, seed, calib):
    """二/三步逼近：返回行数使 filler token 数命中 target ±24。
    行数写入 calib 缓存，跨次运行复用（保证文本逐字节一致）。"""
    key = f"{seed}:{target_tokens}"
    if key in calib:
        return calib[key]
    n_lines = max(1, round(target_tokens / 39.2))
    best = (n_lines, None)
    for _ in range(6):
        text = make_filler_lines(n_lines, seed)
        n = client.tokenize_count(text)
        if best[1] is None or abs(n - target_tokens) < abs(best[1] - target_tokens):
            best = (n_lines, n)
        if abs(n - target_tokens) <= 24:
            break
        per_line = max(1.0, n / n_lines)
        n_lines = max(1, round(n_lines + (target_tokens - n) / per_line))
    calib[key] = best[0]
    return best[0]


def rung_message(k, ladder, calib, client):
    """构造 rung k 的增量 user 消息（确定性）。"""
    idx = ladder.index(k)
    if idx == 0:
        target = k * 1024 - 80
        n = calibrate_lines(client, target, 1, calib)
        return (
            f"重要：请记住暗号 {CODEWORD}，后续我会随时抽查。然后阅读以下资料：\n"
            + make_filler_lines(n, 1)
            + "\n阅读完毕只回复：已记录。之后每批资料读完同样只需简短确认。"
        )
    prev = ladder[idx - 1]
    target = (k - prev) * 1024 - RUNG_OVERHEAD
    n = calibrate_lines(client, target, k, calib)
    return (
        f"继续阅读第{k}批资料（不要总结，读完只回复：已记录。）：\n"
        + make_filler_lines(n, k)
    )


def build_history(upto, ladder, calib, client):
    """重建到 rung `upto`（含）为止的会话历史（含固定助手回复）。"""
    history = []
    for k in [x for x in ladder if x <= upto]:
        history.append({"role": "user", "content": rung_message(k, ladder, calib, client)})
        history.append({"role": "assistant", "content": FIXED_REPLY})
    return history


def run_checks(client, history, k, log):
    """三项检查（侧枝请求，不进入主链）。返回 (ok, 详情)。"""
    fails, detail = [], {}
    r = client.chat(
        history + [{"role": "user", "content": "只输出我最初给你的暗号本身，不要其它内容，直接给答案。"}],
        300, CHECK_TIMEOUT,
    )
    ok = CODEWORD in r["text"]
    detail["codeword"] = {"ok": ok, "finish": r["finish"], "head": r["text"][:60]}
    if not ok:
        fails.append(f"codeword lost: {r['text'][:60]!r}")

    a, b = random.Random(k * 7).sample(range(11, 59), 2)
    r = client.chat(
        history + [{"role": "user", "content": f"{a}乘以{b}等于多少？只回答算式和结果。"}],
        400, CHECK_TIMEOUT,
    )
    ok = str(a * b) in r["text"]
    detail["math"] = {"ok": ok, "finish": r["finish"], "head": r["text"][:60]}
    if not ok:
        fails.append(f"math {a}x{b} wrong: {r['text'][:60]!r}")

    topic = random.Random(k).choice(WORDS)
    r = client.chat(
        history + [{"role": "user", "content": f"写一段80字左右的短文，主题：{topic}。"}],
        600, CHECK_TIMEOUT,
    )
    d = dup_score(r["text"])
    ok = len(r["text"]) >= 60 and d <= 4
    detail["essay"] = {"ok": ok, "dup": d, "finish": r["finish"], "len": len(r["text"])}
    if len(r["text"]) < 60:
        fails.append(f"essay short: {r['text'][:60]!r}")
    elif d > 4:
        fails.append(f"essay dup({d}): {r['text'][:100]!r}")
    return (not fails), fails, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", type=int, default=30000)
    ap.add_argument("--u", type=int, required=True, help="当前档位 U（用于命名/记录）")
    ap.add_argument("--stages", default="", help="逗号分隔 rung 列表（K token），默认内置 1M 阶梯")
    ap.add_argument("--snap", default="/var/hicache-snaps/ctx1m")
    ap.add_argument("--dir", default="/var/ctx1m")
    ap.add_argument("--recovery-wait", type=int, default=900)
    ap.add_argument("--no-flush", action="store_true",
                    help="跳过 flush+快照 load，沿用进程内已有树（配合 --assume-rung；"
                         "用于换档重启后借同类请求重建的 device 常驻主干继续爬）")
    ap.add_argument("--assume-rung", type=int, default=0,
                    help="直接假定进度到该 rung（prompt_tokens 取自历史结果文件）")
    args = ap.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    lock_path = os.path.join(args.dir, "ladder.lock")
    lock_f = open(lock_path, "w")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"another ladder is running (lock: {lock_path})", file=sys.stderr)
        sys.exit(5)

    log_path = os.path.join(args.dir, f"u{args.u}.log")
    import logging
    log = logging.getLogger(f"ctx_ladder.u{args.u}")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(log_path)          # 追加模式，永不截断
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(fh)
    log.addHandler(sh)

    ladder = sorted(int(x) for x in args.stages.split(",")) if args.stages else list(LADDER)
    client = Client(args.port, log, args.recovery_wait)

    results_path = os.path.join(args.dir, f"u{args.u}.json")
    progress_path = os.path.join(args.dir, "progress.json")
    calib_path = os.path.join(args.dir, "calib.json")
    results = load_json(results_path, {"u": args.u, "rungs": {}})
    results["status"] = "running"
    results.pop("fail", None)
    progress = load_json(progress_path, {"last_passed": 0, "last_prompt_tokens": 0, "by_tier": {}})
    calib = load_json(calib_path, {})

    log.info(f"=== ctx_ladder start: u={args.u} ladder={ladder} snap={args.snap} ===")
    log.info(f"progress: last_passed={progress['last_passed']}K by_tier={progress['by_tier']}")

    # ---- 1. 清空在树 KV，随后尽量从磁盘快照恢复 ----
    manifest = os.path.join(args.snap, "manifest.json")
    if args.assume_rung:
        # 显式假定进度：从历史档位结果取该 rung 的 prompt_tokens
        pt = 0
        import glob
        for f in sorted(glob.glob(os.path.join(args.dir, "u*.json"))):
            try:
                pt = json.load(open(f))["rungs"][str(args.assume_rung)]["prompt_tokens"]
                break
            except Exception:
                continue
        if not pt:
            pt = args.assume_rung * 1024
        progress.update({"last_passed": args.assume_rung, "last_prompt_tokens": pt})
        progress["by_tier"][str(args.u)] = args.assume_rung
        save_json_atomic(progress_path, progress)
        log.info(f"assume-rung {args.assume_rung}K (prompt_tokens={pt})")

    if args.no_flush:
        log.info("no-flush: keeping in-process tree (device-resident trunk reuse)")
        if progress["last_passed"] >= ladder[0]:
            history = build_history(progress["last_passed"], ladder, calib, client)
            ok, fails, _ = run_checks(client, history, progress["last_passed"], log)
            if not ok:
                for f in fails:
                    log.error(f"resume check FAILED at {progress['last_passed']}K: {f}")
                results["status"] = "resume_check_failed"
                results["fail"] = {"rung": progress["last_passed"], "kind": "resume_check_failed", "detail": fails}
                save_json_atomic(results_path, results)
                sys.exit(3)
            log.info(f"resume check at {progress['last_passed']}K passed (in-process trunk sane under u={args.u})")
    else:
        client.post_idle("/flush_cache")
        log.info("cache flushed (in-tree KV cleared)")
        if os.path.exists(manifest):
            r = json.loads(client.post_idle("/hicache/snapshot/load", {"path": args.snap}))
            log.info(
                f"snapshot loaded: nodes={r['nodes_loaded']} tokens={r['tokens_loaded']} "
                f"skipped={r['nodes_skipped']} {r['elapsed_s']:.1f}s"
            )
            if progress["last_passed"] >= ladder[0]:
                # 换档重启后：用已载入前缀做一次复查（前缀命中，代价极小）
                history = build_history(progress["last_passed"], ladder, calib, client)
                ok, fails, _ = run_checks(client, history, progress["last_passed"], log)
                if not ok:
                    for f in fails:
                        log.error(f"resume check FAILED at {progress['last_passed']}K: {f}")
                    results["status"] = "resume_check_failed"
                    results["fail"] = {"rung": progress["last_passed"], "kind": "resume_check_failed", "detail": fails}
                    save_json_atomic(results_path, results)
                    log.error(journal_tail())
                    sys.exit(3)
                log.info(f"resume check at {progress['last_passed']}K passed (snapshot sane under u={args.u})")
        elif progress["last_passed"] > 0:
            results["status"] = "snapshot_missing"
            save_json_atomic(results_path, results)
            log.error(f"progress says {progress['last_passed']}K passed but {manifest} missing; refusing to re-prefill from zero")
            sys.exit(4)

    todo = [k for k in ladder if k > progress["last_passed"]]
    if not todo:
        log.info(f"nothing to do: all rungs ≤ {progress['last_passed']}K already passed")
        results["status"] = "pass_tier"
        save_json_atomic(results_path, results)
        sys.exit(0)

    history = build_history(progress["last_passed"], ladder, calib, client)
    prev_actual = progress["last_prompt_tokens"]

    # ---- 2. 逐 rung：增量填充 → 检查 → 快照 ----
    for k in todo:
        prev_rung = ladder[ladder.index(k) - 1] if ladder.index(k) > 0 else 0
        est = prev_actual + (k - prev_rung) * 1024
        if est + 2048 > 1048576:
            results["status"] = "exceeds_ctx"
            results["fail"] = {"rung": k, "kind": "exceeds_ctx", "detail": f"est prompt {est}"}
            save_json_atomic(results_path, results)
            log.error(f"rung {k}K would exceed ctx (est {est}); stopping")
            sys.exit(2)

        msg = rung_message(k, ladder, calib, client)
        save_json_atomic(calib_path, calib)
        prev_rung = ladder[ladder.index(k) - 1] if ladder.index(k) > 0 else 0
        log.info(f"[rung {k}K] filling (target +{(k - prev_rung) * 1024} tok)...")
        try:
            r = client.chat(history + [{"role": "user", "content": msg}], 24, FILL_TIMEOUT)
        except StageFailure as e:
            _record_failure(results, results_path, log, k, e)
            sys.exit(2)
        new_tok = r["prompt_tokens"] - prev_actual if prev_actual else r["prompt_tokens"]
        rate = new_tok / max(0.001, r["latency"])
        log.info(
            f"[rung {k}K] fill: prompt={r['prompt_tokens']} (+{new_tok}) "
            f"{r['latency']:.0f}s ~{rate:.0f} tok/s finish={r['finish']} | {vram()}"
        )
        log.info(f"[rung {k}K] reply(discarded): {r['text'][:80]!r}")
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": FIXED_REPLY})

        try:
            ok, fails, checks = run_checks(client, history, k, log)
        except StageFailure as e:
            _record_failure(results, results_path, log, k, e)
            sys.exit(2)
        for f in fails:
            log.info(f"[rung {k}K] check fail: {f}")
        log.info(f"[rung {k}K] checks: {json.dumps(checks, ensure_ascii=False)}")
        if not ok:
            e = StageFailure("check_failed", "; ".join(fails))
            _record_failure(results, results_path, log, k, e)
            sys.exit(2)

        try:
            s = json.loads(client.post_idle("/hicache/snapshot/save", {"path": args.snap}))
        except StageFailure as e:
            _record_failure(results, results_path, log, k, e)
            sys.exit(2)
        log.info(
            f"[rung {k}K] snapshot: nodes={s['nodes_saved']} tokens={s['tokens_saved']} "
            f"pages={s['pages_saved']} {s['bytes_written']/1e9:.2f}GB {s['elapsed_s']:.1f}s"
        )

        results["rungs"][str(k)] = {
            "ts": time.strftime("%F %T"),
            "prompt_tokens": r["prompt_tokens"], "new_tokens": new_tok,
            "fill_s": round(r["latency"], 1), "tok_s": round(rate),
            "checks": checks, "snapshot": s,
        }
        save_json_atomic(results_path, results)
        progress.update({"last_passed": k, "last_prompt_tokens": r["prompt_tokens"]})
        progress["by_tier"][str(args.u)] = k
        save_json_atomic(progress_path, progress)
        prev_actual = r["prompt_tokens"]

    results["status"] = "pass_tier"
    results["finished"] = time.strftime("%F %T")
    save_json_atomic(results_path, results)
    log.info(f"=== u={args.u} PASSED full ladder to {todo[-1]}K ===")
    sys.exit(0)


def _record_failure(results, results_path, log, k, e):
    results["status"] = "failed"
    results["fail"] = {"rung": k, "kind": e.kind, "detail": e.detail,
                       "ts": time.strftime("%F %T")}
    save_json_atomic(results_path, results)
    log.error(f"[rung {k}K] FAILED kind={e.kind} detail={e.detail}")
    log.error("---- journal tail ----\n" + journal_tail())
    # 快照不保存：进度文件保持在上一个通过的 rung


if __name__ == "__main__":
    main()
