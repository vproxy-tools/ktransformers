#!/bin/bash
# 干净地停掉所有 sglang 相关进程（scheduler/detokenizer/launch_server）。
# 放在脚本文件里避免 pkill/pgrep 匹配到调用者自身的命令行。
for pat in 'sglang\.launch_server' 'sglang::scheduler' 'sglang::detokenizer'; do
    PIDS=$(pgrep -f "$pat" | tr '\n' ' ')
    [ -n "$PIDS" ] && kill $PIDS 2>/dev/null
done
for i in $(seq 1 20); do
    sleep 2
    LEFT=$(pgrep -f 'sglang\.launch_server|sglang::' | wc -l)
    [ "$LEFT" -eq 0 ] && break
done
for pat in 'sglang\.launch_server' 'sglang::scheduler' 'sglang::detokenizer'; do
    pkill -9 -f "$pat" 2>/dev/null
done
sleep 2
exit 0
