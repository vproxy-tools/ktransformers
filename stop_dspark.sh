#!/bin/bash
# 精确停止 30001 实验实例（按监听端口找 PID，不误伤其他 sglang）
set -e
PIDS=$(ss -tlnp 2>/dev/null | grep ':30001 ' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
if [ -z "$PIDS" ]; then echo "no listener on 30001"; exit 0; fi
for p in $PIDS; do
  # 杀掉该 PID 所在进程组（sglang 有 scheduler/tokenizer 子进程）
  PGID=$(ps -o pgid= -p $p | tr -d ' ')
  [ -n "$PGID" ] && kill -9 -"$PGID" 2>/dev/null || kill -9 "$p" 2>/dev/null || true
  echo "killed pgid $PGID (pid $p)"
done
sleep 1
ss -tlnp 2>/dev/null | grep ':30001 ' || echo "30001 clear"
