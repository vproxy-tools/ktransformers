#!/bin/bash
# 停掉实验实例（所有不在 ds4f.service cgroup 里的 sglang 进程），不影响生产服务
ME=$$
PIDS=$(for p in $(pgrep -f "sglang" 2>/dev/null); do
  [ "$p" = "$ME" ] && continue
  grep -qs "ds4f.service" /proc/$p/cgroup 2>/dev/null && continue
  grep -qs "sglang" /proc/$p/comm 2>/dev/null || grep -qs "launch_server" /proc/$p/cmdline 2>/dev/null || continue
  echo $p
done)
if [ -z "$PIDS" ]; then echo "no experiment processes"; exit 0; fi
echo "killing: $PIDS"
kill $PIDS 2>/dev/null
for i in $(seq 1 15); do
  sleep 2
  ALIVE=""
  for p in $PIDS; do kill -0 $p 2>/dev/null && ALIVE="$ALIVE $p"; done
  [ -z "$ALIVE" ] && break
  kill $ALIVE 2>/dev/null
done
for p in $PIDS; do kill -9 $p 2>/dev/null; done
sleep 2
echo "done"
