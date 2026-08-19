#!/bin/bash
# 用法: bisect_ctx.sh CTX1 CTX2 ...  —— 依次以给定 ctx 重启并跑短探针，输出 CLEAN/CRUPT
set -u
cd /home/wkgcass/ktransformers
for CTX in "$@"; do
  PIDS=$(pgrep -f 'sglang\.launch_server' | tr '\n' ' ')
  [ -n "$PIDS" ] && kill $PIDS 2>/dev/null
  for i in $(seq 1 20); do
    sleep 3
    pgrep -f 'sglang\.launch_server' >/dev/null || break
  done
  pkill -9 -f 'sglang\.launch_server' 2>/dev/null
  sleep 3
  echo "=== CTX=$CTX starting ==="
  DSPARK=1 CTXLEN=$CTX MAXTOK=$CTX MEMFRAC=0.60 nohup ./run_dspark.sh > /tmp/dspark_bisect.log 2>&1 &
  ok=0
  for i in $(seq 1 50); do
    sleep 10
    if grep -q "The server is fired up and ready to roll" /tmp/dspark_bisect.log; then ok=1; break; fi
    if ! pgrep -f 'sglang\.launch' >/dev/null; then break; fi
  done
  if [ "$ok" != 1 ]; then
    echo "=== CTX=$CTX SERVER_FAILED ==="
    tail -4 /tmp/dspark_bisect.log
    continue
  fi
  if python3 probe_dspark.py 30001 >/tmp/probe_out.txt 2>&1; then
    echo "=== CTX=$CTX CLEAN ==="
  else
    echo "=== CTX=$CTX CORRUPT ==="
    cat /tmp/probe_out.txt
  fi
done
echo BISECT_DONE
