#!/bin/bash
# DSv4-Flash 实验实例（与生产 ds4f 并行跑在 30001 端口，共享巨页权重缓存）
# 用法: run_exp.sh [--extra-sglang-args ...]   环境变量: EXTRA_ENV="A=1 B=2"
set -e
cd /home/wkgcass/ktransformers
source .venv/bin/activate

export SGLANG_DSV4_MODE=2604
export SGLANG_DSV4_2604_SUBMODE=2604B
export FLASHINFER_CUDA_ARCH_LIST=8.9
export TORCH_CUDA_ARCH_LIST=8.9+PTX
export SGLANG_ENABLE_THINKING=1

# 实验旋钮（默认与生产一致，可被外部环境覆盖）
export KT_CPUINFER="${KT_CPUINFER:-44}"
export KT_TP_COUNT="${KT_TP_COUNT:-2}"

exec python -m sglang.launch_server \
  --host 127.0.0.1 --port 30001 \
  --model /var/deepseek-v4-flash/0731 \
  --kt-weight-path /var/deepseek-v4-flash/0731 \
  --kt-method MXFP4 \
  --kt-num-gpu-experts 0 \
  --kt-cpuinfer "$KT_CPUINFER" \
  --kt-threadpool-count "$KT_TP_COUNT" \
  --tensor-parallel-size 1 \
  --context-length 8192 \
  --attention-backend flashinfer \
  --mem-fraction-static "${MEMFRAC:-0.30}" \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 2048 \
  --max-running-requests 1 \
  --watchdog-timeout 1200 \
  --disable-shared-experts-fusion \
  --trust-remote-code \
  --disable-radix-cache \
  --skip-server-warmup \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  $EXTRA_ARGS
