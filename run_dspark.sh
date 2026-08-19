#!/bin/bash
# DSv4-Flash + DSpark 实验实例（主线 sglang dspark-kt 分支 + kt CPU 专家）
# 端口 30001，独立 venv: /var/deepseek-v4-flash/venvs/dspark（不影响生产 ds4f）
# 用法:
#   ./run_dspark.sh              # kt MXFP4 混合推理（无投机，基线）
#   DSPARK=1 ./run_dspark.sh     # + --speculative-algorithm DSPARK（0731 自带 draft head）
# 环境变量: EXTRA_ARGS / MEMFRAC / KT_CPUINFER / KT_TP_COUNT
set -e
cd /home/wkgcass/ktransformers
source /var/deepseek-v4-flash/venvs/dspark/bin/activate

export FLASHINFER_CUDA_ARCH_LIST=8.9
export TORCH_CUDA_ARCH_LIST=8.9+PTX
export SGLANG_DEFAULT_THINKING="${SGLANG_DEFAULT_THINKING:-1}"
# SM89: deep_gemm (SM90+) paths off, use torch/triton fallbacks
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
# SM89: topk v2 kernel uses thread-block clusters (SM90+ only)
export SGLANG_OPT_USE_TOPK_V2=0
# SM89: indexer logits via torch fallback (deep_gemm is SM90+)
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1

export KT_CPUINFER="${KT_CPUINFER:-44}"
export KT_TP_COUNT="${KT_TP_COUNT:-2}"

SPEC_ARGS=""
if [ "${DSPARK:-0}" = "1" ]; then
    export SGLANG_RAGGED_VERIFY_MODE="${SGLANG_RAGGED_VERIFY_MODE:-static}"
    SPEC_ARGS="--speculative-algorithm DSPARK"
    # cuda graph 默认开启（2026-08-19 修复 kt-kernel KExpertsCPUBuffer 的
    # pinned-buffer 生命周期 bug 后正确且更快：39.6 vs 34.3 tok/s，见
    # DSv4F-Opt.md §7.4）。前 2 步 verify 由 dspark_verify.py 强制 eager 预热
    # （首次 verify 的 hidden 输出 buffer 未物化问题）。EAGER=1 回退无损 eager。
    if [ "${EAGER:-0}" = "1" ]; then
        case " $EXTRA_ARGS " in
            *" --disable-cuda-graph "*) ;;
            *) EXTRA_ARGS="$EXTRA_ARGS --disable-cuda-graph" ;;
        esac
    fi
fi

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
  $SPEC_ARGS \
  $EXTRA_ARGS
