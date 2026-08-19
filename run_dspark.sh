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

# 131072（513 页宽）下 prefill 的 torch indexer gather 峰值 ~17GB 会 OOM；
# expandable_segments 消碎片 + chunk 减为 512（gather 峰值 ~4-7GB，1024 在
# >100K 重预填充时仍差 ~0.01GB）。
# 2026-08-19 实测两者都必要；只影响长 prompt 的 prefill 速度，decode 不受影响。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export KT_CPUINFER="${KT_CPUINFER:-44}"
export KT_TP_COUNT="${KT_TP_COUNT:-2}"

SPEC_ARGS=""
if [ "${DSPARK:-0}" = "1" ]; then
    export SGLANG_RAGGED_VERIFY_MODE="${SGLANG_RAGGED_VERIFY_MODE:-static}"
    # 2026-08-19: verify 元数据图外构建。图内录制的构建在 ctx>111.4K 时损坏生成
    # （复读、远程注意力劣化；110592 及以下无恙）。图外执行同一组构建器，
    # 实测吞吐无回退（同机 A/B 27.8 vs 28.2 tok/s，噪声内）。详见 DSv4Flash.md §9.3。
    export SGLANG_DSv4_VERIFY_META_OUT_OF_GRAPH="${SGLANG_DSv4_VERIFY_META_OUT_OF_GRAPH:-1}"
    SPEC_ARGS="--speculative-algorithm DSPARK"
    # draft 权重计入静态预算，DSPARK 下 0.30 起不来；默认抬到 0.60
    if [ -z "${MEMFRAC:-}" ]; then MEMFRAC=0.60; fi
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

# 显存右移：单请求只需 CTXLEN+余量的 KV 池（0.60 mem-frac 默认会分到 ~801k
# token，白占 ~6GB）。覆盖 CTXLEN 时记得同步调 MAXTOK（须为 256 倍数）。
# 默认 131072（模型上限；需 SGLANG_DSv4_VERIFY_META_OUT_OF_GRAPH=1，上面已默认开启）。
MAXTOK="${MAXTOK:-135168}"

exec python -m sglang.launch_server \
  --host 127.0.0.1 --port 30001 \
  --model /var/deepseek-v4-flash/0731 \
  --kt-weight-path /var/deepseek-v4-flash/0731 \
  --kt-method MXFP4 \
  --kt-num-gpu-experts 0 \
  --kt-cpuinfer "$KT_CPUINFER" \
  --kt-threadpool-count "$KT_TP_COUNT" \
  --tensor-parallel-size 1 \
  --context-length "${CTXLEN:-131072}" \
  --attention-backend flashinfer \
  --mem-fraction-static "${MEMFRAC:-0.30}" \
  --max-total-tokens "$MAXTOK" \
  --chunked-prefill-size "${PREFILL:-512}" \
  --max-prefill-tokens "${PREFILL:-512}" \
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
