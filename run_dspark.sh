#!/bin/bash
# DSv4-Flash + DSpark 实验实例（主线 sglang dspark-kt 分支 + kt CPU 专家）
# 端口 30001，用仓库 .venv（不影响生产 ds4f）
# 用法:
#   ./run_dspark.sh              # kt MXFP4 混合推理（无投机，基线）
#   DSPARK=1 ./run_dspark.sh     # + --speculative-algorithm DSPARK（0731 自带 draft head）
#   SOFT=1 ./run_dspark.sh       # Soft Token Latent Reasoning（与 DSPARK 互斥；
#                                #   radix cache 保持开启：latent 位置用内容派生
#                                #   的合成 key，见 DSv4F-Opt.md §10）
# 环境变量: EXTRA_ARGS / MEMFRAC / KT_CPUINFER / KT_TP_COUNT
set -e
cd /home/wkgcass/ktransformers
source "$(dirname "$0")/.venv/bin/activate"

export FLASHINFER_CUDA_ARCH_LIST=8.9
export TORCH_CUDA_ARCH_LIST=8.9+PTX
export SGLANG_DEFAULT_THINKING="${SGLANG_DEFAULT_THINKING:-1}"
# SM89: deep_gemm (SM90+) paths off, use torch/triton fallbacks
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
# SM89: topk v2 kernel uses thread-block clusters (SM90+ only)
export SGLANG_OPT_USE_TOPK_V2=0
# SM89: indexer logits via torch fallback (deep_gemm is SM90+)
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
# 索引器 tilelang 融合内核替代 torch eager 回退（SM89 实测编译通过；
# prefill 383→493 tok/s，decode 无回退。TILELANG=0 回退）
export SGLANG_OPT_USE_TILELANG_INDEXER="${TILELANG:-1}"

# expandable_segments 消碎片（长 prompt prefill 的 indexer gather 峰值）。
# chunk 1024 在 tilelang 索引器下 >100K 重预填充实测安全（grow_probe 125K 通过）；
# 若配置变更后出现 >100K prefill OOM，PREFILL=512 回退。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export KT_CPUINFER="${KT_CPUINFER:-48}"
export KT_TP_COUNT="${KT_TP_COUNT:-2}"

# 实验实例与生产的持久巨页 arena 物理隔离（2026-09-08 根因确认，
# DSv4F-Opt.md §10.8）：本实例不开 KT_CPU_INT8_PREFILL，fp4-only 的 cursor
# 交错 ([fp4]×41) 与生产 ([fp4][int8]×41) 不同——共享默认 arena 时本实例的
# fp4 写入会覆写生产的 I8V1 INT8 镜像段（9-07/09-08 两次生产乱码事故）。
# 巨页池装不下两套 arena（664G 池 vs 生产 502G + 实验 ~194G），独立目录
# 试过会半建崩溃；故默认直接关持久 arena 走堆分配（功能正常，每次重启
# 全量转换 ~3min）。要持久化可显式 KT_HUGEPAGE_WEIGHTS=1 + 独立目录。
export KT_HUGEPAGE_WEIGHTS="${KT_HUGEPAGE_WEIGHTS:-0}"

# Soft Token Latent Reasoning 模式（可与 DSPARK 组合：THINKING 态的请求会把
# 整个 decode 批降级为普通 target decode（dspark_worker_v2._forward_decode
# 顶部 soft 门控，与 DSPARK_MAX_CTX 门控同机制）；回到 NORMAL 态
# （</think> 之后）自动恢复投机。见 DSv4F-Opt.md §10.10）
SOFT="${SOFT:-0}"
RADIX_ARGS="--disable-radix-cache"
if [ "$SOFT" = "1" ]; then
    export SGLANG_SOFT_THINKING=1
    # 连续 latent 步数兜底（思维链内 argmax=EOS 会被吞，可能不收尾）
    export SGLANG_SOFT_LATENT_MAX_STEPS="${SOFT_MAX_STEPS:-2000}"
    # radix cache 保持开启（latent 位置用内容派生 key）；overlap 调度
    # server_args 会强制关闭，这里显式写出避免告警
    RADIX_ARGS="--disable-overlap-schedule --enable-cache-report"
    # HiCache（L2 host + L3 file）默认挂（HICACHE=0 关闭）；实验用独立存储
    # 目录避免与生产 /var/hicache 互相干扰。合成 key 的 u64 哈希支持见
    # DSv4F-Opt.md §10.9。
    if [ "${HICACHE:-1}" = "1" ]; then
        export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="${HICACHE_DIR:-/var/hicache-exp}"
        RADIX_ARGS="$RADIX_ARGS --enable-hierarchical-cache --hicache-ratio 2 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --hicache-storage-backend file --hicache-storage-prefetch-policy wait_complete"
    fi
fi

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
  --chunked-prefill-size "${PREFILL:-1024}" \
  --max-prefill-tokens "${PREFILL:-1024}" \
  --max-running-requests 1 \
  --watchdog-timeout 1200 \
  --disable-shared-experts-fusion \
  --trust-remote-code \
  $RADIX_ARGS \
  --skip-server-warmup \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  $SPEC_ARGS \
  $EXTRA_ARGS
