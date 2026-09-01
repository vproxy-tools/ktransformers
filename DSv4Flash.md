# DeepSeek-V4-Flash 构建与启动实录（RTX 4090D / ktransformers optimize-latest 分支）

本文记录从源码构建 ktransformers（sglang `dspark-kt-fix` fork + kt-kernel）并启动
DeepSeek-V4-Flash-0731 推理服务的完整步骤。第 11 节为 vision-exp 多模态部署（当前
`ds4f.service` 所服务的模型）。

## 1. 环境信息

> **路径约定**：为保持可移植，正文命令统一使用以下变量（本机实际取值见
> `ds4f.service`；`$USER` 为运行服务的系统用户）：
>
> ```bash
> export KT_ROOT=~/ktransformers                    # ktransformers 仓库根目录
> export MODEL_DIR=/path/to/DeepSeek-V4-Flash-0731  # 模型权重目录（0731 版）
> ```
>
> venv 一律指仓库根 `.venv`。

| 项目 | 值 |
|---|---|
| GPU | 1× NVIDIA GeForce RTX 4090 D（48GB，SM_89 / Ada Lovelace） |
| 驱动 / CUDA toolkit | 550.144.03（最高支持 CUDA 12.4）/ 系统 nvcc 12.6（`/usr/bin/nvcc`） |
| CPU | AMD EPYC 9275F（48 核 96 线程，AVX512 全家桶：F/BW/VL/VNNI/BF16/VBMI，无 AMX） |
| 内存 | 1.5TB（2 NUMA 节点） |
| 系统 | Ubuntu 24.04，gcc 13.3，cmake 3.28，ninja |
| Python venv | 仓库根 `.venv`（Python 3.12，torch 2.11.0+cu128） |
| 模型 | `$MODEL_DIR`（0731 版 checkpoint，156GB，48 个 safetensors 分片，DeepseekV4ForCausalLM） |
| 代码分支 | ktransformers `optimize-latest`；submodule third_party/sglang = **`dspark-kt-fix` 分支**（官方 main `d06762282` + 27 个自研补丁的线性化谱系，见第 9 节与 DSv4F-Opt.md §5.18 补记） |

关键结论：

- **驱动 550 可以跑 cu128 的 torch**：CUDA 12 minor version compatibility，`torch==2.9.1`（PyPI 默认即 cu128 构建）在 r525+ 驱动上均可用。文档要求的 "CUDA 12.8+" 主要是针对 5090 的 SM_120；SM_89 用系统自带 nvcc 12.6 编译完全可行，无需升级 toolkit。
- pyproject 里的 `torch-cu130` 索引只对 `uv` 生效（`[tool.uv.sources]`）；`install.sh` 用的是普通 pip，x86_64 会从 PyPI 装 cu128 版 torch，正好适配本机驱动。

## 2. 构建步骤

> **路径约定**：正文命令使用以下变量，按实际环境替换（本机实际取值见
> `ds4f.service`；`$USER` 为运行服务的系统用户）：
>
> ```bash
> export KT_ROOT=~/ktransformers                    # ktransformers 仓库根目录
> export MODEL_DIR=/path/to/DeepSeek-V4-Flash-0731  # 模型权重目录（0731 版）
> ```
>
> venv 一律指仓库根 `.venv`。

```bash
cd $KT_ROOT

# 2.0 创建 venv
python3.12 -m venv .venv && source .venv/bin/activate

# (可选) pip 缓存挪到大盘，避免根分区被写满
export PIP_CACHE_DIR=$HOME/.pip-cache

# 2.1 初始化 submodule（dspark-kt-fix 分支，指针已记录在仓库）
git submodule update --init --recursive

# 2.2 安装依赖 + sglang（editable）
#     关键版本：torch 2.11.0+cu128、flashinfer-python[cu12] 0.6.15.post1、
#     transformers 5.12.1、tilelang 0.1.11、cuda-python 13.3.1；
#     sgl-deep-gemm 0.1.5.post3+cu129 需 docs.sglang.ai 的 wheel 索引。
#     venv 最初为手工组装（约 200 个包，完整清单 pip freeze），
#     下面的 -e 安装让 sglang 依赖其 pyproject 解析；若个别 cu13 系
#     依赖与驱动不匹配（本机驱动仅支持 CUDA 12.x），参照 pip freeze
#     的实测版本手动 pin。
#     2026-08-25 上游拉齐后注意（DSv4F-Opt.md §5.18）：
#     a) 【重要】上游 pyproject 已 pin torch==2.13.0 + 多个 cu13 wheel
#        （sglang-kernel 0.4.6.post1 / humming-kernels / cutlass-dsl 等），
#        在本机（驱动 550，仅支持 CUDA 12）裸 pip install -e 会把 venv
#        装坏——重装一律加 --no-deps；重建全新 venv 以仓库根
#        requirements-freeze-2026-08-25.txt 为基线；
#     b) editable 重装要 venv 里有 setuptools-scm，且 submodule 需先
#        git fetch sgl --tags（否则版本号回退 0.0.0）；
#     c) sglang-kernel 版本下限已被 fork 放宽到 0.4.5（本地 cu129 构建），
#        不要照 pyproject 装 0.4.6.post1（cu13-only，550 驱动加载不了）。
pip install -e third_party/sglang/python --no-deps --no-build-isolation

# 2.3 编译安装 kt-kernel（自动检测 CPU：NATIVE + AMX=OFF + AVX512 全家桶 ON）
#     必须带 --no-deps：kt-kernel 的 requirements pin 老 torch，裸装会降级依赖
cd kt-kernel && pip install . --no-deps --no-build-isolation && cd ..

# 2.4 验证
.venv/bin/python -c "import torch, sglang, kt_kernel; print('ok', torch.__version__)"
kt doctor    # 应全部"正常"，kt-kernel 显示 v0.6.4
```
## 3. 启动命令（RTX 4090D / SM_89 适配版）

### 3.1 本机调优配置（DSpark 投机解码 + cuda graph，日常使用）

生产使用 DSpark 投机解码栈（sglang `dspark-kt-fix` 分支，详见第 9 节）。
**当前稳态（2026-08-28 更新）：3F+27U + 1M 上下文 + 并发 2 + HiCache
手动快照模式 + decode top-4@3-20**，与仓库根 `ds4f.service` 完全一致（各 env 的注释、回退
开关、显存/池测算原始记录都在该文件与 DSv4F-Opt.md §5.14/§5.17/§9）：

```bash
cd $KT_ROOT

# 架构变量：文档示例是 5090(SM_120)，4090/4090D 要改成 8.9
export FLASHINFER_CUDA_ARCH_LIST=8.9
export TORCH_CUDA_ARCH_LIST="8.9+PTX"

# 思考模式默认开启 + 等级拉满（单请求可用请求体 reasoning_effort 覆盖）
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max

# DSpark + SM89 回退栈必需（缺一不可，见 DSv4F-Opt.md §1）
export SGLANG_RAGGED_VERIFY_MODE=static
# verify 元数据图外构建（ctx>111K 图内录制损坏生成，见 9.3）
export SGLANG_DSv4_VERIFY_META_OUT_OF_GRAPH=1
# 长上下文 prefill 的 indexer gather 峰值需要消碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
# 索引器 tilelang 融合内核（prefill 主力项，见 DSv4F-Opt.md §5.2）
export SGLANG_OPT_USE_TILELANG_INDEXER=1
# 部分常驻专家层 Marlin + 关闭 decode 全 CPU 相位切换（见 §5.7）
export SGLANG_V4_MARLIN_PARTIAL=1
export SGLANG_KT_GPU_EXPERTS_PREFILL_ONLY=0
# DSpark draft 专家挪 CPU（见 §5.11，省 ~9.5GB 显存）
export SGLANG_KT_DSPARK_CPU_EXPERTS=1
# CPU prefill INT8(VNNI) 镜像（分块内核，47K prefill +55%；decode 比特不变；
# 切换需清巨页池冷转，runbook 见 DSv4F-Opt.md §8.4/§8.5）
export KT_CPU_INT8_PREFILL=1
# decode 相位减少激活专家（2026-08-28 定型，DSv4F-Opt.md §9）：decode/verify
# 批对 MoE 层 3-20 只激活 top-4（0-2/21-42 与 prefill 不变），decode +10%；
# 实测与回退开关见 §9.4；勿与 SGLANG_KT_DECODE_GPU_CAP_HALF 同开
export SGLANG_KT_DECODE_TOPK_LAYERS=3-20
# DSpark 长上下文门控：0=关闭（>256K 已随上游 4a5d7d3 修复验证干净，
# 见 DSv4F-Opt.md §5.17/§5.18；要保险丝可设 262144）
export SGLANG_DSPARK_MAX_CTX=0
# HiCache 磁盘 L3 目录（NVMe；勿用 /tmp——根分区已满）
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/var/hicache
# 注：KT_EXPERT_DIST_TRACK=1 是临时测量项（DSv4F-Opt.md §6.10），平时不开

$KT_ROOT/.venv/bin/python -m sglang.launch_server \
  --host 0.0.0.0 --port 30000 \
  --model $MODEL_DIR \
  --kt-weight-path $MODEL_DIR \
  --kt-method MXFP4 \
  --kt-expert-placement-strategy hybrid \
  --kt-num-gpu-full-layers 3 \
  --kt-num-gpu-experts 27 \
  --kt-cpuinfer 48 \
  --kt-threadpool-count 2 \
  --tensor-parallel-size 1 \
  --context-length 1048576 \
  --attention-backend flashinfer \
  --mem-fraction-static 0.96 \
  --max-total-tokens 1052672 \
  --chunked-prefill-size 1024 \
  --max-prefill-tokens 1024 \
  --max-running-requests 2 \
  --watchdog-timeout 1200 \
  --disable-shared-experts-fusion \
  --trust-remote-code \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-write-policy write_through \
  --hicache-io-backend kernel \
  --hicache-mem-layout page_first \
  --hicache-storage-backend file \
  --hicache-storage-prefetch-policy wait_complete \
  --hicache-manual-mode \
  --enable-metrics \
  --skip-server-warmup \
  --speculative-algorithm DSPARK \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4
```

1M 档实测备查（DSv4F-Opt.md §5.17/§5.19）：**27U 是 1M 上下文的最大 U**
（28U 在 800K prefill 中 torch OOM 崩溃、27U 全阶梯通过到 1000K，三重
检查全绿）；KV 池 1052672 token、启动 avail ~2.85GB；1000K 处 prefill
~186 tok/s、全阶梯吞吐 522→186 tok/s 随上下文衰减；单请求上限仍是
"一大一小"（两条接近 1M 会排队等池空间）。512K 档（35U）历史实测：
450K 冷 prefill ~1190s（均 ~378 tok/s）、命中缓存后 ~7.5s。

131072 档（1F+28U）历史配置、当时实测与参数表已移至
`DSv4F-Opt.md` §5.20（部署文档只保留当前稳态；历史进展归 Opt 账本）。

### 3.2 文档基准配置（备用参考，短上下文 / 双并发）

与 3.1 的差异：`--kt-num-gpu-experts 10`、`--kt-enable-dynamic-expert-update`、
`--kt-cpuinfer 60`、`--context-length 16384`、`--mem-fraction-static 0.85`、
`--max-running-requests 2`，无 parser 参数。完整命令见官方文档 `doc/en/DeepSeek-V4-Flash.md`
Step 2（架构环境变量仍按上文 8.9 设置）。该配置未在本机当前栈复测，仅作参考。

### 3.3 思考模式（thinking）

**V4-Flash 出厂默认不思考**（官方 `encoding/encoding_dsv4.py` 的 `thinking_mode` 默认 `"chat"`），
`--reasoning-parser deepseek-v4` 只负责把输出里的 `<think>…</think>` 拆到 `reasoning_content` 字段，
不是开关。开启方式（优先级：请求参数 > 环境变量）：

| 方式 | 写法 | 说明 |
|---|---|---|
| 全局默认开（本服务已启用） | 启动环境变量 `SGLANG_DEFAULT_THINKING=1` | 已写入 `ds4f.service`；输出侧的 `</think>` 切分依赖本 fork 对 `serving_chat.py` 的补丁（explicit_thinking 探测模式下输出侧原本忽略该 env） |
| 单请求开启 | 请求体加 `"chat_template_kwargs": {"thinking": true}` | 不设环境变量时的开启方式 |
| 单请求关闭 | `"chat_template_kwargs": {"thinking": false}` | 覆盖环境变量的全局默认 |
| 推理强度 | `"reasoning_effort": "low"/"medium"/"high"/"xhigh"/"max"` | 按官方文档映射：low→low（简洁思考前缀），medium/xhigh→high，high→high（默认档），max→max（最大强度前缀）。也可用环境变量 `SGLANG_DSV4_REASONING_EFFORT`（ds4f.service 已设 max；旧名 `SGLANG_REASONING_EFFORT` 仍被接受但已标记废弃） |
| 官方开关写法 | `"thinking": {"type": "enabled"}` 或 `{"type": "disabled"}` | 与 `chat_template_kwargs.thinking` 等效，对齐 api-docs.deepseek.com |
| Anthropic 风格 | `"reasoning": {"effort": "none"}` / `{"effort": "max"}` | effort=none 关思考；也接受 `"enabled": true/false` |

开启后的请求示例（`reasoning_content` 为思考过程，`content` 为最终答案）：

```bash
curl -s -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "$MODEL_DIR",
    "messages": [{"role": "user", "content": "9.11和9.9哪个大？"}],
    "temperature": 0.0,
    "max_tokens": 400,
    "chat_template_kwargs": {"thinking": false}
  }'
```

> 注意：手动方式（不经 systemd，直接跑 3.1 命令）命令里已含
> `export SGLANG_DEFAULT_THINKING=1`；systemd 方式由 service 文件注入，无需手动设置。

### 投机解码

已启用 **DSpark**（第 9 节）：`--speculative-algorithm DSPARK` 一个参数，
0731 自带 draft。**不要用 EAGLE/MTP**（draft 权重命名不匹配，见 9.1）。

## 4. 验证

```bash
# 模型列表
curl http://127.0.0.1:30000/v1/models

# 文档的 Decode 测试
curl -s -X POST http://127.0.0.1:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Explain quantum computing in detail:",
    "sampling_params": {"temperature": 0.0, "max_new_tokens": 256}
  }'

# OpenAI 兼容对话
curl -s -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "$MODEL_DIR",
    "messages": [{"role": "user", "content": "用三句话介绍一下你自己。"}],
    "temperature": 0.7,
    "max_tokens": 200
  }'

# 交互式聊天
kt chat --host 127.0.0.1 --port 30000 --temperature 0.7 --max-tokens 2048
```

吞吐与正确性请以 `tests/bench_dspark.py 30000` 为准（正确性 5/5 PASS、
DSpark 单请求吞吐参考区间 32–36 tok/s，口径澄清见 DSv4F-Opt.md §5.20）。

## 5. 排障记录

| 现象 | 原因 | 解决 |
|---|---|---|
| >100K 上下文 prefill OOM（513 页宽 gather 峰值 ~17GB） | indexer torch 回退按全 ctx 派生页宽 gather | 旧方案 chunk 512；现栈 tilelang 索引器（`SGLANG_OPT_USE_TILELANG_INDEXER=1`）+ `expandable_segments` 下 chunk 1024 已实测安全（§5.4；1M 阶梯全档复测，§5.19）。仍 OOM 时回退 512 |
| 生产 30000 与实验 30001 同时启动 → 反复 OOM 崩溃循环 | 两个实例挤一张 48GB 卡 | 实验前停生产（systemd stop），跑完先停实验（DSv4F-Opt.md 3 的 GPU 独占规则） |
| 手动 kill 后服务 30s 拉起失败循环 | ExecStart 指向的 venv/路径失效，或 StartLimitBurst(500/天) 耗尽 | `systemctl reset-failed ds4f && systemctl start ds4f`（sudo） |
| RSS 看起来持续增长 | 紧循环测量含"已 free 未归还"驻留页 | 以 malloc_trim 后读数为准（DSv4F-Opt.md 3.6） |
| 输出损坏（复读/数学错） | 先跑 `tests/probe_dspark.py`，损坏与 ctx 相关时用 `tests/bisect_ctx.sh` 二分 | 已知两类损坏均已修复（DSv4F-Opt.md §4）；复现即新问题 |
| 刷屏 `Received output for rid=... but the state was deleted`，且 `#running-req` 不降 | 客户端断连后请求变僵尸继续解码（§5.18 合并引入，详见 DSv4F-Opt.md §1.9） | 已修复（submodule `70a56a4e0a`，线性化谱系；原 merge 谱系哈希 `0960802076` 见 §5.18 补记）。修复前清场：`curl -X POST /abort_request -d '{"abort_all": true}'`（连所有在跑请求一起停） |
| 升级 sglang 后启动报 `sglang-kernel ... less than 0.4.6.post1` | 上游抬高了启动期版本下限，但 PyPI 0.4.6.post1 wheel 是 CUDA 13 构建（`libcudart.so.13`，需 580+ 驱动），本机 550 驱动装不上 | fork 已放宽下限到 0.4.5（`1ca9971e6c`，本地 0.4.5+cu129 构建符号审计无缺口）；要真正上 0.4.6+ 需从 `python/sglang/kernels/aot` 用 CUDA 12 工具链本地构建 |
## 6. 服务管理（systemd）

工程根目录的 `ds4f.service` 即当前稳态 unit（**DSpark + 3F+27U + 1M ctx +
并发 2 + HiCache 手动快照模式**，2026-08-26 定型，演进过程见
`DSv4F-Opt.md` §5.10/§5.17/§5.19）。曾并存过 no-DSpark 与 DSpark 双
unit；合一后 DSpark 即默认配置，早期 DSpark 专用 unit
`ds4f-dspark.service` 已删除（历史参数见 `DSv4F-Opt.md` §5.10）。

均为崩溃自动拉起（30s 延迟；限流 `StartLimitIntervalSec=86400` /
`StartLimitBurst=500`，即每天最多 500 次重启）。

安装（需要 sudo）：

```bash
sudo cp $KT_ROOT/ds4f.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ds4f        # 开机自启 + 立即启动
```

常用操作：

```bash
systemctl status ds4f                  # 状态（start→ready 约 60~100s）
journalctl -u ds4f -f                  # 跟踪日志，出现 "fired up and ready to roll" 即就绪
sudo systemctl restart ds4f            # 重启（改了 service 文件后需先 daemon-reload）
sudo systemctl stop ds4f               # 停止（先 SIGTERM 给 120s 干净退出，超时强杀）
sudo systemctl disable ds4f            # 取消开机自启
```

手动方式（不经 systemd 时）：

```bash
pkill -f "sglang.launch_server"        # 杀主进程，调度器子进程随之退出
curl -s http://127.0.0.1:30000/v1/models || echo "stopped"   # 确认端口释放
nvidia-smi --query-gpu=memory.used --format=csv               # 确认显存归零
```

## 7. 修改代码后的增量构建安装（迭代开发）

### 7.1 改 sglang（third_party/sglang，纯 Python 包）

`.venv` 里的 sglang 是 **editable 安装**（`dspark-kt-fix` 分支）——改 `third_party/sglang` 源码后
**只需重启服务**（见 7.3），无需重装。运行期间不要切 sglang 分支；改分支=改生产代码。

### 7.2 改 kt-kernel（含 C++/CUDA 编译）

kt-kernel 是拷贝安装，改 C++ 后需重编并装入 `.venv`。**必须带 `--no-deps`**
（requirements pin 老 torch，裸装会把 venv 的 torch 降级）：

```bash
source $KT_ROOT/.venv/bin/activate
cd $KT_ROOT/kt-kernel
python3 -m pip install . --no-deps --no-build-isolation   # 增量重编+装入
```

### 7.3 重启使改动生效

```bash
sudo systemctl restart ds4f
# 无 sudo 时：kill 主进程（systemctl show ds4f -p MainPID --value），30s 后自动拉起
```

- 重启成本：30s 延迟 + 权重加载 + CUDA Graph，约 60~140s。
- 自动拉起受 `StartLimitBurst=500`/天 限制，耗尽需 `systemctl reset-failed`。
- 改动是否生效看新进程启动时间：`systemctl status ds4f` 或 `journalctl -u ds4f`。
## 8. 内存侧权重持久巨页（persistent hugepages）

DeepSeek-V4-Flash 的 CPU 侧常驻权重（MXFP4 路由专家，每个 NUMA 约 81 GiB）不再走
`std::aligned_alloc` 堆分配，而是放在 **每个 NUMA 节点自己的持久化 hugetlbfs 文件** 中：

- 数据文件：`/dev/hugepages/kt_weights/node{N}/weights.bin`（1 GiB 大页，节点内各层共用一个文件，
  固定虚拟地址映射，扩容不搬移）；
- 校验标记：`/var/lib/kt-hugepage-weights/node{N}/L{layer}_E{...}.done`（普通磁盘文件，记录
  布局指纹 + physical→logical expert 映射指纹，全部通过才判定可复用）；
- 进程退出后大页仍被这些文件占用（持久化）。重启加载时 Python 侧先核对标记，
  命中则完全跳过 safetensors 读取（日志 `safetensors skipped`），C++ 侧直接 mmap 已驻留
  大页（日志 `REUSED from persistent hugepages`），CPU 权重加载从约 20+ 秒降到毫秒级。

改动文件：`kt-kernel/cpu_backend/hugepage_weights.hpp`（新增分配器）、
`kt-kernel/operators/amx/{moe_base,fp4-moe}.hpp`（分配与复用钩子，本机实际运行路径）、
`kt-kernel/operators/avx2/{moe_base,mxfp4-moe}.hpp`（镜像改动，编译验证）、
`kt-kernel/python/utils/hugepage_cache.py`（新增）与 `kt-kernel/python/utils/amx.py`（复用快路径）。

环境变量：`KT_HUGEPAGE_WEIGHTS=0` 关闭（回退堆分配）；`KT_HUGEPAGE_WEIGHT_DIR` /
`KT_HUGEPAGE_WEIGHT_META_DIR` 可改目录。模型目录或 `config.json` 变化会自动使全部缓存失效。

运维注意：
- 模型更新后需手动 `rm -rf /var/lib/kt-hugepage-weights /dev/hugepages/kt_weights`（并重建属主
  运行服务的用户），否则旧大页一直占用池子（每节点约 81 GiB）；
- 大页池容量（当前 node0=300、node1=364 个 1 GiB 页）需 ≥ 常驻权重 + 其它大页用户；
- 首次冷加载若中途崩溃，未写标记的层会在下次启动自动重写，无需人工干预。

**运维要点：数据目录需 root 一次性创建。** `/dev/hugepages`（hugetlbfs 挂载点）
为 root 所有；若挂载点被重建（如虚机启动把 `kt_weights/` 清掉），普通用户无法
自行创建目录，C++ `hugepage_weights::enabled()` 会**静默回落堆分配**（日志无任何
hugepage 行，服务照常运行）。重新接通只需 root 一次性：

```bash
sudo mkdir -p /dev/hugepages/kt_weights && sudo chown $USER:$USER /dev/hugepages/kt_weights
# 首次冷加载约 1-2 分钟/全模型，之后 REUSED 秒级
```

目录建好后可先离线验证（不碰 GPU、可与生产共存，产物直接被服务器复用）：

```bash
KT_MODEL_DIR=$MODEL_DIR $KT_ROOT/.venv/bin/python $KT_ROOT/tests/hp_weight_check.py 0   # 第一遍:冷(转换+commit)
KT_MODEL_DIR=$MODEL_DIR $KT_ROOT/.venv/bin/python $KT_ROOT/tests/hp_weight_check.py 0   # 第二遍:热(打印 REUSED)
```

之后 ds4f 任意一次重启自然接通（冷一次，此后每次 REUSED、`load_weight` 显著缩短）。
另：所有 kt-kernel MoE 基准脚本已在文件头强制 `KT_HUGEPAGE_WEIGHTS=0`——合成权重
绝不写入持久 arena（任何一个 bench 跑一遍都会按 cursor 顺序覆盖真实层、打断全部
marker 复用链，虽然下次启动会自愈但整轮冷加载）。若挂载点再次被重建（虚机重启），
重复上面的 sudo 两步即可；即使忘了 sudo，启动也只是回落堆分配，不报错。

### 8.1 GPU 参数的持久巨页缓存

回答一个常见问题：加载是**区分** GPU/CPU 参数的——路由专家（`layers.*.ffn.experts.*`，约 146 GiB）
走 kt-kernel CPU 路径（8 节的 BufferB 巨页缓存），其余约 8.8 GiB GPU 参数（attention、dense、
共享专家、embedding/lm_head、MTP）由 sglang `DefaultModelLoader` 从 safetensors 读取上 GPU。

GPU 参数同样缓存到持久巨页：`third_party/sglang` 新增
`sglang/srt/model_loader/gpu_hugepage_cache.py`，并接入 `weight_utils.py` 的
`safetensors_weights_iterator`。首次加载时写入 **GPU 所在 NUMA 节点**的
`/dev/hugepages/kt_weights/node{N}/gpu_weights.bin`（本机 4090D 在 node1，经
`/sys/bus/pci/devices/<pci>/numa_node` 探测；探测不到时 fallback node0），manifest 在
`/var/lib/kt-hugepage-weights/node{N}/gpu_weights.json`（按文件 size+mtime 校验）。
重启后这些张量直接从驻留大页 `torch.frombuffer` 读出（`fill=0, served=8.79 GiB` 日志确认），
不再读盘。改动 sglang 后需重装：
`cd third_party/sglang && SGLANG_KT_VERSION=0.6.4 pip install --no-deps --no-build-isolation ./python`。
`KT_HUGEPAGE_GPU_WEIGHTS=0` 可关闭。实测重启到就绪约 40 秒，推理输出正常。

若挂载点重建把数据文件清掉（manifest 还在但已失配），下次启动按 size+mtime
校验自动 miss 并重新填充一次，之后恢复正常复用，无需人工干预（数据目录的
重建见第 8 节开头的说明，GPU/CPU 两个缓存共用该目录）。

## 9. DSpark / 推测解码（speculative decoding）支持情况

**已支持并在生产启用。** sglang 走 `dspark-kt-fix` 分支（third_party/sglang，
kt CPU 专家引擎移植；2026-08-25 拉齐官方 main `d06762282`（+986 提交），
带入上游三修复——`4a5d7d3` draft tokens > 4 的静默 KV/compressed-state 损坏、
`8549cce` c128 ragged prefill 竞态、`154f0ac` DSpark+DP/EP metadata；
2026-08-26 历史线性化为 main + 27 个自研 cherry-pick（`dspark-kt-merge`
留档，详见 DSv4F-Opt.md §5.18 补记），DSpark 投机解码 + cuda graph
全部打通：**39.6 tok/s**（无投机基线 26.0，+52%；eager 模式 34.3）。
`--speculative-algorithm DSPARK` 一个参数即可，0731 自带 draft（`mtp.0.*`，
4705 个权重键；config: block_size=5, markov_rank=256, target_layers=[40,41,42]）。
**不要用 EAGLE**（见 9.1）。

### 9.1 通用 EAGLE/MTP 路径仍不可用

`--speculative-algorithm EAGLE --speculative-draft-model-path ...` 启动后 draft
模型（`deepseek_v4_nextn`）权重校验报错（`model.e_proj.weight` 等未初始化）：
draft 期望 MTP 适配层挂在 `model.e_proj/...` 顶层命名，而本 checkpoint 的
`mtp.*` 布局映射后带 `model.layers.43.*` 前缀。DSpark 路径不受影响（走
`deepseek_v4_dspark` 专用模型，布局对齐）。

### 9.2 栈结构与关键修复

- venv：仓库根 `.venv`（torch 2.11.0+cu128；flashinfer 未单独安装，
  flash-ops 由 sgl-kernel 0.4.5+cu129 本地构建提供；sgl-deep-gemm
  cu129 索引；sglang editable 指向
  third_party/sglang@`dspark-kt-fix`，kt-kernel 以 torch 2.11 头文件重编）。
- SM89 适配：sparse 注意力走 Triton 回退、索引器 logits 走 torch 回退、
  topk v1（详见 DSv4F-Opt.md §1）。`paged_mqa_metadata` 的 smem 钳制已被
  上游 2026-08-25 合并带来的批次自适应三 kernel 重写取代（tiny/small
  静态 smem + 大 batch 走 workspace scratch，SM89 天然安全）。
- draft 保持纯 GPU（约 10.6GB，MEMFRAC 需 ≥0.60），target 专家全在 CPU。
- **kt-kernel pinned-buffer 生命周期修复**（graph 损坏的根因）：CPU 专家的
  pinned 中转 buffer 曾走单槽 temp 缓存，graph 捕获后任何 prefill 换槽都会
  释放并被复用，replay 读写别人内存。修复后 graph 默认开启（细节见 DSv4F-Opt.md §4）。
- **输出侧 thinking 切分补丁**（serving_chat.py）：V4 模板的
  explicit_thinking 探测模式下，`SGLANG_DEFAULT_THINKING=1` 只影响 prompt 侧、
  输出侧不切 `</think>`（全部落 content）。补丁让输出侧同样遵循
  请求 > env 的优先级。

### 9.3 长上下文（>111K）注意

`--context-length 131072` 依赖 `SGLANG_DSv4_VERIFY_META_OUT_OF_GRAPH=1`
（verify 元数据若录制进 cuda graph，超过 ~111.4K 后生成会确定性损坏；该问题
已修复并在 125K 上下文端到端验证，run_dspark.sh / ds4f.service 均已默认导出）。
定位与修复过程见 `DSv4F-Opt.md` §4。

### 9.4 验证与工具

- 正确性/吞吐：`tests/bench_dspark.py`（5 提示词贪心）；快速探针：`tests/probe_dspark.py`
  （数学/翻译/重复词检测，退出码 0=干净）；答题电池：`tests/qa_battery.py`
  （19 题数学/逻辑/常识硬校验，退出码 0=全对）。
- 长上下文：`tests/ctx_ladder.py`（现行工具：确定性阶梯 + HiCache 快照续测，
  1M 定档战役与用法见 DSv4F-Opt.md §5.19；`--u` 标注档位、`--stages=` 自定义、
  日志与逐档结果落 /var/ctx1m）。早期 `tests/grow_probe.py`（单会话逐级加长、
  暗号回忆/数学/重复度检查）仍可用于快速手测。
- ctx 阈值二分：`tests/bisect_ctx.sh CTX1 CTX2 ...`（逐 ctx 重启+短探针，判
  CLEAN/CORRUPT；损坏与实际序列长度无关时尤其快）。
- 实验实例：`run_dspark.sh`（30001 端口，CTXLEN/MAXTOK/MEMFRAC/PREFILL/
  EAGER 环境变量可覆盖；`EAGER=1` 回退无损 eager）；`stop_sglang.sh` 安全
  清理所有 sglang 进程（避免 pkill 自匹配；**会连 30000 生产一起停**）。
- 以上工具均已集中在仓库根的 **`tests/` 目录**（早期的
  bench/profiler 脚本也一并移入，另含 prefill 优化轮新增：`bench_prefill.py`
  prefill 吞吐、`bench_moe_sweep.py` MoE 微基准、`kern_test.cpp` 单核内核
  试验台、`analyze_trace.py` profiler 聚合、`test_routing_v4.py` 路由单测）；
  启动/停止脚本（`run_dspark.sh`、`stop_sglang.sh` 等）仍在仓库根。
  完整说明（执行前提、结果解读、通过/不通过分界、清理要求、prefill 优化轮
  全部数字的测法）面向开发者，见 `DSv4F-Opt.md` §3 与 §6；本节仅速查。

## 10. 环境变量使用纪律

下面这些环境变量已逐一审计/实测，**全部不需要显式设置**——4 个默认已开、
2 个无读取点、2 个开启有害（勿引入）：

| env | 本栈状态 | 结论 |
|---|---|---|
| `SGLANG_OPT_FUSE_WQA_WKV=1` | `environ.py:1393` 默认 `EnvBool(True)` | 冗余，不设即开 |
| `SGLANG_OPT_USE_JIT_NORM=1` | `environ.py:1583` 默认 `EnvBool(True)` | 冗余，不设即开 |
| `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=1` | `environ.py:1394` 默认 `EnvBool(True)`（仅 HIP 分支强制关） | 冗余，不设即开 |
| `SGLANG_OPT_USE_FUSED_STORE_CACHE=1` | `environ.py:1582` 默认 `EnvBool(True)` | 冗余，不设即开 |
| `SGLANG_KT_WOA_FP8_TRITON=1` | 无任何读取点（仅 environ.py 定义+docstring） | 已废弃 |
| `SGLANG_OPT_USE_OVERLAP_STORE_CACHE=1` | 无任何读取点 | 已废弃 |
| `SGLANG_OPT_MXFP4_FUSE_RSF_SHARED_ADD=1` | 默认 False；**无融合消费方**，=1 只是跳过 `output.mul_(rsf)` | **禁用**：模型 rsf=1.5，=1 会静默丢掉 ×1.5，三处 GPU MoE 路径（marlin/triton_kernels/trtllm）全部算错 |
| `SGLANG_KT_FP8_LMHEAD=1` | 读取点在 `logits_processor.py:756`（head.weight BF16 129280×4096 满足条件） | **禁用**：见下 |

审计与实测依据见 `DSv4F-Opt.md` §4；ds4f.service 注释中已标注禁止引入。

## 11. vision-exp 多模态部署（2026-09-01，当前 ds4f.service）

**模型**：`/var/deepseek-v4-flash/vision-exp`（0731 + 32 层 ViT 塔与 aligner + 图像
特殊嵌入 + gate.bias_vl）。`config.json` 的 architectures 已改为
`DeepseekV4VisionForCausalLM`（备份 `config.json.bak-pre-vision`）；LLM 主干 0-42 层
与 0731 完全一致，视觉塔挂在 `self.vision` 下、绝不进 `model.layers`。

### 11.1 代码改动（sglang submodule `3b9e8ed316` 起）

| 位置 | 内容 |
|---|---|
| `models/deepseek_v4_vision.py` | 视觉模型：ViT+aligner（按 checkpoint 自带 inference/vision.py 逐算子移植）；`SGLANG_DSV4_VISION_MODE=hybrid`（默认，attn GPU bf16 / FFN CPU INT8）/`gpu`（全 GPU）；权重流式分流加载 |
| `multimodal/processors/deepseek_v4_vision.py` | mm processor：safe_resize/patch 化/N-layout 哨兵展开/pad_value（radix 前缀缓存友好） |
| `layers/moe/vl_routing.py` + topk/hash_topk/MoEGate | gate.bias_vl：image span 内 token 用 bias_vl 选专家（hash 层走 (scores+bias_vl).topk），mask 为 thread-local + 长度防御（DSpark 复用 ForwardBatch 对象会残留旧 mask） |
| `deepseek_v4_backend._extend_swa_windows_for_images` | image span 内 token 的滑窗扩展为整图双向可见（参考 get_window_topk_idxs_visible，索引集离线逐点校验一致） |
| `encoding_dsv4` + `serving_chat` | dsv4 编码保留 image 块（process_image_messages），请求直传 tokenized ids + 图片记录 |
| 架构注册 | model_config/server_args/overrides/speculative_hook 等 9 处按架构名键控的列表补 `DeepseekV4VisionForCausalLM`（backend 强制 dsv4、draft 架构替换、hybrid-swa 等） |
| 分块 prefill | 恢复启用（chunk 1024，与 0731 稳态一致）：调度器 PrefillAdder._mm_extend_chunk_to_spans 在分界点落入图片 span 时把本 chunk 延到 span 结束（span ≤388 token，越界有界），保住 span 内双向注意力契约；带图长 prompt 正常分多批 prefill |

### 11.2 CPU 侧视觉 FFN 的 INT8 方案（参考 kt-kernel int8-prefill.hpp）

权重加载时一次性反量化为 INT8：K 二分组、每 (out, K/2) 静态 fp32 尺度；每步对激活
按 (token, K/2) 动态量化；GEMM 走 `torch._int_mm`（oneDNN VNNI）。选 K/2 粒度是实测
结果：全 K 单尺度 11.4 TFLOPS 但精度差，16×64 窄分片触发 oneDNN 病态路径（~0.06
TFLOPS），K/2 在 6.8–13.6 TFLOPS 且塔输出 cos≈0.9976（gpu 模式与参考实现逐位一致，
cos=1.0）。EPYC 9275F 上 2223-ViT-token 图：塔时延 GPU 235ms / hybrid 427ms（离线）。

### 11.3 使用与实测（hybrid 默认；2026-09-01）

```bash
# 图片请求：image_url 支持本机路径 / data:base64 / http(s) URL
curl -s -X POST http://127.0.0.1:30000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "/var/deepseek-v4-flash/vision-exp",
  "messages": [{"role":"user","content":[
    {"type":"image_url","image_url":{"url":"/home/wkgcass/vision-sample.png"}},
    {"type":"text","text":"表格里 Terminal Bench 一行三个模型的分数分别是多少？"}]}],
  "temperature": 0, "max_tokens": 100,
  "chat_template_kwargs": {"thinking": false}}'
```

- 正确性：vision-sample.png（benchmark 表）读数全部与图内真值一致（83.9/82.7/85.0、
  Cybergym 75.3、ZeroBench 仅两列等）；多图交错、纯色小图、缓存命中（同图换问题）
  均正确。文本侧 probe_dspark CLEAN、bench_dspark 5/5、qa_battery 18/19（唯一"FAIL"
  为脑筋急转弯答"一样重"，语义正确）。
- 性能（DSpark+3F+27U，1M ctx 已恢复）：文本 38.1 tok/s（0731 参考区间内）；新图请求
  gpu 模式 1.10s / hybrid 2.20s（含视觉塔+prefill+少量解码），同图缓存命中 ~1.1s。
- 数值校验：`tests/vision_tower_check.py`（gpu 模式逐位一致、窗口索引集一致、
  hybrid INT8 噪声 cos>0.995）。
- 切换视觉模式：改 `SGLANG_DSV4_VISION_MODE`（ds4f.service）后 `daemon-reload +
  restart`；gpu 模式视觉塔占 ~0.95GB 显存，hybrid ~0.4GB。

### 11.4 与 0731 配置的差异 / 已知限制

- 配置与 0731 稳态完全一致（1M ctx / 1052672 池 / chunk 1024 / 27U），仅多
  `SGLANG_DSV4_VISION_MODE` 与视觉模型路径；启动 avail ~2.49GB（0731 ~2.85GB，
  差值=视觉塔权重）。若长上下文 prefill OOM，优先降 27U→26U（每 U ≈0.5GB）。
- **带图长上下文实测（2026-09-01，27U 未降档）**：`tests/vision_long_ctx_test.py`
  —— 图片+暗号开头+填充文本的 ~397K（全冷）与 ~699K（~400K 前缀命中 + ~303K 新增）
  两个档位均 PASS：暗号与表格数值双对、每批恒 1024 token、无 OOM；~397K 全冷
  prefill 约 10.5 分钟（~630 tok/s 均值）。注意：禁用分块 prefill 的旧方案里
  `max_prefill_tokens` 并不切批也不封单请求上限（调度器一把梭整个序列，长 prompt
  必 OOM）——这是恢复分块 + span 边界保护的原因。
- **文/图/文交叉实测（2026-09-01）**：`tests/vision_interleave_test.py` ——
  [文字+暗号1][图1=benchmark表][文字+暗号2][图2=生成的代码图][文字][四问]，
  ~72.7K 与 ~98.9K 两档全 PASS（双暗号 + 表格 82.7 + 图2 代码 OMEGA-881 全对，
  每批 ≤1024）。两个坑及修复：① span 长度依赖占位符位置（compress_pad=
  3-start%4，同图 283~286 变长）而 mm embedding 缓存/pad_value 只按图片内容
  哈希——已把 span 长度混入 item hash；② 最初的"分界落 span 内则延展本批"会把
  批撑过 KT staging 上限（1129>1024 断言）——改为**收缩优先**（截到 span 起点前，
  批恒 ≤预算；仅 span 恰在批首时延展，≤388）。
- **prefill 性能对比（同机同配置，flush 后全冷单发）**：
  **同口径复测（bench_prefill.py --tokens 8192，47,303 token）：820.6 tok/s vs
  0731 文档 828.8/831.3（-1.1%，噪声内）；100K 档 99,245 token：762.8 vs 文档
  771.7（-1.2%）——文本 prefill 路径无回归**。其余档位（均落衰减曲线上）：
  纯文本 72K=787.7、交叉 72.7K（2 图）=787、交叉 98.9K（2 图）≈785、纯文本
  430K=532.7 tok/s——交叉与纯文本同长度几乎零差异（视觉塔 ~0.4s/图 占 <1%）。
  注意 bench_prefill.py 的 --tokens 按词估算、实际 token ~5.9×，读数以输出
  prompt= 为准；flush 后需等几秒再发（异步清树，抢跑会全量命中缓存得到
  荒谬的 9 万 tok/s）。
- **decode 性能（2026-09-01）**：标准口径 bench_dspark ×5 轮 38.13-38.47
  （均值 38.3，±0.5%）vs 0731 同配置 40.44-40.67（-6%）。步时核算：76.5ms/步
  vs 0731 75.2ms（+1.7%，噪声内）——**decode 基础设施无回归**，差额全部来自
  accept 长度 2.93 vs 3.05（vision-exp 权重自身的 draft 特性；DSpark tok/s
  内容敏感带 ±10%，见 DSv4F-Opt.md §5.20/§9.6 口径警告）。长上下文 99K 纯
  decode：文本 512 tok/8.1s = 63 tok/s（99,072 前缀命中）；带图（2 个 span
  驻 KV）347 tok/6.0s ≈ 58 tok/s——图 span 对 decode 无拖累（decode 窗口把
  其按普通 token 处理）。注：decode 日志的 gen throughput 单点采样波动大
  （29-50 tok/s），以 e2e×token 计数为准。
- 换模型后首次启动需冷转巨页（CPU 专家 ~100s + INT8 镜像，随后 REUSED 秒级）；
  若 `/dev/hugepages/kt_weights` 属主不对按第 8 节修复。
- vision-exp 的 mm 请求 usage 里 prompt_tokens 计的是展开前的文本长度（sglang mm
  计数口径），实际序列含 ~283 token/图 的 span（日志 `[dsv4-vision] expanded`）。

### 11.5 适配后审查与修复（2026-09-01 下午）

对 `me/dspark-kt-fix..HEAD` 三个 vision 提交做了全量静态审查 + 无污染 A/B 实测
（重启全新进程、未服务图片、intra-op=1），结论与修复：

**性能回退定论：非实现问题，是权重差异。** ① 纯净进程复测：47K 冷 prefill
815.8/821.4/822.5（vs 污染进程 820.6、0731 基线 828.8/831.3），bench_dspark
38.16（vs 污染 38.3）——`set_num_threads` 污染与否无可分辨影响；② decode
步时 72.9-75.9ms 与 0731 的 ~73ms 一致，bench tok/s 差额全部来自 accept len
2.93 vs 3.05；③ 直接比对 checkpoint 张量证实 **vision-exp 的 LM 主干权重与
0731 位级不同**（`layers.10.attn.wq_a.weight`、`layers.0.ffn.gate.weight`、
`shared_experts.w1` 等全部不等；vision-exp = 0731 + 新增视觉张量）——主干重新
训过导致路由/draft 分布不同，~1.2% prefill 差与 accept 差均为权重特性。

**修复的正确性问题**（均附触发用例，`tests/vision_encoding_unit_test.py`，
33 项检查全 PASS，且用 `git show HEAD:` 加载修复前模块证实旧版确实触发 bug）：

1. **tool_result 重排图文静默错配**（encoding_dsv4.py，最严重）：原流水线先
   `process_image_messages` 收集图片、后 `sort_tool_results_by_call_order` 重排
   tool_result 块 → 占位符换位而图片记录不变。修复：**先 merge+sort 后抽取图片**
   （`merge_tool_messages` 改为保留 parts-list 原始块）；为保渲染逐字节一致，
   新增 24 项新旧 prompt 逐字节 parity 检查（chat/thinking × 6 类输入全同）。
   附带补齐 tool_result 字符串内容的占位符注入校验（流水线重排后的校验对等）。
2. **`pad_input_ids` fallback 必崩**（deepseek_v4_vision.py）：把单个
   `MultimodalInputs` 当 list 迭代 → TypeError；改为直接迭代 `mm_items`
   （对齐 deepseek_vl2 契约；scheduler.py:2650/2970 传单对象）。
3. **context 图片被丢弃**：`processed_context, _ = ...` 丢弃记录 → 占位符在
   prompt 前缀而记录缺失。修复：context 图片保留且排在本轮图片之前。
4. **stale vl_mask 等长穿透**（deepseek_v4_vision.py forward）：DSpark
   verify/draft 复用 prefill 的 ForwardBatch 时，旧防护仅长度不等检查，等长时
   文本 token 会被按图像 token 路由（bias_vl+SWA 扩展）。修复：forward 的
   finally 里显式清掉 `mm_token_mask/mm_span_left/mm_span_right`，长度检查
   降为纵深防御；并更正了"embed_mm_inputs 会清 mm_inputs"的错误注释。
5. **SWA 图窗扩展未复用 `req_pool_indices_repeated`**（deepseek_v4_backend.py）：
   自建 `repeat_interleave(extend_lens)` 不含 cuda-graph padding/CP 重排。修复：
   复用 `expand_prefill_casually` 的现成结果；`cp_v2_active` 时显式 assert
   拒绝（重排会错位 mask）；padding 尾部补 False/0。
6. **`@dataclass` 误删**（forward_batch_info.py `DSV4OutCacheLoc`，NPU 路径
   实例化必崩）：已恢复装饰器。
7. **`set_num_threads(32)` 进程级泄漏**：原实现首次服务图片后永久抬高 scheduler
   进程的 torch intra-op 线程数（1→32）。修复：改为 `_cpu_gemm_threads_scope()`
   仅在 vision INT8 MLP forward 期间临时生效、结束即恢复（CPU GEMM 是同步的）。
   实测验证：同进程先服务 2 张图再测 47K 冷 prefill = 820.1 tok/s，与从未服务
   图片的纯净进程一致。
8. `ds4f.service` 注释矛盾（Description/底部注释写 256K，ExecStart 实际 1M）：
   已更正为 1M 稳态口径。

**复测（修复后生产实例）**：probe_dspark CLEAN；bench_dspark ALL PASS
（38.62 tok/s，权重特性区间内）；qa_battery 19/19（"一磅羽毛"题答案
"Neither—they both weigh exactly one pound"语义正确，checker 放宽收
`neither`——该 FAIL 在修复前的 §11.3 已记录为 18/19 已知项，与本次修复无关）；
vision_interleave 100K PASS（双暗号+表格+代码图全对，90 批全 ≤1024，hybrid
路径跑在新 threads scope 上）；47K 冷 prefill 820.1（无回归）。
注：`tests/vision_tower_check.py`（离线逐位对照）需停服务跑（GPU 仅剩 ~1GB），
本次未重跑——其数值路径未被改动，hybrid 数值已由服务端文/图交叉用例覆盖。

**追加修复（2026-09-01 中午，真实请求打挂服务后的热修）**：并发场景——文本请求
在 DSpark verify 解码，另一带图请求仍在分块 prefill——`ForwardBatch.init_new`
对 TARGET_VERIFY 批也跑 `compute_mm_token_mask`，此时 schedule batch 的
`multimodal_inputs`（含未在 verify 批里的 prefill 请求）与 extend 长度表不对齐，
`zip(strict=True)` 直接 ValueError 拖垮整个服务。修复：mask 只对 extend 模式
（EXTEND/MIXED/SPLIT_PREFILL）计算（`is_extend_or_draft_extend_or_mixed()`）——
verify/decode/draft 的 token 流本来就没有图像 span。复现用例
`tests/vision_concurrent_verify_test.py`（长解码文本请求 × 带图 9K prefill
并发）：未修复进程必崩（同款 zip ValueError，12:04 复现确认），修复后 PASS
且 probe/bench_dspark/vision_interleave 100K 全绿。

**追加修复 2（2026-09-01 午后，真实会话 OOM 热修）**：多轮长上下文会话
（~162K，前缀全命中）尾声发了一张接近上限的图（span 367，ViT 42x74=3108
patch）→ 视觉塔 SDPA OOM（要 590MiB，当时全卡仅剩 545MiB）拖垮服务。根因：
本 torch 构建（2.11.0+cu128, sm89）**不含任何 flash/cudnn/mem-efficient SDPA
内核**（连续 bf16 理想输入也报 "No available kernel"），视觉注意力只能走 math
路径，其实例化 [heads,n,n] 注意力矩阵 + fp32 softmax 缓冲（3.4K patch ≈0.6GiB）；
而会话中 free VRAM 会从启动时的 ~2.49GiB 衰减（长上下文 decode 工作区驻留 +
triton 懒加载等，13:25 时仅 0.47GiB）。修复 `_vision_sdpa`：先试完整 math SDPA
（与参考逐位一致的快路径），OOM 则 empty_cache 后按 query 分块重试（每块内
对全量 K/V 做 per-row softmax，数学语义相同，瞬时占用降到 ~0.1GiB@512；
CPU fp32 下分块与整体逐位一致）。旋钮：`SGLANG_DSV4_VISION_SDPA_QCHUNK`
（默认 512）；测试钩子 `SGLANG_DSV4_VISION_SDPA_FORCE_CHUNK=1` 强制走分块
（服务端文/图交叉用例在该钩子下 PASS）。注意：懒加载警告
"Triton kernel device-loaded after serving started" 提示 free mem 已低——
大图请求前若日志出现该提示，视为 OOM 前兆信号（回退路径已兜底，不会再崩）。
另外 13:26 启动日志里大量 libtorchcodec/FFmpeg/libavutil 报错为每次启动都有
的良性噪音（mimo 音频 processor 探测，与本栈无关），tokenizer backend 提示
同理，均可忽略。
