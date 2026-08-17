# DeepSeek-V4-Flash 构建与启动实录（RTX 4090D / ktransformers optimize-latest 分支）

本文记录在本文机器上从源码编译 ktransformers（sglang-kt + kt-kernel）并启动
DeepSeek-V4-Flash-0731 推理服务的完整步骤，包括与官方文档
（`doc/en/DeepSeek-V4-Flash.md`）不一致之处及原因。已在 2026-08-17 全流程验证通过。

## 1. 环境信息

| 项目 | 值 |
|---|---|
| GPU | 1× NVIDIA GeForce RTX 4090 D（48GB，SM_89 / Ada Lovelace） |
| 驱动 / CUDA toolkit | 550.144.03（最高支持 CUDA 12.4）/ 系统 nvcc 12.6（`/usr/bin/nvcc`） |
| CPU | AMD EPYC 9275F（48 核 96 线程，AVX512 全家桶：F/BW/VL/VNNI/BF16/VBMI，无 AMX） |
| 内存 | 1.5TB（2 NUMA 节点） |
| 系统 | Ubuntu 24.04，gcc 13.3，cmake 3.28，ninja |
| Python venv | 仓库根目录 `.venv`（Python 3.12.3） |
| 模型 | `/var/deepseek-v4-flash/0731`（156GB，48 个 safetensors 分片，DeepseekV4ForCausalLM） |
| 代码分支 | ktransformers `optimize-latest`（submodule third_party/sglang = kvcache-ai fork @ bc7f0058f） |

关键结论：

- **驱动 550 可以跑 cu128 的 torch**：CUDA 12 minor version compatibility，`torch==2.9.1`（PyPI 默认即 cu128 构建）在 r525+ 驱动上均可用。文档要求的 "CUDA 12.8+" 主要是针对 5090 的 SM_120；SM_89 用系统自带 nvcc 12.6 编译完全可行，无需升级 toolkit。
- pyproject 里的 `torch-cu130` 索引只对 `uv` 生效（`[tool.uv.sources]`）；`install.sh` 用的是普通 pip，x86_64 会从 PyPI 装 cu128 版 torch，正好适配本机驱动。

## 2. 构建步骤

```bash
cd /home/wkgcass/ktransformers
source .venv/bin/activate

# (可选) pip 缓存挪到大盘，避免根分区被写满
export PIP_CACHE_DIR=/var/pip-cache

# 2.1 初始化 submodule（仓库已带则跳过）
git submodule update --init --recursive

# 2.2 安装 sglang-kt（含 torch 2.9.1+cu128 等全部依赖，下载约 10GB）
#     系统依赖 libhwloc-dev、pkg-config 已预装，故跳过 deps 子命令（避免 sudo 交互）
./install.sh sglang

# 2.3 编译安装 kt-kernel（自动检测 CPU：NATIVE + AMX=OFF + AVX512_BF16/VNNI/VBMI=ON）
cd kt-kernel && ./install.sh build && cd ..

# 2.4 flashinfer 升级并保证 python/cubin 版本一致
#     注意：cubin 在 PyPI 最高只有 0.6.13，所以两边都固定 0.6.13
#     （--upgrade 会装成 python 0.6.17 + cubin 0.6.13 的错位组合）
pip install "flashinfer-python==0.6.13" "flashinfer-cubin==0.6.13"

# 2.5 tilelang：必须用 0.1.13 + apache-tvm-ffi 0.1.12（见第 4 节排障记录）
pip install "tilelang==0.1.13"   # 会自动带上 apache-tvm-ffi==0.1.12

# 2.6 验证
kt doctor          # 应全部"正常"，kt-kernel 显示 v0.6.4 (AVX512_BF16)
python -c "from transformers import DeepseekV4Config; print('ok')"   # transformers-kt 5.6.0.post1 自带
```

### 与官方文档不同的两处依赖处理

1. **transformers 不需要降到 4.57.1**。
   文档说 sglang-kt 不固定 transformers、需手动 pin `transformers==4.57.1`；但本分支
   pyproject 已固定 `transformers-kt==5.6.0.post1`（kvcache-ai 自己的 fork，就是为修复
   transformers 5.x 的 `DeepSeekV4Config` dataclass TypeError 而生），实测导入正常。
   直接装完即可，不要再用 pip 装 4.57.1 覆盖。

2. **tilelang 用 0.1.13，不要按文档装 0.1.8**。
   文档的验证组合 `tilelang==0.1.8 + apache-tvm-ffi<0.1.12` 对应 main 分支旧代码，
   在本分支（submodule 更新后）实测均崩：
   - tvm-ffi ≤ 0.1.11：tilelang JIT 编译内核时 `AttributeError: '_NestedLoopCheckVisitor'
     object has no attribute '_inst'`（tilelang 自带 TVM 的 `TVMDerivedObject.__setattr__`
     与旧版 tvm-ffi C 层 Object 协议不兼容）；
   - tilelang 0.1.8 + tvm-ffi ≥ 0.1.12：tilelang 导入时 registry 崩
     `AttributeError: attribute '__dict__' of 'type' objects is not writable`。
   - **tilelang 0.1.13 + apache-tvm-ffi 0.1.12**（0.1.13 的原生依赖配对）编译、运行均正常。

## 3. 启动命令（RTX 4090D / SM_89 适配版）

### 3.1 本机调优配置（长上下文 / 单并发，日常使用）

```bash
cd /home/wkgcass/ktransformers
source .venv/bin/activate

# 必需的两个 DSV4 变量（本分支 sglang-kt 硬性要求，不设第一个直接 NotImplementedError）
export SGLANG_DSV4_MODE=2604          # 0731 模型 rope factor=16、无 qk_nope_head_dim 字段 → 2604 模式
export SGLANG_DSV4_2604_SUBMODE=2604B # V4-Flash MXFP4 路径的 SwiGLU clamp=10.0，须与 --kt-method MXFP4 搭配

# 架构变量：文档示例是 5090(SM_120)，4090/4090D 要改成 8.9
export FLASHINFER_CUDA_ARCH_LIST=8.9
export TORCH_CUDA_ARCH_LIST="8.9+PTX"

python -m sglang.launch_server \
  --host 0.0.0.0 --port 30000 \
  --model /var/deepseek-v4-flash/0731 \
  --kt-weight-path /var/deepseek-v4-flash/0731 \
  --kt-method MXFP4 \
  --kt-num-gpu-experts 0 \
  --kt-cpuinfer 44 \
  --kt-threadpool-count 2 \
  --kt-gpu-prefill-token-threshold 4096 \
  --tensor-parallel-size 1 \
  --context-length 131072 \
  --attention-backend flashinfer \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 2048 \
  --max-running-requests 1 \
  --watchdog-timeout 1200 \
  --disable-shared-experts-fusion \
  --trust-remote-code \
  --cuda-graph-bs 1 \
  --cuda-graph-max-bs 1 \
  --disable-radix-cache \
  --skip-server-warmup \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4
```

2026-08-17 实测：约 **46 秒**就绪（0 GPU 专家省去权重 swizzle），显存约 **32.2GB / 48GB**，
解码 **19.6 tok/s**（128 token，含首 token 延迟），中英文生成与 reasoning/tool-call 解析均正常。

参数说明：

| 参数 | 值 | 说明 |
|---|---|---|
| `--kt-num-gpu-experts 0` | 全部 256 专家驻留 CPU（双 NUMA 线程池） | 0 GPU 专家时不要加 `--kt-enable-dynamic-expert-update`（无意义） |
| `--kt-cpuinfer 44` | CPU 推理线程数 | 48 物理核留 4 核给系统 |
| `--context-length 131072` | 长上下文 | **必须是 page_size(256) 的倍数**（131070 这类非整页数会被向下取整浪费页，虽然能跑） |
| `--mem-fraction-static 0.90` | 静态显存占比 | KV 池按需分配，调高不占便宜；0.95 会在长 prompt 分块 prefill 时挤压 workspace，有 OOM 风险 |
| `--reasoning-parser deepseek-v4` / `--tool-call-parser deepseekv4` | 输出解析器 | 两个名字均已在注册表中（注意写法不同：一个带 `-` 一个不带） |

注意事项：

- **长 prompt 首 token 延迟是分钟级**：131k 上下文 ÷ 2048 分块 ≈ 64 次前向，且专家全在 CPU。
  常用长 prompt 可把 `--chunked-prefill-size` / `--max-prefill-tokens` 提到 4096 改善 TTFT
  （代价是 prefill 显存峰值略升，mem-fraction-static 需相应留余量）。
- `--max-running-requests 1` 与 `--cuda-graph-bs 1` 保持一致；要并发就同步调大并测显存。

### 3.2 文档基准配置（备用参考，短上下文 / 双并发）

与 3.1 的差异：`--kt-num-gpu-experts 10`、`--kt-enable-dynamic-expert-update`、
`--kt-cpuinfer 60`、`--context-length 16384`、`--mem-fraction-static 0.85`、
`--max-running-requests 2`，无 parser 参数。完整命令见官方文档 `doc/en/DeepSeek-V4-Flash.md`
Step 2（架构环境变量仍按上文 8.9 设置）。实测首请求 11.3 tok/s、显存 27.4GB、启动 4~5 分钟。

### 3.3 思考模式（thinking）

**V4-Flash 出厂默认不思考**（官方 `encoding/encoding_dsv4.py` 的 `thinking_mode` 默认 `"chat"`），
`--reasoning-parser deepseek-v4` 只负责把输出里的 `<think>…</think>` 拆到 `reasoning_content` 字段，
不是开关。开启方式（优先级：请求参数 > 环境变量）：

| 方式 | 写法 | 说明 |
|---|---|---|
| 全局默认开（本服务已启用） | 启动环境变量 `SGLANG_ENABLE_THINKING=1` | 已写入 `ds4f.service` |
| 单请求开启 | 请求体加 `"chat_template_kwargs": {"thinking": true}` | 不设环境变量时的开启方式 |
| 单请求关闭 | `"chat_template_kwargs": {"thinking": false}` | 覆盖环境变量的全局默认 |
| 推理强度 | `"reasoning_effort": "high"` 或 `"max"` | 只接受这两个值；默认 low（无前缀注入）。也可用环境变量 `SGLANG_REASONING_EFFORT` |

开启后的请求示例（`reasoning_content` 为思考过程，`content` 为最终答案）：

```bash
curl -s -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/var/deepseek-v4-flash/0731",
    "messages": [{"role": "user", "content": "9.11和9.9哪个大？"}],
    "temperature": 0.0,
    "max_tokens": 400,
    "chat_template_kwargs": {"thinking": false}
  }'
```

> 注意：手动方式（不经 systemd，直接跑 3.1 命令）需要自己 `export SGLANG_ENABLE_THINKING=1`
> 才默认思考；systemd 方式由 service 文件注入，无需手动设置。

### 可选：MTP 投机解码（约 1.2× 提速）

在上述命令末尾追加：

```bash
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-moe-runner-backend auto
```

（需额外设置 `SGLANG_FIX_MTP_HC_HIDDEN=1` 与 2604 模式配合，未在本机验证。）

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
    "model": "/var/deepseek-v4-flash/0731",
    "messages": [{"role": "user", "content": "用三句话介绍一下你自己。"}],
    "temperature": 0.7,
    "max_tokens": 200
  }'

# 交互式聊天
kt chat --host 127.0.0.1 --port 30000 --temperature 0.7 --max-tokens 2048
```

2026-08-17 实测（temperature=0，max_new_tokens=128，含首 token 延迟）：约 **11.3 tok/s**，
中英文输出正常。文档标称 5090 为 20+ tok/s，4090D 该成绩符合预期。

## 5. 排障记录（本次构建实际踩过的坑）

| 现象 | 原因 | 解决 |
|---|---|---|
| 启动即崩 `deepseek_v4.py ... raise NotImplementedError` | 未设 `SGLANG_DSV4_MODE`，代码只接受 2601/2604 | `export SGLANG_DSV4_MODE=2604` |
| tilelang JIT 崩 `_NestedLoopCheckVisitor ... no attribute '_inst'` | tilelang 0.1.8 自带 TVM 与 tvm-ffi ≤0.1.11 的 C 层 Object 协议不兼容 | 升级 tilelang 0.1.13 |
| tilelang 导入崩 `attribute '__dict__' of 'type' objects is not writable` | tilelang 0.1.8 + tvm-ffi ≥0.1.12 的 registry 冲突（即文档提到的 TVM FFI 冲突的另一表现） | tilelang 0.1.13 + tvm-ffi 0.1.12（其原生配对） |
| `pip install --upgrade flashinfer-python flashinfer-cubin` 后版本错位（0.6.17 / 0.6.13） | PyPI 上 cubin 最高 0.6.13 | 两个包都显式固定 `==0.6.13` |
| `install.sh` 的 deps 步骤要 sudo 密码 | 只为装 libhwloc-dev / pkg-config，本机已预装 | 跳过 deps，直接跑 `./install.sh sglang` 与 `kt-kernel/install.sh build` |
| pip 解析器报 sglang-kt 依赖不满足（flashinfer/cutlass-dsl 版本警告） | 文档要求升级 flashinfer 突破了 sglang-kt 的 pin（`flashinfer_python==0.6.3`），属预期行为 | 忽略，运行时正常 |
| 开思考后 `reasoning_content` 为 null，思考全文（含 `</think>`）混在 `content` 里 | sglang-kt 0.6.4 的 bug：V4 思考模式把 `<think>` 预填在 **prompt** 里，生成流开头没有 `<think>`；而 `_get_reasoning_from_request()` 只认请求里的 `chat_template_kwargs.thinking`，不认 `SGLANG_ENABLE_THINKING` 环境变量 → 解析器按"必须看到 `<think>` 才算思考"创建，拆分失败 | 打补丁（见下），已验证三种路径全部正常 |

### 5.1 补丁：SGLANG_ENABLE_THINKING 环境变量的解析器适配

**修复位置（源码，已提交）**：`third_party/sglang` 分支 `based-on-bc7f0058f`，
提交 `312c1ee75`（"fix(dsv4): honor SGLANG_ENABLE_THINKING env in reasoning parser decision"），
已推送至 `git@github.com:vproxy-tools/sglang`。
改动文件：`python/sglang/srt/entrypoints/openai/serving_chat.py`（`_get_reasoning_from_request`）。

```python
# 原代码：只看请求参数
return (
    request.chat_template_kwargs is not None
    and request.chat_template_kwargs.get("thinking") is True
)

# 补丁后：请求参数优先，否则回退到环境变量（与 prompt 渲染侧逻辑一致）
if (
    request.chat_template_kwargs is not None
    and request.chat_template_kwargs.get("thinking") is not None
):
    return request.chat_template_kwargs.get("thinking") is True
return envs.SGLANG_ENABLE_THINKING.get()
```

原理：`force_reasoning=True` 会让解析器按"输出开头就在思考中，直到 `</think>`"解析
（DeepSeek-R1 风格），正好匹配 V4 思考模式"prompt 预填 `<think>`、输出只有 `</think>`"的格式。
补丁让环境变量开启的思考也走这条路径。

**生效方式**（sglang-kt 是普通 wheel 安装，改源码不会自动生效，需重装）：

```bash
source .venv/bin/activate
export SGLANG_KT_VERSION=0.6.4
cd third_party/sglang
pip install --no-deps --no-build-isolation ./python
sudo systemctl restart ds4f    # 或 kill 主进程触发 on-failure 自动拉起
```

已验证三种路径全部正常：环境变量默认思考 / 请求显式 `thinking: true` / 请求显式 `thinking: false`。

⚠️ 补丁在 `based-on-bc7f0058f` 分支上；若 submodule 被 checkout 回 `origin/main`（bc7f0058f）
或更新到其他基线，补丁会丢失，需重新应用并重装。父仓库 ktransformers 的 submodule 指针
已指向 `312c1ee75`，记得在父仓库一并提交这个指针变更。

## 6. 服务管理（systemd）

工程根目录提供了 `ds4f.service`（内容即 3.1 节的启动命令 + 必需环境变量），
崩溃自动拉起（30s 延迟、每天最多 12 次，防止异常状态下反复加载 150GB 权重）。

安装（需要 sudo）：

```bash
sudo cp /home/wkgcass/ktransformers/ds4f.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ds4f        # 开机自启 + 立即启动
```

常用操作：

```bash
systemctl status ds4f                  # 状态（启动约 46s 后就绪）
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

> **与第 2 节的区别**：第 2 节是首次完整构建（`./install.sh`，带全部依赖和 extras）。
> 本节是**改完代码后让改动生效**的正确姿势。两者不可混用——见下面的注意点 1。

### 7.1 改 sglang-kt（third_party/sglang，纯 Python 包）

```bash
source .venv/bin/activate
export PIP_CACHE_DIR=/var/pip-cache
export SGLANG_KT_VERSION=0.6.4          # 与 install.sh 行为一致（读 ktransformers/version.py）
cd third_party/sglang
pip install --no-deps --no-build-isolation ./python
sudo systemctl restart ds4f             # 重启后生效（见 7.3）
```

注意点：

1. **不要用 `./install.sh sglang` 重新装**。它执行 `pip install "./python[all]"`，会重新解析
   全部依赖：把 flashinfer 降回 pin 的 0.6.3（覆盖我们调好的 0.6.13）、重装 [all] extras、
   触发 st_attn 等 sdist 源码编译。改码迭代必须用上面的 `--no-deps` 增量安装，
   只替换 sglang-kt 自己，不碰其他包。
2. **`--no-build-isolation`**：省去构建时另建临时环境下载 setuptools 等，快且行为可控。
3. **`SGLANG_KT_VERSION`**：不设的话 wheel 版本号会取 pyproject 默认值，与 install.sh
   装出来的不一致（`pip show sglang-kt` 显示会变）。
4. **改源码不生效是正常的**：sglang-kt 是普通 wheel 安装（非 editable），运行进程加载的是
   site-packages 里的拷贝。改了 `third_party/sglang` 后必须重装 + 重启。
   （反过来：不要直接改 `.venv/.../site-packages/` 下的文件——重启即被绕过、重装即丢失，
   5.1 节的补丁最初就吃过这个亏，最终落到了源码并提交。）
5. **验证安装是否到位**（重启前先做，省一轮重启）：
   ```bash
   # 安装包与源码是否一致（无输出即一致）
   diff third_party/sglang/python/sglang/srt/.../xxx.py \
        .venv/lib/python3.12/site-packages/sglang/srt/.../xxx.py
   ```

### 7.2 改 kt-kernel（含 C++/CUDA 编译）

```bash
source .venv/bin/activate
cd kt-kernel
./install.sh build              # 默认会清 build/ 全量重编（几分钟）
# 频繁迭代可加 --no-clean 保留编译缓存，只重编改动部分
```

kt-kernel 的 requirements（torch==2.9.1）与现环境一致，直接跑不会动依赖。

### 7.3 重启使改动生效

```bash
sudo systemctl restart ds4f
# 无 sudo 时的等价方式：kill 主进程触发 on-failure 自动拉起
pkill -9 -f "sglang.launch_server"     # systemd 会在 30s 后拉起新进程
```

- 重启成本：30s 延迟 + 权重加载 + CUDA Graph，约 80~140s。
- kill 方式受 `StartLimitBurst=12`/天 限制，超了会被 systemd 放弃，
  需 `sudo systemctl reset-failed ds4f && sudo systemctl start ds4f`。
- 改动是否生效看新进程的启动时间：`systemctl status ds4f` 或 `journalctl -u ds4f`。

### 7.4 高频改码的替代：editable 安装

```bash
pip uninstall sglang-kt
cd third_party/sglang && pip install --no-deps -e "./python"
```

之后改 sglang 源码**只需重启服务、无需重装**（python 代码进程启动时重新 import）。
代价：环境从"wheel 拷贝"变成"指向源码树的链接"，别人复现时行为不同；
正式部署前建议切回 wheel 安装（按 7.1 重装即可覆盖回普通模式）。

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
  `wkgcass`），否则旧大页一直占用池子（每节点约 81 GiB）；
- 大页池容量（当前 node0=300、node1=364 个 1 GiB 页）需 ≥ 常驻权重 + 其它大页用户；
- 首次冷加载若中途崩溃，未写标记的层会在下次启动自动重写，无需人工干预。

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

## 9. DSpark / 推测解码（speculative decoding）支持情况

**结论：当前栈（ktransformers + sglang-kt 0.6.4）不支持 DSpark，通用 EAGLE/MTP 替代路径
也走不通（2026-08-17 实测），暂无可用使用方式。**

### 9.1 调查结论

1. **checkpoint 素材齐备**：`/var/deepseek-v4-flash/0731` 的 `config.json` 自带 DSpark
   元数据：`dspark_block_size=5`、`dspark_markov_rank=256`、
   `dspark_target_layer_ids=[40,41,42]`、`dspark_noise_token_id=128799`、
   `num_nextn_predict_layers=1`；权重含完整 3 层 draft（`mtp.0/1/2.*`，结构与主层相同，
   含 256 专家 MoE）和 DSpark 马尔可夫头（`mtp.2.markov_head.markov_w1/w2`）。
2. **sglang-kt 无 DSpark 实现**：整个代码库（源码与 venv 安装副本）没有任何 dspark
   引用。社区的 DSpark 部署基于定制 `dspark-vllm` 镜像（如 DGX Spark 双机 vLLM TP=2
   方案），与本机的 ktransformers CPU 专家栈不兼容。
3. **通用 EAGLE/MTP 路径实测失败**：以下参数启动（整卡显存充足时）：
   ```
   --speculative-algorithm EAGLE \
   --speculative-draft-model-path /var/deepseek-v4-flash/0731 \
   --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3
   ```
   draft 模型（`deepseek_v4_nextn`）权重校验报错：
   `Some weights are not initialized: {model.e_proj.weight, model.h_proj.weight,
   model.enorm.weight, model.hnorm.weight, ...}`。原因：draft 模型期望 MTP 适配层挂在
   `model.e_proj/...` 顶层命名，而本 checkpoint 的 `mtp.*` 布局经
   `deepseek_v4.py::remap_weight_name_to_dpsk_hf_format` 映射后带 `model.layers.43.*`
   前缀，两边不匹配——sglang-kt 对该 checkpoint 布局的 nextn 适配是缺口。
4. **显存**：目标 + draft 共需额外约 2~3GB 显存（draft 的 GPU 侧权重）；生产
   `--mem-fraction-static 0.90` 下有富余，不是障碍。

### 9.2 若未来要启用（移植工作量估计）

本机 decode 两侧都是带宽/延迟型瓶颈（GPU ~440GB/s 显存墙 + CPU MoE ~240µs/层），
推测解码的 verify 一次前向读同样多权重可裁决多个 token，理论收益大（DSpark 官称
单用户加速 60%~85%）——值得做但需要：

1. 修 sglang-kt 的 nextn 权重命名适配（`mtp.*` → draft 模型参数名，涉及
   `remap_weight_name_to_dpsk_hf_format` 的 nextn 分支与 `deepseek_v4_nextn` 的期望
   命名对齐，可能还需处理 draft 层内 attn/ffn 的层号映射）；
2. draft 模型的 MoE 接入 kt CPU 专家路径（draft 层同样有 256 专家，`kt-method MXFP4`
   对 draft 的行为需验证）；
3. verify 批次（bs×draft_tokens）的 CUDA graph 形状适配（当前 `--cuda-graph-bs 1`）;
4. DSpark 本体（置信度调度 + 半自回归块生成 + markov 头）需等上游 sglang/vLLM 实现，
   或整体切换到 dspark-vllm 定制栈（放弃 ktransformers CPU 专家方案，需大显存 GPU）。

启用后的预期参数形态（修好适配后）：
```
--speculative-algorithm EAGLE --speculative-num-steps 2-4 \
--speculative-eagle-topk 1 --speculative-num-draft-tokens 3-5
```
（DSpark 原生 block_size=5。）
