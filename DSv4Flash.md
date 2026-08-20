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

### 3.1 本机调优配置（DSpark 投机解码 + cuda graph，日常使用）

2026-08-19 起生产切换到 DSpark 投机解码栈（sglang `dspark-kt` 分支，详见第 9 节）。
启动命令与 `ds4f.service` 一致：

```bash
cd /home/wkgcass/ktransformers

# 架构变量：文档示例是 5090(SM_120)，4090/4090D 要改成 8.9
export FLASHINFER_CUDA_ARCH_LIST=8.9
export TORCH_CUDA_ARCH_LIST="8.9+PTX"

# 思考模式默认开启（fork 时代的 SGLANG_ENABLE_THINKING 已废弃）
export SGLANG_DEFAULT_THINKING=1

# DSpark + SM89 回退栈必需（缺一不可，见 DSv4F-Opt.md §7）
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1

/var/deepseek-v4-flash/venvs/dspark/bin/python -m sglang.launch_server \
  --host 0.0.0.0 --port 30000 \
  --model /var/deepseek-v4-flash/0731 \
  --kt-weight-path /var/deepseek-v4-flash/0731 \
  --kt-method MXFP4 \
  --kt-num-gpu-experts 0 \
  --kt-cpuinfer 44 \
  --kt-threadpool-count 2 \
  --tensor-parallel-size 1 \
  --context-length 131072 \
  --attention-backend flashinfer \
  --mem-fraction-static 0.60 \
  --max-total-tokens 135168 \
  --chunked-prefill-size 512 \
  --max-prefill-tokens 512 \
  --max-running-requests 1 \
  --watchdog-timeout 1200 \
  --disable-shared-experts-fusion \
  --trust-remote-code \
  --disable-radix-cache \
  --skip-server-warmup \
  --speculative-algorithm DSPARK \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4
```

2026-08-19 实测（DSpark + cuda graph + KV 池右移）：**39.6 tok/s** 持续
（峰值 49.8，accept 2.2-3.8），比无投机基线 26.0 **+52%**；显存约
**24.5GB / 48GB**（KV 池右移后；右移前 30.7GB），就绪约 80~90s（巨页权重
缓存命中时）。同日放开到 131072（9.3 节修复，`--max-total-tokens 135168`），
显存约 24.4GB。

**吞吐口径澄清（2026-08-19 深夜复测）**：DSpark 下 tok/s = accept/周期，
高度依赖提示词内容——数数类（高可预测）70.8 tok/s、bench 5 题混合
32.5~35.1（ALL PASS）、单一散文题 soak ~28（该类内容 accept 仅 ~2.3）。
机器本身无损：MXFP4 MoE 微基准 234.2µs/层，与 8/17 基线 233.5µs 逐微秒
持平；GPU 时钟正常；QEMU 虚机（node0、~1.2 核）对 node1 专家核与 MoE
无可测影响。不同日子的"持续 tok/s"差异主要来自 soak 提示词的 accept
分布，而非性能回退。

参数说明（与旧栈的差异）：

| 参数 | 值 | 说明 |
|---|---|---|
| `--speculative-algorithm DSPARK` | 0731 自带 draft（`mtp.0.*`） | 无需 draft path；**不要用 EAGLE**（9.1 节） |
| `--context-length 131072` | 模型上限 128K | 需 `SGLANG_DSv4_VERIFY_META_OUT_OF_GRAPH=1`（9.3 节修复，run_dspark.sh 已默认导出）；须是 page_size(256) 倍数 |
| `--chunked-prefill-size 512` | 131072 必需 | 513 页宽下 torch indexer gather 的 prefill 瞬时峰值 ~17GB（2048 chunk）会 OOM；512 后峰值 ~4-7GB。须配合 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。仅影响 prefill 速度，decode 不变；90K 以下 prompt 可用 1024 |
| `--max-total-tokens 135168` | context + 4096 余量 | KV 池右移：0.60 mem-frac 默认分到 801536 token（单请求永远用不满），右移后省 ~6GB 且**无性能差异**（同机 A/B：43.1 vs 42.5 tok/s，噪声内）；覆盖 context 时同步调大（256 倍数） |
| `--mem-fraction-static 0.60` | draft 权重 ~10.6GB 计入预算 | 0.90 会 graph 捕获失败；池大小由 max-total-tokens 决定后此值仅作预算校验 |
| `--kt-cpuinfer 48` | 与旧栈一致 | 2026-08-19 复验：bench 32.9/35.1（44 时 32.5/32.8），≥44，噪声内偏正 |
| cuda graph | 默认开 | kt-kernel pinned-buffer 修复后正确（DSv4F-Opt.md §7.4）；前 2 步 verify 自动 eager 预热 |
| 不再需要 | `SGLANG_DSV4_MODE/2604_SUBMODE` | 新栈从 config 读（swiglu_limit）；`--kt-gpu-prefill-token-threshold`、`--cuda-graph-bs 1` 也不再需要 |

注意事项：

- **venv 是 `venvs/dspark` 且 sglang 为 editable**（指向 third_party/sglang 的
  `dspark-kt` 分支检出）。运行期间不要切 sglang 分支；改分支=改生产代码。
- **长 prompt 首 token 延迟是分钟级**：108K 上下文 ÷ 2048 分块 ≈ 54 次前向，
  且专家全在 CPU（实测 4168-token prompt prefill+回答 14.8s）。
- `--max-running-requests 1` 保持；要并发需同步评估 draft/显存后重测。
- 旧栈（.venv + fork）保留可回滚，回滚步骤见 ds4f.service 文末注释块。

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
| 全局默认开（本服务已启用） | 启动环境变量 `SGLANG_DEFAULT_THINKING=1` | 已写入 `ds4f.service`；输出侧的 `</think>` 切分依赖 dspark-kt 分支对 `serving_chat.py` 的补丁（explicit_thinking 探测模式下输出侧原本忽略该 env） |
| 单请求开启 | 请求体加 `"chat_template_kwargs": {"thinking": true}` | 不设环境变量时的开启方式 |
| 单请求关闭 | `"chat_template_kwargs": {"thinking": false}` | 覆盖环境变量的全局默认 |
| 推理强度 | `"reasoning_effort": "low"/"medium"/"high"/"xhigh"/"max"` | 按官方文档映射：low→low（简洁思考前缀），medium/xhigh→high，high→high（默认档），max→max（最大强度前缀）。也可用环境变量 `SGLANG_REASONING_EFFORT` |
| 官方开关写法 | `"thinking": {"type": "enabled"}` 或 `{"type": "disabled"}` | 与 `chat_template_kwargs.thinking` 等效，对齐 api-docs.deepseek.com |
| Anthropic 风格 | `"reasoning": {"effort": "none"}` / `{"effort": "max"}` | effort=none 关思考；也接受 `"enabled": true/false` |

> 注意：fork 时代的 `SGLANG_ENABLE_THINKING` 在新栈已废弃（新栈读
> `SGLANG_DEFAULT_THINKING`，且输出侧切分要求上面提到的 serving_chat 补丁）。

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
>
> **2026-08-19 起**：生产栈 venv 是 `/var/deepseek-v4-flash/venvs/dspark`（不再
> 是 `.venv`）。其中 sglang 是 editable 安装（改 `third_party/sglang` 源码只需
> 重启服务）；kt-kernel 是拷贝安装（改后需按 7.2 重编并把产物同步进该 venv）。
> 以下命令中的 venv 激活请按目标栈选择。

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

**2026-08-20 现状：缓存未接通（数据目录缺失）。** `/dev/hugepages`（hugetlbfs 挂载点）为
root:root 755，8/19 08:58 虚机启动时挂载点被重建，`kt_weights/` 目录消失，普通用户无法
自行创建 → C++ `hugepage_weights::enabled()` 静默回落堆分配（日志无任何 hugepage 行）。
代码钩子本身完好（C++ alloc/commit/reuse + python `check_reusable` 快路径都在，与
MXFP4 layerwise prefill 的 host 写出兼容——它读转换后的 BufferB）。重新接通只需 root 一次性：

```bash
sudo mkdir -p /dev/hugepages/kt_weights && sudo chown wkgcass:wkgcass /dev/hugepages/kt_weights
# 陈旧标记已清理（2026-08-20）；下次启动冷加载约 1-2 分钟/全模型，之后 REUSED 秒级
```

目录建好后可先离线验证（不碰 GPU、可与生产共存，产物直接被服务器复用）：

```bash
cd /var/deepseek-v4-flash/venvs/dspark
./bin/python /home/wkgcass/ktransformers/hp_weight_check.py 0   # 第一遍:冷(转换+commit)
./bin/python /home/wkgcass/ktransformers/hp_weight_check.py 0   # 第二遍:热(打印 REUSED)
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

## 9. DSpark / 推测解码（speculative decoding）支持情况

**结论（2026-08-19 更新）：已支持并在生产启用。** sglang 换到主线基底 + kt 引擎
移植的 `dspark-kt` 分支（third_party/sglang，基底 4ad990ba7），DSpark 投机解码
+ kt CPU 专家 + cuda graph 全部打通：**39.6 tok/s**（无投机基线 26.0，+52%；
eager 模式 34.3）。`--speculative-algorithm DSPARK` 一个参数即可，0731 自带
draft（`mtp.0.*`，4705 个权重键；config: block_size=5, markov_rank=256,
target_layers=[40,41,42]）。**不要用 EAGLE**（见 9.1）。

### 9.1 通用 EAGLE/MTP 路径仍不可用

`--speculative-algorithm EAGLE --speculative-draft-model-path ...` 启动后 draft
模型（`deepseek_v4_nextn`）权重校验报错（`model.e_proj.weight` 等未初始化）：
draft 期望 MTP 适配层挂在 `model.e_proj/...` 顶层命名，而本 checkpoint 的
`mtp.*` 布局映射后带 `model.layers.43.*` 前缀。DSpark 路径不受影响（走
`deepseek_v4_dspark` 专用模型，布局对齐）。

### 9.2 栈结构与关键修复

- venv：`/var/deepseek-v4-flash/venvs/dspark`（torch 2.11.0+cu128、flashinfer
  cu12 系、sgl-kernel/sgl-deep-gemm cu129 索引；sglang editable 指向
  third_party/sglang@`dspark-kt`，kt-kernel 以 torch 2.11 头文件重编）。
  生产旧栈 `.venv`（fork）保留可回滚。
- SM89 适配：sparse 注意力走 Triton 回退、索引器 logits 走 torch 回退、
  topk v1、paged_mqa_metadata smem 钳制（详见 DSv4F-Opt.md §7）。
- draft 保持纯 GPU（约 10.6GB，MEMFRAC 需 ≥0.60），target 专家全在 CPU。
- **kt-kernel pinned-buffer 生命周期修复**（graph 损坏的根因）：CPU 专家的
  pinned 中转 buffer 曾走单槽 temp 缓存，graph 捕获后任何 prefill 换槽都会
  释放并被复用，replay 读写别人内存。修复后 graph 默认开启（DSv4F-Opt.md 7.4）。
- **输出侧 thinking 切分补丁**（serving_chat.py）：V4 模板的
  explicit_thinking 探测模式下，`SGLANG_DEFAULT_THINKING=1` 只影响 prompt 侧、
  输出侧不切 `</think>`（全部落 content）。补丁让输出侧同样遵循
  请求 > env 的优先级。

### 9.3 已修复：verify 元数据图外构建，context 放开到 131072

**症状（2026-08-18 发现）**：DSpark + cuda graph 下，`--context-length` 超过
约 111.4K 后生成确定性损坏（重复短语、远程注意力劣化；实际序列长度无关——
ctx=131072 下 4K 短 prompt 也坏）。二分边界：111360 ✅ / 111616 ❌。

**排除项**（均实测）：实际序列长度（短 prompt 也坏）、KV 池容量（池 114688
与 135168 都出现过干净/损坏组合）、page 宽度（111360 与 111616 同为 437 页）、
draft 图（draft 图开 + verify eager = 干净）、verify 元数据数值（图回放后逐字段
对比 eager 重建，page_table/swa/c4/c128/positions 全等）、压缩计划字节
（plan_c/plan_w 全等）、两条元数据构建路径互比（raw 路径 vs _old 路径全等）。

**根因（定位到机制层面）**：`SGLANG_PREP_IN_CUDA_GRAPH=1`（默认）把 verify 的
raw→full 元数据构建（`make_forward_metadata_from_raw_verify` 一族 triton/torch
算子）**录制进 verify cuda graph**。录制后所有可读产物都正确，但生成损坏——
即损坏源于"构建被录制"这一形态本身（疑似捕获期内存池别名/算子时序效应，
回放后值正确、前向中途被污染），在 req_to_token 宽度超过 ~437 页时触发。
`SGLANG_PREP_IN_CUDA_GRAPH=0`（全局图外）可修但有 `.tolist()` CPU 同步。

**修复**：新增 `SGLANG_DSv4_VERIFY_META_OUT_OF_GRAPH=1`——仅对 TARGET_VERIFY
bucket，把 raw→full 升级移到图外按步执行（同一组纯 GPU 构建器，无 CPU 同步），
图内只录模型层；draft decode 保持图内快速路径。改动：
`deepseek_v4_backend.py`（out_graph TARGET_VERIFY 分支）+ `environ.py`。
run_dspark.sh 已默认导出。

**验证（2026-08-19，ctx=131072 + 池 135168 + chunk 512）**：probe CLEAN；
bench 5/5 PASS（32.8 tok/s）；长上下文增长探针单会话 19.5K / 109K /
**125,211 token 三级全 PASS**（位置 ~10 埋的远程暗号在 125K 上下文仍可回忆、
新数学题正确、零复读；第四级 ~134K 超出 131072 被 400 拒绝，属预期）。
性能：同机 A/B（当日有 QEMU 虚机抢 CPU）图内 27.8 vs 图外 28.2 tok/s——
**修复零回退**；当日绝对值 ~27-33 与历史 39.6 的差距全部来自虚机竞争
（虚机停后可复测）。另：>100K 的 prefill 需 chunk 512 + expandable_segments
（513 页宽 gather 峰值），已写入 run_dspark.sh / ds4f.service。

### 9.4 验证与工具

- 正确性/吞吐：`bench_dspark.py`（5 提示词贪心）；快速探针：`probe_dspark.py`
  （数学/翻译/重复词检测，退出码 0=干净）。
- 长上下文：`grow_probe.py`（单会话逐级加长 20/96/112/120K，含位置 ~10 埋的
  远程暗号回忆、新数学题、重复度检测；`--stages=` 可自定义）。
- ctx 阈值二分：`bisect_ctx.sh CTX1 CTX2 ...`（逐 ctx 重启+短探针，判
  CLEAN/CORRUPT；损坏与实际序列长度无关时尤其快）。
- 实验实例：`run_dspark.sh`（30001 端口，CTXLEN/MAXTOK/MEMFRAC/PREFILL/
  EAGER 环境变量可覆盖；`EAGER=1` 回退无损 eager）；`stop_sglang.sh` 安全
  清理所有 sglang 进程（避免 pkill 自匹配；**会连 30000 生产一起停**）。
- 以上工具的完整说明（执行前提、结果解读、通过/不通过分界、清理要求）面向
  开发者，见 `DSv4F-Opt.md` §9；本节仅速查。

### 9.5 SyncArgs 泄漏修复与图捕获 use-after-free 事故（2026-08-20）

`kt-kernel/cpu_backend/cpuinfer.h` 的 `sync_with_cuda_stream()` 每次 `new SyncArgs`
从不释放（实测 ~32 B/次 ≈ 生产 ~12 MB/h）。修复时踩过一个必须记录的坑：

**第一版修复（回调里无条件 `delete args`）导致生产 SIGSEGV 崩溃循环。** 根因：
decode/verify 的 cuda graph **捕获期间**也调用 `sync_with_cuda_stream`——
`cudaLaunchHostFunc` 连同 `args` 指针被录成图内 host 节点，**每次图回放都用同一
指针重跑回调**。首次回放 delete 后，第二次回放变成 use-after-free + double-free →
堆损坏 → 主线程死在 `pthread_mutex_lock`（faulthandler 栈可见）。也就是说，原代码
的"泄漏"里有一部分是**被捕获图的函数性需求**（args 必须永生）。

**正确修复**（已部署 ds4f，三级验证通过）：启动时 `cudaStreamIsCapturing()` 探测——
eager 一次性回调标记 `owned=true` 回调自删；捕获流标记 `owned=false` 永生（回放零分配，
只在捕获时分配一次，量级为启动期常数）。验证：eager 1M 次 RSS 增长 -0.9 B/次
（malloc_trim 后）；图 5000 次回放 × 4 节点无崩溃；生产 probe CLEAN + bench
5/5 PASS 33.54 tok/s；MoE 微基准 M=1 227µs（基线 233µs）无回退。
诊断要点：紧循环 RSS 读数含"已 free 未归还"的驻留页（纯 C 跨线程 malloc/free
模式本身读出 ~28 B/次），**必须 malloc_trim 后再测**；CPU-only `sync()` 路径因
指针不逃逸被编译器优化掉分配，泄漏只在 CUDA 路径。

## 10. 旧栈 8 个 perf env 在新栈（dspark-kt）的复验（2026-08-20）

旧栈（.venv / sglang-kt 主线基底）为性能设过 8 个 env。逐个在新栈代码审计 +
可执行处实测后的结论——**一个都不要搬，4 个默认已开、2 个已废弃、2 个开启有害**：

| 旧栈 env | 新栈状态 | 结论 |
|---|---|---|
| `SGLANG_OPT_FUSE_WQA_WKV=1` | `environ.py:1303` 默认 `EnvBool(True)` | 冗余，不设即开 |
| `SGLANG_OPT_USE_JIT_NORM=1` | `environ.py:1316` 默认 `EnvBool(True)` | 冗余，不设即开 |
| `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=1` | `environ.py:1317` 默认 `EnvBool(True)`（仅 HIP 分支强制关） | 冗余，不设即开 |
| `SGLANG_OPT_USE_FUSED_STORE_CACHE=1` | `environ.py:1315` 默认 `EnvBool(True)` | 冗余，不设即开 |
| `SGLANG_KT_WOA_FP8_TRITON=1` | 新栈无任何读取点（仅 environ.py 定义+docstring） | 已废弃 |
| `SGLANG_OPT_USE_OVERLAP_STORE_CACHE=1` | 新栈无任何读取点 | 已废弃 |
| `SGLANG_OPT_MXFP4_FUSE_RSF_SHARED_ADD=1` | 默认 False；新栈**无融合消费方**，=1 只是跳过 `output.mul_(rsf)` | **禁用**：模型 rsf=1.5，=1 会静默丢掉 ×1.5，三处 GPU MoE 路径（marlin/triton_kernels/trtllm）全部算错 |
| `SGLANG_KT_FP8_LMHEAD=1` | 读取点在 `logits_processor.py:788`（head.weight BF16 129280×4096 满足条件） | **禁用**：见下 |

`SGLANG_KT_FP8_LMHEAD` 实测细节（单元级，未动生产）：FP8 GEMV 内核数学正确
（vs 手工反量化参照误差 0.037），T=1 提速 **1.96×**（2242→1143µs，读带宽减半），
但 T=2 持平、T=4 反而 0.5×（einsum 按 T 逐行读权重），DSpark 下仅 draft 步受益。
决定性否决点是 **stash 构建有数据竞争**：`build_lmhead_fp8` 的
`weight[...].to("cpu", non_blocking=True)` 后立即在 CPU 上量化，同一权重两次构建
产物**比特不同**且权重重建误差 ~0.007（健康 fp8 应为 ~0.0003，差 20×），会把
logits 静默算坏。若未来要启用：先给 D2H 后补 `torch.cuda.synchronize()`，复测
重建误差回到 ~2-3% 再说。旧栈当时开着此 env，质量影响未评估（存疑）。
ds4f.service 回滚块中的旧栈 env 清单已同步加注（FUSE_RSF 标注禁止、两个废弃）。
