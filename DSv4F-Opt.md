# DSv4F-Opt — DeepSeek-V4-Flash decode 性能优化记录

> 环境：RTX 4090D 48GB（改装卡，显存半速，见 §2.2）+ 2×EPYC 9275F（每 NUMA 24 物理核），
> 全部 256 专家驻留 CPU 巨页，单流 decode（`--max-running-requests 1`）。
> 日期：2026-08-17。

## 1. 成果

| 指标 | 基线 | 第一轮后 | 第二轮后 | 累计提升 |
|---|---|---|---|---|
| 单流 decode 吞吐（稳态，256 token，忽略前 16） | 23.36 tok/s | 30.0 tok/s | **31.2 tok/s** | **+33.6%** |
| 每 token 延迟 | 42.8 ms | 33.4 ms | 32.1 ms | −25% |

测量脚本：`tests/bench_decode.py <max_tokens> <runs> [port]`（流式，排除首 token 与预热）。
所有"每层/每步"数据来自 torch profiler（`/start_profile`，仅统计相邻 decode graph launch 之间的区间）
与 kt-kernel 微基准（`bench/bench_fp4_moe_cold.py`，随机路由击败 L2）。

## 2. 基线剖析（优化前，42.8 ms/token）

### 2.1 时间构成（profiler 实测）
- GPU kernel 执行 ≈ 27 ms/步（占 63%），其中：
  - `_w8a8_block_fp8_matmul`（wq_b/wo_b/共享专家等 FP8 投影）12.6 ms/步
  - cublas bf16 GEMV **`wo_a` einsum 152.7 µs/层**（gemvx [256,1,8]，每层读 64MB bf16）
  - 稀疏注意力本体仅 1.2 ms/步（不是瓶颈）
- CPU MoE（GPU 空闲间隙）≈ 11.7 ms/步 = 40 层 × ~282 µs/层（微基准复现 282 µs）
- CUDA graph 必须开启：eager 模式仅 12–14 tok/s（每层 ~0.23 ms Python/launch 开销）

### 2.2 关键硬件事实：显存带宽墙 ≈ 440 GB/s
48GB 4090D 为改装卡：`nvidia-smi -q -d CLOCK` 显示显存运行于 **5001 MHz（max 10501）**，
持续拷贝实测 ~431 GB/s。因此 GPU 侧一切 GEMV/GEMM 已接近或可达的极限就是 ~440 GB/s，
优化方向 = **减少每 token 读取的字节数**（bf16→FP8 直接减半），而非指望更高带宽。
CPU 侧：MoE 内核冷/热只差 6% → 非纯带宽瓶颈，受 FP4→BF16 解码 ALU 与每线程访存并行度限制
（每线程仅 ~8 GB/s）。

### 2.3 次要发现
- Triton FP8 调优配置按设备名查文件：`4090 D` 与 `4090` 名字不同 → 全部落到默认配置
  （BM=64），`wqkv_a`（N=1536）连默认都没有、且实测默认配置比调优差 60%。
- CPU 线程扫描（每 NUMA）：12:402µs / 16:326 / 20:285 / 22:282 / **24:274** / 28:323 / 48:崩溃。
  SMT 有害，24 线程/NUMA（`--kt-cpuinfer 48`）最优。

## 3. 优化项（按贡献排序）

### 3.1 FP8 wo_a Triton GEMV（~+2.0 tok/s，最大单项）
- 问题：wo_a（[8192,4096] einsum "tgd,grd->tgr"）在 SM89 无 deep_gemm，只能 bf16 → cublas
  GEMV 440 GB/s = 147 µs/层。checkpoint 本身存 FP8+ue8m0 分块 scale，只是加载时被反量化。
- 改动：新文件 `sglang/srt/layers/woa_fp8_triton.py`（Triton kernel：激活 per-128-group
  FP8 量化 + 权重 [128,128] 块 scale，fp32 累加，bf16 输出；GRID (G, R/64, T)，T≤4 走此路，
  大 T prefill 回退 bf16 einsum）；`deepseek_v4.py` 加载路径 stash 原始 FP8 权重
  （`_dequant_fp8_wo_a`），前向 `_ensure_woa_fp8()` 惰性绑定（在 eager 的内存 profiling
  前向完成，早于图捕获）。
- 结果：147→76.7 µs/层（437 GB/s，达带宽墙），每 token 省 ~3 ms。数值与 SM90 deep_gemm
  路径同配方（FP8 量化误差 ~3-4% 相对）。
- 开关：`SGLANG_KT_WOA_FP8_TRITON=0` 关闭（回退 bf16 einsum）。显存 +1.44 GB。

### 3.2 MXFP4 GEMV 内核：VBMI vpermb 解码 + 软件预取（~+1.5 tok/s）
- 问题：FP4→BF16 解码每 32 值需 ~18 条 shuffle/unpack 指令（PSHUFB 路径）；GEMV 每线程
  MLP 不足（~8 GB/s/线程）。
- 改动（`kt-kernel/operators/amx/fp4-moe.hpp`，仅 `__AVX512VBMI__ && __AVX512BF16__` 路径）：
  1. `mxfp4_to_bf16_32` 改用 64B 表 VPERMB（idx = 2×nibble + 字节奇偶），每 32 值 ~10 条指令；
     输出元素序变为 [lo 半|hi 半]，配套在 `ActivationBF16` 构造中加一次
     `_mm512_permutexvar_epi16` 将激活重排为 [偶|奇] 位，两者匹配后 dot 语义不变
     （mat-vec 与 mat-mat 两路径共用同一结构，一致生效）。
  2. 主循环内 `(g&3)==0` 时对 4 行权重+尺度 `_mm_prefetch` +16 组（256B）。
- 结果：282→~240 µs/层（A/B 交替实测 −6% 预取收益；vpermb −11%）。位级等价单测：
  `kt-kernel/tests/test_vpermb_decode.cpp`（对 256 个字节值与随机向量全过）。
- 预取距离扫描：+8/16/24/32 组 → 239.8/234.3/238.3/246.7 µs，取 +16。

### 3.3 Triton FP8 调优配置补齐（wqkv_a −37%，整体 ~+0.5 tok/s）
- `configs/N=1536,K=4096,device_name=NVIDIA_GeForce_RTX_4090_D,...json`（新形状，
  34.2→21.5 µs）+ 其余 7 个 V4-Flash 形状的 4090→4090_D 副本（原本因设备名不匹配全走默认配置）。
- 其余形状离线实测已在带宽墙（wq_b 75µs=447GB/s、wo_b 75µs、gate_up 39µs、down 22µs），
  服务内较高数值（93µs 等）来自运行时干扰。

### 3.4 代码内已有的 opt-in 开关（组合 ~+2.3 tok/s）
`~/.config/sglang/env`（见 §5）启用：
- `SGLANG_OPT_FUSE_WQA_WKV=1`：wq_a+wkv 融合为 4096→1536 单 GEMM（省一对 quant+GEMV）
- `SGLANG_OPT_USE_JIT_NORM=1`：q 头 RMSNorm jit 内核
- `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=1`：kv/compressor/indexer 走旁路流（GPU idle 11.7→2.3 ms/步）
- `SGLANG_OPT_USE_FUSED_STORE_CACHE=1` / `SGLANG_OPT_USE_OVERLAP_STORE_CACHE=1`：融合/重叠 KV 存储
- `SGLANG_OPT_MXFP4_FUSE_RSF_SHARED_ADD=1`：routed_scaling_factor 折入共享专家相加

### 3.5 `--kt-cpuinfer 44→48`（~+0.3 tok/s）
每 NUMA 22→24 线程（见 §2.3 扫描；勿超 24，SMT 反降）。

### 3.6 FP8 lm_head GEMV（第二轮，~+1.3 tok/s）
- 问题：lm_head [129280, 4096] bf16（tie_word_embeddings=False，独立权重）每 token
  读 1.06GB → cublas GEMV 2.27ms/token（grid 32320，profiler 单 kernel 最大项）。
- 改动：`logits_processor.py::_compute_lm_head` 加 `SGLANG_KT_FP8_LMHEAD=1` 分支：
  `woa_fp8_triton.build_lmhead_fp8` 在首次 eager 前向（内存 profiling，早于图捕获）
  分块在 CPU 上量化为 FP8 [128,128] 块（峰值显存只增最终 0.53GB），decode T≤4 走
  `fp8_lmhead_gemv`（复用 wo_a kernel 的 G=1 特例），否则回退 bf16 matmul。
- 结果：2.27→~1.2ms/token。质量验证：八大行星/56/6300 公里/巴黎/光速 299,792,458 米
  均正确。注意两个实现坑：① stash 构建不能放在 T≤4 条件内（捕获前唯一的 eager 前向
  T 很大，否则 stash 永远建不成、图内永久回退）；② 不能整体 `.float()` 量化（2GB
  临时显存会 OOM，需分块 CPU 量化）。

### 3.7 env 覆盖机制（使上述无需改 unit 文件）
- `sglang/srt/environ.py`：import 时读 `$SGLANG_ENV_FILE` 或 `~/.config/sglang/env`，
  KEY=VALUE 逐行 `os.environ.setdefault`（显式环境变量 > 文件 > 默认）。
- `sglang/launch_server.py`：`SGLANG_EXTRA_ARGS` 追加到 argv 尾部（argparse 后者胜），
  用于覆盖 `--kt-cpuinfer` 等 CLI 参数。

## 4. 分阶段验证数字（实验实例，30001 端口）

| 配置 | tok/s |
|---|---|
| 基线（复现生产） | 23.62 |
| + §3.3/§3.4（配置+env 开关） | 25.94 |
| + §3.1（FP8 wo_a） | 27.93 |
| + N=1536 配置 + 48 线程 | 28.27 |
| + §3.2（vpermb+预取，修正表错误后） | 29.69 |
| 最终（复测） | 30.02 |
| 第二轮：+ FP8 lm_head（修正 stash 构建时机与 OOM 后） | 30.84 |
| 第二轮最终（实验复测 / 生产 30000） | 30.65 / **31.20** |

正确性验证：太阳系八大行星介绍正确；`12×11=132`、`长江约6300公里`；
3431-token 长上下文 prefill（走 mat-mat 修改路径）答案正确（合成数据答案 "0,0"）；
vpermb 解码位级单测全过。
（注：vpermb 首版有两处 bug——LUT i=8 项抄错、vpermb 参数顺序反了——靠输出质量检查+单测发现，
已修正；教训是此类位操作改动必须有位级测试。）

## 4.5 输出质量影响评估（A/B 实测）

数值上分三类：
1. **数学等价类**（vpermb 解码+预取、Triton 调优配置、env 开关、48 线程）：只改变浮点
   求和顺序，1-ulp 级舍入差异，不构成质量回退。vpermb 解码本身位级相同（有单测）。
2. **FP8 wo_a**：权重用 checkpoint 自带的 FP8（模型原生量化，非我们引入），新增误差仅
   在激活侧 per-128-group FP8 量化，实测该投影输出相对误差 ~3%。与 SM90 上游
   deep_gemm.fp8_einsum 路径同配方。
3. **FP8 lm_head（最敏感）**：checkpoint 存 bf16，FP8 为加载时量化，logits 相对误差
   ~2-4%，理论上可翻转接近的 top-1。

A/B 对照（8 个固定提示词，贪心，256 token，唯一变量=两个 FP8 开关）：
- 事实类答案全部一致且正确（137×24=3288、120×2.5=300 公里、矩形面积 72cm²、
  大气五层、论语翻译等两侧逐字级一致或等价正确）；
- 平均约 25-30 个 token 后出现首个分歧，全部发生在**开放式续写**（模型自编的
  多轮示例、格式化示例的选择），这正是小幅 logit 扰动的预期特征——确定性强的
  事实 token 不翻转，接近绑定的自由生成 token 偶尔翻转；
- 未做正式评测集（MMLU/困惑度等）对比。若需严格质量保障，可独立关闭：
  `SGLANG_KT_FP8_LMHEAD=0`（损失 ~1.25 tok/s）或 `SGLANG_KT_WOA_FP8_TRITON=0`
  （损失 ~2 tok/s），两者互不影响。

## 5. 生产启用方式

正式配置已显式写入仓库中的 **`ds4f.service`**（`--kt-cpuinfer 48` + 各 `Environment=` 行），
安装到系统：
```
sudo cp $KT_ROOT/ds4f.service /etc/systemd/system/ds4f.service
sudo systemctl daemon-reload && sudo systemctl start ds4f
```
`~/.config/sglang/env` 保留同名 KEY=VALUE 作为兜底（unit 未含对应行时才生效，
语义为 setdefault；进程显式环境变量 > unit > 该文件）。
- 回滚单项：注释 unit 中对应 `Environment=` 行（或删除兜底文件对应行）；
  回滚内核：`cd kt-kernel && git checkout <旧commit> && ./install.sh build`
  （旧栈 `.venv` 流程；**dspark 生产栈**重编必须 `--no-deps`，见 DSv4Flash.md 7.2）。
- 说明：代码侧（venv 内已装）不依赖 env 文件也可运行——`SGLANG_KT_WOA_FP8_TRITON=0`
  时 wo_a 回退 bf16 einsum，其余 SGLANG_OPT_* 默认关闭。

## 6. 第二轮试验记录与未做的后续机会

第二轮额外试验（均已实测）：
- `--num-continuous-decode-steps 4`：29.79 vs 29.59 tok/s，噪声内，无效（overlap 调度器
  已隐藏调度开销），已回退。
- GEMV 8 行分组（提高每线程 MLP）：242-248µs vs 4 行 240µs，寄存器压力抵消收益，已回退。
- 剩余 gemv 群（compressor/线性注意力旁路流，~1.9ms/步求和）：实测与主流并发执行，
  暴露成本远小于求和；但它们的带宽抢占使 wq_b 服务内 93µs vs 离线 75µs。

未做的后续机会：
- ue8m0 scale 以 uint8 存储（省 15% MoE 字节）——但 MoE 实为延迟/MLP-bound（热冷差 6%），
  预期收益有限；需改 BufferB 布局与巨页缓存指纹。
- 旁路流 gemv（bf16/fp32 cublas GemmEx）换 Triton GEMV，减少与主流的带宽抢占。
- ratio-0 的第 0/1 层 wo_a 仍未走 FP8 路径（~0.3 ms/token）。
- CPU MoE ~235µs/层平台期：需 scale-权重交错布局或 SNC 级调优才能突破。
- systemd StartLimitBurst 已改为 500（运维调整）。

## 7. DSpark 投机解码（2026-08-19，主线 sglang 移植）

### 7.1 背景与路线

官方 DeepSeek-V4-Flash-0731 自带 DSpark draft head（config: `dspark_block_size=5`,
`dspark_target_layer_ids=[40,41,42]`, `dspark_noise_token_id=128799`,
`dspark_markov_rank=256`；权重在 `mtp.0.*`，4705 个键），target/draft 同源，只需
`--speculative-algorithm DSPARK`。上游 sglang 主线（sgl-project）在 PR #30261 支持。

kvcache-ai fork（基于 2 月主线）没有 DSpark，因此开新分支 **`dspark-kt`**
（third_party/sglang）：基底取主线 4ad990ba7（2026-08-06，最后一个钉 torch 2.11 的
提交，避开 cu13 依赖墙——驱动 550 只支持 CUDA 12.x），在其上：

- **移植 fork 的 kt CPU 专家引擎**（4477 行 kt_ep_wrapper + mxfp4_deepseek +
  v4_marlin/v4_triton_kernels + quant_method_registry + jit_kernel 包 +
  linear_bf16_fp32 等），主线 V4 模型挂 `_try_kt_plugin` side-effect 注册。
- **pick 我们 3 个仍有效的提交**：巨页缓存、perf pack（environ/4090D 调优 config/
  woa_fp8_triton/EXTRA_ARGS 钩子）、FP8 lm_head GEMV。entrypoint 三个提交
  （thinking env / reasoning_effort max / 官方对齐）被主线原生实现取代，丢弃
  （主线有 SGLANG_DEFAULT_THINKING、REASONING_EFFORT_PROFILES preview/official）。
- **SM89 适配**（主线假设 SM90+/SM120）：paged_mqa_metadata 128KB 动态共享内存按
  设备 optin 上限钳制；sparse decode/prefill 注意力走 fork 的 Triton 回退
  （debug_flash_mla_adapter）；索引器 logits 走 torch 回退；topk v1（v2 用
  SM90 线程块集群）。
- **DSpark draft 保持纯 GPU**：`build_draft_tp_worker` 包进
  `speculative_kt_ep_disabled_context()`，draft 专家不上 CPU（约 10.6GB GPU）。

### 7.2 环境

- 独立 venv：`$DSPARK_VENV`（本机实际路径见 ds4f.service；torch 2.11.0+cu128、
  flashinfer 0.6.15.post1[cu12]、sgl-kernel/sgl-deep-gemm 自 docs.sglang.ai
  cu129 索引、transformers 5.12.1；sglang 为 editable、kt-kernel 为本地构建
  拷贝安装）。旧栈 `.venv`（torch 2.9.1）保留未动，仅作回滚。
- 启停：`run_dspark.sh` / `stop_dspark.sh`（30001 端口）。`DSPARK=1` 开投机，
  默认 cuda graph 开 + `SGLANG_RAGGED_VERIFY_MODE=static`（7.4 修复后正确且
  更快；`EAGER=1` 回退无损 eager）。MEMFRAC：无投机 0.30，DSPARK 需 ≥0.60
  （draft 权重计入预算）。
- 依赖分支状态：third_party/sglang 指针已记录在 optimize-latest（分支 `dspark-kt`，
  venv 内 sglang 为 editable 安装）。kt-kernel 以 torch 2.11 头文件重编
  （`pip install --no-build-isolation --no-deps .`）。
  **2026-08-19/20 更新：dspark 栈已转正为生产**——ds4f.service 直接使用本 venv
  （editable sglang + 拷贝 .so），systemd 部署于 30000。本节撰写时（8/19 实验期）
  生产还是 `.venv` 旧栈，那一状态已作废；旧栈 `.venv` 现仅作回滚用途。

### 7.3 实测（greedy，5 提示词，30001 端口）

| 配置 | 平均吞吐 | accept len / rate | 质量 |
|---|---|---|---|
| 主线基线（kt MXFP4，无投机，cuda graph） | 26.0 tok/s | — | 正确（3288 ✓、散文流畅） |
| DSpark + cuda graph（修复前） | 38.9 tok/s | 恒 2.00 / 0.20 | **损坏**（重复词，见 7.4） |
| DSpark + eager + static | 34.3 tok/s（峰值 38-46） | 2.9-3.3 / 0.38-0.46 | 正确 |
| **DSpark + cuda graph + static（修复后，推荐）** | **39.6 tok/s**（峰值 49.8） | **2.2-3.8 / 正常波动** | **正确** |

- 修复后 graph 配置 4 项 soak（2479 token 连续重负载）：39.57 tok/s 持续，数学
  （水池题 4小时48分 ✓）/散文/翻译/英文总结全部正确，零段错误。
- 对比：vs eager +15%，vs 无投机基线 **+52%**；`tests/bench_dspark.py` 5/5 PASS
  （38.1 tok/s，该 prompt 集 thinking 较短）。
- 首请求含 Triton JIT 预热（~3-6 tok/s），稳态请以第二请求起算。

### 7.4 已修复：verify 的 cuda-graph 回放损坏（2026-08-19 二轮定位 + 修复）

**根因（kt-kernel/python/experts_base.py `KExpertsCPUBuffer.get_buffer`）**：
CPU 专家的 pinned 中转 buffer（input/ids/weights/output/bsz，共 5 类 × 2 slot）由
单槽 temp 缓存管理——sglang 从不调用 `set_capture_batch_sizes`，`capture_bs` 为空，
**所有尺寸都走 temp 路径**。CUDA graph 捕获时（verify 的 6/12/24… token 形状），
录制的 D2H/H2D 拷贝与 `cudaLaunchHostFunc` host node（submit/sync_with_cuda_stream，
CUDA 11.1+ 可捕获为 graph host node，replay 时由 driver 线程重新执行 CPU 专家计算）
都烤死了这些 pinned 裸指针；此后任何一次**不同 batch size 的前向（如 prefill 的
256/2048 chunk）换掉 temp 单槽**，graph 引用的实例失去最后 Python 引用而被 GC，
pinned 块回到缓存分配器，被下一个同尺寸分配**精确复用**（已独立脚本实证地址重合）。
replay 从此读写别人的活跃张量——专家 ID/权重变垃圾 → 确定性输出损坏（重复词、
accept 掉 1.0）；垃圾 ID 无界 → C++ 专家表越界 → 原生段错误（栈无 python 帧）。
任何一次 eager submit 重新分配同尺寸 buffer 落回同地址 → "每步 eager 重跑可治愈"
的假象；请求 2 起（其 prefill 换槽后）必坏也与时间线吻合。

**修复**：get_buffer 在 `torch.cuda.is_current_stream_capturing()` 为真期间分到
（或命中 temp）的尺寸提升进 `capture_buffers` 永久保活；prefill 尺寸（从不捕获）
仍走单槽 temp，内存零增长。修复后：多请求无劣化、4 项 soak 39.57 tok/s 全对、
零段错误，graph 修复收益 +15%。

**遗留的独立小问题（已有 workaround）**：首个 verify（graph 或 eager 均是）返回
的 hidden_states 是未物化的输出 buffer（|x|≈2.9e5）；`dspark_verify.py` 前 2 步
verify 强制 eager 预热（env `SGLANG_DSv4_VERIFY_EAGER_WARMUP`，默认 2）。

**调试技法存档**（代码已清理）：同进程 graph→clone→eager 三连位对比；
`_dbg_full_meta_list` 抓 graph 池内 metadata 张量 replay 后读值；KV 池 uint8
checksum 对比；pinned 指针复用独立复现脚本。

### 7.5 后续机会

- 索引器 logits 从 torch 回退换 tilelang（`SGLANG_OPT_USE_TILELANG_INDEXER=1`，
  需验证 SM89 编译）——当前 torch 回退是 eager 路径的主要 GPU 开销之一。
- 把主线自带的 SGLANG_OPT_FP8_WO_A_GEMM 等消费者级优化在 SM89 上验证开启，
  以及我们移植的 SGLANG_KT_WOA_FP8_TRITON / SGLANG_KT_FP8_LMHEAD
  （environ 已带开关，默认关）。
- SPS 置信度调度表（`--speculative-dspark-sps-table-path`）离线构建。
- kt-kernel C++ 小修：`CPUInfer::sync_with_cuda_stream` 的 `new SyncArgs` 从不
  delete（每层每步 16B 泄漏，长跑约 12MB/h）——**已修（2026-08-20，c082623）**：
  eager 一次性回调自删、图捕获型 args 永生（无条件 delete 曾引发回放 use-after-free
  崩溃循环，事故记录见 DSv4Flash.md 9.5）；回归测试 `tests/sync_leak_check.py`（9.6 节）。
- 生产切换评估：dspark-kt 分支当前基线比生产 fork 新 6782 个主线提交，行为
  差异（SWA 池、调度器、入口）需完整回归后再考虑替换 ds4f。

## 8. 相关文件
- 基准：`tests/bench_decode.py`；线程占用：`tests/thread_util.py`；profiler 分析：`tests/analyze_prof*.py`（均已归入 `tests/`）
- GPU 微基准：`tests/bench_gpu_ops.py` / `tests/bench_gpu_cold.py` / `tests/scan_w8a8_cfg.py`
- CPU MoE 微基准：`kt-kernel/bench/bench_fp4_moe_cold.py`（随机路由，L2 冷）
- 实验实例脚本：`run_exp.sh` / `stop_exp.sh`（与生产并行，30001 端口，共享巨页权重缓存）
- DSpark 实验：`run_dspark.sh` / `stop_dspark.sh`（30001 端口，venv-dspark；
  `KEEP_GRAPHS=1` 开 graph 调试）；基准 `tests/bench_dspark.py`（5 提示词贪心，校验+吞吐）；
  sglang 分支 `third_party/sglang@dspark-kt`
- **测试工具全集（功能/前提/执行/清理/结果解读/通过分界）：见 §9**
  —— 均在 `tests/`：`probe_dspark.py` / `bench_dspark.py` / `grow_probe.py` / `bisect_ctx.sh` /
  `hp_weight_check.py` / `sync_leak_check.py`；启动脚本留在仓库根：`run_dspark.sh`+`stop_sglang.sh`；
  `kt-kernel/bench/bench_fp4_moe*.py`

## 9. 测试工具参考（开发者）

面向开发者；普通用户视角的构建/部署/运行见 `DSv4Flash.md`（其 9.4 节有工具简表并指回本节）。
命令中的 `$KT_ROOT` / `$MODEL_DIR` / `$DSPARK_VENV` 含义见 DSv4Flash.md 开头的路径约定。
通用前提：所有探针/基准都向目标端口发**真实生成请求**（temperature=0 贪心），
默认超时 600–1200s；除特别注明外用系统 python3 即可（只依赖 urllib）。

**GPU 独占规则（重要）**：生产 ds4f(30000) 与实验实例（run_dspark.sh，30001）**不能
同时占 GPU**——2026-08-20 早晨生产部署时实验实例残留 24GB 显存，导致生产 OOM 崩溃
循环 11 次。要跑实验（tests/bisect_ctx.sh / A/B 重启类），先停生产（`sudo systemctl stop
ds4f`，或 kill 主进程靠 systemd 30s 后拉起、注意 StartLimitBurst 预算）；跑完把实验
实例停干净再把生产拉回。探针/基准（不重启服务器）与生产共存没问题。

### 9.1 `tests/probe_dspark.py` —— 快速损坏探针

- **功能**：3 个短生成（数学 12×3、中译英、百字短文），检查数学正确性、翻译可辨、
  思考/正文切分、重复词（dup_score：8+ 字符 chunk 在邻近 60 字内重复次数）。
- **前提**：目标端口有活的 sglang 实例（生产或实验均可）。
- **执行**：`python3 tests/probe_dspark.py [port]`（默认 30001；对生产用 30000）。~1 分钟。
- **清理**：无需。服务端无状态残留（--disable-radix-cache，不污染缓存）。
- **解读**：输出 `CLEAN` 或 `CORRUPT:` + 逐条失败原因（math wrong / translate bad /
  reasoning dup / essay dup 等）。
- **通过分界**：**退出码 0=CLEAN，1=CORRUPT**，可直接接 CI/脚本判断。偶发单条
  失败先重跑一次（贪心下应稳定复现才算真损坏）。

### 9.2 `tests/bench_dspark.py` —— 正确性 + 吞吐基准（5 提示词）

- **功能**：5 个贪心提示词（算术/翻译/事实/作文/代码），逐条硬校验 + 记录
  completion_tokens/耗时，汇总 tok/s。
- **前提**：同上；建议目标实例已跑 warmup（首条请求含 Triton JIT 时会偏慢）。
- **执行**：`python3 tests/bench_dspark.py [port]`。~20–60s（含思考输出的提示词较慢）。
- **清理**：无需。
- **解读**：每条 `[i] PASS/FAIL  N tok / Ts = X tok/s` + 输出前 80 字；末行
  `TOTAL: ... tok/s` 与 `ALL PASS` / `SOME FAILED`。
- **通过分界**：`ALL PASS` = 通过；吞吐参考区间（131072 ctx + cpuinfer 48 +
  单请求）**32–36 tok/s**，低于 30 需查（先看当条 accept 长度：tok/s = accept/周期，
  强依赖提示词的 accept 分布，见 DSv4Flash.md 3.1 的说明，勿直接当回归）。逐条
  FAIL 的判据是硬校验（如乘法结果字符串），与吞吐无关。

### 9.3 `tests/grow_probe.py` —— 长上下文增长探针

- **功能**：单会话逐级加长（默认 20/96/112/120K），第 1 级埋远程暗号 `XK-42Q7`
  （~19.5K 填充文本，每行 ~39 token 已按本分词器校准），每级做暗号回忆 + 新数学题 +
  重复度检测——区分"长程注意力丢失"（暗号丢）与"当步生成损坏"（数学错/复读）。
- **前提**：**ctx=131072 配置**的实例（`run_dspark.sh` 默认；池 135168）。
- **执行**：`python3 tests/grow_probe.py [port] [--stages=20,96,112,120]`。每级 ~1–3 分钟，
  全程 10–20 分钟（输出逐级 flush，可中途看进度）。
- **清理**：无需（radix cache 已禁用，会话结束即释放；服务端 KV 池按页回收）。
- **解读**：每级一行 `[stage NK] prompt~P => PASS/FAIL codeword=.. math=.. dup=..`，
  失败项在下一行给样例。判损坏"出现的实际序列长度"区间取首个 FAIL 级。
- **通过分界**：**全部级 PASS = 通过**。任一级 codeword 丢失而 math 正常 → 长程
  注意力问题；math 错/dup>4 → 当步生成损坏（曾用于定位 >111K verify 损坏，见
  DSv4Flash.md 9.3）。~134K 级被 400 拒绝属预期（超 131072 上限）。

### 9.4 `tests/bisect_ctx.sh` —— context 阈值二分（重启循环）

- **功能**：对给定的一组 ctx 值，逐个以该 ctx 重启 30001 实验实例并跑 9.1 探针，
  输出逐 ctx 的 CLEAN/CORRUPT——损坏只与静态 ctx 配置相关（与实际序列长度无关）
  时定位最快。
- **前提**：**GPU 归实验用**（脚本会反复重启 30001；已改为只杀 cmdline 含
  `--port 30001` 的实例，生产 30000 不受影响，但两者仍不能同时驻留显存——先停生产）。
- **执行**：`./tests/bisect_ctx.sh CTX1 CTX2 ...`（如 `98304 106496 110592 111616`）。
  每个 ctx 启动 ~2 分钟 + 探针 ~1 分钟；全程 = ctx 数 × ~3 分钟。
- **清理**：跑完脚本**不会自动停最后一个实例**——生产要用 GPU 时先
  `./stop_sglang.sh`（会连生产一起杀，慎用；或按 9.7 的方式精确停 30001）。
- **解读**：逐行 `=== CTX=N CLEAN/CORRUPT ===`（CORRUPT 附探针失败明细）；
  `SERVER_FAILED` 表示该 ctx 起不来（OOM 等），看 `/tmp/dspark_bisect.log`。
- **通过分界**：无（是定位工具）。相邻一档 CLEAN、一档 CORRUPT 即损坏阈值边界
  （如 110592 CLEAN / 111616 CORRUPT）。

### 9.5 `tests/hp_weight_check.py` —— 巨页权重缓存冷/热链路验证（CPU-only）

- **功能**：用真实模型某一层走与服务器**完全一致**的 NativeMoE MXFP4 加载路径：
  冷进程做 safetensors 读取 + 转换写入持久 arena + commit 标记；热进程 python
  `check_reusable` 命中 → 跳过 safetensors，C++ 直接 mmap 驻留大页。产物（marker +
  weights.bin 分段）**就是服务器要复用的内容**（layer key/stamp/pfp 与 ds4f 一致）。
- **前提**：`/dev/hugepages/kt_weights` 已存在且当前用户可写（root 一次性
  mkdir+chown，见 DSv4Flash.md 8）；**必须用 dspark venv 的 python**（要 import
  kt_kernel）。
- **执行**：连跑两遍（必须是两个独立进程——同进程第二次 alloc 时 arena cursor 已
  前移，marker 偏移不再相等，测不出复用）：
  `KT_MODEL_DIR=$MODEL_DIR $DSPARK_VENV/bin/python $KT_ROOT/tests/hp_weight_check.py [layer_idx]`
- **清理**：**不需要，也不要清**——marker/大页内容留给服务器复用。只有换模型/换
  布局时才清 `/var/lib/kt-hugepage-weights` 与 `kt_weights/`（DSv4Flash.md 8）。
- **解读**：第 1 遍应见 `[hugepage_weights] layer N tp X: ... allocated in persistent
  hugepages`（转换+commit，单层秒级）；第 2 遍 `[pre-check] check_reusable = True`、
  `weights REUSED from persistent hugepages (safetensors skipped)`、`... REUSED from
  persistent hugepages`、总耗时从数百 ms 掉到 ~60ms。marker 可核对
  `pfp=9bcd0b02fd234216`（0731 模型 + AMXFP4_KGroup_MOE 的参考部署期望值；
注意 pfp 的 stamp 含模型目录 realpath——**换路径部署后期望值会变**，以当次
冷加载 commit 出的 marker 为准）。
- **通过分界**：**第 2 遍打印 REUSED 且 check_reusable=True = 通过**；第 2 遍仍
  allocated（冷）= 复用链断（marker/指纹不一致，删除标记重跑）；日志完全没有
  `[hugepage_weights]` 行 = 缓存未启用（目录缺失或 KT_HUGEPAGE_WEIGHTS=0）。

### 9.6 `tests/sync_leak_check.py` —— SyncArgs 泄漏 / 图回放 UAF 回归

- **功能**：双路回归 kt-kernel `CPUInfer::sync_with_cuda_stream` 的修复（DSv4Flash.md
  9.5）：eager 路 100 万次调用后 malloc_trim RSS 增长应≈0；含 4 个 sync host 节点的
  cuda graph 5000 次回放应无崩溃无增长（捕获型 args 永生、回放零分配）。
- **前提**：GPU 可用（占用 <1.5GB，**与生产共存安全**）；dspark venv python（编译进
  venv 的 .so 才是被测对象——先按 DSv4Flash.md 7.2 重编并装入 venv 再测源码改动）。
- **执行**：`$DSPARK_VENV/bin/python $KT_ROOT/tests/sync_leak_check.py`，~2 分钟。
- **清理**：无需（纯本地进程）。
- **解读**：`[eager] ... (X B/次)` 与 `[graph] ... 无崩溃`；测量陷阱：紧循环 RSS 读数
  含"已 free 未归还"的驻留页（纯 C 跨线程 malloc/free 模式本身 ~28B/次），所以
  **必须看 malloc_trim 之后**的数字——脚本已内置 trim。
- **通过分界**：末行 **PASS**（eager ≤8 B/次、graph 增长 ≤64MB、无崩溃）退出码 0；
  FAIL 或中途 Segmentation fault = 不通过（后者=图回放 UAF 回来了）。

### 9.7 `run_dspark.sh` / `stop_sglang.sh` —— 实验实例启停

- **功能**：`run_dspark.sh` 在 30001 起 dspark 实验实例（独立 venv，不动生产），
  环境变量覆盖：`CTXLEN`（默认 131072）、`MAXTOK`（135168）、`MEMFRAC`（DSPARK 下
  默认 0.60）、`KT_CPUINFER`（48）、`PREFILL`（512）、`EAGER=1` 回退无损 eager；
  `stop_sglang.sh` 停**所有** sglang 进程（含生产！）。
- **前提**：GPU 独占规则（见节首）。要精确只停 30001：按 cmdline 过滤
  `--port 30001` 后 kill（tests/bisect_ctx.sh 里的写法可抄）；要停生产交给 systemd。
- **执行**：`DSPARK=1 CTXLEN=... ./run_dspark.sh`（前台日志；或 nohup 重定向）；
  就绪判定 `grep "fired up and ready to roll"`，全量启动 ~2 分钟。
- **清理**：实验完停实例即完成清理；日志在自选的重定向文件。
- **解读/通过分界**：启动脚本非测试工具；就绪后接 9.1/9.2 判定。

### 9.8 `kt-kernel/bench/bench_fp4_moe.py`（+ `_cold` 变体）—— CPU MoE 微基准

- **功能**：合成 V4-Flash 形状（E256/H4096/I2048/top6/g32）的 MXFP4 权重，测 AMX MoE
  逐层内核 µs 级性能；`--routing balanced|concentrated` 控制专家命中分布，
  `--m-list` 批次列表；`_cold` 变体按 token 轮换随机路由（模拟真实 L3 冷访问）。
  结果追加进 `bench/bench_fp4_moe.jsonl`（含 git commit 与时间戳，作历史基线）。
- **前提**：无 GPU 需求（CPU-only，**与生产共存安全**）；用 dspark venv python。
  机器健康对照时注意**同负载条件**——服务器在跑（尤其加载权重期）或其它吃核任务
  会显著抬高 M=1（曾把 233µs 读成 321µs）。
- **执行**：`cd kt-kernel && ../../venvs路径/bin/python bench/bench_fp4_moe.py --m-list 1,4`
  （cold：`bench_fp4_moe_cold.py [--iters 300]`）。
- **清理**：无需（jsonl 历史有意保留；脚本已强制 `KT_HUGEPAGE_WEIGHTS=0`，
  合成权重不会碰持久巨页 arena）。
- **解读**：每行 `M=  N  per-iter= X us  tok/s=`；历史基线 M=1 ≈ **227–235µs**
  （2026-08-17/19/20 三日一致）。
- **通过分界**：M=1 落在历史 ±5% 内 = 机器/内核健康；显著偏高先查负载再查回归
  （对照 jsonl 里最近记录的 commit 定位改动）。
