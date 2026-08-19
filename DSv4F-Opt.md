# DSv4F-Opt — DeepSeek-V4-Flash decode 性能优化记录

> 环境：RTX 4090D 48GB（改装卡，显存半速，见 §2.2）+ 2×EPYC 9275F（每 NUMA 24 物理核），
> 全部 256 专家驻留 CPU 巨页，单流 decode（`--max-running-requests 1`）。
> 日期：2026-08-17。

## 1. 成果

| 指标 | 基线 | 第一轮后 | 第二轮后 | 累计提升 |
|---|---|---|---|---|
| 单流 decode 吞吐（稳态，256 token，忽略前 16） | 23.36 tok/s | 30.0 tok/s | **31.2 tok/s** | **+33.6%** |
| 每 token 延迟 | 42.8 ms | 33.4 ms | 32.1 ms | −25% |

测量脚本：`bench_decode.py <max_tokens> <runs> [port]`（流式，排除首 token 与预热）。
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
sudo cp /home/wkgcass/ktransformers/ds4f.service /etc/systemd/system/ds4f.service
sudo systemctl daemon-reload && sudo systemctl start ds4f
```
`~/.config/sglang/env` 保留同名 KEY=VALUE 作为兜底（unit 未含对应行时才生效，
语义为 setdefault；进程显式环境变量 > unit > 该文件）。
- 回滚单项：注释 unit 中对应 `Environment=` 行（或删除兜底文件对应行）；
  回滚内核：`cd kt-kernel && git checkout <旧commit> && ./install.sh build`。
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

- 独立 venv：`/var/deepseek-v4-flash/venvs/dspark`（torch 2.11.0+cu128、
  flashinfer 0.6.15.post1[cu12]、sgl-kernel/sgl-deep-gemm 自 docs.sglang.ai
  cu129 索引、transformers 5.12.1；sglang 与 kt-kernel 均 editable/本地构建）。
  生产 `.venv` 完全不动。
- 启停：`run_dspark.sh` / `stop_dspark.sh`（30001 端口）。`DSPARK=1` 开投机，
  默认 `--disable-cuda-graph` + `SGLANG_RAGGED_VERIFY_MODE=static`（见 7.4）。
  MEMFRAC：无投机 0.30，DSPARK 需 ≥0.60（draft 权重计入预算）。
- 依赖分支状态：third_party/sglang 指针已记录在 optimize-latest（分支 `dspark-kt`，
  venv 内 sglang 为 editable 安装）。kt-kernel 以 torch 2.11 头文件重编
  （`pip install --no-build-isolation --no-deps .`）。注意：生产 ds4f 仍用
  `.venv` 内的 fork 拷贝，不受此指针影响；从本分支全新构建会得到 dspark 实验栈。

### 7.3 实测（greedy，5 提示词，30001 端口）

| 配置 | 平均吞吐 | accept len / rate | 质量 |
|---|---|---|---|
| 主线基线（kt MXFP4，无投机，cuda graph） | 26.0 tok/s | — | 正确（3288 ✓、散文流畅） |
| DSpark + cuda graph（ragged 默认） | 38.9 tok/s | 恒 2.00 / 0.20 | **损坏**（重复词） |
| DSpark + cuda graph + static | ~38 tok/s | 恒 2.00 / 0.20 | **损坏** |
| **DSpark + eager + static（推荐）** | **34.3 tok/s**（峰值 38-46） | **2.9-3.3 / 0.38-0.46** | **正确** |

- DSpark(eager,static) 比无投机基线 **+32%**（34.3 vs 26.0 tok/s，5 提示词 1812
  token 总平均；decode 峰值 46 tok/s 出现在 accept 高的数学题上）。
- 首请求含 Triton JIT 预热（~3-6 tok/s），稳态请以第二请求起算。
- 2026-08-19 复验（`bench_dspark.py`，5 提示词 702 token）：eager+static 5/5 PASS、
  输出正确，30.6 tok/s（该 prompt 集思考更短/accept 略低，非回退）。

### 7.4 已知问题：verify 的 cuda-graph 回放损坏（SM89 回退栈，2026-08-19 二轮深挖）

现象：开图时长生成出现系统性重复词（"散文 散文"），accept len 恒 2.00；关图后同样
提示词完全正确。同进程 A/B（graph 先跑、结果 clone 后 eager 重跑同输入）证明：

1. **graph 重放的数值计算完全正确**——logits 与 hidden_states 对 eager 逐位相等
   （maxdiff=0.0000，连续 120+ 步）；以下每步状态也逐位一致（graph 后 vs eager 后
   checksum 相同）：SWA KV、c4/c128 压缩 KV、全部注意力 metadata（seq_lens_casual/
   swa_lens/page_table/swa_out_cache_loc，均随 replay 正确刷新）。
2. **首个 verify（graph 或 eager 均是）返回的 hidden_states 是未初始化池内存**
   （|x|≈2.9e5），commit_hidden 会把垃圾注入 draft KV——已加 workaround：前 2 步
   verify 强制 eager 预热（`dspark_verify.py`，env `SGLANG_DSv4_VERIFY_EAGER_WARMUP`，
   默认 2，`--disable-cuda-graph` 时自动失效）。修掉它之后首个请求完全正确。
3. **残余 bug 与数值无关、确定性**：纯 graph 连续 replay 约 15-20 步后输出开始劣化
   （token 重复、accept 掉到 1.0），两种配置下损坏输出逐字节相同；偶发原生段错误
   （faulthandler 栈为纯 native 线程，无 python 帧——指向 kt C++ 线程池）。每步
   跟一次 eager 重跑（副作用）即可永久保持正确 → 缺的不是计算而是**每步一次的
   eager submit**。
4. 已排除：draft 侧 graph（禁 draft 图后损坏不变）、kt 的 `_cpu_stream` 跨流
   （`SGLANG_KT_HYBRID_NO_CPU_STREAM=1` 无效）、FP8 LUT、staging buffer 轮换、
   metadata copy_ 链。

结论：残余根因在 **kt-kernel C++ wrapper 的 replay 记账**——`kt_ep_wrapper.apply`
的 CPU 专家 submit/sync 是宿主侧调用；纯 replay 期间 python 不执行，C++ 侧的每次
submit 才推进的内部状态（任务队列/缓冲索引）在连续 replay 下与录制的 D2H/H2D
不同步。修复需要 kt-kernel C++ 侧支持 replay 触发（或在 verify 步 graph 外重放
submit），属 kt 内核工程，暂以 `--disable-cuda-graph` 规避（eager 无损）。修好后
预计 ~38-39 tok/s（再 +15%）。调试技法（已从代码移除、此处留档）：同进程
graph→clone→eager 三连对比；`_dbg_full_meta_list` 抓 graph 池内 metadata 张量
replay 后读值；KV 池 uint8 checksum 对比。

### 7.5 后续机会

- 修 verify 图残余 bug：在 kt-kernel C++ wrapper 加 replay 触发路径（见 7.4 结论）。
  修好 + KEEP_GRAPHS=1 实测对比。
- 索引器 logits 从 torch 回退换 tilelang（`SGLANG_OPT_USE_TILELANG_INDEXER=1`，
  需验证 SM89 编译）——当前 torch 回退是 eager 路径的主要 GPU 开销之一。
- 把主线自带的 SGLANG_OPT_FP8_WO_A_GEMM 等消费者级优化在 SM89 上验证开启，
  以及我们移植的 SGLANG_KT_WOA_FP8_TRITON / SGLANG_KT_FP8_LMHEAD
  （environ 已带开关，默认关）。
- SPS 置信度调度表（`--speculative-dspark-sps-table-path`）离线构建。
- 生产切换评估：dspark-kt 分支当前基线比生产 fork 新 6782 个主线提交，行为
  差异（SWA 池、调度器、入口）需完整回归后再考虑替换 ds4f。

## 8. 相关文件
- 基准：`bench_decode.py`；线程占用：`thread_util.py`；profiler 分析：`analyze_prof*.py`
- GPU 微基准：`bench_gpu_ops.py` / `bench_gpu_cold.py` / `scan_w8a8_cfg.py`
- CPU MoE 微基准：`kt-kernel/bench/bench_fp4_moe_cold.py`（随机路由，L2 冷）
- 实验实例脚本：`run_exp.sh` / `stop_exp.sh`（与生产并行，30001 端口，共享巨页权重缓存）
- DSpark 实验：`run_dspark.sh` / `stop_dspark.sh`（30001 端口，venv-dspark；
  `KEEP_GRAPHS=1` 开 graph 调试）；基准 `bench_dspark.py`（5 提示词贪心，校验+吞吐）；
  sglang 分支 `third_party/sglang@dspark-kt`
