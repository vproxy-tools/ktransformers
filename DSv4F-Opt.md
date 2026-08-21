# DSv4Flash 优化与开发者笔记（DSv4F-Opt）

## 5. Prefill 优化轮（2026-08-20）：306 → 494 tok/s

目标：prefill 300+ → 800。本轮落地四项改动，47K prompt 实测
**306 → 493.9 tok/s（+61%）**，decode 39.55 tok/s 无回退，全部正确性
门槛通过（probe CLEAN / bench 5/5 PASS / grow_probe 125K 三级 PASS）。
800 未达，剩余差距的量化分析见 5.5。

### 5.1 CPU MoE 内核 v2（MXFP4 fold 路径）

改动文件：`kt-kernel/operators/amx/fp4-moe.hpp`、
`kt-kernel/operators/amx/la/amx_raw_buffers.hpp`（BufferA 预置换）、
`kt-kernel/operators/amx/la/amx_buffers.hpp`（BufferB 增加 se 数组）。

单核试验台（/tmp 流式 DRAM 权重，m=12×n=2048×k=4096）定位的三处瓶颈与修法：

| 改动 | 原理 | 单核收益 |
|---|---|---|
| 激活预置换（PERMUTE_ACT） | `BufferA::from_mat` 一次性把激活置换到 vpermb 解码序 [偶\|奇]，内层每 group 的 permutexvar（每个 n-tile 重复执行同一行）彻底删除 | ~6% |
| scale 折入指数（fold） | E8M0 尺度是纯 2 的幂 → 反量化出的 bf16 权重直接 int16 加 `e<<7`（指数域加法，**精确**），删掉每 (token,row,group) 的 set1+fmadd（FMA 端口一半负载） | ~20% |
| 运行期 rows 循环 → 4 路编译期实例化 | 运行期边界使编译器无法展开 token 循环（栈上指针寻址），丢掉大部分 fold 收益——这是最大单点 | ~25% |

其它：tile 形状 sweep 证明 **4×4 是寄存器上限下的最优**（12×2/8×2 等大 tile
全部因 zmm 溢出反而慢 2-4×，勿在未重测前加大 MB×NB）；权重预取距离 64→128
组（+3%）。单核 DRAM 流式 87.5 → ~120 GMAC/s。

- 数值：fold 与 legacy 数学等价（同乘积、不同累加序），合成权重校验
  max_rel_err 2.7e-4（bf16 舍入噪声内）；fp4 零值用 nz-LUT 映射 ±2^-95
  避免负指数尺度下指数加法回绕成垃圾。
- 安全网：加载时 `scan_scale_pow2` 并行扫描全部 group scale，任何非 2 幂
  （合成基准）都会把全局 fold 开关降级为 false，内核自动回退 legacy 路径。
- **BufferB 布局变更**（新增每 group int16 的 se 数组，~+12% 尺度区）改变了
  巨页指纹——升级后首次启动会整体冷转换一次（1-3 分钟），之后恢复 REUSED。
- 微基准：M=512 每 27.5→20.2ms，M=1024 48.9→37.8ms（bench_moe_sweep.py）。
- in-situ 每核 ~65-70 GMAC/s（vs 试验台 ~110+）：差距来自 Poisson 路由的
  ragged tile 尾部（~12%）+ 负载不均 + 相位开销；无 perf counter（paranoid=4）
  无法进一步定位。

### 5.2 tilelang 索引器（SM89 可用）

`SGLANG_OPT_USE_TILELANG_INDEXER=1`（run_dspark.sh / ds4f.service 已默认）：
用 tilelang 融合内核替换 indexer 的 torch eager 回退（SM89 上原本只有 SM120
才自动启用）。**383 → 493 tok/s（+29%）**——torch 回退的几十个小 kernel 的
python/launch 串行时间是 prefill 侧最大单项浪费。TILELANG=0 回退。

### 5.3 相位切换 GPU 专家（prefill 用 GPU、decode 全 CPU）

GPU 常驻专家（`--kt-num-gpu-experts`）对两相位效果相反：
prefill +7~9%（空闲 GPU 算力 + CPU 侧 DRAM 减压），decode 却 -40%
（M=1 时路由构建 + matmul_ogs 开销远超省下的 CPU 时间）。

修法（kt_ep_wrapper.py `KTEPWrapperMethod.apply`）：C++ 的
`should_skip_expert` 从 mask 张量**内存实时读**——但注意它读的是
kt-kernel wrapper 在 `BaseMoEWrapper.__init__` 里 **clone 出来的 pinned
副本**，不是 sglang 侧持有的那张（第一次实现写错了张量，decode 图重放
读到的永远是初始快照 → front-loading 下层 0-3 在 decode 整层缺失，
输出退化为复读 prompt；uniform 下每层缺 24 个专家 ~9% MoE 质量，
探针门槛内侥幸通过——两处都必须翻 pinned 副本）。
`SGLANG_KT_GPU_EXPERTS_PREFILL_ONLY=1`（默认开）下，≥64 token 的批次
（prefill chunk）恢复真 mask（CPU 只算非 GPU 专家，GPU 算常驻的），
更小批次（decode / DSpark verify / 尾块）清零 mask（CPU 算全部 256 个）；
GPU 侧同步旁路（decode 图捕获时走 bypass 分支，图内无 GPU MoE kernel）。
**第二个坑**：prefill 层末的清零必须先 `self._sync_done_event.synchronize()`
——`sync_forward` 只是 cudaLaunchHostFunc 异步挂载，不等事件就清零会
抢在本层 C++ 读取之前，静默退化为"CPU 全量 + GPU 双算"（prefill -7%）。
事件等待的流水线代价 ~1.5%。**依赖调度器串行调度
（--max-running-requests 1）**；`SGLANG_KT_KEEP_MASK=1` 可禁用清零
（回到常驻行为，decode 会缺 GPU 专家，仅诊断用）。

实测（DSpark + 24 专家 + 131072 ctx，事件门控版）：prefill 460.6→487.0
（0 专家对照 460.6），decode 41.4（全量 256 专家正确计算，ALL PASS），
grow_probe 8/96/112K（125K 上下文）全 PASS。阈值可用
`SGLANG_KT_GPU_EXPERTS_PREFILL_MIN` 调。

**整层放置（front-loading）实测否决（性能原因，非正确性）**：同显存把
层 0-3 整层放 GPU（`--kt-expert-placement-strategy front-loading`，
24×43=1032 个专家）。最初观测到的"第 5 题确定性 FAIL/复读退化"经定位
是上面第一个 mask bug 在整层场景的放大（4 个整层缺失 ≫ 每层缺 24 个），
修复后 bench ALL PASS、decode 41.5。但 prefill 只有 461.9（vs uniform
487.0，-5%）：整层放置的 GPU 份额无法与 CPU 重叠（GPU 算那 4 层时 CPU
空闲、反之亦然），而分裂层的 GPU 份额在每层内与 CPU 并行。结论：
**分裂层（uniform）在 prefill 上严格优于整层，维持 uniform + 相位切换。**
诊断工具：`SGLANG_KT_DEBUG_FULL_LAYER_DIFF=1` 可对 GPU 层做
GPU-vs-全专家-CPU 在线对拍（注意双 submit 的 hack 有偶发全零假象，
仅作参考）。

**混合放置（hybrid）已实现并实测**：`--kt-expert-placement-strategy
hybrid --kt-num-gpu-full-layers F --kt-num-gpu-experts U`——前 F 个 MoE
层整层上 GPU，其余每层撒 U 个专家叠加收益（一个整层 ≈3.2GB，一个专家
跨 43 层 ≈542MB）。实测（DSpark、相位切换修复版）：

| 配置 | 专家显存 | prefill | decode | 上下文 |
|---|---|---|---|---|
| uniform-24（生产默认） | 13GB | 487.0 | 41.4 | 131072 |
| hybrid 2 整层+16/层 | 14.7GB | 485.5 | 41.4 | 131072 |
| hybrid 2 整层+24/层 | 18.9GB | 495.1 | 39-41* | 池压至 ~92K |

（*decode 为 accept 内容波动，两次 bench 32.8/39.4 均 ALL PASS。）
结论：**同显存下整层与分裂的 prefill 收益几乎线性等价（整层略亏
~5%，因 GPU 份额无法与 CPU 跨层重叠）**；激进 hybrid 换 +1.7% prefill
要牺牲 KV 池。生产维持 uniform-24；hybrid 留作显存富余场景的调节手段
（如无 DSpark 的 prefill 模式、更大显存的卡）。

配套修复：`v4_triton_kernels_moe.py` 的 `_make_routing_data_v4` 原依赖
`triton_kernels.routing`（本 venv 的 0.1.0 精简版没有）——改走 SparseMatrix
+ `make_ragged_tensor_metadata`（vLLM 同款构造），topk 6→8 的 2 幂 padding
保持不变以满足 triton 元数据内核假设；cuda graph 捕获安全（已单测）。

### 5.4 chunk 1024 与显存预算

`--chunked-prefill-size/--max-prefill-tokens 1024`：MoE 每 token 摊销更好
（M=1024 vs 512 提升 ~8%）。tilelang 索引器下 >100K 重预填充实测安全
（grow_probe 125K PASS；旧栈 torch 回退时 1024 在 >100K 差 0.01GB 的限制
不再成立）。显存：DSpark+24 专家需 MEMFRAC 0.85（实测 36.8GB/48GB）。
prefill 专用配置（无 DSpark、56 专家、MAXTOK 66048、MEMFRAC 0.92）可到
**536.5 tok/s**，适合批量灌注场景（上下文上限 ~62K）。

#### 5.7 终局：partial Marlin 解锁常驻 GPU 专家的 decode 收益

用户再提出：prefill 尽量用 GPU 专家；decode 按路由分流、至少一半专家给
CPU，让 CPU 权重带宽与 GPU 算力并行。物理约束修正：GPU 只能算**权重常驻
显存**的专家，"一半"意味着 ~69GB 显存（128/层×43×12.6MB）——本卡不可行；
实际可常驻 ~11% 的 pair。但机制方向正确，此前被一个内核问题掩盖：

**profile 定位**：常驻模式的 decode 慢在 `_matmul_ogs_NNT_bf16...16x256x128`
内核 **762µs/层**（M=6 只跑出 2.4 TFLOPS，tile 形状完全不适合小 M）——
这是当初 decode 40→24.5 塌掉的全部原因，与"GPU 参与 decode"无关。
代码里已有小 M 优化的 **Marlin 路径**（`_MARLIN_MXFP4_CAPS` 含 SM89），
但被 `_resident_partial` 条件禁止用于部分常驻层。

修法：`SGLANG_V4_MARLIN_PARTIAL=1`（mxfp4_deepseek.py）放开限制，
`prepare_v4_mxfp4_marlin` 打包常驻子集、路由 remap 兼容 -1 掩码。

**终版配置**（常驻 28 专家 + partial Marlin，关闭相位切换）：
prefill **499.3**、decode **43.13**（双超相位切换版的 487.0/41.4），
probe CLEAN / bench ALL PASS / grow_probe 125K 全 PASS，显存 ~39.6GB。
32 专家与 28 持平（饱和）。相位切换机制保留作 Marlin 不可用时的回退。

| 配置 | prefill | decode |
|---|---|---|
| 相位切换 + matmul_ogs（§5.3） | 487.0 | 41.4 |
| 常驻 28 + partial Marlin | 499.3 | 43.13 |
| **+ hybrid 1 整层叠加（1F+28U，真终版）** | **506.9** | **47.93** |

**hybrid 叠加式胜出的前提是 Marlin**（用户提议）：之前 hybrid 测出持平/
负收益有两个原因——整层被迫走 matmul_ogs，且当时是"替换式"花显存（从
uniform 份额里拨）。Marlin 解锁后"uniform-28 保持 + 真余显存填 1 整层"
是纯增量：整层省 38.6ms CPU/层、暴露 ~4ms GPU、零通信，decode 该层走
Marlin 再省 ~1.1ms/步。显存 42.0GB（再填一层会挤压图捕获）。
probe CLEAN / bench ALL PASS / grow_probe 125K 全 PASS。

### 5.8 显存账本（实测，hybrid 1F+28U 终版配置）

数据来源：服务器启动日志的 avail mem 里程碑 + `DSV4 memory calculation` 行
（hybrid 1F+28U 配置，2026-08-20 实测，总卡 47.5GB / 标称 48GB）：

| 项 | 大小 | 依据 / 算法 |
|---|---|---|
| 目标模型加载合计 | **29.04 GB** | log `Load weight end ... mem usage=29.04 GB` |
| ├─ 路由专家（hybrid 1F+28U） | 18.40 GB | 3.23（整层）+ 28×43×12.6MB（见下） |
| └─ 骨架（attention/索引器/共享专家/embedding/lm_head/norm） | ~8.8 GB | 差额 + safetensors 分类 |
| DSpark draft（mtp.0） | **10.37 GB** | log 第二次 `Load weight end` |
| KV 池（135168 token，fp8） | **~1.0-1.25 GB** | `bytes_per_full_token=9259.90` × 135168 |
| CUDA 图 + 工作区 + allocator 保留 | ~2-3 GB | `Memory pool end avail=6.86` → 稳态占用 42.0 |
| **稳态合计** | **42.0 GB** | nvidia-smi 实测 |

专家粒度换算（MXFP4，fp4 权重 + E8M0 尺度）：

| 单位 | 显存 |
|---|---|
| 1 个专家（单层；gate 4.0MiB + up 4.0MiB + down 4.0MiB + 尺度） | **~12.6 MB** |
| 1 个"专家列"（同一 id × 43 层，即 +1 个 uniform 专家） | ~542 MB |
| 1 个完整层（256 专家 × 12.6MB） | **~3.23 GB** |
| 当前配置每层专家占用（28 个） | ~353 MB/层（第 1 层另整层 3.23GB） |

**KV↔专家兑换率（重要结论）**：每 token KV 仅 **9,259.9 B**（DSA 压缩缓存，
fp8；log `DSV4 memory calculation` 行直接给出）。因此：
- 整个 131072 上下文池 ≈ **1.25GB，不到半个完整专家层**（3.23GB）；
- "砍上下文换一个整层"在本模型上**不可行**——把上下文砍到 0 也凑不出一层；
- 显存的大头是**权重**不是 KV。想再换整层，唯二杠杆：关 DSpark（10.37GB ≈
  3.2 个整层）或从每层专家数里挪（28→25 省 1.6GB ≈ 半层）。

**"关 DSpark 换 3 整层"实测（2026-08-20 A/B，不值）**：关 DSpark 后 hybrid
填到 4F+28U（4×256 + 39×28 = 2116 专家，目标权重 37.56GB、稳态 40.25GB，
余 8.81GB）——prefill **532.4**（vs 506.9，**+5.0%**，约 8.5 tok/s/整层）；
decode 稳态 **28.73**（bench_dspark 27.78，5/5 PASS，vs 47.93，**−42%**）。
DSpark 的 accept×~1.7 加速远超 3 个整层的带宽减负；且 4F+28U+DSpark =
37.56+10.4GB > 48GB 物理装不下，二者只能二选一。**维持 1F+28U+DSpark**。
（复现：去掉 `--speculative-algorithm DSPARK` 与其 2 个专属 env，改
`--kt-num-gpu-full-layers 4`；bench 口径同 §6.1/6.2，log /tmp/nodspark_4f.log）

### 5.8 为什么是 499 不是 800（量化结论）

47K prompt 每 token 2.00ms = CPU MoE ~1.45ms + 非 MoE（attention/dense/
索引/glue）~0.55ms。要到 800（1.25ms/token）需 CPU MoE 再降 ~2×：
- 内核指令组合上限（dpbf16 32 MAC/条 + fold 后非 FMA 端 ~2 条/MAC 组）
  单核 ~110 GMAC/s 已接近；in-situ 65-70，理论剩余 1.4×。
- GPU 专家受显存限制（DSpark 共存时 ≤24-28 个 ≈ 9-11% 对），线性外推
  无法覆盖剩余差距。
- prefill CUDA graph（本可消掉每层 ~4ms 串行 glue）：2026-08-21 已深入到
  图回放层并定位故障（详见 §5.9）——绑定包换 12.9.7 后捕获成功、kt MoE
  以 eager 断点接入，但**任何真实图回放均产出损坏结果**（tier512 乱码 /
  tier1024 illegal access），+14% 的吞吐是在损坏计算上测得，不可用。
  维持禁用；生产不受影响。
- 单卡 TBO 不适用（无通信可重叠）。

### 5.6 结果汇总（47K prompt，131072 ctx，DSpark）

| 配置 | prefill tok/s | decode tok/s |
|---|---|---|
| 优化前基线（chunk 512，0 专家，torch 索引器） | 306 | 39.6 |
| + 内核 v2 | 353 | — |
| + 24 GPU 专家（常驻） | 374.6 | 24.5（回退！） |
| + chunk 1024 | 382.8 | — |
| + tilelang 索引器（0 专家对照） | 460.6 | 40.6 |
| + 相位切换 v1（mask 写错张量，decode 缺 9%） | 493.9 | 39.55 |
| **+ 相位切换 v2（pinned 双写 + 事件门控，生产）** | **487.0** | **41.4** |

（相位切换 v1 的 prefill 493.9 带着竞态红利且 decode 静默缺 24 专家/层，
不作数；v2 是全量正确下的诚实数字。）

### 5.9 BCG prefill 图攻坚（2026-08-21）：换绑定成功、回放损坏已定位、未修

**依赖变更（已生效并保留）**：venv `cuda-python/cuda-bindings 13.3.1 →
12.9.7`。13.3.1（CUDA 13 绑定）对 550 驱动一律 error 35（连 CUDA 10 时代
的 `cudaStreamGetCaptureInfo` 都拒）；12.9.7 同时满足 torch 声明的
`>=12.9.4,<13` 与 550 驱动（cu12 minor 兼容）。注意 sglang pyproject 声明
`cuda-python>=13.0`，本机属有意偏离。**换包前快照：
`requirements-backup-20260821.txt`**（恢复：
`.venv/bin/pip install -r requirements-backup-20260821.txt`）。

**打通的部分**：
1. error 35 消失，`--cuda-graph-backend-prefill breakable` 显式锁定可绕过
   DSV4 自动禁用规则（server_args.py `_disable_breakable_cudagraph_...`
   的规则对显式设置不生效）；捕获成功（单档 1024：6.3s/1.11GB；38 档全量
   ~56-111s/~6GB，1F+28U+DSpark 显存装得下，memfrac 0.87）。
2. **kt 混合 MoE 必须是 eager 断点**：`KTEPWrapperMethod.apply` 加
   `@eager_on_graph(True)`（kt_ep_wrapper.py）——BCG 回放只重放录制段 +
   标注断点，apply 的 host 编排（CPU 专家提交/路由元数据/同步）不标注则
   回放吃陈旧状态（短 prompt 数学题复读式错误）。非 BCG 场景装饰器直接
   透传，零开销。
3. **上游 NamedTuple 弱引用 bug 修复**（breakable_cuda_graph.py
   `_weak_ref_if_tensor`）：NamedTuple（StandardDispatchOutput/TopKOutput）
   被拆成普通 tuple，断点回放 `dispatch_output.hidden_states` 直接
   AttributeError；现按 `type(x)(*elems)` 重构。
4. 调试工具（保留）：`SGLANG_BCG_DEBUG_SYNC=1` 每段/每断点后同步打点；
   `SGLANG_BCG_DEBUG_KERNELS=1` 附带段内内核名转储（cuGraphGetNodes +
   cuKernelGetName，需 keep_graph，已联动）。

**未解决——图回放损坏（全部实测，1F+28U、无 radix、单请求）**：

| 配置 | 真实图回放结果 |
|---|---|
| tier 1024 + ctx131072 + 无 DSpark | 首次回放即 illegal access（logits 处浮出） |
| tier 1024 + ctx131072 + DSpark | 不崩但多 chunk 输出损坏（grow_probe 8K FAIL：暗号丢+数学错+复读 67） |
| tier 512 + ctx131072 | 多 chunk 完成但输出乱码（"Repeat w0" → `" ( 8 (,1"`） |
| 短 prompt（<chunk，走 eager） | 全部正确（probe CLEAN / bench 5/5）——曾经的"短 OK"均为 eager，非图 |

bench_prefill 47K 实测 579.6 tok/s（+14.3%）是在损坏计算上的吞吐，
**不可作为收益结论**。基线（同机同晚、无 BCG）grow_probe 同口径 PASS，
损坏归因于 BCG 图回放本身。

**定位数据**（SGLANG_BCG_DEBUG_SYNC 下，崩溃固定发生在第一个 MoE 断点后
的 segment 2/87 回放；断点本体 apply 同步后通过）：segment 2 内核序列 =
`Mul`（合并加权）→ `add`（残差）→ `mhc_post/pre_*_tilelang`（前后置融合
norm）→ `per_token_group_quant` → `_w8a8_block_fp8_matmul`（wqa/wkv 投影）
→ `fused_q_norm_rope` → **index_elementwise + direct_copy(cast)** →
`fused_k_norm_rope_flashmla`。已排除：dedup（默认关）、显存压力
（memfrac 0.78 余 11.5GB 仍崩）、DSpark（无亦崩）、多档捕获（单档亦损）、
CUDA_LAUNCH_BLOCKING（与捕获互斥不可用）。

**第二轮深挖（同日续，sanitizer 路线受阻后的等效定位）**：
- compute-sanitizer 与 sglang 多进程 spawn 死锁（TreeLauncher 起，
  scheduler 子进程不产 GPU 工作，15min 无权重加载）——不可用于本服务形态。
- 等效工具链（全部保留在分支）：`SGLANG_BCG_DEBUG_SPLIT=after_mlp,...`
  （deepseek_v4.py 内空断点二分器）+ `SGLANG_BCG_NO_WEAK_REF=1`（禁弱引用）
  + 既有 SYNC/KERNELS 转储。
- **新修复（必要非充分）**：`_compute_kv_to_cache`（DSV4 k-norm+rope+
  **直写分页 KV**，经 `set_swa_key_buffer_radix_fused_norm_rope`）加
  `@eager_on_graph(True)`——录制进段则回放写陈旧 slot；与 attention 同类
  的 KV 副作用，必须 eager。
- **expandable_segments 定性**：关掉后 illegal access 消失（变静默损坏）——
  崩溃=VMM unmap 已释放显存 + 录制内核悬垂指针；保留时崩溃点稳定在 MoE
  断点后段（强引用也不救 → 悬垂不在断点参数里，在段间中间量）。
- **损坏本质（no-expandable + grow_probe）**：8K 阶段 FAIL，模式=
  `codeword=False / math=True / dup=0`——**前缀注意力跨 chunk 丢失**，
  当前步正常；无 BCG 同配置对照 PASS（expandable 有无均 PASS，确认与
  expandable 无关的纯 BCG 回放缺陷）。
- 再排除：overlap 调度竞态（--disable-overlap-schedule 仍 FAIL）、
  MAX_SEQ_LEN 钳制（=131072 全长）、元数据刷新面
  （refresh_for_breakable_cuda_graph_replay_ 对 raw_out_loc/seq_lens/
  positions/c4/c128/page_table/flashmla 字段全覆盖，审查未见缺口）。
- 剩余嫌疑集中在：refresh 的 `reference_assign_fields`（page_table/
  swa_page_indices/c128_page_indices/flashmla 元数据走**引用替换**而非
  原地拷贝——若某录制内核按捕获期地址读这些张量，刷新换对象即失联）
  与 c128 在线压缩的跨 chunk 状态机。下一步：对这两组字段做原地拷贝
  实验、或上游 issue（DSV4 的 BCG 兼容声明只在全 GPU 栈上成立过）。

**第三轮（同日，收束）**：
- `SGLANG_BCG_REFRESH_INPLACE=all`（reference_assign_fields 全量改原地
  拷贝，deepseek_v4_backend.py，两段式防部分拷贝；默认关）：grow_probe
  结果与引用替换**完全相同**（8K FAIL 同签名）——该假设**证伪**。工具
  保留供后续逐字段二分。
- tilelang 索引器对照（SGLANG_OPT_USE_TILELANG_INDEXER=0）在 ctx131072
  下不可行：torch 回退路径捕获期 kvcache gather 峰值 OOM（2GB×，有无
  expandable、memfrac 0.70-0.87 均炸）——此线关闭。
- **单 chunk 图回放也损坏**（~819-token 数学题 prompt，pad 到 1024 档，
  GRAPH，输出空）——损坏**并非前缀特有**，而是普遍性回放损坏；grow_probe
  的"暗号丢失"只是其在长上下文下的表现。另观察到同次启动里图使用后的
  短 eager 请求也出错（一次性样本，疑与捕获期 dummy 前向污染持久状态
  有关，未深究）。
- 结论：BCG 在 DSV4+kt 栈上是**多重损坏**（悬垂指针崩溃 + 普遍回放
  数值损坏 + 疑似捕获期状态污染），远超单点修复；建议上游 issue 附
  §5.9 全部证据（sglang 分支含全部诊断开关）。生产维持禁用。

**生产影响**：ds4f.service 无 BCG 参数，行为不变；cuda-python 12.9.7 下
eager+decode 图路径已验证（bench_dspark 5/5 ALL PASS、短 prompt probe
CLEAN）；长 prefill eager 在 12.9.7 下建议上线前补一轮 grow_probe。

## 1. DSpark 投机解码（主线 sglang 移植）

### 1.1 背景与路线

官方 DeepSeek-V4-Flash-0731 自带 DSpark draft head（config: `dspark_block_size=5`,
`dspark_target_layer_ids=[40,41,42]`, `dspark_noise_token_id=128799`,
`dspark_markov_rank=256`；权重在 `mtp.0.*`，4705 个键），target/draft 同源，只需
`--speculative-algorithm DSPARK`。上游 sglang 主线（sgl-project）在 PR #30261 支持。

为此开分支 **`dspark-kt`**（third_party/sglang）：基底取主线 4ad990ba7（最后一个钉 torch 2.11 的
提交，避开 cu13 依赖墙——驱动 550 只支持 CUDA 12.x），在其上：

- **移植 kt CPU 专家引擎**（4477 行 kt_ep_wrapper + mxfp4_deepseek +
  v4_marlin/v4_triton_kernels + quant_method_registry + jit_kernel 包 +
  linear_bf16_fp32 等），主线 V4 模型挂 `_try_kt_plugin` side-effect 注册。
- **并入三个既有提交**：巨页缓存、perf pack（environ/4090D 调优 config/EXTRA_ARGS 钩子）、
  FP8 lm_head GEMV（后者后经复验判定不可用，见 DSv4Flash.md 10）。
- **SM89 适配**（主线假设 SM90+/SM120）：paged_mqa_metadata 128KB 动态共享内存按
  设备 optin 上限钳制；sparse decode/prefill 注意力走 Triton 回退
  （debug_flash_mla_adapter）；索引器 logits 走 torch 回退；topk v1（v2 用
  SM90 线程块集群）。
- **DSpark draft 保持纯 GPU**：`build_draft_tp_worker` 包进
  `speculative_kt_ep_disabled_context()`，draft 专家不上 CPU（约 10.6GB GPU）。

### 1.2 环境

- venv：仓库根 `.venv`（torch 2.11.0+cu128、flashinfer 0.6.15.post1[cu12]、
  sgl-deep-gemm 0.1.5.post3+cu129（docs.sglang.ai 索引）、tilelang 0.1.11、
  cuda-python 13.3.1、transformers 5.12.1；sglang 为 editable、kt-kernel 为
  本地构建拷贝安装，~200 个包，完整清单以 `.venv/bin/pip freeze` 为准）。
- 实验实例启停：`run_dspark.sh` / `stop_dspark.sh`（30001 端口，与生产同用 `.venv`）。`DSPARK=1` 开投机，
  默认 cuda graph 开 + `SGLANG_RAGGED_VERIFY_MODE=static`（修复后正确且
  更快；`EAGER=1` 回退无损 eager）。MEMFRAC：无投机 0.30，DSPARK 需 ≥0.60
  （draft 权重计入预算）。
- 依赖分支状态：third_party/sglang 指针已记录在 optimize-latest（分支 `dspark-kt`，
  venv 内 sglang 为 editable 安装）。kt-kernel 以 torch 2.11 头文件重编
  （`pip install --no-build-isolation --no-deps .`）。
  生产 ds4f.service（30000 端口）与本节实验实例共用此 venv。
  venv 最初为手工组装、逐条命令未留档；新机器复刻：按上述版本装齐依赖
  （deep-gemm 走 docs.sglang.ai 索引）→ `pip install --no-deps -e $KT_ROOT/third_party/sglang/python`
  → `$KT_ROOT/kt-kernel` 下 `pip install --no-deps --no-build-isolation .`。

### 1.3 实测（greedy，5 提示词，30001 端口）

| 配置 | 平均吞吐 | accept len / rate | 质量 |
|---|---|---|---|
| 主线基线（kt MXFP4，无投机，cuda graph） | 26.0 tok/s | — | 正确（3288 ✓、散文流畅） |
| DSpark + cuda graph（修复前） | 38.9 tok/s | 恒 2.00 / 0.20 | **损坏**（重复词，见 §4） |
| DSpark + eager + static | 34.3 tok/s（峰值 38-46） | 2.9-3.3 / 0.38-0.46 | 正确 |
| **DSpark + cuda graph + static（修复后，推荐）** | **39.6 tok/s**（峰值 49.8） | **2.2-3.8 / 正常波动** | **正确** |

- 修复后 graph 配置 4 项 soak（2479 token 连续重负载）：39.57 tok/s 持续，数学
  （水池题 4小时48分 ✓）/散文/翻译/英文总结全部正确，零段错误。
- 对比：vs eager +15%，vs 无投机基线 **+52%**；`tests/bench_dspark.py` 5/5 PASS
  （38.1 tok/s，该 prompt 集 thinking 较短）。
- 首请求含 Triton JIT 预热（~3-6 tok/s），稳态请以第二请求起算。

### 1.4 后续机会

- 索引器 logits 从 torch 回退换 tilelang（`SGLANG_OPT_USE_TILELANG_INDEXER=1`，
  需验证 SM89 编译）——当前 torch 回退是 eager 路径的主要 GPU 开销之一。
- 把主线自带的 SGLANG_OPT_FP8_WO_A_GEMM 等消费者级优化在 SM89 上验证开启
  （SGLANG_KT_WOA_FP8_TRITON / SGLANG_KT_FP8_LMHEAD 已判定不可用，见 §4）。
- SPS 置信度调度表（`--speculative-dspark-sps-table-path`）离线构建。

## 2. 相关文件
- 测试工具：全部在 `tests/`（探针/基准/巨页与泄漏回归等，见 §3）；
  GPU 微基准 `tests/bench_gpu_ops.py` / `tests/bench_gpu_cold.py` /
  `tests/scan_w8a8_cfg.py`；CPU MoE 微基准 `kt-kernel/bench/bench_fp4_moe*.py`
- 实验实例：`run_dspark.sh` / `stop_dspark.sh`（30001 端口，`KEEP_GRAPHS=1` 开
  graph 调试）；基准 `tests/bench_dspark.py`；sglang 分支 `third_party/sglang@dspark-kt`
- **测试工具全集（功能/前提/执行/清理/结果解读/通过分界）：见 §3**
  —— 均在 `tests/`：`probe_dspark.py` / `bench_dspark.py` / `grow_probe.py` / `bisect_ctx.sh` /
  `hp_weight_check.py` / `sync_leak_check.py`；启动脚本留在仓库根：`run_dspark.sh`+`stop_sglang.sh`；
  `kt-kernel/bench/bench_fp4_moe*.py`

## 3. 测试工具参考（开发者）

面向开发者；普通用户视角的构建/部署/运行见 `DSv4Flash.md`（其 9.4 节有工具简表并指回本节）。
命令中的 `$KT_ROOT` / `$MODEL_DIR` 含义见 DSv4Flash.md 开头的路径约定；venv 一律指仓库根 `.venv`。
通用前提：所有探针/基准都向目标端口发**真实生成请求**（temperature=0 贪心），
默认超时 600–1200s；除特别注明外用系统 python3 即可（只依赖 urllib）。

**GPU 独占规则（重要）**：生产 ds4f(30000) 与实验实例（run_dspark.sh，30001）**不能
同时占 GPU**——曾有实验实例残留 24GB 显存，导致生产部署时 OOM 崩溃循环。要跑实验（tests/bisect_ctx.sh / A/B 重启类），先停生产（`sudo systemctl stop
ds4f`，或 kill 主进程靠 systemd 30s 后拉起、注意 StartLimitBurst 预算）；跑完把实验
实例停干净再把生产拉回。探针/基准（不重启服务器）与生产共存没问题。

### 3.1 `tests/probe_dspark.py` —— 快速损坏探针

- **功能**：3 个短生成（数学 12×3、中译英、百字短文），检查数学正确性、翻译可辨、
  思考/正文切分、重复词（dup_score：8+ 字符 chunk 在邻近 60 字内重复次数）。
- **前提**：目标端口有活的 sglang 实例（生产或实验均可）。
- **执行**：`python3 tests/probe_dspark.py [port]`（默认 30001；对生产用 30000）。~1 分钟。
- **清理**：无需。服务端无状态残留（--disable-radix-cache，不污染缓存）。
- **解读**：输出 `CLEAN` 或 `CORRUPT:` + 逐条失败原因（math wrong / translate bad /
  reasoning dup / essay dup 等）。
- **通过分界**：**退出码 0=CLEAN，1=CORRUPT**，可直接接 CI/脚本判断。偶发单条
  失败先重跑一次（贪心下应稳定复现才算真损坏）。

### 3.2 `tests/bench_dspark.py` —— 正确性 + 吞吐基准（5 提示词）

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

### 3.3 `tests/grow_probe.py` —— 长上下文增长探针

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

### 3.4 `tests/bisect_ctx.sh` —— context 阈值二分（重启循环）

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

### 3.5 `tests/hp_weight_check.py` —— 巨页权重缓存冷/热链路验证（CPU-only）

- **功能**：用真实模型某一层走与服务器**完全一致**的 NativeMoE MXFP4 加载路径：
  冷进程做 safetensors 读取 + 转换写入持久 arena + commit 标记；热进程 python
  `check_reusable` 命中 → 跳过 safetensors，C++ 直接 mmap 驻留大页。产物（marker +
  weights.bin 分段）**就是服务器要复用的内容**（layer key/stamp/pfp 与 ds4f 一致）。
- **前提**：`/dev/hugepages/kt_weights` 已存在且当前用户可写（root 一次性
  mkdir+chown，见 DSv4Flash.md 8）；**必须用仓库 `.venv` 的 python**（要 import
  kt_kernel）。
- **执行**：连跑两遍（必须是两个独立进程——同进程第二次 alloc 时 arena cursor 已
  前移，marker 偏移不再相等，测不出复用）：
  `KT_MODEL_DIR=$MODEL_DIR $KT_ROOT/.venv/bin/python $KT_ROOT/tests/hp_weight_check.py [layer_idx]`
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

### 3.6 `tests/sync_leak_check.py` —— SyncArgs 泄漏 / 图回放 UAF 回归

- **功能**：双路回归 kt-kernel `CPUInfer::sync_with_cuda_stream` 的修复（背景见 §4）：eager 路 100 万次调用后 malloc_trim RSS 增长应≈0；含 4 个 sync host 节点的
  cuda graph 5000 次回放应无崩溃无增长（捕获型 args 永生、回放零分配）。
- **前提**：GPU 可用（占用 <1.5GB，**与生产共存安全**）；仓库 `.venv` 的 python（编译进
  `.venv` 的 .so 才是被测对象——先按 DSv4Flash.md 7.2 重编并装入再测源码改动）。
- **执行**：`$KT_ROOT/.venv/bin/python $KT_ROOT/tests/sync_leak_check.py`，~2 分钟。
- **清理**：无需（纯本地进程）。
- **解读**：`[eager] ... (X B/次)` 与 `[graph] ... 无崩溃`；测量陷阱：紧循环 RSS 读数
  含"已 free 未归还"的驻留页（纯 C 跨线程 malloc/free 模式本身 ~28B/次），所以
  **必须看 malloc_trim 之后**的数字——脚本已内置 trim。
- **通过分界**：末行 **PASS**（eager ≤8 B/次、graph 增长 ≤64MB、无崩溃）退出码 0；
  FAIL 或中途 Segmentation fault = 不通过（后者=图回放 UAF 回来了）。

### 3.7 `run_dspark.sh` / `stop_sglang.sh` —— 实验实例启停

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

### 3.8 `kt-kernel/bench/bench_fp4_moe.py`（+ `_cold` 变体）—— CPU MoE 微基准

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
  （多日复测一致）。
- **通过分界**：M=1 落在历史 ±5% 内 = 机器/内核健康；显著偏高先查负载再查回归
  （对照 jsonl 里最近记录的 commit 定位改动）。

## 4. 已修复问题档案（均已修复并验证，折叠备查）

<details>
<summary><b>DSpark verify 的 cuda-graph 回放损坏（两轮定位，均已修复）</b>——症状：graph 开启时输出损坏/段错误；现状：graph 默认开启且正确</summary>

<b>第一轮：pinned 中转 buffer 生命周期（修复在 kt-kernel/python/experts_base.py `KExpertsCPUBuffer.get_buffer`）</b>

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

修复：get_buffer 在 `torch.cuda.is_current_stream_capturing()` 为真期间分到
（或命中 temp）的尺寸提升进 `capture_buffers` 永久保活；prefill 尺寸（从不捕获）
仍走单槽 temp，内存零增长。修复后：多请求无劣化、4 项 soak 39.57 tok/s 全对、
零段错误，graph 收益 +15%。

调试技法存档（代码已清理）：同进程 graph→clone→eager 三连位对比；
`_dbg_full_meta_list` 抓 graph 池内 metadata 张量 replay 后读值；KV 池 uint8
checksum 对比；pinned 指针复用独立复现脚本。

<b>第二轮：verify 元数据图内构建损坏（context > ~111.4K）——已修复，context 放开到 131072</b>

症状：`--context-length` 超过约 111.4K 后生成确定性损坏（重复短语、远程注意力
劣化；与实际序列长度无关——ctx=131072 下 4K 短 prompt 也坏）。二分边界：
111360 干净 / 111616 损坏。

排除项（均实测）：实际序列长度、KV 池容量（池 114688 与 135168 都出现过
干净/损坏组合）、page 宽度（111360 与 111616 同为 437 页）、draft 图（draft
图开 + verify eager = 干净）、verify 元数据数值（图回放后逐字段对比 eager
重建，page_table/swa/c4/c128/positions 全等）、压缩计划字节（plan_c/plan_w
全等）、两条元数据构建路径互比（raw vs _old 全等）。

根因（定位到机制层面）：`SGLANG_PREP_IN_CUDA_GRAPH=1`（默认）把 verify 的
raw→full 元数据构建（`make_forward_metadata_from_raw_verify` 一族 triton/torch
算子）**录制进 verify cuda graph**。录制后所有可读产物都正确，但生成损坏——
即损坏源于"构建被录制"这一形态本身（疑似捕获期内存池别名/算子时序效应，
回放后值正确、前向中途被污染），在 req_to_token 宽度超过 ~437 页时触发。
`SGLANG_PREP_IN_CUDA_GRAPH=0`（全局图外）可修但有 `.tolist()` CPU 同步。

修复：新增 `SGLANG_DSv4_VERIFY_META_OUT_OF_GRAPH=1`（run_dspark.sh/ds4f.service
已默认导出）——仅对 TARGET_VERIFY bucket，把 raw→full 升级移到图外按步执行
（同一组纯 GPU 构建器，无 CPU 同步），图内只录模型层；draft decode 保持图内
快速路径。改动：`deepseek_v4_backend.py` + `environ.py`。

验证（ctx=131072 + 池 135168 + chunk 512）：probe CLEAN；bench 5/5 PASS；
长上下文增长探针单会话 19.5K / 109K / 125,211 token 三级全 PASS（远程暗号在
125K 上下文仍可回忆、新数学题正确、零复读）。性能 A/B 图内 27.8 vs 图外
28.2 tok/s——零回退。另：>100K 的 prefill 需 chunk 512 + expandable_segments。

独立小怪癖（不影响正确性，已自动规避）：首个 verify（graph 或 eager 均是）
返回的 hidden_states 是未物化的输出 buffer；`dspark_verify.py` 前 2 步 verify
自动强制 eager 预热（env `SGLANG_DSv4_VERIFY_EAGER_WARMUP`，默认 2）。

</details>

<details>
<summary><b>SyncArgs 泄漏与图捕获 use-after-free 事故（均已修复）</b>——症状：长跑 RSS 缓涨 ~12MB/h；首版修复曾引发崩溃循环</summary>

`kt-kernel/cpu_backend/cpuinfer.h` 的 `sync_with_cuda_stream()` 每次 `new SyncArgs`
从不释放（实测 ~32 B/次 ≈ 生产 ~12 MB/h）。修复时踩过一个必须记录的坑：

<b>第一版修复（回调里无条件 delete args）导致生产 SIGSEGV 崩溃循环。</b> 根因：
decode/verify 的 cuda graph 捕获期间也调用 `sync_with_cuda_stream`——
`cudaLaunchHostFunc` 连同 `args` 指针被录成图内 host 节点，**每次图回放都用同一
指针重跑回调**。首次回放 delete 后，第二次回放变成 use-after-free + double-free →
堆损坏 → 主线程死在 `pthread_mutex_lock`（faulthandler 栈可见）。也就是说，原代码
的"泄漏"里有一部分是**被捕获图的函数性需求**（args 必须永生）。

正确修复（已部署生产，提交 c082623）：启动时 `cudaStreamIsCapturing()` 探测——
eager 一次性回调标记 `owned=true` 回调自删；捕获流标记 `owned=false` 永生（回放
零分配，只在捕获时分配一次，量级为启动期常数）。验证：eager 1M 次 RSS 增长
-0.9 B/次（malloc_trim 后）；图 5000 次回放 × 4 节点无崩溃；生产 probe CLEAN +
bench 5/5 PASS 33.54 tok/s；MoE 微基准 M=1 227µs（基线 233µs）无回退。
回归测试：`tests/sync_leak_check.py`（见 3.6）。

诊断要点：紧循环 RSS 读数含"已 free 未归还"的驻留页（纯 C 跨线程 malloc/free
模式本身读出 ~28 B/次），**必须 malloc_trim 后再测**；CPU-only `sync()` 路径因
指针不逃逸被编译器优化掉分配，泄漏只在 CUDA 路径。

</details>

<details>
<summary><b>禁用环境变量的审计依据（FUSE_RSF / KT_FP8_LMHEAD 等）</b>——结论汇总见 DSv4Flash.md 10</summary>

`SGLANG_OPT_MXFP4_FUSE_RSF_SHARED_ADD`：默认 False；本栈三处 GPU MoE 路径
（marlin/triton_kernels/trtllm）里它只控制是否跳过 `output.mul_(rsf)`，**没有
融合消费方**。模型 routed_scaling_factor=1.5，=1 会静默丢掉 ×1.5——不是性能
开关，是正确性地雷。

`SGLANG_KT_FP8_LMHEAD`（单元级实测，未动生产）：FP8 GEMV 内核数学正确（vs
手工反量化参照误差 0.037），T=1 提速 1.96×（2242→1143µs，读带宽减半），但
T=2 持平、T=4 反而 0.5×（einsum 按 T 逐行读权重），DSpark 下仅 draft 步受益。
决定性否决点是 **stash 构建有数据竞争**：`build_lmhead_fp8` 的
`weight[...].to("cpu", non_blocking=True)` 后立即在 CPU 上量化，同一权重两次
构建产物**比特不同**且权重重建误差 ~0.007（健康 fp8 应为 ~0.0003，差 20×），
会把 logits 静默算坏。若未来要启用：先给 D2H 后补 `torch.cuda.synchronize()`，
复测重建误差回到 ~2-3% 再说。

其余（WOA_FP8_TRITON / OVERLAP_STORE_CACHE）在新栈无任何读取点；FUSE_WQA_WKV /
USE_JIT_NORM / USE_MULTI_STREAM_OVERLAP / USE_FUSED_STORE_CACHE 为 `EnvBool(True)`
默认值，显式设置=冗余。

</details>

## 6. 测试方法学（§5 全部数字的测法，逐项详记）

### 6.1 通用约定

- **实验实例**：`run_dspark.sh`（端口 30001）承载全部 A/B；env 覆盖
  `MEMFRAC / MAXTOK / PREFILL / DSPARK / EXTRA_ARGS`，一条命令 = 一个配置。
  生产 ds4f(30000) 与实验实例互斥占卡（GPU 独占规则，§3 节首）。
- **每次 A/B 只改一个变量**；同配置连跑 2-3 次取总均值，跑间噪声
  **±0.5-1%**（<1.5% 的差异一律视为持平，不下结论）。
- **首请求预热**：含 Triton/tilelang JIT 的第一条请求会偏慢，一律丢弃
  （bench 脚本取第 2 次起的迭代；或先打一条不计数的请求）。
- **巨页状态**：改 kt-kernel 布局后首启是冷转换（1-3 分钟），之后
  `REUSED from persistent hugepages`；A/B 前确认日志出现 REUSED，
  避免把冷启动 IO 计入。
- **判断门槛**：prefill 以 ±1% 为噪声带；decode 的 bench_dspark 吞吐
  是 accept 内容敏感的（同配置两次可差 32.8→39.4，均 ALL PASS），
  decode 结论必须结合 accept len 看 server log，只有步时
  （accept÷tok/s）才是稳定口径。

### 6.2 prefill 吞吐 —— `tests/bench_prefill.py`

- **方法**：构造 ~47K token 的合成 prompt（"tok{i}-{rand}" 词表， tokenizer
  实测 ~47,303 token），`/generate` + `max_new_tokens=1`，从响应
  `meta_info.e2e_latency` 取 TTFT——此时 e2e ≈ prefill + 1 步 decode，
  prefill tok/s = prompt_tokens ÷ e2e。`--tokens N` 调整规模、`--iters K`
  迭代次数。
- **用法**：`python3 tests/bench_prefill.py 30001 --tokens 8192 --iters 3`。
- **口径**：47K 参考值 506.9（hybrid 终版）；不同 prompt 长度的注意力成本
  不同（4K/16K/47K 不可直接互比），只比同长度。
- **不校验正确性**（纯速度）；正确性用 6.4/6.5。

### 6.3 CPU MoE 内核微基准 —— `tests/bench_moe_sweep.py` + `tests/kern_test.cpp`

- `bench_moe_sweep.py`：V4 形状（E256/H4096/I2048/top6/gs32）合成
  **E8M0 纯 2 幂尺度**权重（触发 fold 快路径；非 2 幂自动回退 legacy，
  可用于 A/B），走与服务器一致的 CPUInfer 前向。`--tpn` 扫每 NUMA 线程数，
  `--m` 扫批大小，`--check` 对 torch 参考实现比对数值
  （max_rel_err 应 <1e-3；实测 2.7e-4）。脚本已强制
  `KT_HUGEPAGE_WEIGHTS=0`，合成权重不碰持久巨页。
- `kern_test.cpp`：**单核内核试验台**（g++ -O3 -march=native 秒级迭代），
  400MB 权重 arena 超出 L3 → 真实 DRAM 流式；扫 tile 形状（4×4/8×2/…）
  与预取距离（pf16-128）。§5.1 的"4×4 是寄存器上限最优、大 tile 溢出
  反慢 2-4×"即出自它。坑：计时需按模板 MB 数正确折算 MAC（曾有两处
  记账 bug 得出虚高数字）。
- 判读：单核 DRAM 流式 ~87.5（legacy）→ ~120（fold）GMAC/s；
  in-situ 每核 ~65-70。差距 = ragged tail(~12%) + 负载不均 + 相位开销。

### 6.4 正确性 —— `tests/probe_dspark.py`（快）/ `tests/bench_dspark.py`（全）

- probe：3 个短生成（数学/翻译/短文），校验数学结果、翻译可辨、思考切分、
  重复度（8+ 字符 chunk 邻近 60 字重复计数）。**退出码 0=CLEAN** 可接脚本。
- bench：5 个贪心提示词逐条**硬校验**（如乘法结果字符串）+ 吞吐；
  `ALL PASS` 为通过。注意吞吐项 accept 敏感（见 6.1）。
- 每次改放置/内核/相位逻辑后至少跑这两样。

### 6.5 长上下文 —— `tests/grow_probe.py`

- 单会话逐级加长（默认 20/96/112/120K；`--stages=` 自定义），第 1 级
  ~19.5K 处埋暗号，每级做暗号回忆 + 新数学题 + 重复度——区分"长程注意力
  丢失"（暗号丢）与"当步损坏"（数学错/复读）。
- 终版配置跑法：`--stages=20,96,112`（112K 级实际 prompt 125,216 token，
  即验证到 125K 上下文）。~134K 级会 400（超 131072 上限，预期行为，
  脚本未捕获该异常是已知瑕疵）。全程 10-20 分钟。

### 6.6 GPU 侧剖析 —— torch profiler + `tests/analyze_trace.py`

- 抓取：
  ```bash
  mkdir -p /tmp/prof_out
  curl -s -X POST http://127.0.0.1:30001/start_profile -H "Content-Type: application/json" \
    -d '{"output_dir":"/tmp/prof_out","num_steps":40,"activities":["CPU","GPU"]}'
  # 发一个负载请求，等几秒，trace 自动落盘 /tmp/prof_out/*.trace.json.gz
  ```
- 分析：`python3 tests/analyze_trace.py <trace.json.gz> [--window 0.3]`
  （按 kernel 名聚合 GPU 时间；`--window` 跳过 prefill 前段只看 decode）。
  §5.7 定位 matmul_ogs 762µs/层、§5 前期测 GPU 利用率 21%/每层 22ms 空闲
  间隙，均出自此法。
- 已知限制：perf_event_paranoid=4 且无 sudo → perf 不可用，只能靠
  torch profiler（GPU 侧）+ FORWARD_TIME_PROFILE（kt-kernel 相位，需
  重编，用完还原）+ 单核试验台（CPU 侧）三件套拼图。

### 6.7 显存账本测法

- 启动日志里程碑：`Load weight end ... mem usage=`（分目标/draft 两次）、
  `DSV4 memory calculation: bytes_per_full_token=...`（每 token KV 字节，
  直接读）、`Memory pool end. avail mem=`。
- 稳态：就绪后 `nvidia-smi --query-gpu=memory.used --format=csv,noheader`。
- 逐项核算与兑换率见 §5.8。

### 6.8 单元测试 —— `tests/test_routing_v4.py`
GPU 专家路由构造（SparseMatrix 路径）+ **CUDA graph 捕获安全**双校验：
`.venv/bin/python tests/test_routing_v4.py`，末行 PASS 为通过。
改 `_make_routing_data_v4` / triton_kernels 版本后必跑（曾有 torch 回退
版本在图捕获期崩溃的前车之鉴）。

### 6.9 脚本索引（全部集中在本仓库 `tests/`）

| 脚本 | 用途 | 依赖 |
|---|---|---|
| `bench_prefill.py` | prefill 吞吐（47K 口径） | 系统 python3，仅 urllib |
| `bench_moe_sweep.py` | CPU MoE 微基准 + 数值校验 | `.venv`（kt_kernel+torch） |
| `kern_test.cpp` | 单核内核试验台（tile/预取扫描） | g++ -O3 -march=native |
| `probe_dspark.py` | 快速损坏探针（退出码判断） | 系统 python3 |
| `bench_dspark.py` | 5 题硬校验 + decode 吞吐 | 系统 python3 |
| `grow_probe.py` | 长上下文分级探针（到 125K） | 系统 python3 |
| `analyze_trace.py` | profiler trace 聚合分析 | 系统 python3 |
| `test_routing_v4.py` | 路由构造 + 图捕获单测 | `.venv` |
| `test_expert_dist.py` | 专家分布计数/清零/dump 单测 | `.venv`（GPU） |
| `analyze_dist.py` | SIGUSR2 dump 的放置收益分析 | 系统 python3 |
| `sync_leak_check.py` / `hp_weight_check.py` 等 | 既有回归（§3） | 见 §3 |

### 6.10 专家路由分布 —— 真实负载测量（SIGUSR2 dump）

- **开关**：`KT_EXPERT_DIST_TRACK=1`（默认关）。关=热路径每层仅一次布尔
  判断（实测 prefill 506.7 vs 基线 506.9，无差）；开=每层前向 2 个微型
  CUDA 内核（int64 cast + scatter_add 到常驻 [43,256] int64 计数器），
  CUDA graph 可捕获 → **decode/verify 图回放同样计数**。开销 A/B（同机
  同晚、含 QEMU 干扰负载）：prefill 507.8 vs 506.7；bench_dspark 41.10
  vs 42.75（accept 内容敏感噪声带内）；日志步时同 ~66ms。结论：开≈免费。
- **实现**：sglang 分支 `kt_ep_wrapper.py`（计数 + SIGUSR2 handler，
  handler 在关闭时也安装 → 误发 USR2 只写提示文件、不杀进程）+
  `cuda_graph_setup.py`（图捕获 begin/end 钩子；end 清零捕获期的
  dummy 路由，且 dump 在捕获期拒绝执行）。
- **用法**：服务带 env 启动 → 跑真实负载 → `pkill -USR2 -f
  'sglang::scheduler'`（**只发 scheduler 进程**，systemctl kill 会打到
  无 handler 的进程导致退出）→ 读 `/tmp/kt-distribute.txt`：TOTAL/
  DELTA_SINCE_LAST_DUMP 两个 43×256 矩阵 + 每层 SUMMARY。再跑一段负载
  再 dump 一次，DELTA 即该窗口的分布（两次 dump 相减，免停服）。
- **校验口径**：每层 pairs 必须全部相等（每 token 每层恰好 6 对）；
  实测 47,303-token prefill + 447 verify 步 = 299,922 对/层，逐位吻合
  （verify 窗口 6 token×6 专家/步也计入——图回放路径生效的直接证据）。
- **首轮真实负载结论（47K 合成 + 5 题生成）**：连续 ID 0-27 驻留只承接
  **11.3%** 路由量；每层最热 28 专家可承接 **67.0%**（5.9×，零额外
  显存）——若做按热度放置（fork 已有 `frequency` 策略 +
  `--init-expert-location *.pt` 基础设施），是比"换层填满"大一个量级的
  下一步。分析：`python3 tests/analyze_dist.py`（tests/test_expert_dist.py
  为单测）。
