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
（2026-08-22 起为双稳态并存：no-DSpark 5F+28U 在线 / 本节 1F+28U+DSpark
为备选稳态，见 §5.10。）
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
- **处置决策（2026-08-21，用户确认）：BCG 线正式关闭**——判定为上游
  段机制多重损坏、本地不可修，不再投入。留档备查：issue 草稿
  `upstream-bcg-issue.md`（含复现/症状矩阵/12 项排除/诊断开关说明，
  日后若重启此线可直接贴上游）、sglang 分支 `305cc43ca` 的 5 个诊断
  开关与 3 处必要修复（默认零开销，随分支保留）。生产维持 BCG 禁用
  （506.9/47.9 现状）。下一个优先优化候选：按热度专家放置（见 §6.10
  首轮实测 11.3%→67.0% 路由承接、5.9× 余量，零额外显存）。

**生产影响**：ds4f.service 无 BCG 参数，行为不变；cuda-python 12.9.7 下
eager+decode 图路径已验证（bench_dspark 5/5 ALL PASS、短 prompt probe
CLEAN）；长 prefill eager 在 12.9.7 下建议上线前补一轮 grow_probe。

### 5.10 双稳态配置（2026-08-23 更新）：no-DSpark 与 DSpark 并存，均为 3F+40U

**仓库提供两个并存的稳态 unit，二选一部署：**

| 文件 | 配置 | prefill | decode | 适用 |
|---|---|---|---|---|
| `ds4f.service` | 无 DSpark，3F+40U（2368 专家），memfrac 0.90 | ~540（DSpark 口径实测 540.1，§5.13） | ~28-29 | temp>0 业务行为观察期 |
| `ds4f-dspark.service`（当前在线） | DSpark + CPU draft，3F+40U（2368 专家），memfrac 0.95 | 540.1 | bench 41.29（贪心） | decode 吞吐 + prefill 兼顾 |
| （备选）同上下调 3F+28U | 长上下文 prefill OOM 时的回退（减 480 专家 ~6GB） | ~525 | 噪声内相同 | 显存余量敏感时 |
| （历史）5F+28U | 2026-08-23 前稳态；§5.13 证明与 3F+40U 性能无差 | 539.6 | 42.51 | — |
| （历史）DSpark GPU draft 1F+28U，memfrac 0.87 | 上一版 DSpark 稳态 | 506.9 | 47.9（accept×~1.7） | 极致 decode 步时 |

切换（两份文件除 DSpark 相关行外完全一致）：

```bash
sudo cp <选定的 unit> /etc/systemd/system/ds4f.service
sudo systemctl daemon-reload && sudo systemctl restart ds4f
```

二者互斥的物理原因：DSpark draft 占 ~10.6GB，4F+28U+DSpark =
37.56+10.4GB > 48GB 装不下（§5.8），所以 DSpark 稳态只能 1F。
（2026-08-22 起有第三条路：`SGLANG_KT_DSPARK_CPU_EXPERTS=1` 把 draft
专家挪 CPU，draft 显存降到 ~0.9GB，DSpark+多整层成为可能，见 §5.11。）

背景与依据：

- 触发：真实业务反馈开 DSpark 后 temp>0 输出异常（莫名完结、代码混乱）。
  逐层排查结论见 §1.5——采样链路受探针覆盖的部分统计保真，问题轴
  未定位，no-DSpark 稳态用于对照观察。
- **加载期瞬态修复（mxfp4_deepseek.py，只影响加载不影响运行）**：
  整层常驻的 Marlin 打包原为"原始整层 + 打包整层"同时在场（~6.6GB
  瞬态），而加载器在打包前已把全部原始权重放上 GPU，5+ 整层配置下
  打包第 1 层即 OOM。修为逐张量 分配→转换→立即释放（峰值降到单张
  w13 一份 ~2.2GB）。附 `SGLANG_V4_MEMDBG=1` 打点（默认关）。
- **5F/6F 边界**：5F+28U = 2344 专家、目标权重 ~42.5GB，是 48GB 卡
  上限；6F 在 create_weights 阶段即 OOM（即便有上面的瞬态修复）。
- memfrac 0.87→0.90（仅 no-DSpark 稳态）：无 draft 省 ~10.6GB，余量
  喂给专家权重；0.90 是实测可稳定完成图捕获的上限，再往上挤压图
  捕获工作区。
- 折中方案（减 uniform 专家换更多整层，如 3F+17U）实验后放弃
  （log /tmp/exp_3f17u.log）——最终选择保持每层 28 个 uniform 专家
  不动，只用纯余显存叠加整层。

### 5.11 DSpark draft 专家挪 CPU（2026-08-22）：`SGLANG_KT_DSPARK_CPU_EXPERTS=1`

**功能**：开关开启时，DSpark draft（mtp.0/1/2 三个 stage）的 256 个路由
专家全部走 KT CPU 引擎，draft 不再在 GPU 上放专家权重。默认关（维持
纯 GPU draft 现状）。

**收益与代价（实测，1F+28U+DSpark，memfrac 0.87）**：

| 指标 | GPU draft（基线） | CPU draft | 变化 |
|---|---|---|---|
| 稳态显存 | 42.0 GB | **32.2 GB** | **−9.8 GB** |
| decode 步时（accept÷tok/s） | ~66-67.8 ms | ~68.8 ms | +1~3 ms（≈3%） |
| bench_dspark | 38.1-42.75 tok/s | 43.11 tok/s | 噪声带内 |
| 正确性 | — | probe CLEAN + bench 5/5 PASS | — |

accept len 2.1-3.9 正常波动，证明 CPU 上的 draft 专家计算结果正确
（否则 accept 会塌到 ~1）。省下的 9.8GB 约等于 3 个完整专家层
（§5.8 换算）。

**5F+28U+DSpark+CPU draft 实测（2026-08-22，memfrac 0.89）**：启动成功，
稳态 **43.2GB**；probe CLEAN、bench 5/5 PASS。prefill **533.3 tok/s**
（vs 1F DSpark 506.9，+5.2%，追平 no-DSpark 4F 的 532.4）；但 decode
步时 **84.2ms**（bench 36.37 tok/s），比 1F CPU draft 的 68.8ms 慢
~15ms——与 §5.3 规律一致（M=1 下 GPU 专家的 routing 构建 + matmul_ogs
开销主导，整层越多 decode 越慢；3F+17U 当年 71.1ms 同理）。结论：
DSpark+多整层**物理上可行**（0.87→0.89 即可过 KV 分配线 0.8754），
prefill 受益、decode 付出代价，按业务侧重点选用。

**4F/5F 追加实测（2026-08-22/23）**：5F 稳态 43.2GB 余量偏薄曾降为 4F；
随业务默认切贪心（§1.6 末尾，generation_config temperature=0.0），
DSpark 走 folded greedy accept，5F decode 步时从 84.2ms（temp=1.0 eager
采样 accept）降到 **70.1ms**（bench 44.55 tok/s，5/5 PASS）——与 4F 的
70.3ms 持平，5F 重新成为生产配置（memfrac 0.95；KV 池被
--max-total-tokens 封顶，0.95 仅抬高预算线过分配检查，物理占用与
0.89 相同）。注意 bench 数字此时是贪心口径，与历史 temp=1.0 口径
（38-43 tok/s）不可直接比。

**实现要点**：

- 开关：`environ.py` `SGLANG_KT_DSPARK_CPU_EXPERTS`（EnvBool，默认 False）。
  开启时 `draft_worker_common.py` 用 `dspark_draft_cpu_experts_context()`
  替代 `speculative_kt_ep_disabled_context()` 构建 draft。
- `kt_ep_wrapper.py`：`create_kt_config_from_server_args` 在 draft 构建
  全局置位时返回全 CPU 配置——`gpu_experts_mask` 全 False、
  `weight_base_key="mtp.{stage}"`、`wrapper_layer_idx=1000+stage`。
  合成 idx 是为了避开巨页 marker（`L{idx}`）与 target 层 0-2 碰撞；
  sglang 侧簿记（dist 统计、last-layer 逻辑）仍用真实 stage id。
  draft 的 physical_to_logical_map 强制恒等（不借 target 的放置表）；
  draft 不参与专家分布统计（避免污染 target 层 0-2 的统计）。
- kt-kernel（需 `pip install . --no-deps --no-build-isolation` 重装）：
  `KTMoEWrapper`/`BaseMoEWrapper`/`NativeMoEWrapper` 增加
  `weight_base_key` 透传；设置后 safetensors 查找直接用
  `{base}.ffn.experts.*`（命中 `mtp.N.ffn.experts.*`），不再试
  `model.layers.{idx}` 前缀。
- 顺修 `mxfp4_deepseek.py`：0 常驻专家层（全 CPU）跳过 Marlin/TK 转换
  并释放原始占位张量——此前 MARLIN_PARTIAL=1 下 0 专家维度的 repack
  直接 CUDA invalid configuration 崩溃。
- draft 与 target 共享全局 CPUInfer 单例（48 线程/2 池），decode 图
  捕获内嵌 CPU 专家 host node 的模式与 target verify 图相同，已验证。
- 巨页缓存：draft stage 以 L1000/L1001/L1002 键入 persistent hugepages
  （每 stage 2.21GB×2 NUMA），首启冷转换后热启复用。

### 5.12 整层数扫描（2026-08-23）：3F/4F/5F prefill/decode 对比

同一配置只改 `--kt-num-gpu-full-layers`：DSpark + CPU draft
（`SGLANG_KT_DSPARK_CPU_EXPERTS=1`）+ hybrid 28U + Marlin partial +
memfrac 0.95，贪心（bench 显式 temp=0）。三轮均 5/5 PASS。

| 配置 | 稳态显存 | prefill（47.3K tok ×3 均值） | decode（bench_dspark 总计） |
|---|---|---|---|
| 3F+28U | 38.1 GB | 523.0 tok/s | 42.30 tok/s |
| 4F+28U | 41.0 GB | 529.3 tok/s | 40.64 tok/s |
| 5F+28U | 43.9 GB | 539.6 tok/s | 42.51 tok/s |

结论：

- **prefill 随整层数单调上升**：3F→5F 共 +3.2%（每层约 +8 tok/s），
  稳定但有限——与 §5.11 中 1F→5F +5.2% 的趋势一致。
- **decode 对整层数无可分辨影响**：三组 40.6~42.5 tok/s 全在
  bench_dspark 的噪声带内（prompt 4/5 生成长度每次不同，分母不稳；
  4F 最低即噪声）。贪心口径下 decode 瓶颈仍在 CPU 侧专家带宽，
  不再像 temp=1.0 eager accept 时那样随整层数恶化（§5.11 的
  84.2ms→70.1ms 变化）。
- **显存每层约 +2.9GB**（38.1→41.0→43.9）。

选用建议：显存充裕选 5F 买 prefill；紧张时 3F/4F 的 decode 不吃亏。
测量脚本 `/tmp/bench_F_sweep.sh`（逐配置启停 + 两个 bench），原始日志
`/tmp/bench_F{3,4,5}.log`。

### 5.13 自定义放置（custom 策略）与 NF+topN 全扫描（2026-08-23）

**背景**：§5.12 的连续常驻（id0-27）对路由质量的捕获纯属碰运气——按
8-21 真实业务分布快照（`/tmp/expert_hit_probs.csv`，SIGUSR2 dump 导出，
每层 256 专家、每层总 pairs 恒等 1,061,826），连续 28U 每层只覆盖
7~16%。本节实现任意逐层放置并全参数扫描。

**新增启动参数**（sglang dspark-kt，`server_args.py` + `kt_ep_wrapper.py`）：

```
--kt-expert-placement-strategy custom \
--kt-expert-placement-map "0=F,1=F,2=F,3=0-2-4"
```

key = MoE 层序号（0 起，只数 MoE 层）；`F` = 整层上 GPU，否则短横线
分隔的专家 id；未列出的层 0 常驻。custom 下 `--kt-num-gpu-experts`
不再需要（总数从 map 推导）。非法输入（越界/重复 key/残缺列表）全部
启动即报错；mask 下游本来就吃任意布尔形状（frequency 策略先例），
无需其他改动。解析器单测 + "custom 复现 hybrid 5F+28U 逐位等价"已验证。

**转换脚本** `tests/gen_placement.py`：CSV → map spec。

```
SPEC=$(python3 tests/gen_placement.py /tmp/expert_hit_probs.csv \
       --max-experts 1579 --full-layers 0,1,2)
```

`--full-layers` 整层；`--max-experts` = 非整层的常驻名额预算，按
share_of_layer_pairs **全局贪心**选优（每层总质量相等，share 跨层可比，
贪心即最大捕获分配；与每层等额配额实测差 <0.5pt，见下）。spec 走
stdout、统计走 stderr，可直接 shell 替换。

**每层方差**（选择整层的依据）：layer 0-2（hash-MoE 前层）var≈3.5e-6
结构性均匀；layer 3+ 2e-5~7e-5 且随深度走高（最高 37/38/26/36/28）。
方差最小 10 层：2,1,0,19,9,3,5,21,11,10。低方差层任何部分常驻最多吃
23~41%，整层是唯一 GPU 化手段；高方差层 top-28 频率选优即可吃 50~61%。

**NF+topN 扫描**（固定 ~2347 GPU 专家包络 ≈ 44GB；整层按方差最小序列
递进，其余名额全局 top；DSpark + CPU draft + Marlin + memfrac 0.95，
贪心；全部 5/5 PASS）：

| 配置 | 整层 | topN | 预测捕获质量 | 稳态显存 | prefill | decode |
|---|---|---|---|---|---|---|
| 3F+top1579 | 0,1,2 | 1579 | 59.6% | 44.2 GB | 530.6 | 42.79 |
| 4F+top1323 | +19 | 1323 | 57.5% | 44.2 GB | 533.0 | 42.53 |
| 5F+top1067 | +9 | 1067 | 54.8% | 44.1 GB | 533.5 | 40.20 |
| 6F+top811 | +3 | 811 | 51.5% | 44.1 GB | 528.2 | 39.90 |
| 7F+top555 | +5 | 555 | 47.0% | 44.1 GB | 530.5 | 41.26 |
| 8F+top299 | +21 | 299 | 40.1% | 44.0 GB | 529.3 | 43.45 |
| 9F+top43 | +11 | 43 | 26.5% | 43.9 GB | 530.0 | 39.04 |

最干净的对照（同 3F、同 ~44GB 包络，唯一变量 = 频率选优 vs 连续）：

| 3F 对照 | 预测捕获质量 | 稳态显存 | prefill | decode |
|---|---|---|---|---|
| +top1579 频率选优 | 59.6% | 44.2 GB | 530.6 | 42.79 |
| +40U 连续（hybrid，每层 id0-39） | 22.5% | 44.2 GB | **540.1** | 41.29 |

参照（§5.12，连续 28U，捕获 22.2%）：5F+28U = 539.6 / 42.51 tok/s。

结论：

- **放置方式对性能无可分辨影响**：捕获质量 22%~60% 的巨大差异完全不
  转化为速度——prefill 全部落在 528~534 tok/s（含 9 个整层也没有超过
  连续 5F 的 539.6，"整层越多 prefill 越快"在 5F 以上不成立），decode
  全部落在 39~43.5 噪声带。decode 瓶颈不在"多少专家对落在 CPU"（M=1
  下 CPU MoE 边际成本非主导项，步时被 draft/verify、同步、attention
  等固定开销吃掉）。
- 3F 直接 A/B 为此提供了最硬的证据：捕获质量差 2.6 倍（59.6% vs
  22.5%），速度反而是连续版略优（prefill 540.1 vs 530.6，decode 噪声
  内）——540.1 也是全部 9 组实测中的 prefill 最高值。
- 因此在固定包络下**放置选择只看显存预算**；custom + 频率选优的价值
  在于压显存场景（如退回 38GB 以下）捕获质量 3× 于连续常驻，
  degradation 更小。
- 注意频率放置会随业务漂移"腐烂"，需定期重测刷新（SIGUSR2 →
  kt-distribute.txt → CSV → gen_placement.py）。

spec 样例 `/tmp/placement_{3F_1579,4F_1323,5FL_1067,6F_811,7F_555,8F_299,9F_43}.txt`，
日志 `/tmp/bench_{custom3F,4Ftop,custom5FL,6Ftop,7Ftop,8Ftop,custom9F,3F40U}.log`，
扫描脚本 `/tmp/bench_NFtop_sweep.sh`、`/tmp/bench_3F40U.sh`。

### 5.14 HiCache 磁盘 KV 缓存 + prefill 命中统计 + 320K 上下文（2026-08-24）

一次落地三件事（ds4f 稳态，全部实测验证）。上下文先开到 512K 后回收到
320K，过程如下。

**配置变更**（ds4f-dspark.service）：

- 上下文 128K→**320K**（`--context-length 327680`、`--max-total-tokens
  331776`；曾试 512K，见容量调参）；
- 去掉 `--disable-radix-cache`（首部署起的保守默认，无技术原因），tree
  cache 切到 **UnifiedRadixCache**（FULL+SWA 双组件）；
- 启用 **HiCache 磁盘 L3**：`--enable-hierarchical-cache --hicache-ratio 2
  --hicache-write-policy write_through --hicache-io-backend kernel
  --hicache-mem-layout page_first --hicache-storage-backend file
  --hicache-storage-prefetch-policy wait_complete`，存储目录
  `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/var/hicache`（**勿用默认
  /tmp：根分区 97% 满**；/var 是 NVMe 余量 1.1TB）。走的是上游为 DSv4 专门
  实现的 HiCache 栈（`build_deepseek_v4_hicache_stack`，KV+SWA+C4/C128 压缩
  池+DSpark draft 池全覆盖，回归测试自带 DSv4-Flash-DSpark+L3 file 配置）；
- `--enable-metrics`：prefill 命中统计现成——`/metrics` 的
  `sglang:cached_tokens_total`（按 device/host/storage 来源细分）、
  `sglang:prompt_tokens_total`，另有每 batch 日志 `#cached-token` 与每请求
  `meta_info.cached_tokens`。

**容量调参（关键坑）**：开 radix 后 KV 池由 profiled 值封顶
（`kv_cache_configurator.py _profile_available_bytes`：预算 = 权重后 avail
− 47.02GB×(1−memfrac)），memfrac 从此变成真实旋钮：0.95→池 358K、0.97→467K、
0.96+减 2U（38U）→满池 528384。DSv4 池真实边际 **7.7KB/token**（含
swa/c4/c128/state 侧池，比只算 KV 的 5.7KB 大）。长上下文 prefill 的瞬时
占用随上下文增长（~1.15GB 固定 + ~1.9B/token，chunk 1024 并不封顶它）：
38U 时启动 avail 仅 1.82GB，450K prefill 在 ~360K 上下文处 OOM 崩服务；
35U（avail 3.27GB）实测 450K 全程 free ≥1.3GB。容量对照：**35U≈512K、
38U≈400K、40U≈320K**（建议值，各留 ~0.7GB 峰值余量）。因 35U↔40U 无可测
性能差（decode 42.28 vs 41.29 tok/s 噪声带内），最终按用户选择定
**3F+40U + memfrac 0.96 + 320K**：启动 avail 2.66GB，300K 边界压测
714.2s 冷 / 6.7s 热，无 OOM。

**实测结果**：

- 缓存命中：100K 前缀冷 209.4s → 热 3.5s（~60×），
  `cached_tokens_total{cache_source="device"}` 增长；
- **磁盘持久化**：重启后同前缀 2.4s，
  `cache_source="storage_HiCacheFile"` 命中（/var/hicache 落盘 2169
  页/2.4GB per 100K；注意目录属主必须是运行用户，root 创建会 EACCES，
  只在日志报 "Failed to save tensor"）；
- 512K 时代实测（santi.txt/wxkb.txt 各截 450K token，35U 配置）：冷 prefill
  **1190s/1194s**（全程均 ~378 tok/s——吞吐随上下文增长从 ~540 衰减到
  ~320，注意力成本使然），热 **7.5s/7.6s**；2316K 超长请求 HTTP 400 优雅
  拒绝。320K 边界压测（40U，santi 截 300K）：冷 714.2s（~420 tok/s）、
  热 6.7s；
- 回归（radix 新路径零覆盖，重点复验）：bench_dspark 5/5 PASS；
  长生成 seqlen 17075（>13312 SWA 池界）finish=stop、0 retract（§1.7 修复
  在 radix 路径下成立）；流式 session-request 3/3 content 完整（§1.8 修复
  不受影响）；
- 已知代价：启动时间 50s→~80s（host 池分配）；device 池装不下多篇超长
  文档时跨文档靠磁盘层命中（write_through 落盘即持久）；
- 磁盘限额：`SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=200G` +
  `MIN_FREE_SPACE=100G`（默认无上限会长满盘；实测磁盘占用 ~11GB/1M
  token，200G ≈ 18M token 前缀，LRU 超上限驱逐到 90% 水位）；
- 性能复测（干净环境）：decode 42.28 tok/s（历史 41.29 噪声带内，radix+
  HiCache 无可测开销）、热 prefill ~75K tok/s；35U 时冷 prefill 527.9
  tok/s（47.3K，较 540.1 低 2%，补回 40U 后应回到 ~540）。注意 HiCache
  写回线程活跃时会干扰基准（曾测出 32.67 的假象），测性能要等写回落后。

### 5.15 HiCache 手动快照模式（2026-08-25）：`--hicache-manual-mode` + save/load 接口

**动机**：自动 write_through 写回在 prefill 期间异步落盘会干扰基准（§5.14
末尾的 32.67 假象），且快照时机不可控。改为显式手动控制：自动保存/加载/
限额/自动清理逻辑全部保留但默认不触发。

**实现**（third_party/sglang）：

- `server_args.py` 新增 `--hicache-manual-mode`（默认 False）。开启后：
  `_finish_write_through_ack`（unified_radix_cache.py，唯一自动 H→磁盘
  写回点）与 `scheduler._prefetch_kvcache`（唯一请求到达自动预取点）被
  旁路；host 层、load_back H→D、LRU/限额逻辑不变。
- 两个管理接口（均要求 `is_fully_idle()`，非空闲返回 400）：
  - `POST /hicache/snapshot/save {"path": ...}`：排空 D→H 后 DFS 整树，
    把所有 backuped 节点的页（KV+SWA+C4/C128+state+draft 全池）写入
    `{path}`（新建独立 HiCacheFile 实例，max_size=0 不自清，不动主
    backend），并把树结构写 `{path}/manifest.json`；
  - `POST /hicache/snapshot/load {"path": ...}`：先 flush（清树/device/
    host，不动磁盘），再按 manifest 父先子后逐节点复用
    `prefetch_from_storage` 灌回 host 并挂树（期间临时把
    storage_backend 指向快照目录、threshold→1、容量限制放开，finally
    恢复）；
  - 清空 GPU 侧用现有 `POST /flush_cache`，不新增。
- **关键约束**：磁盘页文件名是纯内容哈希（`{sha256链}[.{pool}]{suffix}.bin`），
  文件本身不含 token 序列，离开 radix 树无法反解前缀——manifest 因此必须
  存树结构（紧凑格式：每节点只存自己的 token_ids + parent 下标，父先于子，
  O(总 token 数)）。实现见新文件 `mem_cache/hicache_snapshot.py`（纯
  stdlib，原子写，10 例单测 `tests/test_hicache_snapshot_manifest.py`）。
- 配置配套（ds4f-dspark.service）：加 `--hicache-manual-mode`；限额 env
  `SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE/MIN_FREE_SPACE` 注释关闭（manual
  模式下无自动写，限额失去意义；恢复快照外自动模式时再开）。

**典型工作流**：跑业务/prefill → 空闲时 snapshot/save 到指定目录 → 随便
测别的（/flush_cache 清场）→ 需要时 snapshot/load 秒级恢复现场。快照目录
整体可复制归档（内容寻址，同前缀跨快照天然去重）。

**实测**（2026-08-25，3F+40U + 320K 稳态，276K token 快照）：

- manual mode 下多轮 prefill 全程 /var/hicache 文件数不变（36047），自动
  写回确已关闭；请求到达也不再自动预取（重启后同前缀需显式 load）；
- save：540 节点 / 275,968 token / 4690 页 / 2.38GB，**0.72s**（页已在
  host 池，纯 H→盘）；二次 save 幂等去重（只写新增页）；
- flush_cache：清树/device/host，磁盘不动；
- load：**275,968 token 全量回挂在 0.13~2.8s**（NVMe 冷/热页缓存差异），
  回放后同前缀请求 TTFT **1.1~1.2s**（对照：全冷 prefill ~9 分钟）；
- save → flush → load → 请求 完整往返两轮稳定，probe_dspark CLEAN；
- 修掉的两个 bug（初版实测暴露）：(a) 逐节点回放会被 TRAILING_PAGES 池
  （swa/state，只有窗口尾部页有文件）的尾部存在性门控全部否决——回放粒度
  改为 root→leaf 整路径（对齐正常请求的查询语义）；(b) manifest JSON 出来
  的 list 直接挂树会与正常请求的 `array("q")` key 撞上
  `RadixKey.match` 的严格类型断言（Scheduler 崩）——回放全程保持 array。
- 注意：回放后树结构是"每路径一个合并大节点"（再次 save 时 nodes=2），
  前缀匹配语义等价；快照目录 `/var/hicache-snap1`（2.4GB）留作样本。

### 5.16 思考等级默认拉满 + 默认 temperature 回 1.0（2026-08-25）

- **思考等级**：DSv4 编码器（`encoding_dsv4.py`）本就有 `reasoning_effort`
  概念——请求体可传 `reasoning_effort`（本模型 checkpoint 检测为
  official profile：**low / high / max** 三档，靠 system prompt 注入不同
  努力程度指令实现）；请求不传时读 env `SGLANG_DSV4_REASONING_EFFORT`
  兜底（serving_chat.py:1225-1237）。checkpoint 自带默认是 **low**。
  现稳态在 ds4f-dspark.service 设 `SGLANG_DSV4_REASONING_EFFORT=max`，
  即默认最高档；单请求可用 `reasoning_effort: "low"/"high"` 降级。
- **默认 temperature 回到 0**：bench 复现期曾把模型目录
  `generation_config.json` 改成 `temperature: 0.0, do_sample: false`，
  2026-08-25 白天短暂恢复过 1.0（`.bak` 原值），同日按用户决定**改回
  0.0**（确定性输出优先；请求显式传 temperature 仍优先）。

### 5.17 上下文档位调整、并发 2 与 DSpark 长上下文门控（2026-08-25）

一天内走了 320K→512K→（短暂 256K 作罢）→**512K+35U+并发2+DSpark
门控**的最终形态。过程：上午从 40U+320K 改到 **35U+512K**（§5.14 已
验证的组合）并把 `--max-running-requests 1→2`；实测 512K 期间发现
**DSpark 在 >256K 上下文输出不稳定（重复字符/复读倾向明显增加，根因
未查——与 §1.7 已修复的 SWA 驱逐 bug 是不同现象）**；一度准备把上下文
上限压回 256K，最终改为动态门控方案，上下文保持 512K。

**DSpark 长上下文门控**：`SGLANG_DSPARK_MAX_CTX`（默认 **262144**，
0=永不禁用；实现在 `dspark_worker_v2.py`，单测
`tests/test_dspark_max_ctx.py`）。batch 内任一请求上下文达到阈值 → 整
batch 退化为普通 eager decode（越线请求只升不降；长请求结束后 batch
自动恢复投机，日志 `degrading to plain target decode` /
`resuming speculative decode`）。切换点在 worker 的 `_forward_decode`
顶部（`_forward_decode_plain` 补齐 input_ids/out_cache_loc、target
forward + `inject_target_hidden` 保持 draft KV 同步，verify 图宽度
不匹配自动走 eager）。per-request 混合判定为不可行（ragged verify 里留
越线请求仍走 verify 专用 attention 路径——>111K 损坏史（§9.3）恰恰出
在那里），故取 batch 级语义。

GPU 验证（阈值临时设 4096）：(a) prompt 7999 越线 → 首个 decode 步即
降级，accept len 1.00、eager、~12.4 tok/s；(b) prompt 2626 + 长生成
中途跨过 4096 → 日志在 ctx_lens=[4097] 处准点降级；(c) 越线请求结束
后同 batch 恢复投机（accept len 回升 5.8+、83 tok/s）；(d) 降级路径
输出质量正常（8K 文档总结请求，思考链/正文/收尾全正常）；(e) 修掉
一个首测崩溃：spec 路径 `sampling_info.penalizer_orchestrator` 恒为
None（forward-only 副本），降级步不能引用它。**已知代价**：降级步
~12.4 tok/s（eager + 每步 draft KV 注入，比非投机图基线 ~34 慢）——
长上下文求稳的代价；越线请求的 draft KV 成为死重（请求结束才释放）。

512K 档实测备查：池 528384 token、启动 avail 3.21GB、verify 图按
bs=[1,2] 捕获、双请求并行 decode 生效（日志 `#running-req: 2`）。

**各上下文档位的完整参数组合**（改动时三个参数一起换，memfrac 0.96 不动）：

| 场景 | `--kt-num-gpu-experts` | `--context-length` | `--max-total-tokens` | 启动 avail |
|------|------------------------|--------------------|----------------------|-----------|
| 512K（当前，配 DSpark 门控） | 35 | 524288 | 528384 | ~3.2GB |
| 400K | 38 | 409600 | 413696 | ~2.7GB（推算：38U+512K 实测 1.82GB + 池缩小 0.86GB） |
| 320K | 40 | 327680 | 331776 | ~2.66GB |
| 256K | 40 | 262144 | 266240 | 充裕（池比 320K 档还小 0.5GB） |

320K 档（40U）即 2026-08-24 的稳态：池 331776 ≈ 2.56GB，300K 边界
压测 714.2s 冷 / 6.7s 热通过（§5.14）。并发注意：KV 池由
`--max-total-tokens` 封顶、所有并发会话共享，两条都接近满上下文的请求
会排队等池空间（按可用 token 准入，不会 OOM）。

> **2026-08-25 晚补记（上游拉齐后）**：本节记录的 ">256K 输出不稳定
> （根因未查）" 头号嫌疑即上游 `4a5d7d3` —— DSv4 speculative decoding
> 在 **draft tokens > 4** 时的静默 KV / compressed-state 损坏；本机
> `speculative_num_draft_tokens=6` 恰好踩中，且上下文越长损坏概率越
> 可观测。该修复已随 §5.18 合并进入生产，**升级后 grow_probe 240K/280K
> 级已复测干净**（详见 §5.18）。`SGLANG_DSPARK_MAX_CTX` 当前仍置 0
> （门控关闭）；若后续业务长上下文持续无异常，可保持 0 跑满投机收益，
> 也可恢复 262144 默认值作为保险丝（两种都合理，按风险偏好选）。

### 5.18 上游拉齐（2026-08-25 晚）：merge sgl/main d06762282，带入 DSpark/DSv4 三修复

动机与范围：把 `third_party/sglang`（`dspark-kt` 分支）从 8-06 基底
（`4ad990ba7` + 25 个自研提交）merge 官方 main `d06762282`（+986 提交），
目标修复——`4a5d7d3`（**DSv4 投机解码 draft tokens > 4 的静默 KV /
compressed-state 损坏**，本机 6 draft tokens 踩中，§5.17 ">256K 不稳定"
的头号嫌疑）、`8549cce`（c128 ragged prefill 压缩计划 GPU 竞态）、
`154f0ac`（DSpark + DP/EP metadata）。`eb4f9c2`（SM120 mHC/NCCL buffer
alias）为 SM120 专属，本机 SM89 跳过。merge commit `0c7838d6c`（submodule），
父仓库指针 `42b3fd3`。

合并要点（11 个冲突文件，详见 merge commit message）：

- 三个修复承载文件（`c_plan.cuh` / `deepseek_v4_memory_pool.py` /
  `pool_configurator.py` / `dspark_draft.py`）与上游逐字节一致，零覆盖；
- `dflash_info_v2.py` 取上游 `page_aligned_decode_alloc_lens` 重构，§1.7
  的 SWA 驱逐修复存活（重复调用已去重）；
- DSML 流式检测器（§1.8 修复）保留 fork `normal_parts` 方案 + 采纳上游
  异常路径加固（清 buffer 防重发），功能级单测通过；
- `paged_mqa_metadata.cuh` 取上游批次自适应三 kernel 重写，SM89 smem
  钳制（§1）被取代（新设计 tiny/small 静态 smem + 大 batch workspace
  scratch，天然 SM89 安全）；
- fork 全部 25 个自研提交（KT 引擎、max-ctx 门控、hicache 手动快照、
  SM89 路径等）与所有自研文件完好。

sglang-kernel 版本门（dspark-kt `3c4c5c59f`）：上游启动期硬性要求
`>= 0.4.6.post1`，但 PyPI 该版本 wheel 是 CUDA 13 构建（`libcudart.so.13`，
需 580+ 驱动），本机 550 驱动无法加载；fork 放宽下限到 0.4.5（本地
`0.4.5+cu129` 构建）。符号级审计：排除 NPU 路径与 aot 构建树后，运行时
代码引用的全部 `sgl_kernel` 符号在 0.4.5 均存在。长期项：用
`python/sglang/kernels/aot` + CUDA 12 工具链本地构建 0.4.6+cu129 后
恢复上游下限。

依赖取舍：torch 2.11.0+cu128（上游 pin 2.13.0，受驱动约束保留）、
sglang-kernel / humming-kernels / tilelang / quack-kernels / cuda-tile
维持现装版本（懒加载可选路径，深度 import 链验证不触发）；新增 termcolor、
setuptools-scm；flashinfer 维持不装（sgl-kernel 自带 flash_ops）。
备份：submodule 分支 `backup/dspark-kt-pre-sync-20260825` +
`~/sglang-upgrade-backup-20260825/`（前后 pip freeze）。

升级后验证（2026-08-25 晚，生产 30000 实例，512K+35U+DSpark+CPU draft）：

- 语法/导入：合并涉及全部 .py 编译通过；serving 全链 import 通过；
- `tests/probe_dspark.py 30000`：**CLEAN**；
- `tests/bench_dspark.py 30000`：**5/5 ALL PASS**，预热后 **37.33 tok/s**
  （首跑 29.89 含 JIT 预热；参考区间 32–36，勿把单次吞吐当回归）；
- `tests/test_dspark_max_ctx.py` / `tests/test_hicache_snapshot_manifest.py`
  （10 例）/ `tests/test_routing_v4.py`（含图捕获）：**全 PASS**；
- 启动链路：hugepages 权重 REUSED（target 31.9s / draft 6.9s），verify
  图 bs=[1,2] 捕获正常，`spec_num_draft_tokens=6` 下 verify 37 次无异常；
- 长上下文 `grow_probe.py 30000 --stages=8,120,240,280`（跨 >111K 历史
  损坏边界与 §5.17 >256K 不稳定区）：**4/4 全 PASS**（codeword 回忆 /
  数学 / 重复度逐级干净，exit=0）。280K 级为升级后首次在 >256K 上下文
  得到干净输出——印证 §5.17 补记的 `4a5d7d3` 嫌疑判断；探针期间
  spec_accept_length 3.4 / verify 971 次无异常。prefill 计时备查：
  120K 新前缀 235s、120K→240K 增量 303s、240K→280K 增量 114s
  （hicache 前缀复用生效）。

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

### 1.5 temp>0 分布保真排查（2026-08-22）：采样路径无统计学偏差

起因：真实业务反馈开 DSpark 后输出"莫名其妙完结、代码混乱"。对
temp=1.0 采样做了逐层排查（脚本在 /tmp：`dist_prefix.py` / `dist_prefix2.py` /
`dist_cmp.py` / `finish_test.py` / `test_chain_kernel.py`，置换检验 N=400-800）：

- **accept kernel**（`chain_speculative_sampling_triton`）：受控 p/q 单测与 CPU
  精确参考逐位一致，TV=0.0000。
- **folded 图内 draft 采样**（`SGLANG_DSPARK_FOLDED_SAMPLING=0` 对照）与
  **torch.multinomial 采样路径**（`SGLANG_DSPARK_FAST_SAMPLING=0`）：均无影响。
- **Marlin/GPU 专家**：全 CPU verify（`SGLANG_KT_GPU_EXPERTS_PREFILL_ONLY=1`）
  下结论不变，排除。
- **固定前缀第 1 token**（extend 路径，N=800）：on vs off TV=0.068，
  p=0.52，无差异。同配置重复采样噪声地板 TV≈0.10（p=0.66）。
- **第 2 token 联合分布**（verify 5-token 前向路径，N=800）：TV=0.101，
  p=0.115，无差异。
- **提前完结**（数到 30 任务，N=100）：on/off 均 100% finish=stop、
  长度恒定 60 token，无差异。
- 早期"链深位置分布偏移 p=0.0005"被证伪：**两条合法 decode 路径**
  （全 CPU decode vs Marlin decode，都关 DSpark）之间差异同样 p=0.0005
  ——该指标会把微小数值差沿轨迹混沌放大，不能用作 bug 判据。

结论：DSpark 的 extend/verify/accept/采样全链路在受探针覆盖的场景下
统计保真。**未覆盖的轴**：thinking 开启、流式、工具调用、并发批处理。
若业务问题重现，需要一份具体失败样本（prompt、采样参数、是否流式、
并发度、输出）来定位。

### 1.6 EOS 收尾专项审计（2026-08-23）：算法无 EOS 改写，贪心全上下文干净

起因：业务（temp=1.0）反馈句尾多出残片（"Three.js + add"）/ 突然完结，
仅开 DSpark 时出现。两轮代码审计 + 三组探针：

- **算法链无 EOS 改写**：draft 采样（dspark_draft.py / SampleStepTokens）、
  accept 内核（accept_greedy / 链式拒绝采样）、bonus 落位
  （BuildOutTokens）、commit 切片（accept_lens=correct_len+1）、scheduler
  收尾（_check_token_based_finish 逐 token 扫描本步全部接受 token、
  finished_len 截断）均为教科书实现，无任何 EOS 屏蔽/替换/跳过逻辑。
- **"阈值"答疑**：chain_speculative_sampling_triton 的
  threshold_single/threshold_acc=1.0 实际未被 classic 内核使用（形参
  保留），拒绝判定就是标准的 coin·q < p；confidence/cap 阈值只缩 verify
  窗口（影响速度），截断处正确重取 target 分布的 bonus，不改分布。
- **folded 图内 accept 只走 greedy**（dspark_worker_v2.py 有显式守卫：
  `sampling_info.is_all_greedy` 才 fold，temp>0 批次强制 eager accept）。
- **temp=0 A/B**（/tmp/greedy_battery.py，8 题；同 4F+28U 放置）：DSpark
  vs 无 DSpark 内容 7/8 分叉但双侧均连贯、正常 stop——分叉是 verify 窗
  前向与逐 token decode 的数值差沿轨迹放大，非 bug 特征。
- **长上下文收尾 A/B**（/tmp/longctx_probe.py，temp=0，一句话总结任务）：
  18K 双侧干净收尾；DSpark 侧 61.5K、111.5K 亦干净（finish=stop、
  语义正确、无残尾）。

结论：temp=0 下 0~111K 上下文 DSpark 收尾无缺陷；EOS 被改写/吞掉的
假设证伪。业务 temp>0 症状若仍复现，剩余嫌疑集中在**采样接受路径在
真实业务参数组合下的保真度**（§1.5 未覆盖轴）。判别实验：同一批句尾
场景 N 次，统计"K token 内以句号/EOS 收尾"比例，DSpark 开/关对照。
另可开 SGLANG_DSPARK_ENABLE_SPS_RECORD / STS_COLLECT_PATH 在生产抓取
真实失败样本。

**后续处置（2026-08-23）**：模型目录 `generation_config.json` 改为
`temperature=0.0, do_sample=false`（原件备份 `generation_config.json.bak`），
不显式传 temperature 的请求默认走贪心——同时让 DSpark 落在验证最充分的
folded greedy accept 路径上。实测：默认值与显式 temperature=0 行为一致
（低/中熵 prompt 三次输出逐字相同）；高熵开放题仍有逐 run 分叉（内核
非确定性的近似平票翻转，非 bug）。显式传 temperature 的客户端不受影响。
回退：恢复 .bak 并重启 ds4f。

### 1.7 DSpark decode 期 SWA 窗口驱逐缺失（2026-08-23）：长生成 retract abort 的根因与修复

症状：生产（3F+40U DSpark）长生成时报
`Out of memory even after retracting all other requests in the decode
batch. Aborting the last request.` journal 显示同类 abort 自 8-21 起已有
5 次（4F/5F 时代同样存在，非 3F+40U 引入）。

根因链：

- DSv4 混合注意力分 full/swa 两 KV 池；`swa_full_tokens_ratio` 对
  DeepseekV4 自动取 0.1，`--max-total-tokens 135168` 封顶 full 池
  → **swa 池仅 13312 槽**。
- decode 期 SWA 窗口驱逐只在两条路径存在：非投机
  `alloc_for_decode`（allocation.py:547）与 EAGLE
  （`eagle_prepare_for_decode` 开头一行 `batch.maybe_evict_swa()`）。
- **DSpark/DFLASH 路径绕过了二者**：
  `prepare_for_decode → spec_prepare_for_decode → is_dflash_family()
  → DFlashDraftInputV2.prepare_for_decode`（dflash_info_v2.py），
  后者从不调 `maybe_evict_swa()`，也从不递增 `req.decode_batch_idx`
  （而驱逐闸门要求 `decode_batch_idx >= 1`）。
- 后果：DSpark 下长生成的 swa 足迹 ~1:1 随 seqlen 增长，越过 ~13K
  即 abort。铁证：同进程探针请求 full=111616 时 swa 仅 2304（增长在
  prefill 期、prefill 驱逐正常），出事请求 full=14080 时 swa=13312
  （0.95×seqlen，零驱逐）。

修复（third_party/sglang `dflash_info_v2.py`
`DFlashDraftInputV2.prepare_for_decode`，镜像 EAGLE）：

1. 方法开头加 `batch.maybe_evict_swa()`；
2. per-req 循环内 `req.decode_batch_idx += 1`。

> 2026-08-25 上游拉齐注记（§5.18）：上游同期独立加了同样的
> `maybe_evict_swa()`（放在 `bs==0` 早退之后），merge 时两处已去重
> （保留上游位置 + 本节注释）；`decode_batch_idx` 递增改由上游重构后的
> 循环后统一 tick 承担，语义不变。单测 + bench 复验通过。

安全性：驱逐只动窗口外槽位；DSpark reserved_len = committed +
2×block_size 的 over-allocation 语义不受影响。

验证（30001，DSpark+CPU draft+3F+40U+memfrac 0.95，/tmp/bench_swa_fix*.sh）：

- "数到 3500"：finish=stop，seqlen 9543，swa 占用峰值 0.04~0.08
  （修复前同位置 ~0.7），0 retract；
- **越界决定性用例"数到 6000"**：finish=stop，seqlen **17047 >
  13312**，输出完整至 6000，全程 swa 占用 0.04~0.08，0 retract；
- 正确性回归 bench_dspark 5/5 PASS（42.83 tok/s）。

### 1.8 流式 tool_call 响应 content 截断（2026-08-24）：detector 丢弃前导文本

症状：业务（streaming）content 在 tool_call 前突然中断，如
"…then build the "（半截子句）、"…download Three"（截在词中）。
仅开 DSpark 时出现。

排查与定位（level-3 请求日志抓现行，日志含原始 text + output_ids）：

- 非流式复现 0/57；token 层面 `" Three"`(13475) 与 `".js"`(15135)
  是独立 token，一度怀疑模型自选断点。
- `--log-requests --log-requests-level 3` 抓到三类实例对比：
  - fin2：原始 token 流含完整 "…then build the project.\n\n"，但客户端
    收到 "…then build the " —— **服务端在流式路径丢了 "project."**；
  - fin6/fin7：原始 token 流本身就是 "…Let me fix it\n\n" + DSML ——
    模型在该贪心轨迹上自主选择直接进工具调用（无句读），服务端忠实转发。
- 根因（deepseekv32_detector.py `parse_streaming_increment`，v4 detector
  的父类）：当某个流式 delta 同时含 content 尾部与 DSML 起始标记时，
  进入 invoke 匹配循环后**返回值恒为 `normal_text=""`**，缓冲区里
  标记前的正文被静默丢弃。
- 为何只在大块 delta 下出现：非投机逐 token 流式时 DSML 标记几乎总是
  独占一个 delta（正文此前已流出）；DSpark 每步提交 ~5 token，
  "project.\n\n<｜DSML｜tool_calls>" 常落在同一 delta → 稳定触发。

修复（同文件，invoke 匹配循环开头）：找到缓冲区中第一个 `"<｜DSML｜"`
标记，将其前的正文剥离（并像 early-return 路径一样剥掉 eot/invoke_end
残留标记）后随 `normal_text` 返回；异常路径同理保留已剥离前缀。

验证：

- 单元级（直接驱动 DeepSeekV4Detector）：逐字符、整段单 blob、30 组
  随机 burst 切分、双工具调用夹正文，33/33 通过——修复前 single-blob
  场景 content 返回空串；
- 实机流式 10/10 content 完整收尾、tool args 为合法 JSON（其中 1 例为
  模型发并行双 invoke，客户端拼接显示问题，非服务端）；
- 模型侧"裸断句"类（fin6/fin7）仍会偶发，属贪心轨迹选择，非丢 token。

注：排查期间在 ds4f.service 临时加了 `--log-requests --log-requests-level 3`
（全量输入输出+token ids 落 journal，量大），确认无复发后可撤掉。

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
