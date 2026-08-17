# DSv4F-Opt — DeepSeek-V4-Flash decode 性能优化记录

> 环境：RTX 4090D 48GB（改装卡，显存半速，见 §2.2）+ 2×EPYC 9275F（每 NUMA 24 物理核），
> 全部 256 专家驻留 CPU 巨页，单流 decode（`--max-running-requests 1`）。
> 日期：2026-08-17。

## 1. 成果

| 指标 | 基线 | 优化后 | 提升 |
|---|---|---|---|
| 单流 decode 吞吐（稳态，256 token，忽略前 16） | 23.36 tok/s | **≈30.0 tok/s** | **+28%** |
| 每 token 延迟 | 42.8 ms | ≈33.4 ms | −22% |

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

### 3.6 env 覆盖机制（使上述无需改 unit 文件）
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

正确性验证：太阳系八大行星介绍正确；`12×11=132`、`长江约6300公里`；
3431-token 长上下文 prefill（走 mat-mat 修改路径）答案正确（合成数据答案 "0,0"）；
vpermb 解码位级单测全过。
（注：vpermb 首版有两处 bug——LUT i=8 项抄错、vpermb 参数顺序反了——靠输出质量检查+单测发现，
已修正；教训是此类位操作改动必须有位级测试。）

## 5. 生产启用方式（当前已生效）

代码在 venv 与仓库中，env 文件 `~/.config/sglang/env`：
```
SGLANG_OPT_FUSE_WQA_WKV=1
SGLANG_OPT_USE_JIT_NORM=1
SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=1
SGLANG_OPT_USE_FUSED_STORE_CACHE=1
SGLANG_OPT_USE_OVERLAP_STORE_CACHE=1
SGLANG_OPT_MXFP4_FUSE_RSF_SHARED_ADD=1
SGLANG_KT_WOA_FP8_TRITON=1
SGLANG_EXTRA_ARGS=--kt-cpuinfer 48
```
- 回滚单项：删除对应行；回滚内核：`cd kt-kernel && git checkout <旧commit> && ./install.sh build`。
- 等价的 unit 文件写法（若想显式化）：在 `/etc/systemd/system/ds4f.service` 加
  `Environment=SGLANG_...=1` 各行并把 `--kt-cpuinfer 44` 改 48。

## 6. 未做的后续机会
- ue8m0 scale 以 uint8 存储（当前转 float32 存，占 MoE 权重带宽 ~17%）+ 整数指数加法应用
  ——需改 BufferB 布局与巨页缓存指纹。
- GEMV 8 行分组提高每线程 MLP；scale 与权重交错布局消除独立 scale 流。
- ratio-0 的第 0/1 层 wo_a 仍未走 FP8 路径（~0.3 ms/token）。
- 本机 systemd `StartLimitBurst=12/天` 已用 11 次，当日仅剩 1 次自动重启额度。

## 7. 相关文件
- 基准：`bench_decode.py`；线程占用：`thread_util.py`；profiler 分析：`analyze_prof*.py`
- GPU 微基准：`bench_gpu_ops.py` / `bench_gpu_cold.py` / `scan_w8a8_cfg.py`
- CPU MoE 微基准：`kt-kernel/bench/bench_fp4_moe_cold.py`（随机路由，L2 冷）
- 实验实例脚本：`run_exp.sh` / `stop_exp.sh`（与生产并行，30001 端口，共享巨页权重缓存）
