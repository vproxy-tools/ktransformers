# Upstream issue draft: Breakable CUDA graph (BCG) prefill replay broken on DeepSeek-V4-Flash with KT hybrid CPU/GPU MoE

> 留档：BCG 线已于 2026-08-21 经用户决策正式关闭（DSv4F-Opt.md §5.9），
> 本草稿不再计划提交；若日后重启该线，贴上游前需先把含诊断开关的分支
> （`305cc43ca` 及祖先）推到公开 fork。对应本地调查记录：DSv4F-Opt.md
> §5.9（三轮证据链）。

---

**Title:** [Bug] Breakable CUDA graph (BCG) prefill replay silently corrupts output / illegal memory access on DeepSeek-V4-Flash (DSV4 + KT hybrid MoE), while eager fallback is correct

## Environment

- sglang: fork `dspark-kt` @ `305cc43ca` (contains all diagnostic switches below; model-side integration commits `d33777f70`, `df5e8c156`)
- torch 2.11.0+cu128, cuda-python/cuda-bindings 12.9.7 (cu12 bindings required on driver 550 — cu13 bindings gate everything behind CUDA_ERROR_INSUFFICIENT_DRIVER 35, even CUDA-10-era APIs like `cudaStreamGetCaptureInfo`)
- Driver 550.144.03, RTX 4090D (SM89), single GPU, TP=1
- Model: DeepSeek-V4-Flash 0731 — 43 MoE layers, 256 routed experts/top-6, hash-MoE layers 0–2, DSA-style attention (dsv4 backend), SWA + c4/c128 compressed caches
- MoE execution: ktransformers hybrid CPU/GPU (CPU experts via `kt_kernel` AVX512, partial Marlin GPU residents: 28 experts/layer + 1 full layer), `moe_a2a_backend=none`
- `--speculative-algorithm DSPARK` off in the minimal repro (fails with and without it)

## Reproduction

```bash
python -m sglang.launch_server \
  --model /path/to/deepseek-v4-flash-0731 \
  --kt-weight-path /path/to/deepseek-v4-flash-0731 --kt-method MXFP4 \
  --kt-expert-placement-strategy hybrid --kt-num-gpu-full-layers 1 \
  --kt-num-gpu-experts 28 --kt-cpuinfer 48 --kt-threadpool-count 2 \
  --context-length 131072 --chunked-prefill-size 1024 --max-prefill-tokens 1024 \
  --cuda-graph-backend-prefill breakable --cuda-graph-bs-prefill 1024 \
  --mem-fraction-static 0.87 --max-total-tokens 135168 \
  --disable-radix-cache --max-running-requests 1 --skip-server-warmup \
  --attention-backend flashinfer --disable-shared-experts-fusion \
  --trust-remote-code --disable-decode-cuda-graph   # decode graphs irrelevant
# NOTE: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True changes the failure
# mode from silent corruption to illegal memory access (see below).
```

Requests (any of these reproduces):

1. Single-chunk, zero prefix (~819 real tokens, padded to the 1024 bucket,
   `Prefill batch ... cuda graph: True` in the log):
   `"Background data: item0 item1 ... item399\n\nQuestion: What is 17*23? Answer with just the number."`
   → **empty output** (immediate EOS). Eager fallback answers `391`.
2. Multi-chunk: a staged long-context probe (20K tokens) → long-range recall
   fails (`codeword=False`) while current-step math stays correct, no repetition.
3. Same requests against the identical server **without**
   `--cuda-graph-backend-prefill breakable` → all pass (with and without
   `expandable_segments`).

## Symptom matrix

| Config | Real graph replay result |
|---|---|
| bucket 1024 + `expandable_segments:True` | illegal memory access; async, surfaces at the eager logits tail (or at the first post-replay triton launch) |
| bucket 1024, no expandable_segments | silent corruption (single chunk: empty output; multi chunk: prefix recall lost) |
| bucket 512 | silent corruption (garbage symbol output) |
| eager fallback (any) | correct |

With per-segment fault-localizing syncs (see `SGLANG_BCG_DEBUG_SYNC` below) the
illegal access is deterministic: `segment 0 ok → attention break ok →
segment 1 ok → MoE break ok → replay segment 2 → fault`. That is, the very
first recorded segment after the first MoE eager break. Strong-referencing the
break args (`SGLANG_BCG_NO_WEAK_REF=1`) moves the surfacing point but does not
fix it → the dangling buffer is not one of the break arguments; it is a
segment-to-segment intermediate. Faulting segment kernel list (via
`SGLANG_BCG_DEBUG_KERNELS`):
`vectorized_elementwise(Mul)` → `CUDAFunctor_add` → `mhc_post_tilelang_kernel`
→ `mhc_pre_gemm_sqrsum_splitk` → `mhc_pre_big_fuse_with_norm_tilelang` →
`per_token_group_quant_flat` → `_w8a8_block_fp8_matmul` (×2) →
flashinfer rmsnorm → `fused_q_norm_rope` → `index_elementwise` +
`direct_copy(cast)` → `fused_k_norm_rope_flashmla`.

## Model-side integration we had to add (included in the fork)

1. `KTEPWrapperMethod.apply` (KT hybrid MoE forward) wrapped with
   `@eager_on_graph(True)` — its host choreography (CPU expert submission,
   routing metadata rebuild, CPU/GPU sync) must re-run every replay, else
   replay consumes capture-time state.
2. `MqaAttentionBase._compute_kv_to_cache` wrapped with `@eager_on_graph(True)`
   — the fused k-norm+RoPE writes the paged KV cache directly
   (`set_swa_key_buffer_radix_fused_norm_rope` with `swa_loc` derived from
   `forward_batch.out_cache_loc`); recorded into a segment it replays with
   capture-time slots.
3. Upstream bug fixed in `_weak_ref_if_tensor` (breakable_cuda_graph.py):
   NamedTuple args (e.g. `StandardDispatchOutput`/`TopKOutput`) were degraded
   to plain tuples, so break replay fns failed with
   `'tuple' object has no attribute 'hidden_states'`. Now reconstructed via
   `type(x)(*elems)`.

Even with 1–3, replay remains broken — the residual fault appears to be inside
the recorded-segment machinery itself (allocation lifetime between manual
`capture_begin`/`capture_end` cycles vs graph-pool use-counting, or the DSV4
model's cross-layer deferred state: `hc_post`/`hc_pre` fused norms thread
`residual/post/comb` tensors between layer iterations).

## Ruled out (12 hypotheses, all tested on the minimal repro)

dedup (default off) · capture-time memory pressure (memfrac 0.78, 11.5 GiB free
— identical failure) · DSPark on/off · multi-tier vs single-bucket capture ·
weak-ref reclamation (`SGLANG_BCG_NO_WEAK_REF=1`) · overlap-scheduler race
(`--disable-overlap-schedule`) · `MAX_SEQ_LEN_FOR_CAPTURE` clamping (=131072) ·
metadata refresh coverage (`refresh_for_breakable_cuda_graph_replay_` fields
audited) · reference-swap → in-place copy of `reference_assign_fields`
(`SGLANG_BCG_REFRESH_INPLACE=all`, two-pass safe copy — identical failure) ·
tilelang indexer swap (untestable: torch fallback OOMs at 131K ctx during
capture) · tier size · "prefix-specific" corruption (single chunk breaks too).
Also: compute-sanitizer cannot be used against this server form factor (its
TreeLauncher + sglang's spawn-based scheduler deadlock; scheduler never
initializes CUDA).

## Diagnostic switches on the fork (all env-gated, default off)

- `SGLANG_BCG_DEBUG_SYNC=1` — `torch.cuda.synchronize()` + log around every
  segment replay and eager break (turns async faults into precise tracebacks)
- `SGLANG_BCG_DEBUG_KERNELS=1` — dump each segment's kernel names via
  cuda-python (`cuGraphGetNodes`/`cuKernelGetName`; forces `keep_graph=True`)
- `SGLANG_BCG_DEBUG_SPLIT=after_mlp,after_hcpost` — no-op eager-break points
  inserted in deepseek_v4.py for within-segment bisection
- `SGLANG_BCG_NO_WEAK_REF=1` — strong refs instead of weak-ref pool reclamation
- `SGLANG_BCG_REFRESH_INPLACE=all|f1,f2` — in-place copy of the
  normally-reference-swapped metadata tensor fields

## Ask

The DSV4 auto-disable rule (`_disable_breakable_cudagraph_if_incompatible`,
"DeepSeek-V4 (heavy capture-pool memory pressure)") keeps BCG off for DSV4 by
default; overriding it via the explicit backend flag exposes the breakage
above on our KT-hybrid stack. Could someone familiar with the BCG segment/pool
lifetime design point at what the recorded segments can legally reference
across the manual capture cycles — or confirm whether DSV4 (+ any
host-coupled MoE eager breaks) was ever validated under BCG on a full-GPU
stack? Happy to run any instrumentation you suggest on this exact repro.
