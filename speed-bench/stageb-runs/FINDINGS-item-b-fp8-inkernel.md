# Item (b) — in-kernel FP8 KV cache (the memory win) — spec §25 Phase 3

**Goal:** real E4M3 byte-packed K/V on the FLAT/default cache path (the path
deployment uses — the paged store is only the dev/gate path), for the ~2×
KV-memory reduction the 220K DoD needs. Env `DS4_QWEN4_KV_FP8` (user-approved).
BF16 default path is byte-for-byte untouched (all changes behind `if (args.fp8)`).

## Design (mirrors the (a)-certified `qwen4_fp8sim_vec`)
- attn_prep (qwen4.metal): per (tok,kv-head) simdgroup computes per-64-block
  absmax (simd_shuffle_xor over 8 lanes), power-of-2 scale = exp2(ceil(log2(
  max(amax,1e-4)/448))), E4M3-encodes post-RoPE K (`row[]`) and raw V (`vs[]`)
  to uint8 (`dsv4_e4m3fn_encode`), writes per-block fp16 scale.
- attn_mm (prefill T>8) + attn_decode (T≤8): decode uint8→half via
  `(half)(dsv4_e4m3fn_decode(byte)*scale)` into the same buffers the BF16 math
  used — downstream simdgroup-matrix math unchanged.
- Storage (ds4.c): full-attn layers allocate uint8 K/V (1 B/elem) + fp16 per-64
  scale INSTEAD of half K/V; the half cache is not allocated (the memory win).
  Snapshot save/restore + kv_cache_bytes are FP8-aware. FP8 disables the paged
  path (mutually exclusive). E4M3 helpers shared from dsv4_kv.metal (concatenated
  library); `dsv4_e4m3fn_exp_scale` table == the CPU table → value/dequant match.

## Validation
### BF16 regression (unchanged default) — PASS
`tests/test_qwen4_kernels` (synthetic, no model): Metal library compiles with the
new kernels; all attn mm/decode parity tests pass → the BF16 path is bit-unchanged.

### Correctness vs the (a)-certified fp8sim — PASS
`speed-bench/run-fp8-flat-gate.sh 512` (BF16 vs fp8sim vs in-kernel FP8, frontier
logits on the same prompt):
```
[FP8 vs BF16]    cosine=0.98483219  top1 fp=198  bf=198
[FP8SIM vs BF16] cosine=0.98493781
[FP8 vs FP8SIM]  cosine=0.99374443  top1 fp=198  sim=198
```
In-kernel FP8 diverges from BF16 by the SAME amount as the (a)-certified fp8sim
(0.98483 vs 0.98494) with matching greedy top-1 → quality-equivalent to the
certified reference. (FP8 and fp8sim are two independent E4M3 quantizers — float-
row+metal-scale vs f16-row+CPU-scale — so they are not bit-identical; ground rule
6 gate is quality-based, and both land at the certified quality.) Criterion:
top1(FP8)==top1(BF16) AND cos(FP8,BF16) >= cos(FP8SIM,BF16)-5e-3 → PASS.

### Direct in-kernel eval (definitive) — PASS
`DS4_QWEN4_KV_FP8=1 ds4-eval --suite core --questions 12 --retry-incomplete`:
**12/12 passed** (runtime 36m) — identical to BF16 12/12 and fp8sim 12/12. Every core
case produced the same correct answer on the real byte-packed in-kernel FP8 path.
Case AIME-02 went incomplete on its first 16000-token attempt and passed on the
32000-token retry (16914 tokens) — the same retry BF16/fp8sim needed for that case.
Receipt: speed-bench/stageb-runs/fp8-eval/fp8-inkernel-scores.txt.

### Memory win — PASS
kv_cache_bytes (ds4-bench CSV, ctx 512):
```
BF16 = 56,471,040 B (0.053 GiB)
FP8  = 34,588,512 B (0.032 GiB)   => 1.63x reduction
```
K/V tensors alone drop ~1.94× (512→264 B/head-vec); the aggregate is 1.63×
because the f32 indexer-K cache (`layer_ik_cache`) is NOT FP8'd (a later item).
Ratio is ctx-independent, so at 220K (BF16 KV ~6.99 GiB) FP8 saves ~2.6 GiB —
budget that the Phase-4 memory manager (item c) can redirect to the expert cache.

## Files
metal/dsv4_kv.metal (encode/decode), metal/qwen4.metal (3 kernels + args +
instantiation), ds4.c (storage/accessors/callsites/snapshot/kv_cache_bytes/flag),
ds4_metal.m (dispatch), ds4_gpu.h (prototypes), tests/test_qwen4_kernels.c,
speed-bench/run-fp8-flat-gate.sh.
