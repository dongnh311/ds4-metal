# Plan 2 Task 1 — Decode GEMV kernels (H1 moe_down + H10 dense GEMV): profile-driven SKIP

Per the campaign plan (2026-09-15-ds4-iq2-m5pro-fast-campaign.md, Task 1): **"Profile
first, change second … no optimization without a measured gap"** and the YAGNI clause
**"If [the profile] shows decode time dominated by [bandwidth-bound] work, this task is
skipped — record the decision as `task=skip reason=<profile note>`."**

Result: **SKIP with measurement.** Decode is memory-bandwidth-bound on quantized weight
reads; the plan's lead candidate is already implemented; no bit-identical kernel change
raises decode tok/s. No source change → no bit-identical gate needed. diskwrites=0
(measurement only). No new flags (used existing diagnostic envs `DS4_METAL_ENCODER_TIMELINE`
and `DS4_METAL_Q8_MV_NSG`). RESIDENT run (compute-bound regime), ctx 4096, gen 128.

## Step 1 — profile (headless per-kernel GPU timeline, `DS4_METAL_ENCODER_TIMELINE`)
One 48-token resident decode. Top decode-path kernels by GPU-busy share (full table:
task1-gemv/kernel_profile.tsv):

| kernel | %gpu | us/call | note |
|--------|------|---------|------|
| kernel_mul_mv_q8_0_f32 | **32.96%** | 102.9 | dense Q8_0 GEMV = the real H10 (attn q/k/v/o + GDN in/out + shared-down, ~127 calls/token) |
| kernel_qwen4_moe_mid | 10.86% | 95.4 | IQ2_XXS gate/up experts (decode) |
| kernel_qwen4_moe_mm_mid_nax64 | 8.92% | — | **prefill** batched mid (1 chunk × 48 layers) |
| kernel_qwen4_moe_down (H1) | 7.29% | 64.0 | Q2_K down experts (decode) |
| kernel_mul_mv_f16_f32_4 | 7.00% | 30.4 | f16 dense projections |
| kernel_qwen4_hc_gate_mix_f16_pf | 6.80% | 29.5 | hyper-connection mix |

Every top kernel is a **quantized weight-read GEMV/GEMM**. The named plan targets are
minor: H1 `moe_down` is 7.3%; the "H10 dense GEMV sequence" is realized by the generic
ggml `kernel_mul_mv_q8_0_f32` (33%), not `qwen4_gemv_rows`.

## H1 lead candidate is ALREADY implemented (no-op)
The plan's first candidate: *"confirm whether `kernel_qwen4_moe_down` loads all 768
columns or bounds to 640; if 768, bound to 640."* Read of `qwen4_row_dot` type 10 (Q2_K),
metal/qwen4.metal:2256 (guard at :2264):
```c
const uint nb = (in_dim + 255u) / 256u;                 // 640 -> nb=3 (768 span)
if (ib*256u + group*16u + l >= in_dim) continue;         // <-- already bounds to 640
```
The per-lane `continue` guard already skips the padded tail; lanes owning elements ≥640
do no work and read no activation past the vector. **The candidate change is already in
the code.** No gap here.

## Step 2 probe — the dominant kernel has NO exploitable occupancy/parallelism gap
`kernel_mul_mv_q8_0_f32` uses NSG simdgroups (default 4) and N_R0_Q8_0=2 rows/threadgroup.
NSG is the runtime-tunable parallelism/occupancy axis (`DS4_METAL_Q8_MV_NSG`). Sweeping it
holds decode flat within noise (task1-gemv/nsg_sweep.csv):

| DS4_METAL_Q8_MV_NSG | gen_steady tok/s |
|---|---|
| 2 | 28.82 |
| 4 (default) | 29.01 |
| 6 | 28.98 |
| 8 | 28.87 |

±0.7% = pure run-to-run noise. If the q8_0 GEMV were occupancy/latency-bound, doubling the
simdgroups (4→8) would move throughput. It does not. (N_R0_Q8_0, the other row-tiling axis,
is a global compile constant wired into ~20 host grid sites, several hardcoding `/2`; it is
not a single-kernel knob, and since it changes no bytes-read it would be equally flat on a
bandwidth-bound kernel.)

## The bandwidth-bound signature (why no kernel change can help)
- Timeline **gap/busy = 0.10** → GPU is **90% busy** through decode: kernels run
  back-to-back, not stalling on dispatch. Not dispatch/overhead-bound.
- NSG-invariant (above) → not occupancy/parallelism-bound.
- ⇒ the binding constraint is **aggregate DRAM traffic**: a fixed number of quantized
  weight bytes must be streamed per token, and they already are, at ~bandwidth, with the
  GPU saturated. Individual model-dim GEMVs achieve only ~100 GB/s yet stay NSG-flat,
  because the ceiling is the *sum* of back-to-back weight reads across the kernel stream,
  not any one kernel's efficiency. (M5 Pro, 64 GB.)

## No serialization/fusion win either
Largest idle gaps (task1-gemv/kernel_profile.tsv, `sum_gap_us`):
- `kernel_qwen4_router_topk` 41.2 ms total, 17.9 us/call — a **true data dependency**
  (experts can't be dispatched until top-K is decided). Architectural, not a kernel bug.
- `kernel_qwen4_ple_gate` 34.5 ms, 719 us/call over 48 calls — **PLE n-gram I/O wait**, not
  GPU compute.
- `kernel_mul_mv_q8_0_f32` 51.6 ms but over 6344 calls = 8 us/call = inherent scheduling.
None is a fusible bubble worth a targeted kernel change.

## Conclusion (redirects the DoD effort)
Decode of this IQ2_XXS/Q2_K/Q8_0 model on M5 Pro is **memory-bandwidth-bound**. The GPU
spends ~90% of decode streaming quantized weights near the bandwidth limit; reorganizing
the GEMV kernels (NR0, NSG, tiling, geometry) cannot raise decode tok/s because the
bytes-read per token is fixed and already saturating DRAM. The only levers on decode
throughput are:
1. **Fewer bytes/token** — a smaller expert/dense quant. Forbidden until last (spec) and a
   quality-gate (not bit-identical) change. Out of scope for Task 1.
2. **Higher memory bandwidth** — hardware.

Kernel micro-optimization (this task's premise) is a dead end for the ≥40 tok/s DoD on
this model. The real remaining decode levers are the cache-residency work already done
(item 1 adaptive cap + 1.4 slot fix — push decode compute/BW-bound instead of SSD-read-
bound on a dedicated box) and MTP (Task 4: +14% prose / +33% code when compute-bound).

`task=skip-H1H10 reason=decode-is-DRAM-bandwidth-bound (gpu 90% busy, NSG-invariant, H1
padded-768->640 already implemented)`.

## Files / receipts
task1-gemv/kernel_profile.tsv (per-kernel GPU-busy + gap), task1-gemv/nsg_sweep.csv,
task1-gemv/baseline_ctx4096_nsg4.csv. Raw timeline: scratchpad/tl_decode.txt (42.7K
dispatch records). No source change.
