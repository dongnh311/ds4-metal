# V4.1 decode glue fusions (upstream #1042) on the streaming pipeline, 2026-09-25

Branch `feature/ds41-decode-fusions` (from `feature/ds4.1-flash` after sub-project 1).
DoD 2 sub-project 2, first step. Measurement point as before: ctx 8192, `switch`
prompt, 512 teacher-forced tokens, V4.1 alone (oMLX and watchdogs down).

## Result

| A/B (interleaved A, B, B, A) | A t/s | B t/s | Ratio |
| --- | ---: | ---: | ---: |
| #1042 off (`DS4_METAL_DISABLE_V41_HC_FUSE=1`) vs on, auto cache (`ab-fusions.json`) | 11.07 | 11.28 | 1.0190 |
| + fused router (`DS4_METAL_ENABLE_V41_ROUTER_FUSE=1`, opt-in) (`ab-router-fused.json`) | 11.18 | 11.34 | 1.0143 |
| Phase-0 configuration vs shipped defaults (`ab-final-fused.json`) | 8.98 | 11.35 | 1.2641 |

Phase-0 configuration: 24 GB cache target, no layer queue under streaming, no
async load, no #1042. Shipped defaults: auto cache (35.62 GiB budget at ctx
8192), layer queue, async load, #1042 without the fused router.

## What was merged and how it was resolved

`8b75381` merges antirez/ds4 #1042 (Adrian Galilea, open PR, head `6e92e92`).
It fuses the single-box decode glue: HC 19 -> 6 dispatches, router 11 -> 1 on
M5, the shared expert, and the attention tail. The PR predates both Ivan's
fusions and this branch's streaming pipeline, so the merge needed these
resolutions:

- **ds41_matmul:** Ivan's Q8_0 decode+BF16 kernel runs first; #1042's
  `matvec_bf16` covers F16 after it.
- **Attention:** the resident QKV overlap keeps its unfused KV tail. Otherwise
  #1042's fused projection does the KV tail (`kv_done`), followed by the fused
  heads rounding + RoPE and the output rounding.
- **MoE:**
  - The fused router and the shared gate/up/SwiGLU run before the async-load
    start.
  - Their error exits abandon the async job.
  - The resident parallel FFN skips when the fused shared expert ran.
- **Kernel name clash** with Ivan's `dense.metal`: #1042's shared kernel is now
  `kernel_dsv41_shared_gate_up_swiglu_mid_q8_0`.
- **Softplus:** #1042's router used `log(1 + exp(v))`. The upstream sync had
  added a series below `exp(v) = 1/32`, and the PR's own kernel test caught
  the mismatch. The fused router now uses the same expression term for term.
- **Per-call switches (`dad0a56`):** the fusion switches were cached on first
  use, so `--stream-control` could not flip them between its two sessions. They
  are now read per call (model-free check `--v41-fuse-switches`).
- **Fused router is opt-in (`2c40b6f`).** On an exact score tie at the top-k
  boundary, the fused select keeps the lower expert index, while the standalone
  argsort kept the other expert.
  - The 64 GB streaming check caught it at pos 2053, layer 12: experts 281 vs
    282, with equal logits and probs. That is one tie in about 5200 layer
    steps.
  - The selected set, the weights and the logits then differ.
  - The router fusion now needs `DS4_METAL_ENABLE_V41_ROUTER_FUSE`.

## Exactness

**Kernel tests on M5** (`tests/test_deepseek41_metal`), all byte-identical:
- `--hc-fuse`: 10 -> 3 dispatches, 0.422 -> 0.302 ms;
- `--moe-fuse`: 15 -> 4-5 dispatches, 1.002 -> 0.828 ms;
- `--attn-fuse`.

**Streaming exactness** (`--stream-control`, 65 steps after prefixes 511 and
2047), all PASS:
- `DS4_METAL_DISABLE_V41_HC_FUSE` (everything) at 8 and 24 GB;
- `_MOE_FUSE`, `_ATTN_FUSE` and `_MATVEC_BF16` at 8 GB;
- router log: 1280 lines, logits identical.

**How the router tie was found.**
- Bisect of the first failing run:
  - a no-op switch passes, so the path is deterministic;
  - `ATTN_FUSE`, `SHARED_FUSE` and `MATVEC_BF16` pass;
  - `MOE_FUSE` and `ROUTER_FUSE` fail at pos 2054.
- A throwaway fused-vs-standalone router dump then showed the tie.

**Qwen:** fast tier PASS (`dsv4_hc.metal` is shared). Full tier PASS on
`2c40b6f`:
- needle at 214,672 tokens hit;
- wired steady 45.91 GiB;
- paired A/B against PROD, medians 44.08 / 39.59 (PROD) vs 42.90 / 41.46
  (branch); the gate passed.

## GPU budget per stage (ms/token, ctx 8192, auto cache, 40 tokens)

`DS4_METAL_GPU_STAGE_TIMESTAMPS_DETAIL` read with `speed-bench/v41/stages.py`
(overlap removed; `stages-before.txt`, `stages-after.txt`). "Achieved" is the
stage's weight bytes over its time, against 290 GB/s.

| Stage | Before | After #1042 | Weights read | Achieved after |
| --- | ---: | ---: | --- | ---: |
| pre_attn (HC mix/collapse/norm, Engram projection) | 2.34 | 2.30 | Engram 0.6 GiB (2 layers), HC F16 | ~95 % |
| attn_project (q_a, q_b, kv) | 8.54 | 7.69 | 2.07 GiB Q8_0 | ~97 % |
| attn_kv (RoPE, FP8, window) | 0.40 | 0.15 | — | — |
| attn_select (indexer) | 1.61 | 1.56 | 80 MiB F16 + keys | latency |
| attn_core (attention, heads BF16 + RoPE) | 2.85 | 2.61 | KV rows | latency |
| attn_out (output_a, output_b) | 12.74 | 12.62 | 2.99 GiB Q8_0 | ~88 % |
| post_attn (HC) | 1.15 | 0.69 | HC F16 | — |
| moe_route (F32 router) | 2.45 | 2.42 | 300 MiB F32 | ~45 % |
| moe_shared | 7.36 | 6.56 | 1.40 GiB Q8_0 | ~79 % |
| moe_routed (IQ2_XXS gate/up, Q2_K down) | 13.18 | 13.67 | 2.22 GiB | ~60 % |
| moe_tail + after_moe | 0.42 | 0.00 | folded into the expand | — |
| logits | 2.40 | 2.35 | 671 MiB Q8_0 | ~100 % |
| **GPU busy** | **55.44** | **52.65** | byte floor 38.4 | |

The step time also pays the miss path: pread 13.4 + readahead 15.4 ms/token at
a 0.87 hit rate.

## Where the remaining GPU time is (about 14 ms over the byte floor)

1. **Routed MoE, about 5.5 ms.** A code survey found that the V4.1 kernels
   already have the Qwen M5 fixes (per-type kernels, 4 rows per simdgroup, IQ2
   tables in threadgroup memory). What is left is smaller:
   - shape function constants for 5120/2304;
   - float4 activation loads;
   - a parallel SwiGLU epilogue;
   - nsg 4;
   - a Q2_K down lane map that leaves about 25 % of the lanes idle on 2304-wide
     rows. Remapping it changes the summation order, so it is not bit-exact
     and needs the user's approval.
   - The kernels are shared with V4-Flash and GLM streaming, so any change
     belongs in new V4.1/M5 pipelines.
2. **Attention select + core, about 3.5 ms** (latency-bound).
3. **attn_out, about 1.6 ms**, and **shared expert, about 1.4 ms** (bandwidth
   efficiency).
4. **Router, about 1.3 ms.** The fused router recovers it (+1.4 %) if a
   canonical top-k tie order is accepted for both paths.

## Long context at the auto cache

ctx 32768, shipped defaults, auto cache (2644 cached experts): 8.84 t/s, hit 0.787,
wired steady 37.2 / peak 48.2 GiB, no swap, not contaminated (Phase 0 at 32K /
24 GB: 7.85 t/s). This closes the review's open item that no run covered 32K at
the auto cache with every feature on.
