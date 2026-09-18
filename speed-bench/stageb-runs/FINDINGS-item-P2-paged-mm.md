# Item P2 (spec §25 Phase 2 kernel) — paged "mm" prefill: bit-identical full prefill

**Goal (from item 1.3 NOTE + USER DECISION 2026-09-17):** make the SSD-paged path
use the tiled `mm` GEMM at prefill (T>8) so that a **full** prefill is
resident-vs-paged **bit-identical**. Until now the gate could only prove bit-identity
by forcing per-token on both sides (`--prefill-chunk 8`); at the default chunk the
resident path used the tiled GEMM (T>8) while paged was always per-token, so the two
differed by FP accumulation order (~1.88 max logit diff at ctx 512, argmax stable) —
inherent to any tiled GEMM, not a pager bug. This item removes that gap.

## Mechanism (why it is bit-identical, not just close)
The `mm` kernels index expert weights by **global** expert id:
`gate_base + e*expert_bytes` (qwen4.metal mid/down/nax). So if the paged path stages
each used expert into a buffer at the **same global-id offset** `e*expert_bytes`, the
kernel's addressing is byte-for-byte what it computes against the resident model map
(`m->map`). Same kernel + same weight bytes (gate #5 already proved the bundle bytes
== the GGUF expert bytes) + same routing lists ⇒ identical FP op order ⇒ identical
logits. Not an approximation.

## Implementation (one item, two files)
- **ds4.c** `qwen4_graph_moe_paged_mm(g,m,l,T,layer_idx)`: build_lists →
  `end_commands()` (flush so the CPU can read counts *after* router/topk/build_lists
  are queued — same ordering fix as item 1.3) → read `moe_counts` → collect the
  used-expert **union** (`counts[e]>0`) → lazy-alloc three global-id staging buffers
  (`DS4_N_EXPERT * bundle_size`, one set reused across all layers) → `pager_ensure()`
  the union per tensor and `tensor_write` each used expert into slot
  `used_ids[i]*bundle_size` (staged **once per layer**, not once per token) →
  `begin_commands()` → `mm_mid_with_bufs` (routed) + shared-expert dense gemv +
  swiglu → `mm_down_with_bufs` + shared-down gemv → `reduce`. A paged-mm branch at the
  top of `if (ok && g->pager)` in `qwen4_graph_moe` triggers for T>8 under the SAME mm
  eligibility test as the resident path (type has_mm, gate==up type, dims %64,
  shared-experts dense_ok); decode (T≤8) keeps the per-token path.
- **ds4_metal.m** `ds4_gpu_qwen4_moe_mm_mid_tensor_with_bufs`: rewritten to mirror the
  resident `ds4_gpu_qwen4_moe_mm_mid_tensor` exactly (full NAX/NAXC/NAXF path, correct
  args `in_dim/ff_dim/row_bytes/expert_bytes/tail_base`, mid output bound, tg=128, the
  6/7/8-bind dispatches, the M5-Pro NAX64 tensor path for IQ2_XXS gate/up + Q2_K down),
  differing from resident only in that `b[0]/b[1]` bind the gate/up **staging** buffers
  (bound size `expert_bytes*n_expert`). `_mm_down_tensor_with_bufs`: fixed the down bind
  size from `expert_bytes` → `expert_bytes*n_expert` (it must span all global-id slots).

## Gate — bit-identical (ground rule 5, run BEFORE any bench)
`run-logit-gate.sh 512 512` — resident (no --ssd-streaming, mm GEMM at T=512) vs paged
(--ssd-streaming, now paged-mm GEMM at T=512>8):
```
max_abs_diff=0.0 at index=-1
value_mismatches=0/248320
resident_dump_sha256=7d23b5e11a9af3384a17c6ce0bf29ec3dd52334620cc79e1289b84032430b263
paged_dump_sha256   =7d23b5e11a9af3384a17c6ce0bf29ec3dd52334620cc79e1289b84032430b263
GATE PASS: bit-identical   (gate_rc=0)
```
This is the **default-chunk** bit-identity the item 1.3 NOTE said would require paged
"mm" — now delivered. Regression: `run-logit-gate.sh 512 8` (per-token path) still
PASS bit-identical (unchanged).

## Scope / honesty
- diskwrites unaffected: staging is `tensor_write` (host→GPU RAM), the pager stays
  O_RDONLY; SSD read-GB unchanged (same expert misses as per-token — each used expert
  read once per layer either way; paged-mm only cuts the *staging memcpy* count, not
  the SSD reads).
- Staging RAM: three global-id buffers ≈ 762 MiB total (gate 512×422400 + up 512×422400
  + down 512×645120), one set reused across all layers, lazy-allocated on first
  paged-mm prefill.
- BF16/default per-token and resident paths byte-for-byte unchanged (paged-mm is a new
  branch gated on T>8 + pager + mm eligibility).
- This is the correctness/bit-identity milestone. Prefill throughput vs the per-token
  paged path is a separate bench (paged-mm stages each used expert once per layer
  instead of once per token — far fewer copies — but SSD reads are unchanged).

## Files / receipts
ds4.c (struct fields paged_mm_{gate,up,down}, free path, qwen4_graph_moe_paged_mm
helper, paged-mm branch), ds4_metal.m (mm_mid_with_bufs rewrite + mm_down bind fix).
Gate log: scratchpad/pagedmm_gate_512.log; regression: pertoken_gate_512_8.log.
