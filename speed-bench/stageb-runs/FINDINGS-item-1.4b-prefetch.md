# Item 1.4b — prefetch tuning: the predictive prefetch is dead + ineffective; reclaim its RAM

Investigated the pager's Stage C/D "predictive next-layer prefetch" as a lever to raise
hit-rate / hide SSD-read stalls. Conclusion: it is not wired, cannot help a balanced
MoE as designed, and only wasted RAM. Removed its activation; reclaimed ~136 MiB for the
expert cache (the proven lever, item 1.4).

## What was there
- `ds4_expert_pager_async_start()` + `alloc_double_bufs()` are called at pager enable
  (ds4.c), starting an idle worker thread and malloc'ing **48 layers × 3 tensors × 2 ×
  bundle = ~136 MiB** of L2 double-buffers.
- But NOTHING ever calls `async_prefetch` / `prefetch_predicted` / `predict_next_layer`
  / `swap_buffer` / `get_l1_buf` from any binary (grep across ds4.c, ds4_bench/cli/
  agent/server: zero hits). The worker gets no jobs; the 136 MiB is zeroed and never
  read. `async_stop()` is never called either (worker leaked at exit).

## Why it cannot help as designed
- The predictor (`predict_next_layer`) returns the **top-10 globally-hottest experts**
  by cumulative count. A load-balanced MoE spreads routing across all 512 experts by
  design, so those 10 have near-zero precision against the next layer's actual per-token
  routed set — prefetching them would mostly fetch experts the tokens don't use.
- MoE expert identity is known only **immediately before** the GEMM needs it: the router
  runs just ahead of the experts, so there is no compute window within a layer to hide
  the SSD read behind, and the next layer's routing depends on this layer's output
  (sequential) — so speculative cross-layer prefetch needs a real routing predictor,
  which the global-frequency stub is not.
- The cache + LRU/scored eviction already exploit the temporal locality that a simple
  prefetch could (recent experts stay resident), so a weak predictor adds ~nothing.

## Action + result
Removed the `async_start()` / `alloc_double_bufs()` call block (ds4.c); left the pager's
async API in place for a future routing-aware predictor. Verified (ctx 4096 gen 128,
ssd-streaming):
```
                      hit_rate   gen_steady   rss_peak      swap_growth
with dead prefetch:   53.54%     6.05 tps     6.92 GiB      0
prefetch removed:     53.54%     6.18 tps     6.79 GiB (-141 MiB)   0
```
Zero regression (identical hits 128794/111776), ~136–141 MiB reclaimed. That RAM now
counts as free/grabbable, so on the shared box it lowers footprint/swap pressure and on
a dedicated box it feeds item (1)'s adaptive cache cap. Bit-identical:
run-logit-gate.sh 512 512 (removing an unused allocation cannot change the math).
diskwrites=0.

## Recommendation
Do NOT invest in MoE expert prefetch for this model: the routing dependency + balanced
load make it low-ROI, and the cache-size lever (item 1 adaptive cap + item 1.4 slot
fix) is the proven path. A future routing-aware predictor (e.g. per-layer per-token
locality across consecutive decode steps) could revisit it, but it is a research effort
with uncertain payoff.

## Files / receipts
ds4.c (removed the Stage C async_start/double-buffer block). Receipts:
prefetch-1.4b/prefetch_removed_ctx4096.csv, before_removal_ref_ctx4096.csv,
gate_after_prefetch.log.
