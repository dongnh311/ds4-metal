# Item P2 — fix memory-manager over-allocation into swap (item-(c) budget bug)

The A/B follow-on (FINDINGS-item-P2-mtp-ab-speedup.md) found that `DS4_QWEN4_MEMMGR=1`
**over-allocated the expert cache and forced ~18–20 GiB of swap**, halving throughput
and violating the zero-swap DoD. This fixes it.

## Root cause (ds4.c memory-manager block, item c)
The manager sized the expert cache from **non-routed** model bytes under
--ssd-streaming — `model_bytes = weights_streaming_non_routed_bytes` (6.32 GiB) —
then handed the whole `RAM − model − 20% headroom` remainder to the cache (43.32 GiB
at ctx 4096). But the process's real committed footprint tracks the **full GGUF span**
(~41.72 GiB — the engine's own memory plan charges exactly that and is proven
swap-free at the default cache). Sizing from 6.32 told the planner ~35 GiB more was
free than actually is, so on a busy 64 GiB machine (other sessions hold ~12 GiB
active+wired) the 43 GiB cache did not fit → ~18–20 GiB swap. Measured (ctx 4096 gen
128, DS4_QWEN4_MEMMGR=1, no MTP):
```
BEFORE:  expert budget 43.32 GiB -> swap_growth 20.45 GiB, swap_peak 23.06, gen_steady 2.24 tps  (WORSE than default 5.45)
```
(An early hypothesis — OS file-cache double-caching of the bundle — was tested with
`F_NOCACHE` on the pager fd and REJECTED: swap moved only 20.45→18.41 GiB. Reverted.
A `host_statistics64` available-RAM cap was also tried and rejected: `inactive` and
`external` page counts overlap → it reported 84 GiB > physical. Reverted. The reliable
signal is the engine's own full-span charge.)

## Fix (ds4.c, memory-manager block only, env-gated — default path byte-unchanged)
1. Charge the **full model span** (`e->startup_model_span_bytes`, ~41.72 GiB) — the
   same figure the engine's own plan uses and which is empirically swap-free — instead
   of the optimistic non-routed subset. This keeps the grant inside the engine's safe
   envelope.
2. Cap the grant at the **whole bundle size** (`pager->bundle_size`, 34.31 GiB) —
   caching beyond the data that exists is pure waste.

## Result — swap eliminated AND throughput improved (ctx 4096 gen 128, no MTP)
```
AFTER:   expert budget 7.93 GiB -> swap_growth 0.00 GiB, swap_peak 3.36 (no growth),
         gen_steady 6.96 tps, pager_hit_rate 0.68, rss_peak 11.4 GiB
```
Versus the default (no memmgr) baseline gen_steady 5.45 tps / hit 0.50: the manager now
grants the **largest swap-safe** cache (7.93 vs the engine's 1.86 GiB), lifting hit rate
0.50→0.68 and decode **5.45→6.96 tps (+28%)** with **zero swap growth**. So the memory
manager now *helps* where it previously hurt. Logit gate (run-logit-gate.sh 512 512)
still PASS bit-identical (the change is behind DS4_QWEN4_MEMMGR; cache size never
affects the math). diskwrites=0.

## Honesty / DoD implications
- This corrects item (c)'s over-optimistic receipt (its "expert cache 38→40 GiB at
  220K" figures were products of the same 6.32-GiB error; the swap-safe budget is far
  smaller). The FP8→expert-cache lever is still real but relative: at 220K, KV
  dominates (BF16 ~6.82 / FP8 ~4.18 GiB), so with the full-span charge the expert
  budget is only a few GiB and FP8 roughly doubles it — a genuine but modest lever, not
  a whole-bundle cache.
- **A whole-bundle (34.31 GiB) resident cache is NOT achievable on this 64 GiB machine
  without swap** once the full model span + KV are charged. That is a hardware/
  environment limit, not a code bug. The remaining DoD levers for the 220K read-GB wall
  are: FP8 *experts* (shrink the 34 GiB bundle itself — a future item), or resolving
  whether the routed-expert GGUF pages are truly resident (if they are not, the span
  charge could be relaxed to grant more cache — needs an RSS measurement, not a plan
  figure).

## Files / receipts
ds4.c (memory-manager block: full-span model charge + bundle-size cap; removed the
rejected non-routed override). Receipt: memmgr-fix/memmgr_fixed_off_ctx4096.csv
(after); before-fix receipt: mtp-ab/paged_memmgr_off.csv. Logs (scratchpad):
memfix2_off.log, gate_after_memmgrfix.log.
