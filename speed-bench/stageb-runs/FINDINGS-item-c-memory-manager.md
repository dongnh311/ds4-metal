# Item (c) — wire the unified memory manager into ds4.c (Plan 4 Task 3 Step 3)

**Goal:** turn the standalone `ds4_memory_manager` stub (committed 76a15e1, never
called) into a live 4-way budget planner (model + KV + expert cache + workspace),
so the KV savings from item (b) FP8 are redirected to the expert cache — the lever
for the 220K SSD read-GB / hit-rate wall (night report Blocker #2).

## What was wired (ds4.c, session create, qwen4 Metal + --ssd-streaming)
- Opt-in via **DS4_QWEN4_MEMMGR** (diagnostic-env family, ground rule 7). Default
  path is byte-for-byte unchanged — the whole block is behind the env guard.
- At the pager-cache-enable point: init the manager with total RAM
  (`glm_graph_host_memory_bytes`, sysctl hw.memsize), set the **resident** model
  bytes (non-routed weights via `weights_streaming_non_routed_bytes` — routed
  experts are paged and budgeted separately as the expert cache, NOT counted as
  resident model), compute the plan for the run's ctx with an FP8-aware KV
  estimate (`qwen4_kv_bytes_estimate`), report it, and set the pager cache budget
  to the plan's expert budget when that exceeds the engine's figure.
- Fixed `ds4_memory_manager.c` `compute_plan`: it split available RAM 60/40
  KV/expert (giving KV 60% even when it needed far less, starving the expert
  cache). Now **KV gets exactly what it needs; the expert cache gets the
  remainder** — which is what makes FP8's smaller KV free expert-cache budget.
  (No committed unit test referenced the old split.)

## Result — the FP8→expert-cache lever, measured at ctx 220000
Memory-manager report (64 GiB machine, --ssd-streaming, resident model 6.32 GiB):
```
BF16 KV:  KV budget 6.82 GiB  ->  Expert cache budget 38.06 GiB
FP8  KV:  KV budget 4.18 GiB  ->  Expert cache budget 40.70 GiB
```
FP8 KV (item b) frees **2.64 GiB** (6.82→4.18) that goes straight to the expert
cache (38.06→40.70 GiB). Before this wiring the pager cache budget was whatever
`ds4_engine_dynamic_expert_cache_bytes` returned (0 with no --qwen4-expert-cache
flags → uncached, an SSD read on every expert access); the manager now grants the
expert cache the real remainder of RAM. Receipt: memmgr/budget-220k.txt.

## Scope / honesty
- This delivers the WIRING + the correct KV-aware 4-way budget + the demonstrated
  FP8→expert-cache lever. It does NOT by itself hit the 40 tok/s @220K DoD — that
  remains multi-phase (paged-mm prefill P2, dispatch tuning, etc., per the night
  report). What (c) removes is the expert-cache starvation: at 220K the cache now
  gets ~40 GiB instead of starving, and FP8 widens that by the KV savings.
- diskwrites unaffected: the manager only plans; the pager stays O_RDONLY.
- Default runs (no DS4_QWEN4_MEMMGR) are unchanged.

## Files
ds4.c (include, qwen4_kv_bytes_estimate, memory-manager block at session create),
ds4_memory_manager.c (compute_plan allocation policy). Receipt: memmgr/budget-220k.txt.
