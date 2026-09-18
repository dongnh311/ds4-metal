# Item 2.1 (spec §25 Phase 2) — paged KV was INERT; fixed + honestly re-gated

## Integrity finding (why the prior gate was a false positive)
Commit 50a5c0e ("Phase 2.1+2.2 ... gate bit-identical") wired Design A
"materialize" but the paged path NEVER RAN in the live session flow, so the
committed gate #KV silently compared the resident path against ITSELF
(trivially bit-identical). Two compounding bugs:

1. **init-after-alloc ordering.** ds4_kv_cache_init ran at ds4.c:73134, AFTER
   qwen4_graph_alloc (73118). graph_alloc's `memset(g,0,sizeof(*g))` (57871)
   zeroes paged_kv_cache, and the paged_*_staging_full buffers are allocated
   INSIDE graph_alloc gated on `paged_kv_cache.n_layers>0` (57972) — which was 0
   during alloc. So staging_full stayed NULL and qwen4_paged_kv_active()
   (needs staging_full[0]!=NULL) was ALWAYS false → the roundtrip early-returned
   and kcache/vcache fell back to the resident flat cache.

2. **ring never sized to ctx.** ds4_kv_cache_init IGNORED its max_tokens arg and
   hard-coded `initial_pages = total_pairs*4` (4 pages/ring). get_page indexes
   `(token_pos/page_size) % pages_per_ring`, so beyond 4*page_size (=2048 @512)
   the ring WRAPS and aliases earlier tokens — silent corruption for any real
   context. (The committed gate only used ctx≤512, one page, so this never
   surfaced even had the path been active.)

## Fix
- ds4.c: moved the DS4_QWEN4_KV_PAGED init block to BEFORE qwen4_graph_alloc;
  graph_alloc now preserves paged_kv_cache across its memset (like g->pager).
  Free the cache on the graph_alloc error path. One-time "paged KV path ACTIVE"
  log in the roundtrip so a gate can PROVE the path ran.
- ds4_kv_cache.c: size pages_per_ring = ceil(max_tokens/page_size)+1 (cap at
  DS4_KV_CACHE_MAX_PAGES with a warning); pre-allocate the first page of EACH
  ring (was the first N flat slots).
- speed-bench/run-kv-gate.sh: HARD-ASSERT "paged KV path ACTIVE" in the paged
  run's stderr — a missing log now FAILS the gate (no more false positives).

## Honest re-gate (paged path CONFIRMED active)
- ctx 512:  paged path CONFIRMED ACTIVE ("paged KV path ACTIVE, page=512");
  max_abs_diff=0.0, value_mismatches=0/248320 → **GATE PASS bit-identical**.
  (Pre-fix the same run FAILED the new activation assert — proof the path was inert.)
- ctx 8192: ring sized to **18 pages/ring (covers ~9216 tokens/head)** — the
  ring-size fix engaged (old hard-coded 4-page ring = 2048 max would have
  wrapped/aliased tokens 2048..8191). paged path ACTIVE; max_abs_diff=0.0,
  value_mismatches=0/248320 → **GATE PASS bit-identical**. Passing here proves
  the ring fix is both necessary and correct.
- ctx 8192 chunk 2048 (4 chunks, multi-chunk roundtrip): paged path ACTIVE;
  0/248320 → **GATE PASS bit-identical** (cross-chunk incremental commit +
  growing-range materialize verified).

## Caveat (memory reality — NOT a win yet)
Design A keeps the flat cache AND staging_full AND the paged store → ~3x BF16 KV
memory. It is a CORRECTNESS milestone (paged store proven bit-identical), env-
gated (DS4_QWEN4_KV_PAGED, off by default), NOT the default path. The memory win
requires Design B (kernels read the paged pool directly, drop flat+staging_full)
and then Phase 3 FP8 (halve the pool). Those are the next steps.
