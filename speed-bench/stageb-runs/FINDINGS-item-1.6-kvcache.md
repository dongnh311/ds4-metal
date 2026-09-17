# Item 1.6 — root-cause: "kvcache_bytes=0 at ctx >= 30720"

## Conclusion: DISPROVEN as stated; it was a column conflation.

The reporting function is `ds4_session_kv_cache_bytes(ds4_session*)` at
ds4.c:62468. It returns 0 only when `!s`, `s->distributed`, or
`!s->qwen4_graph_ready`; otherwise it sums, over all layers,
`layer_k_cache[il] + layer_v_cache[il] + layer_ik_cache[il]` plus
`paged_kv_cache.total_bytes`.

Measured on the locked artefact at **ctx 32768 (>= 30720)**, post leak-fix,
the bench CSV reports:
  - `kv_cache_bytes    = 3,621,301,760  (3.373 GiB)`   <- NON-zero, correct
  - `snapshot_payload_bytes = 0`                        <- this is what was 0

So kv_cache_bytes is NOT zero at ctx >= 30720. The value that IS zero is the
separate CSV column `snapshot_payload_bytes`, produced by
`ds4_session_payload_bytes()` (ds4.c:62511), which returns 0 when the session
falls back to prefix-replay instead of a resident snapshot (the >1 GiB
snapshot fallback the progress doc described). The earlier
"kvcache_bytes=0 ... snapshot > 1GB -> prefix replay" note conflated these two
adjacent CSV columns: the prefix-replay fallback zeroes snapshot_payload_bytes,
not kv_cache_bytes.

The only way `ds4_session_kv_cache_bytes` returns 0 at high ctx is
`qwen4_graph_ready == false` — i.e. the graph never finished building. That is
exactly the pre-fix state: the pager eviction leak (commit 1171359) OOM-killed
the process during/after graph build, so any 0 observed at >=30720 before the
fix was "graph not ready due to OOM", not a KV accounting bug.

Evidence: speed-bench/stageb-runs/stageB_32k_mtpoff.csv (kv_cache_bytes column).
No code change required; the function is correct.
