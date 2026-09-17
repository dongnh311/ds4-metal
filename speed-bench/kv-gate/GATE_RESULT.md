# Gate #KV (spec §25 item 2.2): DS4_QWEN4_KV_PAGED=1 vs resident — PASS bit-identical

## Design (item 2.1): paged KV via "materialize" (Design A)
When DS4_QWEN4_KV_PAGED is set, paged_kv_cache is allocated with 2*Hkv heads
([0,Hkv)=K, [Hkv,2Hkv)=V). Per layer, after attn_prep writes the flat resident
K/V for tokens [pos0,pos0+T), qwen4_paged_kv_roundtrip (ds4.c) commits those rows
INTO the paged store, then materializes the whole [0,pos0+T) range back OUT into
the full staging buffers (paged_{k,v}_staging_full), interleaved to the kernels'
token-major [tok][Hkv][D] layout; the two attn_decode reads then consume the
staging buffers (qwen4_kcache/qwen4_vcache selectors). Pure memcpy round-trip
(no arithmetic) → bit-identical by construction iff the paged addressing is
correct. attn_prep's GPU write is committed (end_commands) before the CPU reads
the flat cache — same ordering the SSD pager fix required.

This is the PLUMBING milestone: it validates the paged K/V store end-to-end but
does NOT reduce memory (the flat cache still exists; paged kv_cache_bytes is ~2x
resident: 104 MiB vs 55 MiB at ctx 512). The memory win is Design B (in-kernel
page table), a follow-up — the paged per-head layout must first be reconciled
with the kernels' token-major access.

## Results
Gate: speed-bench/run-kv-gate.sh [CTX] [CHUNK]; resident (env unset) vs paged
(DS4_QWEN4_KV_PAGED=1), both resident-experts, frontier-logit dump + byte diff.
- ctx=8            : max_abs_diff=0.0, 0/248320 mismatches, resident sha==paged sha.
- ctx=512          : max_abs_diff=0.0, 0/248320, identical dump sha.
- ctx=512 chunk=128 (4 chunks): max_abs_diff=0.0, 0/248320 — multi-chunk mirror +
  growing-range materialize + sparse/QSA read path all bit-identical.
- decode (gen 16, greedy): identical generated text resident vs paged
  ("influenza ne dee procedere, che benigna, e propitia,"); only perf/memory CSV
  columns differ (paged uses more memory, expected).

gguf sha256 = ed238d8d… (asserted by size). diskwrites: no --ssd-streaming here
(resident experts); no writes to the model volume.

## Not done (follow-ups, per plan order)
- 2.3 bench page sizes 256/512/1024 × ctx — pending (Design A has no memory win,
  so page size only affects the paged-store overhead; the meaningful bench needs
  Design B).
- GDN recurrent state + QSA indexer caches (ik_cache/block_key) are NOT paged
  (separate storage, out of the paged_kv scaffold).
