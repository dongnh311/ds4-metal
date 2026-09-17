# Item 1.1 — ctx 220000 paged (MTP-off): DNF-with-diagnosis (not a clean number)

Attempted `--ssd-streaming --ssd-streaming-cache-experts 10GB` (forced cache,
allowed flag) because the AUTO plan starves the expert cache to 2 slots at 220K
(see FINDINGS-item-1.4). Result after 72 min of wall-clock:
  - prefill did NOT complete (no CSV data row); process was I/O-bound at ~30% CPU
    (CPU time advanced ~1.3s per 6s wall), synchronous per-token expert preads at
    ~64 MB/s + swap-in waits.
  - physical footprint stable 24.8 GiB, but writable regions ~49% swapped_out
    (~19 GiB); swap grew to 5.76 GiB with only 383 MiB swap free (near exhaustion)
    -> cut with SIGTERM (rc=143) to protect the machine.

Conclusion: the paged 220K path is IMPRACTICAL on the current Stage-B
implementation. Two independent walls, both measured:
  1. Per-token MoE (qwen4_graph_moe paged branch loops per token; only non-MoE
     batches) -> prefill is serialized on 0.5 MiB random preads at ~64 MB/s.
     At 32K this was 34.6 tps prefill / 4.5 tps decode; 220K is ~7x tokens with
     larger attention, and would read far more than the ~166 GiB seen at 32K.
  2. Memory 3-way squeeze at 220K: KV 6.99 + buffers 10.27 + a useful expert
     cache all compete with the 95 GiB NNgram file cache on 64 GiB -> a forced
     anon cache swaps (near swap-exhaustion here); the auto plan avoids swap only
     by starving the cache (re-read storm).

DoD "≥40 tok/s @220K, zero swap" is therefore NOT achievable on Stage-B alone;
it needs the later ordered phases: paged "mm" prefill (P2 kernel) to batch expert
reuse, FP8 KV (P3) to halve the 6.99 GiB KV, and the memory manager (P4) 4-way
budget. This is the honest Phase-1 baseline, reported as a blocker per the
mission's report format (not self-resolved beyond ground rules).

Clean high-ctx data point delivered: ctx 32768 MTP-off completes rc=0
(stageB_32k_mtpoff.csv). exit@55K is gone (item 1.5): the 220K run ran 72 min
with bounded footprint (24.8 GiB, no leak-climb) far past the old ~55K crash.
