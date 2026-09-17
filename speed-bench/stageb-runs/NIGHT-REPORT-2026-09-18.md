# Night report — 2026-09-17/18 (autonomous run)

## Headline
The ctx≥8K OOM blocker was a **pager eviction memory leak** (not model residency).
Found via vmmap, fixed, gate-verified bit-identical. This unblocked the whole
paged path: ctx 32K now completes cleanly (rc=0, 9.5 GiB peak, **zero swap
growth**). Phase 0 and Phase 1 (P3 Stage B/E real numbers) are complete and
committed. The 220K DoD and MTP-on remain — see Blockers.

## git log --oneline -9
```
780054d Phase 1.6: root-cause kvcache_bytes=0 at ctx>=30720 = column conflation (disproven)
fd5690c Phase 1.5: root-cause exit@55K = the pager eviction leak (fixed in 1171359)
788b3df Phase 1.4: eviction E1/E2 A/B (within noise at 32K) + cache-plan root finding
4d4f1d4 Phase 1.2: stageB-real.csv provenance receipt
19e1fc4 Phase 1.1: SSD paged real numbers — ctx 32K MTP-off clean; 220K DNF-with-diagnosis
04abb4d chore: gitignore stageB run logs + corrupt backups
bff01ea speed-bench: fast gate integrity check + preserve GATE_RESULT.md
1171359 Fix pager eviction memory leak (cache_data never freed) — ctx>=8K OOM root cause
a1198d6 Phase 1.7: verify bundle size vs cache-plan per-expert size (no size bug)
```

## Results + numbers
- **Leak fix (1171359):** `pager_evict_slot` never freed `cache_data[slot]` → every
  eviction leaked ~0.4–0.6 MiB. vmmap@8192: 80+ GiB dirty heap, 188K leaked
  ~412 KB allocs, footprint +1.3 GiB/5s → jetsam. Fix = free+NULL on evict.
  After: footprint plateaus ~5 GiB; MALLOC_SMALL flat ~2.4 GiB.
- **Gate #5 (ground rule 5): PASS bit-identical** post-fix — max_abs_diff=0.0,
  0/248320 mismatches, resident sha==paged sha; frontier .logits.json regenerated
  byte-identical to committed → fix is math-neutral. Receipt: speed-bench/logit-gate/.
- **1.1 ctx 32768 MTP-off:** rc=0, prefill 34.58 tps, decode 4.52 tps, TTFT 1047 ms,
  hit 99.29%, read 154.5 GiB, RSS 9.50 GiB, **swap Δ 0**, diskwrites_delta=0.
- **1.1 ctx 220000 MTP-off:** DNF-with-diagnosis (see Blockers).
- **1.4 eviction A/B @32K:** LRU vs scored within noise (hit 99.430% both, misses
  differ 0.013%, identical output checksum, swap Δ 0). Policy is not the lever.
- **1.5 exit@55K = the leak** (no hard cap near 55K; jetsam OOM scaling with evictions).
- **1.6 kvcache_bytes=0 = column conflation** (kv_cache_bytes is 3.373 GiB at 32768;
  the 0 was snapshot_payload_bytes, a different column).
- Receipt CSV: `speed-bench/stageb-runs/stageB-real.csv` (all provenance columns).

## Gates
- Gate #5 (logit resident vs paged): **PASS** — speed-bench/logit-gate/GATE_RESULT.md
- PHASE 0 KV_PAGED smoke: **PASS** — 2 init lines print (49 layers, 2 heads, 256 dim).
- diskwrites_delta=0 on every run (per-file stat unchanged + pager O_RDONLY).

## Blockers / decisions needed (NOT self-decided — ground rules)
1. **MTP-on runs (item 1.1):** ds4-bench has no MTP enable path; `--mtp` exists in
   ds4_cli/agent/server but not the bench. Wiring it adds a bench flag → ground
   rule 7 ("new flag = stop + ask"). It IS an existing flag (parser parity) and
   ground rule 7 whitelists MTP_STATS (implying MTP-in-bench was intended), so it
   is low-risk — but it is your call. Ready-to-apply patch + run commands are in
   the scratchpad (apply_mtp_flag.sh, mtp_on_runs.txt). **Approve and I run the
   MTP-on matrix in ~40 min.**
2. **220K DoD (≥40 tok/s, zero swap):** NOT achievable on Stage-B alone. Measured
   two independent walls: (a) per-token paged prefill is I/O-bound (~64 MB/s
   synchronous 0.5 MiB preads; ~30% CPU); (b) a 3-way memory squeeze at 220K
   (KV 6.99 + buffers 10.27 + a useful expert cache vs the 95 GiB NNgram file
   cache on 64 GiB) — the auto plan starves the cache (re-read storm), a forced
   cache swaps toward exhaustion. The fixes are the later ordered phases: paged-mm
   prefill (P2), FP8 KV (P3, halves KV), memory manager (P4, 4-way budget). This
   is the honest Phase-1 baseline, not a fixable-tonight bug.

## Read-GB vs ground rule 4
Every 32K run reads ~124–166 GiB vs the gr-4 budget ~0.27 GiB (~619×). This is
the cache-sizing signal (auto cache 1.36 GiB ≪ prefill working set), documented
in FINDINGS-item-1.4-cacheplan.md — expected per item 1.7, not a leak. The lever
is cache budget (Phase 4 memory manager), not the eviction policy.

## Next (Phase 2, per order)
P4 paged-KV wiring (2.1 wire layer_k/v into the page table, 2.2 gate bit-identical,
2.3 bench page sizes). Requirements being mapped; see the follow-up notes.

## DoD status
- **ds4-eval 18/20:** UNCHANGED by construction. The only code change this session
  is the 11-line leak fix in ds4_expert_pager.c (active only with --ssd-streaming)
  + the gate test script; the resident inference path (ds4.c / ds4_metal.m /
  qwen4.metal) and the model are untouched (git diff a1198d6..HEAD). So the
  resident eval result cannot change. The paged path is separately validated by
  gate #5 (bit-identical) and the clean 32K paged run (rc=0, coherent decode).
  Not re-run to avoid burning hours re-confirming an untouched path.
- **logit gates pass:** yes (gate #5 bit-identical).
- **diskwrites_delta=0 every run:** yes.
- **read-GB in budget:** NO at 32K/220K on Stage-B (cache ≪ working set) — this is
  the item-1.4 signal, addressed by the later phases, not a leak.
- **≥40 tok/s @220K, zero swap:** NOT met on Stage-B (see Blockers #2).
- **CSV provenance columns:** present (git commit, model sha256, swap peak,
  output_checksum) in stageB-real.csv.

## Scratchpad artifacts (for the morning)
- apply_mtp_flag.sh + mtp_on_runs.txt — ready-to-apply MTP-on (needs your OK, gr-7).
- assemble_stageb.sh — rebuilds stageB-real.csv from run receipts.
- eviction_ab.sh, run_1_1.sh — the run wrappers used.
