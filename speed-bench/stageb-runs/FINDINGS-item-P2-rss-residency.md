# Item P2 (1) — RSS residency measurement + adaptive available-RAM cache budget

Follow-on to the memmgr over-allocation fix (0760384). That fix charged the FULL GGUF
span (41.72 GiB) to stay swap-safe, which was a conservative PROXY. This item measures
what is actually resident, then replaces the proxy with a measurement-driven adaptive
budget.

## Measurement (vmmap / footprint / ps, ssd-streaming, ctx 4096 gen 512, default cache)
Sampled at t≈25/45/70/100 s of decode — all stable:
```
GGUF mapping  7000000000-7a6eaf0000  [ 41.7G virtual   6.3G resident   0 dirty   0 swapped ]  r--/r-x SM=SHM
phys_footprint: 6.0 GiB (peak 6.0)   — does NOT grow during decode
Writable regions: 7.0G total, 5.8G resident
IOAccelerator (GPU): 5.4G total, ~4.7G resident   — the expert cache lives here (unified memory)
CSV rss_peak: 6.03 GiB   swap_growth: 0
```

### Findings
1. **The routed-expert region is NOT resident.** The 41.7 GiB model mapping holds only
   **6.3 GiB resident**; the routed experts (~35 GiB) are mapped but never faulted —
   they are read from the bundle, not the GGUF. The 6.3 GiB resident == the non-routed
   weights (`weights_streaming_non_routed_bytes` = 6.32 GiB). So the model's true
   resident cost is ~6.3 GiB, and the memmgr's 41.72 GiB span charge over-charged by
   ~35 GiB.
2. **The expert cache is GPU / unified memory** (IOAccelerator region), i.e. it counts
   as physical RAM on Apple Silicon and competes with every other process. So the
   earlier 43 GiB cache swapped because the *cache* (not the model) exceeded free RAM
   on this shared box — the real constraint is available unified RAM, not model size.

## Refined fix (ds4.c memmgr block, env-gated — default path byte-unchanged)
Replace the span-charge proxy with the measurement-driven budget:
1. Charge the model at its **real resident cost** (non-routed weights, 6.32 GiB).
2. Cap the grant by **grabbable RAM** = `free + speculative + purgeable`
   (`ds4_host_available_ram_bytes`, macOS host_statistics64), reserving KV + ~4 GiB
   graph buffers + 2 GiB safety, and by the bundle size.
   - `inactive` is deliberately EXCLUDED: on this shared dev box the ~40 GiB inactive
     bucket is mostly OTHER sessions' reclaimable working set — counting it reported
     ~50 GiB "free" when only ~8 GiB was truly grabbable (an earlier available formula
     that summed inactive+external reported 84 GiB > physical, and a 34 GiB grant
     swapped 18 GiB). free+speculative+purgeable ≈ what the OS can hand out without
     evicting anyone's live pages.

### Behaviour — adaptive, always swap-safe
- **This shared box right now** (5.85 GiB grabbable): cap → 0, keeps the engine's
  1.86 GiB budget, gen_steady 5.25 tps, **swap_growth 0**. Correct: there is no free
  RAM to grant, so it grants none (the static span-charge would have handed out 7.93
  GiB, which at this load would swap).
- **A dedicated 220K box** (free ≈ 50 GiB): grabbable ≈ 50 → cap ≈ 50 − KV − 6 →
  bounded by the 34.31 GiB bundle → grants the WHOLE bundle → ~100% hit → the
  DoD-friendly regime. The span-charge could NEVER reach this (it caps ~2.7 GiB at
  220K regardless of free RAM). So this is strictly better for the DoD while remaining
  swap-safe on a busy box.

Logit gate (run-logit-gate.sh 512 512) still PASS bit-identical (change is behind
DS4_QWEN4_MEMMGR; cache size never affects the math). diskwrites=0.

## Honest status
- This makes the memmgr correct and adaptive: it grants the largest cache that fits in
  *actually-free* unified RAM. On this loaded shared box that is near zero (the box is
  the limit, not the code); on a dedicated deployment box it unlocks the whole-bundle
  cache — the direct 220K read-GB lever.
- Supersedes the span-charge heuristic from 0760384 with a measurement-driven budget.

## Files / receipts
ds4.c (ds4_host_available_ram_bytes helper; memmgr block: non-routed model charge +
grabbable-RAM cap + bundle cap). Receipts: rss/ (vmmap/footprint samples via
measure_rss.out), memfix3_off.csv (adaptive cap, swap 0), gate_after_availcap.log.
