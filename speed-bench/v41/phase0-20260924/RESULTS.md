# V4.1-Flash Q2 on M5 Pro 64 GB — Phase 0(b) results (2026-09-24)

This is the measured input for spec `docs/V41_64GB_BUILD.md` §7: the measured
roofline, the largest per-token term, and the candidate targets for DoD 2.
Raw tables: `RESULTS.generated.md`, `results.csv`, `*.locality.json` and
`bytes.json` in this directory.

## 1. Setup

| Item | Value |
| --- | --- |
| Model | `DeepSeek-V4.1-Flash-Q2.gguf` (antirez/deepseek-v4.1-flash-gguf), 365 713 686 528 bytes, SHA-256 `1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42` (verified by `download_model.sh`) |
| Binary | `ds4-bench` built from the C/Metal sources at `9a57ece`; no C/Metal change through `e466453` |
| Driver | `speed-bench/v41/phase0.py` at `71613f7` (4 locality runs, first 3 speed runs) and `1ea00cb` (last 9 speed runs; Python-only fix, section 7) |
| Date | 2026-09-24, 14:43–15:29 +07 |
| Machine | M5 Pro, 64 GB, macOS 26 (Darwin 25.6); idle wired 2.6–2.7 GiB before every run; no other ds4, oMLX stopped, swap delta 0 in all 16 runs |
| Diagnostics env | `DS4_V41_DECODE_PROFILE=1 DS4_METAL_GPU_BUSY_PROFILE=1 DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1`; locality runs add `DS4_V41_ROUTER_LOG` |
| Decode | `--teacher-forced-decode`, `--ssd-streaming`, prompt text from `phase0.py prompts` |

The cache target includes a **7.12 GiB prefill expert reserve** (ctx ≤ 32K,
prefill cap 2048). The decode cache is what is left:

| `--ssd-streaming-cache-experts` | decode cache | `cache_experts` (9.49 MiB each) |
| ---: | ---: | ---: |
| 4 GB | none | 1 |
| 8 GB | 0.88 GiB | 95 |
| 16 GB | 8.88 GiB | 958 |
| 24 GB | 16.88 GiB | 1821 |

One token routes 240 (layer, expert) units (40 layers × 6), so the 4 and 8 GB
points mostly measure direct reads. At 8 GB the engine itself warns: "expect
heavy thrashing".

## 2. Per-token decomposition vs the spec §2.1 estimate

Best clean `switch` row: ctx 8192, cache 24 GB, **9.04 t/s**, hit rate 0.795.

| Term | Spec §2.1 estimate | Measured | Note |
| --- | ---: | ---: | --- |
| Byte floor (resident + routed at 290 GB/s) | ~36 ms | 38.4 ms | resident 8.14 GiB/token (spec said ~7.6; it left out 0.59 GiB of Engram projection weights) + routed 2.22 GiB |
| GPU busy | ≥ 36 ms | **57.0 ms** | 1.48× the byte floor |
| Router → load sync + miss loads | 12 + 5–17 ms | **53.6 ms** (pread 16.6 + host gaps 37.0) | 2–3× the estimate |
| Engram | ~1 ms | 0.6 ms | as expected |
| **Step** | 54–66 ms (15–19 t/s) | **111.1 ms (9.0 t/s)** | |

**Kernel-efficiency factor = 57.0 / 38.4 = 1.48×.** The GPU path is close to
its byte floor. The Qwen MoE ran at ~3.5×, so V4.1's kernels are not the
problem the spec feared. Host gaps (`step − GPU − pread − Engram`) are a
residual: they hold the per-layer router → host → load round trips and the
CPU work between command buffers.

## 3. Cache size

Per cache target, ctx 8192 (4K and 32K follow the same shape):

| Cache target | t/s | decode hit rate | wired steady / peak GiB |
| ---: | ---: | ---: | ---: |
| 4 GB | 5.57 | 0.000 | 12.4 / 19.3 |
| 8 GB | 7.00 | 0.268 | 13.2 / 22.4 |
| 16 GB | 8.76 | 0.705 | 21.2 / 28.4 |
| 24 GB | 9.04 | 0.795 | 29.2 / 36.2 |

- **The #810 shape does not reproduce.** Decode rises monotonically with the
  cache up to 24 GB: #810 saw 8 GB ≥ 16 GB > 32 GB on V4-Flash. This runtime
  fills its cache with `pread` into slot buffers, not mmap-backed views, so the
  #638 wiring penalty does not appear. Wired memory grows by the cache size
  and nothing more.
- **Returns fall off after 16 GB:** 16 → 24 GB adds only 0.28 t/s (+3 %).
- The highest peak is 40.5 GiB (ctx 32K, 24 GB). That leaves 12 GiB under the
  52.5 GiB user wire limit, so a 32 GB point is safe to try in §8 step 3.
- Context costs speed. At 24 GB, 32K runs 13 % slower than 8K (+3 ms GPU,
  +6 ms pread, +8 ms host gaps). The 4K rows sit below 8K most likely because
  teacher forcing decodes a different part of the `switch` text there (hit rate
  0.708 vs 0.795), not because of the context itself.

## 4. Locality (router log, 2000 tokens per workload, LRU after 200 warm-up tokens)

| Workload | LRU 8 GiB | LRU 16 GiB | LRU 32 GiB | LRU 48 GiB | pair overlap | cover K=1 | K=2 | K=4 | K=8 | new pairs/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code | 0.534 | 0.707 | 0.866 | 0.931 | 0.310 | 0.310 | 0.399 | 0.501 | 0.608 | 1.46 |
| docs | 0.558 | 0.693 | 0.828 | 0.898 | 0.335 | 0.335 | 0.429 | 0.520 | 0.610 | 1.20 |
| it | 0.681 | 0.795 | 0.896 | 0.943 | 0.455 | 0.455 | 0.536 | 0.621 | 0.698 | 1.68 |
| switch | 0.583 | 0.722 | 0.859 | 0.916 | 0.364 | 0.364 | 0.450 | 0.538 | 0.627 | 1.34 |

- **Every workload is in the < 80 % bucket at 8–16 GiB.** The spec's 98 % and
  93 % hit cases (§2.1) do not hold on this box. The LRU simulation agrees with
  the engine once the prefill reserve is taken out of the target: switch 8 GiB
  sim 0.583 vs 958 experts measured 0.588 at 4K. The engine beats LRU only below
  the 240-unit working set (95 experts: 0.20–0.27 measured vs 0 for LRU).
- **Prefetch ceiling is low.** The union of the last 8 tokens covers only
  61–70 % of the next token's experts, and one-token overlap is 31–46 %,
  consistent with the ~36 % Qwen prior. A predictive prefetch can hide some
  pread but cannot make misses rare.
- **The working set is small:** 1.2–1.7 never-seen units per token (0.6 % of
  240). A larger cache keeps paying off (48 GiB LRU → 0.90–0.94), but 64 GB
  cannot hold it next to the 9.4 GiB resident floor.

## 5. Largest term

At the best point, **GPU busy is the largest term (57.0 ms, 51 %)**. It is
also the one with the least headroom: 18.6 ms above the byte floor. **Host
gaps (37.0 ms, 33 %) are the largest removable term.** The spec estimated
~12 ms for the whole sync. pread is third (16.6 ms, 15 %).

Ordered lever list for §8:

1. **Host gaps / per-layer sync (§8 step 4)**: ~25 ms above the spec's 12 ms
   estimate. Best ms per unit of work.
2. **Cache size and policy (§8 step 3)**: try a 32 GB target (the LRU sim
   gives switch 0.859 at 32 GiB); shrink the 7.12 GiB prefill reserve where the
   prefill chunk allows.
3. **Overlap pread with GPU work (§8 steps 3–4)**: up to 16.6 ms.
4. **Metal on the resident bytes (§8 step 6)**: at most 18.6 ms to the floor.
5. **Above the roofline (§8 steps 8–9)**: DSpark/MTP, cheaper non-routed
   weights.

## 6. Target options for DoD 2

Step times are derived from the best clean row (111.1 ms). Each lever row
assumes the rows above it are done.

| Scenario | Step ms | t/s | Band |
| --- | ---: | ---: | --- |
| Measured today (ctx 8192, 24 GB) | 111.1 | 9.0 | 2 (5–10) |
| Host gaps 37 → 12 ms (spec sync estimate) | 86.1 | 11.6 | 3 |
| + GPU 57 → 38.4 ms (byte floor) | 67.5 | 14.8 | 3 |
| + pread fully hidden under GPU work | 50.9 | 19.6 | 4 |
| Byte floor alone (no I/O, no sync) | 38.4 | 26.1 | 5 |

- **(a) ~9 t/s, band 2.** The measured best, with no code change. It is below
  the DoD-2 "< 10 t/s → re-profile" line, so it works only as a floor.
- **(b) 12–15 t/s, band 3.** The host gaps come down to the spec's sync
  estimate and the GPU moves most of the way to its byte floor. Levers 1–4 on
  this runtime only; nothing above the roofline.
- **(c) ≥ 20 t/s, band 5.** Even with every overhead gone, streaming Q2 on this
  box tops out near 19.6 t/s (26.1 t/s with I/O fully hidden). Beyond that it
  needs levers above the roofline: DSpark/MTP (§8 step 8), or cheaper
  non-routed weights (§8 step 9, user approval plus the §9 quality gate).
  Resident weights are 79 % of the bytes per token.

**Chosen (user, 2026-09-24): (c) ≥ 20 t/s, keeping model quality.** Lossless
levers come first (runtime work, then speculative decoding with exact
verification). Cheaper non-routed weights come last and need approval plus the
§9 quality gate. Recorded as spec DoD 2.

The upstream anchors are consistent with this: V4.1 on an M5 Max (more memory
bandwidth than this box) reached 13.39 t/s warm (`077a257`). V4-Flash Q2 on
this exact box reached 11.37 t/s (#810).

## 7. Run notes

- Task 9's `--engram-parallel-ssd` exact check could not run on 64 GB. The
  upstream fixture asks for a 64 GiB cache, which is capped to 47 GiB and
  leaves no room for its second session. The Engram merge resolution in
  `d3bf293` ran in every Phase-0 run (default parallel path) without an
  exactness check.
- The speed plan first stopped after 3 runs. The kernel still held 10 GiB of
  the previous run's Metal wiring for a few seconds after exit. `1ea00cb`
  makes the driver wait up to 120 s for it to drain.
- No run was contaminated, and the decode-window wired sampling worked in
  every row (`wired_window = decode`).
