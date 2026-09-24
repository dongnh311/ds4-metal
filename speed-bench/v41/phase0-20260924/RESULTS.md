# V4.1-Flash Q2 on M5 Pro 64 GB — Phase 0(b) results (2026-09-24)

This is the measured input for spec `docs/V41_64GB_BUILD.md` §7: the measured
roofline, the largest per-token term, and the candidate targets for DoD 2.
Raw tables: `RESULTS.generated.md`, `results.csv`, `*.locality.json` and
`bytes.json` in this directory. The readahead and pread-rate figures below
come from the `streaming expert timing total` line in each run's raw stderr
(`~/orca/workspaces/ds4-metal-data/v41-phase0/20260924/`, not in git).

## 1. Setup

| Item | Value |
| --- | --- |
| Model | `DeepSeek-V4.1-Flash-Q2.gguf` (antirez/deepseek-v4.1-flash-gguf), 365 713 686 528 bytes, SHA-256 `1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42` (verified by `download_model.sh`) |
| Binary | `ds4-bench` built from the C/Metal sources at `9a57ece`; no C/Metal change through `e466453` |
| Driver | `speed-bench/v41/phase0.py` at `71613f7` (4 locality runs, first 3 speed runs) and `1ea00cb` (last 9 speed runs; Python-only fix, section 7) |
| Date | 2026-09-24, 14:43–15:29 +07 |
| Machine | M5 Pro, 64 GB, macOS 26 (Darwin 25.6); idle wired 2.6–2.7 GiB before every run; no other ds4, oMLX stopped; swap delta within ±8 MiB in all 16 runs |
| File cache | warm: runs went back to back, the model file is not opened with `F_NOCACHE`, and nothing else used memory (see §2) |
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

Best clean `switch` row: ctx 8192, cache 24 GB, **9.04 t/s** (bench-measured;
the 111.1 ms mean step gives 9.0), hit rate 0.796.

| Term | Spec §2.1 estimate | Measured | Note |
| --- | ---: | ---: | --- |
| Byte floor (resident + routed at 290 GB/s) | ~36 ms | 38.4 ms | resident 8.14 GiB/token (spec said ~7.6; it left out 0.59 GiB of Engram projection weights) + routed 2.22 GiB |
| GPU busy | ≥ 36 ms | **57.0 ms** | 1.48× the byte floor |
| Miss path | 5–17 ms | **34.1 ms** | `F_RDADVISE` readahead 17.5 + pread 16.6; 465 MiB/token |
| Router → load sync and other host work | ~12 ms | **19.5 ms** | residual: step − GPU − pread − readahead − Engram |
| Engram | ~1 ms | 0.6 ms | as expected |
| **Step** | 54–66 ms (15–19 t/s) | **111.1 ms** | |

- **Kernel-efficiency factor = 57.0 / 38.4 = 1.48×** (whole-GPU busy time over
  the byte floor). The GPU path is close to its floor.
- **The miss path is warm-file-cache bound, not SSD bound.** pread moves
  465 MiB/token in 16.6 ms (27 GiB/s), and 13 GiB/s counting the readahead.
  Both are above the spec's measured random SSD rate of ~10 GB/s, so most
  misses were served from the OS file cache. That cache is not wired and stayed
  warm from run to run. The numbers hold for this box's intended use (one large
  model at a time, §1). With a cold cache the miss path will cost more.
- **Readahead runs outside the timed pread.** Each miss first issues
  `F_RDADVISE` (`ds4_metal.m`, `DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD`
  turns it off), so the raw `host_gap_ms` column in `RESULTS.generated.md`
  includes it. In all 16 runs the readahead bytes equal the decode pread bytes
  exactly.

Miss path by cache size (ctx 8192):

| Cache target | readahead ms | pread ms | miss path ms | pread GiB/s | other host ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 GB | 28.7 | 47.0 | 75.8 | 47.3 | 36.0 |
| 8 GB | 25.5 | 37.6 | 63.1 | 43.3 | 20.1 |
| 16 GB | 18.2 | 20.0 | 38.2 | 32.8 | 18.7 |
| 24 GB | 17.5 | 16.6 | 34.1 | 27.4 | 19.5 |

## 3. Cache size

Per cache target, ctx 8192 (4K and 32K follow the same shape):

| Cache target | t/s | decode hit rate | wired steady / peak GiB |
| ---: | ---: | ---: | ---: |
| 4 GB | 5.57 | 0.000 | 12.4 / 19.3 |
| 8 GB | 7.00 | 0.269 | 13.2 / 22.4 |
| 16 GB | 8.76 | 0.706 | 21.2 / 28.4 |
| 24 GB | 9.04 | 0.796 | 29.2 / 36.2 |

- **Consistent with #810 up to 16/24 GB; 32 GB untested.** #810 (V4-Flash, this
  box) measured 8 GB → 11.37, 16 GB → 11.71 and 32 GB → 8.1–9.5 t/s. Here decode
  rises monotonically up to 24 GB, and the sweep stopped before the point where
  #810 fell.
- **No #638 wiring penalty.** This runtime fills the cache with `pread` into
  slot buffers, not mmap-backed views. Wired memory grows by the cache size and
  nothing more.
- **Bigger cache, smaller file cache.** Every extra wired GiB leaves less room
  for the OS file cache that serves the misses (§2). That may be what bends
  #810's curve at 32 GB, so measure 32 GB before assuming it helps.
- **Returns fall off after 16 GB:** 16 → 24 GB adds only 0.28 t/s (+3 %).
- The highest peak is 40.5 GiB (ctx 32K, 24 GB). That leaves 12 GiB under the
  52.5 GiB user wire limit, so a 32 GB point is safe to try in §8 step 3.
- **Context cost is mostly unexplained.** At 24 GB, 32K runs 13 % slower than
  8K: +3 ms GPU, +6 ms pread, +8 ms other. Only the GPU part is clearly caused
  by the context. The hit rate also drops (0.722 vs 0.796), because teacher
  forcing decodes a different stretch of the `switch` text at each context.
  The 4K rows sit below 8K for the same reason (0.708 vs 0.796). Separating
  context from text needs runs that decode the same text at every context.

## 4. Locality (router log, 2000 tokens per workload, LRU after 200 warm-up tokens)

| Workload | LRU 8 GiB | LRU 16 GiB | LRU 32 GiB | LRU 48 GiB | pair overlap | cover K=1 | K=2 | K=4 | K=8 | new pairs/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code | 0.534 | 0.707 | 0.866 | 0.931 | 0.310 | 0.310 | 0.399 | 0.501 | 0.608 | 1.46 |
| docs | 0.558 | 0.693 | 0.828 | 0.898 | 0.335 | 0.335 | 0.429 | 0.520 | 0.610 | 1.20 |
| it | 0.681 | 0.795 | 0.896 | 0.943 | 0.455 | 0.455 | 0.536 | 0.621 | 0.698 | 1.68 |
| switch | 0.583 | 0.722 | 0.859 | 0.916 | 0.364 | 0.364 | 0.450 | 0.538 | 0.627 | 1.34 |

- **Every workload is in the < 80 % bucket at 8–16 GiB.** The spec's 98 % and
  93 % hit cases (§2.1) do not hold on this box.
- **The engine roughly matches LRU**, once the prefill reserve is taken out of
  the target. This comparison is loose: the simulation uses 863 units at 8 GiB
  over tokens 200–2000, while the engine had 958 units over the whole 512-token
  run. switch: sim 0.583 vs engine 0.589 at 4K. Below the 240-unit working set
  the engine does better than LRU: 95 units measured 0.19–0.28, where LRU
  gets 0.
- **Prefetch ceiling is low.** The union of the last 8 tokens covers only
  61–70 % of the next token's experts. One-token overlap is 31–46 %, in line
  with the ~36 % Qwen prior. A predictive prefetch can hide some miss time but
  cannot make misses rare.
- **Few new units, long reuse distance.** Only 1.2–1.7 never-seen units appear
  per token (0.6 % of 240). Still, even a 48 GiB LRU reaches only 0.90–0.94, and
  64 GB cannot hold that next to the 9.4 GiB resident floor.

## 5. Largest term

At the best point, **GPU busy is the largest term (57.0 ms, 51 %)**. It also
has the least headroom: 18.6 ms above the byte floor. **The miss path (34.1 ms,
31 %) is the largest removable term.** The other host work (19.5 ms, 18 %) is
7.5 ms above the spec's 12 ms sync estimate.

Ordered lever list for §8:

1. **Miss path (§8 steps 3–4).** Up to 34 ms.
   - Overlap readahead and pread with GPU work instead of running them inline.
   - Take fewer misses: try a 32 GB target (the LRU sim gives switch 0.859 at
     32 GiB), and shrink the 7.12 GiB prefill reserve where the prefill chunk
     allows.
   - First, measure a cold-file-cache run and a
     `DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD=1` run, so the plan is built
     on the real cost.
2. **Per-layer sync and other host work (§8 step 4).** About 7.5 ms above the
   spec's estimate.
3. **Metal on the resident bytes (§8 step 6).** At most 18.6 ms to the floor.
4. **Above the roofline (§8 steps 8–9).** DSpark/MTP, cheaper non-routed
   weights.

## 6. Target options for DoD 2

Step times are derived from the best clean row (111.1 ms). Each lever row
assumes the rows above it are done.

| Scenario | Step ms | t/s | Band |
| --- | ---: | ---: | --- |
| Measured today (ctx 8192, 24 GB) | 111.1 | 9.0 (bench 9.04) | 2 (5–10) |
| Miss path (34.1 ms) fully hidden under GPU work | 77.0 | 13.0 | 3 |
| + other host work 19.5 → 12 ms (spec sync estimate) | 69.5 | 14.4 | 3 |
| + GPU 57.0 → 38.4 ms (byte floor) | 50.9 | 19.6 | 4 |
| Byte floor alone (no I/O, no sync) | 38.4 | 26.1 | 5 |

- **(a) ~9 t/s, band 2.** The measured best, with no code change. It is below
  the DoD-2 "< 10 t/s → re-profile" line, so it only works as a floor.
- **(b) 12–15 t/s, band 3.** Hide the miss path and bring the other host work
  down to the spec's sync estimate. GPU time stays as it is.
  - Also taking the GPU all the way to its byte floor gives 19.6 t/s on paper
    (band 4). That needs perfect kernels, so it is a ceiling, not a plan.
- **(c) ≥ 20 t/s, band 5.** This needs levers above this roofline. With the
  miss path hidden, the GPU at its byte floor, and only the spec's 12 ms of
  sync left, streaming Q2 on this box tops out at 19.6 t/s. The byte floor
  alone, with no I/O and no sync at all, is 26.1 t/s. The options:
  - DSpark/MTP (§8 step 8): more than one token per pass over the weights.
  - Cheaper non-routed weights (§8 step 9): needs user approval plus the §9
    quality gate. Resident weights are 79 % of the bytes per token.

**Chosen (user, 2026-09-24): (c) ≥ 20 t/s, keeping model quality.**
- Lossless levers come first: runtime work, then speculative decoding with
  exact verification.
- Cheaper non-routed weights come last and need approval plus the §9 quality
  gate.
- Recorded as spec DoD 2, together with how the target is measured.

The upstream anchors are consistent with this: V4.1 on an M5 Max (more memory
bandwidth than this box) reached 13.39 t/s warm (`077a257`). V4-Flash Q2 on
this exact box reached 11.37 t/s (#810).

## 7. Run notes

- **Task 9 model checks.** The router-log check passed: "V4.1 router log: 1280
  lines; logits identical with and without it PASS" (32 steps × 40 layers,
  SSD, 8 GB).
- **Engram exact check not run.** The upstream `--engram-parallel-ssd` check
  cannot run on 64 GB: its fixture asks for a 64 GiB cache, which is capped to
  47 GiB and leaves no room for the second session. The Engram merge resolution
  in `d3bf293` ran in every Phase-0 run on the default parallel path, but
  without an exactness check. This stays open in spec §0.
- **Speed plan stopped once.** It stopped after 3 runs because the kernel still
  held 10 GiB of the previous run's Metal wiring for a few seconds after exit.
  `1ea00cb` makes the driver wait up to 120 s for the wiring to drain.
- **All rows clean.** No run was contaminated, and decode-window wired sampling
  worked in every row (`wired_window = decode`).
