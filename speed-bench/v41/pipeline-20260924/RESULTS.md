# V4.1 SSD-streamed decode pipeline: results (sub-project 1 of DoD 2), 2026-09-24

Spec: `docs/superpowers/specs/2026-09-24-v41-stream-decode-pipeline-design.md`.
Plan: `docs/superpowers/plans/2026-09-24-v41-stream-decode-pipeline.md`.
Baseline: `speed-bench/v41/phase0-20260924/RESULTS.md`.

## Result

Single-stream DeepSeek-V4.1-Flash Q2 decode on the M5 Pro 64 GB (ctx 8192,
`switch` prompt, 512 teacher-forced tokens), interleaved A/B in one run:

| Side | Configuration | t/s (runs) |
| --- | --- | ---: |
| A | Phase-0 configuration: 24 GB cache, no queue under streaming, no async load | 9.01 (9.00, 9.02) |
| B | Shipped defaults: auto cache (35.62 GiB budget, 3075 experts), queue + async load | **11.12** (11.10, 11.13) |

Ratio **x1.2336** (`ab-final3.json`). Every change is bit-exact against its
switch. The sub-project target (>= 12 t/s) is **not met**: 11.12 t/s.

## Setup

- Runtime commit `65071ee` on `feature/ds4.1-flash`; model
  `DeepSeek-V4.1-Flash-Q2.gguf` (365,713,686,528 bytes, SHA-256 `1ce6a8f8…6f42`).
- V4.1 alone: oMLX and both watchdogs down, no other ds4 process; zero swap
  growth in every run.
- Phase-0 diagnostics env (`DS4_V41_DECODE_PROFILE`, `DS4_METAL_GPU_BUSY_PROFILE`,
  `DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY`).
- Speed is judged only by interleaved A/B (A, B, B, A); raw runs are in
  `~/orca/workspaces/ds4-metal-data/v41-pipeline/20260924*/`.

## Steps

Details and per-term tables: `probes.md`. A is the feature off, B on, 24 GB
target unless noted.

| Step | Switch | A t/s | B t/s | Ratio | Kept |
| --- | --- | ---: | ---: | ---: | --- |
| a1 readahead off | `DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD` (B) | 9.11 | 8.88 | 0.9753 | readahead stays on |
| a2 cache 32 GB | `--b-cache 32` | 9.22 | 9.54 | 1.0342 | — |
| a2 cache 40 GB | `--b-cache 40` | 9.25 | 10.09 | 1.0903 | best size 40 GB |
| b queue decode layers under streaming | `DS4_METAL_DISABLE_V41_STREAM_DECODE_QUEUE` | 8.92 | 9.53 | 1.0684 | yes |
| c1 async load of missed experts | `DS4_METAL_DISABLE_V41_ASYNC_LOAD` | 9.65 | 10.05 | 1.0420 | yes |
| c1 re-check: readahead off with async | `DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD` (B) | 10.13 | 9.88 | 0.9753 | readahead stays on |
| c2 split from one miss | `DS4_METAL_DISABLE_V41_SPLIT_LOW` | 10.28 | 10.27 | 0.9985 | no, reverted |
| d1 auto cache vs 40 GB (queue + async) | `--a-cache auto` / `--b-cache 40` | 10.96 | 11.21 | 1.0233 | no code (below) |
| d2 eviction scan | — | — | — | — | skipped: 1.51 ms/token < 2 |

- **d1.** The auto plan (35.62 GiB at ctx 8192, 31.63 at 32768) already sits
  under the measured best and cannot exceed 38.8 GiB on this machine, so the
  planned 40 GiB cap would never bind. Moving the default to 40 GB (+2.3 %,
  within run spread) means relaxing the engine's safe-bytes headroom for V4.1:
  left to the user.
- **c2.** The split ran (14273 vs 2472 split layers per run), but GPU busy rose
  4.3 ms/token, cancelling the overlap.

## Regression found at the end, and its fix

The first grid re-run with queue + async on showed the 8 GB rows 8-12 % slower
than Phase 0 (`results-before-8gb-fix.csv`). Hit rate fell from 0.20-0.27 to
0.000.

| Config at 8 GB, ctx 4096, 128 tokens | Hit rate | Split layers |
| --- | ---: | ---: |
| queue + async | 0.000 | 0 |
| async off | 0.191 | 3498 |
| queue off | 0.191 | 3498 |
| both off | 0.191 | 3498 |

- **Cause.** The Metal expert cache marks entries in flight until a host wait
  on committed command buffers (`done_seq` moves only then). Queue alone keeps
  the per-layer selected-id readback wait; async alone keeps the end-of-layer
  drain. With both on, no host wait happens until the last layer, so every
  entry the token has used still looks in flight.
- **Effect.** The worker can only evict entries the token has not used yet,
  which are the hot experts of later layers. When every entry looks in flight,
  `take_reusable_batch` reaches `wait_inflight`, which flushes the batch from
  the worker thread (a data race). The same path, reached on the main thread,
  aborts with a committed-command-buffer assertion under
  `DS4_METAL_DISABLE_STREAMING_EXPERT_EARLY_LOAD=1`.
- **Fix (`65071ee`).** Below twice one token's routed working set
  (2 x 40 x 6 = 480 cached experts, the engine's own thrash-warning bound),
  V4.1 waits for committed command buffers before starting the worker (new
  `ds4_gpu_wait_committed_commands`, called only by V4.1). In-flight entries
  within a token never exceed 240, so larger caches always keep idle victims
  and skip the wait.
- **Why only below 480.** An unconditional wait cost x0.93 at 16/24 GB
  (`ab-final2.json`: 10.08 t/s; host +7.5 ms/token, one host wake-up per
  layer).
- **Check at the smallest cache that skips the wait (12 GB, 526 slots).** Hit
  rate 0.486 with async on and 0.486 with it off.

## Grid vs Phase 0

`results.csv` (final) vs `speed-bench/v41/phase0-20260924/results.csv`; hit
rates are identical row by row.

| ctx | cache GB | Phase 0 t/s | Now t/s | Ratio | Hit |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 4 | 5.53 | 6.24 | 1.128 | 0.000 |
| 4096 | 8 | 6.51 | 6.74 | 1.035 | 0.203 |
| 4096 | 16 | 7.86 | 8.67 | 1.103 | 0.589 |
| 4096 | 24 | 7.92 | 8.89 | 1.122 | 0.708 |
| 8192 | 4 | 5.57 | 6.33 | 1.136 | 0.000 |
| 8192 | 8 | 7.00 | 7.16 | 1.023 | 0.269 |
| 8192 | 16 | 8.76 | 10.06 | 1.148 | 0.706 |
| 8192 | 24 | 9.04 | 10.35 | 1.145 | 0.796 |
| 32768 | 4 | 5.28 | 5.74 | 1.087 | 0.000 |
| 32768 | 8 | 6.25 | 6.47 | 1.035 | 0.209 |
| 32768 | 16 | 7.60 | 8.37 | 1.101 | 0.605 |
| 32768 | 24 | 7.85 | 8.61 | 1.097 | 0.722 |

No row swapped or was contaminated. The decode-window wired peak is at most
40.6 GiB in the grid, and 47.9 GiB at the auto cache.

The 8 GB rows gain least. At 95 slots the wait is on every layer, and the hit
rate only matches Phase 0; the hits-first split does not improve.

## Per-token decomposition at ctx 8192 (ms/token)

| Term | Phase 0 (24 GB) | Now (defaults, auto cache) |
| --- | ---: | ---: |
| GPU busy | 57.0 | 57.0 |
| pread | 16.6 | 13.2 |
| `F_RDADVISE` readahead | 17.5 | 15.5 |
| Engram | 0.6 | 0.6 |
| Other host work and per-layer sync | 19.5 | 4.0 |
| **Step** | **111.1** | **90.3** |

Hit rate 0.796 -> 0.87 (bigger auto cache). The host term fell 15.5 ms: the
end-of-layer drain and the selected-id readback wait are gone. The pread and
readahead timers include the worker's time, so these two rows count work that
now runs off the main thread. The step time still pays them almost in full:
the worker starts only one shared-expert time before the routed MoE needs the
experts.

## What is left, for sub-projects 2-3

- **12 t/s needs about 7 ms/token more.**
- **The miss path, 28.7 ms/token, is the largest term that I/O can remove.**
  Only a load that starts earlier can hide it: predicting layer N+1's experts
  during layer N (out of scope here; #849 lost 5.5 % alone), or a larger cache
  (+2.3 % at 40 GB).
- **GPU busy is 57.0 ms against the 38.4 ms byte floor.** That is
  sub-project 2 (kernel fusions, antirez #1042 / Ivan `kernelpool-1073`).
- **>= 20 t/s (50 ms/token) is below today's GPU time alone.** It needs
  sub-project 2 and exact-verify speculative decoding (sub-project 3).

## Checks run on the final binary

- **Exactness.** Streaming exactness (`tests/test_deepseek41_graph --stream-control`,
  logits, history and every KV/state span for 65 steps after prefixes 511 and
  2047):
  - both switches together at 8 and 24 GB;
  - async at 4 GB;
  - each switch alone at 8 GB (before the fix);
  - queue in quality mode.

  Router log: 1280 lines, logits identical with and without it.
- **Qwen.** Fast tier PASS on the final binary. Full tier PASS on the final
  binary:
  - byte-identical PROD replies;
  - needle at 214,672 tokens hit;
  - wired steady 45.78 GiB;
  - paired A/B against PROD, medians 43.90 / 43.88 (PROD) vs 43.90 / 43.97
    (branch).
