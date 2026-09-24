# Streaming pipeline probes (step a), 2026-09-24

Spec: `docs/superpowers/specs/2026-09-24-v41-stream-decode-pipeline-design.md`, step a.
Point: ctx 8192, `switch` prompt, 512 teacher-forced tokens, Phase-0 diagnostics env,
binary at `4419e6c` (no runtime change yet). V4.1 alone: oMLX and both watchdogs down.
Each row is one interleaved A/B (A, B, B, A) from `speed-bench/v41/ab.py`; A is the
24 GB cache target with default settings. B terms are per decode token (ms), means of
the two B runs; `readahead_ms` is the `F_RDADVISE` time, `host_ms` the host gap without it.
Raw runs: `~/orca/workspaces/ds4-metal-data/v41-pipeline/20260924/`.

| Label | A t/s | B t/s | Ratio | B GPU | B pread | B readahead | B host | B hit | B wired steady / peak GiB | B swap MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| readahead-off | 9.11 | 8.88 | 0.9753 | 56.9 | 36.5 | 0.0 | 19.1 | 0.796 | 29.2 / 38.2 | 0 |
| cache-32 | 9.22 | 9.54 | 1.0342 | 56.3 | 14.3 | 15.9 | 18.2 | 0.852 | 37.3 / 44.4 | 0 |
| cache-40 | 9.25 | 10.09 | 1.0903 | 58.0 | 11.3 | 13.8 | 19.1 | 0.891 | 45.3 / 45.4 | 0 |

A side (24 GB, default): GPU 56.6, pread 17.0, readahead 16.9, host 17.7 ms/token, hit 0.796.

## Verdicts

- **a1 `readahead_verdict = keep`** (ratio 0.9753 < 1.01). Without `F_RDADVISE` the
  inline pread doubles (16.9 -> 36.5 ms/token): the advisory read costs about what it
  saves when it runs on the main thread. Step c1 moves it to the worker; Task 5 re-checks.
- **a2 `best_cache_gb = 40`**: 24 GB 9.25 t/s, 32 GB 9.54 (x1.0342), 40 GB 10.09
  (x1.0903); 32 GB is not within 1 % of 40 GB. Hit rate rises 0.796 -> 0.852 -> 0.891,
  pread + readahead falls 33.9 -> 30.3 -> 25.1 ms/token, GPU busy rises 1.5 ms at 40 GB.
  No row is refused: decode-window wired peak <= 45.4 GiB, no swap growth. A 48 GB
  target does not fit `vm.user_wire_limit` (52.48 GiB) with the 7.12 GiB prefill reserve,
  so 40 GB is the largest practical target at ctx 8192.

## Runtime steps (A = feature switched off, B = on; 24 GB target)

| Step | Switch | A t/s | B t/s | Ratio | B GPU | B pread | B readahead | B host | Kept |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| b queue | `DS4_METAL_DISABLE_V41_STREAM_DECODE_QUEUE` | 8.92 | 9.53 | 1.0684 | 58.5 | 17.1 | 17.7 | 11.4 | yes |
| c1 async load | `DS4_METAL_DISABLE_V41_ASYNC_LOAD` | 9.65 | 10.05 | 1.0420 | 59.4 | 17.0 | 18.3 | 4.7 | yes |
| c1 re-check: readahead off (async on) | `DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD` (B) | 10.13 | 9.88 | 0.9753 | 59.6 | 35.5 | 0.0 | 6.1 | no: `readahead_verdict = keep`, Task 7 skipped |
| c2 split from 1 miss | `DS4_METAL_DISABLE_V41_SPLIT_LOW` | 10.28 | 10.27 | 0.9985 | 62.4 | 18.3 | 18.0 | -1.5 | no: reverted |
| d1 auto cache vs 40 GB target (queue + async on) | `--a-cache auto` / `--b-cache 40` | 10.96 | 11.21 | 1.0233 | 60.5 | 11.3 | 14.3 | 5.6 | no code (see d1) |

Under the async load, pread + readahead (about 35 ms/token) stay on the critical path: the
worker starts only one shared-expert time earlier. The c1 gain comes from dropping the
selected-id readback wait and host work (host 11.2 -> 4.7 ms/token). The timing counters
include the worker's time, so `host_ms` is a residual, not main-thread time.

c2 did split (14273 split layers per run vs 2472 at threshold 3), but GPU busy rose
4.3 ms/token (58.0 -> 62.4), which cancels the pread overlap: the extra command stage and
second routed bind cost what upstream's comment predicts. The code was reverted; the A/B
summary stays here as the record.

**d1 (default cache size).** The V4.1 Metal auto plan already sits below the measured best:
expert budget before the prefill reserve 35.62 GiB at ctx 8192 (3075 cached experts) and
31.63 GiB at ctx 32768, from the engine's safe-bytes limit (45 / 41 GiB for model +
experts, 56 GiB recommended working set). On this machine it can never exceed
0.86 x 56 - 9.37 = 38.8 GiB, so a 40 GiB cap would never bind, and on larger machines it
would cut a cache the measured trend says is faster. No cap was added. Auto vs an explicit
40 GB target at ctx 8192 (queue + async on): 10.96 vs 11.21 t/s (x1.0233, within the
2-3 % run spread; 40 GB wired steady 45.3 GiB, auto 40.9 GiB, no swap). Reaching 40 GB by
default would mean relaxing the engine's safe-bytes headroom for V4.1: a memory-policy
decision, left to the user.

**d2 (eviction scan).** Skipped: `reuse_scan` is 1.51 ms/token at 24 GB with queue + async
on (0.72 at 40 GB), under the 2 ms threshold.
