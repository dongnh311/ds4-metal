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
