# Task C — --ple sidecar ported onto #1062 multi-session batching (DELIVERED)

Branch `origin/orca-rebase-ple-batched` (commit 8c53ac7) = orca-rebase
(upstream/main 8db1d1d + Q4_K dense wiring, carries antirez/ds4 PR #1062 native
multi-session batched speculative decode) + the external PLE Q4_1 sidecar (--ple)
grafted additively. Built in isolated worktree; PROD untouched.

## What was done
The 64 GiB light qwen4exp model requires --ple (30 GiB external n-gram; the
upstream embedded BF16 table is ~95 GiB, won't fit). #1062's batching lives on the
upstream lineage that dropped --ple. Port makes the two n-gram paths COEXIST:
- Single choke point `qwen4_ngram_row` gains a sidecar branch (dequant Q4_1 row
  from the sidecar CPU mmap via qwen4_ref_row); every gather path (single-session
  stage_inputs, CPU ref, batched qwen4_batch_stage_ngrams) bottoms out there.
- `qwen4_ngram_read`/`_part` use an active-tensor helper so the embedded and
  sidecar tables share the batched multithreaded readers.
- Per-session PLE recurrent state (ple_prev/ple_hist per ds4_session graph) was
  already generalized by #1062 — no batching redesign needed.
- ds4_model +ple_model/ple_tensor; ds4_engine +embedded ple_model; --ple open
  block before weights_bind; BF16-only validation relaxed to admit Q4_1 sidecar;
  --ple flag+help re-added across cli/server/agent/bench/eval. ~9 files, ~200
  substantive lines in ds4.c.

## Validation (imat model, M5 Pro 64 GiB, full quality — verify-exact)
- **Single-stream greedy --mtp output BYTE-IDENTICAL to PROD orca-bugfixes**
  (0857670), twice (before and after diagnostic removal) => zero quality change.
- **Multi-session throughput** (ds4-server --batched-session, --mtp, temp0,
  200 tok/req, 0 errors, full quality):
  | config | aggregate t/s | vs single | per-stream |
  |---|---|---|---|
  | 1 stream (ref) | 39.2 | 1.0x | 39.2 |
  | 8 concurrent (ctx 4096) | 67.5 | 1.72x | 8.4 |
  | 16 concurrent (ctx 4096) | 79.7 | 2.03x | 5.0 |
  Plain batched (no --mtp) 8 concurrent: ~68 t/s (similar).

## Honest ceiling: ~2x aggregate, NOT 65x — and why
Decode on this model is GPU-BW-saturated by a SINGLE stream (Phase 4: 92% GPU).
Batching multiple streams shares the same saturated GPU, so aggregate throughput
improves utilization (~2x, plateauing ~80 t/s from 8->16) rather than scaling with
stream count. The PR's 65x figure is from a compute-bound-per-stream regime (larger
model / hardware where one stream leaves the GPU idle); it does not transfer to this
light BW-bound model on M5 Pro. The ~2x is still a real, full-quality serving win.

## Memory constraint
Each batched session needs its own KV cache. At ctx 32768, 8 sessions OOM the GPU
(kIOGPUCommandBufferCallbackErrorOutOfMemory) beside the 40.67 GiB model. ctx 4096
fits 16 sessions. So concurrency is bounded by ctx x sessions x KV vs 64 GiB — a
config choice (smaller ctx or fewer sessions for high concurrency), not a bug.

## Shipping decision (needs user)
This port moves the engine base from orca-bugfixes to the upstream lineage
(orca-rebase), gaining #1062 batching + all upstream fixes while keeping --ple and
Q4_K. Single-stream is byte-identical, so adoption is safe, but it is a base switch
(re-port the 4 Tier-1 fixes forward, re-run the shipped-quality gate). Branch is
pushed for review; PROD remains orca-bugfixes until you approve the switch.
