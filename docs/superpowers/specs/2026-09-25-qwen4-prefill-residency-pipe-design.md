# qwen4: prefill residency and drain-free staging pipe — design

Date: 2026-09-25. Status: approved in conversation, pending spec review.

## Problem

Qwen PROD (unc48L `...-Q4KDownPad768-DenseQ4Kselimat-MTP.gguf`, `--ssd-streaming`,
`DS4_QWEN4_STREAM_FULL_LAYERS=32`, 6 GB expert cache, `DS4_QWEN4_KV_GROW=1`, `-c 262144`,
`--prefill-chunk 2048`, `--mtp`) prefills long prompts at 358-480 t/s. A 20K-token agent turn waits
about a minute before the first token. Slow prefill is cause #2 of "ds4 feels slow".

A spike on 2026-09-25 (CLI, 36,553-token code prompt) measured the GPU busy only 51.6 s of a 79.8 s
prefill (35 % idle). Two causes account for the idle time:

| Cause | GPU idle per 36K prefill | Mechanism |
|---|---|---|
| Command-buffer launch latency | ~17 s | Under `--ssd-streaming`, the 17 resident model-view spans (36.75 GiB) are in no `MTLResidencySet` (`ds4_gpu_model_residency_request_views` returns early in streaming mode). The driver revalidates them lazily, so `GPUStartTime - commit` reaches ~0.5 s on the command buffer after layer 47 of every chunk and ~0.13 s on the resident-phase buffer. |
| Per-layer staging drain | 11-20 s | `ds4_gpu_qwen4_stream_stage_layer` ends the batch and waits for the GPU at each of the 16 streamed layers, then reads the selected experts (~0.79 GiB) before any GEMM can run. The 15 GiB streamed region does not fit the page cache (cyclic LRU thrash), so these reads mostly come from the SSD at ~11 GiB/s. |

Spike results, all outputs byte-identical to baseline:

| Prompt | Baseline prefill | Residency | Residency + pipe (all chunks) | Decode-safe variant |
|---|---|---|---|---|
| 36K code | 460 t/s (399-489) | 497-542 | 592-607 | 563-585 |
| 134K code | 358-367 | - | 570 | 516 |
| 5.3K code | 476-497 | - | 543-548 | 505-520 |
| ~50-token chat | 78-84 | 124-125 | - | - |

The command-buffer launch latency summed 17.4 s without residency and 0.4 s with it.

Decode after a long prefill lost 2-7 % with residency + pipe on every chunk. The cause is that streamed-expert
miss loads rose from ~600 to ~900 us, because pinned spans and full-layer reads leave less useful page
cache. The decode-safe variant (last chunk unchanged, pipe reads with `F_NOCACHE`) measured -0.7 % at
36K and -2.5 % at 134K. Per-process CLI decode A/B was too noisy at 5K (-4 % to -15 %), because a previous
run's page cache leaks into the next. Settling this decode question needs a persistent-server paired A/B.

Rejected in the spike:
- `F_RDADVISE` window ahead: no effect.
- 18 pread threads: no effect.
- `--prefill-chunk 4096`: +5 % prefill, -4 % decode, not bit-exact.

The spike patch is kept outside git at `~/orca/workspaces/ds4-metal-data/spikes/qwen4-prefill-spike-20260925.patch`
(reference only; this design is implemented from scratch).

## Goal

- Cut qwen4 prefill time under `--ssd-streaming` by removing the launch latency and the per-layer
  staging drain.
- Keep output byte-identical in every mode.
- Choose the default mode by a paired server A/B that keeps decode at or above 97 % of PROD.

## Non-goals

- Other model families (GLM, DeepSeek V4/V4.1): their streaming paths stay byte-for-byte as today.
- `--prefill-chunk` changes, kernel changes, decode-path changes.
- Staging through the decode expert cache (deeper prefetch that would evict the decode cache).
- Non-streaming (fully resident) runs: they already request residency.
- PROD deployment: that is a separate, user-approved step (see Rollout).

## Design

### Mode switch and policy (`ds4.c`)

`DS4_QWEN4_PREFILL_MODE=off|safe|max`. Unset, empty or any other value means `off`. The engine
default is `off`; the PROD registry sets the mode chosen under Measurement (same pattern as
`DS4_QWEN4_KV_GROW`).

The graph gets a chunk-role field `prefill_role` with values `NONE`, `MIDDLE`, `LAST`. It defaults to `NONE`. Only
the two prompt loops set it:
- the CLI prompt loop (next to `g->stream_seed_last`, ds4.c:61194);
- the server session loop (ds4.c:77652).

Every other caller of `qwen4_graph_forward_tokens` (eval/logit paths, MTP verify, decode) keeps
`NONE` and therefore behaves exactly as `off`.

The pure function `qwen4_prefill_policy(mode, role, T, streaming)` returns three flags:
`{residency, pipe, pipe_nocache}`.

| Condition | residency | pipe | pipe_nocache |
|---|---|---|---|
| mode `off`, role `NONE`, `T <= 64`, or not streaming | no | no | no |
| `max`, role `MIDDLE` or `LAST` | yes | yes | no |
| `safe`, role `MIDDLE` | yes | yes | yes |
| `safe`, role `LAST` | no | no | no |

`T <= 64` is the existing threshold below which streamed layers do not take the staged GEMM path
(`mm_min`).

Streamed layers are the layers for which `qwen4_stream_expert_cache_addr_layout_supported` holds, in
ascending order. The layer staged after streamed layer `L` is:
- the next streamed layer, if there is one;
- after the last streamed layer of a `MIDDLE` chunk, the first streamed layer (its read overlaps the
  next chunk's resident layers);
- after the last streamed layer of a `LAST` chunk, none.

This is also a pure function.

### Unit A: prefill residency (`ds4_metal.m`)

- API: `ds4_gpu_prefill_residency_begin(void)` and `ds4_gpu_prefill_residency_end(void)`.
- The first `begin` builds a residency set from the mapped model views (`g_model_views`, the 17 spans
  under streaming).
  - It only commits the set: no `requestResidency`, and no `addResidencySet:` on the queue.
  - Decode command buffers and every other family therefore never carry it.
  - A view-generation counter bumped wherever the model views are rebuilt or released triggers a rebuild of the set at the next `begin`.
  - `ds4_gpu_cleanup` releases the set.
- While the scope is open, `ds4_gpu_new_command_buffer` (ds4_metal.m:1389) calls `useResidencySet:` on
  every command buffer it creates. This covers `begin_commands`, `flush_commands` and the pipe's commit.
  The queue keepalive thread creates its command buffers directly and is unaffected.
- The scope is a flag, not a nesting counter. `end` clears it.
- On macOS < 15, when `useResidencySet:` is unavailable, or when creating the set fails, the unit logs once and runs
  without residency. Output is unaffected either way.
- Call site: `qwen4_graph_forward_tokens` (ds4.c:60645) calls `begin` before `glm_graph_begin_commands_if_needed`
  when `policy.residency` holds. It calls `end` after the final `ds4_gpu_end_commands` (ds4.c:60793) on
  every exit path, including errors.

### Unit B: drain-free staging pipe (`ds4_metal.m`, bookkeeping in C)

**Entry point.** `ds4_gpu_qwen4_stream_stage_layer_pipe(...)` takes the same arguments as
`ds4_gpu_qwen4_stream_stage_layer` plus:
- the next streamed layer (its index and its gate/up/down offsets, or none);
- the `pipe_nocache` flag.

Only the prefill call site in `qwen4_graph_moe` (ds4.c:60420) uses it, and only when `policy.pipe` holds.
The decode fallback call site (ds4.c:60559) keeps calling the old function and never touches the pipe.

**Buffers.** Two staging buffers, each holding one whole layer: `need = 2 * gate_bytes + down_bytes`,
0.93 GiB on unc48L.
- Buffer 0 is also the union path's buffer (`g_qwen4_stage.buf`).
- Buffer 1 is allocated on the first pipe use in a prompt and released after the prompt's `LAST` chunk.
- Both fit inside the planner's existing prefill headroom (`DS4_STREAMING_PREFILL_HEADROOM_LAYERS`
  full layers, 1.86 GiB, already carved out of the 6 GB cache budget). The memory plan does not change.

**Jobs.** At most one job per buffer. A job holds:
- a state: idle, queued, done or failed;
- a sequence number;
- the layer and its three offsets;
- the model map;
- the `nocache` flag;
- the command buffer to wait for.

A stage call accepts a done job only if the layer, all three offsets and the model map match.

**Reader thread.** One thread, created on first use, serves queued jobs in sequence order. For each job it:
1. calls `waitUntilCompleted` on the job's wait command buffer (skipped if none);
2. reads the three whole tensors into the buffer at the same regions the union path uses (gate at 0,
   up at `gate_bytes`, down at `2 * gate_bytes`). It reads 32 MiB pieces with 8 reader threads of its own. It does not use the
   shared pread pool, so a union-path read on the main thread cannot collide with it. With `nocache`, it reads
   through a second model fd opened with `F_NOCACHE`.

**Per-buffer last reader.** Each buffer remembers the command buffer that last carried GEMMs reading it:
- When a stage call activates buffer `s` for layer `L`, `L`'s GEMMs are encoded into the open batch.
- The next pipe commit records that command buffer as buffer `s`'s last reader.
- `ds4_gpu_qwen4_stream_stage_chunk_end(last)`, called by `ds4.c` after the chunk's `ds4_gpu_end_commands`, clears every last reader (everything has completed).

**Stage call with a matching job for `L` in buffer `s`:**
1. Commit the open batch without waiting (flush) and keep the committed command buffer.
2. Queue the read of the next streamed layer (if any) into buffer `1-s`. Its wait command buffer is buffer `1-s`'s last reader, which may be the batch just committed.
3. Wait for `L`'s job (normally done already).
4. If it failed, run the union path for `L` into buffer `s`.
5. Point the gate/up/down binds at buffer `s` through the existing `g_qwen4_stage` redirect.

**Stage call without a matching job** (the first streamed layer of a prompt, or after a failure or invalidation):
1. Wait for any running job.
2. Mark both buffers idle.
3. Run the existing union path (drain, read `selected`, union read) into buffer 0.
4. Queue the next streamed layer into buffer 1 with no wait command buffer, since everything is drained.

**Chunk end.** `ds4.c` calls `ds4_gpu_qwen4_stream_stage_chunk_end(last)` after every prefill chunk whose
role is not `NONE`, whether or not that chunk used the pipe (a `safe` `LAST` chunk does not, but still ends
the prompt). It clears the last readers. With `last`, it also waits for any queued job, marks both
buffers idle and releases buffer 1.

**Failures and cleanup.**
- A read failure sends that layer to the union path.
- A buffer allocation failure disables the pipe for the process (logged once); the union path is used from then on.
- `DS4_QWEN4_STREAM_SEED_TOKENS > 0` forces the union path for `LAST` chunks, where seeding happens.
- `ds4_gpu_cleanup` stops the reader thread (stop flag, broadcast, join), then releases both buffers and the `F_NOCACHE` fd.

**Bookkeeping in C.** The job and buffer bookkeeping lives in a small plain-C header, `ds4_qwen4_stage_pipe.h`, shared by `ds4_metal.m` and the tests, so it is testable without Metal. It covers:
- slot choice;
- sequence order;
- the match rule;
- last-reader transitions;
- chunk-end reset;
- spare release.

### Diagnostics

Keep the spike's launch-latency measure under the existing `DS4_METAL_GPU_IDLE`. At exit it prints the
whole-run GPU busy union and the summed `GPUStartTime - commit` of waited command buffers. With
`DS4_METAL_STREAMING_EXPERT_PREAD_PROFILE`, the pipe prints one line per stage call: the layer, the main-thread
wait, the read time and the wait on the command buffer.

### Touch points (develop 500a306)

| File | Place | Change |
|---|---|---|
| ds4.c | graph struct next to `stream_seed_last` (58949) | `prefill_role` |
| ds4.c | new, near `qwen4_stream_seed_tokens` (8474) | mode parser, `qwen4_prefill_policy`, next-streamed-layer function |
| ds4.c | `qwen4_graph_moe` staging call (60420) | pipe entry when `policy.pipe` |
| ds4.c | `qwen4_graph_forward_tokens` (60645, 60793) | residency scope, `chunk_end` |
| ds4.c | CLI loop (61194), server loop (77652) | set and reset `prefill_role` |
| ds4_gpu.h | next to the stage API | residency scope, pipe entry, `chunk_end` |
| ds4_metal.m | `ds4_gpu_new_command_buffer` (1389) | attach the set inside the scope |
| ds4_metal.m | `ds4_gpu_idle_note_cb` / `ds4_gpu_finish_command_buffer` (1595, 1675) | launch-latency diagnostic |
| ds4_metal.m | next to `ds4_gpu_qwen4_stream_stage_layer` (51016) | pipe module |
| ds4_metal.m | `ds4_gpu_cleanup` (11866) | stop the pipe, release the set, buffers and fd |
| new C header | `ds4_qwen4_stage_pipe.h` | pure bookkeeping |
| tests/ | new model-free test + Makefile target | policy, next layer, bookkeeping |

## Testing

**Model-free** (TDD, part of the normal build, no GPU needed):
- `qwen4_prefill_policy` truth table: mode × role × T (64/65) × streaming.
- Mode parsing: `off`/`safe`/`max`/garbage/empty.
- Next streamed layer: middle, last streamed layer of `MIDDLE` (wraps) and `LAST` (none), a single streamed
  layer, and no streamed layers.
- Pipe bookkeeping:
  - queue and serve in sequence order;
  - match and mismatch (layer, offsets, map);
  - last-reader recorded on commit and cleared on chunk end;
  - failed job goes to the union path;
  - chunk end with `last` releases the spare and leaves no queued job.

**With the model** (GPU; only when the user allows and the machine is free, oMLX and watchdogs down and restored after):
- **Exactness:** CLI output byte-identical across `off`, `safe` and `max` on a ~50-token chat, 5K, 36K and 134K
  prompts. `speed-bench/qwen-regression/run.sh fast` passes with each mode set.
- **Quality:** the `qwen_gate` full-tier long needle is found with the chosen mode.
- **Memory:** peak wired at a ~256K prompt stays within the full-tier rule (baseline + 0.5 GiB).
- **Diagnostic:** summed launch latency at 36K is under 1 s with `safe` or `max` (17 s with `off`).

## Measurement and default selection

Add a long-prompt paired mode to `speed-bench/qwen-regression/qwen_gate.py`:
- Fresh servers in the existing `AB_ORDER` (PROD, branch, branch, PROD).
- Each server receives, back to back, a ~32K-token request and then a ~5K-token request. Each request asks for 300 tokens.
- Every request uses a distinct prompt (a different slice of the filler text), and each server gets a scratch KV
  directory, so the server's prefix cache can never skip the prefill being measured.
- It records prefill t/s and decode t/s per request.
- Each server gives one sample per request type. The verdict per request type compares the mean of the two branch servers
  with the mean of the two PROD servers, with the same 97 % floor as `speed_ab_failures`.
- The branch side runs with `DS4_QWEN4_PREFILL_MODE` set; there is one gate run per mode.

Rule:
1. Default `max` if its decode is at least 97 % of PROD on both requests.
2. Otherwise `safe` if it meets the same bar.
3. Otherwise stay `off` and report.

The chosen mode must also show at least +10 % prefill on the 32K request.

Results go to `speed-bench/qwen-prefill-pipe/RESULTS.md`.

## Rollout

- Merge to `develop` with the engine default `off`.
- Document `DS4_QWEN4_PREFILL_MODE` next to the other qwen4 env knobs.
- PROD needs a separate approval. When given: cut `prod/<feature>-YYYYMMDD` from `develop` with
  `deploy-ai-gateway.sh`, then add `DS4_QWEN4_PREFILL_MODE=<chosen>` to the registry's ds4 command.
- Rollback: remove the env (the binary behaves as today), or reinstall the previous prod branch.

## Risks

- **Decode regression on real traffic.** Mitigated by the paired server gate and the `safe` fallback.
  The mode is one env away from `off`.
- **Residency set pinning memory under pressure at 256K.** Mitigated by the full-tier wired check. The set is
  only attached during prefill chunks.
- **Hazard bugs in the pipe (reading a buffer the GPU still reads).** Mitigated by the last-reader rule,
  model-free bookkeeping tests, and byte-identical checks on long prompts where every chunk boundary and
  wrap is exercised.
- **Background reads during decode.** No job is queued past a `LAST` chunk, and the spare is released there.
