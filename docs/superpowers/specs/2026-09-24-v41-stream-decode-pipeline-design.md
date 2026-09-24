# V4.1 SSD-streamed decode pipeline (sub-project 1 of DoD 2) — design

Date: 2026-09-24. Status: approved in conversation (three sections), pending spec review.
Parent spec: `docs/V41_64GB_BUILD.md` (DoD 2, §8 steps 3–4).
Measured input: `speed-bench/v41/phase0-20260924/RESULTS.md`.

## Problem

DeepSeek-V4.1-Flash Q2 decodes at **9.04 t/s** on the M5 Pro 64 GB with SSD
expert streaming (ctx 8192, `switch` prompt, cache target 24 GB). Each token
takes 111.1 ms:

| Term | ms/token |
| --- | ---: |
| GPU busy | 57.0 |
| Miss path: `F_RDADVISE` readahead 17.5 + pread 16.6 | 34.1 |
| Other host work and per-layer sync | 19.5 |
| Engram | 0.6 |

The code map (read-only survey, 2026-09-24) shows why the host terms are so
large. Line numbers are at `4d5b307`; `C` is `ds4.c`, `M` is `ds4_metal.m`.

- **Two GPU waits per layer.** Streaming turns off the solo layer queue
  (C:41696), so every layer drains at its end (C:41726–41733). The routed MoE
  also waits once to read back the 6 selected ids (M:42200–42236).
- **The miss path runs on the main thread while the GPU is idle.** Readahead,
  slot preparation, pread (9-thread pool, but the caller blocks) and install
  all run inline (M:17258–17668).
- **V4.1 uses none of the existing overlap mechanisms.** V4-Flash already
  loads selected experts on a worker, overlapping the shared expert, with an
  early commit (C:26985–27096). None of it is wired for V4.1.
- **The one overlap V4.1 has is the "split" path.** It needs at least 3 misses
  in a layer (M:13345), but V4.1 averages about 1.2 misses per layer.

## Goal

Raise single-stream V4.1 Q2 decode from 9.04 to **at least 12 t/s** (band 3),
without changing any output. Conditions:

- same model and machine;
- V4.1 running alone: the gateway switched to V4.1, Qwen PROD and oMLX
  stopped (user decision 2026-09-24, answer "A");
- measured as in spec DoD 2.

This is the first of three sub-projects toward DoD 2 (≥ 20 t/s):

1. Streaming decode pipeline: this spec.
2. GPU kernels: fusions from antirez #1042 / Ivan `kernelpool-1073`, to move
   the 57 ms of GPU time toward the 38.4 ms byte floor.
3. DSpark with SSD streaming: exact-verify speculative decoding. No upstream
   code does this yet.

## Constraints

1. **Bit-identical output.** With each new feature on or off, logits, history
   and KV state match bit for bit. Every feature has a
   `DS4_METAL_DISABLE_V41_*` switch that restores the old behavior; it serves
   as the exactness control and as the rollback.
2. **V4.1-only changes.** Functions shared with the Qwen PROD streaming path
   must not change behavior. These are the pread pool and `pread_tasks`,
   `load_batch`, `prepare_load_buffers` / `take_reusable`, `install_loaded` /
   `set_addr_slot`, `prune_*`, `note_*`, `end_commands` and the slabs
   (survey §3, M:13608–16624, M:11704). New behavior is selected at V4.1 call
   sites or behind V4.1-only parameters. Functions that V4-Flash and GLM also
   use (`routed_moe_one_tensor`, `split_worthwhile`) keep their current
   behavior for those models.
3. **Zero swap and machine limits.** Wired memory stays at or below
   52.48 GiB (`vm.user_wire_limit`), with at least 40 GiB of disk free (swap
   files live on that disk).
4. **Speed is judged by interleaved A/B only** (off, on, on, off, in one run).
   Never against a stored number: the same binary drifts 6–12 % from one start
   to the next.

## Design

### Tooling (before any runtime change)

- **A/B driver.** Add an `ab` mode to `speed-bench/v41/phase0.py`. It runs one
  workload/context/cache point with an environment toggle in the order off,
  on, on, off. It reuses `run_one`'s guards: foreign-ds4 refusal, wired-drain
  wait, contamination poll, decode-window wired, atomic result files. It
  reports the mean of per-run steady t/s for each side, the ratio, and the
  profile terms (GPU, pread, readahead, other host).
- **Streaming exactness check that fits 64 GB.** Add a new mode to
  `tests/test_deepseek41_graph.c`, modeled on `check_decode_control`
  (`tests/test_deepseek41_graph.c:1212`). It opens the engine with SSD streaming and a caller-chosen
  cache (default 8 GB), runs a control session with a given
  `DS4_METAL_DISABLE_*` switch set and a candidate session without it, and
  compares logits, history and every KV/state span. It covers 65 decode steps
  after prefixes 511 and 2047. The upstream fixture asks for a 64 GiB cache and
  two sessions, which does not fit this box. The new mode also closes the open
  item in spec §0: the Engram resolution in `d3bf293` gets an exact check
  (`DS4_METAL_DISABLE_V41_ENGRAM_PARALLEL` as the switch).

### Step a — quick probes (no runtime change)

All at ctx 8192, `switch` prompt, 512 tokens, through the A/B driver.

- **a1. Readahead.** `DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD=1` vs
  default, at cache 24 GB. Readahead costs 17.5 ms/token inline. antirez #533
  reports that on an M5 Pro 64 GB it made preads 3–4× faster when it ran on a
  worker. The result decides step c: drop readahead for V4.1, or keep it off
  the main thread.
- **a2. Cache size.** Targets 24, 32 and 40 GB, recording t/s, hit rate,
  steady and peak wired, and swap. #810 fell at 32 GB on V4-Flash, and a bigger
  wired cache leaves less OS file cache for the misses.
  - The 7.12 GiB prefill reserve backs no buffer during decode (survey §4), so
    raising the target by 7 GB gives decode the same memory as reusing the
    reserve would. No reserve-reuse code is written.
- Cold-file-cache runs are out of scope, because V4.1 runs alone (Goal).

### Step b — keep decode layers queued under streaming

- **Change.** In `ds41_graph_step`, add
  `g->tp_world == 1 && g->streaming && !g->quality` to the solo `queue_layers`
  condition. `DS4_METAL_DISABLE_V41_STREAM_DECODE_QUEUE` switches it off. This
  is the shape of antirez #1034 (+11.3 % on M2 Ultra, same Q2 file), adapted
  to HEAD's solo-queue code.
- **Effect.** The end-of-layer drain goes away, leaving one GPU wait per layer:
  the selected-id readback. The host encodes layer N+1's attention while the
  GPU runs layer N's MoE. `queued_logits` then submits the vocabulary head
  before the final drain.
- **Kept drains.**
  - The last layer, before the token is published.
  - Layer 13 when `DS4_METAL_DISABLE_V41_ENGRAM_INPUTS` is set.
  - The id readback inside the routed MoE.
- **Safety.**
  - Cache slots used by in-flight command buffers are never evicted, via the
    existing in-flight tracking (M:15728–15765).
  - The Engram inputs are already double-buffered (C:41708–41725).
  - The plan must check that `ds41_router_log_capture` reads ids only after
    they are valid under queuing.
- **Acceptance.** The exactness check passes with the new switch as control.
  A/B shows a gain. It becomes the default only if both hold.

### Step c — load missed experts off the main thread

- **c1. Worker load, V4.1 call site.** The pieces already exist:
  `begin_selected_load` (M:16962), the pending-load wait in
  `load_selected_missing` (M:17294–17310), and
  `ds4_gpu_routed_moe_set_selected_override` (M:40998, consumed at M:42158).
  - After the router top-k is encoded (C:40951), signal an event and flush.
    Then encode the shared expert.
  - A worker waits for the event and reads the 6 ids. It resolves hits and
    misses and starts the readahead (as step a1 decides) and the preads into
    slots, all off the main thread.
  - The main thread passes the ids through the override. That removes the
    readback wait: only the pending load for that layer is awaited.
  - The load now starts one shared-expert time earlier and overlaps the
    shared expert plus host encoding. Estimated gain is 5–10 of the 34 ms.
  - Pattern: V4-Flash's `metal_graph_selected_async_load_*` (C:23620–23805,
    C:26985–27096). Switch: `DS4_METAL_DISABLE_V41_ASYNC_LOAD`.
- **c2. Split threshold for V4.1.** Allow the hits-first split from 1 miss
  instead of 3, for V4.1 only (V4-Flash and GLM keep 3). Resident experts'
  gate/up then run on the GPU while the missed ones are read. Switch:
  `DS4_METAL_DISABLE_V41_SPLIT_LOW`.
- **Shared functions.** Qwen's pread pool, `load_batch` and install code are
  called unchanged. `begin_selected_load` and `load_selected_missing` are not
  on Qwen's path (survey §3).

### Step d — cache

- **d1. Default cache size.** When the user gives no
  `--ssd-streaming-cache-experts`, V4.1 streaming uses the best size from
  step a2. It is capped by the wired headroom left at the active context and
  recomputed per context, never hard-coded (parent spec §8 step 3). An
  explicit flag still wins.
- **d2. Eviction victim scan.** Replace the full scan on each miss
  (M:15711–15750, about 1 ms/token today) only if the post-step-c profile
  shows it above 2 ms/token.

### Out of scope

- Lookahead prefetch (predicting layer N+1's experts). #849 lost 5.5 % on its
  own. Revisit only if the miss path is still the largest term after step d.
- Kernel fusions: sub-project 2.
- DSpark / MTP: sub-project 3.
- Any change to weight precision.
- Qwen code paths.

## Testing and acceptance

- **Per feature:** the new streaming exactness check passes, with the
  feature's switch as control, at cache 8 GB and at the default size. Existing
  model-free checks (`--router-log-format`, `--decode-profile-format`) and the
  87 Python tests pass.
- **Per commit touching `ds4_metal.m` or shared streaming code:** Qwen fast
  tier (`speed-bench/qwen-regression/run.sh fast`).
- **Per step:** A/B against the previous step with the driver; keep a step
  only if it is faster. Record the result in
  `speed-bench/v41/pipeline-<date>/`.
- **End of sub-project:**
  - Re-run the Phase-0 speed plan (16-row grid) on the same machine state and
    compare row by row with `phase0-20260924`.
  - The Qwen full tier is green.
  - The exactness check passes for all switches together (all off vs all on).
  - Target: ≥ 12 t/s at ctx 8192, `switch`, default cache.
- **Machine protocol:** GPU runs happen only after the user confirms the box is
  free (oMLX and the watchdogs down); restore them afterwards. Every run uses
  `caffeinate`, a swap watch and a stall watch (log unchanged for 15 min). A
  process stuck in `U` in `pread` is not killed with `-9`; ask for a reboot.

## Risks

- **Queued layers change completion order and hit a latent race** in the
  shared cache's in-flight tracking. Mitigation: exactness check at small
  caches (more evictions), and repeated runs.
- **Worker loads contend with the GPU for memory bandwidth, so GPU time
  rises.** Mitigation: the A/B profile terms show GPU busy per side; keep only
  net wins.
- **The 12 t/s target is not reached with b–d.** The remaining terms then go
  to sub-projects 2–3 with measured numbers. This spec's success is the
  measured decomposition plus every net-positive, exact change.
