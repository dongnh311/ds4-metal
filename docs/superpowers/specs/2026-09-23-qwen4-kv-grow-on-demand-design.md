# qwen4: grow-on-demand KV and indexer buffers — design

Date: 2026-09-23. Status: approved in conversation, pending spec review.

## Problem

`qwen4_graph_alloc` sizes every context-dependent GPU buffer for the full `-c` capacity at session
start. On M5 Pro 64 GB with SSD expert streaming (`DS4_QWEN4_STREAM_FULL_LAYERS=32`, 6 GB expert
cache), those buffers are wired up front even when the conversation is short, which takes page cache
away from the model file. Streamed-expert cache misses then go to the SSD instead of the page cache.

Measured on unc48L (`...-Q4KDownPad768-DenseQ4Kselimat-MTP.gguf`), same prompt, byte-identical output,
`-c 262144` vs `-c 65536`:

| case | -c 262144 | -c 65536 | cost |
|---|---|---|---|
| short chat, decode | 41.7 t/s | 42.9 t/s | -2.8% |
| 37K code summary, decode | 38.3 t/s | 39.6-40.0 t/s | -3.4..-4.5% |
| 37K prefill | 330 t/s | 404 t/s | -18% |
| peak wired (37K prompt) | 49.16 GiB | 46.38 GiB | +2.8 GiB |
| streamed miss load | 638 us | 474 us | +35% |
| misses per miss-gate | 2.61 | 2.61 | same |

PROD runs `-c 229376`, so almost every request pays this. At a genuinely full context the memory is
needed anyway, so there is nothing to win there.

## Goal

Allocate the context-dependent buffers for the context actually in use, grow them on demand, and
shrink them at a session boundary. Output must stay byte-identical to the full-capacity allocation.
Metal kernels do not change.

## Non-goals

- Paged KV (kernels addressing KV through a page table). Rejected: every attention/prep/score kernel
  would change, which risks byte-exactness, for no gain over copy-on-grow.
- CUDA and ROCm paths, one-shot `generate_qwen4_metal_argmax`, and test graphs keep full allocation.
- The startup memory plan (`ds4_context_memory_estimate_with_prefill_mode`, ds4.c:39749) keeps
  planning for the full `-c` (worst case, used for admission and placement). It overstates the 4-bit
  KV by about 3.8x; fixing that is a separate change because the estimate also feeds placement.
- Shrinking inside a session (for example after a rewind).

## Buffers affected

Per attention layer (13: trunk layers with `(il+1)%4==0` plus the nextn/MTP layer), sized by
capacity in rows:

- `layer_k_cache_fp8`, `layer_v_cache_fp8` (4-bit default: `rows*kv_dim/2` bytes; FP8: `rows*kv_dim`)
- `layer_k_scale`, `layer_v_scale` (`rows*(kv_dim/64)*2` bytes)
- `layer_k_cache`, `layer_v_cache` (half cache, only when not FP8/4-bit: `rows*kv_dim*2` bytes)
- `layer_block_key` (`(rows/4+1)*Di*2` bytes)
- `layer_ik_cache` only when the ik ring is off (`DS4_QWEN4_IK_RING=0`); with the ring on it depends
  on `cap_tokens` only.

Per graph: `pos3` (`rows*4` uint32), and the scratch `score` (`T*(rows/4+1)` floats) and `tile_max`
(`T*((rows/4+1)/8+1)` uint32), which may live in the shared engine arena.

All of these are row-major by position (or by block), so a prefix copy preserves them exactly. Kernels
take the buffers as raw pointers bound on every dispatch; nothing caches them (no argument buffers,
residency sets or reused command buffers). `score` rows use the current `n_blocks` as stride, not the
capacity.

## Design

### Capacity model

- `ctx_cap` keeps its meaning: the logical limit from `-c`. It still decides the maximum position, the
  ik-ring choice, the checkpoint header fields, and the startup memory plan.
- New `alloc_cap`: rows actually allocated. Capacity-sized buffers use `alloc_cap`; block-sized ones
  use `alloc_cap/4+1`.
- `DS4_QWEN4_KV_GROW=1` enables the feature for session graphs on Metal. Unset or `0`:
  `alloc_cap = ctx_cap`, identical to today.

### Policy

- Initial capacity 32768 rows (`DS4_QWEN4_KV_INIT_CAP` overrides), always a multiple of 4096 and never
  above `ctx_cap`.
- Reserve on sync: `need = prompt_len + 4096`. If `need > alloc_cap`, grow.
- Grow on demand: before every forward (prefill chunk, decode, MTP verify/draft, span verify, batched
  rows), if `pos + T + 64 > alloc_cap`, grow (64 rows covers the widest verify, 17 rows, with room).
- Grow target: `max(need, 2*alloc_cap)` rounded up to 4096, clamped to `ctx_cap`; if the result is
  above `ctx_cap/2`, jump straight to `ctx_cap` to avoid a second large copy.
- Shrink at a session boundary: when `ds4_session_sync` resets the graph (the new prompt does not
  extend the checkpoint) and `need <= alloc_cap/4`, reallocate at `max(init_cap, need)`. The 1/4
  hysteresis avoids grow/shrink thrash.

### Mechanism

`qwen4_graph_ensure_cap(g, need)` runs only at safe points: the start of a qwen4 entry point, before
`begin_commands`. Every qwen4 entry point ends with `ds4_gpu_end_commands`, which waits for all
command buffers, so the GPU holds no reference to the old buffers there. This is the same pattern as
`qwen4_batch_scratch_ensure` and the lazy snapshot allocation.

1. Allocate the new set at the new capacity.
2. Copy the used prefix on the CPU (Shared storage): KV/scale/`pos3` rows `[0, pos)`, block-key blocks
   `[0, pos/4]` (plus the ik rows when the ring is off).
3. Swap the pointers, free the old buffers, set `alloc_cap`.
4. On allocation failure, keep the old buffers and report the same error as "context full"; state is
   untouched.

Shared arena: its `score`/`tile_max` grow to the largest `alloc_cap` among borrowing sessions, and each
session re-reads its borrowed scratch pointers from the arena at the start of every forward, so no
session keeps a dangling pointer.

Transient memory: during a copy both sets exist. Worst case 128K to 256K adds about 0.9 GiB for
10-20 ms (peak about 50.2 GiB against the 52.48 GiB user wire limit).

### Touch points (ds4.c line numbers at develop 3f24d94)

- Allocation: `qwen4_graph_alloc` 58149-58346 (split capacity sizing into a helper reused by grow);
  session allocation 73794.
- Bound checks that guard buffers, preceded by `ensure_cap` and compared against `alloc_cap`: 59424
  (forward), 59768 and 59862 (MTP steps/chain), 78363 (decode), 75108-75139 and 75253 (spec/span room),
  batched 79548, 79878, 79921, 80983, 81066. Logical "context full" stays `pos >= ctx_cap`.
- Metal wrappers receive `cache_cap = alloc_cap` (ds4_metal.m:49624-49647 size checks).
- Batched rows: `ensure_cap` for each participating session before `qwen4_batch_row_graph` and staging
  (the by-value `rowg[]` copies are taken after that).
- Session sync 76261-76310: reserve after the checkpoint/reset decision; shrink in the reset branch.
  The plan must confirm every sync variant (`ds4_session_sync`, `_lockstep`, `_multimodal`) reaches it.
- Checkpoint load 63539-63660: `ensure_cap(rows)` before writing KV (check at 63553). Save is unchanged.
- Arena: borrow 58213-58217, transfer 58112-58125, fit check 73771-73773 and 80020.
- Log on grow/shrink: `ds4: qwen4 KV capacity <old> -> <new> rows (<delta> GiB)`.

## Testing

Correctness gate: with the flag on, greedy output is byte-identical to the flag off in the same
configuration (unc48L, K=32, 6 GB cache, MTP + 64K draft vocab, 4-bit KV, `-c 262144`):

1. Short prompt (no growth).
2. 40K prompt (growth at reserve).
3. 31K prompt + 2K generated tokens (growth mid-decode across the 32K mark, MTP active).
4. 256K prompt (jump to `ctx_cap`).
5. Batched session: two concurrent requests, one crossing a growth threshold.
6. KV disk checkpoint: restart the server and restore a checkpoint longer than 32K; compare flag on vs
   off with the same procedure.
7. Shrink: a 100K session followed by an unrelated short prompt; the log shows the shrink, system
   wired memory drops, output still matches.
8. Existing kernel tests still build and pass (`test_qwen4_kernels`, `test-qwen4-q2`).

Performance: paired A/B at `-c 262144`, flag on vs off, for short, 37K, 135K and 256K prompts: decode,
prefill, peak wired. Expected: short/medium close to the `-c 65536` numbers, 256K unchanged.

## Rollout

Implement on a branch in a separate worktree off `develop`, test-first. Merge to `develop` with the
flag off by default. Deploying to PROD follows docs/DEPLOY_AI_GATEWAY.md (`prod/<feature>-YYYYMMDD`,
`deploy-ai-gateway.sh cut/install/smoke`) and adds `DS4_QWEN4_KV_GROW=1` to the registry command, as a
separate step approved at that time. Turning the flag on by default is a later decision.
