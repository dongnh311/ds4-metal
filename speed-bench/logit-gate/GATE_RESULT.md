# Gate #5 result: resident vs paged logit dump (item 1.3)

Status: **FAIL** — not "not bit-identical", but total corruption.

## Runs
- ctx=256: `/tmp/logit-repro/frontier_000256.logits.json` (paged only, ad hoc repro, not kept in repo)
- ctx=512: `speed-bench/logit-gate/resident/frontier_000512.logits.json` vs
           `speed-bench/logit-gate/paged/frontier_000512.logits.json`

## Finding
At both ctx=256 and ctx=512, the SSD-streaming (paged) path's frontier dump is
**0/248320 finite logits** — every lane is NaN. The resident path is fully
finite (248320/248320) with a sane argmax (id=198, logit=14.145 at ctx=512).
This reproduces at two different context sizes, so it is not a one-off.

argmax_id=0 on the paged side is an artifact of NaN-vs-NaN comparisons all
being false in a typical argmax loop (first candidate wins by default), not
a real prediction.

## Ruled out this session
- The ctx-8192 expert-cache floor fix (item 1.4, this session): does not
  engage at ctx 256/512 (`plan.cache_experts` from the base plan is already
  4526, far above the floor), so it cannot be the cause.
- Stage C async double-buffer prefetch: allocates its own buffers
  (`g_pager_async_l2_bufs`) but `grep` finds **no caller** anywhere in ds4.c
  for `ds4_expert_pager_get_l1_buf` / `swap_buffer` / `async_prefetch` /
  `async_wait` — it is dead infrastructure that cannot be feeding garbage
  into the real forward pass.
- Stage D predictive prefetch: same story, `record_selection` /
  `predict_next_layer` / `prefetch_predicted` have no callers either.

## Root cause (found)
Not a runtime bug at all -- **`qwen38-experts.bin`'s down-expert bundles are
corrupt**, produced by a one-line typo in the extraction tool.

`gguf-tools/qwen4_expert_bundle.py:46`:
```python
TYPE_BYTES = {
    ...
    10: 66,   # IQ2_XXS (256 elems per block)   <-- WRONG: GGUF type 10 is Q2_K
    ...
    16: 66,   # IQ2_XXS (duplicate mapping; verified by size analysis)
}
```
GGUF type 10 is **Q2_K** (84 bytes/256-element block), not IQ2_XXS (66
bytes/block) -- type 16 is IQ2_XXS. The script's own extraction-loop comment
even says so correctly two lines above the bug ("blk.N.ffn_down_exps.weight
type=10 (Q2_K)"), but the lookup table used to size the read was wrong.

`Reader.read_tensor_rows()` (`qwen4_expert_bundle.py:126-150`) uses
`TYPE_BYTES[t]` to compute `total_bytes`, `elem_size`, and therefore
`last_dim_bytes` (the byte stride between experts) for every tensor it
extracts. For the down tensor (type 10, dims `[768, 2560, 512]`), using 66
instead of 84 makes every computed per-expert byte range **both the wrong
size and the wrong start offset** (`start_byte = row_start * last_dim_bytes`
compounds the error for `expert > 0`). Gate/up are unaffected: they are
genuinely type 16 (IQ2_XXS), correctly mapped to 66.

Verified numerically against the real files:
- `qwen38-experts.index.json` records every down bundle as `"type": 10`,
  `"dims": [768, 2560]`, `"size": 506880`.
- Q2_K math for those dims: `(768/256) * 84 * 2560 = 645120` bytes -- the
  correct size, and (by construction) what the resident/GGUF-mmap path
  reads directly from the untouched, locked GGUF (which is why resident
  mode is correct).
- IQ2_XXS math for the same dims: `(768/256) * 66 * 2560 = 506880` bytes --
  **exactly** the bundle's recorded/actual size. Offsets between
  consecutive bundles in the index confirm the file really only contains
  506880 bytes per down expert, not 645120: it is genuinely truncated/
  wrong data on disk, not just a mislabeled-but-correct size.
- Every one of the 24576 down bundles (48 layers x 512 experts) is affected
  identically, which matches "every paged layer produces NaN" (confirmed by
  temporary per-layer tracing: first non-finite value appears at layer 2,
  the first routed layer that isn't the one "off the slab size class" layer
  bypassing the pager via direct mmap reads).

## Ruled out along the way
- The ctx-8192 expert-cache floor fix (item 1.4, this session): does not
  engage at ctx 256/512 (`plan.cache_experts` from the base plan is already
  4526, far above the floor).
- Stage C async double-buffer prefetch / Stage D predictive prefetch: dead
  infrastructure, no callers anywhere in `ds4.c`.
- Batching (the "mm" fast path, `T > 8`): capping the prefill chunk to 1
  token (forcing every paged forward call through the same T=1 shape as
  decode) did **not** fix it -- confirmed the bug is not about multi-token
  staging overwrite. A direct ctx=1 (single-token) run reproduces the same
  all-NaN output, which is what led to finding the real cause above instead.
- Everything in `ds4.c` / `ds4_metal.m` (`qwen4_bind_buf`, the `_with_bufs`
  kernel wrappers, `ds4_expert_pager_ensure`): these correctly assume
  standard Q2_K/IQ2_XXS sizing that matches the GGUF ground truth; they are
  not the bug, the bundle file handed to them is.

## Fix (not yet applied -- needs a decision, see below)
1. `gguf-tools/qwen4_expert_bundle.py:46`: change `10: 66` to `10: 84`.
2. Regenerate `qwen38-experts.bin` + `qwen38-experts.index.json` from the
   locked GGUF (sha256 `ed238d8d50c6fb4b504c2796c3bc82b2a9f6c21566ad29bdbfdbadfe01e28d7a`).
   Down bundles grow from 506880 to 645120 bytes each; total bundle size
   grows by `48*512*(645120-506880) = 3,398,553,600` bytes (+3.16 GiB), from
   31.12 GiB to roughly 34.3 GiB. This re-reads a large fraction of the
   147 GiB source GGUF and writes ~34 GiB fresh -- a real, one-time, large
   disk operation, not something to run silently.

## Consequence for the ledger
Item 1.4's just-committed eviction A/B numbers (hit rate, read volume,
tok/s) came from this same paged path at ctx 8192. The pager-level telemetry
(hits/misses/bytes read) is almost certainly still accurate -- it is counted
independently of whether the computed values are finite -- but the actual
generated tokens in that run were very likely NaN/garbage output, same as
here. This does not invalidate the pager cache-sizing conclusion (both E1
and E2 read the same bytes either way), but it means no correctness claim
should be drawn from that run's *output*, only from its cache statistics.

## Per ground rule 5
"Gate bit-identical ... TRUOC khi bench; fail = revert + bao." Stopping here
to report rather than deciding a revert target unilaterally (ground rule 8):
I cannot yet point to the single commit that introduced this without a
bisection, and reverting a broad range of Phase 1 SSD-streaming commits is a
bigger call than this item's scope.
