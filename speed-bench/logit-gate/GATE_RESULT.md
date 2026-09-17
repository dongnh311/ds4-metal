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

## Not yet root-caused
The real data path is the synchronous `ds4_expert_pager_ensure()` call and
whatever in ds4.c's qwen4 graph consumes the returned pointers (staging
write / `qwen4_bind_buf` binding, from the prior session's Stage B wiring
work). Item 1.5's kernel fix (metal/qwen4.metal, non-finite router logit ->
no-expert routing) is already in the tree and did not prevent this, so
either a different code path than the router is corrupting expert weights,
or the router-logit case itself is still reachable via a path the fix
doesn't cover.

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
