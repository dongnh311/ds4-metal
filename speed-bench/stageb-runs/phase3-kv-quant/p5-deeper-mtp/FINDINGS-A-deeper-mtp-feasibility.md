# Task A — deeper MTP draft (2-deep): feasibility (full-quality single-stream lever)

Goal: raise single-stream --mtp by drafting 2 tokens/cycle instead of 1, keeping
verify-exactness (only commit drafts == target argmax => zero quality loss).

## Timing model (from measured Phase-4 data)
- T=1 decode GPU = 36.5 ms. MTP cycle (T=2 verify) ~= 36-40 ms for up to 2 tokens
  (the 2nd verify row is nearly free — weight-stationary batch), giving today's
  1.67 tok/cycle => 42.8 t/s.
- A batched 3-row verify [first, d0, d1] should also cost ~37 ms (3rd row ~free).
  If both drafts accepted (est. 0.67 * ~0.5 = ~33% of cycles) => 3 tok / 37 ms.
  Projected ceiling: +11..19% at full quality => ~47-50 t/s (NOT 60).

## Why it is NOT a quick/safe change (blocking infra, verified in ds4.c @0857670)
1. `g->n_logit_rows = mtp ? 2u : 1u` (55109) — logits buffer holds 2 rows; a 3-row
   verify needs 3. Buffer + readback resize.
2. **Single snapshot point.** `snap_after_first` (54923) = "snapshot state after
   row 0" only; kernels take the snap target with a hardcoded row index `0u`
   (e.g. 55381). A 2-deep needs S2 = state AFTER ROW 1 so the "d0 accepted, d1
   rejected" case (est. ~33% of cycles) restores in O(1). Without S2 the only
   correct options are (a) an extra full ~36 ms forward, or (b) a full replay of
   [first,d0] — both make the deep path a timing WASH by the model above.
   Adding S2 means new snapshot buffers AND snapshot-at-arbitrary-row plumbing in
   the GDN linear-recurrent kernel, the PLE conv kernel, and the attention KV
   snapshot — i.e. Metal recurrent-state kernel surgery.
3. `T > 2u` guard in qwen4_graph_mtp_steps (55901) — the MTP head processes <=2
   rows; producing a 2nd draft d1 before the verify needs head chaining (d1 from
   d0's state), which is not wired (the state after d0 isn't available until the
   verify forwards it — a dependency the current single-draft design sidesteps).

## Verdict
A done as a genuine win = batched 3-row verify + a second recurrent-state snapshot
(S2) + head-chaining for d1. That is a substantial, delicate change to the GPU
recurrent-state snapshot path. A subtle S2 bug silently corrupts the GDN/PLE/KV
recurrent state and degrades long-context quality in ways a short byte-identical
test may not catch — directly against the "keep quality" mandate. And even done
perfectly it projects to ~47-50 t/s, not 60.
Recommendation: pursue only as a dedicated, byte-exact-gated effort if a modest
full-quality single-stream gain is wanted for its own sake; it is NOT a path to 60.
Prioritized C (multi-session throughput) instead — quality-safe and far higher
value. This analysis stands on the infra; no risky partial was shipped.
