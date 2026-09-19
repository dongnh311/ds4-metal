# Span-verify feasibility probe: mechanism works, but byte-exactness at T>3 needs kernel work

Follow-on to FINDINGS-prompt-lookup-prototype.md (which showed PLD at the fixed 2-3
verify width gives only +5.5%, capped by width). This probes the multi-token SPAN
verify — the path to the real win — to decide feasibility before the deep build.

## What was built (probe, worktree exp-pld)
`DS4_QWEN4_PLD_SPAN=N`: when prompt-lookup finds a >=3-token context match, verify
`[first_token, span_1..span_N]` in ONE forward (`qwen4_graph_forward_tokens`, T=N+1,
all_rows), accept the longest argmax-matching prefix, then (probe: ALWAYS) restore the
pre-block snapshot `snap0` and replay the committed tokens for a clean recurrent state.
Widened `n_logit_rows` to N+1 (env-gated) so the forward can emit N+1 logit rows.
Fixed a real bug found en route: `qwen4_verify_logits` was sized `4*V` by the normal
path; the span path read `(N+1)*V` into it -> SIGSEGV. Now sized `max(4,n_logit_rows)*V`.

## Result (imat, M5 Pro, greedy)
| prompt                | baseline | span probe | accepted/cycle | byte-exact vs baseline |
|-----------------------|---------:|-----------:|:--------------:|:----------------------:|
| verbatim copy (echo)  | 46.8 t/s | 37.3 t/s*  | ~4.5 (447%)    | **IDENTICAL ✓**        |
| add docstrings/hints  | 41.2 t/s | 30.2 t/s*  | ~confident+near-tie | DIFFERS (see below) |
*probe always-replays (2 forwards/span) so it is deliberately slower; speed is not the
probe's question — correctness is.

- **Copy case: byte-identical.** The span mechanism (accept-k, snap0 save/restore,
  replay) is correct and commits ~4.5 tokens per cycle. Feasibility of the mechanism:
  PROVEN.
- **Docstring case: identical for the first 1651 characters, then ONE near-tie flip**
  (base "Args: r (float)" vs span "Args type?"), after which both continue coherently.
  This is a numerical argmax flip, not a bug.

## Why the near-tie flip: verify_rows_exact is T<=3 only
The engine's exact speculative verify (`verify_rows_exact`, `qwen4_graph_fused`) splits
attention into per-row sub-batches so each verified row's numerics BIT-MATCH the T=1
decode. That path exists only for T<=3. My span forward uses the T>3 prefill path
(batched attention) whose numerics differ by ~sub-logit amounts — enough to flip the
argmax on a near-tie (~1 per 1600 chars here). On confident tokens (verbatim copy) it
never flips, so copy is byte-exact; on uncertain prose it occasionally diverges.

## Verdict / what a shippable version needs
1. **Byte-exactness:** extend `verify_rows_exact` (per-row exact attention) from 3 rows
   to N rows — deep Metal kernel surgery on the attention path. Without it, the span
   verify is NOT byte-exact on non-verbatim output (mild, within the M5 tensor-route
   noise envelope, but a real change from baseline).
2. **Speed:** drop the always-replay on FULL accept (state is already correct there);
   replay only on partial accept. Easy, and turns the copy case into a real win
   (~4.5 tok/cycle in ~1 forward). Partial-accept-heavy output stays neutral/slower.

Net: the mechanism is proven and the buffer bug is fixed, but a byte-exact, faster span
verify requires the N-row exact-attention kernel work (item 1) — the high-risk piece.
Decision point: invest in the N-row exact attention, accept the mild non-byte-exact
divergence for the code-speed win, or stop at the safe +5.5% fixed-width PLD.

## Artifacts
- Worktree exp-pld (this commit): `pld.patch` covers the fixed-width PLD; span-verify
  code is in ds4.c (qwen4_pld_span2, qwen4_span_verify, DS4_QWEN4_PLD_SPAN).
- Repro: `DS4_QWEN4_PLD=1 DS4_QWEN4_PLD_SPAN=8 ds4 --temp 0 ...` in the worktree.
