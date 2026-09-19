# Span-verify DONE: byte-exact N-row prompt-lookup, +25-34% on echo output (conditional win)

Completed the deep task the user chose (#1): make the multi-token prompt-lookup SPAN
verify byte-exact by generalizing the engine's per-row exact attention from T<=3 to
arbitrary T. Built in the isolated worktree exp-pld (commit 668de4b); PROD untouched.

## What was needed (and done)
1. **N-row exact attention + hc_mix split.** The engine's `verify_rows_exact` splits the
   attention core and hc gate/mix into <=2-row sub-batches so each verified row's numerics
   BIT-MATCH the T=1 decode. It was gated to T==3; generalized both to arbitrary T (the
   attention split was already a `sub0+=2` loop; the hc_mix split was hardcoded 2+1 ->
   made a loop). The Metal-side geometry gates (matvec nsg, GDN scan grouping, MoE rows)
   are byte-INVARIANT by design (confirmed in-source), so they needed no change, and GDN
   is batch-invariant. Result: the T>3 span forward is byte-identical to baseline decode.
2. **Skip replay on full accept.** On full accept the verify forward already advanced the
   recurrent state over all committed tokens -> take the last row's logits, no replay
   (this is the speed win). On partial accept: restore the pre-block snapshot `snap0` and
   replay the accepted prefix.
3. **Confidence gates so it never meaningfully hurts:** require a >=5-token context match
   (`min_ng`), cap the span to the match length, only trigger for spans >= `minspan` (8),
   and a `cooldown` back-off after a wasted (partial) verify.
4. Fixed a real SIGSEGV: `qwen4_verify_logits` was sized `4*V` by the normal verify path;
   the span reads `(N+1)*V` -> overflow. Now `max(4, n_logit_rows)*V`.
Enable with `DS4_QWEN4_PLD=1` (span 8 default).

## Byte-exactness: PROVEN (independent of machine load)
Greedy output is BYTE-IDENTICAL to baseline on every prompt tested -- verbatim copy,
docstring rewrite, realistic code refactor, and chat prose. This is the hard guarantee
(the verify only commits the target argmax) and held across all runs.

## Speed (model resident, healthy machine; adjacent A/B pairs)
| workload                         | baseline | PLD span | delta   |
|----------------------------------|---------:|---------:|--------:|
| verbatim copy / echo-heavy       | ~47 t/s  | 58-63    | **+25-34%** |
| realistic code refactor (transform) | ~43   | 40-42    | ~ -2% (minspan 8) |
| chat prose                       | ~41      | ~40      | ~ -2%   |

## Honest verdict
- The mechanism is DONE and byte-exact. It is a **conditional** win, not universal:
  - **Big win (+25-34%) when the output echoes the context** -- pasting a file back,
    "regenerate the whole file", RAG/long quotes, repetitive boilerplate.
  - **~neutral (-2%) on genuine code transformation and chat** (spans rarely match long,
    so the batching rarely fires; the confidence gates keep the overhead small).
- **The win requires the model RESIDENT.** Under memory pressure (model paged to SSD) the
  bottleneck becomes disk and the batching gives nothing -- observed when a 18-load
  benchmark marathon swap-thrashed the box to a flat ~37 t/s for base AND PLD alike. The
  production gateway keeps the model resident, so the resident numbers are the relevant
  ones; but it means PLD is not a rescue for a memory-starved run.
- Because it is byte-exact and env-gated (OFF by default), it is safe to ship as an
  opt-in: it never touches the default path, helps echo-heavy sessions, and is
  ~neutral otherwise.

## Recommendation / decision
Ship as **opt-in** (`DS4_QWEN4_PLD=1`), off by default -- zero risk to normal serving,
real gains on paste-back / RAG / regenerate-file coding. It strictly subsumes the
earlier fixed-width +5.5% PLD. If you would rather not carry a conditional feature,
fall back to #2 (that fixed-width version) or drop PLD entirely.

## Artifacts
- Worktree exp-pld @ 668de4b; `span-verify.patch` (this dir) = full diff vs PROD base.
- Repro: `DS4_QWEN4_PLD=1 ds4 --temp 0 -m <imat> --ple <sidecar> --prompt-file <code>`.
  Tunables: DS4_QWEN4_PLD_SPAN (max span), _MIN_NG, _MINSPAN.
- NOTE: live re-measurement is currently blocked -- the box has ~28 GiB stale swap from
  the benchmark marathon; clean absolute numbers need `sudo purge` or an idle drain.
