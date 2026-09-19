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

## CLEAN RE-MEASURE 2026-09-19 16:41 (corrects the "swap-thrash / needs purge" note above)
Certified A/B, model RESIDENT (swapouts+0MB every run, base ~44-46 confirms residency),
ctx 8192, -n 300, --temp 0, imat + PLE demand-paged (DS4_QWEN4_PLE_PREFETCH_FULL=0):

| workload | base `--mtp` | PLD `--mtp` + DS4_QWEN4_PLD=1 | delta |
|---|---|---|---|
| copy / echo | 46.58 t/s | 59.23 t/s | **+27.2%** |
| realcode transform | 42.61 t/s | 41.26 t/s | -3.2% |
| chat | 41.33 t/s | 42.77 t/s | +3.5% |

**ROOT CAUSE of the earlier "flat ~37, no win, needs purge" runs: the A/B harness omitted
`--mtp`.** PLD lives inside `ds4_session_qwen4_spec_cycle`, which the engine only enters
when `engine->glm_mtp` is set (CLI `--mtp`). Without `--mtp` decode takes the MTP-off path
(~36.5 t/s) and the PLD branch never executes -> base == PLD == ~37, which looked like a
thrashed box but was a missing flag. The ~28 GiB stale swap is real but IRRELEVANT: it is
never touched during decode (swapouts+0 with the model resident), so no `sudo purge` is
needed for a clean measure. **Correct A/B: `--mtp` on BOTH sides; certify residency by
swapouts delta (0) and base landing at ~44-46, not ~37.**

CORRECTED repro (note `--mtp`):
  base: ds4 --mtp --temp 0 -m <imat> --ple <sidecar> --ctx 8192 -n 300 --prompt-file <p>
  PLD : DS4_QWEN4_PLD=1 ds4 --mtp --temp 0 -m <imat> --ple <sidecar> --ctx 8192 -n 300 --prompt-file <p>
