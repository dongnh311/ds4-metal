# Prompt-lookup (n-gram) speculative drafting: works & byte-exact, but width-capped (+5.5% on echo)

No-GPU, no-training attempt to raise single-stream decode for the user's code/chat
workload (after the retrain path was ruled out for cost). Prototyped in an isolated
worktree of the PROD engine (orca-rebase-ple-batched); PROD untouched. Env-gated
`DS4_QWEN4_PLD=1` (diagnostic env; no new user flag).

## What it does
At the qwen4 MTP draft sites (`ds4_session_qwen4_spec_cycle`), when a match exists,
override the nextn head's drafted token(s) with a **prompt-lookup** proposal: find the
most recent earlier occurrence of the recent suffix (ending in the token the draft
follows) in the committed history (`s->checkpoint`, which includes the prompt) and
propose the tokens that followed it. Require a **≥3-token context match** (strong-only,
so it never displaces the trained nextn head on non-repeating output). It is
**verify-exact**: the draft is only a proposal; the existing MTP verify still commits a
token only when it equals the target argmax — so the output is byte-identical.

## Measured (imat, M5 Pro, greedy, worktree ds4)
| prompt                         | PLD off | PLD on | accept off→on | byte-exact |
|--------------------------------|--------:|-------:|:-------------:|:----------:|
| verbatim copy (echo-heavy)     | 46.70   | **49.25 (+5.5%)** | 88.6% → **95.3%** | identical ✓ |
| add docstrings/hints (non-echo)| 40.81   | 40.81 (neutral)   | 63.1% → 63.0%     | identical ✓ |

(An earlier version without the ≥3-gram gate slightly HURT the non-echo case, 64.5→59.2%
accept, by displacing the nextn head — the gate fixed that.)

## Verdict
- **The mechanism works and is safe.** PLD raises draft acceptance on output that echoes
  the context (88.6→95.3% on verbatim copy) and is byte-exact; with the strong-match
  gate it is neutral (never hurts) on non-repeating output.
- **But the current win is small (+5.5%) because the MTP verify is capped at 2–3 tokens
  per cycle.** Even when PLD knows the next 20 tokens verbatim, the engine commits at
  most ~2–3 per verify forward. Acceptance rose to 95% but throughput barely moved —
  **verify WIDTH, not draft source, is the binding constraint.**

## The real win needs a multi-token span verify (bigger, riskier)
To capture PLD's actual advantage (accept a long copied span in ONE forward), extend
`qwen4_graph_forward_tokens` verify from 2–3 rows to N rows, with N-depth GDN
linear-attention state snapshot/rewind (generalizing the existing `snap_after_first`/
`snap_after_second` machinery) and an accept-longest-prefix loop. On code-heavy /
copy-heavy output this could plausibly give 2–3× on those sections (vs +5.5% now); on
free-form chat, little. This is real engine surgery on the recurrent-state snapshot path
— the high-value, higher-risk follow-on. Decision point for the user.

## Status / artifacts
- Prototype: worktree `/Users/dongnh/orca/workspaces/ds4-metal/exppld`, branch `exp-pld`.
  PROD (orca-rebase-ple-batched) untouched. Not shipped.
- `pld.patch` (this dir): the full diff (helpers + 3 draft-site overrides, ~50 lines).
- `sample_copy_output.txt`: sample greedy output (identical PLD on/off).
- Reproduce: build ds4 in the worktree, `DS4_QWEN4_PLD=1 ds4 --mtp-timing --temp 0 ...`.
