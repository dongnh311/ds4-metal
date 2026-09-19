# De-risk probe: is the more-MTP-heads GPU retrain worth it? Local evidence says BET, not sure thing

Before recommending GPU rental for the retrain path (SCOPE-more-mtp-heads.md, Option A),
measured the draft-depth acceptance/speed curve on the shipping imat model LOCALLY (no
training, no GPU) to see whether deeper drafting has headroom on THIS model + 2-bit quant.

## Method
- PROD engine (orca-rebase-ple-batched) `ds4 --mtp-timing --temp 0 -n 512` on a natural
  essay prompt (free-gen, greedy), imat main + PLE sidecar, ctx 4096.
- Swept draft depth via the existing `DS4_QWEN4_MTP_DEPTH` env (Task-A machinery):
  MTP-off / forced depth-2 / forced depth-3 / adaptive (default). Recorded generation
  t/s and the engine's own verify-cycle acceptance line.
- Note: the rank-histogram diagnostic (`DS4_QWEN4_MTP_STATS`) lives only on the scallop
  branch, whose bench lacks `--ple`; the PROD bench has `--ple` but not the histogram.
  So this uses the depth sweep + accept% (both available in PROD) instead — which
  measures the decision-relevant quantity (does deeper drafting pay) directly.

## Result (imat, M5 Pro, greedy, 512 tok)
| config            | gen t/s | verify cycles | accepted | accept% |
|-------------------|--------:|--------------:|---------:|--------:|
| MTP-off           |  36.49  | —             | —        | —       |
| depth-2 (forced)  |  41.73  | 309           | 202      | 65.4%   |
| **depth-3 (forced)** | **37.59** | 246       | 167      | 67.9%   |
| adaptive (default)|  42.15  | 307           | 201      | 65.5%   |

## Verdict — the GPU retrain is a genuine BET, not certain; local evidence tilts cautious
1. **MTP is already at its useful depth.** MTP-off→depth-2 = +14% (36.5→41.7); adaptive
   ≈ depth-2 (42.2). Going DEEPER with the current head **loses speed**: depth-3 = 37.6,
   BELOW depth-2 and barely above MTP-off. The 3rd drafted token is rejected often
   enough on general prose that its draft+overhead cost outweighs the wins. The adaptive
   policy already knows this (it engages depth-3 only on deterministic stretches), so
   there is **no free headroom left in the existing head**.
2. **The model's tokens are only ~65% predictable one-ahead** (first-draft accept 65%).
   That is a property of the model + 2-bit quant, and it caps how much ANY drafter helps.
3. **What a retrain would have to prove:** a DEDICATED trained head (predicting t+2/t+3
   directly, DeepSeek-V3 style) must beat the re-used head by enough to first turn
   depth-3 net-POSITIVE, then more to reach ~2.3–2.5 accepted/cycle for 60 t/s. The
   literature (DeepSeek-V3 ~85–90% 2nd-token accept, EAGLE 2.5–3.5×) is on FULL-PRECISION
   dense models; on a 2-bit quantized MoE with only ~65% first-draft accept, the
   achievable head acceptance is uncertain and likely below those numbers.
4. **Honest odds:** plausible outcomes range from "~45–50 t/s (misses 60)" to "~60–66".
   Rough estimate ~40–60% chance of clearing 60 — a real bet with a real downside, NOT a
   guaranteed improvement.

## Cheaper de-risk BEFORE a full retrain (recommended)
Run a bounded **pilot head-train**: train ONE dedicated extra nextn head (base frozen)
on a small self-distillation corpus for a few GPU-hours (~$50–200 rental), then measure
its depth-3 acceptance with this same sweep. Decision gate:
- If the pilot head makes depth-3 net-POSITIVE (t/s > depth-2) and 2nd/3rd-token accept
  climbs well above the re-used head's → the full retrain is likely to pay → commit.
- If the pilot head barely moves depth-3 → stop; ~43 t/s is the M5-Pro single-stream
  ceiling and multi-session ~2× is the only local lever.
This converts an open-ended GPU bet into a small, measured go/no-go.

## Receipts (this dir)
`depth_sweep.log` (the four runs above). Reproduce: `DS4_QWEN4_MTP_DEPTH=<2|3> ds4 --mtp-timing
--temp 0 -n 512 -m <imat> --ple <sidecar> -p <essay prompt>`. See SCOPE-more-mtp-heads.md
for the full option matrix and [[long-ctx-fidelity-and-mtp-retrain-scope]].
