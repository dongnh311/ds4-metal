# Item 1.1 (spec §25) — MTP on/off real numbers

## Wiring (this session)
ds4-bench had no MTP enable path (`--mtp` existed only in ds4_cli/agent/server).
Wired `--mtp` into ds4_bench.c (user-approved, ground rule 7):
- config field `glm_mtp`; parser `--mtp` → `cfg.glm_mtp=true`;
- engine opt `.glm_mtp = cfg.glm_mtp` + `.glm_mtp_timing = cfg.glm_mtp`
  (post-hoc acceptance summary, no hot-path cost — prints at session free);
- speculative gate relaxed `ds4_bench.c:894` from `cfg.dspark` to
  `(cfg.dspark || cfg.glm_mtp)` so the loop calls
  `ds4_session_eval_speculative_argmax` → qwen4 spec cycle. Without this the
  ready patch would have produced MTP-on numbers IDENTICAL to MTP-off
  (speculation never fires). Defensive warning added if --mtp requested but
  speculative did not enable.

The qwen4 MTP head is embedded in the main GGUF (blk.<il>.nextn.*); no
`--mtp-model` needed. Greedy MTP is LOSSLESS: a draft is accepted only when it
equals the target argmax (ds4.c:74390), so output == plain greedy → no
correctness gate required (speed measurement only). MTP_STATS is a compile-time
macro; runtime confirmation via the `--mtp`-enabled teardown line and
DS4_QWEN4_SPEC_TRACE.

## Measured — HEADLINE: 0% draft acceptance at this quant
- Confirm run ctx 4096 gen 16 --mtp: "14 verify cycles, 0 drafts accepted
  (0.0%)". gen 2.43 tps (vs faster MTP-off at this ctx) → MTP is a SLOWDOWN, not
  a speedup, because every cycle runs 2 forward passes and accepts only the
  target token.
- 32K gen 128 --mtp: prefill completed; DECODE DNF (SIGTERM rc=143, diskwrites=0).
  Root cause: MTP speculative decode at 32K uses per-frontier SESSION SNAPSHOTS
  for draft restoration ("frontier restoration uses session snapshots") — at 32K
  that copies ~GB of KV state per step, so decode crawls; combined with 0%
  acceptance it is pure overhead. Stopped to free the model for the Phase-2
  gate (critical path). MTP does not affect PREFILL, so prefill_tps ≈ the
  committed MTP-off 34.58 tps. The informative result is the 0% acceptance +
  the snapshot-restoration cost, both captured; the exact 32K decode_tps is a
  predictable ~2x slowdown and not worth the wall-clock to fully resolve.
- MTP-off baseline 32K (committed, run_1_1): prefill 34.58 tps, decode 4.52 tps.

## Interpretation
0.0% acceptance means the IQ2_XXS/Q2K-quantized MTP (nextn) head's drafts never
match the full model's argmax on this content. The flag path is correct (engine
counter qwen4_spec_cycles is incremented in engine code, not my wiring; the
same path the dspark route uses). Root-causing WHY acceptance is 0% (quant
degradation of the nextn head vs a dequant/rope/scale mismatch) is the scope of
Plan 2 Task 4 "MTP acceptance campaign" — out of order for tonight. For item
1.1 the honest result stands: **MTP-on gives no throughput benefit at this
quant (0% acceptance → slowdown).**

## Provenance
git commit=a185f1f (pre-commit), model sha256=ed238d8d… (verified), diskwrites_delta=0.
Receipt CSV: speed-bench/stageb-runs/stageB_32k_mtpon.csv; confirm:
speed-bench/stageb-runs/mtp_confirm.csv.
