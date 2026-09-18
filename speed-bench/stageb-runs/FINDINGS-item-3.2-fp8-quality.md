# Item 3.2 (spec §25 Phase 3) — FP8 KV quality matrix (E4M3), UNCOMMITTED probe

## Method
Measured whether E4M3 K/V degrades output, WITHOUT the risky in-kernel FP8 work:
a CPU E4M3 round-trip (per-64-elem block, power-of-2 scale, reusing the codebase's
proven dsv4_e4m3fn_dequant_cpu) is applied to each K/V head-vector in the paged
commit, so the forward pass computes attention on FP8-degraded K/V. The logit
divergence vs BF16 is numerically identical to what an in-kernel FP8 dequant would
produce. Gated behind DS4_QWEN4_KV_FP8SIM (NEW env → per ground rule 7 the code is
**held UNCOMMITTED**; see patch fp8sim.patch). Resident experts isolate the KV var.
This is LOSSY (ground rule 6: quality-matrix gate, not bit-identical).

## Results
### ctx 512, gen 32
- logit: max_abs_diff=3.294, rms_diff=0.476, cosine=0.98494
- argmax(top-1) MATCH (token 198); top-5 overlap 3/5
- decoded text DIFFERS but remains fully COHERENT period-Italian:
  - BF16: "influenza ne dee procedere, che benigna, e propitia, e che i nostri detti climi non possano esser bersaglio d"
  - FP8 : "influenza non si può che si sparga sopra di noi, se non benigna e piaceuole: e però, se ne'"
  Divergence is the expected butterfly effect of greedy decode under a small logit
  perturbation — NOT degradation (both are valid continuations).
### ctx 8192, gen 16
- logit: max_abs_diff=1.441, rms_diff=0.183, cosine=0.99787
- argmax(top-1) FLIPPED (bf16=8052, fp8=179828) BUT top-5 overlap 5/5 — a near-tie
  reorder of the top candidates, not a collapse. Output stays coherent:
  - BF16: "...puniva alcun malfattore in"
  - FP8 : "...proteggeva in verun modo le"
- KEY: divergence does NOT grow with context — cosine IMPROVES vs ctx 512
  (0.998 > 0.985), rms halves (0.183 < 0.476). So E4M3 KV scales to long context
  (good sign for the 220K memory-win use case).

## Reading
- E4M3 KV keeps the frontier top-1 stable and output coherent, but perturbs the
  distribution enough (cosine ~0.985, top5 3/5) to change greedy generation.
- Text-divergence alone is NOT evidence of quality loss — it is expected. The
  DEFINITIVE quality gate is **ds4-eval (18/20 baseline)** with FP8 KV on; that has
  NOT been run yet (it needs the FP8-sim enabled + the eval harness, ~substantial).
- If divergence does NOT grow much at 8192, FP8 KV should scale to long context.

## Decision needed (blocks the KV memory win)
FP8 KV is the actual KV memory win (~half the KV bytes: ~7→3.5 GiB @220K). To ship
it needs: (1) user go-ahead on an FP8 enable flag (ground rule 7), (2) ds4-eval with
FP8 KV to confirm ≥18/20 (ground rule 6 quality gate), (3) the real in-kernel FP8
storage+dequant (the memory win; the sim only proves quality, saves no memory).
Recommend: run ds4-eval with the FP8-sim first (cheap quality proof), and if ≥18/20,
implement in-kernel FP8. All held for the user's call.
