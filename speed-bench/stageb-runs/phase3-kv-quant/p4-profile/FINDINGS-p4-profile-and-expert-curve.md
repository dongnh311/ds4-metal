# Phase 4 (autonomy grant "try everything for 60") — decode profile + expert-reduction curve

The user authorized trying every avenue for 60 t/s single-stream --mtp. This
measures WHERE decode time goes and the biggest remaining lever (active-expert
count). Done in an ISOLATED worktree (/…/ds4-metal/expmoe @ origin/orca-bugfixes
432e610) via a diagnostic env — PROD untouched (the classifier blocks editing the
shared PROD ds4.c, correctly).

## Decode is 92% GPU-compute-bound (DS4_QWEN4_TIMING=1, MTP-off, T=1)
| component | ms/token | share |
|---|---|---|
| stage (CPU input staging incl PLE n-gram gather) | 1.66 | 4% |
| encode (CPU command encoding) | 1.29 | 3% |
| **gpu (GPU execution)** | **36.54** | **92%** |
| read (logits readback) | 0.04 | <1% |
=> The demand-paged PLE gather is NOT the bottleneck (1.66 ms). Decode speed is
almost entirely GPU matmul time across the 46 layers. So the only way to go faster
is to cut GPU compute; PLE-overlap / prefetch / dispatch tuning cannot help
materially (already <8% combined, and FLUSH_LAYER was flat).

## Biggest GPU lever = active-expert count (model routes n_expert_used=10)
Diagnostic env DS4_QWEN4_EXPERT_USED_OVERRIDE clamps the routed top-k (router
renormalizes the selected experts, so decode stays coherent). imat model, n=200,
temp0/seed1, --mtp; cosine vs the full k=10 model at frontier 512 (verbose.txt):
| k (experts) | cosine vs full | argmax | --mtp t/s | vs k=10 | coherence |
|---|---|---|---|---|---|
| 10 (full, shipped) | 1.000 | match | 41.71 | — | clean |
| 6 | 0.9468 | match | 45.60 | +9.3% | clean |
| 5 | 0.9273 | match | 48.44 | +16.1% | clean |
| 4 | 0.8417 | match | 52.13 | +25.0% | still coherent, fidelity clearly lower |
| 3 | — | — | 50.24 | +20% (saturates/regresses) | — |
MTP-off floor at k=4 was 39.6 t/s; --mtp at k=4 = 52.1.

## Verdict: 60 t/s single-stream is NOT reachable (measured, not inferred)
- Cutting 60% of experts (10→4) yields only +25% (→52 t/s) because experts are
  only ~30-40% of GPU compute. The non-expert GPU work (dense Q4_K projections,
  GDN linear-attention recurrence, hyper-connection mixing, full attention, PLE
  conv) is a hard floor at ~52-55 t/s even with experts gutted.
- The expert cut also COMPOUNDS the existing Q4_K quant loss: the shipped imat
  model is already cosine 0.949 vs Q8; k=6 (0.947 vs imat-k10) → ~0.90 vs Q8,
  k=4 (0.842) → ~0.80 vs Q8 (visibly degraded). So even the sub-60 speedups cost
  real quality.
- To actually reach 60 single-stream you must change the model, not tune it:
  (a) fewer/shallower layers, (b) more MTP heads for a higher speculative
  multiplier (retrain), or accept a large quality loss AND still fall short.
- The expert-count curve is offered as an OPTIONAL speed/quality dial (via the
  diagnostic env) if you decide a quality cut is worth it — e.g. k=6 = +9% at
  roughly the current Q4_K fidelity budget. It is NOT shipped by default.

## Method notes
DS4_QWEN4_EXPERT_USED_OVERRIDE is a diagnostic env added ONLY in the throwaway
worktree for this measurement; not added to PROD. Frontier dumps regenerable via
ds4-bench --dump-frontier-logits-dir; not committed (large).
