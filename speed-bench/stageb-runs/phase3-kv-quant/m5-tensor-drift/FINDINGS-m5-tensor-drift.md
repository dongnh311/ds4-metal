# M5 tensor-route drift on M5 Pro: real but modest (argmax-stable to 4K), quant validated

Follow-up to the upstream finding (Ivan's branch `fix/m5-tensor-drift`, a11bf74):
on M5 the Metal4 `matmul2d` TensorOps accumulate path drifts from the legacy
simdgroup kernels; Ivan measured (M5 Max) whole-logit-unit drift with greedy
divergence and a first-token flip at ~3.7K context, and withholds the auto tensor
route on M5 by default. My runtime has it ON ("Metal 4 tensor API enabled",
tensor_matmul=on) — auto-enabled for M5/M6/A19/A20 at ds4_metal.m:2900-2918; the
only off-switch in this (pre-fix) build is `--quality` (g_quality_mode gates
`ds4_gpu_mpp_available` at :2757, but ALSO changes other kernels — so the A/B
below is "default vs quality-mode", not a pure tensor-route isolation).

## Measured on M5 Pro 64 GiB, light Orca (logit dumps @ frontier, cosine vs the other path)
| comparison | cosine | max_abs (logit units) | argmax |
|---|---|---|---|
| **runtime drift** Q8 tensor ON vs OFF @512 | 0.9936 | 1.05 | **preserved**, top5 5/5 |
| **runtime drift** Q8 tensor ON vs OFF @4096 | 0.9885 | 1.45 | **preserved**, top5 5/5 |
| runtime drift imat tensor ON vs OFF @512 | 0.9931 | 1.29 | FLIPS 16->17 (top-2 near-tied) |
| greedy 128-tok imat ON vs OFF | — | — | outputs DIVERGE (both coherent) |

## Verdict (this hardware/model/build — differs from Ivan's M5 Max)
- The tensor route DOES perturb logits (~1-1.5 units, cosine 0.988-0.994) and the
  drift GROWS with context (0.9936@512 -> 0.9885@4096). It is real.
- BUT on Q8 it does NOT flip the argmax through 4096 ctx (top5 identical) — much
  milder than Ivan's M5 Max (flip @3.7K). Likely because this build already runs
  the drift-patch compensations (hc_stable=on, norm_unify=on) and/or M5 Pro's
  TensorOps differ from M5 Max. It only flips on the more-quantized imat @512,
  where the top-2 are near-tied and ~1 logit of noise is the tie-breaker — not a
  gross error. Longer contexts (>4K) may eventually flip Q8; not tested here.
- Practical impact at typical contexts is modest: the quantization gap
  (Q8->imat, cosine ~0.954) is LARGER than the tensor-route gap (~0.99) at short
  ctx, so the drift is not the dominant fidelity factor here.

## Bonus (important): the shipped quant is validated / slightly better than reported
Isolating the drift (both models on the legacy/OFF path): **Q8-off vs imat-off
cosine = 0.9536** — HIGHER than the 0.9494 measured with the tensor route on both
sides. So the imatrix-Q4_K quantization is sound; the earlier 0.949 headline was
mildly pessimized by the shared tensor-route noise. The shipped HF models are fine.

## Recommendation
- Not urgent on M5 Pro given the modest measured impact (argmax-stable to 4K on Q8).
- For maximum fidelity (very long ctx, or close-call tokens), the legacy path is
  more faithful. Clean fix = adopt Ivan's withhold (a11bf74: tensor route
  off-by-default on M5, opt-in via env) when rebasing onto upstream/main. Neither
  upstream/main nor this build has the withhold merged — it's M5-hardware hygiene,
  independent of the quant work.

## Receipts (this dir)
drift_cosines.txt (all A/B cosines), greedy_on_vs_off.txt (divergence evidence),
drift_prompt.txt (fixed test prompt). Logit JSONs not committed (large; regenerable
via ds4-bench --dump-frontier-logits-dir [--quality]). See
[[../../../..]] upstream-antirez-ivan-learnings memory for the root-cause context.
