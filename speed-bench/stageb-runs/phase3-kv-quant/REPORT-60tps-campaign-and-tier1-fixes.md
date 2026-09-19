# Consolidated report — 60 t/s campaign + Tier-1 fixes (overnight run, 2026-09-19)

Model: `Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-DenseQ4Kselimat-MTP`
(imatrix-selective Q4_K dense) + Q4_1 PLE sidecar. Hardware: Apple M5 Pro 64 GiB.
Engine: PROD `~/.local/share/ai-gateway/ds4-metal`, branch orca-bugfixes.
All benches: run from PROD dir, temp0/seed1, model volume read-only (diskwrites=0);
receipts under speed-bench/stageb-runs/phase3-kv-quant/{p0-baseline,p1-validate,
p2-throughput,p3-gates}.

## TL;DR
1. **Four Tier-1 latent-bug fixes delivered and validated** (softplus log1p accuracy,
   moe_reduce isfinite guard, ignore_eos under --mtp, #1050 tool-call message).
   No quality regression, decode t/s unchanged. Engine commit orca-bugfixes 0857670.
2. **60 t/s single-stream --mtp is NOT reachable** on this model/hardware — now
   MEASURED, not inferred (Phase 4). Decode is **92% GPU-compute-bound** (PLE gather
   only 4%). The biggest lever, cutting active experts 10→4 (major quality loss,
   cosine 0.842), reaches only **52 t/s** because experts are ~30-40% of GPU compute
   and the non-expert work (dense/GDN/HC/attn) floors at ~52-55 t/s. Honest baseline
   is **~42.8 t/s**. Every in-scope engine lever (FLUSH_LAYER, MTP tuning, #1056
   port) yields nothing; see Phase 2 & 4. The ceiling is structural.
3. **All Tier-2 risk gates pass** (repetition, MTP verify-exactness, ignore_eos,
   M5 tensor-drift argmax-stable to 8K).

## Phase 0 — baseline
| config | mean t/s | MTP acceptance |
|---|---|---|
| single-stream **--mtp** (target metric) | **42.84** | 67.9% (161/237) |
| MTP-off (reference) | 36.55 | — |
MTP speedup +17.2%. Gap to 60 = +40.1%.

## Phase 1 — Tier-1 fixes (all validated; receipt p1-validate)
| # | fix | file | effect |
|---|---|---|---|
| 1 | softplus naive log(1+e) → Kahan log1p (no MSL log1p builtin) | metal/qwen4.metal | matches CPU ref in the recurrent GDN decay gate; argmax-stable, gain grows with ctx |
| 2 | isfinite guard on each expert partial | metal/qwen4.metal (moe_reduce) | one NaN/Inf slot can no longer poison the output + HC residual; bit-identical healthy path |
| 3 | ignore_eos honored under --mtp | ds4_server.c | MTP block no longer early-stops on eos when ignore_eos set (A/B proven) |
| 4 | clearer tool-name-before-parameter error | ds4_agent.c | #1050 diagnostic |
Validation: coherent; decode --mtp 42.29 vs 42.84 (unchanged); imat-vs-Q8 frontier
cosine 0.9909→0.9904 @512 (no regression), argmax preserved; log1p effect isolated
by metal-swap A/B (Q8/imat old-vs-new cosine 0.995, argmax MATCH).

## Phase 2 — why 60 t/s single-stream is not reachable (receipt p2-throughput)
- **2a FLUSH_LAYER sweep**: FLAT 42.1–43.0 across 0..12 (default 2); ~2% noise.
  Submission overlap already saturated → not a lever.
- **2b MTP tuning**: qwen4 MTP draft is hardcoded 2 with DS4_N_NEXTN_PREDICT=1 (one
  nextn head); `--mtp-draft`/`--mtp-margin` do nothing on this path. Greedy accept
  is strict exact-match (no margin). Acceptance 67% is the single-head ceiling. GPU
  argmax + swap-restore + next-draft batch-prep already on by default → not a lever.
- **2c #1056 kernel port**: dedicated git analysis of the row-reuse/predictor bundle
  found NO portable single-stream win — the good qwen4/M5 MTP kernels are already in
  orca-bugfixes; 666f9c7 is superseded by existing mv_ext r1ptg (and its fast path is
  M1-M4-only); 891c005 fixes a regression we never had; the rest are TP/DeepSeek/GLM/
  MXFP4/multi-session.
- **Root cause of the ceiling**: decode is dominated by per-token *overhead*, not
  weight bandwidth — 46-layer kernel dispatch + GDN linear-attention recurrent state
  (sequential per token) + CPU-side PLE n-gram gather (demand-paged; 30 GiB can't be
  resident beside the 40.67 GiB model on 64 GiB). Experts are already at the 2-bit
  floor and full-Q4K dense = +13% (same as selective), so quant won't close the gap.

### The only real paths to more single-stream speed (all out of current scope)
- (a) **More nextn/MTP heads** → more accepted tokens/cycle. Needs a model retrain.
- (b) **A shallower/smaller model** (fewer than 46 layers). Different model.
- (c) **Overlap/cut the PLE gather** (engine change; risk; needs a design + likely a
  new knob → would require your go-ahead per the no-new-flag rule).
- For **aggregate** throughput (NOT single-stream latency): the #1062 multi-session
  batching (65x @16 streams) via a --ple forward-port — the deferred big project.

## Phase 4 — "try everything for 60" (measured profile + expert-reduction curve)
After you authorized trying every avenue, I profiled decode and drove the biggest
lever in an isolated worktree (PROD untouched). Receipt: p4-profile/.
- **Decode = 92% GPU compute** (DS4_QWEN4_TIMING=1, MTP-off): gpu 36.54 ms/token vs
  stage/PLE 1.66, encode 1.29, read 0.04. PLE demand-paging is NOT the bottleneck.
- **Active-expert curve** (model routes 10; diagnostic env clamps top-k; --mtp;
  cosine vs full at frontier 512):
  | k | cosine vs full | --mtp t/s | vs full |
  |---|---|---|---|
  | 10 (shipped) | 1.000 | 41.7 | — |
  | 6 | 0.947 | 45.6 | +9% |
  | 5 | 0.927 | 48.4 | +16% |
  | 4 | 0.842 | 52.1 | +25% |
- **Even gutting experts to k=4 (visible quality loss) only hits 52 t/s** — the
  non-expert GPU floor (~52-55) blocks 60. The expert cut also compounds the Q4_K
  gap (k=4 ≈ 0.80 vs Q8). 60 single-stream needs a model change (fewer layers or
  more MTP heads = retrain), not tuning. The k-curve is an OPTIONAL speed/quality
  dial you can choose (e.g. k=6 = +9% near the current fidelity budget), not shipped.

## Phase 3 — risk gates (receipt p3-gates): ALL PASS
- ignore_eos under --mtp: fixed & A/B-proven (28→120 tokens). Gap noted:
  /v1/completions never parses ignore_eos (chat endpoint only; pre-existing).
- #1045 repetition: no degeneracy (distinct-line ratio 1.000, no repeated n-grams).
- MTP verify-exactness: greedy --mtp output byte-identical to plain greedy.
- M5 tensor-drift: argmax stable to 8192 (max_abs grows to 4.1 but top token holds).

## Shipping status
- Engine fixes committed to orca-bugfixes (0857670); receipts committed to
  dongnh311/scallop. NOT pushed (awaiting your review).
- HF models UNCHANGED: the fixes are engine-side; the gguf artifacts are
  byte-identical, so no re-upload is needed. (The log1p accuracy gain benefits any
  user running our engine; a README note could mention it if desired.)
- No new user flags introduced. SSD stayed at 302 GiB free throughout.

## Recommended decision for you
The Tier-1 correctness/quality wins are done and safe to keep. For "60 t/s": it is
not achievable single-stream without (a) a retrain with more MTP heads or (b) a
quality/scope change. If the real goal is *serving throughput*, the highest-value
next project is the #1062 --ple multi-session port (65x aggregate). Tell me which
direction and I'll scope it.
