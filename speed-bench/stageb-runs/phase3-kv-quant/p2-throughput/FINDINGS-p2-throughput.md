# Phase 2 — throughput campaign to 60 t/s single-stream --mtp: engine levers exhausted

Baseline (Phase 0): --mtp 42.84 t/s, accept 67.9%; target 60 (+40%). All three
planned engine levers were investigated; NONE opens a path to 60 on this model.

## 2a. DS4_QWEN4_FLUSH_LAYER sweep (free env) — FLAT (not a lever)
Swept 0,1,2,3,4,6,8,12 (default 2), --mtp, n=400, temp0/seed1, 2 runs each:
| FL | 0 | 1 | 2(def) | 3 | 4 | 6 | 8 | 12 |
|--|--|--|--|--|--|--|--|--|
| gen t/s | 42.1/42.6 | 42.1/42.5 | 42.5/42.7 | 42.9/42.9 | 42.8/42.8 | 42.7/42.6 | 42.8/42.7 | 43.0/42.2 |
Range 42.1-43.0, all within ~2% noise; acceptance constant 66.9%. FL=3 a hair
best but not significant. The queue-without-waiting submission is already
saturated; decode is not command-submission bound. **Not a path to 60.**

## 2b. MTP tuning (--mtp-draft / --mtp-margin / exact-sampling) — INAPPLICABLE
Code inspection: for qwen4, `ds4_engine_mtp_draft_tokens` returns a HARDCODED 2
(ds4.c:58907) and the model has DS4_N_NEXTN_PREDICT=1 (one nextn head), so
`--mtp-draft` is ignored and the MTP block is at most [seed + 1 draft].
Greedy accept is strict exact-match (`sample_argmax(rows)==d`, ds4.c:70227) with
NO margin path (mtp_margin is only used by an unrelated CUDA greedy split-kv
kernel), so `--mtp-margin` does nothing here either. Acceptance 66.9% is the
model's single-head MTP accuracy — architecturally fixed. The only "knob" that
raises it, DS4_QWEN4_SPEC_FORCE_ACCEPT, commits UNVERIFIED drafts (breaks
greedy-exactness) → rejected. The MTP path already runs GPU argmax
(DS4_QWEN4_MTP_GPU_ARGMAX default on), state swap-on-reject
(DS4_QWEN4_MTP_SWAP_RESTORE default on), and next-draft batch-prep
(DS4_QWEN4_NO_MTP_BATCH unset) — all optimizations already enabled. **Not a lever.**

## 2c. Port #1056/#1047 freestanding kernels — NO VIABLE COMMIT (deep git analysis)
A dedicated analysis of the antirez/upstream row-reuse + predictor-state bundle
(666f9c7, 891c005, 548cdaf, e2cba0b, a504136 family, 7b5498d, 6c01e38, d7ade9f,
a154af7, ...) concluded NONE is portable to orca-bugfixes for single-stream --mtp:
- The genuine single-stream qwen4/M5 MTP decode kernels (Sep 6-8 wave) are ALREADY
  MERGED into orca-bugfixes (verified `git merge-base --is-ancestor`).
- 666f9c7 "reuse Q8 weights across 3 verify rows" is a DSpark/DeepSeek kernel AND
  is functionally superseded by orca-bugfixes's existing `ds4_gpu_mv_ext_r1ptg`
  (returns r1ptg=3 for the 3-row verify) — plus its token_pair/triple path is
  gated `pre_m5_apple_silicon` (M1-M4 only; dead on M5 Pro).
- 891c005 (+10% single-stream) FIXES a regression (f16_batch mis-catch) that
  orca-bugfixes never had (no f16_batch logic in its qwen4_gemv_rows).
- The a504136 family (Speculate over a batch of sessions) is multi-session infra
  ABSENT from orca-bugfixes (0 occurrences of qwen4_batch_mtp_drafts /
  ds4_sessions_eval_batch_speculative_argmax); its wins are all C=16, C=1 = 1.00x.
- The rest are TP-only / DeepSeek-MLA / GLM-prefill / MXFP4-format / prefill-only.
**No un-applied single-stream kernel win exists to port.**

## Conclusion: 60 t/s single-stream --mtp is NOT reachable on this model/hardware
Decode is dominated by per-token overheads that the available levers do not touch:
46-layer kernel dispatch, the GDN linear-attention recurrent state (inherently
sequential per token), and the CPU-side PLE n-gram gather (demand-paged, 30 GiB
can't be RAM-resident beside the 40.67 GiB model on 64 GiB). FLUSH_LAYER flat +
weight-quant already at the 2-bit floor (experts) confirms decode is NOT pure
weight-BW bound, so further quant won't help either (full-Q4K measured +13% =
same as selective; the Q8 attn is a small byte fraction). MTP already gives its
architectural max (+17%, 1 head, exact-match). The only paths to 60 are OUT of
the current single-stream/no-retrain/no-new-flag scope:
  (a) more nextn/MTP heads → more accepted tokens/cycle (model RETRAIN);
  (b) a smaller/shallower model (different model);
  (c) reduce PLE gather cost / overlap it with GPU (engine change, risk).
For AGGREGATE throughput (not single-stream) the #1062 multi-session batching
(65x @16 streams) via a --ple forward-port is the real win — the DEFERRED project
already noted, and it does not improve single-stream latency.
