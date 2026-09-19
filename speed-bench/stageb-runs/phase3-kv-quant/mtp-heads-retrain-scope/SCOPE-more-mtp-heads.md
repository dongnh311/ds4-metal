# Scope: raising single-stream decode past ~43 t/s via more MTP heads (retrain path)

Design/feasibility scope ONLY — no training is done here. Answers "what would it take,
what gain, at what cost" for the one lever left after the engine hit its single-stream
ceiling (see [[tier1-fixes-and-60tps-ceiling]] and REPORT-60tps-campaign-*). This is a
performance path; it does not touch model behavior/uncensoring.

## 1. Why this is the only remaining lever
Measured ceiling on M5 Pro 64 GiB: single-stream decode is **92% GPU-compute-bound**;
every in-engine lever (FLUSH_LAYER, expert-cut to k=4, #1056 row-reuse) tops out
~52 t/s, and the honest baseline is **~42.8 t/s MTP-off-equiv / ~43 t/s `--mtp`**.
The `--mtp` speedup is +17% at **~67% draft acceptance** with **one** MTP head
(`n_nextn_predict = 1`, GGUF key `qwen4exp.nextn_predict_layers`).

Speculative decode throughput ≈ `accepted_tokens_per_cycle / verify_forward_cost`.
Because decode is bandwidth/compute-bound by ONE target forward, a verify pass costs
~1 forward regardless of how many tokens it checks — so raising accepted-tokens/cycle
multiplies throughput almost linearly until draft+overhead dominates. Today: 1 head →
~1.67 accepted/cycle → +17%. To reach 60 t/s needs ~2.3–2.5 accepted/cycle; to reach
~75 needs ~3. That requires better/more DRAFT prediction, which is a **trained**
artifact — hence a (partial) retrain.

## 2. Engine readiness (what already exists)
- The nextn/MTP block is the model's **final layer(s)**: `il + DS4_N_NEXTN_PREDICT >=
  DS4_N_LAYER` marks nextn layers (ds4.c:1026, 5423). The whole layer-count math keys
  off `DS4_N_NEXTN_PREDICT`, so bumping it to 2–3 is structurally anticipated.
- Adaptive deeper-MTP (Task A, shipped): the engine can already draft up to 2 tokens
  from the SINGLE head via a 3-token verify (`snap_after_second`, `snap2_lin_state`,
  ds4.c:57726+) — verify-exact, +5–6%. So multi-token drafting and recurrent-state
  snapshot/rewind for GDN linear-attention already work.
- **Gap**: running 2–3 DISTINCT trained heads sequentially per draft cycle (vs
  re-running one head) is incremental on top of the adaptive-depth machinery, not a
  rewrite. Tree-verify (Medusa) and an external-feature draft loop (EAGLE) are NOT in
  the qwen4 path and would be larger engine additions.

## 3. Options (accept ↑, engine work, train cost)
| option | how | accept gain (lit.) | engine work | train cost |
|---|---|---|---|---|
| **A. More native MTP heads** (DeepSeek-V3 style, depth 2–3) | train 1–2 extra nextn blocks on the frozen base; `DS4_N_NEXTN_PREDICT=2/3` | ~2.2–2.8 acc/cycle | **low** (reuses nextn path + adaptive depth) | medium |
| B. EAGLE-2/3 | feature-level autoregressive draft head using target hidden states + token embeds | ~2.5–3.5× throughput (SOTA) | **high** (new draft loop) | medium |
| C. Medusa-2 | K independent FFN heads on last hidden state + tree attention | ~2× | **high** (tree verify) | low |
| D. External small draft model | tiny vocab-compatible Qwen drafts, target verifies | ~1.8–2.5× | medium (external-drafter spec in qwen4) | none (reuse a released small model) |

Recommendation: **Option A** first — it stacks directly on the shipped verify-exact
nextn + adaptive-depth machinery (lowest engine risk, keeps byte-exact greedy), and 2
good heads plausibly clear 60 t/s. Option B is the higher ceiling if bigger engine
work is acceptable. Option D avoids any target retrain but needs a suitable small
drafter and still needs engine spec-decode-with-external-drafter for qwen4.

## 4. Feasibility & cost (the honest gate)
- **Training does NOT run on the user's M5 Pro 64 GiB.** Head training needs the base
  in ≥bf16 with activations + optimizer state for backprop through the head; even with
  the base frozen and gradient checkpointing, the working set for this MoE far exceeds
  64 GiB unified memory. It needs **datacenter GPUs** (e.g. multi-A100/H100 80 GB, or
  a large single GPU with int8 base + checkpointing). This is the blocking dependency.
- **Data**: a self-distillation corpus (base model's own next-token targets over a
  general corpus). Native-MTP/Medusa heads are light — typically ~1e8–1e9 tokens;
  EAGLE wants more. Reuse the existing imatrix calibration corpus + general text.
- **Base access**: training targets the FULL-precision base (`orcarouter/
  Qwen3.8-Flash-Next-Uncensored` or the Qwen base), not the IQ2 light GGUF. The new
  heads are then quantized into the light packaging with the existing requant pipeline
  (heads likely kept at higher precision, like dense projections, so 2-bit doesn't
  erase them — cf [[WHY-quant-dilutes-abliteration]] on small-delta erasure).
- **Verify-exactness**: any new head MUST preserve byte-identical greedy verify (the
  current MTP is verify-exact). This is a hard gate on the engine port, not the head.

## 5. Expected outcome (honest ranges)
- Depth-3 native MTP (Option A), acceptance ~2.3–2.8/cycle → decode ≈ 43 ×
  (2.5/1.67) ≈ **~64 t/s**, minus draft/overhead → realistic **~58–66 t/s**. This is
  the first credible route through the 60 t/s wall.
- EAGLE-2 could push higher (~70–80) but at higher engine cost and less certainty on a
  BW-bound MoE.
- All figures are single-stream latency; multi-session aggregate (the shipped #1062
  path) is an orthogonal, already-available ~2× and does not need this.

## 6. Effort / sequence if greenlit
1. Secure GPU compute (external) — **the gate**; without it, stop here.
2. Engine: extend nextn drafting to N trained heads (build on adaptive-depth A);
   keep verify-exact gate green. (est. small–moderate, in-house.)
3. Train 1–2 extra nextn heads on the frozen base (self-distill). (est. medium, GPU.)
4. Requant heads into the light GGUF; re-run shipped-quality + verify-exact gates;
   re-bench single-stream `--mtp`; upload to HF per the model-change rule.

## 7. Decision needed from the user
Do you have / will you rent **GPU training compute**? If yes → Option A is the plan and
step 2 (engine, in-house on the Mac) can start in parallel. If no → this path is
blocked and ~43 t/s single-stream stands as the M5-Pro ceiling; the multi-session ~2×
remains the only speed lever available locally.
