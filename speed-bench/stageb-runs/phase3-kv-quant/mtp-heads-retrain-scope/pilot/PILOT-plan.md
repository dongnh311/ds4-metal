# Pilot plan: prove-or-kill the extra-MTP-head speedup with bounded GPU spend

Turnkey plan the user green-lit ("đi hướng pilot"). Goal: convert the open-ended GPU
retrain bet (odds ~40–60% to clear 60 t/s, see FINDINGS-accept-ceiling-derisk.md) into
a small, measured go/no-go BEFORE committing to a full retrain. Everything here is
prepared in-house; the training steps require an external GPU + the HF base (neither
runs on the M5 Pro).

## Ground truth (measured from the shipped model metadata, this session)
- Base (full precision, trainable): **`Qwen/Qwen3.8-Flash-Next`** on HuggingFace
  (Ivan converted the DS4 GGUF from HF snapshot `de4b8e4d43...`). Download for training.
- Architecture: **n_layer ≈ 49** (blk.0–48). The MTP head is **blk.48** and it is a
  FULL MoE transformer block — attn q/k/v/o + q/k norms, 10-expert MoE (ffn_*_exps) +
  shared expert (ffn_*_shexp) + gate, hyper-connection layers (hc_attn_*, hc_ffn_*) —
  PLUS the nextn adapter: `enorm`, `hnorm` (norm the token-embedding and the hidden
  state), `eh_proj` (13.9 MB Q8_0 ≈ hidden≈2560 combine of [emb, hidden]),
  `hc_head_norm/up/down`. Only **ONE** nextn module exists (Qwen's shipped MTP).
- Implication: a NATIVE extra head is a whole MoE layer's worth of parameters — heavy
  to train and to hold, NOT a lightweight Medusa FFN head. This drives the two-tier plan.

## Why deeper drafting is the target (recap)
Decode is verify-forward-bound → throughput ≈ accepted-tokens/cycle. Measured: MTP-off
36.5 → depth-2 41.7 t/s; the existing head re-run for a 3rd token is NET-NEGATIVE
(depth-3 = 37.6). First-draft accept ~65%. So the current head is tapped out; only a
head TRAINED for t+2 (and t+3) can add accepted tokens. The pilot tests whether such a
head can, on THIS 2-bit MoE.

## Two-tier pilot (cheap signal first)

### Tier-A — predictability probe (cheapest; ~a few GPU-hours, single GPU)
Question it answers: *from the base's last hidden state at position t, how accurately can
ANY head predict token t+2 (and t+3) on real text?* This is the ceiling every drafter
lives under; if it is low, no native head will reach 60.
1. **Precompute (inference only, no training):** run the FROZEN base over a corpus
   (~2–5M tokens), dumping per-position (last_hidden_t, embed_{t+1}, id_{t+1}, id_{t+2},
   id_{t+3}). Inference-only → fits int8/fp8 base on one big GPU; the dump is the only
   base pass. (Cannot be done locally: we hold only the 2-bit GGUF, which does not
   expose hidden states; needs the HF base.)
2. **Fit small heads on the dump (CPU/GPU, minutes):** train 2 lightweight heads
   (RMSNorm + 2-layer MLP over [hidden_t, emb of the previously-drafted token], tied to
   the base's output embedding) to predict id_{t+2} and id_{t+3}. Report top-1 accuracy.
3. **Gate:** t+2 top-1 accuracy
   - **≥ ~70%** → strong: a native (heavier) head will beat this → proceed to Tier-B.
   - **~55–70%** → marginal: native head *might* clear 60; proceed to Tier-B with eyes open.
   - **< ~55%** → the tokens are not predictable enough two-ahead on this quant → STOP;
     ~43 t/s is the ceiling, keep multi-session ~2×. (Saves the full-retrain spend.)
   Cost: dominated by the single precompute pass; the head fit is cheap. Est. small.

### Tier-B — one native nextn head (only if Tier-A passes; heavier GPU)
Train ONE additional native MTP module mirroring blk.48 (full MoE block + nextn adapter),
**warm-started from the existing blk.48 weights** (fast convergence), base frozen,
next-next-token CE (teacher-forced on real text). Then:
1. Convert the trained head to the DS4 GGUF nextn layout via `gguf-tools/
   qwen4_exp_convert.py` (the same safetensors→GGUF path Ivan used); quantize it into
   the light packaging at a precision that survives 2-bit (keep the adapter/norms high,
   cf [[WHY-quant-dilutes-abliteration]]).
2. Engine: extend nextn drafting to run 2 trained heads sequentially (incremental on the
   shipped adaptive-depth machinery `snap_after_second`); keep the verify-exact gate.
3. **Measure with the exact depth sweep from FINDINGS-accept-ceiling-derisk.md.** Gate:
   depth-3 with the trained head must be NET-POSITIVE (t/s > depth-2 41.7) and trend
   toward ~2.3–2.5 accepted/cycle. If it clears ~55–60 t/s → commit to the full retrain
   (more heads / more data / EAGLE). If not → stop at the measured gain.

## Compute / cost (order-of-magnitude; user to confirm against live GPU prices)
- Tier-A: 1 inference pass of the base over ~2–5M tokens + tiny head fit. A single
  80 GB GPU (int8/fp8 base) for a few hours. Small rental.
- Tier-B: training a full MoE block (frozen base) with warm start — larger GPU/VRAM and
  more hours; still bounded (one head, small corpus). Mid rental.
- The base download (~full-precision Qwen3.8-Flash-Next) is a one-time large transfer.

## What is prepared in-house (this dir)
- This plan.
- `train_pilot_probe.py` — Tier-A skeleton: precompute-dump loader + the two small heads
  + top-1 accuracy report + the go/no-go print. Marked where the base + GPU are needed;
  structurally follows the real blk.48 nextn layout above. UNTESTED here (no GPU/base).
- Reuse: `gguf-tools/qwen4_exp_convert.py` (convert), the depth-sweep recipe (measure).

## Decision flow
Secure GPU + base → run Tier-A → gate → (pass) run Tier-B → gate → (pass) full retrain.
Each arrow is a measured stop point, so spend scales with evidence.
