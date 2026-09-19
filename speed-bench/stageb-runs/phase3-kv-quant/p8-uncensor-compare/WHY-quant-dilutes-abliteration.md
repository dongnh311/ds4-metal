# Why 2-bit quantization dilutes abliteration (measured, this model)

Explains the observed result in `FINDINGS-uncensor-base-vs-orca.md`: after packaging
into the DS4-IQ2 light layout, our OrcaUncensored build differs from the base model
in **only 36 tensors** (all `ssm_out.weight`), and behaves "partially uncensored"
(complies on ~half the mid probes, still refuses phishing/hotwire/meth). This is not
a bug — it is the expected interaction between how abliteration edits weights and how
low-bit quantization rounds them. This document explains the mechanism only; it is
not a recipe to re-apply or strengthen abliteration.

## 1. What abliteration does to the weights
Abliteration ("refusal-direction ablation", Arditi et al. line of work) finds a small
number of directions in the residual stream — the "refusal direction" `r̂` (unit norm)
— by contrasting activations on harmful vs harmless prompts. It then edits many weight
matrices `W` that write into the residual stream so they can no longer project onto
that direction:

    W' = W − r̂ (r̂ᵀ W)        (remove the component that writes toward "refuse")

Two properties matter here:
- **The edit is broad**: it touches every residual-writing projection (attention
  output, MLP/expert down, GDN `ssm_out`, …) across most layers.
- **The per-weight change is tiny**: for each weight element the change is the part of
  that row aligned with a *single* direction out of thousands. In magnitude it is
  usually a small fraction of the weight's own value — often `|ΔW_ij| ≪ |W_ij|`. The
  behavioral effect comes from the *coherent sum* of many tiny, aligned nudges, not
  from any one large edit.

That second property is the whole story: abliteration is a **low-amplitude, spread-out
perturbation**.

## 2. What quantization does to a tiny perturbation
Quantization snaps each weight to the nearest point on a discrete grid. The grid
spacing (quant step `Δq`) depends on the format:

| format (this model) | where used | bits | grid step `Δq` (per block, rough) |
|---|---|---|---|
| Q8_0 | dense projections in the Q8 main; attn q/k/v/o always | 8 | ≈ range/256 — **fine** |
| Q4_K (+imatrix) | less-sensitive dense projections in the imat main | ~4.5 | ≈ range/16 — **moderate** |
| IQ2_XXS | expert gate/up | ~2.06 | ≈ range/4 — **very coarse** |
| Q2_K (pad768) | expert down | ~2.6 | ≈ range/5 — **very coarse** |

A quantizer erases any change smaller than half its step:

    round_q(W + ΔW) == round_q(W)      whenever   |ΔW| < Δq / 2

The abliteration delta `ΔW` from §1 is exactly the kind of small change that loses
this contest. On the **2-bit expert tensors**, `Δq` is large (only ~4–5 code points
across the whole block range), so `|ΔW| < Δq/2` holds for the overwhelming majority of
elements: `base` and `abliterated` weights **quantize to identical bytes**. The
abliteration is rounded away — "swallowed by quantization noise". That is why 1219 of
1255 tensors came out byte-identical between base and OrcaUncensored.

## 3. Why it survived only in `ssm_out`
`ssm_out.weight` is the output projection of the GDN linear-attention blocks (~75% of
layers in this qwen4exp model). It is a **dense** tensor, so in the Q8 build it is kept
at **Q8_0** — the *fine* grid. There, `Δq` is ~64× smaller than on the 2-bit experts,
so a good share of the abliteration delta clears the `Δq/2` threshold and **survives
the round-trip**. Those 36 surviving tensors are the entire measured footprint, and
they are enough to flip behavior on the *easier* probes (lock-picking, profanity) but
not on the ones whose refusal was carried mostly by the (now-erased) expert and other
projections (phishing, hotwire, meth). Hence: **partial** uncensoring.

Summary of the causal chain:

    abliteration = tiny broad perturbation
        └─ on 2-bit experts:  |ΔW| < Δq/2  →  rounded away  →  abliteration erased
        └─ on Q8 ssm_out:     |ΔW| > Δq/2  →  preserved     →  residual uncensor

## 4. Consequences / corollaries (for understanding, not a fix recipe)
- **Precision ∝ preservation.** Any property encoded as a small-amplitude weight
  perturbation (not just abliteration — also fine LoRA-style deltas, subtle style
  fine-tunes) degrades in proportion to how coarse the quantization is on the tensors
  that carry it. Q8 preserves it, 2-bit largely does not.
- **The comparison above was measured at Q8** (both mains). In the imat Q4_K variant,
  `ssm_out` at Q4_K sits between Q8 and 2-bit, so it would preserve *less* of the
  ssm_out delta than the Q8 main — expect equal-or-weaker uncensoring, never stronger.
- **This is intrinsic to lossy low-bit quantization**, not specific to DS4 or to
  abliteration. The same math is why aggressive 2-bit quant can wash out other
  delicate behaviors and why quality-sensitive tensors are commonly kept at higher
  precision.
- **The source model is the ceiling.** `orcarouter/Qwen3.8-Flash-Next-Uncensored`
  (full precision) is as uncensored as this lineage gets; every lossy step downstream
  can only equal or dilute it. The light package cannot exceed the source, and 2-bit
  experts guarantee it falls short of it.

## References / grounding
- Measurement: `FINDINGS-uncensor-base-vs-orca.md`, `uncensor_compare.txt` (this dir).
- Quant layout: IQ2_XXS gate/up + Q2_K-pad768 down experts; dense Q8_0 (Q8 main) or
  selective Q4_K+imatrix (imat main); attn q/k/v/o always Q8_0.
- Technique background (public): refusal-direction ablation, Arditi et al.,
  "Refusal in LLMs is mediated by a single direction".
