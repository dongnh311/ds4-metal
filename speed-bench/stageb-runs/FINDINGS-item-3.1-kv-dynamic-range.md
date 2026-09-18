# Item 3.1 (spec §25 Phase 3) — K/V dynamic range (FP8 feasibility prep)

## What & why
Phase 3's first step ("K/V dynamic range") measures the numeric range of the
attention K and V that the paged KV cache stores, to choose an FP8 format before
implementing lossy FP8. Instrumented the now-working paged roundtrip
(qwen4_paged_kv_roundtrip) to accumulate, as a PURE READ of the f16 K/V (no
output change → gate #KV stays bit-identical), per-element absmax/RMS/[min,max]
and the mean per-(token,head) absmax. Printed at qwen4 session teardown. Reuses
DS4_QWEN4_KV_PAGED (whitelisted env) — NO new flag.

FP8 formats: E4M3 = 4 exp / 3 mantissa bits, max ±448, finer near zero; E5M2 =
5 exp / 2 mantissa, max ±57344, coarser. KV quant normally uses PER-TOKEN
scaling, so what matters is (a) whether per-token absmax fits the format after
scaling and (b) mantissa precision vs the value spread.

## Measured (ctx 8192, single prompt = promessi_sposi, all full-attn layers)
K (50,356,224 elems): absmax=15.7266  mean|tok|=4.8938  rms=1.4673  range=[-15.7266, 12.9844]
V (50,356,224 elems): absmax=34.3438  mean|tok|=1.9000  rms=0.6031  range=[-34.3438, 22.5156]

## Reading
- Both K (|max| 15.7) and V (|max| 34.3) sit FAR inside E4M3's +/-448 range, with
  ~13x-28x headroom. E5M2's larger range (+/-57344) is unnecessary and would waste
  a mantissa bit. **Recommend E4M3.**
- Per-token(-head) scaling IS warranted: global absmax >> mean per-(token,head)
  absmax — K 15.7 vs 4.89 (~3.2x), V 34.3 vs 1.90 (~18x). V especially has a wide
  per-token spread, so a single global scale would waste E4M3's precision on most
  tokens; a per-token(-head) scale maps each vector near +/-448 for max mantissa use.
- Precision expectation: E4M3 has 3 mantissa bits (~2^-3 = 12.5% relative step at a
  given exponent). With per-token scaling the RMS quantization error is well below
  that, but whether it degrades logits/eval is the OPEN question — that is exactly
  what the FP8 quality matrix must answer before shipping.
- Caveat: single-prompt sample (promessi_sposi) at ctx 8192; a couple more prompts
  would confirm the tails, but the fit (well inside E4M3) is unambiguous.

## Status / next
This is measurement only — no FP8 stored, no output change. Implementing FP8 KV
(the memory win: ~half the KV bytes) is the next Phase-3 step and it is LOSSY
(ground rule 6: quality-matrix gate, NOT bit-identical) AND needs a new enable
toggle (ground rule 7: new flag = stop + ask). So FP8 itself awaits the user's
go-ahead + a quality-matrix review. This range data is the input to that decision.
