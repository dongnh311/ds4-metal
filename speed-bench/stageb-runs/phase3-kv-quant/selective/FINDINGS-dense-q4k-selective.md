# Selective dense Q4_K: keep full-attention q/k/v/o at Q8_0, requant the rest (+~13% decode, better quality)

Follow-on to the full dense-Q4_K win (FINDINGS-dense-q4k-kernel-decode-win.md,
prod 432e610). That variant requantized ALL 271 per-layer dense projections to
Q4_K and measured decode +13.3% at cosine 0.915 vs Q8. This variant tests the
hypothesis that the most quality-sensitive dense tensors -- the full-attention
q/k/v/o projections -- can be kept at Q8_0 to recover quality at near-zero decode
cost, because they are a small share of the per-token dense byte traffic.

## What changed vs the full variant
requant_dense.py SKIP_SUBSTR adds `.attn_q.`, `.attn_k.`, `.attn_v.`,
`.attn_output.` (dotted, so the fused `attn_qkv` stays Q4_K). Kept at Q8_0 now:
output.weight, token_embd, nextn_*, ple_*, hc_*, ffn_down_shexp(640), PLUS the
per-layer full-attention q/k/v/output projections. Still Q4_K: attn_qkv, attn
gate, ssm_out, GDN indexer, shexp gate/up, ffn gate/up.
- Requant: 232 tensors Q8_0->Q4_K (vs 271 full); 105 kept Q8_0 (vs 66 full).
- Dense bytes: 2.23 -> 1.18 GiB requantized region (full was 2.67 -> 1.42 GiB).
- Variant gguf: 40.68 GiB, 234 Q4_K tensors. No binary/kernel change (same
  432e610 Q4_K dense path).

## Result (light Orca, M5 Pro 64 GiB) -- all vs the SAME Q8 reference this session
| metric | Q8_0 | Full Q4_K | Selective Q4_K |
|---|---|---|---|
| decode t/s (MTP-off, ctx4096, clean 2-way A/B) | 31.02 | 36.47* | **35.29 (+13.8%)** |
| decode t/s (3-way A/B mean, noisy round) | 29.6 | 33.8 | 33.2 |
| logit cosine vs Q8 @512 | 1.000 | 0.9147 | **0.9296** |
| logit rms_diff vs Q8 @512 | 0 | 0.761 | **0.681** |
| logit max_abs_diff vs Q8 @512 | 0 | 4.30 | **3.83** |
| argmax @512 | ref | match | **match** |
| top5 overlap @512 | 5/5 | 4/5 | 4/5 |
| coherence (fibonacci, --mtp temp0) | ok | ok | **ok** |
| uncensor smoke vs Q8 (non-det probe) | baseline | -- | **matches Q8 pattern** |

*Full Q4_K decode 36.47 is from the prior session's clean A/B (Q8 32.19 that
session); this session's Q8 baseline ran ~31.0 under heavier background load, so
compare deltas, not absolutes. Selective vs its own-session Q8 = +13.8%.

## Conclusion
Selective Q4_K is a strict improvement over full Q4_K: it KEEPS essentially the
full decode gain (~+13-14%, selective ~= full within machine noise -- the
attention q/k/v/o projections are a small fraction of per-token dense bytes) while
IMPROVING quality on every logit metric (cosine 0.915 -> 0.930, rms 0.761 ->
0.681, max_abs 4.30 -> 3.83). argmax preserved; the single top5 flip (rank 5:
21->19) is IDENTICAL in both variants, so it originates in the dense layers BOTH
requantize (gate/up/ssm), not in attention -- keeping attention at Q8 cannot
recover it, and it costs 1 rank-5 slot on 1 probe.

Both variants are PLAIN Q4_K (no imatrix). Recommended operating point among the
Q4_K variants: selective (better quality, ~same speed). Whether to ship over Q8
is still a quality-vs-speed user call: +13% decode for cosine 0.930 (a notch
below Q8's exactness; well above the -4%/0.882 Q4_0 dead end).

## Receipts (this dir)
selab_q8_*.csv / selab_sel_*.csv (clean 2-way decode A/B),
w3_{q8,full,sel}_*.csv (3-way A/B), cosine_q8_vs_sel_512.txt,
../FINDINGS-dense-q4k-kernel-decode-win.md (full variant + kernel root-cause).
smoke_q8.out / smoke_selective.out (tests/test_orca_uncensored_smoke.py,
non-deterministic default temp 1.0): SAME behavior class -- coding returns code
both, safe clean both, the borderline "stun device" probe trips the broad refusal
regex on caveated-but-compliant text in both (Q8 Y/Y, selective N/Y; selective
refused LESS this run). Bonus: selective gen t/s (29.7-32.2) > Q8 (28.5-29.9) on
all 6 probes, corroborating the decode win outside the benchmark path.
coherence_selective.txt (coherent fibonacci).
