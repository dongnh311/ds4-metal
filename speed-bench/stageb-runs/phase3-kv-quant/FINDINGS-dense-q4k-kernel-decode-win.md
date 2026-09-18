# Breaking the decode ceiling: wire Qwen dense to the shipped fast Q4_K GEMV (+13%)

Follow-on to the KV/quant experiment (which showed KV quant can't break the ceiling
and a naive dense Q4_0 was -4%). Here the ceiling IS broken for the light Orca model
by quantizing the dense projections to Q4_K and routing them to DS4's existing
decode-grade dense Q4_K kernel.

## Root cause the earlier Q4_0 attempt missed (kernel efficiency, not bytes)
Decode timelines at ctx 4096 (per-token dense-GEMV effective bandwidth):
| dense kernel | bytes/token | eff. bandwidth |
|---|---|---|
| Q8_0 `kernel_mul_mv_q8_0_f32` (specialized) | 3.00 GiB | **185 GiB/s** |
| Q4_0 `kernel_mul_mv_ext_q4_0_f32_r1_1` (generic) | 1.59 GiB | **93 GiB/s** |
Q4_0 read half the bytes at half the efficiency -> no decode win. The fix is a
decode-grade 4-bit kernel, not a different byte count.

## The lever: DS4 already ships kernel_mul_mv_q4_K_dense_f32
The GLM/DeepSeek dense-quant path uses a fast classic Q4_K matvec
(metal/moe.metal:3372, ~530-650 GB/s on M5). The Qwen path never wired to it. Three
one-line ds4.c edits (prod commit 432e610) route Qwen dense Q4_K into it:
tensor_type_is_qwen4_dense +Q4_K, qwen4_gemv_rows case DS4_TENSOR_Q4_K, and
qwen4_graph_dense_ok +Q4_K. No new Metal code; no new user flag.

## Requant: gguf-tools/requant_dense.py (Q8_0 -> Q4_K)
271 per-layer dense tensors requantized (attn q/k/v/o, attn_qkv/gate, ssm_out/lin_*,
indexer q/k, ffn_gate/up_shexp). KEPT Q8_0 (special forward paths / not 256-aligned):
output.weight, token_embd, nextn_*, ple_*, hc_*, ffn_down_shexp (ne0=640). Dense
2.67 -> 1.42 GiB. Variant 40.47 GiB, 273 Q4_K tensors.

## Result (light Orca, M5 Pro 64 GiB)
| metric | Q8_0 baseline | Dense Q4_K | delta |
|---|---|---|---|
| decode t/s (MTP-off, ctx4096, 3x mean) | 32.19 | **36.47** | **+13.3%** |
| decode t/s (--mtp, single) | ~40.2 | **42.56** | ~+6% |
| prefill t/s | 509 | 528 | +3.7% |
| logit cosine vs Q8 @512 | 1.000 | **0.915** | argmax preserved, top5 4/5 |
| ds4-eval core (6-case subset, cap 1024) | 3/6 (others cap-incomplete) | **3/6, same cases, same answers** | no accuracy regression |
| uncensor probes (lockpicking, adult) | comply | **comply** | preserved |

**+13.3% decode is the first quant change that actually raises decode** (Q4_0 was
-4%). Quality is acceptable (argmax preserved, eval matches Q8 on completed cases,
coherent, uncensored) though not free (cosine 0.915 < FP8-KV's 0.985 -- the dense
backbone at 4.5-bit is inherently lossier; PLAIN Q4_K, no imatrix).

## Caveats / next
- Quality was checked via cosine + a 6-case eval subset (token-capped). A full
  ds4-eval core + longer budgets would be more definitive before shipping.
- **imatrix Q4_K** would raise quality further (requant_dense.py uses plain Q4_K).
- Only 271/337 dense converted (special paths excluded). Wiring the excluded
  fused/special paths (hc, nextn, GDN lin_*) to Q4_K could add a few more % but
  needs per-path kernel work.
- The +13% is at ctx 4096; the dense share of decode is ctx-independent (weights,
  not KV) so the gain should hold across context.

## Receipts
q4kab_q8_*.csv / q4kab_q4k_*.csv (decode A/B), decode_kernel_timeline_q{8,40}_4k.txt
(185 vs 93 GiB/s), eval_q8_vs_q4k_subset.txt, gguf-tools/requant_dense.py,
docs/patches/orca-bugfixes/0001-Phase-3-wire-Qwen-dense-Q4_K-*.patch (prod 432e610).
