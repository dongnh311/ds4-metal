# Long-context fidelity: imat Q4_K holds to Q8 at 4K–64K (no drift accumulation)

Closes the open item from the packaging-fidelity work: prior imat-vs-Q8 logit A/B
was measured only at short context (@512 cosine 0.9494 with tensor-route on, 0.9536
isolated). This extends it to **4096 → 65536** — up to 16× the earlier evidence — to
answer: does the selective-Q4K+imatrix dense quant lose fidelity to the Q8 dense main
as context grows?

## Method
- Models: shipping imat main (`...DenseQ4Kselimat-MTP.gguf`) vs Q8 reference main
  (`...Q2KDownPad768-MTP.gguf`), IDENTICAL IQ2/Q2_K experts, differ only in dense
  quant. Same Q4_1 PLE sidecar, same PROD engine (orca-rebase-ple-batched).
- `ds4-bench --gen-tokens 0` (pure prefill) dumps the full next-token logit vector
  (vocab 248320) at each frontier. Corpus: `promessi_sposi.txt` (~350K real tokens,
  non-repeating prose), sliced at 4096/8192/16384/32768/65536.
- Config: tensor-route ON (`quality=false` in dumps) = the SHIPPING configuration, so
  this is the fidelity users actually get. `compare.py` computes logit cosine,
  KL(Q8‖imat) over softmax, top1/top5 agreement, max abs logit diff.

## Result (imat vs Q8, both shipping config)
| ctx   | cosine | KL(Q8‖imat) | top1 argmax | top5 | max_abs (logit) |
|------:|-------:|------------:|:-----------:|:----:|----------------:|
| 4096  | 0.9767 | 0.186 | match         | 3/5 | 4.79 |
| 8192  | 0.9900 | 0.240 | **flip** (near-tie) | 5/5 | 3.16 |
| 16384 | 0.9888 | 0.051 | match         | 4/5 | 2.48 |
| 32768 | 0.9647 | 0.014 | match         | 5/5 | 3.20 |
| 65536 | 0.9728 | 0.287 | match         | 4/5 | 2.46 |

## Verdict — fidelity does NOT erode with context length
- **Cosine stays in a 0.965–0.990 band across the whole 4K–64K range** — no downward
  trend, and equal-to-better than the ~0.949–0.954 short-context (@512) baseline. The
  dense-Q4K quantization gap to Q8 is a roughly constant per-token perturbation; it
  does not compound as the context grows.
- **max_abs logit diff does not grow with ctx** (largest at 4096 = 4.79, smallest at
  65536 = 2.46). No runaway drift — consistent with the M5 tensor-route note that any
  route drift is bounded, not accumulating, on this hardware.
- **Argmax matches at 4 of 5 frontiers.** The single flip (@8192, q8=179828 vs
  imat=171399) is a near-tie: top5 overlap is 5/5, i.e. both candidates sit in each
  other's top-5 and ~1–3 logit units of quant noise broke the tie. This is the same
  near-tie flip behavior seen at short ctx, not a new long-ctx failure mode.
- KL(Q8‖imat) bounces 0.014–0.287 with no upward trend — distributions stay close.

**Conclusion:** the shipped imat Q4_K main is a faithful long-context stand-in for the
Q8 main to at least 64K; the +14% decode does not cost accumulating quality at length.
For maximum fidelity on close-call tokens the Q8 main remains the reference, but the
gap is constant, not context-growing.

## Honesty note on the harness (not the model)
The bench itself was NOT zero-swap at 32K+: the Q8 model (44.8 GiB) plus the PLE
sidecar demand-paged working set (which grows as more distinct n-gram rows are touched
over a long prefill) pushed ~16 GiB to swap on this 64 GiB box even with
`DS4_QWEN4_PLE_PREFETCH_FULL=0` and `DS4_QWEN4_PLE_EVICT_TOKENS=512`. This is a
**bench-harness limitation** (ds4-bench has no KV-disk offload), NOT a model/runtime
regression: the ai-gateway serves long context zero-swap via `--kv-disk-dir` +
`--kv-cache-cold-max-tokens` + tuned PLE eviction. Swap affects speed only; the logit
VALUES compared here are deterministic given (tokens, model, engine), so the fidelity
numbers are unaffected. A prior run that swap-thrashed for 38 min was a stuck
full-PLE-prefetch process (76 GiB footprint), SIGKILLed before these measurements.

## Receipts (this dir)
- `fidelity_table.txt` — the table above (raw parser output).
- `compare.py` — cosine/KL/top-k parser (reads the per-frontier logit JSONs).
- `imat_prefill.csv`, `q8_prefill_4k-32k.csv`, `q8_prefill_65k.csv` — prefill t/s +
  kv bytes per frontier (also show where swap slowed prefill: 361→140 tps at 32K).
- Full-logit JSONs NOT committed (large ~3 MB × 10; regenerable via the method above),
  matching the m5-tensor-drift receipt convention. See [[m5-tensor-drift]] for the
  route-drift context and the @512 baseline in ../selective-imatrix/.
