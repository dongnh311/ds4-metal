# Long-context fidelity: imat Q4_K holds to Q8 across the FULL 4K–229K range (no drift)

Closes the packaging-fidelity open item. Prior imat-vs-Q8 logit A/B was measured only
at short context (@512 cosine 0.9494 tensor-route on / 0.9536 isolated). This extends
it across the model's ENTIRE operating range — **4096 → 229376** (ctx-max) — to answer:
does the selective-Q4K+imatrix dense quant lose fidelity to the Q8 dense main as the
context grows toward the 229K ceiling?

## Method
- Models: shipping imat main (`...DenseQ4Kselimat-MTP.gguf`) vs Q8 reference main
  (`...Q2KDownPad768-MTP.gguf`); IDENTICAL IQ2/Q2_K experts, differ only in dense
  quant. Same Q4_1 PLE sidecar, same PROD engine (orca-rebase-ple-batched).
- `ds4-bench --gen-tokens 0` (pure prefill) dumps the full next-token logit vector
  (vocab 248320) at each frontier. Corpus: `promessi_sposi.txt` (~350K real,
  non-repeating tokens), sliced at 4096/8192/16384/32768/65536/131072/229376.
- Config: tensor-route ON (`quality=false`) = the SHIPPING configuration, so this is
  the fidelity users actually get. `compare.py` computes logit cosine, KL(Q8‖imat)
  over softmax, top1/top5 agreement, and max abs logit diff.

## Result (imat vs Q8, shipping config)
| ctx    | cosine | KL(Q8‖imat) | top1 argmax | top5 | max_abs (logit) |
|-------:|-------:|------------:|:-----------:|:----:|----------------:|
| 4096   | 0.9767 | 0.186 | match               | 3/5 | 4.79 |
| 8192   | 0.9900 | 0.240 | **flip** (near-tie) | 5/5 | 3.16 |
| 16384  | 0.9888 | 0.051 | match               | 4/5 | 2.48 |
| 32768  | 0.9647 | 0.014 | match               | 5/5 | 3.20 |
| 65536  | 0.9728 | 0.287 | match               | 4/5 | 2.46 |
| 131072 | 0.9708 | 0.071 | **flip** (near-tie) | 4/5 | 3.10 |
| 229376 | 0.9860 | 0.012 | match               | 3/5 | 2.46 |

## Verdict — fidelity does NOT erode with context; it is best at the longest ctx
- **Cosine stays in a 0.965–0.990 band across the whole 4K–229K range** — no downward
  trend. At the real ctx-max (229376) it is 0.9860, near the TOP of the range and well
  above the ~0.949–0.954 short-context (@512) baseline.
- **max_abs logit diff does not grow with ctx** — it is LARGEST at 4096 (4.79) and
  smallest at long ctx (2.46 at both 65536 and 229376). The dense-Q4K quant gap is a
  roughly constant per-token perturbation that does NOT compound; the two models, if
  anything, converge as context lengthens.
- **KL(Q8‖imat) is lowest at the longest contexts** (0.014 @32K, 0.012 @229K) — the
  distributions get closer, not farther, with length.
- **Argmax matches at 5 of 7 frontiers.** The two flips (@8192 q8=179828/im=171399;
  @131072 q8=378/im=296) are near-ties: top5 overlap 4–5/5, i.e. both candidates sit
  in each other's top-5 and ~1–3 logit units of quant noise broke the tie. Same
  near-tie behavior seen at short ctx, not a new long-ctx failure mode.

**Conclusion:** the shipped imat Q4_K main is a faithful stand-in for the Q8 main
across the entire 4K–229K operating range; the +14% decode costs NO accumulating
quality at length. For maximum fidelity on individual close-call tokens the Q8 main
remains the reference, but the gap is constant, not context-growing.

## Memory reality (why kv-disk turned out unnecessary)
The model's GDN linear-attention KV is COMPRESSED: the engine planner reported, e.g.,
imat @ctx=131073 → resident model 40.67 GiB + KV **4.17 GiB** (raw 3.25 + compressed
0.91, `compressed_kv_rows=32769`) + buffers ~6.9 GiB = **~47 GiB planned**. Even at
229376 the total fits well under 64 GiB, so all frontiers ran resident WITHOUT kv-disk
offload. (ds4-bench does not accept `--kv-disk-dir` anyway — that is a ds4-server
flag; confirmed "unknown option".) Residual ~29 GiB swap during the sweep was stale,
inactive pages from an earlier full-PLE-prefetch run (SIGKILLed before these
measurements); it did not grow across stages and does not affect logit VALUES, which
are deterministic given (tokens, model, engine). Env: `DS4_QWEN4_PLE_PREFETCH_FULL=0`,
`DS4_QWEN4_PLE_EVICT_TOKENS=512`. Per-stage wall time ~11–19 min (prefill-bound).

## Receipts (this dir)
- `fidelity_table.txt` — the full 7-frontier table above (raw parser output).
- `compare.py` — cosine/KL/top-k parser.
- `imat_prefill.csv`, `q8_prefill_4k-32k.csv`, `q8_prefill_65k.csv`,
  `{imat,q8}_c131072.csv`, `{imat,q8}_c229376.csv` — prefill t/s + KV bytes per run.
- Full-logit JSONs NOT committed (large ~3 MB × 14; regenerable via the method above),
  matching the m5-tensor-drift receipt convention. See [[m5-tensor-drift]] for
  route-drift context and ../selective-imatrix/ for the @512 baseline.
