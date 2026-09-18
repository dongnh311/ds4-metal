# imatrix-weighted selective Q4_K: quality up (cosine 0.930 -> 0.949), decode unchanged (+15%), ZERO C work

Cheap-probe result for the user-chosen "unblock imatrix-Q4_K for dense" direction.
Instead of building a from-scratch dense imatrix collector into the qwen4 Metal
graph (which exploration showed is a large/risky C subsystem — the existing
collector is expert-only and wired to the DEEPSEEK4 graph, and neither Ivan's
upstream nor antirez/ds4 have a qwen4/dense imatrix to port), we reused the
ALREADY-PUBLISHED llama.cpp importance matrix and fed it to DS4's existing
weighted K-quant kernel via the Python requant. It works, and it makes the dense
imatrix collector unnecessary for shipping.

## Method (Python only; no engine change, no rebuild)
- Source imatrix: `unsloth/Qwen3.8-Flash-Next-GGUF` @ `c8b5954` file
  `imatrix_unsloth.gguf_file` (553 MiB GGUF, general.type=imatrix). It is a
  standard llama.cpp imatrix over the SAME qwen4exp tensor decomposition DS4 uses
  (fused `attn_qkv`, `ssm_out`, `indexer.q/k_proj`, `hc_*`, shexp, experts) — so
  the dense entries map to DS4 tensor names 1:1.
- `gguf-tools/requant_dense.py --imatrix <file>`: new `load_gguf_imatrix()` reads
  each `<name>.in_sum2` (F32[ncols]) / `<name>.counts`, importance = in_sum2/count,
  and passes it as the per-column weight into `GGMLQuantizer.encode(f,"Q4_K",
  imatrix=...)` (already wired to `ds4q_quantize_chunk`). Column alignment checked:
  **228/232 selective targets weighted, exact length match; 4 fall back to plain
  Q4_K (all blk.48, the last trunk layer, absent from the unsloth calibration).**
- Same selective policy as before: full-attention q/k/v/output stay Q8_0.

## Result (light Orca, M5 Pro 64 GiB) — all vs the SAME Q8 reference this session
| variant | logit cosine vs Q8 @512 | rms | max_abs | decode t/s (MTP-off, 3x) | decode vs Q8 |
|---|---|---|---|---|---|
| Q8_0 (exact) | 1.0000 | 0 | 0 | 30.29 | ref |
| full Q4_K (plain) | 0.9147 | 0.761 | 4.30 | ~34 | +13% |
| selective Q4_K (plain) | 0.9296 | 0.681 | 3.83 | 35.29 | +13.8% |
| **selective Q4_K + imatrix** | **0.9494** | **0.589** | **2.88** | **34.83** | **+15.0%** |

- imatrix closes ~27% of the plain-selective quality gap to Q8 (0.070 -> 0.051)
  and ~40% of the full-Q4K gap, for FREE: decode is unchanged (encode-only; same
  234 Q4_K tensors, same bytes/kernel — the +15% vs +13.8% is machine noise, same
  operating point), argmax preserved, top5 4/5 (the lone rank-5 flip is the same
  21->19 seen in every Q4_K variant, from gate/up + the 4 blk.48 fallbacks).
- Coherence: clean fibonacci with type hints + docstring (--mtp temp0, gen 44.7 t/s).
- Uncensor smoke (smoke_imat.out): matches Q8's pattern (harmful Y/Y, coding=code,
  safe clean) — same behavior class; gen t/s ~32 > Q8 ~29 across probes.

## Why this is the answer (no collector needed to ship)
The dense imatrix collector for the qwen4 graph is real future work IF a
DS4-self-calibrated imatrix is wanted, but the published llama.cpp imatrix already
covers 98% of the dense targets with exact column alignment and delivers the
quality lift now. Recommended operating point among all Q4_K variants: **selective
+ imatrix** (best quality at the same speed).

## Provenance
- imat variant gguf: 43,677,591,168 bytes, sha256
  `b1dd08509231126b5f1596603fd2589ff8bdf8eae221420f3e4409110fee8bd4`, 234 Q4_K.
- Requant: `requant_dense.py --imatrix imatrix_unsloth.gguf_file` (228/232 weighted).

## Receipts (this dir)
cosine_q8_vs_imat_512.txt, imab_{q8,imat}_*.csv (decode A/B), coherence_imat.txt,
smoke_imat.out. See ../selective/ for the plain-selective baseline and
../FINDINGS-dense-q4k-kernel-decode-win.md for the kernel root-cause.
