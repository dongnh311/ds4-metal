# KV / quant quality experiment — can a quant change break the decode ceiling? (No, on this runtime)

Goal (user): change quantization to push decode past its DRAM-bandwidth ceiling
(~34 tok/s MTP-off / ~40 --mtp on the light Orca model, M5 Pro 64 GiB), accepting
a quality cost if the speed is worth it. Profile-first, then measured on the real
artifact. Conclusion: NO quant change breaks the decode ceiling here.

## Step 1 — where does decode bandwidth go? (decode timeline @ ctx 32K)
Parsed the per-kernel GPU time of the decode command-buffers (decode_timeline_32k.txt):
| group | decode GPU % |
|-------|-------------:|
| DENSE (Q8_0 GEMV, `mul_mv_q8_0` 39.8%) | **53.2%** |
| EXPERTS (IQ2_XXS + Q2_K, sparse ~10/512) | 24.6% |
| other (norms / GDN / router / hc) | 18.4% |
| **ATTN / KV** | **3.8%** |
Corroborated by the flat decode curve (8K 29.5 -> 220K 30.5 tok/s MTP-off): KV read
does not grow into the bottleneck even at 220K. **Decode is bound by per-token
WEIGHT reads, dominated by the DENSE Q8_0 projections** (read in full every token;
3.0 GiB/token) — not KV, not the sparse experts.

## Step 2 — KV -> FP8: MEMORY win, NOT a decode-speed win
KV is only 3.8% of decode, so halving it (FP8 E4M3, already built + quality-gated on
the dev branch: ds4-eval 12/12, cosine 0.985) can lift decode by <=~2% — noise.
FP8 KV's real value is MEMORY: ~1.63x smaller KV (~2.6 GiB freed @220K), i.e. longer
zero-swap context / larger expert cache — not throughput. (Porting the Qwen4 FP8-KV
path from dev to the prod runtime was therefore not pursued for *speed*.)

## Step 3 — DENSE Q8_0 -> Q4_0: no decode gain + large quality loss (worst of both)
Built a variant requantizing all 336 eligible per-layer dense Q8_0 projections to
Q4_0 (the only sub-8-bit dense type the runtime accepts; ds4.c:5257), keeping
output.weight at Q8_0. Tool: gguf-tools/requant_dense.py (numpy block_q4_0 packer;
the shipped quantizer cannot emit Q4_0). Dense 3.00 -> 1.59 GiB (-1.41 GiB).
Measured (ctx 4096, MTP-off, ds4-bench):
| | prefill t/s | decode t/s |
|---|---:|---:|
| Q8_0 (baseline) | 458 (cold) / ~535 | **31.9** |
| Dense Q4_0 | 530 | **30.7** (−4%) |
- **Decode does NOT improve — it is ~4% slower.** Prefill improves ~16% (batched
  GEMM benefits from fewer bytes), but at DECODE (batch=1 GEMV) the specialized
  `kernel_mul_mv_q8_0_f32` is faster per call than the Q4_0 dense path; the 4-bit
  dequant overhead + a less-optimized kernel eat the bandwidth saving. So decode at
  batch=1 is partly kernel/latency-bound, not purely DRAM-bound (refines Task 1).
- **Quality drops sharply:** cosine(Q8, DenseQ4_0) logits = **0.882** (vs FP8-KV's
  0.985), argmax token FLIPS (16 -> 17), top-5 overlap 3/5, max|Δlogit| 4.6. Q4_0 is
  a crude 4-bit format on the sensitive dense backbone.
=> Dense Q4_0 is the worst of both worlds (no speed, big quality hit). Not adopted;
the variant gguf was deleted.

## Bottom line
On this DS4 M5 runtime the decode ceiling cannot be broken by quantization:
- KV (3.8% of decode) is too small a share — FP8 KV is a memory lever, not speed.
- Experts are already 2-bit (IQ2_XXS / Q2_K).
- Dense is the real per-token cost (53%), but its only accepted sub-8-bit type
  (Q4_0) is not faster at decode here and costs large quality.
Remaining real decode levers are HARDWARE (more memory bandwidth) or NEW optimized
low-bit dense GEMV kernels (e.g. a tuned Q4_K/Q5_K dense path with a decode-grade
kernel — kernel engineering, out of scope). The shipped Q8_0 dense + MTP is the
right operating point for this model on this hardware.

## Receipts
decode_timeline_32k.txt (per-kernel decode profile), dense_q8.csv / dense_q40.csv
(A/B), gguf-tools/requant_dense.py (the Q4_0 requant tool). FP8-KV quality/mem:
dev-branch commit 3c6f473 (ds4-eval 12/12, cosine 0.985, KV 1.63x).
