# SCALE-3.0: RAM/tok-s frontier vs number of streamed layers (2026-09-22)

`DS4_QWEN4_STREAM_FULL_LAYERS=K` keeps the first K routed layers resident under
`--ssd-streaming` and streams the rest. Every streamed layer pays one host sync
per token whatever the hit rate, so the number of streamed layers, not the cache
size, is the main RAM/speed knob.

Method (same as SCALE-2c): M5 Pro 64 GB, ctx 8192, 600 tokens, greedy, seed 1,
no MTP, prompt `longgen_2c.txt`. RAM = steady system-wide wired pages from
`vm_stat` during decode (includes ~3.3 GiB machine baseline).

## PROD (`...DenseQ4Kselimat-Q2KDownPad768-MTP`, 40.7 GiB, 48 streamable layers)

| streamed | cache | wired GiB | tok/s | vs resident |
|---:|---:|---:|---:|---:|
| 0 | - | 49.17 | 35.82 | 100% |
| 6 | 2GB | 45.45 | 32.10 | 90% |
| 12 | 4GB | 43.19 | 31.39 | 88% |
| 12 | 2GB | 41.19 | 27.32 | 76% (dominated) |
| 24 | 8GB | 38.66 | 28.39 | 79% |
| 24 | 4GB | 34.65 | 24.99 | 70% |
| 48 | 16GB | 29.61 | 21.65 | 60% |
| 48 | 8GB | 21.62 | 18.82 | 53% |

## unc31 (`...SelQ4Down18-48-DenseQ4Kselimat-MTP`, 47.7 GiB)

Layers 0-17 (Q2_K down) are off the slab class and already resident, so only
the 30 Q4_K layers 18-47 can stream: K=0 streams 30, K=24 streams 24, K=36
streams 12, K=42 streams 6.

| streamed | cache | wired GiB | tok/s | vs resident |
|---:|---:|---:|---:|---:|
| 0 | - | 56.16 | 33.86 | 100% |
| 6 | 3GB | 51.71 | 30.53 | 90% |
| 6 | 2GB | 50.82 | 27.05 | 80% (dominated) |
| 12 | 6GB | 49.15 | 29.36 | 87% |
| 12 | 4GB | 47.14 | 28.21 | 83% |
| 24 | 12GB | 44.04 | 26.45 | 78% |
| 24 | 8GB | 40.01 | 25.74 | 76% |
| 30 | 16GB | 42.50 | 25.26 | 75% (dominated by 24/8GB) |
| 30 | 8GB | 34.42 | 21.83 | 64% |

The all-streamed points of both models reproduce SCALE-2c (PROD 21.58/19.4 and
29.59/21.4; unc31 34.42/22.0 and 42.50/24.8), and both resident baselines match.

## Reading

- Cost model, fitted on both models with the cache near half of the streamed
  experts: about 0.3 ms per token per streamed layer plus about 0.6-0.8 ms fixed
  once streaming is on. Tighter caches add SSD misses (0.5-0.7 ms per layer).
- No sharp knee. 90% of resident speed saves 3.7 GiB (PROD) or 4.5 GiB (unc31);
  the middle of the curve loses about 0.65 tok/s per GiB saved.
- At equal RAM, more streamed layers with a larger cache beat fewer layers with a
  small cache (PROD 24/8GB over 12/2GB; unc31 24/8GB over 30/16GB).
- unc31 with 12 streamed layers and a 6GB cache wires 49.15 GiB, the same as PROD
  resident (49.17), at 29.36 tok/s (87% of unc31 resident).
- The drain share printed by `DS4_QWEN4_STREAM_TIMING` is not a waste measure: it
  includes waiting for the GPU work of every resident layer encoded before the
  sync. Compare tok/s against resident instead.

## Output

Every streaming run of a model is byte-identical to every other (PROD
`0fbb2a7a...`, unc31 `8179896f...`), and PROD matches exp-pld c9d494c streaming,
so the knob does not change output. Streaming output differs from resident
output on this prompt for both models, in the parent as well; that divergence
predates this change. A `1GB` cache fails at prefill in the parent too.

## unc31, 12 streamed layers @ 6GB, with MTP: 8K vs 124K real context

`--mtp --mtp-timing`, greedy, `DS4_QWEN4_PLE_PREFETCH_FULL=0`. 124K rows use
`DS4_QWEN4_KV_FP8=1 --prefill-chunk 2048 --ctx 131072` and the `needle.txt`
prompt (124,141 tokens, code at the start, question at the end).

| ctx | mode | peak wired | decode wired | prefill t/s | gen t/s | MTP accept | needle |
|---:|---|---:|---:|---:|---:|---:|---|
| 8K | resident | 56.40 | 56.29 | (short prompt) | 38.20 | 70.9% | - |
| 8K | 12 @ 6GB | 49.40 | 49.18 | (short prompt) | 34.46 (90%) | 73.8% | - |
| 124K | resident | 55.62 | 54.94 | 587.6 | 34.20 | 82.9% | HIT |
| 124K | 12 @ 6GB | 48.41 | 45.56 | 106.2 (18%) | 23.88 (70%) | 85.0% | HIT |

- MTP does not hurt streaming at 8K (90% with MTP vs 87% without).
- At 124K the two runs produce byte-identical output; both answer the needle.
- Decode keeps only 70% at 124K (overhead ~12.7 ms/token vs ~2.8 ms at 8K).
  Not isolated; this run did not set `DS4_QWEN4_STREAM_TIMING`.
- Prefill is the blocker: 124K takes ~1169 s streamed vs ~211 s resident.
  The extra ~958 s over ~61 chunks x 12 layers is ~1.3 s per layer per chunk to
  load roughly the whole layer (~950 MiB, a 2048-token chunk touches nearly all
  512 experts), i.e. ~0.7 GB/s effective, far below sequential SSD bandwidth.
  That points at per-expert reads, not the SSD. Each layer's gate/up/down
  experts are contiguous tensors, so a chunk whose union covers most of a layer
  could read it in bulk.
- Resident unc31 already fits at 124K with FP8 + chunk 2048 (peak 55.62 GiB), so
  streaming is not needed for fit at this context; it buys ~7 GiB of headroom.

## Prefill fix (b704550): GEMMs stay on non-streamed layers

Root cause of the slow streaming prefill and of the streaming/resident output
difference: `qwen4_graph_moe` disabled the prefill expert GEMMs (T > 64) on every
layer whenever `--ssd-streaming` was on, so resident and pinned layers also ran
the per-token row kernels. Streaming with zero streamed layers (K=48) was 116
t/s against 555 resident on a 16K prompt, and chunk 4096 was no faster than
2048, so the cost was per token, not per chunk or per read. The gate is now per
layer, using the span planner's predicate.

unc31, 124K needle (124,141 tokens), FP8 KV, MTP, 12 streamed layers @ 6GB:

| build | chunk | prefill t/s | 124K prefill time | gen t/s | peak GiB | output |
|---|---:|---:|---:|---:|---:|---|
| resident | 2048 | 587.6 | ~211 s | 34.20 | 55.62 | reference |
| streaming, before fix | 2048 | 106-108 | ~1160 s | 23.9-25.7 | 48.4 | == resident |
| streaming, after fix | 2048 | 230.4 | ~540 s | 26.10 | 48.36 | == resident |
| streaming, after fix | 4096 | 229.0 | ~542 s | 24.58 | 49.83 | == resident |

Prefill goes from 18% to 39% of resident. Chunk 4096 buys nothing and costs
1.5 GiB. The remaining gap is the 12 streamed layers, which still use the row
kernels in prefill because no GEMM reads through the expert address table yet.

16K prompt, same model: 0 streamed 522 t/s, 6 streamed 303, 12 streamed 233
(resident 555); every output matched resident.

## Expert staging (SCALE-3.2): GEMM prefill for streamed layers, and a working fallback

`ds4_gpu_qwen4_stream_stage_layer` loads the experts one dispatch needs into a
buffer laid out like the layer's gate/up/down tensors, and `qwen4_bind_weight`
redirects binds of exactly those tensors to it while it is active. The unchanged
GEMM and row kernels then run on streamed layers. Staged prefill is on by
default (`DS4_QWEN4_STREAM_STAGE_PREFILL=0` disables it); the fallback always
stages when a cache load fails for a streamed layer.

unc31, 124K needle, FP8 KV, MTP, 12 streamed layers @ 6GB, chunk 2048:

| build | prefill t/s | 124K prefill time | gen t/s | peak GiB | output |
|---|---:|---:|---:|---:|---|
| resident | 587.6 | ~211 s | 34.20 | 55.62 | reference |
| streaming, original | 106.2 | ~1170 s | 23.88 | 48.41 | == resident |
| per-layer GEMM gate (b704550) | 230.4 | ~540 s | 26.10 | 48.36 | == resident |
| + expert staging | 541.3 | ~230 s | 23.69 | 48.18 | == resident |

16K prompt: 12 streamed 455 t/s (82% of resident 555), 6 streamed 474 t/s;
staging off 214 t/s. All outputs matched resident. The staging buffer is 0.93
GiB (Q4_K layer) or 0.71 GiB (PROD Q2_K layer); measured peak rose only ~0.24
GiB at 16K and not at all at 124K, since prefill no longer fills cache slots.

Fallback: the two configs that used to fail at prefill with "not covered by
mapped model views" now run and match resident output: PROD with a `1GB`
cache (budget reads as 1 slot; gen 31.4 t/s) and unc31 K=47 with 6GB (mlock
fails; peak 57.4 GiB, a mis-sized cache for one streamed layer).

Decode at 124K stays at ~70% of resident (23.7 vs 34.2); it is untouched by
these changes and its cause is still not isolated.

## Correction: long-context decode is ~88% of resident, not ~70%

The 124K "decode ~70%" figures above came from the needle prompt, whose answer
ends at EOS after about 75 tokens (41 verify cycles + 34 accepted drafts), and
they started from a cold expert cache: with staging, prefill no longer fills
the cache. Re-measured with the same 124K context but a closing instruction to
count to 400, so both runs decode ~595 tokens (unc31, FP8, MTP, chunk 2048):

| mode | gen t/s | vs resident | prefill t/s | MTP accept |
|---|---:|---:|---:|---:|
| resident | 35.08 | 100% | 588.3 | 75.5% |
| 12 streamed @ 6GB | 30.73 | 88% | 538.2 | 75.5% |

Outputs are byte-identical. The streaming cost is 4.0 ms/token for 12 layers,
0.34 ms per streamed layer, the same per-layer cost measured at 8K, so long
context adds no decode cost of its own.

Cache warmup after a long prefill: the first ~250 decode calls with misses
average 6.3 missed experts per call, settling to ~1.7. The first 480 calls
spend 531 ms loading against ~90 ms per 480 calls afterwards, and drain adds
~0.45 s, so warmup costs ~0.5-0.9 s once per prompt: 3-4% of a 600-token reply,
20-30% of a 75-token one, which is what the earlier figure measured.
