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
