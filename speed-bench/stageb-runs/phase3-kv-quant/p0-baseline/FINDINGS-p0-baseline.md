# Phase 0 — baseline + ceiling (imat model, M5 Pro 64 GiB)

Binary: PROD `~/.local/share/ai-gateway/ds4-metal` @ orca-bugfixes 432e610 (pre-fix).
Model: `...-DenseQ4Kselimat-MTP.gguf` (40.67 GiB resident) + PLE Q4_1 sidecar
(demand-paged, insufficient RAM headroom for full prefetch). ctx 4096, n=400,
temp 0, seed 1, fixed prompt (prompt.txt). Run from PROD dir (matching metal/).

## Target metric: single-stream `--mtp` decode t/s
| config | run1 | run2 | run3 | mean | MTP acceptance |
|---|---|---|---|---|---|
| **--mtp** (target) | 42.60 | 42.80 | 43.13 | **42.84** | **67.9%** (161/237 cycles) |
| MTP-off (ref) | 35.92 | 36.89 | 36.85 | 36.55 | — |

- MTP wall-clock speedup over MTP-off: **+17.2%**.
- MTP effective tokens: 237 verify cycles + 161 accepted drafts = 398 ≈ n=400;
  avg 0.68 accepted draft tokens/cycle (acceptance is the --mtp multiplier).
- Deterministic (temp0/seed1): acceptance identical across runs; t/s jitter <1.5%.

## Gap to goal
Target = **60 t/s single-stream --mtp** → need **+40.1%** over the 42.84 baseline.

## Ceiling note (honest framing)
- Decode is memory-bandwidth bound on the per-token working set (dense attn Q8 +
  dense proj Q4_K touched every token; top-k experts IQ2_XXS/Q2_K; KV read; PLE
  n-gram rows demand-paged from SSD — the page-fault path adds latency + variance).
- MTP-off 60 t/s (27.4->16.7 ms/token, -39%) is at/above the DRAM-BW ceiling for
  this working set on M5 Pro → out of scope for single-stream without shrinking
  bytes/token (further quant). MTP-off is the reference, not the goal.
- The --mtp goal (60) is reachable ONLY by raising the effective-token multiplier
  (higher accepted-drafts/cycle) and/or cutting per-cycle kernel cost. Levers with
  a path to +40%: (2b) MTP draft-depth/margin tuning (acceptance is 67.9%, the
  direct multiplier), (2c) #1056 IQ2/Q2_K row-reuse kernels (cut expert GEMV cost),
  (2a) FLUSH_LAYER pipelining (default=2; small). PLE full-prefetch is NOT available
  (30 GiB PLE + 40.67 GiB model > 64 GiB RAM).

## Envs/flags used (all pre-existing; no new user flags)
`--mtp`, `--mtp-timing`, `--mtp-draft`, `--mtp-margin`, `--mtp-exact-sampling`,
`DS4_QWEN4_FLUSH_LAYER` (default 2), `DS4_DSPARK_STATS`. diskwrites_delta=0 (model
volume read-only; generation writes only to logs under the worktree).
