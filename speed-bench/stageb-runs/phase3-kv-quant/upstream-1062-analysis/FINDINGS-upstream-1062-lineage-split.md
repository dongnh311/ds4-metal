# Adopting upstream qwen throughput (PR #1062): blocked by the --ple lineage split

Investigated how to bring upstream's qwen decode/throughput wins (d0b7434 +
PR #1062) to the 64GB light Orca model. Result: a clean rebase is blocked by a
model-packaging lineage split; the win requires a feature port, deferred as a
"worth testing" future task (user, 2026-09-18).

## What PR #1062 delivers (antirez/ds4, merged into upstream/main @ 8db1d1d)
Native batched decode + MTP for Qwen3.8-Flash-Next on Metal (33 commits, ds4.c +
ds4_metal.m, Metal only):
- **16 streams, plain @4k: 3.1 -> 203 tok/s (65x)** — multi-session concurrency.
- **1 stream, plain: 50.2 -> 54.3 (+8%)** — via d0b7434 (concurrent n-gram row
  preads) + predictor primed during prefill.
- **1 stream, MTP code: ~50 -> 78-84 (+57-68%)**.
Includes: Q8 batch GEMM on fp32 simdgroup, specialized grouped MoE passes,
GPU-side n-gram gathering, chained draft prediction.

## The blocker: two divergent model-packaging lineages
- **orca-bugfixes lineage (our PROD, old base):** external PLE n-gram sidecar via
  `--ple` (ds4_cli.c:2025, g_ds4_ple_sidecar, qwen4_ple_sidecar_* helpers). The
  64GB light model = light main + demand-paged Q4_1 sidecar. Has NO parallel
  n-gram reader (dispatch_apply count = 0); d0b7434's target `qwen4_ngram_read`
  does not exist here.
- **antirez/upstream/main lineage:** self-contained embedded BF16 n-gram
  (ngram_fd/ngram_tensor; commit 9139e2a "self-contained Qwen BF16 n-gram
  releases"). **DROPPED `--ple`.** d0b7434 optimizes THIS reader. The
  self-contained BF16 n-gram is ~95GB -> does NOT fit 64GB, so the light --ple
  approach is still required; we cannot simply switch models.
Net: upstream/main cannot load our light model ("unknown option: --ple",
verified by building orca-rebase = upstream/main + Q4_K and running it).

## Verified this session
- Built `orca-rebase` (branch on origin) = upstream/main + the 3-edit Qwen dense
  Q4_K wiring re-applied. Builds clean (make ds4 ds4-bench). Confirms Q4_K wiring
  ports forward with no conflict, and d0b7434 + PR #1062 are present natively.
- Cannot run the light model on it (no --ple). B1/B2/B3 already upstream; B4 not
  needed from the build dir.

## Paths to actually get #1062 on the light model (deferred)
- **A (worth testing, unlocks everything):** port the external PLE sidecar
  (--ple loader + qwen4_ple_sidecar_* + g_ds4_ple_sidecar, ~14 refs + helpers)
  forward onto upstream/main, reconciling with the self-contained n-gram
  rearchitecture. Then the light model runs on the batched lineage -> 65x
  multi-session (valuable if the ai-gateway serves concurrent requests) + 8%
  single-stream + MTP. Base branch ready: `origin/orca-rebase`. ~hours, moderate
  risk, PROD backups exist (baseline-6c1e836).
- **C (cheaper, single-stream only):** back-port just the concurrent n-gram-read
  idea into the orca-bugfixes --ple sidecar read path. ~+8% single-stream, no
  batching. ~1h, low risk.

## Status
Deferred (user: "note lại, đây cũng là 1 cách nên test"). PROD unchanged
(orca-bugfixes: Q4_K + imatrix models shipped). No PROD rebuild performed.
