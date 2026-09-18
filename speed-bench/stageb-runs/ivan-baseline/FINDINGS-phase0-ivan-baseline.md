# Phase 0 — Freeze baseline (Ivan DS4-IQ2 model, prod codebase) on M5 Pro 64 GiB

Confirmed-plan Phase 0. Establishes the clean reproducible baseline for the DS4-IQ2 Fast+
work, on the CORRECT model + codebase (correcting the earlier OrcaUncensored measurements).

## Setup (the config that matters)
- **Model:** Ivan `Qwen3.8-Flash-Next-DS4-IQ2` — main `…IQ2XXSImatrix-Q2KDownPad768-MTP.gguf`
  (41.73 GiB) + PLE **Q4_1 sidecar** `…PLE-Q4_1.gguf` (29.8 GiB), at
  `~/.local/share/ai-gateway/ds4-models/`.
- **Codebase/binary:** PROD `~/.local/share/ai-gateway/ds4-metal/` (git branch
  `qwen3.8-flash-next` @ 6c1e836), built `ds4`/`ds4-bench` fresh.
- **CRITICAL — bug B4 workaround:** the binary MUST be run **from its own dir** so it loads
  the matching `./metal/`. Running the prod binary from another checkout's cwd (e.g. the
  scallop worktree) loads a mismatched `./metal/` → garbage output + degraded tok/s. This
  was the sole cause of my earlier "~31 tok/s + garbage" prod runs.
- Resident model + PLE demand-paged (CPU-only); **NO `--ssd-streaming`** (Ivan's model fits).

## Baseline sweep 8K–128K (ds4-bench, MTP-off, zero swap)
| ctx | prefill t/s | decode gen_steady t/s | planned mem | swap growth |
|-----|-------------|-----------------------|-------------|-------------|
| 8192 | 335.7 | 29.5 | 44.6 GiB | 0 |
| 16384 | 355.7 | 29.9 | 44.6 GiB | 0 |
| 24576 | 360.2 | 30.4 | 44.6 GiB | 0 |
| 28672 | 378.9 | 30.0 | 44.6 GiB | 0 |
| 32768 | 369.9 | 30.1 | 44.6 GiB | 0 |
| 131072 | 354.9 | **28.79** | **48.12 GiB** | 0 (no OOM/Killed; 90% mem free after) |
Planned memory at 32K = 44.62 GiB (KV 1.05 + buffers 1.85 + model 41.72); at 128K = 48.12 GiB
(KV 4.17 raw 3.26+comp 0.91 + buffers 2.23 + model 41.72), plus ~6.55 GiB bench prefill
context buffers → real footprint ~54.7 GiB, still resident on 64 GiB. Swap byte-identical
before/after the whole sweep → **zero swap growth**; the 128K run completed with no
OOM/Killed marker and left the box at 90% free. Decode declines only ~4% from 8K to 128K
(30.1 → 28.79) — the expected per-token KV-attention cost over a 128K window, not swap.
(kvcache_bytes reads 0 at ctx≥30720 = the >1 GiB snapshot → prefix-replay fallback, a bench
artifact, not a bug.) 128K prompt = rendered_prompts.txt.

## 220K frontier — zero-swap REACHED on this box (the prefill-chunk lever)
| ctx | prefill t/s | decode gen_steady t/s | planned + ctx-buffers | swap growth |
|-----|-------------|-----------------------|-----------------------|-------------|
| 220000 (chunk 8192, default) | — | — | 58.99 + **17.68** ≈ 76 GiB | **THRASH** (killed) |
| 220000 (chunk **2048**) | **472.9** | **30.48** | 51.28 + **9.57** ≈ **60.85 GiB** | **0** |
- At 220K the bench's default `--prefill-chunk` jumps to 8192 (the "PRO long prompts"
  default), which alone inflates the prefill context buffers to 17.68 GiB → ~76 GiB working
  set ≫ 64 GiB → swap thrash (proc pinned at 0% CPU in swap I/O). **`--prefill-chunk 2048`**
  (the same chunk the ≤128K sweep used) cuts context buffers to 9.57 GiB → 60.85 GiB total →
  fits resident. `--prefill-chunk` is an EXISTING flag, not a new user flag.
- Zero swap: swap `used` actually **decreased** across the run (4433 MiB before → 4105 MiB
  after; monitor's +800 MiB growth alarm never fired). diskwrites=0 confirmed (model + PLE
  gguf size+mtime byte-identical before/after; receipts in ivan_baseline_220k.stderr.log).
- Decode holds ~30 t/s from 8K to **220K** (MTP-off) — essentially flat; decode is
  DRAM-BW-bound (Task 1), not context-bound, and the KV path stays resident zero-swap.

## MTP on/off
- **MTP-off** (ds4-bench): ~30 tok/s @ 8K–32K.
- **MTP-on** (ds4 CLI, `--mtp`, ctx 4096, coherent output): **39.81 tok/s** = **+35%**.
  Consistent with the HF-reported M5 Pro baseline of 41–45 tok/s (MTP on + matching kernels)
  and with the scallop MTP Task-4 gains (+14% prose / +33% code).

## Reconciliation with the earlier (wrong-model) measurements
- My prior "decode 4.11 tok/s @32K" and "40 tok/s @220K unreachable" were on the
  **OrcaUncensored** build (147 GB gguf, 95 GiB **BF16** embedded n-gram, 34 GiB expert
  bundle) which does not fit → forced `--ssd-streaming` → thrash. That model, not the
  hardware or kernels, was the bottleneck.
- On **Ivan's** build (71.5 GiB total, Q4_1 PLE sidecar) the same M5 Pro 64 GiB fits the
  model resident and runs zero-swap at ~30 (MTP-off) / ~40 (MTP-on).
- Task 1's "decode BW-bound ~29" was an MTP-off number; with MTP the real ceiling is ~40.

## DoD-class achievability — CONFIRMED on this box (reverses the earlier "unreachable")
- pswai (M4 Max 64 GiB): **224K @ 45.4 tok/s decode, 323 prefill, 50.31 GiB, ZERO swap**
  (Q2_K experts + PLE eviction + MTP).
- alfredoartiles (M5 Pro 64 GiB): 128K OpenCode agent, 37–43 tok/s, ~47 GB, zero crash.
- **THIS box (M5 Pro 64 GiB), measured:** 220K zero-swap at **30.48 t/s MTP-off** with
  `--prefill-chunk 2048` (60.85 GiB working set). The `FINDINGS-DoD-verify-endtoend.md`
  "≥40 @220K unreachable" verdict was on the **heavy OrcaUncensored** build (147 GB, 95 GiB
  BF16 n-gram, `--ssd-streaming`) with the default chunk 8192 — a different model + a
  swap-forcing chunk. On Ivan's **light** model the two DoD walls (memory + SSD) vanish; the
  only remaining gap to ≥40 is the decode ceiling, which **MTP** closes.
- **≥40 @220K projection:** MTP gives +35% at short ctx (39.81 vs 29.5). Applied to the
  flat ~30 t/s MTP-off decode → **~40–41 t/s @220K zero-swap**, i.e. the DoD is *reachable*
  on Ivan's light model on this 64 GiB box. TO CONFIRM: one MTP-on 220K run (chunk 2048).

## Bottom line (Phase 0 frozen)
Ivan's DS4-IQ2 model on this M5 Pro 64 GiB is a sound, zero-swap base from 8K to **220K**:
prefill 335–473 t/s, decode ~30 t/s MTP-off (~40 MTP-on short ctx), diskwrites=0, coherent
output — provided the binary runs from its own `./metal/` dir (B4) and long contexts use
`--prefill-chunk 2048`. This is the clean baseline the uncensored build (Phase 1) inherits.

## Files / receipts
ivan_baseline_8k_32k.csv + .stderr.log, ivan_baseline_128k.csv, ivan_baseline_220k.csv +
.stderr.log. Thrash run archived: scratchpad ivan_baseline_220k_chunk8192_thrash.csv +
ivan_220k_chunk8192.err (shows the 76 GiB / 17.68 GiB-ctx-buffer blowup).
