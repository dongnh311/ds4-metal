# Item (a) — FP8 KV quality gate (spec §25 Phase 3, ground rule 6)

**User directive:** "Bật FP8 KV và a,b,c đi" — enable FP8 KV, then (a) run the
ds4-eval quality gate to confirm ≥ baseline BEFORE the in-kernel memory win.

## What was measured
ds4-eval `--suite core --questions 12 --retry-incomplete`, two configurations,
same model (gguf sha ed238d8d…, size 147207127040), same harness:
- **BF16** (resident) — the reference.
- **FP8** (E4M3, per-64-block power-of-2 scale) via `DS4_QWEN4_KV_PAGED=1
  DS4_QWEN4_KV_FP8SIM=1` — the CPU E4M3 round-trip (`qwen4_fp8sim_vec`, ds4.c)
  that degrades K/V exactly as the eventual in-kernel FP8 would. It is a QUALITY
  probe (no memory win); ground rule 6 uses the quality matrix, not bit-identity.

Harness: `speed-bench/run-fp8-eval-gate.sh` (env-only, no new user flag).

## Result — PASS
```
BF16 : ds4-eval: 12/12 passed, runtime 00h:42m
FP8  : ds4-eval: 12/12 passed, runtime 00h:48m
```
Every one of the 12 core cases (AIME2025, GPQA Diamond, SuperGPQA) produced the
**same correct answer** under FP8 E4M3 KV as under BF16. FP8 KV preserves eval
quality at the BF16 baseline (12/12 = 100%, well above the ≥18/20 = 90% DoD bar).
Per-case receipts: `speed-bench/stageb-runs/fp8-eval/{bf16,fp8}-scores.txt`.

## Conclusion
E4M3 KV (per-64-block, power-of-2 scale, full 256 dims of post-RoPE K and raw V)
is quality-safe for this model. This certifies the numerics that item (b)
implements in-kernel for the real memory win. See
[FINDINGS-item-b-fp8-inkernel.md].
