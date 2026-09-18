# DoD end-to-end verification (spec §25): ≥40 tok/s @220K zero-swap — MEASURED NOT ACHIEVABLE on this M5 Pro 64 GiB

User directive: run the DoD verification on **this** machine (M5 Pro, 64 GiB), which was
effectively idle at run time (`memory_pressure` 91% free, no competing process >0.8 GiB).
All runs: locked GGUF sha `ed238d8d…e28d7a` (asserted by the pager), `--ssd-streaming` +
qwen38-experts.{bin,index.json}, diskwrites=0 verified (model+bundle stat identical
before/after every run; pager O_RDONLY). Provenance receipts in `dod/`.

## Result: the ≥40 tok/s @220K zero-swap DoD is unreachable on this hardware. THREE
independent walls, each measured; any one alone is fatal.

| config | ctx | cache eff. | hit | **gen tok/s** | swap growth | verdict |
|--------|-----|-----------|-----|---------------|-------------|---------|
| full-resident, no SSD (Task 1) | 4096 | — | — | **29.01** | 0 | compute ceiling |
| adaptive memmgr (swap-safe) | 32768 | 1.36 GiB | 32.5% | **4.11** | ≈0 (+0.21) | zero-swap, SSD-bound |
| forced 16 GiB | 32768 | 14.58 GiB | 39.3% | 1.66 | **+12.98** | swap thrash |
| forced 34 GiB (whole bundle) | 32768 | 34.10 GiB | 84.2% | 1.37 | **+29.95** | swap thrash |
| adaptive memmgr (swap-safe) | **220000** | **0.00 GiB** | — | **<1 (I/O crawl)** | ≈0 | zero-swap, cache starved |

### Wall 1 — Compute ceiling (from Plan 2 Task 1, commit b0f6fe6)
Even **fully resident with zero SSD involvement**, decode is DRAM-bandwidth-bound at
**~29 tok/s** (ctx 4096; NSG-invariant, GPU 90% busy). 40 tok/s is *above* this ceiling
before any context or streaming cost. At 220K, per-token KV attention over 220K tokens
adds work → strictly slower than 29. So 40 tok/s is unreachable on compute grounds alone.

### Wall 2 — Zero-swap starves the expert cache at 220K
The DoD forbids swap. Under the swap-safe adaptive cap (`DS4_QWEN4_MEMMGR=1`) at 220K:
```
KV 7.00 GiB (raw 5.46 + compressed 1.54) + context buffers 17.68 GiB + model + n-gram
grabbable 9.91 GiB, swap-safe cap 0.00 GiB (reserve 12.82) -> expert cache = 0.00 GiB (1 expert)
WARNING: 3 slots cannot hold one step's expert set; re-read on every access
```
KV (7 GiB) + context buffers (**17.68 GiB** at 220K — they scale with context) + the
model's own faulted pages consume the entire swap-safe RAM budget, leaving **nothing** for
the expert cache. Every one of the ~480 routed expert reads per token (48 layers × ~10)
then comes cold from SSD (~1.42 MiB each ≈ 682 MB/token). At the observed ~64 MB/s random
pread that is ≈10 s/token ⇒ **sub-1 tok/s**, and the process sits at ~5.8% CPU (pure I/O
wait). Zero-swap and a useful cache are mutually exclusive at 220K on 64 GiB.

### Wall 3 — Forcing a compute-bound cache causes swap (the opposite failure)
To make decode compute-bound you must cache the whole 34 GiB bundle. Forcing it
(`--ssd-streaming-cache-experts 36GB`) at only 32K already pushed **+29.95 GiB into swap**
(the 34 GiB anon cache + actively-mapped model + 95 GiB n-gram page cache can't coexist in
64 GiB) → decode collapsed to **1.37 tok/s**. 16 GiB forced still swapped +12.98 GiB → 1.66
tok/s. The swap-safe *small* cache (4.11 tok/s) actually beats every forced-large cache
here — confirming the adaptive memmgr cap (items P2-memmgr / P2-(1)) was the correct design.

## Why "dedicated box" in the plan meant *more RAM*, not *idle*
Prior notes said a "dedicated 220K box" would let the adaptive cap grant the whole bundle.
This run shows that on a **64 GiB** box, idle or not, it cannot: the resident working set at
220K is model(~6.3 faulted) + KV(7) + buffers(17.7) + n-gram(95 GiB file, demand-paged) +
whole-bundle cache(34) — far beyond 64 GiB. The DoD needs a machine with materially more
unified memory (≳128 GiB) to hold {KV@220K + 17.7 GiB context buffers + model + 34 GiB
bundle cache} without swap — **and even then decode is capped at the ~29 tok/s BW ceiling
(Wall 1)**, so 40 tok/s additionally requires MTP (Task 4: +14% prose / +33% code when
compute-bound) and/or a smaller expert quant (forbidden until last; a quality-gate change).

## Honest DoD scorecard (this hardware)
- ≥40 tok/s @220K: **NO** — max zero-swap decode is 4.11 tok/s @32K and <1 tok/s @220K;
  compute ceiling is ~29 tok/s even resident.
- zero swap: **achievable** only with the adaptive (near-empty) cache → SSD-bound; any
  useful cache swaps.
- diskwrites=0: **YES** (all runs, model+bundle byte-identical before/after; pager O_RDONLY).
- read-GB ≤ bundle × miss_rate × 1.1: **NO at zero-swap** — cache ≪ working set forces
  re-reads (32K adaptive read 137.9 GB ≫ budget); inherent to a starved cache, not a
  prefetch bug (item 1.2/1.4).
- CSV provenance: **YES** — dod/dod_summary.csv + per-run CSVs + stderr + before/after stat.

## Bottom line
On this M5 Pro 64 GiB, the ≥40 tok/s @220K zero-swap DoD is **not achievable** and the
reason is now measured, not assumed: a compute BW ceiling below 40, and a memory budget
that cannot simultaneously satisfy zero-swap and a compute-bound expert cache at 220K.
The engine behaves correctly at both extremes (adaptive cap keeps swap≈0; forcing a big
cache is the only way to swap, and it hurts). Reaching the DoD requires ≳128 GiB unified
memory (removes Wall 2/3) plus MTP and/or a smaller quant (to clear Wall 1).

## Files / receipts (dod/)
dod_summary.csv; per-run: zs_memmgr_32k.csv (4.11 tok/s, swapΔ0), zs_16g_32k.csv,
dod_32k.csv (forced 34G), zs_memmgr_220k.{grant_evidence.txt,stderr.log} (cap=0 proof);
dod_32k.model_files_{before,after}.txt (diskwrites=0). Compute ceiling: see
FINDINGS-item-P2-task1-decode-gemv.md (commit b0f6fe6).
