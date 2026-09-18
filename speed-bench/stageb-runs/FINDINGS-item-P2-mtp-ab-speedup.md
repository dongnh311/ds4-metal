# Item P2 MTP — net-speedup A/B (MTP on vs off), and a memory-manager over-allocation finding

Follow-on to the MTP paged-experts fix (ec70af4). Question: does 55.6% first-draft
acceptance actually make **decode** faster, given each cycle runs a batched verify
pass plus a draft pass? Answer: **regime-dependent — MTP helps when compute-bound,
hurts when SSD-read-bound.** No code change (MTP is lossless greedy; the fix is
already gated) → no bit-identical gate needed; pure measurement. All runs: ctx 4096,
gen 128, prose (promessi_sposi), model sha ed238d8d…, diskwrites=0.

## Results (gen_steady_tps, on/off ratio)
| config | cache | MTP off | MTP on | on/off | accept | read-GB/tok on/off |
|---|---|---|---|---|---|---|
| **resident** | n/a | 28.93 | **32.52** | **1.124** ✅ | 55.6% | n/a |
| **paged** | default 4.86 GiB | 5.45 | **4.74** | **0.870** ❌ | 55.6% | **1.118** |
| paged + MEMMGR | 43.32 GiB* | 2.24 | 1.92 | 0.857 ❌* | 55.6% | 1.000 |

\* the MEMMGR run swapped (see below) — not a clean datapoint; shown for the read-GB signal only.

Acceptance is identical (55.6%, 81 cycles / 45 accepted) across resident and both
paged configs — confirming the ec70af4 fix makes MTP bit-identical resident↔paged.
Prefill tps is unchanged by MTP (decode-only feature), as expected.

## Interpretation
- **Compute-bound (resident): MTP = +12.4%.** A real, modest win. The gain is far
  below the 55.6% accept rate because each cycle still pays ~1 batched verify forward
  (2–3 rows) + 1 draft forward; the amortization over accepted drafts nets ~+12%.
- **SSD-read-bound (paged, small cache): MTP = −13%.** Root cause in the data: the
  multi-row verify pass routes the speculative rows to *extra* experts, so
  `ses_read_gb_per_token` rises **+11.8%** (0.01324 → 0.01479) at a ~50% hit rate —
  every extra expert is an SSD read. When decode is dominated by expert I/O, doing
  more speculative expert reads per step is a net loss even at 55.6% acceptance, and
  it pushes read-GB the wrong way against the ground-rule-4 budget.
- **The read-GB penalty is a hit-rate artifact.** With the larger cache the MTP
  read-GB penalty **disappears** (on/off 1.118 → 1.000): the speculative rows' experts
  are already cached, so MTP no longer adds SSD reads. This means once decode is
  compute-bound (cache holds the working set), MTP should behave like the resident
  case (+~12%). The DoD scenario (220K with the memory-manager cache ≥ 34.31 GiB
  bundle) is that compute-bound regime — so MTP is expected to help there, IF the
  cache fits without swap.

## Secondary finding — the memory manager over-allocates into swap (item-(c) bug)
The `DS4_QWEN4_MEMMGR=1` run was meant to test the high-hit-rate regime but instead
**swapped 20 GiB** (`swap_growth_bytes` 0 → 20.4 GiB; `swap_peak` 2.6 → 23.1 GiB) and
throughput COLLAPSED (5.45 → 2.24 tps off). Cause: the manager sized the expert cache
from **non-routed** model bytes only —
`memory manager: expert cache budget 1.86 -> 43.32 GiB (model 6.32 + KV 0.13 of 64 RAM)` —
but the process actually maps the **full 41.72 GiB** model span (the engine's own
line: `resident model 41.72 GiB`). 41.72 (mapped) + 43.32 (cache) ≫ 64 GiB RAM →
20 GiB swap. So item (c)'s budget math (`RAM − 6.32 − KV`) is wrong whenever the full
GGUF is mmap-resident: it must subtract the actual resident footprint (the routed
experts are counted twice — once in the 41.72 mmap span, once in the pager cache).
This violates the zero-swap DoD and is the blocker for a clean high-hit-rate paged
MTP measurement. **Recommend fixing the memmgr over-allocation before any 220K MTP
bench** (and it is likely the higher-value DoD lever regardless of MTP).

## Bottom line
- MTP is correct and healthy (55.6% accept, lossless-greedy).
- MTP as-is **helps compute-bound decode (+12%) and hurts SSD-read-bound paged decode
  (−13%)**. It is NOT a win for the paged DoD at the default cache.
- It is expected to help at 220K *if* the expert cache fits the bundle without swap —
  which the current memory manager does not deliver (it over-allocates → swap). That
  memmgr fix, not MTP tuning, is the next DoD lever.

## Receipts
speed-bench/stageb-runs/mtp-ab/{resident,paged_defaultcache,paged_memmgr}_{off,on}.csv
Logs (scratchpad): ab_resident.log, ab_paged.log, ab_paged_memmgr.log.
