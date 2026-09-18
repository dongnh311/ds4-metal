# Item P2 MTP — Task 4: acceptance campaign across workloads + depth-policy validation

Completes the MTP follow-on (Plan 2 Task 4). Measurement/validation only — MTP is
lossless-greedy and no code was changed, so no bit-identical gate. All runs: RESIDENT
(the compute-bound regime where MTP helps — see the A/B finding that paged/read-bound
MTP is net-negative), ctx 4096, gen 128, DS4_QWEN4_MTP_STATS=1, diskwrites=0.

## Workload breadth — MTP value is strongly workload-dependent
| workload | 1st-draft accept | mean #better | MTP-off tok/s | MTP-auto tok/s | speedup |
|----------|------------------|--------------|---------------|----------------|---------|
| prose (promessi_sposi) | 55.6% (45/81) | 57.0 | 28.97 | 32.97 | **+13.8%** |
| code (ds4_expert_pager.c) | **86.4% (57/66)** | **5.3** | 28.10 | **37.45** | **+33.3%** |

Code is far more predictable than prose: 86.4% of first drafts are the exact trunk
argmax (rank1), and the draft-rank distribution is tight (mean #better = 5.3 vs 57 for
prose; 501+ bucket = 0). So MTP delivers a large decode speedup on code/deterministic
content and a moderate one on prose. Both are net wins in the compute-bound (resident)
regime.

## Depth-policy validation (prose) — the adaptive qwen4_spec_depth is well-tuned
| depth | 1st-draft accept | gen tok/s | vs off |
|-------|------------------|-----------|--------|
| off | — | 28.97 | — |
| **auto** (default) | 55.6% | **32.97** | +13.8% |
| forced depth-2 | 55.6% | 32.89 | +13.5% |
| forced depth-3 | 51.4% | **26.60** | **−8.2% (net LOSS)** |

Forcing depth-3 on general prose is a net loss (−19% vs depth-2): the 2nd chained draft
accepts too rarely to cover the extra draft+verify+rewind cost. The auto-policy
(`qwen4_spec_depth`, ds4.c) engages depth-3 ONLY on a perfect recent first-draft window
(bits≥8, cycles≥8, reject2_streak==0) and drops out at the first sign the chain stops
paying — so on prose it stays depth-2 and matches the optimum (auto 32.97 ≈ d2 32.89),
while correctly avoiding the depth-3 loss. **No policy change needed.**

## Scope / honesty
- **Agentic tool-calling workload NOT measured**: it needs a real tool-call trace (JSON
  syntax transitions, exact-sampling boundaries) that cannot be faithfully synthesized
  from a raw --prompt-file; faking it would give a misleading number. Left as a
  follow-on that requires a captured agentic session.
- These are the RESIDENT (compute-bound) numbers. On the paged/SSD-read-bound path MTP
  is net-negative at a small cache (verify pass inflates expert reads — earlier A/B),
  so MTP should be ON for compute-bound deployments (resident, or a dedicated box whose
  expert cache holds the working set) and OFF when decode is SSD-read-bound.
- `DS4_QWEN4_MTP_DRAFT_ROWS/DRAFT_VOCAB` (draft-head narrowing) not swept: the rank
  histograms show the draft head is already accurate (drafts are overwhelmingly rank1),
  so widening the draft head is low-ROI. Left as an optional follow-on.

## Conclusion
- Keep MTP + the auto-depth default; both are correct.
- MTP is a strong decode win on code (+33%) and a solid one on prose (+14%) when
  compute-bound. Combined with the cache work (item 1 adaptive cap + 1.4 slot fix that
  push toward compute-bound), MTP is a real contributor to decode throughput on a
  dedicated deployment.

## Files / receipts
mtp-task4/{prose,code}_{off,auto,d2,d3}.csv, log: scratchpad/mtp_task4.log. No code change.
