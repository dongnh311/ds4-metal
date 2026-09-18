# Item P2 MTP — fix 0% draft acceptance on the paged build (nextn experts are resident-only)

**Headline reversal:** the prior record (item 1.1, commit a185f1f) said MTP gives
"0% draft acceptance at this quant" and concluded MTP is a net slowdown. That was
**wrong about the cause.** MTP is healthy; the 0% was a **paged-path bug** that fed
the MTP head the wrong experts. Fixed — paged MTP now equals resident MTP exactly.

## How it was found — the MTP_STATS diagnostic (allowed env, ground rule 7)
`DS4_QWEN4_MTP_STATS` existed only as a dead compile-time `#ifdef` (ds4.c, never
`-D`'d in the Makefile) that, even if enabled, printed the same cycles/accepted/rate
as `--mtp-timing`. Rebuilt it as a **real runtime `getenv`** diagnostic that, per
verify cycle, ranks the first draft token `d` inside the trunk's row-0 distribution
(`#vocab logits > rows[d]`; rank 1 == draft is the trunk argmax == accepted) and
prints the histogram at teardown. This turns a bare "0%" into a diagnosis: rank-1
misses ⇒ accept bug; a tight rank 2-5 mass ⇒ near-miss (quant); a rank 501+ mass ⇒
the head output is garbage. Env-gated (O(V) scan only when on); default path byte-
for-byte unchanged.

## What the diagnostic showed (ctx 4096, 64 gen, prose = promessi_sposi)
```
RESIDENT (no --ssd-streaming):
  41 verify cycles, 21 accepted (51.2%) | rank1=21 (51.2%) 2-5=4 6-50=10 51-500=4 501+=2 | mean#better=97.8
PAGED (--ssd-streaming), BEFORE fix:
  62 verify cycles,  0 accepted (0.0%)  | rank1=0  2-5=0 6-50=3 51-500=10 501+=49    | mean#better=13421.2
```
Same weights, same head math, but paged drafts rank ~13421/151936 = **garbage**, not
quant near-miss. So the head is fine (resident 51%); paging corrupts its input.

## Root cause (ds4.c)
The nextn (MTP) layer is the appended last layer, `il = DS4_N_LAYER-1 = 48`. The SSD
expert bundle covers **only the 48 trunk MoE layers** (`qwen38-experts.index.json`
`layer_count=48`; 512 experts × 3 tensors × 48 = 73728 bundles), so the nextn
layer's experts are **resident-only** (not in the bundle). But the two nextn MoE call
sites passed **`layer_idx=0u`**:
- `qwen4_graph_mtp_steps` (ds4.c:59486) `qwen4_graph_moe(g,m,l,T,0u)`
- `qwen4_graph_mtp_chain_step` (ds4.c:59576) `qwen4_graph_moe(g,m,l,1,0u)`

With the pager active, `qwen4_graph_moe` took its paged branch and staged **trunk
layer 0's** experts (`layer_idx=0`), then ran them against the **nextn layer's
routing** (`g->selected` from `l`'s router) → wrong expert weights → garbage drafts →
0% acceptance. Resident mode uses `l->ffn_*_exps` (the correct nextn experts from
`m->map`), hence 51%. The logit gate never caught this: it runs `--gen-tokens 0`
(pure prefill), and prefill loops only the 48 trunk layers — the nextn layer runs
only inside the MTP step.

## Fix (ds4.c, 3 edits; `layer_idx` is used ONLY in the paged branch, verified)
1. Guard the pager branch: `if (ok && g->pager && !ds4_qwen4_layer_is_nextn(layer_idx))`
   — the nextn layer always takes the resident `l->ffn_*_exps` path.
2/3. Pass the real nextn index `DS4_N_LAYER-1u` (not `0u`) at both nextn call sites,
   so the guard fires (`is_nextn(48)=true`) and no layer-0 staging buffer is touched.

## Verification
```
PAGED (--ssd-streaming), AFTER fix:
  41 verify cycles, 21 accepted (51.2%) | rank1=21 (51.2%) 2-5=4 6-50=10 51-500=4 501+=2 | mean#better=97.8
  == byte-identical to RESIDENT (same cycles/accepted/histogram)
```
Regression: `run-logit-gate.sh 512 512` (paged-mm prefill) still **GATE PASS
bit-identical** (nextn layer isn't in prefill, so the guard cannot affect it).
diskwrites=0. Resident MTP unchanged (no pager). Trunk layers still page normally.

## Reconciliation with item 1.1
Item 1.1's "0% acceptance → MTP is a slowdown at this quant" was this bug, not quant
damage. On the fixed build MTP accepts **51.2%** of first drafts on prose (bimodal:
exact-argmax or a genuine branch point). MTP-on decode measured 30.76 tok/s at ctx
4096 resident. Net-speedup A/B (on vs off) and the multi-workload acceptance campaign
(code, agentic) + depth-window tuning remain as the follow-on P2 MTP work; this item
delivers the correctness fix that makes any of that meaningful on the paged (DoD)
build. The 32K MTP DNF (item 1.1, session-snapshot restore cost) is a separate,
still-open decode-cost issue.

## Files / receipts
ds4.c (MTP_STATS runtime diagnostic: session-struct rank accumulators,
qwen4_mtp_stats_enabled/_record, teardown print; nextn paged-experts guard + 2 call
sites). Receipts: scratchpad/mtp_stats_probe.log (resident 51%),
mtp_stats_paged.log (paged 0% before), mtp_stats_paged_fixed.log (paged 51% after),
regress_gate_after_mtpfix.log (prefill gate still bit-identical).
