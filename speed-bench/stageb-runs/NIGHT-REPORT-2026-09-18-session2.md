# Night report — 2026-09-18 (autonomous session 2: items 1+2 approved)

You approved items 1 (MTP-on wiring) and 2 (Design B / paged-KV memory work) and
said keep running autonomously. Here is everything, honestly.

## TL;DR
- **Item 1 (MTP): done.** --mtp wired into ds4-bench; measured **0% draft
  acceptance** → MTP is a slowdown at this quant, not a speedup.
- **Item 2 turned into a serious integrity fix.** The previously-committed
  "Phase 2.1/2.2 paged-KV gate bit-identical" (50a5c0e) was a **FALSE POSITIVE** —
  the paged path never actually ran. Found via a fresh code audit, fixed 3 bugs,
  re-gated for real (bit-identical, path CONFIRMED active). This was the right and
  necessary thing before building anything on top.
- **Design B "memory win" — honest finding:** in BF16 it delivers **no** memory
  benefit (the shipping flat cache is already optimal for BF16). The real KV memory
  win is **FP8**, which I took as far as I safely could without your input.
- **Phase 3 started:** measured K/V dynamic range (→ E4M3) and the **FP8 quality
  matrix** (E4M3 KV keeps output coherent; cosine 0.985–0.998; divergence does not
  grow with context). FP8 itself is **blocked on your go-ahead** (new flag = ground
  rule 7; lossy = ground rule 6 quality review).

## Commits this session (on dongnh311/scallop, after a185f1f)
```
ed939b5 Item 3.2 (Phase 3): FP8-KV quality matrix measured (E4M3) — impl held (ground rule 7)
6855945 Item 3.1 (Phase 3): K/V dynamic-range measurement (FP8 format prep)
369c874 Item 2.1 (Phase 2) FIX: paged KV was INERT — now active + honestly re-gated
bf11ccb Item 1.1: wire --mtp into ds4-bench + measure (0% acceptance)
```

## Item 1 — MTP (done)
Wired --mtp (config+parser+engine opt+glm_mtp_timing; relaxed the bench speculative
gate from cfg.dspark to (dspark||glm_mtp); --mtp added to full help). The ready
scratchpad patch was INCOMPLETE (it left the gate on cfg.dspark → would have given
MTP-off numbers); I caught and fixed that. Measured 0% acceptance (confirm ctx4096:
"14 verify cycles, 0 drafts accepted"). Greedy MTP is lossless so output is
unaffected. 32K decode is impractically slow (per-frontier session snapshots copy
~GB of KV per step). Why 0% acceptance = Plan 2 Task 4 (out of order). diskwrites=0.

## Item 2 — Phase 2 paged KV: the integrity fix (the big one)
The 50a5c0e gate was fake-passing. Three compounding bugs (all now fixed, 369c874):
1. init ran AFTER graph_alloc, whose memset zeroed paged_kv_cache before the
   staging-buffer alloc → staging stayed NULL → paged path inert.
2. ds4_kv_cache_init IGNORED max_tokens (hard-coded 4 pages/ring) → the ring wrapped
   and aliased tokens past 2048 → silent corruption at any real context.
3. qwen4_paged_kv_active() probed layer 0, which is a LINEAR layer with no K/V →
   its staging is legitimately NULL → the check was always false.
Fixes: init before graph_alloc + preserve paged_kv_cache across the memset; ring
sized to ctx; active() probes the first full-attn layer; a one-time "paged KV path
ACTIVE" log; and run-kv-gate.sh now HARD-ASSERTS activation so this can't fake-pass
again. Re-gated bit-identical (0/248320) at ctx 512, ctx 8192 (18 pages/ring, >1
page), and ctx 8192 chunk 2048 (multi-chunk). Gate #5 (expert pager) re-verified
PASS after the core graph_alloc change.

## Why "Design B" doesn't deliver a BF16 memory win (important)
Design A (materialize) keeps flat + staging_full + paged store (~3x BF16 KV) — a
correctness milestone, env-gated (off by default), NOT a win. Design B (kernels read
the store directly) removes the duplication → back to ~1x = the SAME as the shipping
flat cache. There is no BF16 layout that beats a single contiguous flat cache; any
paging/staging is ≥ that. So the KV memory win is **entirely FP8**, not Design B.
Building a BF16 Design B would be risk (3 core attention kernels) for zero benefit,
so I did not — I moved to the real lever (FP8) instead.

## Phase 3 — FP8 (the real KV memory win), measured up to the decision point
- **3.1 K/V dynamic range (committed):** K absmax 15.7 / V absmax 34.3, both FAR
  inside E4M3 (±448). → **E4M3 + per-token(-head) scaling** (V per-token spread ~18x).
- **3.2 FP8 quality matrix (committed as findings; impl held as fp8sim.patch):**
  E4M3 KV vs BF16 — ctx512 cosine 0.985 (top1 match, top5 3/5); ctx8192 cosine 0.998
  (top1 flips on a near-tie, top5 5/5). Divergence does NOT grow with context. Output
  stays COHERENT (valid period-Italian both ways); the text difference is the normal
  greedy butterfly effect, not degradation. **FP8 looks viable** — but the definitive
  gate is ds4-eval (task accuracy), not text-match.

## DoD status (honest)
- ≥40 tok/s @220K zero swap: **NOT met, not achievable tonight** — needs FP8 KV +
  memory manager + paged-mm prefill (multi-phase). MTP is a dead end here (0% accept).
- logit gates: **PASS** (gate #5 bit-identical; gate #KV now REAL bit-identical).
- diskwrites=0: **yes**, every run (model file stat unchanged; resident/pager RO).
- ds4-eval 18/20: not re-run (resident path unchanged this session by construction).
- CSV provenance: present.

## DECISIONS I NEED FROM YOU (the KV memory win hinges on these)
1. **Enable FP8 KV?** It needs a new enable flag → ground rule 7 says stop + ask, so
   I held it. The quality matrix says E4M3 is promising. If you say go, the plan is:
   (a) run ds4-eval with fp8sim (DS4_QWEN4_KV_FP8SIM, apply fp8sim.patch) to confirm
   ≥18/20 (ground rule 6); (b) if it holds, implement the real in-kernel FP8
   storage+dequant = the ~half-KV memory win; (c) then the memory manager (Phase 4).
2. Anything you want changed about the fp8sim approach (per-64-block power-of-2 scale,
   reusing dsv4_e4m3fn_dequant_cpu) before I make it real?

## Ready-to-apply artifacts
- speed-bench/stageb-runs/fp8sim.patch — the FP8 quality probe (adds DS4_QWEN4_KV_FP8SIM).
- FINDINGS-item-{1.1-mtp,2.1-paged-kv-fix,3.1-kv-dynamic-range,3.2-fp8-quality}.md
