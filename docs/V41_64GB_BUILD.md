# DeepSeek-V4.1-Flash on M5 Pro 64 GB — Performance Build Spec

Build a **fast SSD-streamed, router-first, cache-heavy** DeepSeek-V4.1-Flash Q2
runtime profile for a single **M5 Pro / 64 GB / 1 TB SSD** box (single-stream
agent workload: Claude Code / Codex), **while treating the existing Qwen3.8
implementation as a hard regression-protected subsystem** that must not be
rewritten, degraded, or modified unless a proven generic fix passes all Qwen
benchmarks — **and gated by a Phase-0 feasibility measurement that must prove
the target tok/s is physically reachable before runtime optimization begins.**

## Definition of Done (the North Star — all five must hold)

Confirmed goal (2026-09-19): **DeepSeek-V4.1-Flash Q2 usable on M5 Pro 64 GB,
single-stream, for agent workloads, without degrading Qwen3.8.** DONE iff:

1. `ds4 --ssd-streaming` loads V4.1-Flash Q2 on 64 GB **zero-swap**;
2. decode **≥10 t/s (floor), target 15–20 t/s**;
3. passes **ds4-eval core** + a real **agent smoke** (coherent + correct);
4. **Qwen3.8 regression green** (no regression on the protected subsystem);
5. reproducible (pinned model/commit, recorded bench).

Non-goals: resident/>30 t/s (SSD-bound); big-machine/TP (Ivan's lane);
ternary/Bonsai is an **optional** accelerator, parked until Phase 0(b) shows a
speed shortfall.

## 0. Status & preconditions

- **DEFERRED.** Build starts when the scallop-branch work completes and frees
  the disk to ~**400 GiB** free. The Q2 GGUF is **340 GiB on disk** (HF lists
  365.71 GB decimal); it will not fit today.
- **Phase 0(a) DONE (2026-09-18) → GO on feasibility** (see §3).
- **Phase 0(b) PENDING** — needs the 340 GiB download; it is the remaining gate.
- Branch: develop on a fresh `feature/ds41-64gb` off `main`. **Never** merge
  `scallop` (Qwen 64GB/MTP) wholesale into `main`.

## 1. Hardware & model target

| | |
| --- | --- |
| Machine | Apple M5 Pro, 64 GB unified memory, ~1 TB SSD, macOS, Metal |
| Workload | single-stream agent (Claude Code / Codex) |
| Model | DeepSeek-V4.1-Flash, calibrated **Q2** (`antirez/deepseek-v4.1-flash-gguf`, `ds41f-q2`), Engram SSD-backed |
| Context rollout | 4K → 8K → 32K → 128K → 256K first; 1M only after 256K is stable |

## 2. Performance gates (re-anchored to measured physics)

| Gate | tok/s | Meaning |
| --- | --- | --- |
| Functional | runs, no OOM | must clear Phase 0(b) |
| Floor | **≥10** | below this = architecture/I-O problem, re-profile |
| Production | **15–20** | the real bar |
| Main opt | 20–30 | **stretch — contingent on the Phase 0(b) hit-rate curve** |
| Stretch | 30–40 | only if locality is unusually favorable |
| 60+ | — | not a commitment |

**Honest anchor:** decode is I/O-bound. Compute floor ≈ 35 ms/token (from Ivan's
28.82 tok/s @8K *resident*, §3). All-miss routed read ≈ 2.22 GiB/token
(142.38 GiB × 6/384) ÷ ~5 GB/s SSD. So **20 tok/s needs ~97% expert-cache hit,
15 tok/s ~93%, 10 tok/s ~85%** — from a cache holding ~21–28% of experts. The
likely band is **10–15 tok/s**; 20–30 is not assumed, it is earned only if
Phase 0(b) shows the locality supports it.

## 3. Measured ground truth (Phase 0a — do not re-derive)

GGUF header of `DeepSeek-V4.1-Flash-Q2.gguf` parsed via HTTP range-read (no full
download). GGUF v3, 1046 tensors, arch `deepseek41`, 40 layers, 384 routed
experts, top-6, alignment 16384.

| Class | Tensors | Size | Fate on 64 GB |
| --- | ---: | ---: | --- |
| **Non-routed RESIDENT FLOOR** | 918 | **8.79 GiB** | always resident |
| Routed experts | 120 (40×3) | 142.38 GiB | SSD-streamed, cached |
| Engram | ~8 | ~189 GiB | disk-only, read per token |

- Cross-check: routed + non-routed = 151.2 GiB ≈ documented 151.77 GiB "main". ✓
- **Verdict: WILL load and run on 64 GB.** Budget `64 − 8.8 floor − ~10 OS/apps
  − ~4 Metal/graph − ~3 KV(MLA-compressed)` ⇒ **expert cache ≈ 30–40 GiB
  (~21–28 % of routed experts resident).**
- **Routing is cache-hostile:** config `topk_method = noaux_tc` = DeepSeek
  aux-loss-free **load-balanced** routing → deliberately flat expert usage →
  works against frequency caching. This is why Phase 0(b) is mandatory.
- **Compute ceiling reference** (upstream `4f9a2e0`, `ds41f-nondspark-optimizations`,
  M3 Ultra 512 GiB, Q4, **resident, no streaming**): decode 28.82 tok/s @8K,
  23.09 @256K (+58.3 % geomean over `main`). This is the compute half we inherit
  by rebasing; the 64 GB **streamed I/O** half is unmeasured upstream — our niche.

## 4. HARD RULE #1 — do not break Qwen3.8

- Never rewrite / degrade the Qwen3.8 implementation, its quant recipe, or its
  Qwen-specific Metal kernels.
- **Qwen regression gate** (`speed-bench/qwen-regression/`) must have a baseline
  before touching any shared runtime code, and must pass on **every** commit that
  touches: pager, cache, memory, KV, SSD, Metal, GGUF loader, generic runtime.
- **If Qwen regression fails → STOP DS4.1 work.** No "fix later".

## 5. HARD RULE #2 — track upstream, do not greenfield

Fork chain: **`antirez/ds4`** → **`ivanfioravanti/ds4-metal`** (git remote
`upstream`) → us. V4.1 **already exists and runs on Metal**
(`ds4_engine_is_deepseek41`, full `deepseek41.*` config, Metal + vision kernels,
DSA indexer). Ivan is actively optimizing it (`upstream/ds41f-optimizations`,
`upstream/ds41f-nondspark-optimizations`, `upstream/ds41f-dspark`).

- **Do NOT create parallel `ds4_v41_*.c`** that duplicate the working `deepseek41`
  path. Extend it; keep the 3-layer separation (§6).
- **Rebase onto `upstream/ds41f-nondspark-optimizations`** to inherit compute
  wins (async decode-layer queueing, parallel Engram reads, BF16 fuse). Re-base
  continuously — Ivan commits daily; divergence itself violates Rule #1.
- Ivan's V4.1 work targets **big machines (M3 Ultra / TP)**. **The 64 GB
  single-stream streaming profile is unclaimed — that is our contribution.**

## 6. Code architecture — three layers

- **Generic (shared, Qwen-safe):** expert cache, SSD `pread`, async worker, LRU
  /frequency/recency, memory accounting, KV allocator primitives, benchmark
  instrumentation. No `if (model == QWEN) … if (model == DS41) …` sprawl.
- **Qwen-specific (frozen):** Qwen graph, expert layout, tensor map, Metal
  kernels, quant recipe, MTP, context behavior.
- **DS4.1-specific (extend existing `deepseek41` code):** router, expert layout,
  memory/cache policy for the 64 GB budget. Do not put Qwen constants into it,
  nor DS4.1 assumptions into the generic layer.

## 7. Phase 0 — GO/NO-GO gate

- **0(a) DONE** — resident floor = 8.79 GiB → load feasible (§3).
- **0(b) TODO (blocks all optimization):** download `ds41f-q2`; on a real
  agent trace, measure **expert-activation locality → hit-rate vs cache-size
  curve** at ~30–40 GiB budgets, plus **Engram bytes/token**. Decision:
  - ≥93 % hit at ~35 GiB → target 15+ tok/s, proceed.
  - ~85 % → target ~10, proceed with tempered expectations.
  - <80 % (flat routing dominates) → re-scope or abandon the fast-target.

## 8. Work plan (consolidate existing + build the net-new 64 GB profile)

Ordered; each step ends with V4.1 tests **and** Qwen regression.

1. **Rebase** onto `ds41f-nondspark-optimizations`; stand up `qwen-regression/`
   baseline; confirm Qwen bit-identical.
2. **Phase 0(b)** locality measurement (§7) → set the real target.
3. **64 GB memory manager** (net-new): recompute budget per context
   (4K…256K), never hard-code; charge resident span, not payload.
4. **Router-first selective I/O** (partly exists — tune): read only the top-6
   experts the router selects; bulk-unique reads in prefill; KPI
   `SSD bytes read ≈ SSD bytes required`.
5. **Expert cache** for the small budget: `(layer, expert_id, kind)` key, O(1),
   LRU first, then frequency+recency / prediction-aware — chosen by data.
6. **Async SSD + Engram parallel reads** (inherit from `4f9a2e0`, tune):
   overlap within the router data-dependency limit (within-layer + speculative
   only; no free layer N→N+1 prefetch). Fold **Engram bytes/token** into the
   budget from the start.
7. **Paged KV adapter** for MLA-compressed KV (port primitives from scallop; do
   not assume Qwen KV == V4.1 KV). Validate paged vs non-paged logit agreement.
8. **Metal opt** only after I/O is right; generic/shared kernels only, then Qwen
   regression.
9. **Predictive prefetch** — must not change exact expert selection; keep only
   if net tok/s rises without inflating SSD traffic.
10. **DSpark / MTP last** — SSD-bound workloads may not benefit; measure
    `SSD bytes / accepted token`.
11. **Agent benchmark** (Claude Code / Codex real tasks) — wall-clock is the
    final metric.

## 9. KPIs (record every benchmark)

tok/s (prefill/decode), TTFT · **SSD GB/token (deciding KPI)**, read ops,
read latency · expert cache hit/miss % · resident/peak RAM, cache size, KV size,
workspace · GPU/CPU util · router/MoE/attn/Engram/I-O-wait time breakdown.

## 10. Non-negotiables

1. Never merge `scallop` wholesale into `main`; never break/alter Qwen3.8.
2. Every shared-runtime change passes Qwen regression; if it fails, STOP.
3. Track `upstream/ds41f-*`; extend the existing `deepseek41` path, don't fork
   parallel `ds4_v41_*.c`.
4. Exact router-first expert selection is mandatory; minimize SSD bytes/token;
   never read all experts when the router determines the subset.
5. No precision drop merely to shrink size; no perf claim without benchmark data.
6. Optimize SSD routing/cache before DSpark/MTP.
7. Targets: ≥10 floor, 15–20 production, 20–30 earned-not-assumed. 10–20 is not
   the final target — if stuck <10 after exact routing + cache + async + Metal,
   re-profile.
