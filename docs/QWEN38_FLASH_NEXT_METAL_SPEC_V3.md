# QWEN3.8-FLASH-NEXT-METAL — SPEC V3 (current-state → gap → ceiling closed)

Status: 2026-09-22. Supersedes V2 (greenfield). V2's fatal premise — "build a new runtime,
use the existing V4.1/Qwen Metal impl as external reference" — is wrong: **that existing impl IS
our own ds4-metal** (worktree exppld, lineage antirez→ivan→dongnh). We are not at Phase 0; we are at
a validated runtime with a MEASURED ceiling. This spec is written from where we actually stand.

## 0. Mission (unchanged)
Native Apple-Silicon runtime for Qwen3.8-Flash-Next on M5 Pro / 64 GB / 1 TB SSD, tuned for the
agentic-coding workload (Claude Code / Codex / OpenCode). Not a general GGUF runner.

## 1. CURRENT STATE — what already exists (do NOT rebuild)
The runtime already RUNS Qwen3.8-Flash-Next end-to-end, byte-exact, with:
- Full arch: 48 layers (36 GDN + 12 QSA), 512 experts / 10 routed + 1 shared, 4-branch gated
  residual, N-gram/PLE (external Q4_1 sidecar, CPU/demand-paged via `--ple`), 1 MTP layer.
- Quant: IQ2_XXS gate/up + padded Q2_K down (PROD ~40.67 GiB resident); SelQ4 31L tier (mixed
  Q2_K L0-17 / Q4_K L18-47, ~48-56 GiB) = strongest quality.
- FP8 KV (256K native usable on 31L), compressed-KV, QSA sparse attention (512 blocks / 2048 tok).
- MTP speculative decode; span-verify + context-copy speculation (shipped PROD as opt-in, byte-exact).
- SSD expert streaming (`--ssd-streaming`, bounded wired cache) = memory-fit mode, byte-exact.
- Measurement stack already built: byte-exact A/B gate, GPU-timestamp profilers
  (STAGE_TS/MOE_TS/TOPK, exp-pld 28f709a), router logger (DS4_QWEN4_ROUTER_LOG, d07c49a),
  wired-memory (vm_stat) harness. **V2 Phases 0-8 are essentially done.**

## 2. MEASURED PERFORMANCE (honest, from real bench logs — NOT targets)
Decode (generation) tok/s, M5 Pro 64 GB, ctx 8192:
| condition | tok/s | note |
|---|---|---|
| PEAK (copy-favorable prompt, copy-speculation) | 63.11 | MTP ~4x accept; special prompt, NOT representative |
| realcode single-run | ~43 | MTP ~72% accept |
| **sustained avg-3 (realcode/chat) PROD** | **~37** | the honest general number |
| sustained 31L (strongest quality) | ~33 | full-resident |
| memory-mode stream 8GB (31L) | ~18-22 | RAM 34.4 GiB, byte-exact |
Memory: PROD ~48.3 GiB planned (model 40.67 + buffers 7.37 + KV 0.26); 31L ~48-56 GiB.
Prefill: 200-820 t/s depending on chunk/ctx (not the bottleneck).

## 3. CEILING — CLOSED WITH EVIDENCE
SPEED campaign closed earlier: decode ceiling is **MoE IQ2 dequant + T=1 (M=1) execution**
(MoE 40.5% decode: mid IQ2_XXS 3.03x over byte-floor, down Q2_K 2.56x; attention only 19%).
The 60 t/s **sustained/general** target: both — and the only two — ceiling-break levers are now
empirically eliminated (offline gates, 2026-09-22):
- **(A) new cheaper-dequant expert format + kernel — NO-GO.** The ~3x overhead is the M=1
  occupancy floor, not the codebook: Q2_K is already codebook-free yet still 2.56x. A format change
  can't drop below that, and cheaper formats cost more bytes. Best case (implausible) 1.18x → 57.8.
- **(B) token-aggregation (M>1 via speculation) — NO-GO.** Real routing (5000 tok, 6 workloads):
  at realistic draft depth K=2-4, 65-78% of dispatched experts are still M=1 (scatter-dominated);
  effective_M only 1.2-1.55 → ~1.1-1.17x → clears 60 for no baseline. Only K≥8 + the most optimistic
  PROD baseline reaches 60, at high CB-rewrite risk and against acceptance decay.
=> **60 t/s sustained is architecturally out of reach on M5 Pro for this model.**

## 4. PERFORMANCE TARGETS (revised)
- **P0:** stable on 64 GB, no pathological swap, correctness parity (byte-exact vs full-resident),
  sustained decode as measured (~33-37 t/s) — no regression.
- **P1:** HOLD the measured ceiling (PROD ~37 / 31L ~33), byte-exact, across ctx 8K-256K and the
  agentic-coding workload. (Replaces the old "≥60 t/s".)
- **P2 (peak, workload-specific, NOT a sustained claim):** exploit context-copy speculation where the
  output reuses context (agentic edits) — already reaches 60+ (63.11 measured) on such content.
  Re-measure honestly: avg-3 on real code showed it ~neutral, so treat the 63 as a special-prompt
  peak, not a general win.
- 60 t/s general = PROVEN-unreachable; any doc must say so and name the bottleneck, never game a prompt.

## 5. THE REAL GAP (what is actually left to do — not "build a runtime")
1. **Memory-fit mode polish** (the banked deliverable): `--ssd-streaming` byte-exact, tune the
   RAM/tok-s curve; goal = run where full-resident OOMs (long ctx, 31L on tight RAM), accept the
   tok/s trade. This is the honest SCALE win.
2. **Quant quality gates:** extend the byte-exact + coherence + needle suite; no harmful-compliance
   evals. Keep gate/up vs down quant independent.
3. **Long-ctx coordination:** one MemoryManager arbitrating KV + PLE cache + expert cache +
   prefetch so 256K doesn't starve paging (P1 8K-256K; 512K/1M are P2, not requirements).
4. **Agentic-workload robustness:** prefix caching + context-copy speculation measured on Claude
   Code / Codex / OpenCode traces, TTFT + sustained TPOT reported separately from prefill.

## 6. WHAT V2 GOT WRONG — dropped
- ❌ Rebuild runtime from Phase 0 (throws away validated, byte-exact work).
- ❌ Greenfield "build the measurement system first" — already built; bottleneck already found.
- ❌ Autonomous Coordinator/7-Workers/Critic + GPU_LOCK harness — over-engineered for 1 dev + Claude.
- ❌ "don't break the separate Qwen runtime" / new `~/qwen38-flash-next/` namespace — phantom: ds4-metal
  IS the Qwen runtime. Real rule below.
- ❌ "≥60 t/s at all costs" — proven unreachable general; keep as peak-only, honestly labeled.

## 7. CONSTRAINTS (in force)
Engine edits/builds/tests in worktree **exppld ONLY**; never port to PROD, never start the gateway.
All new decode behavior behind a flag (unset = byte-identical, full residency). Byte-exact A/B is the
correctness gate. Chat VI; code/doc/commit EN + attribution. Model artifacts → HuggingFace, never git.

## 8. DEFINITION OF DONE (reframed)
[✓] runs Qwen3.8-Flash-Next byte-exact on 64 GB   [✓] arch/quant validated (PROD + 31L)
[✓] profilers + byte-exact gate + router logger    [✓] memory-fit streaming byte-exact
[✓] MTP + span/context-copy speculation (opt-in)   [ ] MemoryManager unifying KV/PLE/expert paging
[ ] agentic-trace bench (CC/Codex/OpenCode) TTFT+TPOT   [ ] 256K long-ctx coordination gate
[×] ≥60 t/s sustained general — PROVEN UNREACHABLE (levers A+B empirically eliminated); peak 63.11
    achieved only on copy-favorable content.

## 9. If anyone still wants >60 sustained
It requires leaving this platform/model combo, not a rewrite: either (a) a fundamentally different
expert execution model that raises M without speculation (not available for causal interactive
decode), or (b) higher-bandwidth hardware (3090 ≈ 940 GB/s + tensor cores vs M5 unified memory).
Both are new projects, and the offline gates say the software path on M5 is closed.
