# Paged/FP8 KV + Memory Manager — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paged KV cache, optional FP8 KV, and a unified memory budget manager (KV ↔ expert cache ↔ Metal workspace ↔ OS headroom) for the Qwen3.8 `qwen4exp` DS4-IQ2 build, so 256K+ context runs without OS swap and the Plan-3 expert pager and KV pages share one accountable budget.

**Architecture:** Paged KV is a Metal-side page table (logical position → physical page) in front of the *unchanged attention math*; FP8 KV is a precision switch on the page storage, independently enable-able; the memory manager is an extension of the existing `ds4_ssd_auto_cache_plan` accounting (`ds4_ssd.c:108`) generalized to a 4-way budget. All work is `DS4_HAS_QWEN4_METAL`-guarded.

**Tech Stack:** C (`ds4.c`, `ds4_metal.m`), new `ds4_kv_cache.{h,c}` + `ds4_memory_manager.{h,c}` modules, `ds4-bench`, logit fixtures.

**Spec:** `DS4-IQ2 Fast+ Final Engineering Specification` §8-§11 (Phases 6-8), executed in global order **after** Plan 3 (expert cache) and **before** the kernel/MTP campaign's MTP tuning.

## Global Constraints

- **Model artifact:** every run in this plan uses the Plan-1 final build `gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-MTP-NNgram.gguf` (self-contained n-grams, sha head `ed238d8d…` — record the full sha256 in Task 0 receipts), not Ivan's upstream GGUF — same reasoning as Plan 3 Task 2 Step 1: lock the artifact first.
- **Context ceiling:** native ceiling is 262144 tokens; beyond 256K requires `DS4_QWEN4_YARN_FACTOR` — Task 3's 320K/512K probes must run with it and record the factor used; 256K and below need no YARN.
- **Strict order (spec §25):** Plan 3 stage G must be done (or at least stages B–E with numbers) before starting here; its memory-pressure receipts are the input to Task 0. Do not start FP8 KV before paged-KV numbers exist (spec §25: "Do not skip directly to DeepSeek-style compressed KV").
- **Attention semantics are frozen:** paged KV changes *where* K/V bytes live and their *precision path*; it must not change what attention computes. Gate: `test_qwen4_logit_dump.py` bit-identical for BF16 paged vs BF16 resident; FP8 has its own quality gate (Task 5), not the bit-identical one.
- **DeepSeek compatibility is out (spec §10):** the DS4-Q4/DeepSeek KV layouts, compressor frontiers, DSpark capture paths are different architectures. This plan adds nothing to them; the new modules are qwen4-only. CUDA/ROCm builds exclude qwen4 (`bab2029`) so no shared-kernel regression is possible; any touch to `ds4_gpu.h` shared structs → ASK the user (AGENT.md CUDA/distributed gate).
- **No new user-facing flags beyond the existing family:** paged KV reuses the `--ctx`/`--kv-cache-boundary-align-tokens` conventions; new behavior is enabled by extending `--ssd-streaming` semantics (paged-KV pages participate in the same budget) — if a genuinely new flag becomes unavoidable, stop and present it to the user first (AGENT.md: "Do not add permanent semantic variants behind flags").
- **Quality gate for FP8 (spec §9, fixed):** BF16-KV vs FP8-KV compare on: greedy token agreement over ≥5 long-context retrieval prompts, coding task, tool-use task, multi-turn conversation, NLL on the `ds4-eval` core suite. FP8 is accepted only if degradation is negligible **and** tok/s at 256K+ justifies the switch. "Looks okay" is not a result.

---

## Task 0: Memory receipts handoff + baseline for KV experiments

**Files:** `speed-bench/orca-iq2-kv-baseline.csv`

**Interfaces:** Consumes Plan 3 Task 7 campaign receipts (per-ctx RAM/swap/KV-bytes/expert-cache-bytes at the 5-point sweep). Produces: the KV baseline (current monolithic KV, BF16) numbers at 32K/64K/128K/220K/256K-probe with the same provenance receipt format (git commit, model sha256, ctx, token-sequence checksum, swap peak).

- [ ] **Step 1: Record current KV cost** — the engine prints planned KV bytes at load; capture per-context point from Plan 3 receipts + a fresh 256K probe (record failure mode if it swaps/OOMs: that is the number this plan attacks).
- [ ] **Step 2: Bench + commit CSV** (`stage=kv-0`).

## Task 1: `ds4_kv_cache.{h,c}` — paged KV, BF16, page sizes 256/512/1024

**Files:** Create `ds4_kv_cache.h/.c` (qwen4-only module, ~400 lines: page allocator, page table `logical_pos → page_id`, per-layer K/V pages); Modify `ds4.c` qwen4 attention hooks (`ds4.c:58099-58130` GEMV sequence + GDN/QSA attention calls at the `qwen4_graph_attention` site) to address K/V through the page table; `metal/qwen4.metal` `kernel_qwen4_attn_decode`/`_merge` (`:1733/:1820`) gain a page-table indirection argument (attention **math** unchanged).

**Interfaces:**
- Consumes: Task 0 receipts (to pick default page size); Metal buffer APIs in `ds4_metal.m`.
- Produces: `ds4_kv_paged_init(ctx_tokens)`, `ds4_kv_paged_grow(pos)`; the 3 page sizes behind ONE env knob `DS4_QWEN4_KV_PAGE_TOKENS` (diagnostic, default 512) until benchmarked.

Key design (locked): pages are fixed-size GPU buffers; attention kernels already loop over KV blocks — replace the contiguous-pointer arithmetic with a per-page pointer array (one `uint32` page-base + offset). No copy-on-grow: grow = allocate new page, never move old ones (this is the whole point vs monolithic reallocation at 220K→256K).

- [ ] **Step 1: Module + page allocator** (CPU-visible page table; GPU gets the base-pointer array; unit test `tests/test_kv_paged.c`: allocation order, grow, per-layer offsets for 3 page sizes).
- [ ] **Step 2: Wire qwen4 attention** — GDN recurrent state and QSA KV both address through the page table; shared-expert/dense paths untouched.
- [ ] **Step 3: Gate — bit-identical** (BF16 paged vs resident):
  ```bash
  make test
  python3 tests/test_qwen4_logit_dump.py --model <GGUF>   # resident, before
  DS4_QWEN4_KV_PAGED=1 python3 tests/test_qwen4_logit_dump.py --model <GGUF>  # paged
  # both must print the same max-diff PASS line and argmax agreement
  ```
- [ ] **Step 4: Bench all 3 page sizes × 5 ctx points** (`stage=kv-256/512/1024`): tok/s, peak RAM, swap, KV bytes. Choose default from data; record losers in CSV.
- [ ] **Step 5: 256K probe** — paged KV must hit 256K where Task 0's resident baseline failed (or fails less badly); record the receipt.
- [ ] **Step 6: Commit** module + tests + CSV.

## Task 2: FP8 KV storage + dequant path (independent switch)

**Files:** `ds4_kv_cache.{h,c}` (FP8 page variant: E4M3 or E5M2 chosen by measured K/V dynamic range in Task 1 receipts — record the measurement, don't guess; `ds4_metal.m` dequant-on-load into attention staging, or a dedicated FP8 attention input path if the kernels accept it cheaply — prefer the staging dequant first, kernel change only if profiling shows it pays).

**Interfaces:** Consumes Task 1 module + its quality baselines. Produces: `DS4_QWEN4_KV_FP8=1` diagnostic switch (off by default; env, not a user flag, per Global Constraints — if it proves to be the shipping feature, present the flag question to the user first).

- [ ] **Step 1: Measure K/V dynamic range** on the 5-point sweep (store K/V histograms into the logit-dump test harness, 20 lines) → pick E4M3 vs E5M2, commit the measurement.
- [ ] **Step 2: FP8 pages + dequant**; attention math on dequantized FP32 inputs (no kernel semantic change).
- [ ] **Step 3: Gate — quality matrix (spec §9), NOT bit-identical:**
  ```bash
  # per workload class × {BF16 KV, FP8 KV}: ds4-eval --suite core NLL,
  # 5 long-context retrieval prompts (needle at 32K/128K/220K), coding task,
  # tool-use task, 8-turn conversation; record greedy token-agreement %
  python3 gguf-tools/quality-testing/score_official <...>  # per that README's protocol
  ```
  Accept threshold: greedy agreement ≥ 99.5% on short/mid context AND retrieval still correct at 128K; below that, FP8 is **rejected** and the stage is recorded as failed (a rejected stage is a result).
- [ ] **Step 4: Bench FP8 vs BF16 at 220K/256K** (tok/s, KV bytes, total RAM) → `stage=kv-fp8` rows.
- [ ] **Step 5: Commit** with the full quality matrix in the commit message.

## Task 3: `ds4_memory_manager.{h,c}` — unified 4-way budget

**Files:** Create `ds4_memory_manager.h/.c` (~200 lines); generalize `ds4_ssd_auto_cache_plan` (`ds4_ssd.c:108`) into `ds4_mem_budget_plan(total_ram, kv_bytes(ctx), model_bytes, workspace_bytes, &out_expert_budget, &out_kv_page_budget, &out_headroom)`; wire at qwen4 engine init and at context grow (`--ctx` / auto-grow at rewind boundaries).

**Interfaces:** Consumes: Task 1 page allocator, Plan-3 pager cache budget. Produces: one budget printout at load (extending the existing `--ssd-streaming` startup report style) + the dynamic re-fit at every 32K context boundary (spec §11: "at 32K larger expert cache, at 220K smaller expert cache + larger KV" — the *existing* expert auto-budget already moves with pressure; this makes KV pages move with it too, through the same function).

- [ ] **Step 1: Budget function + load-time printout** (no behavior change yet; numbers only, eyeball vs actual).
- [ ] **Step 2: Dynamic re-fit at context boundaries** — expert-cache bytes and KV page pool are resized together; expert evictions that fit the new budget use Plan-3's LRU→scored ladder (call the pager's existing API, no new eviction logic here).
- [ ] **Step 3: Gate** — bit-identical logit dump at each ctx point **and** zero-swap check: the manager's job is "never let uncontrolled pressure reach OS swap" (spec §11/§18#10), so the acceptance test is `vm_stat` swap delta ≈ 0 across a 32K→256K grow session.
- [ ] **Step 4: Bench `stage=kv-full`** (paged + FP8-if-accepted + manager) across the 5-point sweep + 320K/512K probe with failure-mode receipts.
- [ ] **Step 5: Commit** module + tests + CSV; update `docs/ORCA_UNCENSORED_DS4_IQ2.md` section "Memory manager".

## Task 4: DeepSeek-style compressed KV — EXPERIMENTAL, separate spec

**Status: NOT in this plan (spec §11/§25.11).** This task exists so the order is explicit: it starts only after Task 3 numbers prove 256K+ works, AND only via a fresh spec (the DSpark/compressor architecture is a different attention design; porting "concepts" (selective retention, block tables) without the architecture is how this becomes a rewrite — AGENT.md forbids it). Placeholder gate: a written mini-spec + user approval, then its own plan. No code in this campaign.

---

## Self-Review

- **Spec coverage:** §8 paged KV → Task 1; §9 FP8 KV (+ quality matrix) → Task 2; §11 memory manager → Task 3; §10 "DeepSeek-inspired only" → Task 4 gated; §17 benchmark matrix rows (`Fast+PagedKV`, `Fast+FP8KV`, `Full Fast+`) → Tasks 1/2/3 CSV stage labels; §18 "no uncontrolled OS swap" → Task 3 Step 3.
- **Order check:** Plan 3 → this plan → Plan 2's MTP tuning (kernel stages of Plan 2 remain available to interleave only if the user re-orders; default is the spec §25 order).
- **Placeholder scan:** none; Task 4 is explicitly out of scope with a gate, not a TODO.
- **Known risks:** (a) QSA block-sparse attention (`kernel_qwen4_idx_select/_expand`, `metal/qwen4.metal:1403/1684`) is the trickiest page-table consumer — if its block selection assumes contiguous KV, Task 1 may need a per-page block bitmap; detect in Task 1 Step 2, report before forcing it. (b) FP8 may legitimately fail the quality gate — the plan treats that as a valid outcome with receipts, not as task failure.
