# DeepSeek-V4.1-Flash on M5 Pro 64 GB — Performance Build Spec (v2)

> **FROZEN 2026-09-25.** V4.1 is kept as a research and benchmark reference,
> not a production target. See §0 for the final state, the two closing spikes
> and the criteria for re-opening.

Build a **fast SSD-streamed, router-first** DeepSeek-V4.1-Flash Q2 runtime
profile for a single **M5 Pro / 64 GB / 1 TB SSD** box (single-stream agent
workload: Claude Code / Codex), **while treating the existing Qwen3.8
implementation as a hard regression-protected subsystem**. The speed target is
**set from Phase 0 measurements, not assumed** (decision "C", 2026-09-23).

v2 (2026-09-23) replaces v1 (2026-09-18). Changes: the expert cache is small
and bounded by wired memory, not "cache-heavy" (§3.1); per-token roofline
estimate and upstream anchors (§2); golden artifact = the pinned upstream Q2
(§3.3); concrete Qwen gate (§4); branch model updated to `develop` (§0);
correctness oracle (§9); target decided after Phase 0 (§7).

## Definition of Done (all five must hold)

1. `ds4 --ssd-streaming` loads V4.1-Flash Q2 on this box **zero-swap**, inside
   the machine wire limits (§1).
2. Decode reaches **≥ 20 t/s (band 5)** single-stream at 8K context on the
   Phase-0 `switch` workload, **without lowering model quality** (user
   decision 2026-09-24, option (c) of
   `speed-bench/v41/phase0-20260924/RESULTS.md`; measured start 9.04 t/s).
   Levers in order: lossless runtime work first (host gaps / per-layer sync,
   cache size and policy, pread overlap, Metal; §8 steps 3–6, worth up to
   ~19.6 t/s on paper), then speculative decoding with exact verification
   (DSpark/MTP, §8 step 8). Cheaper non-routed weights (§8 step 9) are a last
   resort and need the user's approval plus the §9 quality gate. Below 10 t/s
   after the cache, sync and Metal work → re-profile.
   *How it is measured:* `ds4-bench --ssd-streaming` on the Phase-0 `switch`
   prompt, ctx 8192, 512 generated tokens, cache target recorded, warm file
   cache (not the first run after boot; nothing else using memory), the
   Phase-0 diagnostics env (profiling on, as for the 9.04 start), interleaved
   A/B against the previous build. Runtime-only levers are compared with `--teacher-forced-decode`,
   like the 9.04 t/s start. Speculative decoding is judged free-running (greedy),
   because teacher forcing changes acceptance. That result is reported next to
   the teacher-forced rate of the same build.
3. Passes **ds4-eval core**, a real **agent smoke**, and the official-API
   quality fixtures (§9).
4. **Qwen3.8 regression green** (§4).
5. Reproducible: pinned GGUF SHA256, pinned commit, recorded bench.

Non-goals: big-machine / TP (Ivan's lane); rebuilding the Q2 GGUF ourselves
(§3.3); 1M context before 256K is stable.

## 0. Status

- **FROZEN 2026-09-25 (user decision).** Effort moves back to Qwen3.8-Flash-Next
  (PROD: 51.6 GiB, about 35-44 t/s, 256K context, zero swap).
  - **Final state:** `develop` `2a910dc`. Shipped defaults decode at 12.1 t/s
    (ctx 8K; auto cache, queued layers, async load, #1042 fusions, lookahead
    k = 1), bit-exact against the Phase-0 path.
  - **Why:**
    - 20 t/s needs the miss path hidden, about 10 ms less GPU time, and
      MTP/DSpark.
    - No measured quality edge over Qwen on the agent workload.
    - The GGUF takes 341 GiB of the 1 TB disk.
  - **Spike 1, smaller routed experts** (CPU cache simulator on 1100-token
    router logs, byte budget of today's 28.5 GiB decode cache):
    - The artifact is 188.8 GiB Engram (SSD-only, about 0.6 ms/token), 142.4
      GiB routed experts and 9.4 GiB resident.
    - All-IQ1_M experts are still about 111 GiB, all-IQ1_S about 99 GiB, so
      decode keeps streaming.
    - Predicted decode: IQ1_M or ternary 13.2-14.1 t/s; IQ1_S 13.6-15.0; hot
      experts IQ2 and the rest ternary 12.7-13.8.
    - Hotness ranked on one workload covers another's selections almost at
      random.
    - Oracles from the measured terms:
      - no miss on the critical path: 16.6 t/s;
      - every miss already in the page cache: 13.6;
      - GPU at the 38.4 ms byte floor: 15.6;
      - no miss and the GPU at the byte floor: 23.9.
  - **Spike 2, approach-2 staging by top-k:**
    - A layer with at least one miss costs about 1.055 ms whatever its miss
      count, because the pread pool reads in parallel.
    - Top-1 to top-6 staging covers every miss of 31-66 % of those layers.
      That gives 13.2-14.8 t/s if every staged read lands in time. With the
      measured 10 GB/s random SSD reads and a ~1.27 ms window per layer, it
      gives 12.5 t/s at top-1, falling to 4.6 at top-6 (wasted reads).
    - A perfect one-layer predictor reaches 16.4 t/s ideal, 13.8 when bound by
      the SSD. Layer 0 is never predictable.
    - An infinite cache removes 76 % of the layers with a miss, so misses are
      mostly a capacity problem.
    - Letting decode use the idle 7.12 GiB prefill reserve gives 12.7-12.8
      t/s; a 40 GiB cache gives 13.0-13.1.
  - The spike simulators were throwaway and are not in the repo.
  - **GGUF deleted from disk 2026-09-25.** Re-download it with
    `./download_model.sh ds41f-q2`; its SHA-256 is pinned in
    `speed-bench/v41/phase0-20260924/RESULTS.md`.
  - **Re-open only if:**
    - V4.1 gets MTP/DSpark with measured acceptance and a real speedup;
    - upstream hides most of the streaming miss path;
    - DS4.1 measurably beats Qwen on the agent tasks.
  - **Levers in order, if re-opened:** decode use of the prefill reserve or a
    40 GiB cache, top-1 staging into cache slots, GPU MoE and attention
    kernels, then MTP/DSpark.
- **Upstream sync done 2026-09-23** on `feature/ds4.1-flash`: `ec56a05` merges
  `antirez/main` 0aaea5a (25 commits), `d3bf293` merges Ivan's
  `ds41f-nondspark-optimizations` 4f9a2e0. Ivan's `main` had nothing newer
  than 8db1d1d. `make all` is clean. Tests done 2026-09-24: the sync's unit
  targets pass and the Qwen full tier is green against the PROD baseline
  (paired A/B speed check, `speed-bench/qwen-regression/`). The Engram
  resolution in `d3bf293` passes the 64 GB streaming exact check
  (`--stream-control ... DS4_METAL_DISABLE_V41_ENGRAM_PARALLEL 8`, 2026-09-24).
- Static review of the sync: the Qwen compute path is untouched
  (`metal/qwen4.metal` unchanged; shared kernels and host functions only gained
  parameters that default to the old behavior; the new softplus series only
  reaches the DS4/V4.1/GLM router). Qwen-visible change: `d31089d` fixes Qwen
  tool-content streaming in `ds4_server.c`.
- **Disk:** the Q2 GGUF (341 GiB) was downloaded to
  `~/orca/workspaces/ds4-metal-data/gguf/` (deleted at the freeze, 2026-09-25). About 65 GiB stays free (the unc31
  Qwen GGUF was removed; it is on HF). Keep ≥ 40 GiB free: macOS swap files
  live on the same disk.
- **Phase 0(b) DONE 2026-09-24:** Qwen baseline and full tier green; Q2
  downloaded and pinned; 16 clean runs; best 9.04 t/s (ctx 8K, 24 GB cache).
  Results and the measured roofline: `speed-bench/v41/phase0-20260924/RESULTS.md`.
  Target chosen: ≥ 20 t/s without quality loss (DoD 2).
- **Sub-project 1 (streaming decode pipeline) DONE 2026-09-24: 11.12 t/s**
  at ctx 8K with the shipped defaults (x1.2336 over the Phase-0
  configuration, bit-exact; the pipeline code alone is about x1.11, the rest
  is the auto cache vs Phase 0's 24 GB point). Decode layers stay queued under streaming and
  missed experts load on the async worker. Readahead stays on, the V4.1 split
  threshold stays at 3, and the auto cache is unchanged. The >= 12 t/s
  sub-project target was not met.
  Results: `speed-bench/v41/pipeline-20260924/RESULTS.md`.
- **Lookahead prefetch into the page cache, 2026-09-25:**
  - Layer L+1's router runs on layer L's FFN input.
  - The first uncached guess per layer gets `F_RDADVISE`.
  - Gain: ×1.08-1.09 at ctx 8K, bit-exact.
  - Shipped defaults: 12.00 t/s mean over 6 runs (11.48-12.29), against 8.89 for the Phase-0 configuration.
  - Results: `speed-bench/v41/lookahead-20260925/RESULTS.md`.
- **Branches:** work on `feature/ds4.1-flash` (off `develop`), merge to
  `develop`, deploy only via `prod/<feature>-YYYYMMDD` cut from `develop`
  (`deploy-ai-gateway.sh`). v1's `main` / `scallop` wording is retired:
  scallop was merged into `develop` and deleted on 2026-09-23.

## 1. Hardware & model target

| | |
| --- | --- |
| Machine | Apple M5 Pro, 64 GB unified memory, ~1 TB SSD, macOS, Metal |
| Wire limits | `vm.user_wire_limit` = 52.48 GiB (mlock'd cache counts), `iogpu.wired_limit_mb` = 57344 |
| Measured | memory-bound Q8_0 gemv ≈ 290 GB/s; random expert reads from SSD ≈ 10 GB/s |
| Sharing | one large model at a time; the PROD gateway wires ~45–51 GiB when up |
| Workload | single-stream agent (Claude Code / Codex) |
| Model | DeepSeek-V4.1-Flash calibrated **Q2** (`./download_model.sh ds41f-q2`), Engram disk-only |
| Context rollout | 4K → 8K → 32K → 128K → 256K; 1M only after 256K is stable |

## 2. Performance: roofline, anchors, bands

### 2.1 Per-token decode budget (estimate, with the Phase 0 and Phase 1 measurements)

| Term | Size / token | Time / token (estimate) | Measured (Phase 0, ctx 8K, 24 GB) | Measured (Phase 1, ctx 8K, auto cache) |
| --- | --- | --- | --- | --- |
| Non-routed resident weights (8.79 GiB floor − 1.23 GiB `token_embd`) | ~7.6 GiB (measured 8.14: +0.59 GiB Engram projections) | ≥ 28 ms at 290 GB/s | byte floor 30.2 ms | — |
| Routed experts actually used (142.38 GiB × 6/384) | 2.22 GiB | ≥ 8 ms | byte floor 8.2 ms; GPU busy for both rows 57.0 ms (1.48× floor) | GPU busy 57.0 ms |
| Router → load host sync, 40 layers × ~0.3 ms (Qwen measured, pre-gate) | — | ~12 ms | other host work 19.5 ms (residual) | 4.0 ms (queued layers, async load) |
| Misses at ~10 GB/s: 98 % hit / 93 % hit | ~45 / ~160 MiB | ~5 / ~17 ms | miss path 34.1 ms at 0.796 hit: `F_RDADVISE` readahead 17.5 + pread 16.6, 465 MiB/token from a warm OS file cache (27 GiB/s) | 28.7 ms at 0.87 hit: readahead 15.5 + pread 13.2 (worker time included) |
| Engram: 48 random 264-byte rows, parallel reads | tiny | ~1 ms | 0.6 ms | 0.6 ms |
| **Total** | | 54–66 ms | **111.1 ms/token (bench 9.04 t/s)** | **90.3 ms/token (11.12 t/s)** |

Ideal total ≈ 54 ms at 98 % hit (**≈ 19 t/s**) or ≈ 66 ms at 93 % (≈ 15 t/s).
Real kernels do not run at the byte floor (the Qwen MoE runs ~3.5× over it;
Q4_K reaches 56–63 % of peak), so the **realistic band is ~10–13 t/s**. About
three quarters of the bytes per token are **resident non-routed weights, not
SSD traffic**. Past ~19 t/s you need a lever above this roofline:
speculative decoding (§8 step 8) or cheaper non-routed weights (§8 step 9).

### 2.2 Upstream anchors (measured)

| Configuration | Decode | Source |
| --- | ---: | --- |
| V4-Flash Q2, **M5 Pro 64 GB**, SSD, 8 GB cache | 11.37 t/s | antirez/ds4 #810 |
| V4-Flash Q2, M5 Pro 64 GB, SSD, 32 GB cache | 8.1–9.5 t/s | antirez/ds4 #810 |
| V4.1 (quant not stated), M5 Max, SSD, warm | 13.39 t/s | `077a257` |
| V4.1 Q2, M3 Ultra, resident, 131K ctx | 21.06 t/s | `6c00e2d` |
| V4.1 Q4, M3 Ultra, resident, 8K ctx | 28.82 t/s | `4f9a2e0` |

### 2.3 Reporting bands

| Band | t/s | Reading |
| --- | --- | --- |
| 1 | < 5 | architecture problem |
| 2 | 5–10 | functional, not yet acceptable |
| 3 | 10–15 | good baseline, keep optimizing |
| 4 | 15–20 | production-class |
| 5 | 20–30 | needs a lever above the §2.1 roofline |
| 6 | 30–40 | stretch |

The target (DoD 2) is picked from these bands at the end of Phase 0.

## 3. Measured ground truth (Phase 0a — do not re-derive)

GGUF header of `DeepSeek-V4.1-Flash-Q2.gguf` parsed via HTTP range reads (no
full download). GGUF v3, 1046 tensors, arch `deepseek41`, 40 layers, 384
routed experts, top-6, alignment 16384.

| Class | Tensors | Size | Fate on 64 GB |
| --- | ---: | ---: | --- |
| **Non-routed resident floor** | 918 | **8.79 GiB** | always resident |
| Routed experts | 120 (40×3) | 142.38 GiB | SSD-streamed, cached |
| Engram | ~8 | ~189 GiB | disk-only, read per token |

Cross-check: routed + non-routed = 151.2 GiB ≈ the documented 151.77 GiB main.
**It will load and run on 64 GB.** Phase 0(b) byte accounting of the
downloaded file puts the floor at **9.38 GiB**: the table above left out
0.59 GiB of Engram projection weights. 9.38 + 142.38 = 151.76 GiB matches the
documented main size (`speed-bench/v41/phase0-20260924/bytes.json`).

### 3.1 Correction: the expert cache is small

v1 budgeted a 30–40 GiB expert cache. On Metal that is **measured-wrong**.
On this exact hardware, antirez/ds4 #810 measured 8 GB → 11.37 t/s, 16 GB →
11.71, and 32 GB → 8.1–9.5 t/s. The cause (#638): Metal wires the whole
mmap-backed model view on first GPU use, so more mapped expert views means
more wired memory and more pressure. Consequences:

- Start at **8–16 GB** and pick the size by a sweep. Do not maximize hit rate.
  *Phase 0(b) (2026-09-24) on V4.1 is consistent with #810 up to 16/24 GB:*
  decode rose monotonically 4 → 24 GB (5.6 → 9.0 t/s at 8K). 32 GB, where #810
  fell, is untested. Wired memory grew only by the cache size, so there was no
  #638 mmap-wiring penalty. The target also includes a 7.12 GiB prefill reserve
  (8 GB = 95 experts). A bigger wired cache leaves less OS file cache for the
  misses, so measure 32 GB before assuming it helps (§8 step 3).
- Load experts with `pread` into bounded slot buffers. **Never** bind
  mmap-backed expert views.
- The constraint is **wired memory**. Measure it with `vm_stat` "Pages wired
  down" system-wide. RSS and `phys_footprint` miss Metal residency.

### 3.2 Routing is load-balanced

`topk_method = noaux_tc` is DeepSeek's aux-loss-free, **load-balanced**
routing. Expert usage is deliberately flat, which works against caching. That
is why the locality curve in Phase 0(b) is mandatory.

### 3.3 The golden artifact is the pinned upstream Q2

`gguf-tools/deepseek41_quantize.py` recipe `q2` produced the published file:

- IQ2_XXS routed gate/up, Q2_K routed down.
- Q8_0 attention / shared / head.
- F32 router, norms and mHC.
- Engram raw F8_E4M3 + E8M0 rows; `engram_kv` in F16.
- Calibrated.

Record its SHA256 at download (`download_model.sh` verifies it).

Rebuilding needs the source weights plus 341 GiB per variant, which does not
fit next to the golden. Quant research (tensor repair, imatrix variants, a
sensitivity map) is an **optional separate track**, opened only if §9 shows a
quality gap.

## 4. HARD RULE #1 — do not break Qwen3.8

- Never rewrite or degrade the Qwen3.8 implementation, its quant recipe, or
  its Qwen-specific Metal kernels.
- **Fast tier:** runs on every commit that touches shared runtime code (pager,
  cache, memory, KV, SSD, Metal, GGUF loader, generic runtime).
  - `make test-qwen4-kernels test-qwen4-q2 test-qwen4-moe-mm-specialize`.
  - PROD smoke **byte-identical** to the recorded reference
    (`deploy-ai-gateway.sh smoke --ref`).
- **Full tier:** runs at the end of every phase, before merging to `develop`.
  PROD config: unc31, K=36, FP8 KV, MTP, 64K draft vocab, `-c 229376`.
  - Output byte-identical.
  - Decode ≥ 97 % of baseline (median of 3).
  - Steady wired ≤ baseline + 0.5 GiB (`vm_stat`).
  - 229K needle HIT.
- Record the baseline in `speed-bench/qwen-regression/` before §8 step 3 lands.
- Operational: runs need the gateway and other sessions' benchmarks stopped.
  Schedule them with the user.
- **If Qwen regression fails → STOP DS4.1 work.** No "fix later".

## 5. HARD RULE #2 — track upstream, do not greenfield

Fork chain: `antirez/ds4` (remote `antirez`) → `ivanfioravanti/ds4-metal`
(remote `upstream`) → us (`origin`). V4.1 **already runs on Metal**, including
SSD streaming:

- `ds41_graph.streaming`.
- Large prefills overlap the next layer's reads.
- `QA_BEFORE_RELEASES.md` §17 records a "Q2, M5 SSD" quality row.
- `ds4_engram.c` reads with `F_NOCACHE` + `pread`.

Rules:

- Do **not** create parallel `ds4_v41_*.c` that duplicate the `deepseek41`
  path. Extend it.
- Sync often. Ivan and antirez commit daily; divergence itself threatens Rule #1.

Upstream branches not merged yet (each needs its own decision):

| Branch | Ahead | Why it is held |
| --- | ---: | --- |
| `upstream/kernelpool-1073-short-prefill` | 51 | draft V4.1 short-prefill work; revisit at §8 step 5 |
| `upstream/ds41f-dspark` | 31 | DSpark (TP-leaning); revisit at §8 step 8 |
| `upstream/ds41f-optimizations` | 17 | old base (9139e2a); check what is not already in main |
| `upstream/fix/m5-tensor-drift` | 1 | changes the Qwen default on M5; full Qwen tier first |
| `upstream/m5-round5-nax`, `qwen2-nax-tiles`, `exp/m5-tensor-precision` | — | expert-tile experiments, not V4.1 |

## 6. Code architecture — three layers

- **Generic (shared, Qwen-safe):**
  - What goes here: expert cache, SSD `pread`, async worker, LRU /
    frequency / recency, memory accounting, KV allocator primitives,
    instrumentation.
  - The live expert cache is `ds4_gpu_stream_expert_cache_*` in `ds4_metal.m`.
    It serves DS4, GLM **and Qwen PROD**, so any change to it goes through the
    full Qwen tier.
  - No `if (model == QWEN) … if (model == DS41) …` sprawl.
- **Qwen-specific (frozen):** Qwen graph, expert layout, tensor map, Metal
  kernels, quant recipe, MTP, context behavior.
- **DS4.1-specific:** extend the existing `deepseek41` code.
  - Net-new 64 GB policy code may live in new `ds41`-prefixed files.
  - No Qwen constants in it, and no DS4.1 assumptions in the generic layer.
- The old scallop pager / memory-manager / paged-KV files
  (`ds4_expert_pager.*`, `ds4_memory_manager.*`, `ds4_kv_cache.*`) are not on
  `develop`. Recover them from `c3c98d4` only if a step needs them.

## 7. Phase 0 — measure, then set the target (decision "C")

- **0(a) DONE:** resident floor = 8.79 GiB → load feasible (§3).
- **0(b):**
  1. Download `ds41f-q2`; pin the SHA256.
  2. Run the existing `--ssd-streaming` **unchanged** at 4K / 8K / 32K
     context, with 4 / 8 / 16 GB caches plus one 24 GB point to confirm the
     #810 shape. Record decode, prefill, TTFT, steady wired and swap.
  3. Per-token decode decomposition: resident-weight GPU time, per-layer
     sync, miss loads, Engram, host gaps.
  4. Router log over a real agent trace → hit rate vs cache size, and
     token-to-token expert overlap (the prefetch ceiling).
  5. Bytes read per token per tensor class, from the GGUF header. This
     replaces the §2.1 estimate.
- **Output:** a measured roofline, the chosen target (DoD 2) and an ordered
  lever list. Review it with the user before §8 step 3.
  **Done 2026-09-24:** `speed-bench/v41/phase0-20260924/RESULTS.md` (roofline
  §2, cache §3, locality §4, lever list §5, target options §6). The measured
  resident floor is 9.38 GiB, not the 0(a) 8.79 GiB: 0(a) left out the Engram
  projection weights.

## 8. Work plan

Each step ends with the V4.1 tests and the Qwen fast tier. Each phase ends
with the Qwen full tier.

0. **Sync** (done) and record the Qwen baseline.
1. **Phase 0(b)** (§7).
2. **Set the target** (user decision, from the §7 output). Done 2026-09-24:
   ≥ 20 t/s without quality loss (DoD 2).
3. **Memory and cache policy for 64 GB:**
   - Small cache (§3.1), filled by slot `pread`.
   - No mmap-backed expert views.
   - Budget recomputed per context from the wire limits; never hard-coded.
4. **Per-layer sync:** the router → load data dependency forces a host sync
   per layer. The exact experts for layer N+1 cannot be read during layer N.
   The precedent is Qwen SCALE-3A: a GPU poll gate plus a service thread
   gave +3–5 % byte-identical. Generalizing it touches the Qwen path → full
   tier.
5. **Prefill:** bulk-read the union of a chunk's experts into a staging buffer
   so GEMM prefill runs on streamed layers. Qwen SCALE-3.2 reached 92 % of
   resident this way. Upstream already overlaps next-layer reads, so measure
   before building.
6. **Metal** on the resident-bytes term, the largest in §2.1. Generic or
   shared kernels only, then the full tier.
7. **Predictive prefetch:**
   - Must never change exact expert selection.
   - Keep it only if net t/s rises without inflating SSD traffic.
   - Priors from Qwen: token-to-token overlap is ~36 %, and LRU ≈ a frequency
     oracle.
8. **DSpark / MTP** (`upstream/ds41f-dspark`):
   - Measure SSD bytes per accepted token.
   - A verify batch touches the union of its tokens' experts.
9. **Cheaper non-routed weights for speed** (e.g. attention / shared below
   Q8_0):
   - Needs the user's approval and the §9 quality gate.
   - "No precision drop merely to shrink size" still holds.
10. **Agent benchmark:** Claude Code / Codex real tasks. Wall-clock is the
    final metric.

Deferred unless data demands them:

- A paged-KV adapter. V4.1 KV is MLA-compressed, and Qwen's paged KV was inert.
- 1M context, until 256K is stable.
- The quant research track (§3.3).

## 9. Correctness and quality

- **Oracle:** no resident reference fits on 64 GB. Use the same build in
  synchronous streaming (no prefetch, no async). Every cache, async, prefetch
  or staging change must give byte-identical greedy output against it.
  Lesson (Qwen, 2026-09-22): streaming silently disabled prefill GEMM and
  changed outputs on prompts over 64 tokens.
- **External quality:**
  - `gguf-tools/quality-testing/deepseek-v4.1-flash-20260919-router`: 112
    official-API prompts.
  - `QA_BEFORE_RELEASES.md` §17: Q2 M5 SSD General-100, 2697/2994 top-1.
- **Functional:** ds4-eval core plus an agent smoke.

## 10. KPIs (record on every benchmark)

- tok/s (prefill / decode), TTFT.
- **SSD GB per generated token** and per accepted token; read ops; read latency.
- Expert-cache hit / miss %.
- Steady and peak **wired** GiB (`vm_stat`); cache, KV and workspace sizes.
- Per-token time decomposition: resident weights / sync / misses / Engram /
  host gaps.
- GPU / CPU utilization.

## 11. Non-negotiables

1. Never break or alter Qwen3.8. Every shared-runtime change passes the Qwen
   gate. If it fails, STOP.
2. Track `antirez` + `upstream`; extend the `deepseek41` path; no parallel
   graph.
3. Exact router-first selection. Never read experts the router did not
   select, except predictive prefetch that proves a net gain.
4. Minimize SSD bytes per token **and** wired memory.
5. No performance claim without benchmark data. No precision drop merely to
   shrink size.
6. Optimize SSD, cache and sync before DSpark / MTP.
7. The target comes from Phase 0 data; 10–20 t/s is not assumed to be final.
8. GPU runs happen only when the machine is free. Check other sessions and
   ask first.
