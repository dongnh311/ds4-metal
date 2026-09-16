# DS4-IQ2-SSD Expert Cache / Prefetch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the resident `DS4-IQ2-Fast` model (30–40 tok/s @ ~220K context on M5 Pro 64 GB, built by Plan 1) into a two-mode engine: **resident (Fast) = stock Metal path, byte-identical, zero abstraction penalty on RAM hits** + **paged = SSD-backed expert cache with L1/L2/L3 hierarchy and async prefetch**, so long-context (≥256K) workloads stop swapping and target 40+ tok/s.

**Architecture:** Reuse DS4's existing routed-expert streaming infrastructure (built for DeepSeek/GLM: `ds4_ssd.c` budget/lock API + graph-side "read next layer's experts while computing the current layer" overlap, per AGENT.md) and port it to the `qwen4exp` graph behind `DS4_HAS_QWEN4_METAL`. Experts are packed offline into one contiguous `qwen38-experts.bin` bundle (`[layer][expert]{gate IQ2_XXS, up IQ2_XXS, down Q2_K Pad768}`) + JSON index, written by a `gguf-tools/` packer from the Plan-1 GGUF. No quant or kernel changes in Phase 1–2; kernel/dispatch work (Plan 2 territory) stays gated behind measured miss data.

**Tech Stack:** C (`ds4_ssd.c`, qwen4 graph in `ds4.c`, `ds4_metal.m`), Python packer (`gguf-tools/`), `ds4-bench`, logit-dump fixtures, `fio`/`iostat`-style SSD counters via built-in stats env knob.

**Spec:** `docs/superpowers/specs/2026-09-15-orca-uncensored-ds4-iq2-design.md` + in-session decisions: (a) SSD is a cache hierarchy under the execution path, not a separate slow engine; (b) MTP last; (c) GreenBitAI `Qwen3.8-Flash-Next-4bit-paged` used as *design reference only* (their `gbx-lm` runtime source was not found public as of 2026-09-15 — the borrow is their index format, auto-residency idea, and 4-check validation protocol, not code).

## Global Constraints

- **Fast path is sacred (user decision, constraint ⑤):** the RAM-hit path must be the existing Metal qwen4 graph; the pager's only contract is `ensure_expert_resident(layer, expert, ids) -> bool all-hit`. No new copy/abstraction on hits. Verify with `test_qwen4_logit_dump.py`: resident-mode output is byte-identical to pre-port baseline.
- **Bit-identical gate (GreenBitAI protocol):** for paged mode: (1) logit match vs resident on ≥5 prompts to 160 tokens; (2) layer-wise resident-vs-paged comparison over all 48 layers; (3) MTP on/off checked separately. A paged run that drifts = architecture bug, reject the change.
- **AGENT.md:** Qwen4 code is `DS4_HAS_QWEN4_METAL`-guarded (Metal-only; CUDA/ROCm builds exclude it — commit `bab2029`). Any edit to shared structs in `ds4_gpu.h` or to DeepSeek/GLM streaming paths → STOP and ask the user (CUDA machine check + distributed test gate).
- **No new user-facing flags (AGENT.md: "Do not add permanent semantic variants behind flags").** Spec §21's `--expert-cache-size` / `--expert-prefetch-depth` map onto DS4's EXISTING flag family: `--ssd-streaming` enables the pager; `--ssd-streaming-cache-experts <bytes>` is the cache budget, auto-sized by `ds4_ssd_auto_cache_plan` (`ds4_ssd.c:108`) — this IS spec §6's "machine decides residency, not a flag" auto-residency; `--ssd-streaming-preload` covers hot pre-residency. Anything new stays env-only and diagnostic (`DS4_QWEN4_PAGER_STATS`, eviction weights), the same convention as `DS4_QWEN4_NO_FUSE`.
- **SSD is strictly read-only — no writes, no cache files, no wear path:** `qwen38-experts.bin` opens `O_RDONLY`; no stage may create/append/truncate any file on the model SSD (no overflow cache, no write-back, no "spilled" pages — if RAM/KV pressure needs an overflow that is the OS's page-cache/swap territory, and a run that needs it fails the stage receipt instead of adding a write path). `hot_experts.json` and all receipts/CSVs live in the repo working dir, not the SSD. **Verify at every stage:** mount counters `diskwrites` delta across the run = 0 for the model volume (`iostat`/`diskutil` or a `misc/ssd_watch.sh` counter), recorded in the CSV next to the read counters.
- **Read volume is a first-class KPI (endurance = reads count too):** per CSV row record SSD read GB (iostat byte delta on the model volume) alongside `expert misses/token`. Hard budget: ≤ `bundle_bytes × 1 × (miss_rate × 1.1)` total per session at 220K context — i.e. the theoretical worst-case single full sweep plus 10% waste; any stage whose measured read GB exceeds it has a prefetch/predict bug (stage D's predictive reads double-count) and is rejected until the miss math is fixed. Target state: L2 prediction keeps SSD reads ≤ ~1 full bundle sweep per 1000+ decode tokens at steady state (hit ≥ 95%).
- **No requant, no kernel rewrites in Phases 1–2.** IQ2_XXS/Q2_K + current kernels only. Kernel work = separate campaign (Plan 2) after miss-rate data exists.
- **Benchmark protocol (user-specified, fixed):** every staged build reports tok/s, TTFT, expert cache hit %, prefetch hit %, SSD read MB/s, SSD read latency, expert misses/token, stall/token, Metal utilization, RAM, swap. Stage letters A→G below are cumulative and each gets its own CSV row set. "Faster" is accepted only via this table, never by feel.
- Build machine: M5 Pro 64 GB; paged runs need `qwen38-experts.bin` (~36 GiB IQ2/Q2K payload, smaller than GreenBitAI's 70.31 GiB 4-bit store — a core advantage) on a fast local SSD.

## Reference notes (research findings, 2026-09-15)

- `gbx-lm` source: **not found public** (GitHub/HF search empty). GreenBitAI Qwen paged model card (reference-only): `model.safetensors` 3.80 GiB resident + `experts.bin` 70.31 GiB + `ple-q4.rows` 29.80 GiB; 48 layers × 2 draws layer-wise exact; 5 prompts to 160 tokens bit-identical, 1 longer prompt max logit delta 6.8 with identical greedy tokens. Their DeepSeek paged model adds `GBX_PAGING`/`GBX_DEEPSEEK_MTP` envs and "machine decides residency, not a flag" auto-budget. We borrow: bundle+index layout idea, auto budget, validation protocol.
- DS4 already implements, for DeepSeek/GLM Metal: `--ssd-streaming` with auto byte-budget (`ds4_ssd_auto_cache_plan`, `ds4_ssd.c:108`), memory lock (`ds4_ssd_memory_lock_acquire`, `ds4_ssd.c:142`), and graph-level overlap of next-layer expert reads with current-layer shared/routed compute (`AGENT.md` memory policy; `g->ssd_streaming` hooks throughout `ds4.c` from ~line 19760). The port reuses these; the new work is qwen4-graph integration + the offline bundle packer + L1/L2 prediction.

## 3-tier cache design (locked)

```
L1  executing layer's selected experts (held during the 48-layer sweep of one decode step)
L2  predicted next-layer experts, prefetched async during current-layer compute
L3  qwen38-experts.bin on SSD (single pread per bundle)
eviction score = frequency + recency + next-layer probability + load cost  (Phase 2, not LRU)
double buffer: buffer A → Metal compute; buffer B → SSD read; buffer C → L2 prefetch
```

Staged enablement (each stage is a separate commit + bench row set, so regressions are attributable):

| Stage | What turns on | Cumulative? |
|---|---|---|
| A | Plan-1 model resident, no pager (baseline from Plan 1 Task 5) | — |
| B | bundle+index + pager port; L1 only, synchronous read on miss (correctness gate) | A |
| C | + async double-buffer prefetch of *current* layer's experts issued after router result, overlapping dense/GDN compute of that layer | B |
| D | + L2 predictive next-layer prefetch (router history → predicted expert set, issued while layer N computes) | C |
| E | + scored eviction (frequency+recency+next-layer prob+load cost) replacing naive LRU; hot-expert pre-residency from router statistics captured in a dry run | D |
| F | Metal dispatch optimization for paged layout (page-align, batched preads) — only if SSD read latency shows on the critical path | E |
| G | MTP campaign (Plan 2 Task 4 material, re-sequenced here per user's "MTP last") | F |

---

## Task 0: Research + baseline lock (M5 Pro, no pager code)

**Files:** `docs/ORCA_UNCENSORED_DS4_IQ2.md` (append "SSD stage A baseline" section); create `speed-bench/orca-iq2-ssd-baseline.csv`

**Interfaces:** Produces stage-A baseline numbers (tok/s/TTFT at the 5-point sweep 32K/64K/128K/192K/220K, RAM, swap) + a `fio`/`iostat` profile of the target SSD (sequential + 4K random read MB/s, IOPS, latency p50/p99) that every later stage compares against. Each bench point also writes the provenance receipt (spec §3/§20): git commit, model sha256, macOS version, chip, ctx, prompt/gen lengths, temp, mtp on/off, swap peak, and **output_checksum** (sha256 of the generated token-ids at greedy) — the golden-reference token sequence. If 220K or any point beyond is unstable, record the **failure mode** (Metal OOM / swap / stall duration) in the receipt instead of forcing the run. Also records SSD read latency/MBs counters method used in the protocol (e.g., `iostat -d -I -w` sampling script `misc/ssd_watch.sh`, ~40 lines, allowed to live in `misc/` since it's a tool not a feature).

- [ ] **Step 1: Lock the Plan-1 build** — confirm `gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-MTP-NNgram.gguf` (final Plan-1 artifact, self-contained n-grams; record its sha256 — head `ed238d8d…` — in the receipt) exists and Plan 1 Task 5 numbers are on file (ds4-eval core 11/12, smoke PASS, MTP gen ≈36.5–36.6 t/s @4K/32K, resident 41.72 GiB, native ctx ceiling 262144). If Plan 1 is not done, run it first; this plan depends on its artifact, not on Ivan's.
- [ ] **Step 2: Stage A bench** — `ds4-bench` at 4K/32K/220K ctx (greedy + MTP on), `vm_stat` peak swap, TTFT. Write CSV rows labeled `stage=A`.
- [ ] **Step 3: SSD profile** — measure the actual SSD: sequential read, 4K random read, p50/p99 latency, with the model file warm in page cache vs cold (`purge` or reboot-to-measure). These numbers define the prefetch-depth budget: `budget_ms = p50_latency × max_bundles_per_miss`. Commit the profile + CSV.

## Task 1: Offline bundle packer (`gguf-tools/qwen4_expert_bundle.py`)

**Files:** Create `gguf-tools/qwen4_expert_bundle.py`; test `tests/test_qwen4_expert_bundle.py` (new, CPU-only, runs on M1 Pro 16 GB)

**Interfaces:**
- Consumes: Plan-1 main GGUF (Ivan-style `Reader` from `qwen4_pack_to_qwen4exp.py`), no model loading.
- Produces: `qwen38-experts.bin` (bundles packed `[layer][expert]`, each bundle = gate IQ2_XXS bytes + up IQ2_XXS bytes + down Q2_K Pad768 bytes, all contiguous, page-aligned to 4 KiB) + `qwen38-experts.index.json`:
  ```json
  {"version":1,"layer_count":48,"expert_count":512,
   "bundle_bytes":{"gate":<IQ2XXS row bytes>,"up":...,"down":...},
   "bundles":[{"layer":0,"expert":0,"offset":0,"size":...} ...],
   "gguf_sha256":"...","sha256": "..."}
  ```
  + per-bundle sha256 in the index (GreenBitAI-style "build-time verification record").

- [ ] **Step 1: Write the packer** — read the 48 down + 96 gate/up expert tensor payloads straight from the GGUF via the existing `Reader` (byte-copy, no re-quant), pack in `[layer][expert]` order, record offsets, verify: re-read every bundle, sha256-compare against the GGUF tensor payload sha256s from the Plan-1 manifest (same read-back-verify convention as `qwen4_iq2.py:174-188`).
- [ ] **Step 2: Unit test** `tests/test_qwen4_expert_bundle.py`: build a 2-layer × 4-expert synthetic GGUF fixture (reuse `tests/test-vectors/` patterns), pack, verify index offsets/sizes, verify `pread(index.offset, index.size)` returns exactly the GGUF payload bytes. Run: `make test` target for it.
- [ ] **Step 3: Full pack on M5 Pro** — run against the Plan-1 GGUF; expected `qwen38-experts.bin` ≈ 36–40 GiB (IQ2 gate/up + Q2_K down for 48×512 experts; the MTP Q4_K/MXFP4 experts are NOT bundled — MTP stays resident). Commit index JSON + packer (payload is gitignored under `gguf/`).

## Task 2: Pager core for the qwen4 graph (L1 + synchronous miss, stage B)

**Files:** Modify `ds4.c` (qwen4 graph, all edits inside `#ifdef DS4_HAS_QWEN4_METAL` blocks near `qwen4_graph_moe` at `ds4.c:58276` and the graph struct around `ds4.c:57521`), Create `ds4_expert_pager.c/h` (new small module, qwen4-only, not wired into DeepSeek/GLM paths).

**Interfaces:**
- Consumes: `qwen38-experts.bin` + index (Task 1), Plan-1 GGUF (main model now omits the 144 bundled expert tensors? NO — keep them in the GGUF for the resident mode; the pager is an *alternative* source selected at load when `--ssd-streaming` + paged bundle present, mirroring how GLM streaming reads experts from the GGUF file itself).
- Produces: `ds4_expert_pager` API: `ds4_expert_pager_open(bin, index)`, `pager_ensure(pager, layer, expert_ids, n) -> int (0 all-resident, >0 misses)`, miss path = `pread` bundle into double-buffered staging → upload buffer for the qwen4 expert kernels. Stats struct: misses/token, hit %, pread bytes/latency sums (diagnostic env `DS4_QWEN4_PAGER_STATS=1`, printed at exit — same diagnostic convention as Plan 2 Task 0).

Key decisions (locked, implement as stated):
1. Bundle source: pager reads `qwen38-experts.bin` (not the GGUF) — one pread per expert; the GGUF's own expert payloads are unused in paged mode but kept in the file for mode switching without repacking.
2. Metal buffer: paged experts go to a staging `ds4_gpu_tensor` reused per (layer, expert-slot), **no per-expert allocation churn**; upload while the dense part of the same layer executes (this is stage B's minimal overlap, full async is stage C).
3. L1 = the current layer's bundle staging held across the 48-layer sweep of one decode token; eviction of L2/older layers happens only in stage E.

- [ ] **Step 1: `ds4_expert_pager.c/h`** — open/index/pread/stats; ~200 lines; unit-testable without Metal: `tests/test_expert_pager.c` (CPU, fake buffers, asserts offsets/sizes/recorded stats vs a synthetic index).
- [ ] **Step 2: wire into `qwen4_graph_moe`** (`ds4.c:58276`): when `--ssd-streaming` and pager is open, expert weight pointers come from pager staging instead of resident tensors; keep the existing kernel dispatch 100% unchanged (kernels read whatever pointer the graph passes).
- [ ] **Stage B gate — correctness before anything else:**
  ```bash
  make test
  # resident baseline dump (pre-existing):
  python3 tests/test_qwen4_logit_dump.py --model <Plan1-GGUF> > /tmp/resident.txt
  # paged run:
  ./ds4 -m <Plan1-GGUF> --ssd-streaming -p <prompt5> -n 160 --temp 0   (5 prompts, GreenBitAI protocol)
  python3 tests/test_qwen4_logit_dump.py --model <Plan1-GGUF> --ssd-streaming > /tmp/paged.txt
  diff /tmp/resident.txt /tmp/paged.txt   # must be identical (quant layout is shared; paging moves bytes, not math)
  ```
- [ ] **Step 3: bench stage B rows** (4K/32K, tok/s + all protocol metrics incl. `iostat` SSD MB/s during the run) → CSV `stage=B`. Expect: slower than A on misses (synchronous), identical on warm cache.
- [ ] **Step 4: commit** (`ds4_expert_pager.{c,h}`, `ds4.c` hooks, `tests/test_expert_pager.c`, CSV). Message includes stage-B numbers.

## Task 3: Async double/triple buffering (stage C)

**Files:** `ds4_expert_pager.c` (background read worker: 2 read buffers + 1 compute buffer, `dispatch_io` or a GCD worker — Metal build is Apple-only so `dispatch_async` on a global queue is acceptable; keep it 1 worker, no thread pools); `ds4.c` qwen4 per-layer schedule.

**Interfaces:** Consumes stage B. Produces: `stage=C` CSV rows + `prefetch hit %` in stats.

- [ ] **Step 1: worker + triple buffer** — after layer N's router completes, issue preads for layer N's *missing* bundles into buffer-B while the dense/GDN part of layer N computes on buffer-A; layer N's expert GEMM waits on buffer-B ready (it will be, by construction of the overlap window; if not, stall is counted, not hidden — record `stall/token`).
- [ ] **Step 2: gate** — logit dump diff vs resident (must remain identical; async moves timing, not bytes); bench `stage=C` rows (tok/s, prefetch hit %, stall/token, SSD MB/s).
- [ ] **Step 3: commit.**

## Task 4: Predictive next-layer prefetch (stage D)

**Files:** `ds4_expert_pager.c` (L2 predictor), `ds4.c` (router history capture).

**Interfaces:** Consumes stage C + router selection history. Produces `stage=D` rows incl. `prefetch hit %` for L2 predictions.

- [ ] **Step 1: predictor, simplest honest version first** — MoE expert choice is autocorrelated (same prompt segment → same expert set). L2 = *re-issue layer N's expert set as layer N+1's prediction* (cheap, no learned model, no training data needed). Record predicted-vs-actual next-layer match rate in stats.
- [ ] **Step 2: issue L2 preads during layer N compute** (buffer-C rotation); on mismatch, the stale L2 read is wasted bandwidth — measure it, don't pretend it's free.
- [ ] **Step 3: gate + bench `stage=D`** (logit diff unchanged; tok/s and predicted-match-rate in CSV).
- [ ] **Step 4: if predicted match < ~50% on agentic workloads, stop and report** — the next-layer-probability term of stage E only pays off if *some* locality exists; no auto-escalation.

## Task 5: Eviction ladder — LRU first, then scored eviction + hot pre-residency (stage E)

**Files:** `ds4_expert_pager.c` (eviction policy: **stage E1 = plain LRU** (spec §10: "initial implementation may use LRU"); **stage E2 = scored** `score = w1·freq + w2·recency + w3·next_layer_prob − w4·bundle_size`, with `w1..w4` exposed as env-overridable constants `DS4_QWEN4_CACHE_W_FREQ/RECY/PRED/COST` — diagnostic experiment knobs per AGENT.md convention, defaults set from stage D stats, not user-tunable flags), `gguf-tools/` (optional small script to capture router statistics in a dry resident run → `hot_experts.json` pre-residency list).

**Interfaces:** Produces `stage=E1/E2` rows incl. hit/miss/eviction/**re-load rate** counters (spec §10; the pager stats struct in Task 2 already records them — surface all four in the CSV). Never evict bundles in state IN_USE or PREFETCHING (spec §10 hard rule; enforced in the pager state machine from Task 2: COLD → PREFETCHING → READY → IN_USE → EVICTING → COLD).

- [ ] **Step 1: dry-run stats capture** — one resident run over a representative workload records per-(layer, expert) selection counts; pack into `hot_experts.json` (tool, Python, ~80 lines, output data is per-workload and NOT committed to the model artifact).
- [ ] **Step 2: scored eviction** in pager cache; pre-residency fills the byte budget (from `--ssd-streaming-cache-experts` auto-budget, existing `ds4_ssd_auto_cache_plan`) with hot experts first.
- [ ] **Step 3: gate + bench `stage=E`.**
- [ ] **Step 4: decision gate (user number):** if aggregate 220K-context tok/s is still ≤ stage A baseline, the campaign's "30–40 → 40+" claim is not met by paging — record honestly in the doc and recommend Plan-2 kernel work (expert dispatch/GEMV) as the remaining lever, since paging can only remove *wait*, it cannot add compute.

## Task 6: Metal dispatch tuning for paged layout (stage F) — only if F trigger fires

**Files:** `ds4_metal.m` (qwen4 staging upload paths), `metal/qwen4.metal` (no quant changes; at most block-size alignment).

**Trigger (hard gate, from stage D/E metrics):** SSD read latency on the critical path, i.e. `stall/token` × miss rate > 5% of step time while SSD iostat shows the read bandwidth headroom (from Task 0 profile) unused. Otherwise **skip F** — a skipped stage is a result, record it.

- [ ] **Step 1: page-align bundles (already 4 KiB in Task 1) + batched multi-bundle preads (read k bundles in one syscall loop) + merged GPU buffer uploads** (one `memcpy`/`dispatch` for a contiguous run of bundles instead of per-expert).
- [ ] **Step 2: bit-identical gate + bench `stage=F`.**

## Task 7: MTP re-benchmark on the paged model (stage G) + campaign report

**Files:** `speed-bench/orca-iq2-ssd-campaign.csv` (merged A–G), `docs/ORCA_UNCENSORED_DS4_IQ2.md` (new "Paged expert cache" section: protocol, all stage numbers, 3-tier cache design, skip reasons, final recommendation).

**Interfaces:** Consumes stages A–F. MTP here is *measured on top of* the paged build, not tuned (MTP tuning itself remains Plan 2's Task 4; re-sequenced to G per user decision "MTP phải để sau").

- [ ] **Step 1: bench `stage=G`** = best paged stage (E or F) × {MTP off, MTP on, `DS4_QWEN4_MTP_DEPTH` 2/3}, all protocol metrics.
- [ ] **Step 2: write the campaign table** (A→G × {tok/s, TTFT, hit%, SSD MB/s, RAM, swap} at 32K/64K/128K/192K/220K, with 256K/320K/512K probed and failure modes recorded) into the doc; state explicitly which context frontiers paged ≥ resident (the "long-context wins" claim only counts if it's in this table). **Success thresholds (spec §18):** paged ≥ resident tok/s within 5% at every frontier where resident does not swap; first target ≥40 tok/s @220K; 256K+ practical with **zero OS swap** — a paged run whose only win is "runs but swaps" is recorded as a failure mode, not success.
- [ ] **Step 3: user decision gate:** keep paged mode, and whether any stage's change set needs the CUDA/distributed regression check per AGENT.md (pager is qwen4-Metal-only; if no shared struct moved, report "CUDA unaffected by construction" with the `grep -c` evidence; if it did move, ASK).
- [ ] **Step 4: commit docs + CSV; branch review (superpowers:finishing-a-development-branch) before any PR.**

---

## Self-Review

- **User's 5 optimizations mapped:** ① bundles → Task 1/2; ② predictive prefetch → Task 4; ③ scored hot-cache → Task 5; ④ triple buffering → Task 3; ⑤ sacred Fast path → Global Constraint + every stage's bit-identical gate.
- **User's phase order respected:** paging → prefetch → (dispatch = F, gated) → MTP last (G). No requant anywhere.
- **GreenBitAI role honored, no overclaim:** their repo is reference (index/validation/auto-budget ideas); their `gbx-lm` source was not found public, so "mổ source đầu tiên" degrades to model-card design study + DS4's own pager — stated in Reference notes; no task depends on gbx-lm code.
- **Placeholder scan:** stage F and task skips are *gated skips with recorded decisions*, not TODOs. All bench numbers are produced by named commands, not assumed.
- **Known risks:** (a) 220K context at stage D/E may hit page-cache/PLE eviction interaction with PLE's own demand paging (`bd7061b` residency sweep) — note the Plan-1 artifact is the `-NNgram` build (n-grams embedded in the GGUF, no separate PLE sidecar), so at stage B first confirm whether the `bd7061b` paging path is even exercised for this artifact; the combined memory accounting is measured in stage B, not tuned blind; (b) qwen4's `DS4_QWEN4_MTP` embedded experts: MTP expert tensors are NOT in the bundle (resident only) — verify with the packer's `expected set` (48 trunk layers only) and fail loudly if MTP names leak into the bundle.
