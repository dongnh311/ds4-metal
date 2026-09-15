# DS4-IQ2 M5 Pro FAST Campaign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise sustained decode tok/s of the Qwen3.8 `qwen4exp` DS4 IQ2 build on an M5 Pro 64 GB through kernel-only (bit-identical) optimization first, then MTP acceptance tuning, using the Orca Uncensored IQ2 GGUF from the companion plan as the benchmark artifact.

**Architecture:** Phase 1 touches only Qwen4 Metal decode kernels and their `ds4_metal.m` dispatch layers, gated by bit-identical logit fixtures (`test_qwen4_logit_dump.py`) and `ds4-bench` A/B. Phase 2 tunes the existing adaptive MTP depth machinery (`DS4_QWEN4_MTP_DEPTH`, draft-row selection, rewind cost). No quantization changes in this campaign. PLE I/O and SSD expert streaming are tracked as Phase 3 candidates, not started.

**Tech Stack:** Objective-C Metal (`ds4_metal.m`, `metal/qwen4.metal`), C graph scheduling (`ds4.c`, `DS4_HAS_QWEN4_METAL`-guarded blocks), Python logit-dump fixtures, `ds4-bench`.

**Spec:** `docs/superpowers/specs/2026-09-15-orca-uncensored-ds4-iq2-design.md` (tok/s lever section) + in-session decision: Phase 1 kernel-only → Phase 2 MTP → Phase 3 PLE/SSD later. Plan 1 (Orca build, `docs/superpowers/plans/2026-09-15-orca-uncensored-ds4-iq2.md`) must be finished first on the M5 Pro; its Task 5 numbers are this campaign's baseline.

## Global Constraints

- **AGENT.md:** every task must (a) confirm the normal Metal path is not slower (`ds4-bench` before/after), (b) confirm SSD streaming is unaffected (Qwen4 code is `DS4_HAS_QWEN4_METAL`-guarded, Metal-only — verify no shared struct changed), (c) if a change touches shared non-Qwen4 code or `ds4_gpu.h`, ASK the user before distributed testing and before claiming CUDA safety (CUDA machine access is the user's).
- **Bit-identical gate:** any kernel rewrite passes `python3 tests/test_qwen4_logit_dump.py` (teacher-forced dump vs baseline) and `make test-qwen4-kernels` before `ds4-bench` is even run. A faster path with unexplained logit drift is REJECTED (AGENT.md: correctness before speed).
- **No new permanent semantic variants:** diagnostic env knobs are allowed only when they validate the one release path (precedent: `DS4_QWEN4_NO_FUSE`, `DS4_QWEN4_SPEC_FORCE_ACCEPT`).
- **Keep code small/sharp** — prefer modifying the existing fused kernel family over adding parallel variants; do not add C++.
- Benchmark protocol is fixed (Task 0): same prompt, 4K and 32K frontiers, 128-token greedy, `--mtp` on/off, 3 runs each, recorded in `speed-bench/`. No other numbers count.

## Hook Table (verified against `halfbeak` HEAD, 2026-09-15)

| # | Anchor | Role in decode | Phase |
|---|---|---|---|
| H1 | `metal/qwen4.metal:2412` `kernel_qwen4_moe_down` | Q2_K/MXFP4 down GEMV (padded 768 geometry, reads 640) | P1 |
| H2 | `metal/qwen4.metal:2462` `kernel_qwen4_moe_down_mxfp4_pf` | fused down for prefill batches | P1 |
| H3 | `metal/qwen4.metal:2544` `kernel_qwen4_moe_reduce` | expert out × router weight + shared + residual (ds4.c:58495 comment says "the reduce folds the combine in" — verify what is still separate) | P1 |
| H4 | `metal/qwen4.metal:2619` `kernel_qwen4_moe_build_lists` | token→expert list build (gather/scatter) | P1 |
| H5 | `metal/qwen4.metal:2848` `kernel_qwen4_moe_mm_mid` + `_down` (+`_nax` variants) | prefill batched gate/up & down | P1 |
| H6 | `metal/qwen4.metal:911` `kernel_qwen4_router_topk` | 512-expert logits + top-K | P1 |
| H7 | `ds4.c:58276` `qwen4_graph_moe` | per-layer MoE scheduling (reused for chained drafts at :58857) | P1/P2 |
| H8 | `metal/qwen4.metal:550/606/666/3656` GDN `gdn_prep/scan/scan_r4/out/gdn_front` | Gated DeltaNet recurrent decode | P1 |
| H9 | `metal/qwen4.metal:1733/1820` `kernel_qwen4_attn_decode`/`attn_merge` | QSA decode attention | P1 |
| H10 | `ds4.c:58051-58130` `qwen4_gemv_rows` call list (lin_qkv, lin_gate, alpha/beta, lin_out) | dense GEMV sequence per layer | P1 |
| H11 | `metal/qwen4.metal:3576/3622` `kernel_qwen4_mtp_stage`/`_combine` + `ds4.c:58723` `qwen4_graph_mtp_steps` | MTP draft stage | P2 |
| H12 | `ds4.c:73589` `qwen4_spec_depth` (adaptive 2→3, `qwen4_depth3_engaged` window logic) + `:58029` `qwen4_mtp_draft_rows` + `DS4_QWEN4_MTP_DRAFT_VOCAB` | draft depth policy + draft-head row selection | P2 |
| H13 | `ds4.c:83658` `ds4_session_rewind` snap0/snap/snap2 (:57521-57535) | replay cost on MTP rejection (discussion #5 territory) | P2 |
| H14 | `metal/qwen4.metal:760/848` `kernel_qwen4_ple_gate`/`ple_conv` + PLE demand-paging (commit `bd7061b`; ~10.5 GiB resident measured on M5 Pro) | PLE n-gram lookup | P3 (track only) |
| H15 | `ds4_bench.c` (prefill-interval sweep + generation) + `DS4_QWEN4_NO_FUSE=1`, `DS4_QWEN4_SPEC_FORCE_ACCEPT` | measurement harness | all |

Guard split (for the CUDA-safety check): all Qwen4 graph code in `ds4.c` sits under `#ifdef DS4_HAS_QWEN4_METAL` (defined at `ds4.c:54` in the Apple/Metal path; commit `bab2029` explicitly kept the Qwen graph out of CUDA/ROCm builds). `ds4_metal.m` and `metal/*` are Metal-only. **Any change confined to those = no CUDA regression possible; still run `make` for all active build targets at each task end. If a task needs a shared struct change in `ds4_gpu.h` → ask the user (CUDA machine) before proceeding.**

---

## Task 0: Baseline capture (M5 Pro, after Plan 1 Task 5)

**Files:** `speed-bench/` (CSV append), `docs/ORCA_UNCENSORED_DS4_IQ2.md`

**Interfaces:** Produces `speed-bench/orca-iq2-baseline.csv` (columns: ctx, mode=plain|mtp|forced-accept, prefill_tps, decode_tps, mtp_accept1, mtp_accept2, run) used by every later task's before/after comparison.

- [ ] **Step 1: Build + smoke**

```bash
make
./ds4 -m gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-MTP.gguf --ctx 8192 --prefill-chunk 1024 -p "OK"
```
Expected: single-line OK, no Metal OOM. If Plan 1's Orca GGUF isn't ready, use Ivan's template GGUF — but note which in the CSV.

- [ ] **Step 2: Record MTP acceptance counters**

The acceptance window (`s->qwen4_depth_window`, `qwen4_mtp_accepted` at `ds4.c:73599+`) is not printed by default. Add a **diagnostic-only** one-liner behind `DS4_QWEN4_MTP_STATS=1` (env check + `fprintf(stderr)` at session end, ~6 lines) — allowed by the "diagnostic switches validate the one release path" rule. Without it, acceptance data has to be inferred from tok/s deltas, which is too coarse for Phase 2.

```bash
DS4_QWEN4_MTP_STATS=1 ./ds4 -m <GGUF> --ctx 32768 --prefill-chunk 1024 --mtp \
  -p "<128-token agentic prompt>" -n 128
```

- [ ] **Step 3: Run the fixed sweep** (for `DS4_QWEN4_MTP_DEPTH` in unset, 2, 3 and `DS4_QWEN4_SPEC_FORCE_ACCEPT` on/off where sensible):

```bash
./ds4-bench -m <GGUF> --ctx 4096 --prefill-chunk 1024 --gen 128 --mtp      # and --ctx 32768, no --mtp
./ds4-bench -m <GGUF> --ctx 4096 --prefill-chunk 1024 --gen 128            # baseline without MTP
```
Write results to `speed-bench/orca-iq2-baseline.csv` (create `speed-bench/orca-iq2-baseline.csv` with the header row first). Also capture `vm_stat` peak swap during each run (M1 Max thrash was the failure mode in discussion #1).

- [ ] **Step 4: Bit-identical reference dump (the gate fixture for all of Phase 1)**

```bash
python3 tests/test_qwen4_logit_dump.py --model <GGUF>   # check its flags first (repo README/TESTING.md)
mv <dump output> speed-bench/orca-iq2-logits-baseline.txt
```
Commit baseline + stat knob: `git add speed-bench/ ds4.c && git commit -m "Record Orca IQ2 M5 Pro baseline and add MTP stats diagnostic"`

## Task 1: Decode GEMV kernels — H1 (moe_down) + H10 (dense GEMV sequence)

**Files:** `metal/qwen4.metal:2412`, `ds4_metal.m` (qwen4 dispatch ~48213+), test fixture reused.

**Interfaces:** Consumes Task 0 baseline CSV + logit dump. Produces: improved decode bandwidth on the down path; CSV delta row.

- [ ] **Step 1: Profile first, change second.** Run the H1/H10 kernels under Xcode Instruments (GPU profile) or Metal System Trace for one 128-decode run; record threadgroup sizes and memory throughput vs theoretical (M5 Pro ~2.5-2.9 TB/s). One-paragraph note in the commit message with the numbers — no "optimization" without a measured gap.
- [ ] **Step 2: Single targeted kernel change** (the Instruments gap, e.g. Q2_K down load coalescing / register pressure in `kernel_qwen4_moe_down` padded-768 geometry — the 3×256-block layout: confirm whether the kernel loads all 768 columns or already bounds to 640; if it loads 768, bounding to 640 with the same 3-block arithmetic is the first candidate).
- [ ] **Step 3: Gate — bit-identical**

```bash
make && make test-qwen4-kernels
python3 tests/test_qwen4_logit_dump.py --model <GGUF>   # diff vs speed-bench/orca-iq2-logits-baseline.txt → must be identical
```
- [ ] **Step 4: Bench A/B + commit**

```bash
./ds4-bench -m <GGUF> --ctx 4096 --prefill-chunk 1024 --gen 128
./ds4-bench -m <GGUF> --ctx 32768 --prefill-chunk 1024 --gen 128
```
Append CSV row (`task=H1`). Commit kernel + CSV: `git commit -m "Optimize Qwen4 Q2_K down GEMV <what>; +X% decode at 4K/32K"`.

## Task 2: Expert dispatch + MoE reduce — H4 (build_lists), H3 (reduce fusion check)

**Files:** `metal/qwen4.metal:2544-2900`, `ds4.c:58276` `qwen4_graph_moe`

**Interfaces:** Produces CSV row `task=H3H4`.

- [ ] **Step 1: Verify current fusion state.** `ds4.c:58495` claims "the reduce folds the combine in" — read `kernel_qwen4_moe_reduce` and confirm which of {router-weighted sum, shared expert, residual add} are actually inside it vs separate `hc_combine` calls. If all three are separate, one fusion pass (sum+shared+residual, per the +1% cold-prefill precedent in the Baekpica notes) is the change; if already fused, the gap is in `build_lists` gather patterns instead. Pick ONE change based on the read, not both.
- [ ] **Step 2: Implement (single pass change)**
- [ ] **Step 3: Gate** — `make test-qwen4-kernels` + logit dump identical + `DS4_QWEN4_NO_FUSE=1` still runs (the unfused diagnostic path must keep working, per `test_qwen4_mtp_limits.py`).
- [ ] **Step 4: Bench A/B (4K + 32K, plain mode), CSV row, commit.**

## Task 3: Router + GDN/QSA decode — H6, H8, H9 (only if Task 0 profile says they matter)

**Files:** `metal/qwen4.metal:911, 550-726, 1733-1930`

**Interfaces:** Consumes Task 0/1/2 profile notes. If Instruments shows decode time dominated by H1/H3 work, **this task is skipped** (YAGNI) — record the decision in the CSV as `task=skip-H6H8H9 reason=<profile note>`.

- [ ] **Step 1: Profile check gate** (Instruments GPU attribution for one 128-decode run; print top-5 kernel time shares in commit message)
- [ ] **Step 2-4: same pattern as Task 1** (one kernel change → bit-identical gate → bench A/B → commit)

## Task 4: MTP acceptance campaign — H11, H12, H13

**Files:** `ds4.c:58723/73589/83658`, `metal/qwen4.metal:3576/3622`

**Interfaces:** Consumes `DS4_QWEN4_MTP_STATS` data (Task 0 Step 2) + `DS4_QWEN4_SPEC_FORCE_ACCEPT` upper-bound number. Produces tuned depth policy + CSV rows `task=mtp-*`.

- [ ] **Step 1: Measure the acceptance distribution** with the Task 0 stat knob across 3 workload classes: agentic tool-calling (multi-turn `ds4-agent`), prose continuation, code generation. Per class: accept rate draft-1, draft-2, draft-3, effective tok/s at depth 2 vs 3 vs forced-accept. Record in CSV.
- [ ] **Step 2: Tune the depth policy.** The `qwen4_spec_depth` window logic (`bits >= 8` engage, `bits < 6` disengage, `reject2_streak >= 2` exit — `ds4.c:73589-73620`) is the first tuning target: with measured acceptance data, adjust thresholds if a workload class shows draft-2/3 consistently not paying. Keep the policy in-code (no env knob proliferation) — thresholds are constants chosen from the measured data, commit the data table into the commit message.
- [ ] **Step 3: Draft-head row selection.** `DS4_QWEN4_MTP_DRAFT_ROWS`/`DRAFT_VOCAB` (`ds4.c:57958-58031`): if forced-accept gap vs opportunistic MTP is large, test whether a wider draft-head row set narrows it (the draft head scores a frequent-token prefix — wider rows = better drafts but more verify work). One configuration change, bench all 3 workload classes, keep only if effective tok/s improves on at least 2 of 3.
- [ ] **Step 4: Rewind cost check (discussion #5 territory).** With `DS4_QWEN4_STATS`, confirm `qwen4_rewound` count per run: if full-transcript replays appear (log line "PLE sidecar eviction" per 1024 tokens during decode), that is the #5 bug reproducing on our build — do NOT fix it here, record it and hand to the user (runtime change in rewind snapshot logic, needs the same care as the open upstream discussion). Logit-dump gate still applies to any H13 change.
- [ ] **Step 5: Bench all 3 workload classes, CSV, commit.**

## Task 5: Campaign report + decision on Phase 3

**Files:** `speed-bench/orca-iq2-campaign.csv` (merged), `docs/ORCA_UNCENSORED_DS4_IQ2.md` (or `docs/PERFORMANCE.md` section)

- [ ] **Step 1: Merge all Task 0-4 CSV rows; compute headline deltas** (baseline → final, plain and MTP, 4K/32K, plus the 3 workload-class MTP numbers).
- [ ] **Step 2: Write the numbers into the doc** with the fixed protocol from Global Constraints, and a one-paragraph "what did NOT help" list (skipped tasks, rejected changes with reasons).
- [ ] **Step 3: Decision gate (user call):** did Phase 1+2 close the gap the user wants? If yes → stop. If tok/s is still insufficient, **that** is when Phase 3 (PLE I/O tuning behind the `bd7061b` residency-sweep knob, or SSD expert streaming) gets its own spec+plan — do not start it inside this campaign.
- [ ] **Step 4: Commit docs; branch review** (superpowers:finishing-a-development-branch before any PR to upstream).

---

## Self-Review

- **Spec coverage:** user's priority list maps as: 1 MTP → Task 4; 2 IQ2 gate/up + 3 Q2K down kernels → Task 1 (gate/up `moe_mm_mid` prefill variant H5 folds in if profiling shows prefill cost; decode GEMV path is H1/H10); 4 dispatch/gather → Task 2; 5 MoE reduce fusion → Task 2; 6 GDN/QSA → Task 3 (skippable); 7 PLE I/O → Phase 3 gate in Task 5; 8-9 SSD expert streaming → out of scope. Quantization changes explicitly deferred (spec: "chỉ sau cùng mới thử quantization mới").
- **Placeholder scan:** none; every step names exact files/lines/commands. Task 3 is conditionally-skippable by design (recorded, not silent).
- **Consistency:** one benchmark protocol (Global Constraints) used by all tasks; bit-identical gate order (kernel test → logit dump → bench) is the same in every task.
- **Known uncertainty:** the logit-dump script's exact CLI flags — verify against `tests/test_qwen4_logit_dump.py` source before Task 0 Step 4 (the plan says "check its flags first", same as Task 0 Step 1's `--ctx` usage).
