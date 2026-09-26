# Ornith qwen35moe M4: Acceptance (quality + speed) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the Ornith-1.5-35B-A3B `qwen35moe` ds4 build meets the spec's quality bar (gate 2, index ≥ 86.6 on 23G and 25G) and speed bar (gate 3, ds4 decode-with-MTP ≥ live oMLX and ds4 prefill ≥ live oMLX at 2K/32K/128K and a cold ~31K first turn), by adding the Q5_K tiled prefill GEMM, an Ornith stage profile, keep-only-if-exact-and-faster decode levers, and a conditional long-context KV mode, then record a report and update the spec.

**Architecture:** M4 changes nothing about Ornith correctness (M1) or MTP (M2) or serving (M3); it makes the resident build fast and measures it. The one mandatory kernel is the Q5_K tiled expert GEMM — additive entry points in `metal/qwen35.metal` (appended after `metal/qwen4.metal`), new host wrappers in `ds4_metal.m`, one added `||` in the shared pipeline-specialisation branch, and one condition in `ds4_qwen35moe.inc`'s MoE dispatch. Everything else is host-side levers behind `DS4_QWEN35_*` switches, a new A/B harness `speed-bench/ornith/m4_ab.py` that owns one model process at a time and interleaves ds4 against oMLX, and gateway quality tooling copied to the scratchpad and pinned to a staging ds4-server port. `qwen_gate.py` stays byte-for-byte unchanged; it is the Qwen3.8 gate every shared-file commit must pass.

**Tech Stack:** C (`ds4.c`, `ds4_server.c` unchanged here), Objective-C (`ds4_metal.m`), Metal Shading Language (`metal/qwen35.metal`), Python 3 stdlib (harness, eval, probe clients), the AI-Gateway-MLX bakeoff harness (read-only import from `/Users/dongnh/Documents/GitHub/AI-Gateway-MLX`), Homebrew llama.cpp `llama-server` 0.5.0 (the M1 gate-1 oracle, still installed; Task 11 reuses `tests/ornith/gate1.py` against the committed references to bound each KV mode's quantization error).

**Spec:** `docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md` (this plan is milestone M4 of §9). Binding controller decisions: the scratchpad `plan-research/DECISIONS.md` §4 (M4) and §1–§3 (Global Constraints and the M2/M3 interfaces M4 consumes).

## Global Constraints

Copied verbatim from `DECISIONS.md` §1; every task's requirements implicitly include this section.

- The Qwen3.8 production path stays byte-identical and as fast as today (spec §1, §7). Every commit that touches a shared file (`ds4.c` outside Ornith-only functions, `ds4_metal.m`, `ds4_gpu.h`, `metal/*.metal`, `ds4_server.c`, `ds4_agent.c`, `ds4_kvstore.c`) passes `make test-qwen4-kernels test-qwen4-q2` and `speed-bench/qwen-regression/run.sh fast`; the branch passes `run.sh full` before merge.
- Metal only for Ornith; refusals from M1 stay unless the plan lifts one explicitly.
- Ornith knobs use the `DS4_QWEN35_*` prefix; Ornith code reads no family-level `DS4_QWEN4_*` knob (spec §7.3; the kernel A/B switches inside shared qwen4 helpers are the documented exception).
- Kernel changes are additive: new kernels/entry points in `metal/qwen35.metal`; `metal/qwen4.metal` is not edited.
- Code, comments, docs and commit messages in English. Model files never go into git. The 23G GGUF: `DS4_ORNITH_MODEL=$HOME/orca/workspaces/ds4-metal-data/gguf/ornith/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf`. Every model test reads `DS4_ORNITH_MODEL` and fails with a message when it is unset.
- One model process at a time on this 64 GB machine. Never `kill -9` a Metal process. Long runs under `caffeinate -i -s`; monitors poll process liveness, not only a log pattern.
- GPU windows: pausing the live stack needs the user's OK once per execution window. Pause = `launchctl unload ~/Library/LaunchAgents/dev.dongnh.gateway-watchdog.plist ~/Library/LaunchAgents/dev.dongnh.gateway-eval.plist ~/Library/LaunchAgents/dev.dongnh.ai-proxy.plist`, then `kill -TERM $(cat ~/.local/share/ai-gateway/omlx.pid)`; restore = `launchctl load` of the three plists, then wait for `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8090/status` = 200.
- Never add a staging entry to the live gateway registry; never run AI-Gateway `run_ab.py`/`run_ab.sh` as they are (they rewrite live aliases and hot-swap oMLX). Staging ds4-server ports: 18296 (Ornith staging), 18190/18191 stay the oracle/loader test ports.
- No C++. Follow AGENT.md: small readable code; comments explain why.
- Every commit ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2
  ```

Plan-specific constants:
- Scratch gateway port 18131 (D-m4-gateway.md recipe); ds4 staging port 18296. Never :8090, :18085 or :18086 for a staging server.
- Receipts live under `speed-bench/ornith/m4/`; per-run dumps under `speed-bench/ornith/m4/ds4/`, `.../omlx/`, `.../logs/` are git-ignored (already covered by the M1 `.gitignore` glob `speed-bench/ornith/*/ds4/` and `speed-bench/ornith/*/logs/`; add `speed-bench/ornith/*/omlx/` in Task 1).
- The gateway bakeoff Python is imported read-only from `GW = /Users/dongnh/Documents/GitHub/AI-Gateway-MLX`; nothing under `GW/reports/*/run_ab.*` or `GW/scripts/install-launchd.sh`/`proxy-stack.sh`/`POST /runtime/action` is run.
- The Python interpreter that has `jinja2`, `tokenizers`, `numpy`, `pyarrow` for the gateway harness is `~/.local/ai-gateway-env/bin/python3`; never `pip install` into `~/.local/omlx-venv` (the PROD oMLX environment).

## Deviations from the spec wording, decided while planning

1. **Spec §1 goal 3 and §8 gate 3 oMLX constants are stale.** The spec's "about 54.5 t/s decode, 31K-token first turn in about 19 s" comes from oMLX 0.6.2, the *regular* scottlowry Ornith with specprefill ON (`AI-Gateway-MLX/reports/ornith15-vs-qwen38-2026-08-21`). The 54.5 has no traceable source. The live slot (Shiftedx abliterated, oMLX 0.6.4, specprefill off, MTP on) is faster: a 30,373-token cold request had TTFT ~15.2 s (~2,000 t/s prefill), mean decode ~65.4 t/s over 196 events, ~72–80 t/s at 24K, ~42 t/s at 66K. Gate 3 is therefore a **live interleaved A-B-B-A** against oMLX on this machine, at 2K/32K/128K plus a cold ~31K first turn, with the bar being the live oMLX measured in the same run, not the stale constants. Task 13 edits spec §1 goal 3 and §8 gate 3 to say this.
2. **Spec §4 (MoE) "M1 deviation" is retired.** Spec §4 says "The tiled Q5_K GEMM is deferred to M4, where prefill speed is measured." Task 4 ships it; Task 13 removes that deviation paragraph from spec §4 and edits `speed-bench/ornith/m1/RESULTS.md`'s "Deviation: Q5_K prefill" note to point at the M4 receipt.
3. **KV modes.** Spec §5/§8 name only "the existing FP8 KV mode" as the long-context lever. `DECISIONS.md` §4.4/§4.5 allow both FP8 (mode 1) and 4-bit (mode 2). This plan implements a single `DS4_QWEN35_KV` knob covering `f16|fp8|q4`, and either non-F16 mode must pass gate 2 again before it becomes a default (spec §8 gate 3). Task 13 notes this in spec §5.
4. **Gate 2 pass rule is tightened.** Spec §8 says "scores at least 86.6". The harness excludes truncated/errored cases from the rate, so a truncating model can score high on nothing. Per `DECISIONS.md` §4.4 the pass rule is **index ≥ 86.6 AND truncated = errored = 0** on both tiers; the 7-case agentic matrix is 7/7 with no guard-counter movement; the abliteration probes match the oMLX baseline captured first. Task 13 notes this in spec §8 gate 2.

## Pre-execution fix list (dry run 2026-09-26) — apply before Task 1

A dry run applied all 13 tasks on develop + the M2 and M3 plans' code. Everything built with 0 warnings and the kernel tests passed with and without Metal 4 only after the fixes below. The plan text is NOT yet updated for them; the executor (a fresh agent) updates the plan first, then runs it. Rulings are the controller's.

Hard failures:
1. Task 4 extractor: wrap the generated tensor-op templates in `#ifdef DS4_METAL_HAS_TENSOR` (body = mid + down + `"#ifdef DS4_METAL_HAS_TENSOR\n"` + naxmid + naxdown; the appended instantiation string keeps only its `#endif`). Without it `DS4_METAL_DISABLE_METAL4=1` / pre-M5 devices fail to compile the whole library (Qwen3.8 too). Run the kernel tests once with `DS4_METAL_DISABLE_METAL4=1` in Task 4 and Task 11. Make extractor re-runs byte-stable (today each re-run adds a blank line when L12 sits after the END marker).
2. Task 11 Step 3: also update the M1 caller in `test_attn_prep_noindexer` to the 26-argument prep (`(float)base, 1e-6f, kc, vc, kc, vc, 0u)`).
3. Task 11 Steps 5-6: the KV-mode tolerances are below the quantization error (measured fp8 rel 4.4e-2, q4 rel 1.1e-1). Ruling: check prep+decode against a host dequantization of the packed cache (measured fp8 rel 1.5e-7, q4 rel 5.2e-4), plus a loose bound vs the double reference (fp8 1e-1, q4 2.5e-1).
4. Task 11 Step 8: update the forward declaration of `qwen35_payload_body_bytes` to `(uint32_t rows, uint32_t mtp_rows, bool mtp, bool fp8, bool q4)` (one call site + the prototype, not three call sites); write the load-side read block, `test_qwen35_payloads_mode()` for f16/fp8/q4 and a "different KV cache mode" refusal case in full.
5. L12 kernel row mask: `const uint r0 = idx < n0 ? 0u : 1u; for (uint r = r0; r < 2u; r++)` (the plan's `rmax` gave the last key to the wrong row).
6. Ruling on L12 exactness: a verify that is not bit-identical to plain decode breaks M2's guarantee (`--mtp` greedy == plain). L12 is adopted only if exact by construction: Ornith's T=1 decode also runs `kernel_qwen35_attn_decode2` with one row, each row keeps its own split geometry, and the test is `memcmp` of the 2-row call vs two 1-row calls of the same kernel (plus a tolerance check vs the double reference). Otherwise L12 is not adopted.
7. Task 9 test: fill `gmid`/`gpart` with a sentinel before every call; also test down NR1.
8. Task 10 Step 1: `static DS4_MAYBE_UNUSED bool qwen35_mtp_draft_head_load(`.

Contradictions and prose-only steps:
9. L4 belongs in `qwen35_graph_attention_prep` (where the three gemvs are); it also changes the MTP catch-up prep.
10. Task 5 profile and Task 7 flush: write complete code with exact placement (setup before `bool ok = qwen35_graph_carry_h(...)`, marks per layer, final mark + print + `#undef` before `if (!ds4_gpu_end_commands())`), rename the `double last` shadowed by `ds4_gpu_tensor *last`, make the Interfaces line match what is printed, add a T=1 profile mode and the exact `--mtp` 128K profile command.
11. L12 wiring must not redeclare `const float scale` in `qwen35_graph_attend`.
12. Task 4 test: drop the never-printed "tensor mid within 2e-3" Expected line and the unread `DS4_QWEN35_MOE_{MID,DOWN}_TILES`; skip-and-label the tensor-tile case when `ds4_gpu_mpp_available()` is false; free `ref_part`; anchors instead of stale line numbers.
13. Task 9: NR kernels become default only after the A/B; write the default switch and its revert as code.
14. The L12 wrapper must not `getenv` a `DS4_QWEN4_*` knob; call the shared helper that decides merge-wide, or add a `DS4_QWEN35_*` knob.

Measurement steps:
15. Ruling: drop `--omlx-attach` (its `guard_free()` exits while oMLX runs, and attaching would load two models). Levers are measured ds4-vs-ds4 in one run (base env vs lever env, A-B-B-A, oMLX paused); oMLX is compared only in Task 3 and the final Task 12 run. Add that mode to `m4_ab.py` with unit tests.
16. Arms never SIGKILL: after SIGTERM and a bounded wait, raise `SystemExit` naming the pid.
17. Ruling: a lever is exact only if the gate-1 dump JSONs are byte-identical (`cmp`, not `max_delta`) at chunk 2048/64/65 AND `tests/ornith/test_mtp_cli.py`, `make test-qwen35-graph` and `make test-qwen35-mtp` pass with the lever's env (gate 1 never runs `--mtp`).
18. Ruling: drop L5 (the paired GDN path requires `!g->mtp_R`; Ornith PROD runs `--mtp`, so it never fires).
19. Task 10 vocabulary: `frspec_vocab.py` reads only parquet files/directories; `promessi_sposi.txt` gives 0 tokens and is Italian; there is no vi corpus; zero-count ids are dropped (a probe gave 7,218 ids). Reuse the corpora that produced `Qwen3.8-Flash-Next-draft-vocab-vi-en-code-64k.txt` (find its recipe in the repo); fill a short list by ascending id and say so. Its exactness check is item 17 with the vocab env set.
20. Task 12: `export SCRATCH=...` at the top of each block; explicit start/stop of the staging server with the final configuration (levers, KV mode, prefill chunk) for eval, matrix and probes; matrix pass = `metrics.incident_turns == 0` with a guard for a missing `summary.json`; probe regex handles curly apostrophes (the probe set is 4 lines); overwrite the model field in the committed eval summary; count truncated/errored over review cases too.
21. Task 12 Step 5 pass rule also requires ds4 prefill ≥ oMLX per context; build `--ds4-env` from `LEVERS.md` by an exact procedure (no `<kept-levers>` placeholder).
22. Task 13: spec edits as an exact old/new Python replace script (assert each old string occurs once; the quoted "54.5 … 19 s" text does not match the spec verbatim), including spec §4's "ported from the GLM kernels" sentence and the payload-tag comment "F16 KV only".
23. Task 2 Interfaces must match the code (`guard_free` signature, `interleave` arity, no `Arm` dataclass/`cached_tokens` unless added), record `finish_reason`, close the filler file.

## Review Focus

1. **A prompt cache silently skipping the prefill in the A/B**, so ds4 or oMLX reports an unreal ~∞ prefill t/s. Every A/B prompt carries a fresh nonce at its **start** (both prefix caches match from the start), and the harness asserts `usage.prompt_tokens_details.cached_tokens == 0` on ds4 and `cached_tokens == 0` on oMLX for every request; a nonzero value fails the run. Covered by Task 2, `speed-bench/ornith/tests/test_m4_ab.py::test_cached_tokens_zero_asserted` and `::test_nonce_is_prefix`.
2. **oMLX and a ds4-server loaded at the same time**, which on 64 GB swaps or trips the oMLX 48 GB guard and corrupts both arms' numbers. The harness runs exactly one arm's process at a time (starts the ds4 arm only after the oMLX arm's process is gone and vice versa), checks `pgrep -x omlx-server`, `pgrep -fl` the ds4 pattern and `ps -axo pid,stat,comm | awk '$2 ~ /^(E|U)/'` before each start, and records the swap delta. Covered by Task 2, `test_m4_ab.py::test_one_process_guard_refuses_when_other_up` and `::test_swap_delta_recorded`.
3. **Q5_K tile tails at the tile boundary (T = 65) and the multi-tile / partial-expert case (T = 641)**, where the remainder-tile kernels and the empty-expert lists must still match the per-token row kernels. Covered by Task 4, `tests/test_qwen35_kernels.c::test_moe_mm_q5k` at T ∈ {65, 200, 641}, run once with `DS4_QWEN35_MOE_MM_NAX=0` (simdgroup tiles) and once at the default (tensor tiles, within tolerance of the row kernels and of a double reference), plus the gate-1 chunk-65 run in Task 4 Step 10.
4. **A non-F16 KV mode that passes gate 1 within tolerance but regresses quality**, shipped as a default without re-checking. `DS4_QWEN35_KV` fp8/q4 is never a default until it passes gate 2 again; gate 1 is run unchanged against the committed llama.cpp references, whose calibrated tolerance was recorded against an F16 KV and so measures the quantization error directly. Covered by Task 11, Step 7 (gate 1 per mode, unchanged tolerance) and Step 10 (gate 2 per mode before any default).
5. **The 128K MTP verify reading the whole KV twice**, so MTP loses to oMLX at long context and the shared-KV verify kernel is needed but never built. The Ornith stage profile (Task 5) measures the T=2 verify KV cost at 128K; the shared-KV two-row decode kernel (Task 11, L12) is built only when the trigger fires (MTP still below oMLX at 128K after the KV mode), and its output is bit-checked against the per-row single-split decode. Covered by Task 5 (`DS4_QWEN35_PROFILE` 128K receipt) and Task 11 Step 9 (`test_qwen35_kernels.c::test_attn_decode2_matches_perrow`).

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `.gitignore` | Modify | add `speed-bench/ornith/*/omlx/` |
| `speed-bench/ornith/m4/RUN_NOTES.md` | Create | interface re-verification receipt + precondition ledger |
| `speed-bench/ornith/m4_ab.py` | Create | interleaved ds4-vs-oMLX A/B harness (one process at a time) |
| `speed-bench/ornith/tests/test_m4_ab.py` | Create | unit tests for the harness plumbing (no model, no GPU) |
| `metal/qwen35.metal` | Modify | Q5_K tile GEMM (helpers + kernels via extraction), Q5_K decode kernels (L8), shared-KV 2-row decode (L12, conditional) |
| `ds4_metal.m` | Modify | new kernel enum/names, `ds4_gpu_qwen35_moe_mm_{mid,down}_tensor`, one `||` in the MM specialisation branch, `qwen35_moe_mm_nax`, L8/L12 wrappers, `ds4_gpu_qwen35_attn_prep_tensor` KV-mode args |
| `ds4_gpu.h` | Modify | declarations of the new wrappers |
| `ds4_qwen35moe.inc` | Modify | Q5_K tile dispatch, `DS4_QWEN35_PROFILE`, `DS4_QWEN35_FLUSH_LAYER`, `DS4_QWEN35_FUSE_NORM`, `DS4_QWEN35_ATTN_MULTI_GEMV`, `DS4_QWEN35_KV` alloc/attention, `DS4_QWEN35_MTP_DRAFT_VOCAB` wiring |
| `ds4.c` | Modify | `DS4_QWEN35_KV` env parse + mode-aware memory estimate, `qwen35_mtp_draft_head_load` (via a parameterised shared loader), the M3 Ornith disk-KV payload extended to the FP8/Q4 mode (tags, bytes, save/load) |
| `speed-bench/ornith/m4/extract_q5k_tiles.py` | Create | deterministic Q5_K tile-kernel generator from the Q4_K templates |
| `tests/test_qwen35_kernels.c` | Modify | `test_moe_mm_q5k`, `test_moe_q5k_rows` (L8), `test_attn_prep_kv_modes`, `test_attn_decode2_matches_perrow` |
| `tests/ds4_test.c` | Modify | the M3 `--qwen35-payloads` case loops the KV modes (round-trip + cross-mode refusal) |
| `gguf-tools/frspec_vocab.py` | Reuse (no edit) | generates the Ornith draft vocabulary offline |
| `speed-bench/ornith/m4/ornith-draft-vocab-*.txt` | Generate (git-committed text list) | Ornith MTP draft vocabulary (token-id list, not a model file) |
| `speed-bench/ornith/m4/REPORT.md` | Create | the M4 report |
| `speed-bench/ornith/m4/quality/*.json`, `.../speed/*.json`, `.../probes/*.json`, `.../profile/*.txt` | Generate + commit summaries | receipts (raw per-run dumps git-ignored; summary JSON committed) |

---

### Task 1: Preconditions, interface re-verification, model files

No production code. The deliverable is a written ledger proving M2 and M3 landed with the interfaces this plan consumes, the Qwen gate is green, and both GGUF tiers are present.

**Files:**
- Create: `speed-bench/ornith/m4/RUN_NOTES.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes (from M2, `DECISIONS.md` §2): `ds4_engine_mtp_draft_tokens` returns 2 for Ornith opened with `--mtp`; `ds4_session_qwen35_spec_cycle(...)`; `qwen35_graph_alloc(ds4_qwen4_gpu_graph*, uint32_t ctx_cap, uint32_t cap_tokens, bool mtp)`; `qwen35_graph_mtp(...)`; struct fields `mtp_h`, `mtp_h_pos0`, `mtp_h_rows` on `ds4_qwen4_gpu_graph`; `qwen35_graph_forward_tokens(..., float *logits_out, bool all_rows)`; knobs `DS4_QWEN35_SPEC_TRACE`, `DS4_QWEN35_SPEC_FORCE_ACCEPT`; `--mtp-timing` prints `qwen4_spec_cycles`/`qwen4_spec_accepted` at session free.
- Consumes (from M3, `DECISIONS.md` §3): `ds4-server` serves Ornith (M1 refusal removed); server ids `ornith-1.5-35b-a3b` with `-chat`, `-reasoner`, `-nothink`, `-no-think` aliases; `ds4_engine_is_qwen35moe()`; `DS4_QWEN35_PAYLOAD_TAG`; the server renders the embedded template; `chat_template_kwargs.enable_thinking` honoured.
- Produces: `speed-bench/ornith/m4/RUN_NOTES.md` with each symbol confirmed or a STOP ruling; `$DS4_ORNITH_MODEL` (23G) and `$DS4_ORNITH_MODEL_25G` (25G) both present.

- [ ] **Step 1: Confirm the branch base and Qwen gate are green**

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
git log --oneline -1
git rev-parse --abbrev-ref HEAD          # expect feature/ornith-m4 (cut from develop after M3 merged)
git status --short
```
Expected: HEAD is a develop descendant that contains the M2 and M3 merges. If HEAD does not contain them (grep in Step 2 fails), STOP and record a ledger ruling: M4 cannot start before M2 and M3 merge to develop (`DECISIONS.md` §4.1).

- [ ] **Step 2: Re-verify every consumed M2/M3 symbol by grep**

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
# M2
grep -n 'ds4_session_qwen35_spec_cycle' ds4.c | head
grep -n 'qwen35_graph_mtp' ds4_qwen35moe.inc | head
grep -n 'uint32_t mtp_h_pos0\|uint32_t mtp_h_rows\|ds4_gpu_tensor *\*mtp_h' ds4.c
grep -n 'qwen35_graph_alloc(ds4_qwen4_gpu_graph \*g, uint32_t ctx_cap, uint32_t cap_tokens, bool mtp)' ds4_qwen35moe.inc
grep -n 'qwen35_graph_forward_tokens' ds4_qwen35moe.inc | head
grep -n 'DS4_QWEN35_SPEC_TRACE\|DS4_QWEN35_SPEC_FORCE_ACCEPT' ds4.c ds4_qwen35moe.inc
# M2 open gate + draft count
grep -n 'ds4_engine_mtp_draft_tokens' ds4.c | head
# M3
grep -n 'ds4_engine_is_qwen35moe' ds4.h ds4.c ds4_server.c | head
grep -n 'ornith-1.5-35b-a3b' ds4_server.c | head
grep -n 'DS4_QWEN35_PAYLOAD_TAG' ds4.c | head
grep -n 'milestone M3\|arrives in milestone' ds4_server.c ds4_agent.c   # expect NO hits (refusal removed)
```
Expected: every M2/M3 grep matches; the "milestone M3" refusal grep returns nothing. Record each result in `RUN_NOTES.md`. For each symbol whose signature differs from `DECISIONS.md` §2/§3, STOP with a ledger ruling naming the difference (the plan's later code assumes these signatures).

- [ ] **Step 3: Confirm a staging ds4-server serves Ornith and reports the fields the harness needs**

Ask the user's OK once for this execution window before starting any model process (Global Constraints, GPU windows). With the box free (`ps -axo pid,stat,comm | awk 'NR==1 || /ds4|omlx|llama/'` shows no model process, no state E/U), start a throwaway server and probe it, then stop it:

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
mkdir -p /tmp/m4-kv
caffeinate -i -s ./ds4-server --metal -m "$DS4_ORNITH_MODEL" -c 262144 --mtp \
  --kv-disk-dir /tmp/m4-kv --host 127.0.0.1 --port 18296 > /tmp/m4-probe.log 2>&1 &
SRV=$!
for i in $(seq 1 900); do curl -s -o /dev/null http://127.0.0.1:18296/v1/models && break; sleep 1; done
curl -s http://127.0.0.1:18296/v1/models
curl -s http://127.0.0.1:18296/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"ornith-1.5-35b-a3b","messages":[{"role":"user","content":"nonce-8f3 Say hi."}],
       "max_tokens":8,"temperature":0,"stream":false}' | python3 -m json.tool
kill -TERM $SRV; wait $SRV 2>/dev/null; rm -rf /tmp/m4-kv
```
Expected: `/v1/models` lists `ornith-1.5-35b-a3b` (plus `-chat`, `-reasoner`); the completion returns `usage.prompt_tokens_details.cached_tokens` = 0 on a fresh nonce. Record the exact model id and the `usage` shape in `RUN_NOTES.md`. Never use port 18086/18085/8090. Never `kill -9`.

- [ ] **Step 4: Make sure both GGUF tiers are present (25G download needs the user's OK)**

```bash
ls -l "$HOME/orca/workspaces/ds4-metal-data/gguf/ornith/"Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf
df -h "$HOME" | tail -1
```
The 25G tier (`...-25G-ICE.gguf`, 24,849,784,096 bytes) is needed for gate 2's second tier (`DECISIONS.md` §4.4). It is not downloaded. Model downloads go through HuggingFace per the user's rule and cost disk; **ask the user before downloading**. On the user's OK, and only if `df` shows ≥ 40 GiB free:

```bash
cd "$HOME/orca/workspaces/ds4-metal-data/gguf/ornith"
R=gbuzhf/Ornith-1.5-35B-A3B-Abliterated-CyberTiel-Calibrated-MTPv2-ICE-GGUF
~/.local/omlx-venv/bin/hf download "$R" \
  Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-25G-ICE.gguf --local-dir .
ls -l Ornith-*25G-ICE.gguf   # expect 24849784096 bytes
```
Set and record `DS4_ORNITH_MODEL_25G=$HOME/orca/workspaces/ds4-metal-data/gguf/ornith/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-25G-ICE.gguf`. If the user declines, record in `RUN_NOTES.md` that gate 2 runs on 23G only and 25G is deferred; the report (Task 13) states the same.

- [ ] **Step 5: Ignore per-run oMLX dumps**

Append to `.gitignore`:
```gitignore
# Ornith M4 A/B: keep committed summaries only
speed-bench/ornith/*/omlx/
```

- [ ] **Step 6: Commit the ledger**

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
git add speed-bench/ornith/m4/RUN_NOTES.md .gitignore
git commit -m "speed-bench/ornith: M4 preconditions and interface re-verification

M2 and M3 symbols the M4 plan consumes are grepped and recorded; the
Ornith staging server responds on 18296; GGUF tiers noted.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- speed-bench/ornith/m4/RUN_NOTES.md .gitignore
```

---

### Task 2: The A/B harness `speed-bench/ornith/m4_ab.py`

A new harness that interleaves ds4-server against oMLX, one model process at a time, defeating both prefix caches. It reuses `qwen_gate.py`'s client-side SSE timing by import (`qwen_gate.py` is unchanged). All plumbing is unit-tested without a model or GPU first.

**Files:**
- Create: `speed-bench/ornith/m4_ab.py`, `speed-bench/ornith/tests/test_m4_ab.py`

**Interfaces:**
- Consumes: `qwen_gate.sse_timings`, `qwen_gate.request_rates`, `qwen_gate.chat_stream` (imported); `machine.ds4_running`, `machine.swap_used_mib`; `wired.wait_idle_gib`.
- Produces (module `speed-bench/ornith/m4_ab.py`):
  - `build_prompt(nonce: str, target_tokens: int, filler: str, chars_per_token: float) -> str` — nonce first, then a filler slice sized to hit `target_tokens`.
  - `Arm` dataclass: `name` ("ds4"|"omlx"), `base_url`, `model_id`, `start()`, `stop()`, `ready(timeout)`, `cached_tokens(usage)`.
  - `guard_free(other_names: list[str]) -> None` — raises `SystemExit` if any model process runs or any process is in state E/U.
  - `measure_arm(arm, contexts, filler, max_tokens, warmup) -> dict` — per-context streamed request, records `ttft_s`, `prefill_tps`, `decode_tps`, `cached_tokens`, `completion_tokens`, `finish_reason`, `swap_before/after`.
  - `interleave(make_ds4, make_omlx, contexts, order) -> dict` — A-B-B-A over `("omlx","ds4","ds4","omlx")`, one process at a time.
  - `verdict(summary) -> list[str]` — gate-3 failures: for each context, ds4 mean decode ≥ oMLX mean decode and ds4 mean prefill ≥ oMLX mean prefill; and ds4 cold-31K TTFT ≤ oMLX cold-31K TTFT.

- [ ] **Step 1: Write the failing unit tests**

Create `speed-bench/ornith/tests/test_m4_ab.py`:

```python
"""Unit tests for the Ornith M4 A/B harness plumbing (python3 -m unittest).

No model, no GPU: the process guards, prompt building, cached-tokens assertion,
one-process-at-a-time discipline and the gate-3 verdict are checked with fakes.
"""
import os
import sys
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))                       # speed-bench/ornith
sys.path.insert(0, os.path.join(HERE, "..", "..", "qwen-regression"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import m4_ab as m  # noqa: E402


class PromptTest(unittest.TestCase):
    def test_nonce_is_prefix(self):
        p = m.build_prompt("nonce-abc", 2048, "x" * 200000, 3.46)
        self.assertTrue(p.startswith("nonce-abc"))

    def test_target_tokens_scales_chars(self):
        short = m.build_prompt("n", 2048, "x" * 200000, 3.46)
        long = m.build_prompt("n", 32768, "x" * 2000000, 3.46)
        self.assertLess(len(short), len(long))
        # ~3.46 chars/token: a 2048-token prompt is roughly 7K chars, well under 200K
        self.assertLess(len(short), 20000)


class GuardTest(unittest.TestCase):
    def test_one_process_guard_refuses_when_other_up(self):
        # ds4_running returns a non-empty string => refuse
        with self.assertRaises(SystemExit):
            m.guard_free(ds4_running=lambda: "12345 ds4-server", omlx_running=lambda: "",
                         estate=lambda: "")

    def test_one_process_guard_refuses_when_omlx_up(self):
        with self.assertRaises(SystemExit):
            m.guard_free(ds4_running=lambda: "", omlx_running=lambda: "40263 omlx-server",
                         estate=lambda: "")

    def test_one_process_guard_refuses_on_estate(self):
        with self.assertRaises(SystemExit):
            m.guard_free(ds4_running=lambda: "", omlx_running=lambda: "",
                         estate=lambda: "999 E hung")

    def test_one_process_guard_passes_when_free(self):
        m.guard_free(ds4_running=lambda: "", omlx_running=lambda: "", estate=lambda: "")


class CachedTokensTest(unittest.TestCase):
    def test_cached_tokens_zero_asserted(self):
        usage = {"prompt_tokens_details": {"cached_tokens": 7}}
        with self.assertRaises(m.CacheHit):
            m.assert_cache_cold(usage)

    def test_cached_tokens_zero_ok(self):
        m.assert_cache_cold({"prompt_tokens_details": {"cached_tokens": 0}})
        m.assert_cache_cold({"prompt_tokens": 100})  # no details => treated as 0


class SwapTest(unittest.TestCase):
    def test_swap_delta_recorded(self):
        rec = m.measure_arm(
            _FakeArm(), contexts=[2048], filler="x" * 100000, max_tokens=8, warmup=0,
            swap_used=iter([100.0, 140.0]).__next__,
            wait_idle=lambda: None)
        self.assertEqual(rec["swap_before_mib"], 100.0)
        self.assertEqual(rec["swap_after_mib"], 140.0)
        self.assertEqual(rec["swap_delta_mib"], 40.0)


class VerdictTest(unittest.TestCase):
    def test_pass_when_ds4_faster_everywhere(self):
        summary = {"2048": {"ds4_decode": 90, "omlx_decode": 65, "ds4_prefill": 2100, "omlx_prefill": 2000},
                   "cold31k": {"ds4_ttft_s": 14.0, "omlx_ttft_s": 15.2}}
        self.assertEqual(m.verdict(summary), [])

    def test_fail_when_ds4_decode_slower(self):
        summary = {"2048": {"ds4_decode": 60, "omlx_decode": 65, "ds4_prefill": 2100, "omlx_prefill": 2000},
                   "cold31k": {"ds4_ttft_s": 14.0, "omlx_ttft_s": 15.2}}
        self.assertTrue(any("decode" in f for f in m.verdict(summary)))

    def test_fail_when_cold_ttft_worse(self):
        summary = {"cold31k": {"ds4_ttft_s": 20.0, "omlx_ttft_s": 15.2}}
        self.assertTrue(any("TTFT" in f for f in m.verdict(summary)))


class _FakeArm:
    name = "ds4"
    base_url = "http://127.0.0.1:18296"
    model_id = "ornith-1.5-35b-a3b"

    def start(self):
        pass

    def stop(self):
        pass

    def ready(self, timeout):
        return True

    def stream(self, prompt, max_tokens):
        # (timing dict like sse_timings, usage dict) — cold cache, fixed rates
        return ({"prompt_tokens": 2048, "completion_tokens": 8, "ttft_s": 1.0, "decode_s": 0.1},
                {"prompt_tokens_details": {"cached_tokens": 0}})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `cd /Users/dongnh/orca/workspaces/ds4-metal/foxface && python3 -m unittest speed-bench/ornith/tests/test_m4_ab.py -v`
Expected: `ModuleNotFoundError: No module named 'm4_ab'`.

- [ ] **Step 3: Implement `speed-bench/ornith/m4_ab.py`**

```python
#!/usr/bin/env python3
"""Interleaved ds4-vs-oMLX A/B for Ornith (spec §8 gate 3, DECISIONS.md §4.2).

One model process at a time on this 64 GB box: the ds4 arm starts only after
the oMLX arm's process is gone and vice versa. Every prompt carries a fresh
nonce at its START (defeats both prefix caches) and cached_tokens==0 is
asserted. Client-side timing is qwen_gate.sse_timings (imported, unchanged).

  m4_ab.py --ds4-model PATH --out DIR [--contexts 2048,32768,131072] [--cold-tokens 31000]
           [--omlx-attach] [--max-tokens 256] [--warmup 1]

Arms:
  ds4  : Popen an Ornith ds4-server on 18296 with --mtp; model id from /v1/models.
  omlx : Popen the registry omlx process_command (default), or --omlx-attach to
         use an already-running :18085 (the caller pauses the watchdogs first).

Pausing the live oMLX / hitting :18085 touches the live stack: the CALLER gets
the user's OK and runs the launchctl-unload recipe; this script never unloads
LaunchAgents and never edits the registry.
"""
import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "speed-bench", "qwen-regression"))
sys.path.insert(0, os.path.join(ROOT, "speed-bench", "lib"))
import qwen_gate  # noqa: E402  (sse_timings, request_rates)
import machine    # noqa: E402
import wired       # noqa: E402

DS4_PORT = 18296
OMLX_PORT = 18085
OMLX_MODEL = "Shiftedx--ornith-1.5-35b-a3b-abliterated-attention8-bf16recurrence-vision-mtplx"
FILLER = os.path.join(ROOT, "speed-bench", "promessi_sposi.txt")
AB_ORDER = ("omlx", "ds4", "ds4", "omlx")


class CacheHit(Exception):
    pass


def cached_tokens(usage):
    det = (usage or {}).get("prompt_tokens_details") or {}
    return int(det.get("cached_tokens", 0) or 0)


def assert_cache_cold(usage):
    if cached_tokens(usage) != 0:
        raise CacheHit("cached_tokens=%d (prefix cache defeated the prefill)" % cached_tokens(usage))


def build_prompt(nonce, target_tokens, filler, chars_per_token):
    chars = max(0, int(round(target_tokens * chars_per_token)) - len(nonce) - 40)
    if chars > len(filler):
        raise ValueError("filler has %d chars, need %d for %d tokens" % (len(filler), chars, target_tokens))
    return nonce + " " + filler[:chars] + "\n\nSummarize the text above in two sentences."


def guard_free(ds4_running=None, omlx_running=None, estate=None):
    ds4_running = ds4_running or machine.ds4_running
    omlx_running = omlx_running or _omlx_running
    estate = estate or _procs_in_estate
    d, o, e = ds4_running(), omlx_running(), estate()
    if d:
        raise SystemExit("m4_ab: a ds4 process is running; the box must be free:\n" + d)
    if o:
        raise SystemExit("m4_ab: an omlx process is running; the box must be free:\n" + o)
    if e:
        raise SystemExit("m4_ab: a process is in state E/U (wedged Metal); reboot before running:\n" + e)


def _omlx_running():
    p = subprocess.run(["pgrep", "-fl", "omlx-server"], capture_output=True, text=True)
    return p.stdout.strip()


def _procs_in_estate():
    p = subprocess.run(["ps", "-axo", "pid,stat,comm"], capture_output=True, text=True)
    return "\n".join(l for l in p.stdout.splitlines()[1:] if re.match(r"\s*\d+\s+[EU]", l))


def calibrate_chars_per_token(arm):
    """One max_tokens=1 request; usage.prompt_tokens over the prompt char count."""
    probe = "nonce-cal " + open(FILLER, encoding="utf-8", errors="replace").read()[:20000]
    _, usage = arm.stream(probe, max_tokens=1)
    pt = int((usage or {}).get("prompt_tokens", 0) or 0)
    return (len(probe) / pt) if pt > 0 else 3.46


class Ds4Arm:
    name = "ds4"
    base_url = "http://127.0.0.1:%d" % DS4_PORT

    def __init__(self, model_path, out, extra_env=None, extra_args=None):
        self.model_path = model_path
        self.out = out
        self.extra_env = extra_env or {}
        self.extra_args = extra_args or []
        self.proc = None
        self.model_id = "ornith-1.5-35b-a3b"

    def start(self):
        kv = os.path.join(self.out, "ds4-kv")
        shutil.rmtree(kv, ignore_errors=True)
        os.makedirs(kv, exist_ok=True)
        cmd = ["caffeinate", "-i", "-s", os.path.join(ROOT, "ds4-server"), "--metal",
               "-m", self.model_path, "-c", "262144", "--mtp",
               "--kv-disk-dir", kv, "--host", "127.0.0.1", "--port", str(DS4_PORT)] + self.extra_args
        log = open(os.path.join(self.out, "ds4-server.log"), "w")
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                     env={**os.environ, **self.extra_env})

    def ready(self, timeout=900):
        for _ in range(timeout):
            if self.proc.poll() is not None:
                raise SystemExit("m4_ab: ds4-server exited during startup, see ds4-server.log")
            try:
                urllib.request.urlopen(self.base_url + "/v1/models", timeout=2)
                return True
            except OSError:
                time.sleep(1)
        raise SystemExit("m4_ab: ds4-server did not come up in %d s" % timeout)

    def stream(self, prompt, max_tokens):
        return _chat_stream(self.base_url, self.model_id, prompt, max_tokens)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()   # SIGKILL of our OWN clean process on shutdown only; never -9 a wedged one


class OmlxArm:
    name = "omlx"
    base_url = "http://127.0.0.1:%d" % OMLX_PORT

    def __init__(self, out, attach=False):
        self.out = out
        self.attach = attach
        self.proc = None
        self.model_id = OMLX_MODEL

    def _registry_command(self):
        reg = os.path.expanduser(os.environ.get("DS4_GATEWAY_REGISTRY",
                                                 "~/.local/ai-gateway/runtime-registry.json"))
        models = json.load(open(reg))["models"]
        entry = models[OMLX_MODEL]["runtimes"]["omlx"]
        return list(entry["process_command"])

    def start(self):
        if self.attach:
            return
        log = open(os.path.join(self.out, "omlx-server.log"), "w")
        self.proc = subprocess.Popen(["caffeinate", "-i", "-s"] + self._registry_command(),
                                     stdout=log, stderr=subprocess.STDOUT)

    def ready(self, timeout=900):
        for _ in range(timeout):
            try:
                st = json.loads(urllib.request.urlopen(self.base_url + "/api/status", timeout=2).read())
                if any(OMLX_MODEL in m for m in (st.get("loaded_models") or [])):
                    return True
            except OSError:
                pass
            except Exception:
                pass
            if self.proc and self.proc.poll() is not None:
                raise SystemExit("m4_ab: omlx exited during startup, see omlx-server.log")
            time.sleep(1)
        # attach mode: warm it with a tiny request so the model loads
        try:
            _chat_stream(self.base_url, self.model_id, "nonce-warm hi", 1)
            return True
        except Exception:
            raise SystemExit("m4_ab: omlx not ready in %d s" % timeout)

    def stream(self, prompt, max_tokens):
        return _chat_stream(self.base_url, self.model_id, prompt, max_tokens)

    def stop(self):
        if self.attach:
            return
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def _chat_stream(base, model, text, max_tokens):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": text}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": True,
                       "chat_template_kwargs": {"enable_thinking": True},
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    events = []
    t0 = time.time()
    usage = {}
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                events.append((time.time() - t0, chunk))
    return qwen_gate.sse_timings(events), usage


def measure_arm(arm, contexts, filler, max_tokens, warmup=1, cold_tokens=0,
                swap_used=None, wait_idle=None):
    swap_used = swap_used or machine.swap_used_mib
    wait_idle = wait_idle or (lambda: wired.wait_idle_gib())
    wait_idle()
    swap_before = swap_used()
    cpt = getattr(arm, "_cpt", None)
    if cpt is None and hasattr(arm, "stream"):
        cpt = calibrate_chars_per_token(arm)
        arm._cpt = cpt
    for _ in range(warmup):
        arm.stream(build_prompt("nonce-warm", 512, filler, cpt), max_tokens)
    rows = {}
    targets = list(contexts) + (["cold31k"] if cold_tokens else [])
    for ctx in targets:
        n = cold_tokens if ctx == "cold31k" else ctx
        nonce = "nonce-%s-%d" % (arm.name, int(time.time() * 1000) % 100000)
        timing, usage = arm.stream(build_prompt(nonce, n, filler, cpt), max_tokens)
        assert_cache_cold(usage)
        rates = qwen_gate.request_rates(timing)
        rows[str(ctx)] = {**timing, **rates, "cached_tokens": cached_tokens(usage)}
    swap_after = swap_used()
    return {"arm": arm.name, "rows": rows, "chars_per_token": cpt,
            "swap_before_mib": swap_before, "swap_after_mib": swap_after,
            "swap_delta_mib": swap_after - swap_before}


def interleave(make_ds4, make_omlx, contexts, cold_tokens, filler, max_tokens, warmup, order=AB_ORDER):
    runs = {"omlx": [], "ds4": []}
    for which in order:
        guard_free()
        arm = make_omlx() if which == "omlx" else make_ds4()
        arm.start()
        try:
            arm.ready()
            runs[which].append(measure_arm(arm, contexts, filler, max_tokens, warmup, cold_tokens))
        finally:
            arm.stop()
        for _ in range(120):
            if not machine.ds4_running() and not _omlx_running():
                break
            time.sleep(1)
    return _summarize(runs, contexts, cold_tokens)


def _summarize(runs, contexts, cold_tokens):
    summary = {"runs": runs}
    keys = [str(c) for c in contexts] + (["cold31k"] if cold_tokens else [])
    for ctx in keys:
        row = {}
        for side in ("ds4", "omlx"):
            dec = [r["rows"][ctx]["decode_tps"] for r in runs[side] if ctx in r["rows"]]
            pre = [r["rows"][ctx]["prefill_tps"] for r in runs[side] if ctx in r["rows"]]
            ttft = [r["rows"][ctx]["ttft_s"] for r in runs[side] if ctx in r["rows"]]
            if dec:
                row[side + "_decode"] = statistics.fmean(dec)
                row[side + "_prefill"] = statistics.fmean(pre)
                row[side + "_ttft_s"] = statistics.fmean(ttft)
        summary[ctx] = row
    return summary


def verdict(summary):
    failures = []
    for ctx, row in summary.items():
        if ctx in ("runs",):
            continue
        if ctx == "cold31k":
            if "ds4_ttft_s" in row and "omlx_ttft_s" in row and row["ds4_ttft_s"] > row["omlx_ttft_s"]:
                failures.append("cold31k: ds4 TTFT %.2fs > oMLX %.2fs" % (row["ds4_ttft_s"], row["omlx_ttft_s"]))
            continue
        if "ds4_decode" in row and "omlx_decode" in row and row["ds4_decode"] < row["omlx_decode"]:
            failures.append("%s: ds4 decode %.1f < oMLX %.1f t/s" % (ctx, row["ds4_decode"], row["omlx_decode"]))
        if "ds4_prefill" in row and "omlx_prefill" in row and row["ds4_prefill"] < row["omlx_prefill"]:
            failures.append("%s: ds4 prefill %.0f < oMLX %.0f t/s" % (ctx, row["ds4_prefill"], row["omlx_prefill"]))
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds4-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--contexts", default="2048,32768,131072")
    ap.add_argument("--cold-tokens", type=int, default=31000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--omlx-attach", action="store_true")
    ap.add_argument("--ds4-env", default="")   # k=v,k=v applied to the ds4 arm only
    ap.add_argument("--ds4-args", default="")  # space-joined extra ds4-server args
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "ds4"), exist_ok=True)
    os.makedirs(os.path.join(out, "omlx"), exist_ok=True)
    contexts = [int(x) for x in args.contexts.split(",") if x]
    env = dict(kv.split("=", 1) for kv in args.ds4_env.split(",") if "=" in kv)
    extra_args = args.ds4_args.split() if args.ds4_args else []
    filler = open(FILLER, encoding="utf-8", errors="replace").read()
    summary = interleave(lambda: Ds4Arm(args.ds4_model, os.path.join(out, "ds4"), env, extra_args),
                         lambda: OmlxArm(os.path.join(out, "omlx"), attach=args.omlx_attach),
                         contexts, args.cold_tokens, filler, args.max_tokens, args.warmup)
    summary["failures"] = verdict(summary)
    with open(os.path.join(out, "m4_ab.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=1)
    for ctx in [str(c) for c in contexts] + ["cold31k"]:
        row = summary.get(ctx, {})
        print("m4_ab: %-8s ds4 decode %.1f / oMLX %.1f  ds4 prefill %.0f / oMLX %.0f" % (
            ctx, row.get("ds4_decode", 0), row.get("omlx_decode", 0),
            row.get("ds4_prefill", 0), row.get("omlx_prefill", 0)))
    for f in summary["failures"]:
        print("m4_ab: FAIL", f)
    print("m4_ab:", "FAIL" if summary["failures"] else "PASS")
    return 1 if summary["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `cd /Users/dongnh/orca/workspaces/ds4-metal/foxface && python3 -m unittest speed-bench/ornith/tests/test_m4_ab.py -v`
Expected: `Ran 12 tests ... OK`.

- [ ] **Step 5: Commit**

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
git add speed-bench/ornith/m4_ab.py speed-bench/ornith/tests/test_m4_ab.py
git commit -m "speed-bench/ornith: interleaved ds4-vs-oMLX A/B harness for gate 3

One model process at a time, nonce-prefixed prompts, cached_tokens==0
asserted, A-B-B-A per context. Reuses qwen_gate timing helpers unchanged.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- speed-bench/ornith/m4_ab.py speed-bench/ornith/tests/test_m4_ab.py
```

---

### Task 3: Baselines — live oMLX and the M1-resident ds4

Measurement only. Captures the live oMLX numbers (the gate-3 bar) and the current ds4 numbers (before the Q5_K tile GEMM), so the report can show the before/after. Uses the live :18085 and pauses the stack — **needs the user's OK once for this window** (Global Constraints).

**Files:**
- Create: `speed-bench/ornith/m4/speed/baseline.json`, `speed-bench/ornith/m4/speed/BASELINE.md`

**Interfaces:**
- Consumes: `speed-bench/ornith/m4_ab.py` (Task 2).
- Produces: `speed/baseline.json` (oMLX and M1-ds4 rows at 2K/32K/128K + cold-31K).

- [ ] **Step 1: Get the user's OK and pause the live stack**

State plainly: this run pauses the live gateway and oMLX (`launchctl unload` the three plists, `kill -TERM` the omlx pid), then the harness owns both processes for a clean 128K A/B, and restores at the end. Wait for the user's yes. Then:

```bash
launchctl unload ~/Library/LaunchAgents/dev.dongnh.gateway-watchdog.plist \
  ~/Library/LaunchAgents/dev.dongnh.gateway-eval.plist \
  ~/Library/LaunchAgents/dev.dongnh.ai-proxy.plist
kill -TERM "$(cat ~/.local/share/ai-gateway/omlx.pid)" 2>/dev/null || true
for i in $(seq 1 60); do pgrep -x omlx-server >/dev/null || break; sleep 1; done
ps -axo pid,stat,comm | awk 'NR==1 || /ds4|omlx|llama/'
```
Expected: no `omlx-server`, no `ds4` process. If a process is stuck in state E/U, STOP and tell the user a reboot is needed (`metal-kill9-wedges-gguf-vnode`); do not `kill -9`.

- [ ] **Step 2: Run the interleaved baseline (M1-resident ds4 vs oMLX)**

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
S=speed-bench/ornith/m4/speed/baseline
caffeinate -i -s python3 speed-bench/ornith/m4_ab.py \
  --ds4-model "$DS4_ORNITH_MODEL" --out "$S" \
  --contexts 2048,32768,131072 --cold-tokens 31000 --max-tokens 256 --warmup 1 \
  | tee speed-bench/ornith/m4/speed/baseline.txt
cp "$S/m4_ab.json" speed-bench/ornith/m4/speed/baseline.json
```
Expected: a table for 2K/32K/128K + cold31k with ds4 and oMLX decode/prefill. `cached_tokens` = 0 everywhere (a `CacheHit` aborts). This baseline is **expected to FAIL gate 3 on prefill** at 32K/128K (Q5_K row kernels are slow — the whole reason for Task 4); record that as the "before" number, not a stop.

- [ ] **Step 3: Restore the live stack**

```bash
launchctl load ~/Library/LaunchAgents/dev.dongnh.ai-proxy.plist \
  ~/Library/LaunchAgents/dev.dongnh.gateway-watchdog.plist \
  ~/Library/LaunchAgents/dev.dongnh.gateway-eval.plist
for i in $(seq 1 120); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8090/status)" = 200 ] && break; sleep 1; done
curl -s -o /dev/null -w 'gateway status %{http_code}\n' http://127.0.0.1:8090/status
```
Expected: `gateway status 200`. If not 200 after 120 s, report to the user; do not leave the stack down.

- [ ] **Step 4: Write and commit the baseline receipt**

Create `speed-bench/ornith/m4/speed/BASELINE.md` recording, per context: ds4 (M1-resident, Q5_K row kernels) decode/prefill/TTFT and oMLX decode/prefill/TTFT, the swap deltas, and the note that ds4 prefill is expected below oMLX until the Q5_K tile GEMM lands. Include the exact `m4_ab.py` command line and the git HEAD.

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
git add speed-bench/ornith/m4/speed/baseline.json speed-bench/ornith/m4/speed/BASELINE.md
git commit -m "speed-bench/ornith: M4 baseline (M1-resident ds4 vs live oMLX)

Live oMLX is the gate-3 bar; M1-resident ds4 prefill is below it at 32K/128K
(Q5_K row kernels) — the Q5_K tile GEMM in the next task closes that.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- speed-bench/ornith/m4/speed/baseline.json speed-bench/ornith/m4/speed/BASELINE.md
```

---

### Task 4: Q5_K tiled prefill GEMM (mandatory)

The one required kernel: Q5_K routed experts get the tiled expert GEMM the Q4_K layers already use, so a 31K prompt stops spending ≥ 28 s in the 15 Q5_K layers. Additive entry points in `metal/qwen35.metal`; new host wrappers; one `||` in the shared MM specialisation branch (the only shared-file kernel-dispatch edit); one condition in `ds4_qwen35moe.inc`. `metal/qwen4.metal` is not edited.

**Files:**
- Modify: `metal/qwen35.metal` (helpers + extracted kernels), `ds4_metal.m` (enum/names ~49528–49530 and ~49653–49655, one `||` at the branch ~49769, `qwen35_moe_mm_nax`, wrappers after `ds4_gpu_qwen35_moe_down_tensor`), `ds4_gpu.h`, `ds4_qwen35moe.inc` (`qwen35_graph_moe` line ~155), `tests/test_qwen35_kernels.c`, `Makefile` (none; the target exists)
- Create (generated into `metal/qwen35.metal`): the four tile kernels via the extraction script

**Interfaces:**
- Produces (C, `ds4_gpu.h`), same shapes as the Q4_K tile wrappers:
  ```c
  int ds4_gpu_qwen35_moe_mm_mid_tensor(
          ds4_gpu_tensor *mid, const ds4_gpu_tensor *x, const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts,
          const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset,
          uint32_t weight_type, uint32_t n_expert, uint32_t n_tokens, uint32_t n_slots, uint32_t n_out,
          uint32_t in_dim, uint32_t ff_dim, uint32_t list_cap);
  int ds4_gpu_qwen35_moe_mm_down_tensor(
          ds4_gpu_tensor *part, const ds4_gpu_tensor *mid, const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts,
          const void *model_map, uint64_t model_size, uint64_t down_offset,
          uint32_t weight_type, uint32_t n_expert, uint32_t n_tokens, uint32_t n_slots, uint32_t n_out,
          uint32_t ff_dim, uint32_t out_dim, uint32_t list_cap);
  ```
- Produces (Metal): `qwen35_mm_stage16_q5k`, `struct qwen35_raw_q5k`, `qwen35_load_raw_q5k`, `qwen35_dequant_raw_q5k`, and kernels `kernel_qwen35_moe_mm_mid_q5k{,_nt1,_nt2,_nt8}`, `kernel_qwen35_moe_mm_down_q5k{,_nt1,_nt2,_nt8}`, `kernel_qwen35_moe_mm_mid_q5k_nax{,64}`, `kernel_qwen35_moe_mm_down_q5k_nax{,64}`.

- [ ] **Step 1: Write the failing kernel test** (append to `tests/test_qwen35_kernels.c`, before `int main`)

`arena_q5_K` already exists in this file (M1). Add the tile test, modelled on `tests/test_qwen4_kernels.c::test_moe_mm_tiles_exact` (T=641, hot/partial/empty experts) but with Q5_K gate/up/down and the M1 `arena_q5_K` packer. It compares the tiles to the per-token row kernels (`ds4_gpu_qwen35_moe_{mid,down}_tensor`, no shared slot) and to a double reference, at both NAX levels.

```c
/* Q5_K routed experts through the tiled GEMM: the simdgroup tiles
 * (DS4_QWEN35_MOE_MM_NAX=0) are the exact reference for the tensor tiles, and
 * both are bounded against the double reference and against the M1 per-token
 * row kernels.  T=65 exercises the first tile tail, T=641 the multi-tile,
 * partial-expert and empty-expert cases (cf. test_moe_mm_tiles_exact). */
static void test_moe_mm_q5k(arena_t *a, uint32_t T) {
    const uint32_t E = 256, F = 256, NE = 4, slots = 2, n_out = slots, list_cap = T + 7, guard = 16;
    const uint64_t mid_n = (uint64_t)T * n_out * F, part_n = (uint64_t)T * n_out * E;
    const char *keys[] = {"DS4_QWEN35_MOE_MID_TILES", "DS4_QWEN35_MOE_DOWN_TILES", "DS4_QWEN35_MOE_MM_NAX"};
    char *saved[3];
    for (uint32_t i = 0; i < 3; i++) { const char *v = getenv(keys[i]); saved[i] = v ? strdup(v) : NULL; }
    double *gate_w, *up_w, *down_w;
    const uint64_t gate_off = arena_q5_K(a, (uint64_t)NE * F, E, &gate_w, 0.05f);
    const uint64_t up_off = arena_q5_K(a, (uint64_t)NE * F, E, &up_w, 0.05f);
    const uint64_t down_off = arena_q5_K(a, (uint64_t)NE * E, F, &down_w, 0.05f);
    float *x = rand_vec((uint64_t)T * E, 2.0f);
    int32_t *sel = malloc((uint64_t)T * slots * sizeof(int32_t));
    for (uint32_t t = 0; t < T; t++) { sel[t * slots] = 0; sel[t * slots + 1] = (int32_t)(1u + t % 2u); }
    double *mid_ex = malloc(mid_n * sizeof(double)), *part_ex = malloc(part_n * sizeof(double));
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t s = 0; s < slots; s++) {
            const uint32_t e = (uint32_t)sel[t * slots + s];
            for (uint32_t f = 0; f < F; f++) {
                double g = 0.0, u = 0.0;
                for (uint32_t k = 0; k < E; k++) {
                    g += gate_w[((uint64_t)e * F + f) * E + k] * (double)x[(uint64_t)t * E + k];
                    u += up_w[((uint64_t)e * F + f) * E + k] * (double)x[(uint64_t)t * E + k];
                }
                mid_ex[((uint64_t)t * n_out + s) * F + f] = silu_d(g) * u;
            }
            for (uint32_t d = 0; d < E; d++) {
                double p = 0.0;
                for (uint32_t k = 0; k < F; k++) p += down_w[((uint64_t)e * E + d) * F + k] * mid_ex[((uint64_t)t * n_out + s) * F + k];
                part_ex[((uint64_t)t * n_out + s) * E + d] = p;
            }
        }
    ds4_gpu_tensor *gx = upload(x, (uint64_t)T * E);
    ds4_gpu_tensor *gsel = ds4_gpu_tensor_alloc((uint64_t)T * slots * 4);
    ds4_gpu_tensor *glists = ds4_gpu_tensor_alloc((uint64_t)NE * list_cap * 4);
    ds4_gpu_tensor *gcounts = ds4_gpu_tensor_alloc(NE * 4);
    require_ok(gsel && glists && gcounts && ds4_gpu_tensor_write(gsel, 0, sel, (uint64_t)T * slots * 4), "q5k sel");
    require_ok(ds4_gpu_qwen4_moe_build_lists_tensor(glists, gcounts, gsel, T, slots, NE, list_cap), "q5k lists");
    ds4_gpu_tensor *gmid = upload(NULL, mid_n + guard), *gpart = upload(NULL, part_n + guard);
    const uint32_t levels[] = {0u, 64u};   /* 0 = simdgroup tiles; 64 = tensor tiles */
    float *ref_mid = NULL, *ref_part = NULL;
    const float sentinel = -1234.5f;
    for (uint32_t li = 0; li < 2u; li++) {
        char lv[4]; snprintf(lv, sizeof(lv), "%u", levels[li] ? 2u : 0u);   /* NAX level 2 => 64-tok tiles */
        setenv("DS4_QWEN35_MOE_MM_NAX", lv, 1);
        for (uint32_t i = 0; i < 2; i++) setenv(keys[i], "8", 1);
        require_ok(ds4_gpu_tensor_fill_f32(gmid, sentinel, mid_n + guard) &&
                   ds4_gpu_tensor_fill_f32(gpart, sentinel, part_n + guard), "q5k sentinels");
        const bool have = ds4_gpu_qwen35_moe_mm_mid_tensor(gmid, gx, glists, gcounts, a->base, a->size,
                                gate_off, up_off, 13u, NE, T, slots, n_out, E, F, list_cap);
        if (!have && levels[li]) { printf("  q5k tiles nax=%u unavailable, skipped\n", levels[li]); continue; }
        require_ok(have, "q5k mm mid");
        require_ok(ds4_gpu_qwen35_moe_mm_down_tensor(gpart, gmid, glists, gcounts, a->base, a->size,
                                down_off, 13u, NE, T, slots, n_out, F, E, list_cap), "q5k mm down");
        float *gm = download(gmid, mid_n + guard), *gp = download(gpart, part_n + guard);
        for (uint64_t i = mid_n; i < mid_n + guard; i++) require_ok(gm[i] == sentinel, "q5k mid tail guard");
        for (uint64_t i = part_n; i < part_n + guard; i++) require_ok(gp[i] == sentinel, "q5k down tail guard");
        double ew = 0.0, sc = 1e-6;
        for (uint64_t i = 0; i < part_n; i++) { double d = fabs((double)gp[i] - part_ex[i]); if (d > ew) ew = d; if (fabs(part_ex[i]) > sc) sc = fabs(part_ex[i]); }
        char what[64]; snprintf(what, sizeof(what), "q5k tiles nax=%u down vs exact T=%u", levels[li], T);
        printf("  %-40s max|d|=%.3e (rel %.3e)\n", what, ew, ew / sc);
        require_ok(ew <= 3e-3 * sc, what);
        if (levels[li] == 0u) { ref_mid = gm; ref_part = gp; }   /* simdgroup = exact reference */
        else {
            double wm = 0.0, sm = 1e-6;
            for (uint64_t i = 0; i < mid_n; i++) { if (ref_mid[i] == sentinel) continue; double d = fabs((double)gm[i] - ref_mid[i]); if (d > wm) wm = d; if (fabs(ref_mid[i]) > sm) sm = fabs(ref_mid[i]); }
            require_ok(wm <= 2e-3 * sm, "q5k tensor mid within 2e-3 of simdgroup");
            free(gm); free(gp);
        }
    }
    /* cross-check the simdgroup tiles against the M1 per-token row kernels (no shared slot). */
    setenv("DS4_QWEN35_MOE_MM_NAX", "0", 1);
    ds4_gpu_tensor *rmid = upload(NULL, mid_n), *rpart = upload(NULL, part_n);
    require_ok(ds4_gpu_qwen35_moe_mid_tensor(rmid, gx, gsel, a->base, a->size, gate_off, up_off, 13u,
                                             NE, T, slots, E, F, 0, 0, UINT32_MAX), "q5k row mid");
    require_ok(ds4_gpu_qwen35_moe_down_tensor(rpart, rmid, gsel, a->base, a->size, down_off, 13u,
                                              NE, T, slots, F, E, 0, UINT32_MAX), "q5k row down");
    float *rp = download(rpart, part_n);
    double rw = 0.0, rs = 1e-6;
    for (uint64_t i = 0; i < part_n; i++) { double d = fabs((double)rp[i] - ref_part[i]); if (d > rw) rw = d; if (fabs(ref_part[i]) > rs) rs = fabs(ref_part[i]); }
    printf("  q5k tiles vs row kernels down T=%u: max|d|=%.3e (rel %.3e)\n", T, rw, rw / rs);
    require_ok(rw <= 2e-3 * rs, "q5k tiles match row kernels");
    for (uint32_t i = 0; i < 3; i++) { if (saved[i]) { setenv(keys[i], saved[i], 1); free(saved[i]); } else unsetenv(keys[i]); }
    free(x); free(sel); free(mid_ex); free(part_ex); free(gate_w); free(up_w); free(down_w); free(ref_mid); free(rp);
    ds4_gpu_tensor_free(gx); ds4_gpu_tensor_free(gsel); ds4_gpu_tensor_free(glists); ds4_gpu_tensor_free(gcounts);
    ds4_gpu_tensor_free(gmid); ds4_gpu_tensor_free(gpart); ds4_gpu_tensor_free(rmid); ds4_gpu_tensor_free(rpart);
}
```

Add to `main`, after the existing `test_moe_q5k` calls:
```c
    printf("qwen35 moe q5_K tiled GEMM (simdgroup + tensor tiles)\n");
    test_moe_mm_q5k(&arena, 65);
    test_moe_mm_q5k(&arena, 200);
    test_moe_mm_q5k(&arena, 641);
```

- [ ] **Step 2: Build and confirm the test fails**

Run: `make test-qwen35-kernels`
Expected: a compile error — `ds4_gpu_qwen35_moe_mm_mid_tensor` is implicitly declared / undeclared in `tests/test_qwen35_kernels.c` (the wrapper and its `ds4_gpu.h` prototype do not exist yet). The test object fails to compile before any link step.

- [ ] **Step 3: Add the Q5_K staging helpers to `metal/qwen35.metal`**

Append after the M1 Q5_K row kernels (after `kernel_qwen35_moe_down`, before the GDN-out section), so they sit before the extracted kernels. These are new `static inline` helpers; they mirror the Q4_K `qwen4_mm_stage16` / `qwen4_raw16` shape but read the 176-byte Q5_K super-block with its extra high-bit plane. Verified element-for-element against `qwen35_row_dot` (this file, M1).

```metal
/* --- Q5_K tiled prefill GEMM (additive; qwen4.metal untouched) ----------- */

/* Stage 16 consecutive Q5_K values (quarters q0 and q0+1 of 32-block b, q0
 * even) as halves.  Same coordinate as qwen4_mm_stage16's Q4_K branch:
 * sb = b/8, group = b%8, l = q0*8.  Q5_K super-block is 176 bytes: d, dmin,
 * 12 packed 6-bit scale/min bytes, 32 high-bit bytes, 128 nibble bytes.  The
 * 6-bit scale/min unpack is identical to Q4_K; the value is the low nibble at
 * qs plus the group-th bit of qh as the fifth bit. */
template <typename D>
static inline void qwen35_mm_stage16_q5k(device const char *row, uint b, uint q0, uint type, threadgroup D *dst) {
    (void)type;
    const uint sb = b / 8, group = b % 8;
    device const uchar *blk = (device const uchar *)(row + (uint64_t)sb * 176);
    const float d = (float)(*(device const half *)blk);
    const float dmin = (float)(*(device const half *)(blk + 2));
    device const uchar *sc = blk + 4;
    uint s, mn;
    if (group < 4) { s = sc[group] & 63u; mn = sc[group + 4] & 63u; }
    else { s = (sc[group + 4] & 0xFu) | ((sc[group - 4] & 0xC0u) >> 2); mn = (sc[group + 4] >> 4) | ((sc[group] & 0xC0u) >> 2); }
    const float ds = d * (float)s, dm = dmin * (float)mn;
    const uint4 v = *(device const uint4 *)(blk + 48 + (group >> 1) * 32 + q0 * 8);   /* 16 nibble bytes */
    const uint4 h = *(device const uint4 *)(blk + 16 + q0 * 8);                        /* 16 high-bit bytes */
    const uint shift = (group & 1u) * 4u;
    for (uint i = 0; i < 16; i++) {
        const uint lo = (v[i >> 2] >> (8u * (i & 3u) + shift)) & 0xFu;
        const uint hi = (h[i >> 2] >> (8u * (i & 3u) + group)) & 1u;
        dst[i] = (D)(ds * (float)(lo | (hi << 4)) - dm);
    }
}

/* Register prefetch for the tensor-op tiles: 16-byte header (d|dmin|scales),
 * the 16 nibble bytes and the 16 high-bit bytes of one 32-block quarter-pair. */
struct qwen35_raw_q5k { uint4 hdr; uint4 q; uint4 qh; };
static inline qwen35_raw_q5k qwen35_load_raw_q5k(device const char *row, uint b, uint q0, uint type) {
    (void)type;
    const uint sb = b / 8, group = b % 8;
    device const uchar *blk = (device const uchar *)(row + (uint64_t)sb * 176);
    qwen35_raw_q5k r;
    r.hdr = *(device const uint4 *)blk;
    r.q   = *(device const uint4 *)(blk + 48 + (group >> 1) * 32 + q0 * 8);
    r.qh  = *(device const uint4 *)(blk + 16 + q0 * 8);
    return r;
}
static inline void qwen35_dequant_raw_q5k(qwen35_raw_q5k r, uint b, uint q0, uint type, threadgroup half *dst) {
    (void)q0; (void)type;
    const uint group = b % 8;
    const float d = (float)as_type<half>((ushort)(r.hdr.x & 0xFFFFu));
    const float dmin = (float)as_type<half>((ushort)(r.hdr.x >> 16));
    const uint scw[3] = { r.hdr.y, r.hdr.z, r.hdr.w };
#define QWEN35_SCB(i) ((scw[(i) >> 2] >> (8u * ((i) & 3u))) & 0xFFu)
    uint s, mn;
    if (group < 4) { s = QWEN35_SCB(group) & 63u; mn = QWEN35_SCB(group + 4) & 63u; }
    else { s = (QWEN35_SCB(group + 4) & 0xFu) | ((QWEN35_SCB(group - 4) & 0xC0u) >> 2); mn = (QWEN35_SCB(group + 4) >> 4) | ((QWEN35_SCB(group) & 0xC0u) >> 2); }
#undef QWEN35_SCB
    const float ds = d * (float)s, dm = dmin * (float)mn;
    const uint shift = (group & 1u) * 4u;
    for (uint i = 0; i < 16; i++) {
        const uint lo = (r.q[i >> 2] >> (8u * (i & 3u) + shift)) & 0xFu;
        const uint hi = (r.qh[i >> 2] >> (8u * (i & 3u) + group)) & 1u;
        dst[i] = (half)(ds * (float)(lo | (hi << 4)) - dm);
    }
}
```

- [ ] **Step 4: Generate the four tile kernels by the extraction script**

Do NOT hand-copy the ~300-line kernel bodies. Run this deterministic extractor, which slices the exact `metal/qwen4.metal` kernel templates by their unique anchor lines, applies a fixed replacement table, and appends the result to `metal/qwen35.metal`. It emits only the half / non-compensated tensor variants (Q5_K needs no float/compensated tiles, D-m4.md §3.7). Save it as `speed-bench/ornith/m4/extract_q5k_tiles.py`:

```python
#!/usr/bin/env python3
"""Generate the Q5_K tiled-GEMM kernels for metal/qwen35.metal from the Q4_K
templates in metal/qwen4.metal by an exact slice + literal replacement.

Deterministic: it fails loudly if an anchor is missing or matches more than
once, so a qwen4.metal edit can never silently produce the wrong kernel.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(ROOT, "metal", "qwen4.metal")
DST = os.path.join(ROOT, "metal", "qwen35.metal")
MARK = "// BEGIN GENERATED q5_K tiles (extract_q5k_tiles.py) — do not edit by hand"
END = "// END GENERATED q5_K tiles"


def slice_between(text, start_anchor, end_anchor):
    i = text.find(start_anchor)
    j = text.find(end_anchor)
    if i < 0 or j < 0 or text.count(start_anchor) != 1 or text.count(end_anchor) != 1 or j <= i:
        sys.exit("extract: anchors not found uniquely: %r .. %r" % (start_anchor, end_anchor))
    return text[i:j]


def main():
    src = open(SRC, encoding="utf-8").read()
    # mid template + its four instantiations: from the mid doc-comment to the
    # down doc-comment.
    mid = slice_between(src,
        "/* mid[t][slot][r] = silu(gate . x) * (up . x) for every (token, slot) routed",
        "/* part[t][slot][r] = down . mid[t][slot], same tiling with mid as B */")
    # down template + its four instantiations: from the down doc-comment to the
    # rows_f32_to_f16 doc-comment.
    down = slice_between(src,
        "/* part[t][slot][r] = down . mid[t][slot], same tiling with mid as B */",
        "/* rows of floats -> halves (round to nearest even), four values per thread;")
    # the two tensor-op templates (mid then down) live between the header comment
    # and the instantiation macros; take from the mid nax kernel signature to the
    # QWEN4_NAX_MID_SIG_HALF macro (which starts the instantiations).
    naxmid = slice_between(src,
        "template <int NR1, typename XT, bool COMP>\nkernel void kernel_qwen4_moe_mm_mid_nax_t(",
        "template <int NR1, typename XT, bool COMP>\nkernel void kernel_qwen4_moe_mm_down_nax_t(")
    naxdown = slice_between(src,
        "template <int NR1, typename XT, bool COMP>\nkernel void kernel_qwen4_moe_mm_down_nax_t(",
        "#define QWEN4_NAX_MID_SIG_HALF")

    repl = [
        ("kernel_qwen4_moe_mm_mid_nax_t", "kernel_qwen35_moe_mm_mid_q5k_nax_t"),
        ("kernel_qwen4_moe_mm_down_nax_t", "kernel_qwen35_moe_mm_down_q5k_nax_t"),
        ("kernel_qwen4_moe_mm_mid", "kernel_qwen35_moe_mm_mid_q5k"),
        ("kernel_qwen4_moe_mm_down", "kernel_qwen35_moe_mm_down_q5k"),
        ("qwen4_mm_stage16", "qwen35_mm_stage16_q5k"),
        ("qwen4_load_raw16", "qwen35_load_raw_q5k"),
        ("qwen4_dequant_raw16", "qwen35_dequant_raw_q5k"),
        ("qwen4_raw16", "qwen35_raw_q5k"),
    ]

    def apply(block):
        for a, b in repl:
            block = block.replace(a, b)
        return block

    body = apply(mid) + apply(down) + apply(naxmid) + apply(naxdown)
    # The mid/down simdgroup slices already END WITH qwen4's four
    # `template [[host_name("kernel_qwen4_moe_mm_{mid,down}{,_nt1,_nt2,_nt8}")]]`
    # instantiation lines, which the replacement table renames to the q5k
    # names — so those 8 instantiations already exist in `body`.  Do NOT emit
    # them again (a second `template [[host_name(...)]]` for the same
    # specialization is a "duplicate explicit instantiation" and the Metal
    # library fails to compile).  Only the four NAX instantiations are missing:
    # the naxmid/naxdown slices stop at the QWEN4_NAX_*_SIG_* macros, before the
    # tensor-op instantiation lines, and this plan needs only the half /
    # non-compensated variants.
    body += """
#ifdef DS4_METAL_HAS_TENSOR
template [[host_name("kernel_qwen35_moe_mm_mid_q5k_nax")]]   kernel void kernel_qwen35_moe_mm_mid_q5k_nax_t<32, half, false>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const char *, device const int32_t *, device const int32_t *, device const half *, device float *, device half *, device half *, threadgroup char *, uint3, ushort, ushort, ushort);
template [[host_name("kernel_qwen35_moe_mm_mid_q5k_nax64")]] kernel void kernel_qwen35_moe_mm_mid_q5k_nax_t<64, half, false>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const char *, device const int32_t *, device const int32_t *, device const half *, device float *, device half *, device half *, threadgroup char *, uint3, ushort, ushort, ushort);
template [[host_name("kernel_qwen35_moe_mm_down_q5k_nax")]]   kernel void kernel_qwen35_moe_mm_down_q5k_nax_t<32, half, false>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const int32_t *, device const int32_t *, device const half *, device float *, device const half *, threadgroup char *, uint3, ushort, ushort, ushort);
template [[host_name("kernel_qwen35_moe_mm_down_q5k_nax64")]] kernel void kernel_qwen35_moe_mm_down_q5k_nax_t<64, half, false>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const int32_t *, device const int32_t *, device const half *, device float *, device const half *, threadgroup char *, uint3, ushort, ushort, ushort);
#endif
"""
    text = open(DST, encoding="utf-8").read()
    if MARK in text:
        head = text[:text.index(MARK)]
        tail_after = text[text.index(END) + len(END):]
        text = head.rstrip() + "\n\n"
    else:
        tail_after = ""
    text = text.rstrip() + "\n\n" + MARK + "\n" + body.rstrip() + "\n" + END + "\n" + tail_after
    open(DST, "w", encoding="utf-8").write(text)
    print("extract: wrote", DST)


if __name__ == "__main__":
    main()
```

Run it once: `python3 speed-bench/ornith/m4/extract_q5k_tiles.py`. Then read the generated block in `metal/qwen35.metal` and confirm: (a) the `mid` kernel's `qwen35_mm_stage16_q5k(grow, b, quarter0, type, dg)` calls are present, (b) the down kernel calls `qwen35_mm_stage16_q5k(drow, ...)`, (c) the nax kernels use `qwen35_load_raw_q5k`/`qwen35_dequant_raw_q5k`/`qwen35_raw_q5k`, (d) no `kernel_qwen4_moe_mm_*` name survives in the generated block, (e) each `[[host_name(...)]]` appears exactly once — the eight simdgroup instantiations `kernel_qwen35_moe_mm_{mid,down}_q5k`, `..._nt1`, `..._nt2`, `..._nt8` come from the renamed mid/down slices and the four `..._nax`/`..._nax64` from the appended block, with no duplicate (a duplicate explicit instantiation stops the Metal library compiling). `metal/qwen4.metal` is unchanged (`git diff --stat metal/qwen4.metal` shows nothing).

- [ ] **Step 5: Add the kernel enum, names, and the specialisation `||` in `ds4_metal.m`**

In the enum, insert the new entries before `QWEN4_K_COUNT` and after `QWEN4_K_QWEN35_MTP_CONCAT` (the M2 plan added `QWEN4_K_QWEN35_MTP_CONCAT` immediately after `QWEN4_K_QWEN35_GDN_OUT`, so on the M4 branch the tail reads `… GDN_OUT, MTP_CONCAT, COUNT`):
```objc
    QWEN4_K_QWEN35_GDN_OUT,
    QWEN4_K_QWEN35_MTP_CONCAT,
    QWEN4_K_QWEN35_MM_MID_Q5K,
    QWEN4_K_QWEN35_MM_MID_Q5K_NT1,
    QWEN4_K_QWEN35_MM_MID_Q5K_NT2,
    QWEN4_K_QWEN35_MM_MID_Q5K_NT8,
    QWEN4_K_QWEN35_MM_DOWN_Q5K,
    QWEN4_K_QWEN35_MM_DOWN_Q5K_NT1,
    QWEN4_K_QWEN35_MM_DOWN_Q5K_NT2,
    QWEN4_K_QWEN35_MM_DOWN_Q5K_NT8,
    QWEN4_K_QWEN35_MM_MID_Q5K_NAX,
    QWEN4_K_QWEN35_MM_MID_Q5K_NAX64,
    QWEN4_K_QWEN35_MM_DOWN_Q5K_NAX,
    QWEN4_K_QWEN35_MM_DOWN_Q5K_NAX64,
    QWEN4_K_COUNT,
```
In `qwen4_kernel_names[]`, after `"kernel_qwen35_mtp_concat",` (positional; same order as the enum, so after the M2 entry):
```objc
    "kernel_qwen35_gdn_out",
    "kernel_qwen35_mtp_concat",
    "kernel_qwen35_moe_mm_mid_q5k",
    "kernel_qwen35_moe_mm_mid_q5k_nt1",
    "kernel_qwen35_moe_mm_mid_q5k_nt2",
    "kernel_qwen35_moe_mm_mid_q5k_nt8",
    "kernel_qwen35_moe_mm_down_q5k",
    "kernel_qwen35_moe_mm_down_q5k_nt1",
    "kernel_qwen35_moe_mm_down_q5k_nt2",
    "kernel_qwen35_moe_mm_down_q5k_nt8",
    "kernel_qwen35_moe_mm_mid_q5k_nax",
    "kernel_qwen35_moe_mm_mid_q5k_nax64",
    "kernel_qwen35_moe_mm_down_q5k_nax",
    "kernel_qwen35_moe_mm_down_q5k_nax64",
```
In `qwen4_dispatch_resident`, the MM specialisation branch (`ds4_metal.m` ≈49769) currently reads:
```objc
        } else if (kernel >= QWEN4_K_MOE_MM_MID && kernel <= QWEN4_K_MOE_MM_DOWN_NAXC64) {
```
Change the condition to also cover the new range (the only edit to this shared function; the qwen4 kernels take exactly the branch they take today):
```objc
        } else if ((kernel >= QWEN4_K_MOE_MM_MID && kernel <= QWEN4_K_MOE_MM_DOWN_NAXC64) ||
                   (kernel >= QWEN4_K_QWEN35_MM_MID_Q5K && kernel <= QWEN4_K_QWEN35_MM_DOWN_Q5K_NAX64)) {
```
This binds function constants 900 (weight type, or 0) and 905 (`tail_base`) for the Q5_K tiles, which reference constant 905. Nothing else in the branch changes.

- [ ] **Step 6: Add the host wrappers and the NAX helper in `ds4_metal.m`**

Place the NAX policy helper and the two tile wrappers directly after the closing brace of `ds4_gpu_qwen4_moe_mm_down_tensor` (≈53348), **not** after the M1 `ds4_gpu_qwen35_moe_down_tensor` (≈51275): the reused static helpers `qwen4_moe_mm_tiles`, `qwen4_moe_mm_nt`, `qwen4_moe_mm_tails`, `qwen4_moe_mm_grid`, `qwen4_moe_mm_expert_major`, `qwen4_nax_half_operand`, `qwen4_nax_scratch` and the NAX scratch globals `g_qwen4_nax_half_x/_mid`, `g_qwen4_nax_half_mid_for/_count` are all defined between ≈53056 and ≈53348, so placing the wrappers before them fails to compile with undeclared identifiers. `qwen35_expert_row_bytes` (≈51196) is defined earlier and is in scope either way.

```objc
/* Ornith Q5_K tiled prefill GEMM.  Own knob DS4_QWEN35_MOE_MM_NAX (default =
 * the Q4_K policy: 2 = 64-token tensor tiles when the tensor API is available,
 * 0 = simdgroup tiles).  Only the half / non-compensated tiles are built. */
static uint32_t qwen35_moe_mm_nax(void) {
    if (!ds4_gpu_mpp_available()) return 0u;
    const char *v = getenv("DS4_QWEN35_MOE_MM_NAX");
    const long n = (v && v[0]) ? strtol(v, NULL, 10) : 2;
    if (n <= 0) return 0u;
    return (n == 1) ? 32u : 64u;
}

int ds4_gpu_qwen35_moe_mm_mid_tensor(
        ds4_gpu_tensor *mid, const ds4_gpu_tensor *x, const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts,
        const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset,
        uint32_t weight_type, uint32_t n_expert, uint32_t n_tokens, uint32_t n_slots, uint32_t n_out,
        uint32_t in_dim, uint32_t ff_dim, uint32_t list_cap) {
    if (weight_type != 13u) return 0;
    const uint32_t row_bytes = qwen35_expert_row_bytes(weight_type, in_dim);
    const uint64_t expert_bytes = (uint64_t)row_bytes * ff_dim;
    const uint32_t tiles = qwen4_moe_mm_tiles(n_tokens, true);
    const uint32_t nt = qwen4_moe_mm_nt(n_tokens, weight_type, "DS4_QWEN35_MOE_MID_NT");
    const int kernel = nt == 1u ? QWEN4_K_QWEN35_MM_MID_Q5K_NT1 :
                       nt == 2u ? QWEN4_K_QWEN35_MM_MID_Q5K_NT2 :
                       nt == 8u ? QWEN4_K_QWEN35_MM_MID_Q5K_NT8 : QWEN4_K_QWEN35_MM_MID_Q5K;
    qwen4_moe_mm_args args = { n_tokens, n_slots, n_out, in_dim, ff_dim, weight_type, row_bytes, list_cap,
                               expert_bytes, n_expert, tiles, 0, qwen4_moe_mm_expert_major() };
    const bool tails = qwen4_moe_mm_tails(weight_type, nt);   /* type 13 is not in the qwen4 tail set; false on M5 */
    if (tails) args.tail_base = nt * 8u;
    qwen4_bind b[8];
    if (n_tokens == 0 || n_slots == 0 || n_out < n_slots || row_bytes == 0 || (in_dim % 64) != 0 ||
        ff_dim == 0 || n_expert == 0 || n_expert > 512 ||
        !qwen4_bind_weight(&b[0], model_map, model_size, gate_offset, expert_bytes * n_expert, "moe gate experts") ||
        !qwen4_bind_weight(&b[1], model_map, model_size, up_offset, expert_bytes * n_expert, "moe up experts") ||
        !qwen4_bind_tensor(&b[2], lists, (uint64_t)n_expert * list_cap * sizeof(int32_t), "moe lists") ||
        !qwen4_bind_tensor(&b[3], counts, (uint64_t)n_expert * sizeof(int32_t), "moe counts") ||
        !qwen4_bind_tensor(&b[4], x, (uint64_t)n_tokens * in_dim * sizeof(float), "moe input") ||
        !qwen4_bind_tensor(&b[5], mid, (uint64_t)n_tokens * n_out * ff_dim * sizeof(float), "moe mid")) {
        return 0;
    }
    const uint32_t nax = qwen35_moe_mm_nax();
    if (nax) {
        args.tail_base = 0;
        const uint64_t mid_count = (uint64_t)n_tokens * n_out * ff_dim;
        ds4_gpu_tensor *xh = qwen4_nax_half_operand(&g_qwen4_nax_half_x, &g_qwen4_nax_half_x_bytes, x, (uint64_t)n_tokens * in_dim);
        ds4_gpu_tensor *mh = qwen4_nax_scratch(&g_qwen4_nax_half_mid, &g_qwen4_nax_half_mid_bytes, mid_count * sizeof(uint16_t));
        if (!xh || !mh || !qwen4_bind_tensor(&b[4], xh, (uint64_t)n_tokens * in_dim * sizeof(uint16_t), "moe input half") ||
            !qwen4_bind_tensor(&b[6], mh, mid_count * sizeof(uint16_t), "moe mid half")) return 0;
        g_qwen4_nax_half_mid_for = mid;
        g_qwen4_nax_half_mid_count = mid_count;
        const bool nax_tails = nax == 64u && qwen4_moe_mm_tails(weight_type, 8u);
        if (nax_tails) args.tail_base = 64u;
        const int nax_mid = nax == 64u ? QWEN4_K_QWEN35_MM_MID_Q5K_NAX64 : QWEN4_K_QWEN35_MM_MID_Q5K_NAX;
        if (!qwen4_dispatch(nax_mid, &args, sizeof(args), b, 7,
                            qwen4_moe_mm_grid((ff_dim + 63u) / 64u, n_expert, tiles, args.expert_major),
                            MTLSizeMake(128, 1, 1), nax == 64u ? 16384u : 10240u)) return 0;
        if (!nax_tails) return 1;
        args.tiles_per_launch = 1;
        return qwen4_dispatch(QWEN4_K_QWEN35_MM_MID_Q5K_NAX, &args, sizeof(args), b, 7,
                              qwen4_moe_mm_grid((ff_dim + 63u) / 64u, n_expert, 1, args.expert_major),
                              MTLSizeMake(128, 1, 1), 10240u);
    }
    if (!qwen4_dispatch(kernel, &args, sizeof(args), b, 6,
                        qwen4_moe_mm_grid((ff_dim + 31u) / 32u, n_expert, tiles, args.expert_major),
                        MTLSizeMake(128, 1, 1), 0)) return 0;
    if (tails) {
        const MTLSize tg = MTLSizeMake((ff_dim + 31u) / 32u, n_expert, 1);
        if (!qwen4_dispatch(QWEN4_K_QWEN35_MM_MID_Q5K_NT1, &args, sizeof(args), b, 6, tg, MTLSizeMake(128, 1, 1), 0)) return 0;
        if (nt > 2u && !qwen4_dispatch(QWEN4_K_QWEN35_MM_MID_Q5K_NT2, &args, sizeof(args), b, 6, tg, MTLSizeMake(128, 1, 1), 0)) return 0;
        if (nt > 4u && !qwen4_dispatch(QWEN4_K_QWEN35_MM_MID_Q5K, &args, sizeof(args), b, 6, tg, MTLSizeMake(128, 1, 1), 0)) return 0;
    }
    return 1;
}

int ds4_gpu_qwen35_moe_mm_down_tensor(
        ds4_gpu_tensor *part, const ds4_gpu_tensor *mid, const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts,
        const void *model_map, uint64_t model_size, uint64_t down_offset,
        uint32_t weight_type, uint32_t n_expert, uint32_t n_tokens, uint32_t n_slots, uint32_t n_out,
        uint32_t ff_dim, uint32_t out_dim, uint32_t list_cap) {
    if (weight_type != 13u) return 0;
    const uint32_t row_bytes = qwen35_expert_row_bytes(weight_type, ff_dim);
    const uint64_t expert_bytes = (uint64_t)row_bytes * out_dim;
    const uint32_t tiles = qwen4_moe_mm_tiles(n_tokens, false);
    const uint32_t nt = qwen4_moe_mm_nt(n_tokens, weight_type, "DS4_QWEN35_MOE_DOWN_NT");
    const int kernel = nt == 1u ? QWEN4_K_QWEN35_MM_DOWN_Q5K_NT1 :
                       nt == 2u ? QWEN4_K_QWEN35_MM_DOWN_Q5K_NT2 :
                       nt == 8u ? QWEN4_K_QWEN35_MM_DOWN_Q5K_NT8 : QWEN4_K_QWEN35_MM_DOWN_Q5K;
    qwen4_moe_mm_args args = { n_tokens, n_slots, n_out, ff_dim, out_dim, weight_type, row_bytes, list_cap,
                               expert_bytes, n_expert, tiles, 0, qwen4_moe_mm_expert_major() };
    const bool tails = qwen4_moe_mm_tails(weight_type, nt);
    if (tails) args.tail_base = nt * 8u;
    qwen4_bind b[6];
    if (n_tokens == 0 || n_slots == 0 || n_out < n_slots || row_bytes == 0 || (ff_dim % 64) != 0 ||
        out_dim == 0 || n_expert == 0 || n_expert > 512 ||
        !qwen4_bind_weight(&b[0], model_map, model_size, down_offset, expert_bytes * n_expert, "moe down experts") ||
        !qwen4_bind_tensor(&b[1], lists, (uint64_t)n_expert * list_cap * sizeof(int32_t), "moe lists") ||
        !qwen4_bind_tensor(&b[2], counts, (uint64_t)n_expert * sizeof(int32_t), "moe counts") ||
        !qwen4_bind_tensor(&b[3], mid, (uint64_t)n_tokens * n_out * ff_dim * sizeof(float), "moe mid") ||
        !qwen4_bind_tensor(&b[4], part, (uint64_t)n_tokens * n_out * out_dim * sizeof(float), "moe partial")) {
        return 0;
    }
    const uint32_t nax = qwen35_moe_mm_nax();
    if (nax) {
        args.tail_base = 0;
        const uint64_t mid_count = (uint64_t)n_tokens * n_out * ff_dim;
        ds4_gpu_tensor *mh = g_qwen4_nax_half_mid_for == mid && g_qwen4_nax_half_mid_count == mid_count && g_qwen4_nax_half_mid
                             ? g_qwen4_nax_half_mid
                             : qwen4_nax_half_operand(&g_qwen4_nax_half_mid, &g_qwen4_nax_half_mid_bytes, mid, mid_count);
        g_qwen4_nax_half_mid_for = NULL;
        g_qwen4_nax_half_mid_count = 0;
        if (!mh || !qwen4_bind_tensor(&b[3], mh, mid_count * sizeof(uint16_t), "moe mid half")) return 0;
        const bool nax_tails = nax == 64u && qwen4_moe_mm_tails(weight_type, 8u);
        if (nax_tails) args.tail_base = 64u;
        const int nax_down = nax == 64u ? QWEN4_K_QWEN35_MM_DOWN_Q5K_NAX64 : QWEN4_K_QWEN35_MM_DOWN_Q5K_NAX;
        if (!qwen4_dispatch(nax_down, &args, sizeof(args), b, 5,
                            qwen4_moe_mm_grid((out_dim + 63u) / 64u, n_expert, tiles, args.expert_major),
                            MTLSizeMake(128, 1, 1), nax == 64u ? 16384u : 8192u)) return 0;
        if (!nax_tails) return 1;
        args.tiles_per_launch = 1;
        return qwen4_dispatch(QWEN4_K_QWEN35_MM_DOWN_Q5K_NAX, &args, sizeof(args), b, 5,
                              qwen4_moe_mm_grid((out_dim + 63u) / 64u, n_expert, 1, args.expert_major),
                              MTLSizeMake(128, 1, 1), 8192u);
    }
    if (!qwen4_dispatch(kernel, &args, sizeof(args), b, 5,
                        qwen4_moe_mm_grid((out_dim + 31u) / 32u, n_expert, tiles, args.expert_major),
                        MTLSizeMake(128, 1, 1), 0)) return 0;
    if (tails) {
        const MTLSize tg = MTLSizeMake((out_dim + 31u) / 32u, n_expert, 1);
        if (!qwen4_dispatch(QWEN4_K_QWEN35_MM_DOWN_Q5K_NT1, &args, sizeof(args), b, 5, tg, MTLSizeMake(128, 1, 1), 0)) return 0;
        if (nt > 2u && !qwen4_dispatch(QWEN4_K_QWEN35_MM_DOWN_Q5K_NT2, &args, sizeof(args), b, 5, tg, MTLSizeMake(128, 1, 1), 0)) return 0;
        if (nt > 4u && !qwen4_dispatch(QWEN4_K_QWEN35_MM_DOWN_Q5K, &args, sizeof(args), b, 5, tg, MTLSizeMake(128, 1, 1), 0)) return 0;
    }
    return 1;
}
```
Note: the Q5_K down rows are 2048 wide (`ff_dim` 512 for mid, `ff_dim`→`out_dim` 2048 for down); `ff_dim % 64 == 0` holds (512, 2048). Unlike Q2_K/Q4_K padded-down rows, Q5_K down rows here are un-padded (512 divisible by 256), so no `weight_dim` rounding is needed.

Declare both in `ds4_gpu.h` directly after the `ds4_gpu_qwen35_moe_down_tensor` prototype (≈3611), using the Interfaces signatures above.

- [ ] **Step 7: Run the kernel test**

Run: `make test-qwen35-kernels`
Expected:
```
qwen35 moe q5_K tiled GEMM (simdgroup + tensor tiles)
  q5k tiles nax=0 down vs exact T=65   ...
  q5k tiles nax=64 down vs exact T=65  ...
  q5k tensor mid within 2e-3 of simdgroup ...
  q5k tiles vs row kernels down T=65 ...
  ... (T=200, T=641)
qwen35 kernels: ok
```

- [ ] **Step 8: Wire the tiles into the graph** — edit `ds4_qwen35moe.inc` `qwen35_graph_moe`

Line ≈155 currently:
```c
    const bool mm = T > 64u && xt == DS4_TENSOR_Q4_K && dt == DS4_TENSOR_Q4_K;
```
Change to accept a Q5_K pair as well, and route the `mm` branch by type:
```c
    const bool mm = T > 64u && xt == dt && (xt == DS4_TENSOR_Q4_K || xt == DS4_TENSOR_Q5_K);
```
Inside the `if (ok && mm) {` block (≈163), replace the two `ds4_gpu_qwen4_moe_mm_*` calls so Q5_K layers use the qwen35 tile wrappers and Q4_K layers keep the qwen4 ones (the shared `build_lists` call is unchanged):
```c
    if (ok && mm) {
        ok = ds4_gpu_qwen4_moe_build_lists_tensor(g->moe_lists, g->moe_counts, g->selected, T, K, NE,
                                                  g->cap_tokens);
        if (ok) {
            ok = (xt == DS4_TENSOR_Q5_K)
                 ? ds4_gpu_qwen35_moe_mm_mid_tensor(g->mid, g->mixed, g->moe_lists, g->moe_counts, m->map, m->size,
                                                    l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset, xt,
                                                    NE, T, K, K, E, F, g->cap_tokens)
                 : ds4_gpu_qwen4_moe_mm_mid_tensor(g->mid, g->mixed, g->moe_lists, g->moe_counts, m->map, m->size,
                                                   l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset, xt,
                                                   NE, T, K, K, E, F, g->cap_tokens);
        }
        if (ok) {
            ok = (dt == DS4_TENSOR_Q5_K)
                 ? ds4_gpu_qwen35_moe_mm_down_tensor(g->part, g->mid, g->moe_lists, g->moe_counts, m->map, m->size,
                                                     l->ffn_down_exps->abs_offset, dt, NE, T, K, K, F, E, g->cap_tokens)
                 : ds4_gpu_qwen4_moe_mm_down_tensor(g->part, g->mid, g->moe_lists, g->moe_counts, m->map, m->size,
                                                    l->ffn_down_exps->abs_offset, dt, NE, T, K, K, F, E, g->cap_tokens);
        }
    } else if (ok) {
```
Update the comment at `qwen35_graph_moe`'s head (≈141-146): "Q4_K and Q5_K layers take the tiled expert GEMMs above one tile of tokens; single rows and verifies keep the per-token row kernels." The `shared_dense` line (`mm || T > 8u`) is unchanged: with `mm` now true for Q5_K prefill, its shared expert runs as dense projections, matching the Q4_K path.

- [ ] **Step 9: Build ds4 and re-run the kernel + Q2 tests**

Run: `make ds4 test-qwen35-kernels test-qwen4-kernels test-qwen4-q2`
Expected: all three test binaries end without a failure line. `git diff --stat metal/qwen4.metal` shows no change (additive rule).

- [ ] **Step 10: Gate 1 unchanged at chunk 65 (Q5_K tiles now active in layers 0-14)**

The chunk-65 run drives Q5_K layers through the tiles (a 65-token chunk > 64). Its output must still pass gate 1. With the box free and the user's OK for a model run:
```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
python3 tests/ornith/gate1.py --ds4-arg --prefill-chunk --ds4-arg 65 speed-bench/ornith/m4/gate1-chunk65
python3 tests/ornith/gate1.py --ds4-arg --prefill-chunk --ds4-arg 2048 speed-bench/ornith/m4/gate1-chunk2048
python3 tests/ornith/gate1.py --ds4-arg --prefill-chunk --ds4-arg 64 speed-bench/ornith/m4/gate1-chunk64
```
Expected: each prints `gate1: PASS`. The chunk-64 run uses no tiles (control). Copy the three `compare` summaries into `speed-bench/ornith/m4/gate1/`. If chunk-65 now diverges beyond the M1 tolerance where chunk-64 passed, the tile GEMM introduced a real error — STOP with a `systematic-debugging` ruling (the kernel test would usually have caught it; check the tail/empty-expert path).

- [ ] **Step 11: Qwen gate fast (shared files touched: `ds4_metal.m`, `ds4_gpu.h`, `metal/qwen35.metal`)**

Run (box free, user OK): `speed-bench/qwen-regression/run.sh fast`
Expected: `qwen_gate: PASS` and the two `make test-qwen4-*` lines clean.

- [ ] **Step 12: Commit**

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
git add metal/qwen35.metal ds4_metal.m ds4_gpu.h ds4_qwen35moe.inc tests/test_qwen35_kernels.c \
        speed-bench/ornith/m4/extract_q5k_tiles.py speed-bench/ornith/m4/gate1
git commit -m "qwen35: Q5_K tiled prefill GEMM for Ornith

New qwen35_mm_stage16_q5k / qwen35_raw_q5k helpers and the four tile kernels
(simdgroup + tensor), generated from the Q4_K templates by an exact extraction
script; new host wrappers; the shared MM specialisation branch gains one || for
the Q5_K range (binds constants 900/905). Layers 0-14 (Q5_K) now use the tiles
above 64 tokens, closing the prefill gap. qwen4.metal unchanged; gate 1
chunk-65 and the Qwen fast gate pass.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- metal/qwen35.metal ds4_metal.m ds4_gpu.h ds4_qwen35moe.inc tests/test_qwen35_kernels.c speed-bench/ornith/m4/extract_q5k_tiles.py speed-bench/ornith/m4/gate1
```

---

### Task 5: Ornith stage profile `DS4_QWEN35_PROFILE`

A per-stage GPU timer so every later measurement (prefill sweep, decode levers, 128K MTP verify cost) is attributed to a stage, not guessed. Modelled on the qwen4 `prof[]` block (`ds4.c` ≈61144-61236) but Ornith-only and additive to `qwen35_graph_forward_tokens`.

**Files:**
- Modify: `ds4_qwen35moe.inc` (`qwen35_graph_forward_tokens`)

**Interfaces:**
- Consumes: `now_sec()`, `ds4_gpu_end_commands()`, `glm_graph_begin_commands_if_needed()`.
- Produces: with `DS4_QWEN35_PROFILE=1` and T > 1, one stderr line per chunk `ds4: Ornith stage ms/chunk (pos=%u T=%u): norm1 A gdn B attn C moe D head E`; the T=2 verify (T==2) is timed too so the 128K MTP verify KV cost is visible.

- [ ] **Step 1: Write the profile into `qwen35_graph_forward_tokens`**

Replace the layer loop and logits section of `qwen35_graph_forward_tokens` (the M1 body, `ds4_qwen35moe.inc` ≈222-252, plus whatever M2 added for `all_rows`/`mtp_h`; re-read it in Task 1 and preserve those additions) so the per-stage timing wraps each stage. Because `qwen35_graph_layer` is one call, split its timing at the `norm/mixer/moe` boundaries by adding an optional profiler passed down, OR (simpler, no signature change) time whole layers by kind and the head. Use the simpler form: keep `qwen35_graph_layer` as is and time each layer as GDN or attention plus a periodic sync, accumulating into `prof[]`:

```c
static bool qwen35_graph_forward_tokens(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                        const int *tokens, uint32_t T, float *logits_out, bool all_rows) {
    /* ... keep the M2 preconditions, token-range check, pos0, stage_inputs,
     *     glm_graph_begin_commands_if_needed as they are ... */
    static int prof_on = -1;
    if (prof_on < 0) { const char *e = getenv("DS4_QWEN35_PROFILE"); prof_on = e && e[0] && e[0] != '0'; }
    const bool prof = prof_on && T > 1u;
    double p[5] = {0}, last = prof ? now_sec() : 0.0;
#define QWEN35_PROF(idx_) do { if (prof) { ds4_gpu_end_commands(); const double n_ = now_sec(); \
        p[idx_] += n_ - last; last = n_; glm_graph_begin_commands_if_needed(); } } while (0)
    bool ok = true;
    for (uint32_t il = 0; il + DS4_N_NEXTN_PREDICT < DS4_N_LAYER && ok; il++) {
        ok = qwen35_graph_layer(g, m, w, il, pos0, T);
        QWEN35_PROF(ds4_qwen35_layer_is_attention(il) ? 2 : 1);   /* 1 gdn, 2 attn (moe folded in the layer) */
    }
    /* ... keep the M2 logits / mtp_h carry section, then QWEN35_PROF(4) after the head ... */
    QWEN35_PROF(4);
    if (prof)
        fprintf(stderr, "ds4: Ornith stage ms/chunk (pos=%u T=%u): gdn %.1f attn %.1f head %.1f\n",
                pos0, T, 1000.0 * p[1], 1000.0 * p[2], 1000.0 * p[4]);
#undef QWEN35_PROF
    /* ... keep the M1/M2 end_commands, logits read-back, g->pos += T ... */
}
```
(The MoE time is inside each layer's kind bucket; that is enough to answer "is attention or the expert GEMM the prefill cost". A finer split is not needed — the levers below target specific dispatches, not stages.) Because the profiler calls `ds4_gpu_end_commands()` between stages, it must be off (`prof` false) for T==1 decode where the flush lever (Task 7) manages the batch; `T > 1u` guarantees that.

- [ ] **Step 2: Build**

Run: `make ds4`
Expected: clean build.

- [ ] **Step 3: Measure the profile at 2K / 32K / 128K prefill and the T=2 verify**

Box free, user OK. Use the CLI one-shot (`--raw`) so no server is involved; the long prompt comes from the corpus:
```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
mkdir -p speed-bench/ornith/m4/profile
for chars in 7000 112000 450000; do
  head -c $chars speed-bench/promessi_sposi.txt > /tmp/orn-prof.txt
  DS4_QWEN35_PROFILE=1 caffeinate -i -s ./ds4 -m "$DS4_ORNITH_MODEL" --metal --raw \
    --prompt-file /tmp/orn-prof.txt -c 262144 -n 8 --prefill-chunk 2048 \
    > /dev/null 2> speed-bench/ornith/m4/profile/prefill-$chars.txt
  tail -8 speed-bench/ornith/m4/profile/prefill-$chars.txt
done
```
For the T=2 MTP verify cost at 128K, run with `--mtp` and `DS4_QWEN35_SPEC_TRACE=1` on a 128K prompt and capture the T=2 profile lines. Record the per-stage split for each context in `speed-bench/ornith/m4/profile/PROFILE.md`, and state whether attention dominates prefill (drives the Task 6 chunk sweep and the optional prefill-attention work) and whether the T=2 verify's attention time at 128K is roughly double the T=1 decode's (Review Focus 5 trigger for L12 in Task 11).

- [ ] **Step 4: Commit**

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
git add ds4_qwen35moe.inc speed-bench/ornith/m4/profile
git commit -m "qwen35: DS4_QWEN35_PROFILE per-stage prefill/verify timer for Ornith

Additive to qwen35_graph_forward_tokens; T>1 only, so decode's command batch
is untouched. Records the 2K/32K/128K prefill split and the T=2 verify KV cost.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- ds4_qwen35moe.inc speed-bench/ornith/m4/profile
```

---

### Task 6: Prefill chunk sweep

Measurement only. With the tiles in, a larger chunk gives more tokens per expert (fewer re-reads) at the cost of scratch memory. Find the chunk that maximises prefill t/s without exceeding admission.

**Files:**
- Create: `speed-bench/ornith/m4/profile/CHUNK_SWEEP.md`

**Interfaces:**
- Consumes: `DS4_QWEN35_PREFILL_CHUNK` (`ds4.c` `qwen35_prefill_chunk_tokens`, default 2048), the Task 2 harness or the CLI one-shot.

- [ ] **Step 1: Sweep the chunk on a 32K and a 128K prompt**

Box free, user OK:
```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
mkdir -p speed-bench/ornith/m4/profile
for chars in 112000 450000; do
  head -c $chars speed-bench/promessi_sposi.txt > /tmp/orn-chunk.txt
  for chunk in 2048 4096 8192; do
    DS4_QWEN35_PREFILL_CHUNK=$chunk caffeinate -i -s ./ds4 -m "$DS4_ORNITH_MODEL" --metal --raw \
      --prompt-file /tmp/orn-chunk.txt -c 262144 -n 8 \
      2>&1 | grep 'Ornith prefill' | sed "s/^/chars=$chars chunk=$chunk /" \
      | tee -a speed-bench/ornith/m4/profile/chunk-sweep.txt
  done
done
```
Record prefill t/s per (context, chunk) in `CHUNK_SWEEP.md`. Cross-check the scratch cost against `ds4_context_memory_estimate_with_prefill_mode` (`ds4.c` ≈40182): the estimate scales `scratch_bytes` with `T` (the chunk). Pick the chunk with the best prefill t/s whose planned total still admits at 262144 context next to 21 GiB of weights and the F16 KV. Name that chunk as the Ornith deploy default in the report (Task 13); it is a launch-arg (`--prefill-chunk`) or `DS4_QWEN35_PREFILL_CHUNK`, not a code change.

- [ ] **Step 2: Commit**

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
git add speed-bench/ornith/m4/profile/CHUNK_SWEEP.md speed-bench/ornith/m4/profile/chunk-sweep.txt
git commit -m "speed-bench/ornith: M4 prefill chunk sweep (2048/4096/8192)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- speed-bench/ornith/m4/profile/CHUNK_SWEEP.md speed-bench/ornith/m4/profile/chunk-sweep.txt
```

---

### Task 7: Decode levers A — L1 early flush + L2 add+RMSNorm fusion

Two host-only decode levers. Each is behind a `DS4_QWEN35_*` switch, kept as a default only after: (a) gate 1 output identical across the three chunk settings, and (b) an interleaved A/B shows a gain; otherwise reverted with a ledger note.

**Files:**
- Modify: `ds4_qwen35moe.inc` (`qwen35_graph_forward_tokens`, `qwen35_graph_layer`)

**Interfaces:**
- Consumes: `ds4_gpu_flush_commands()`; `ds4_gpu_add_rms_norm_weight_tensor(norm_out, sum_out, a, b, map, size, weight_offset, n, eps)` (single row, grid 1×1×1 — decode T=1 only).
- Produces: knobs `DS4_QWEN35_FLUSH_LAYER` (default 2), `DS4_QWEN35_FUSE_NORM` (default 0 until proven).

- [ ] **Step 1: L1 — early flush in `qwen35_graph_forward_tokens`**

Inside the layer loop (after `qwen35_graph_layer(...)`), submit the first layers while the host encodes the rest, exactly as qwen4 does:
```c
    static int flush_layer = -1;
    if (flush_layer < 0) {
        const char *e = getenv("DS4_QWEN35_FLUSH_LAYER");
        const uint32_t n_trunk = DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
        long v = (e && e[0]) ? strtol(e, NULL, 10) : 2;
        flush_layer = (v >= 0 && (uint32_t)v < n_trunk) ? (int)v : 2;
    }
    /* ... in the loop, after the layer and (if prof) QWEN35_PROF: */
        if (ok && (int)(il + 1u) == flush_layer && !prof) ok = ds4_gpu_flush_commands() != 0;
```
Guard with `!prof` so the profiler's per-stage `end_commands` and the flush do not fight. Flush is order-preserving and retains pending buffers; the existing `ds4_gpu_end_commands()` at the end still waits for both batches before inputs are reused.

- [ ] **Step 2: L2 — add+RMSNorm fusion in `qwen35_graph_layer` (T==1 only)**

`qwen35_graph_layer` does, twice, `add_tensor(R,R,blk)` then a following `rms_norm_weight_rows_tensor(mixed,R,<next norm>)`. Fuse the first pair (attn/GDN add + the same layer's `ffn_norm`) at T==1 behind `DS4_QWEN35_FUSE_NORM`:
```c
static bool qwen35_fuse_norm_enabled(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DS4_QWEN35_FUSE_NORM"); v = e && e[0] && e[0] != '0'; }
    return v;
}
/* ... in qwen35_graph_layer, replacing the add(blk) + ffn_norm pair: */
    if (ok) {
        if (T == 1u && qwen35_fuse_norm_enabled()) {
            ok = ds4_gpu_add_rms_norm_weight_tensor(g->mixed, g->R, g->R, g->blk, m->map, m->size,
                                                    l->ffn_norm->abs_offset, DS4_N_EMBD, DS4_RMS_EPS) != 0;
        } else {
            ok = ds4_gpu_add_tensor(g->R, g->R, g->blk, n) != 0 &&
                 ds4_gpu_rms_norm_weight_rows_tensor(g->mixed, g->R, m->map, m->size, l->ffn_norm->abs_offset,
                                                     DS4_N_EMBD, T, DS4_RMS_EPS) != 0;
        }
    }
```
(The second add, after MoE, keeps the plain `add_tensor`; its following norm is the next layer's `attn_norm`, which crosses the layer boundary and is not fused here.) The fused kernel writes `sum_out = R` and `norm_out = mixed`; both are E-wide, matching. Expected bit-identical; the gate-1 check below decides.

- [ ] **Step 3: Build and gate 1 across the three chunk settings**

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface && make ds4
for chunk in 2048 64 65; do
  DS4_QWEN35_FLUSH_LAYER=2 DS4_QWEN35_FUSE_NORM=1 \
    python3 tests/ornith/gate1.py --ds4-arg --prefill-chunk --ds4-arg $chunk \
      speed-bench/ornith/m4/gate1-l1l2-chunk$chunk
done
```
Expected: `gate1: PASS` on all three, with `max_delta` equal to the Task 4 gate-1 numbers (flush changes only submission timing; the norm fusion must not change output). L1 (flush) never changes arithmetic, so any change means L2 is not bit-exact: if `max_delta` differs from Task 4, treat L2 as a non-exact fusion, leave `DS4_QWEN35_FUSE_NORM` default 0, and record the ledger note (spec §4: a fusion is kept only if output is unchanged).

- [ ] **Step 4: Interleaved A/B for the gain (box free, user OK)**

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
# baseline arm = default env; lever arm = flush+fuse. Both ds4; oMLX is the fixed reference.
python3 speed-bench/ornith/m4_ab.py --ds4-model "$DS4_ORNITH_MODEL" \
  --out speed-bench/ornith/m4/speed/l1l2 --omlx-attach \
  --ds4-env DS4_QWEN35_FLUSH_LAYER=2,DS4_QWEN35_FUSE_NORM=1 --contexts 2048,32768 --cold-tokens 0
```
Compare `speed/l1l2/m4_ab.json` ds4 decode against the Task 3 baseline ds4 decode at 2K/32K. Keep each lever as a default only if it gains (set the default in code: L1 already defaults to 2, so it is on unless it regressed — if it regressed, set the default to 0 in the `strtol` fallback and note it; L2 default stays 0 unless it is both exact and faster, in which case flip the `DS4_QWEN35_FUSE_NORM` default to 1). Record kept/reverted with the measured deltas in `speed-bench/ornith/m4/speed/LEVERS.md`.

- [ ] **Step 5: Qwen gate fast (shared file `ds4_qwen35moe.inc` is Ornith-only, but `run.sh fast` still runs the kernel tests; run it to be safe) and commit**

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
speed-bench/qwen-regression/run.sh fast
git add ds4_qwen35moe.inc speed-bench/ornith/m4/gate1-l1l2-chunk2048 speed-bench/ornith/m4/gate1-l1l2-chunk64 \
        speed-bench/ornith/m4/gate1-l1l2-chunk65 speed-bench/ornith/m4/speed/l1l2 speed-bench/ornith/m4/speed/LEVERS.md
git commit -m "qwen35: decode levers L1 (DS4_QWEN35_FLUSH_LAYER) + L2 (DS4_QWEN35_FUSE_NORM)

Early command flush and an add+RMSNorm fusion at T=1, each behind its own knob.
gate 1 identical across chunk 2048/64/65; kept/reverted per the interleaved A/B
(LEVERS.md).

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- ds4_qwen35moe.inc speed-bench/ornith/m4/gate1-l1l2-chunk2048 speed-bench/ornith/m4/gate1-l1l2-chunk64 speed-bench/ornith/m4/gate1-l1l2-chunk65 speed-bench/ornith/m4/speed/l1l2 speed-bench/ornith/m4/speed/LEVERS.md
```

---

### Task 8: Decode levers B — L4 q/k/v multi-GEMV + L5 paired-GDN A/B

L4 folds the three attention projections into one dispatch (new Ornith code behind a knob). L5 is the shared-helper paired GDN qkv+z GEMV (`DS4_QWEN4_DECODE_FUSIONS`, the documented shared-switch exception, spec §7.3) — measurement only, no new code.

**Files:**
- Modify: `ds4_qwen35moe.inc` (`qwen35_graph_attention`)

**Interfaces:**
- Consumes: `ds4_gpu_qwen4_multi_gemv_tensor(x, n_tokens, in_dim, n_out, outs, map, size, offsets, types, out_rows)` (up to 4 outputs; types 0/1/2/8/30/39); `ds4_gpu_qwen4_decode_fusions_enabled()`.
- Produces: knob `DS4_QWEN35_ATTN_MULTI_GEMV` (default 0 until proven).

- [ ] **Step 1: L4 in `qwen35_graph_attention` (T ≤ 2 only)**

Currently three `qwen4_gemv` calls (qg = attn_q Q8_0; kp = attn_k F16; vp = attn_v F16). Fold them:
```c
static bool qwen35_attn_multi_gemv(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DS4_QWEN35_ATTN_MULTI_GEMV"); v = e && e[0] && e[0] != '0'; }
    return v;
}
/* ... replacing the three qwen4_gemv(qg/kp/vp) calls at the head of qwen35_graph_attention: */
    bool proj;
    if (T <= 2u && qwen35_attn_multi_gemv()) {
        ds4_gpu_tensor *outs[3] = { g->qg, g->kp, g->vp };
        const uint64_t offs[3] = { l->attn_q->abs_offset, l->attn_k->abs_offset, l->attn_v->abs_offset };
        const uint32_t types[3] = { l->attn_q->type, l->attn_k->type, l->attn_v->type };
        const uint32_t rows[3] = { (uint32_t)l->attn_q->dim[1], (uint32_t)l->attn_k->dim[1], (uint32_t)l->attn_v->dim[1] };
        proj = ds4_gpu_qwen4_multi_gemv_tensor(g->mixed, T, DS4_N_EMBD, 3, outs, m->map, m->size, offs, types, rows) != 0;
    } else {
        proj = qwen4_gemv(g->qg, m, l->attn_q, g->mixed, T) &&
               qwen4_gemv(g->kp, m, l->attn_k, g->mixed, T) &&
               qwen4_gemv(g->vp, m, l->attn_v, g->mixed, T);
    }
    return proj &&
           ds4_gpu_qwen35_attn_prep_tensor(...) != 0 &&    /* keep the M1 prep + decode + attn_output as they are */
           ...;
```
The multi-GEMV's reduction order differs from three separate GEMVs, so this is a non-exact lever — gate 1 decides whether the output is unchanged; keep only if identical.

- [ ] **Step 2: Build and gate 1 (three chunk settings)**

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface && make ds4
for chunk in 2048 64 65; do
  DS4_QWEN35_ATTN_MULTI_GEMV=1 python3 tests/ornith/gate1.py \
    --ds4-arg --prefill-chunk --ds4-arg $chunk speed-bench/ornith/m4/gate1-l4-chunk$chunk
done
```
Expected: `gate1: PASS`. If `max_delta` differs from Task 4, L4 changed the output — keep `DS4_QWEN35_ATTN_MULTI_GEMV` default 0 and record the ledger note. (L4 only fires at T ≤ 2, i.e. decode and the T=2 verify; gate 1 runs plain decode, so it exercises the T=1 path.)

- [ ] **Step 3: Interleaved A/B for L4 and, separately, L5 (box free, user OK)**

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
python3 speed-bench/ornith/m4_ab.py --ds4-model "$DS4_ORNITH_MODEL" --out speed-bench/ornith/m4/speed/l4 \
  --omlx-attach --ds4-env DS4_QWEN35_ATTN_MULTI_GEMV=1 --contexts 2048,32768 --cold-tokens 0
# L5: the shared decode-fusions switch (documented §7.3 exception). It fires only at T=1 without an
# active MTP draft (!g->mtp_R), which under MTP is the reject/plain-eval path; measure and expect small.
python3 speed-bench/ornith/m4_ab.py --ds4-model "$DS4_ORNITH_MODEL" --out speed-bench/ornith/m4/speed/l5 \
  --omlx-attach --ds4-env DS4_QWEN4_DECODE_FUSIONS=1 --contexts 2048,32768 --cold-tokens 0
```
For L5, gate 1 must also stay identical (`DS4_QWEN4_DECODE_FUSIONS=1` selects an arithmetic-equivalent kernel path; run one gate-1 chunk-2048 with it set to confirm). Keep L4 (flip its default to 1) and/or L5 (add `DS4_QWEN4_DECODE_FUSIONS=1` to the Ornith deploy command as the documented shared-switch exception) only where gate 1 is identical and the A/B gains. Append the verdicts to `speed-bench/ornith/m4/speed/LEVERS.md`.

- [ ] **Step 4: Commit**

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
git add ds4_qwen35moe.inc speed-bench/ornith/m4/gate1-l4-chunk2048 speed-bench/ornith/m4/gate1-l4-chunk64 \
        speed-bench/ornith/m4/gate1-l4-chunk65 speed-bench/ornith/m4/speed/l4 speed-bench/ornith/m4/speed/l5 \
        speed-bench/ornith/m4/speed/LEVERS.md
git commit -m "qwen35: decode lever L4 (DS4_QWEN35_ATTN_MULTI_GEMV) + L5 A/B

Fold q/k/v into one dispatch behind a knob (kept only if gate 1 is identical
and the A/B gains); measure the shared DS4_QWEN4_DECODE_FUSIONS path (the §7.3
shared-switch exception) for Ornith. Verdicts in LEVERS.md.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- ds4_qwen35moe.inc speed-bench/ornith/m4/gate1-l4-chunk2048 speed-bench/ornith/m4/gate1-l4-chunk64 speed-bench/ornith/m4/gate1-l4-chunk65 speed-bench/ornith/m4/speed/l4 speed-bench/ornith/m4/speed/l5 speed-bench/ornith/m4/speed/LEVERS.md
```

---

### Task 9: Decode lever C — L8 Q5_K decode kernels

The M1 Q5_K decode kernels (`kernel_qwen35_moe_mid/down`) are the generic 2-row-per-simdgroup kernels. Give the Q5_K layers the Q4_K decode treatment: a 1-row mid and a 4-row down that keep each row's per-lane accumulation order (so they are bit-exact vs the 2-row kernels), selected by `DS4_QWEN35_MOE_MR_MID` / `_DOWN`.

**Files:**
- Modify: `metal/qwen35.metal` (two new decode kernels + instantiations), `ds4_metal.m` (enum/names, wrapper row selection), `tests/test_qwen35_kernels.c` (extend `test_moe_q5k` to run the row kernels at NR 1/2/4), `ds4_qwen35moe.inc` (no change; the wrappers pick the kernel)

**Interfaces:**
- Produces (Metal): `kernel_qwen35_moe_mid_q5k_nr1`, `kernel_qwen35_moe_down_q5k_nr4` (and NR2 variants), templated on NR; the host wrappers `ds4_gpu_qwen35_moe_mid_tensor`/`_down_tensor` pick NR from `DS4_QWEN35_MOE_MR_MID`/`_DOWN` (default: mid 1, down 4 on M5, else the M1 2-row kernels).

- [ ] **Step 1: Write the failing NR-equivalence test**

Add to `tests/test_qwen35_kernels.c`, before `int main` (it uses the M1 `arena_q5_K`/`arena_q8_0` packers and the mid/down row wrappers under each `DS4_QWEN35_MOE_MR_*` setting, asserting bit-identical output — the per-row dot order is unchanged, only which rows a simdgroup owns changes):
```c
/* L8: the Q5_K decode kernels at each NR are bit-identical (each row's
 * per-lane dot order is unchanged; only the row-per-simdgroup split changes). */
static void test_moe_q5k_rows(arena_t *a) {
    const uint32_t NE = 16, slots = 8, E = 512, F = 256, T = 2, n_out = slots + 1;
    double *gate_w, *up_w, *down_w, *sg_w, *su_w, *sd_w;
    const uint64_t gate_off = arena_q5_K(a, (uint64_t)NE * F, E, &gate_w, 0.05f);
    const uint64_t up_off = arena_q5_K(a, (uint64_t)NE * F, E, &up_w, 0.05f);
    const uint64_t down_off = arena_q5_K(a, (uint64_t)NE * E, F, &down_w, 0.05f);
    const uint64_t sg_off = arena_q8_0(a, F, E, &sg_w, 0.05f);
    const uint64_t su_off = arena_q8_0(a, F, E, &su_w, 0.05f);
    const uint64_t sd_off = arena_q8_0(a, E, F, &sd_w, 0.05f);
    float *x = rand_vec((uint64_t)T * E, 1.0f);
    float *mid_in = rand_vec((uint64_t)T * n_out * F, 1.0f);
    int32_t *sel = malloc((uint64_t)T * slots * 4);
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t s = 0; s < slots; s++) sel[t * slots + s] = (int32_t)((t * 7u + s * 3u) % NE);
    ds4_gpu_tensor *gx = upload(x, (uint64_t)T * E);
    ds4_gpu_tensor *gsel = ds4_gpu_tensor_alloc((uint64_t)T * slots * 4);
    require_ok(ds4_gpu_tensor_write(gsel, 0, sel, (uint64_t)T * slots * 4), "l8 sel");
    ds4_gpu_tensor *gmidin = upload(mid_in, (uint64_t)T * n_out * F);
    ds4_gpu_tensor *gmid = upload(NULL, (uint64_t)T * n_out * F), *gpart = upload(NULL, (uint64_t)T * n_out * E);
    const char *mk = "DS4_QWEN35_MOE_MR_MID", *dk = "DS4_QWEN35_MOE_MR_DOWN";
    char *sm = getenv(mk) ? strdup(getenv(mk)) : NULL, *sd = getenv(dk) ? strdup(getenv(dk)) : NULL;
    float *ref_mid = NULL, *ref_part = NULL;
    const char *mrv[] = {"0", "1", "2", "4"}, *drv[] = {"0", "2", "4"};
    for (uint32_t mi = 0; mi < 4u; mi++) {
        setenv(mk, mrv[mi], 1);
        require_ok(ds4_gpu_qwen35_moe_mid_tensor(gmid, gx, gsel, a->base, a->size, gate_off, up_off, 13u,
                       NE, T, slots, E, F, sg_off, su_off, 8u), "l8 mid");
        float *got = download(gmid, (uint64_t)T * n_out * F);
        if (ref_mid) { for (uint64_t i = 0; i < (uint64_t)T * n_out * F; i++) require_ok(got[i] == ref_mid[i], "l8 mid NR bit-identical"); free(got); }
        else ref_mid = got;
    }
    for (uint32_t di = 0; di < 3u; di++) {
        setenv(dk, drv[di], 1);
        require_ok(ds4_gpu_qwen35_moe_down_tensor(gpart, gmidin, gsel, a->base, a->size, down_off, 13u,
                       NE, T, slots, F, E, sd_off, 8u), "l8 down");
        float *got = download(gpart, (uint64_t)T * n_out * E);
        if (ref_part) { for (uint64_t i = 0; i < (uint64_t)T * n_out * E; i++) require_ok(got[i] == ref_part[i], "l8 down NR bit-identical"); free(got); }
        else ref_part = got;
    }
    printf("  q5_K decode NR {mid 0/1/2/4, down 0/2/4}: bit-identical\n");
    if (sm) { setenv(mk, sm, 1); free(sm); } else unsetenv(mk);
    if (sd) { setenv(dk, sd, 1); free(sd); } else unsetenv(dk);
    free(x); free(mid_in); free(sel); free(ref_mid); free(ref_part);
    free(gate_w); free(up_w); free(down_w); free(sg_w); free(su_w); free(sd_w);
    ds4_gpu_tensor_free(gx); ds4_gpu_tensor_free(gsel); ds4_gpu_tensor_free(gmidin);
    ds4_gpu_tensor_free(gmid); ds4_gpu_tensor_free(gpart);
}
```
In `main`, add after the M1 `test_moe_q5k` calls: `printf("qwen35 q5_K decode row variants (L8)\n"); test_moe_q5k_rows(&arena);`.

- [ ] **Step 2: Build the test (passes trivially on the M1 kernel)**

Run: `make test-qwen35-kernels`
Expected: the new `test_moe_q5k_rows` line passes — but only trivially: until Step 3 adds the NR kernels and the wrapper honours the env, every `DS4_QWEN35_MOE_MR_*` setting still dispatches the M1 2-row kernel, so all outputs are the same M1 output. Step 4 re-runs it after the NR kernels and the wrapper selection exist, where it actually exercises the NR-1 and NR-4 kernels against the M1 kernel. (This is the intended TDD shape here: the test is meaningful only once the variants exist, so its first run cannot fail on a missing symbol — the wrappers already link.)

- [ ] **Step 3: Add the NR-templated Q5_K decode kernels to `metal/qwen35.metal`**

Append after the M1 `kernel_qwen35_moe_down` (before the generated-tiles marker). These mirror `kernel_qwen4_moe_mid_q4k<NR>` / `kernel_qwen4_moe_down_k<12,NR>` but call `qwen35_row_dot` (Q5_K):
```metal
/* Q5_K decode: NR rows per SIMD group, each row's per-lane dot order identical
 * to the M1 2-row kernel, so output is bit-for-bit the same. */
template <uint NR>
kernel void kernel_qwen35_moe_mid_q5k_nr(
        constant ds4_metal_args_qwen4_moe & args,
        device const char *gate_base, device const char *up_base,
        device const int32_t *selected, device const float *x, device float *mid,
        device const char *sh_gate, device const char *sh_up,
        uint3 tgpig [[threadgroup_position_in_grid]], ushort3 ntg [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y, tok = tgpig.z;
    const uint n_out = args.n_slots + args.has_shared;
    const uint row0 = (tgpig.x * (ntg.x / 32u) + (uint)sgitg) * NR;
    if (row0 >= args.out_rows || slot >= n_out || tok >= args.n_tokens) return;
    const bool shared = slot == args.n_slots;
    const uint type = shared ? args.shared_type : args.weight_type;
    const uint rb = shared ? args.shared_row_bytes : args.row_bytes;
    device const char *gb = shared ? sh_gate : gate_base;
    device const char *ub = shared ? sh_up : up_base;
    const uint64_t ebase = shared ? 0 : (uint64_t)(uint)selected[(uint64_t)tok * args.n_slots + slot] * args.expert_bytes;
    device const float *xt = x + (uint64_t)tok * args.in_dim;
    const uint64_t mb = ((uint64_t)tok * n_out + slot) * args.out_rows;
    for (uint r = row0; r < row0 + NR && r < args.out_rows; r++) {
        const uint64_t off = ebase + (uint64_t)r * rb;
        const float g = qwen35_row_dot(gb + off, xt, type, args.in_dim, tiisg);
        const float u = qwen35_row_dot(ub + off, xt, type, args.in_dim, tiisg);
        if (tiisg == 0) mid[mb + r] = qwen4_silu(g) * u;
    }
}
template <uint NR>
kernel void kernel_qwen35_moe_down_q5k_nr(
        constant ds4_metal_args_qwen4_moe & args,
        device const char *down_base, device const int32_t *selected,
        device const float *mid, device float *part, device const char *sh_down,
        uint3 tgpig [[threadgroup_position_in_grid]], ushort3 ntg [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y, tok = tgpig.z;
    const uint n_out = args.n_slots + args.has_shared;
    const uint row0 = (tgpig.x * (ntg.x / 32u) + (uint)sgitg) * NR;
    if (row0 >= args.out_rows || slot >= n_out || tok >= args.n_tokens) return;
    const bool shared = slot == args.n_slots;
    const uint type = shared ? args.shared_type : args.weight_type;
    const uint rb = shared ? args.shared_row_bytes : args.row_bytes;
    device const char *db = shared ? sh_down : down_base;
    const uint64_t pair = (uint64_t)tok * n_out + slot;
    const uint64_t ebase = shared ? 0 : (uint64_t)(uint)selected[(uint64_t)tok * args.n_slots + slot] * args.expert_bytes;
    device const float *m = mid + pair * args.in_dim;
    for (uint r = row0; r < row0 + NR && r < args.out_rows; r++) {
        const float v = qwen35_row_dot(db + ebase + (uint64_t)r * rb, m, type, args.in_dim, tiisg);
        if (tiisg == 0) part[pair * args.out_rows + r] = v;
    }
}
template [[host_name("kernel_qwen35_moe_mid_q5k_nr1")]] kernel void kernel_qwen35_moe_mid_q5k_nr<1>(constant ds4_metal_args_qwen4_moe &, device const char *, device const char *, device const int32_t *, device const float *, device float *, device const char *, device const char *, uint3, ushort3, ushort, ushort);
template [[host_name("kernel_qwen35_moe_mid_q5k_nr4")]] kernel void kernel_qwen35_moe_mid_q5k_nr<4>(constant ds4_metal_args_qwen4_moe &, device const char *, device const char *, device const int32_t *, device const float *, device float *, device const char *, device const char *, uint3, ushort3, ushort, ushort);
template [[host_name("kernel_qwen35_moe_down_q5k_nr1")]] kernel void kernel_qwen35_moe_down_q5k_nr<1>(constant ds4_metal_args_qwen4_moe &, device const char *, device const int32_t *, device const float *, device float *, device const char *, uint3, ushort3, ushort, ushort);
template [[host_name("kernel_qwen35_moe_down_q5k_nr4")]] kernel void kernel_qwen35_moe_down_q5k_nr<4>(constant ds4_metal_args_qwen4_moe &, device const char *, device const int32_t *, device const float *, device float *, device const char *, uint3, ushort3, ushort, ushort);
```
Add the four enum values before `QWEN4_K_COUNT` (after the Q5_K MM and, if present, the L12 entry — order does not matter as long as the names table matches positionally):
```objc
    QWEN4_K_QWEN35_MOE_MID_Q5K_NR1,
    QWEN4_K_QWEN35_MOE_MID_Q5K_NR4,
    QWEN4_K_QWEN35_MOE_DOWN_Q5K_NR1,
    QWEN4_K_QWEN35_MOE_DOWN_Q5K_NR4,
```
and the matching names in the same positions in `qwen4_kernel_names[]`:
```objc
    "kernel_qwen35_moe_mid_q5k_nr1",
    "kernel_qwen35_moe_mid_q5k_nr4",
    "kernel_qwen35_moe_down_q5k_nr1",
    "kernel_qwen35_moe_down_q5k_nr4",
```
Add the row/group selector next to the M1 `ds4_gpu_qwen35_moe_mid_tensor` (mirroring `qwen4_moe_mr_rows`/`qwen4_moe_mr_groups`):
```objc
/* L8 Q5_K decode split: mid default 1 row / 4 groups, down default 4 rows /
 * 8 groups on M5; 0 (or 2) keeps the M1 2-row / 4-group kernel.  Each row's
 * per-lane dot order is unchanged, so every setting is bit-identical. */
static void qwen35_moe_mr(bool down, uint32_t *nr, uint32_t *groups) {
    const uint64_t v = ds4_gpu_env_u64(down ? "DS4_QWEN35_MOE_MR_DOWN" : "DS4_QWEN35_MOE_MR_MID",
                                       ds4_gpu_device_is_m5_apple_silicon() ? (down ? 4u : 1u) : 0u, 0u, 4u);
    *nr = v >= 4u ? 4u : v == 1u ? 1u : 0u;         /* 0 and 2 -> the M1 2-row kernel */
    *groups = (down && *nr == 4u) ? 8u : 4u;
}
```
In `ds4_gpu_qwen35_moe_mid_tensor`, replace the final `return qwen4_dispatch(QWEN4_K_QWEN35_MOE_MID, ...)` with:
```objc
    uint32_t nr, groups;
    qwen35_moe_mr(false, &nr, &groups);
    const int k = nr == 1u ? QWEN4_K_QWEN35_MOE_MID_Q5K_NR1 : nr == 4u ? QWEN4_K_QWEN35_MOE_MID_Q5K_NR4 : 0;
    if (k)
        return qwen4_dispatch(k, &args, sizeof(args), b, 7,
                              MTLSizeMake((ff_dim + nr * groups - 1u) / (nr * groups), n_out, n_tokens),
                              MTLSizeMake(32u * groups, 1, 1), 0);
    return qwen4_dispatch(QWEN4_K_QWEN35_MOE_MID, &args, sizeof(args), b, 7,
                          MTLSizeMake((ff_dim + 7u) / 8u, n_out, n_tokens), MTLSizeMake(128, 1, 1), 0);
```
and in `ds4_gpu_qwen35_moe_down_tensor`, replace its final `return qwen4_dispatch(QWEN4_K_QWEN35_MOE_DOWN, ...)` with the analogous block (`out_dim` in place of `ff_dim`, and the `_DOWN_Q5K_NR*` enums):
```objc
    uint32_t nr, groups;
    qwen35_moe_mr(true, &nr, &groups);
    const int k = nr == 1u ? QWEN4_K_QWEN35_MOE_DOWN_Q5K_NR1 : nr == 4u ? QWEN4_K_QWEN35_MOE_DOWN_Q5K_NR4 : 0;
    if (k)
        return qwen4_dispatch(k, &args, sizeof(args), b, 5,
                              MTLSizeMake((out_dim + nr * groups - 1u) / (nr * groups), n_out, n_tokens),
                              MTLSizeMake(32u * groups, 1, 1), 0);
    return qwen4_dispatch(QWEN4_K_QWEN35_MOE_DOWN, &args, sizeof(args), b, 5,
                          MTLSizeMake((out_dim + 7u) / 8u, n_out, n_tokens), MTLSizeMake(128, 1, 1), 0);
```
The M1 down wrapper binds 5 buffers (`b`, no `sh_up`); the mid binds 7. `nr` 0 or 2 falls to the M1 2-row kernel. These NR kernels run only where Ornith uses the row kernels (single rows / the 2-row verify's shared-slot path).

- [ ] **Step 4: Build and prove equality + gate 1**

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface && make ds4 test-qwen35-kernels test-qwen4-kernels test-qwen4-q2
for chunk in 2048 64 65; do
  DS4_QWEN35_MOE_MR_MID=1 DS4_QWEN35_MOE_MR_DOWN=4 python3 tests/ornith/gate1.py \
    --ds4-arg --prefill-chunk --ds4-arg $chunk speed-bench/ornith/m4/gate1-l8-chunk$chunk
done
```
Expected: kernel test loop shows NR 1/4 bit-identical to NR 2; `gate1: PASS` with the Task 4 `max_delta` (the per-row dot order is unchanged). If not identical, the row/group split changed the accumulation — revert to the M1 2-row kernel (default 0) and note it.

- [ ] **Step 5: A/B and commit**

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
python3 speed-bench/ornith/m4_ab.py --ds4-model "$DS4_ORNITH_MODEL" --out speed-bench/ornith/m4/speed/l8 \
  --omlx-attach --ds4-env DS4_QWEN35_MOE_MR_MID=1,DS4_QWEN35_MOE_MR_DOWN=4 --contexts 2048,32768 --cold-tokens 0
speed-bench/qwen-regression/run.sh fast
git add metal/qwen35.metal ds4_metal.m tests/test_qwen35_kernels.c speed-bench/ornith/m4/gate1-l8-chunk2048 \
        speed-bench/ornith/m4/gate1-l8-chunk64 speed-bench/ornith/m4/gate1-l8-chunk65 speed-bench/ornith/m4/speed/l8 \
        speed-bench/ornith/m4/speed/LEVERS.md
git commit -m "qwen35: decode lever L8 (Q5_K 1-row mid / 4-row down decode kernels)

NR-templated Q5_K decode kernels, per-row dot order identical to the M1 2-row
kernel (bit-exact); selected by DS4_QWEN35_MOE_MR_MID/_DOWN, default on M5.
gate 1 identical; kept per the A/B (LEVERS.md).

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- metal/qwen35.metal ds4_metal.m tests/test_qwen35_kernels.c speed-bench/ornith/m4/gate1-l8-chunk2048 speed-bench/ornith/m4/gate1-l8-chunk64 speed-bench/ornith/m4/gate1-l8-chunk65 speed-bench/ornith/m4/speed/l8 speed-bench/ornith/m4/speed/LEVERS.md
```

---

### Task 10: Decode lever D — L10 MTP draft vocabulary `DS4_QWEN35_MTP_DRAFT_VOCAB`

Cut the MTP draft head from the full 248,320-row output projection to a frequency-ranked subset, so each cycle reads ~0.13 GiB instead of ~0.50 GiB for the draft. Reuses the qwen4 loader by parameterising its env-name (spec §7.3: Ornith gets its own `DS4_QWEN35_*` knob). The committed artefact is a token-id text list, not a model file.

**Files:**
- Modify: `ds4.c` (parameterise `qwen4_mtp_draft_head_load` to take an env-name; add an Ornith caller), `ds4_qwen35moe.inc` (call it in `qwen35_graph_mtp`'s draft head step — re-read that M2 code in Task 1 and wire at the head GEMM)
- Create: `speed-bench/ornith/m4/ornith-draft-vocab-vi-en-code-64k.txt` (generated by `gguf-tools/frspec_vocab.py`)

**Interfaces:**
- Consumes (from M2, `docs/superpowers/plans/2026-09-26-ornith-m2-mtp.md` Task 2 Step 4): `qwen35_graph_mtp`'s `want_draft` head, which M2 ships as `qwen4_gemv(g->logits, m, w->output, g->mixed, 1u)` (the normed row is `g->mixed`, after `l->nextn_shared_head_norm`) then `ds4_gpu_qwen4_argmax_tensor(g->mtp_argmax, g->mtp_argmax_tmp, g->logits, DS4_N_VOCAB)`, then a single `int32` read of `g->mtp_argmax`. Task 1 re-verified this. If M2 shipped a different head, STOP with a ledger ruling and wire against what M2 shipped.
- Produces: knob `DS4_QWEN35_MTP_DRAFT_VOCAB=<file>`; `ds4.c` `qwen35_mtp_draft_head_load(g, m, w->output)`.

- [ ] **Step 1: Parameterise the shared loader (keeps the qwen4 env name)**

The current loader in `ds4.c` (≈60192) reads a hard-coded env name and prints Qwen3.8-specific messages. Change its signature to take `const char *env_name` and make the messages family-neutral. Apply these exact replacements:

Replace
```c
static bool qwen4_mtp_draft_head_load(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_tensor *w) {
    if (g->draft_head_tried) return g->draft_head != NULL;
    g->draft_head_tried = true;
    const char *path = getenv("DS4_QWEN4_MTP_DRAFT_VOCAB");
    if (!path || !path[0] || w->type != DS4_TENSOR_Q8_0 || w->ndim < 2 || (w->dim[0] % 32u) != 0) return false;
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "ds4: Qwen3.8 MTP draft vocabulary %s: %s\n", path, strerror(errno)); return false; }
```
with
```c
static bool qwen4_mtp_draft_head_load(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_tensor *w,
                                      const char *env_name) {
    if (g->draft_head_tried) return g->draft_head != NULL;
    g->draft_head_tried = true;
    const char *path = getenv(env_name);
    if (!path || !path[0] || w->type != DS4_TENSOR_Q8_0 || w->ndim < 2 || (w->dim[0] % 32u) != 0) return false;
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "ds4: MTP draft vocabulary %s: %s\n", path, strerror(errno)); return false; }
```
Replace the "need 1..V" message
```c
        fprintf(stderr, "ds4: Qwen3.8 MTP draft vocabulary %s: need 1..%" PRIu64 " distinct ids (got %u)\n", path, V - 1u, n);
```
with
```c
        fprintf(stderr, "ds4: MTP draft vocabulary %s: need 1..%" PRIu64 " distinct ids (got %u)\n", path, V - 1u, n);
```
Replace the upload-failed message
```c
        fprintf(stderr, "ds4: Qwen3.8 MTP draft head upload failed\n");
```
with
```c
        fprintf(stderr, "ds4: MTP draft head upload failed\n");
```
Replace the success message
```c
    fprintf(stderr, "ds4: Qwen3.8 MTP draft head: %u of %" PRIu64 " vocabulary rows from %s (%.0f MiB)\n",
            n, V, path, (double)bytes / (1024.0 * 1024.0));
```
with
```c
    fprintf(stderr, "ds4: MTP draft head: %u of %" PRIu64 " vocabulary rows from %s (%.0f MiB)\n",
            n, V, path, (double)bytes / (1024.0 * 1024.0));
```
Update the qwen4 caller (`ds4.c` ≈61519) — replace
```c
    const bool gathered = gpu_argmax && qwen4_mtp_draft_head_load(g, m, w->output);
```
with
```c
    const bool gathered = gpu_argmax && qwen4_mtp_draft_head_load(g, m, w->output, "DS4_QWEN4_MTP_DRAFT_VOCAB");
```
Add the Ornith wrapper directly after `qwen4_mtp_draft_head_load`'s closing brace:
```c
/* Ornith draft head from DS4_QWEN35_MTP_DRAFT_VOCAB (its own knob, §7.3);
 * same gathered-head machinery as Qwen3.8, verify rows use the full output. */
static bool qwen35_mtp_draft_head_load(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_tensor *w) {
    return qwen4_mtp_draft_head_load(g, m, w, "DS4_QWEN35_MTP_DRAFT_VOCAB");
}
```
Both families reuse the same `draft_head`/`draft_ids`/`draft_rows` fields (freed by `qwen4_graph_free`) and `ds4_gpu_qwen4_matmul_q8_0_weights_tensor`; the loader loads once per graph (`draft_head_tried`).

- [ ] **Step 2: Wire the Ornith caller into `qwen35_graph_mtp`'s draft head**

In `ds4_qwen35moe.inc`, `qwen35_graph_mtp` (M2). First, compute the gathered head just before the `if (ok && want_draft) {` head block. Replace
```c
    if (ok && want_draft) {
        ds4_gpu_tensor *x = ds4_gpu_tensor_view(g->mtp_R, (uint64_t)last * row, row);
```
with
```c
    const bool gathered = want_draft && qwen35_mtp_draft_head_load(g, m, w->output);
    const uint32_t head_rows = gathered ? g->draft_rows : (uint32_t)DS4_N_VOCAB;
    if (ok && want_draft) {
        ds4_gpu_tensor *x = ds4_gpu_tensor_view(g->mtp_R, (uint64_t)last * row, row);
```
Then, in the same `&&` chain, replace the final head GEMM + argmax
```c
             qwen4_gemv(g->logits, m, w->output, g->mixed, 1u) &&
             ds4_gpu_qwen4_argmax_tensor(g->mtp_argmax, g->mtp_argmax_tmp, g->logits, DS4_N_VOCAB) != 0;
```
with the gathered/full choice over `head_rows` (`w->output->dim[0]` is the head's in_dim, 2048):
```c
             (gathered ? ds4_gpu_qwen4_matmul_q8_0_weights_tensor(g->logits, g->draft_head,
                             (uint32_t)w->output->dim[0], head_rows, g->mixed)
                       : qwen4_gemv(g->logits, m, w->output, g->mixed, 1u)) &&
             ds4_gpu_qwen4_argmax_tensor(g->mtp_argmax, g->mtp_argmax_tmp, g->logits, head_rows) != 0;
```
Finally, map the gathered argmax back through `g->draft_ids`. Replace the post-read block
```c
    if (ok && want_draft) {
        int32_t token = -1;
        ok = ds4_gpu_tensor_read(g->mtp_argmax, 0, &token, sizeof(token)) != 0 &&
             token >= 0 && token < (int32_t)DS4_N_VOCAB;
        if (ok) *draft_out = token;
    }
```
with
```c
    if (ok && want_draft) {
        int32_t token = -1;
        ok = ds4_gpu_tensor_read(g->mtp_argmax, 0, &token, sizeof(token)) != 0 &&
             token >= 0 && token < (int32_t)head_rows;
        if (ok && gathered) token = g->draft_ids[token];
        if (ok && (token < 0 || token >= (int32_t)DS4_N_VOCAB)) ok = false;
        if (ok) *draft_out = token;
    }
```
No vocab file leaves `gathered` false and `head_rows == DS4_N_VOCAB`, so the M2 head is unchanged. The verify rows always use the full `output` (this only touches the `want_draft` head), so committed tokens are unchanged and only the proposed draft can differ.

- [ ] **Step 3: Generate the Ornith draft vocabulary offline (no GPU, no model run)**

The tokenizer is `gpt2` BPE with 248,320 ids; Vietnamese sits at high ids, so a plain id prefix would collapse Vietnamese acceptance — use `frspec_vocab.py` with weighted corpora (the tool already documents this). Extract `tokenizer.json` from the GGUF or reuse the oMLX snapshot's tokenizer, then:
```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
TOK=$HOME/.local/share/ai-gateway/omlx-models/Shiftedx--ornith-1.5-35b-a3b-abliterated-attention8-bf16recurrence-vision-mtplx/tokenizer.json
~/.local/ai-gateway-env/bin/python3 gguf-tools/frspec_vocab.py \
  --tokenizer "$TOK" --out speed-bench/ornith/m4/ornith-draft-vocab-vi-en-code-64k.txt --size 65536 \
  --corpus code:0.4:"$PWD" --corpus en:0.3:speed-bench/promessi_sposi.txt --corpus py:0.3:"$(python3 -c 'import sys;print(sys.prefix)')/lib"
wc -l speed-bench/ornith/m4/ornith-draft-vocab-vi-en-code-64k.txt
```
`frspec_vocab.py` assumes `V = 248320` (matches Ornith); confirm the header. If `tokenizer.json` is absent, extract it from the GGUF metadata with the stdlib GGUF reader pattern used in `tests/ornith/make_bad_gguf.py`. The list is a committed text fixture (token ids), not a model artefact, so it may go in git.

- [ ] **Step 4: Build; gate 1 unchanged (verify rows use the full head); MTP byte-identical check**

The draft vocabulary changes only which draft is proposed, never the committed token, so at temperature 0 the reply is unchanged. Re-run gate 1 and the M2 MTP byte-identity test (`tests/ornith/test_mtp_cli.py`, run via `make test-qwen35-mtp` on the session path) with `DS4_QWEN35_MTP_DRAFT_VOCAB` set, and confirm `--mtp` output stays byte-identical to plain decode:
```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface && make ds4
export V=$PWD/speed-bench/ornith/m4/ornith-draft-vocab-vi-en-code-64k.txt
DS4_QWEN35_MTP_DRAFT_VOCAB=$V python3 tests/ornith/gate1.py speed-bench/ornith/m4/gate1-l10
DS4_QWEN35_MTP_DRAFT_VOCAB=$V python3 tests/ornith/test_mtp_cli.py speed-bench/ornith/m4/mtpcli-l10
```
Expected: `gate1: PASS`; `test_mtp_cli: PASS` (plain == session == `--mtp` byte-identical across the prefill chunks with the vocab set).

- [ ] **Step 5: A/B, MTP acceptance with `--mtp-timing`, commit**

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
python3 speed-bench/ornith/m4_ab.py --ds4-model "$DS4_ORNITH_MODEL" --out speed-bench/ornith/m4/speed/l10 \
  --omlx-attach --ds4-env DS4_QWEN35_MTP_DRAFT_VOCAB=$PWD/speed-bench/ornith/m4/ornith-draft-vocab-vi-en-code-64k.txt \
  --contexts 2048,32768 --cold-tokens 0
# acceptance stays comparable: a quick --mtp-timing run at 2K with and without the vocab
speed-bench/qwen-regression/run.sh fast
git add ds4.c ds4_qwen35moe.inc speed-bench/ornith/m4/ornith-draft-vocab-vi-en-code-64k.txt \
        speed-bench/ornith/m4/gate1-l10 speed-bench/ornith/m4/mtpcli-l10 speed-bench/ornith/m4/speed/l10 \
        speed-bench/ornith/m4/speed/LEVERS.md
git commit -m "qwen35: decode lever L10 (DS4_QWEN35_MTP_DRAFT_VOCAB)

Parameterise the shared MTP draft-head loader by env name; Ornith gets its own
knob and a frequency-ranked 64K draft vocabulary. Verify rows keep the full
head, so temperature-0 replies are byte-identical. Kept per the A/B (LEVERS.md).

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- ds4.c ds4_qwen35moe.inc speed-bench/ornith/m4/ornith-draft-vocab-vi-en-code-64k.txt speed-bench/ornith/m4/gate1-l10 speed-bench/ornith/m4/mtpcli-l10 speed-bench/ornith/m4/speed/l10 speed-bench/ornith/m4/speed/LEVERS.md
```

---

### Task 11: Long context — `DS4_QWEN35_KV` (conditional) + L12 shared-KV verify (conditional)

**Trigger (KV mode):** run only if the post-lever gate-3 A/B (Task 12 Step 1's interleaved run, or a quick 128K A/B here) shows ds4 decode-with-MTP below oMLX at 128K. **Trigger (L12):** run only if, after the best KV mode, MTP still loses at 128K and the Task 5 profile confirms the T=2 verify's attention time at 128K is roughly double the T=1 decode's (the doubled KV read). Both carry complete code so the executor can implement on trigger without a second planning pass. A non-F16 KV mode passes gate 1 within its own calibrated tolerance and then gate 2 again before it becomes a default (spec §8 gate 3, `DECISIONS.md` §4.5).

**Files:**
- Modify: `ds4.c` (`DS4_QWEN35_KV` env parse next to `qwen35_prefill_chunk_tokens`; the Ornith memory estimate `kv_row`; the M3 Ornith payload `qwen35_payload_body_bytes`/`qwen35_session_save_payload`/`qwen35_session_load_payload` and the `ds4_session_payload_bytes` Ornith guard), `ds4_qwen35moe.inc` (`qwen35_graph_alloc` FP8/Q4 caches, `qwen35_graph_attention_prep`/`qwen35_graph_attend` pass the FP8 caches + mode), `ds4_metal.m` (`ds4_gpu_qwen35_attn_prep_tensor` gains `kv_mode` + the four FP8 tensors; L12 wrapper + enum/name), `ds4_gpu.h`; L12: `metal/qwen35.metal` + `ds4_qwen35moe.inc`
- Modify: `tests/test_qwen35_kernels.c` (`test_attn_prep_kv_modes`, and for L12 `test_attn_decode2_matches_perrow`), `tests/ds4_test.c` (the M3 `--qwen35-payloads` case loops the KV modes)

**Interfaces:**
- Consumes: the mode-aware qwen4 prep/decode kernels (already read modes 0/1/2); `qwen4_kv_fp8_active`/`qwen4_kv_mode`/`qwen4_kcache`/`qwen4_vcache` (`ds4.c`, `static` above the `.inc` include, callable from the `.inc`); the qwen4 FP8/Q4 cache + scale struct fields on `ds4_qwen4_gpu_graph`; `qwen4_payload_kv_fp8_bytes(rows, q4)`, `qwen4_payload_kv_bytes(rows)`, the M3 `qwen35_payload_*` functions and `DS4_QWEN35_PAYLOAD_TAG`.
- Produces: knob `DS4_QWEN35_KV` (`f16`|`fp8`|`q4`); the payload tags `DS4_QWEN35_PAYLOAD_TAG_FP8`/`_Q4`; for L12, `ds4_gpu_qwen35_attn_decode2_tensor` and `kernel_qwen35_attn_decode2`.

- [ ] **Step 1: Decide whether the KV-mode trigger fired**

Box free, user OK; a focused 128K A/B on the best configuration so far (levers kept in Tasks 7–10, F16 KV):
```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
python3 speed-bench/ornith/m4_ab.py --ds4-model "$DS4_ORNITH_MODEL" --out speed-bench/ornith/m4/speed/kv-trigger \
  --omlx-attach --ds4-env <kept-levers> --contexts 131072 --cold-tokens 0
```
If ds4 decode ≥ oMLX at 128K, record "KV mode not needed" in `LEVERS.md` and skip Steps 2–9 (F16 stays the default, spec §5). Otherwise continue.

- [ ] **Step 2: `DS4_QWEN35_KV` env parse + FP8/Q4 caches in `qwen35_graph_alloc`**

In `ds4.c`, add this helper directly after `qwen35_prefill_chunk_tokens` (≈40139), so it is in scope both for the memory estimate below it (Step 4) and for the `.inc` graph alloc (included at ≈61722):
```c
/* Ornith KV storage: f16 (default) | fp8 (E4M3) | q4 (4-bit). Own knob per
 * §7.3.  4-bit needs head dim 256 (the qwen4 4-bit KV path's constraint). */
static DS4_MAYBE_UNUSED void qwen35_kv_mode_env(bool *fp8, bool *q4) {
    const char *e = getenv("DS4_QWEN35_KV");
    *fp8 = *q4 = false;
    if (!e || !e[0] || !strcmp(e, "f16")) return;
    if (!strcmp(e, "fp8")) *fp8 = true;
    else if (!strcmp(e, "q4") && DS4_N_HEAD_DIM == 256u) { *fp8 = true; *q4 = true; }
    else fprintf(stderr, "ds4: DS4_QWEN35_KV=%s ignored (use f16|fp8|q4)\n", e);
}
```
In `qwen35_graph_alloc` (`ds4_qwen35moe.inc`), set the mode right after `g->gdn_silu = true;`:
```c
    g->gdn_silu = true;
    qwen35_kv_mode_env(&g->kv_fp8, &g->kv_q4);
```
Then replace the M2 F16-only cache-allocation block
```c
    const uint64_t kv_bytes = (uint64_t)ctx_cap * kv_dim * 2u;
    const uint64_t state_n = v_dim * DS4_N_LIN_HEAD_DIM;
    const uint64_t hist_n = (uint64_t)(DS4_N_LIN_CONV - 1u) * conv_dim;
    const uint32_t n_layers = mtp ? DS4_N_LAYER : DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
    for (uint32_t il = 0; il < n_layers && ok; il++) {
        if (ds4_qwen35_layer_is_attention(il)) {
            g->layer_k_cache[il] = ds4_gpu_tensor_alloc(kv_bytes);
            g->layer_v_cache[il] = ds4_gpu_tensor_alloc(kv_bytes);
            ok = g->layer_k_cache[il] && g->layer_v_cache[il];
        } else {
```
with the mode-aware version (the GDN branch is unchanged; only the attention branch splits by mode, mirroring `qwen4_ctx_bufs_alloc` at `ds4.c` ≈59644):
```c
    const uint64_t kv_bytes = (uint64_t)ctx_cap * kv_dim * 2u;
    const uint64_t kv_fp8_bytes = g->kv_q4 ? (uint64_t)ctx_cap * kv_dim / 2u : (uint64_t)ctx_cap * kv_dim;
    const uint64_t scale_bytes = (uint64_t)ctx_cap * (kv_dim / 64u) * 2u;
    const uint64_t state_n = v_dim * DS4_N_LIN_HEAD_DIM;
    const uint64_t hist_n = (uint64_t)(DS4_N_LIN_CONV - 1u) * conv_dim;
    const uint32_t n_layers = mtp ? DS4_N_LAYER : DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
    for (uint32_t il = 0; il < n_layers && ok; il++) {
        if (ds4_qwen35_layer_is_attention(il)) {
            if (g->kv_fp8) {
                g->layer_k_cache_fp8[il] = ds4_gpu_tensor_alloc(kv_fp8_bytes);
                g->layer_v_cache_fp8[il] = ds4_gpu_tensor_alloc(kv_fp8_bytes);
                g->layer_k_scale[il] = ds4_gpu_tensor_alloc(scale_bytes);
                g->layer_v_scale[il] = ds4_gpu_tensor_alloc(scale_bytes);
                ok = g->layer_k_cache_fp8[il] && g->layer_v_cache_fp8[il] &&
                     g->layer_k_scale[il] && g->layer_v_scale[il];
            } else {
                g->layer_k_cache[il] = ds4_gpu_tensor_alloc(kv_bytes);
                g->layer_v_cache[il] = ds4_gpu_tensor_alloc(kv_bytes);
                ok = g->layer_k_cache[il] && g->layer_v_cache[il];
            }
        } else {
```
`qwen4_kv_fp8_active(g)` keys off `layer_k_cache_fp8[il] != NULL`, so `qwen4_kcache`/`qwen4_vcache`/`qwen4_kv_mode` return the right buffer/mode. `qwen4_graph_free` already frees `layer_k_cache_fp8`/`layer_v_cache_fp8`/`layer_k_scale`/`layer_v_scale`. `DS4_MAYBE_UNUSED` on the helper avoids an unused warning in the CPU build where the estimate's Metal branch is compiled out.

- [ ] **Step 3: `ds4_gpu_qwen35_attn_prep_tensor` gains the KV mode**

Replace the whole M1 `ds4_gpu_qwen35_attn_prep_tensor` (`ds4_metal.m` ≈50550) with the mode-aware version. It follows the qwen4 prep wrapper's FP8 handling (`ds4_metal.m` ≈50504): `args.fp8 = kv_mode`, the K/V and scale buffer sizes derived from the mode, `b[15..18]` bound to the real FP8/scale tensors (aliased to the F16 cache when `kv_mode == 0`), and the indexer slots still aliased away (Ornith has no indexer):
```objc
/* Ornith attention prep: the qwen4 prep kernel with only its query and KV
 * slots dispatched (grid.x = n_head + n_head_kv), no QSA indexer.  kv_mode 0
 * F16, 1 E4M3, 2 4-bit: the kernel's K/V slot packs the chosen mode. */
int ds4_gpu_qwen35_attn_prep_tensor(
        ds4_gpu_tensor *q_out, ds4_gpu_tensor *gate_out, ds4_gpu_tensor *k_cache, ds4_gpu_tensor *v_cache,
        const ds4_gpu_tensor *qg, const ds4_gpu_tensor *kproj, const ds4_gpu_tensor *vproj,
        const ds4_gpu_tensor *pos3, const void *model_map, uint64_t model_size,
        uint64_t g_q_offset, uint64_t g_k_offset,
        uint32_t n_tokens, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim, uint32_t n_rot,
        uint32_t pos0, uint32_t cache_cap, float rope_base, float eps,
        ds4_gpu_tensor *k_cache_fp8, ds4_gpu_tensor *v_cache_fp8,
        ds4_gpu_tensor *k_scale, ds4_gpu_tensor *v_scale, uint32_t kv_mode) {
    struct {
        uint32_t n_tokens, n_head, n_head_kv, head_dim, n_rot, n_idx_head, idx_dim, pos0, cache_cap;
        float rope_base, eps; uint32_t fp8; float rope_mscale; float rope_freq[32]; uint32_t ik_ring, kv_simq;
    } args = { n_tokens, n_head, n_head_kv, head_dim, n_rot, 0u, 0u, pos0, cache_cap,
               rope_base, eps, kv_mode, 1.0f, { 0 }, 0u, 0u };
    qwen4_rope_fill(args.rope_freq, &args.rope_mscale, n_rot, rope_base);
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)n_tokens * n_head_kv * head_dim * sizeof(float);
    const uint64_t kv_elems = (uint64_t)cache_cap * n_head_kv * head_dim;
    const uint64_t cache_bytes = kv_mode == 2u ? kv_elems / 2u : kv_elems * (kv_mode ? 1u : 2u);
    const uint64_t scale_bytes = kv_mode ? (uint64_t)cache_cap * (n_head_kv * head_dim / 64u) * 2u : cache_bytes;
    ds4_gpu_tensor *kf  = kv_mode ? k_cache_fp8 : k_cache;
    ds4_gpu_tensor *vf  = kv_mode ? v_cache_fp8 : v_cache;
    ds4_gpu_tensor *ksc = kv_mode ? k_scale     : k_cache;
    ds4_gpu_tensor *vsc = kv_mode ? v_scale     : v_cache;
    qwen4_bind b[19];
    if (n_tokens == 0 || head_dim < 32 || head_dim > 256 || (head_dim % 32) != 0 || n_rot > 64 ||
        (n_rot % 2) != 0 || (uint64_t)pos0 + n_tokens > cache_cap || n_head_kv == 0 || (n_head % n_head_kv) != 0 ||
        !qwen4_bind_tensor(&b[0], qg, 2u * q_bytes, "attn q/gate projection") ||
        !qwen4_bind_tensor(&b[1], kproj, kv_bytes, "attn k projection") ||
        !qwen4_bind_tensor(&b[2], vproj, kv_bytes, "attn v projection") ||
        !qwen4_bind_weight(&b[5], model_map, model_size, g_q_offset, (uint64_t)head_dim * sizeof(float),
                           "attn q_norm") ||
        !qwen4_bind_weight(&b[6], model_map, model_size, g_k_offset, (uint64_t)head_dim * sizeof(float),
                           "attn k_norm") ||
        !qwen4_bind_tensor(&b[8], q_out, q_bytes, "attn q") ||
        !qwen4_bind_tensor(&b[9], gate_out, q_bytes, "attn gate") ||
        !qwen4_bind_tensor(&b[10], k_cache, cache_bytes, "k cache") ||
        !qwen4_bind_tensor(&b[11], v_cache, cache_bytes, "v cache") ||
        !qwen4_bind_tensor(&b[14], pos3, (uint64_t)cache_cap * 16u, "rope positions") ||
        !qwen4_bind_tensor(&b[15], kf, cache_bytes, "k cache fp8") ||
        !qwen4_bind_tensor(&b[16], vf, cache_bytes, "v cache fp8") ||
        !qwen4_bind_tensor(&b[17], ksc, scale_bytes, "k scale") ||
        !qwen4_bind_tensor(&b[18], vsc, scale_bytes, "v scale")) {
        return 0;
    }
    b[3] = b[0];  b[4] = b[0];  b[7] = b[5];              /* indexer inputs, never read */
    b[12] = b[8]; b[13] = b[8];                           /* indexer outputs, never written */
    return qwen4_dispatch(QWEN4_K_ATTN_PREP, &args, sizeof(args), b, 19,
                          MTLSizeMake(n_head + n_head_kv, n_tokens, 1), MTLSizeMake(32, 1, 1), 0);
}
```
In `ds4_gpu.h`, replace the M1 `ds4_gpu_qwen35_attn_prep_tensor` prototype with the extended one (same parameters as above). In `ds4_qwen35moe.inc`, update `qwen35_graph_attention_prep` — replace
```c
           ds4_gpu_qwen35_attn_prep_tensor(g->q, g->gate, g->layer_k_cache[il], g->layer_v_cache[il],
                                           g->qg, g->kp, g->vp, g->pos3, m->map, m->size,
                                           l->attn_q_norm->abs_offset, l->attn_k_norm->abs_offset,
                                           T, DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT,
                                           pos0, g->alloc_cap, DS4_ROPE_FREQ_BASE, DS4_RMS_EPS) != 0;
```
with
```c
           ds4_gpu_qwen35_attn_prep_tensor(g->q, g->gate, qwen4_kcache(g, il), qwen4_vcache(g, il),
                                           g->qg, g->kp, g->vp, g->pos3, m->map, m->size,
                                           l->attn_q_norm->abs_offset, l->attn_k_norm->abs_offset,
                                           T, DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT,
                                           pos0, g->alloc_cap, DS4_ROPE_FREQ_BASE, DS4_RMS_EPS,
                                           g->layer_k_cache_fp8[il], g->layer_v_cache_fp8[il],
                                           g->layer_k_scale[il], g->layer_v_scale[il], qwen4_kv_mode(g)) != 0;
```
In `qwen35_graph_attend`, both `ds4_gpu_qwen4_attn_decode_tensor` calls (the `r0 == 0 && !per_row` branch and the per-row loop) pass `g->layer_k_cache[il], g->layer_v_cache[il]` and `NULL, NULL, NULL, NULL, 0u` for the FP8 args; replace each call's cache args with `qwen4_kcache(g, il), qwen4_vcache(g, il)` and its five FP8 args with `g->layer_k_cache_fp8[il], g->layer_v_cache_fp8[il], g->layer_k_scale[il], g->layer_v_scale[il], qwen4_kv_mode(g)` (as the qwen4 attention core does, `ds4.c` ≈60617). When `kv_mode == 0` these resolve to the F16 caches and NULLs behave as today.

- [ ] **Step 4: Memory estimate + context cap**

In the Ornith branch of `ds4_context_memory_estimate_with_prefill_mode` (the M2 version, `ds4.c` ≈40180), replace the fixed `kv_row`
```c
        const uint64_t E = DS4_N_EMBD, kv_row = 2ull * DS4_N_HEAD_KV * DS4_N_HEAD_DIM * 2u;
```
with a mode-aware `kv_row` (K + V bytes per token per attention layer, plus the per-64-block fp16 scales in the packed modes):
```c
        const uint64_t E = DS4_N_EMBD;
        bool kv_fp8 = false, kv_q4 = false;
        qwen35_kv_mode_env(&kv_fp8, &kv_q4);
        const uint64_t kv_row = kv_fp8
            ? (kv_q4 ? (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM            /* K+V 4-bit: 2 * (Hkv*D/2) */
                     : 2ull * DS4_N_HEAD_KV * DS4_N_HEAD_DIM)             /* K+V E4M3: 2 * (Hkv*D) */
              + 2ull * (DS4_N_HEAD_KV * DS4_N_HEAD_DIM / 64u) * 2u        /* K and V per-64-block fp16 scales */
            : 2ull * DS4_N_HEAD_KV * DS4_N_HEAD_DIM * 2u;                 /* F16 K+V */
```
This lets the server admit a 262144 context in FP8/Q4 (2.6 GiB / 1.4 GiB of KV instead of 5.0 GiB). `qwen35_kv_mode_env` is defined in Step 2 above this function.

- [ ] **Step 5: Extend the kernel test to modes 1/2 (prep + decode round-trip)**

The prep's packed layout is exactly what the shared decode kernel reads, so the honest, self-contained check is a round-trip: prep the same K/V in F16 (mode 0) and in mode `m`, decode both with the same query through `ds4_gpu_qwen4_attn_decode_tensor`, and require the mode-`m` attention output within a per-mode tolerance of the F16 output (no hand-reproduction of the per-64-block packing). Add to `tests/test_qwen35_kernels.c`, before `int main`:
```c
/* Ornith attention prep in an FP8/4-bit KV mode: the packed K/V a decode
 * reads.  Prep + decode in mode m must match the F16 prep + decode within the
 * quant's tolerance; the shared decode kernel is the reader, so no per-block
 * packing is reproduced here. */
static void test_attn_prep_kv_modes(arena_t *a, uint32_t mode) {
    const uint32_t T = 4, H = 16, Hkv = 2, D = 256, n_rot = 64, pos0 = 5, cap = 16;
    const double base = 1.0e7;
    const float scale = 1.0f / sqrtf((float)D);
    double *gq, *gk;
    const uint64_t gq_off = arena_f32(a, D, &gq, 0.5f, 1.5f);
    const uint64_t gk_off = arena_f32(a, D, &gk, 0.5f, 1.5f);
    float *qg = rand_vec((uint64_t)T * H * 2 * D, 1.0f);
    float *kp = rand_vec((uint64_t)T * Hkv * D, 1.0f), *vp = rand_vec((uint64_t)T * Hkv * D, 1.0f);
    uint32_t pos3[16 * 4];
    for (uint32_t p = 0; p < cap; p++) { pos3[p * 4] = pos3[p * 4 + 1] = pos3[p * 4 + 2] = p; pos3[p * 4 + 3] = 0; }
    ds4_gpu_tensor *gqg = upload(qg, (uint64_t)T * H * 2 * D);
    ds4_gpu_tensor *gkp = upload(kp, (uint64_t)T * Hkv * D), *gvp = upload(vp, (uint64_t)T * Hkv * D);
    ds4_gpu_tensor *gpos = ds4_gpu_tensor_alloc(sizeof(pos3));
    require_ok(gpos && ds4_gpu_tensor_write(gpos, 0, pos3, sizeof(pos3)), "kv-mode pos");
    /* F16 reference: prep into F16 caches, decode into ref */
    ds4_gpu_tensor *gq0 = upload(NULL, (uint64_t)T * H * D), *ggate0 = upload(NULL, (uint64_t)T * H * D);
    ds4_gpu_tensor *kc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *vc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *o_ref = upload(NULL, (uint64_t)T * H * D);
    require_ok(kc && vc && ds4_gpu_qwen35_attn_prep_tensor(gq0, ggate0, kc, vc, gqg, gkp, gvp, gpos,
                   a->base, a->size, gq_off, gk_off, T, H, Hkv, D, n_rot, pos0, cap, (float)base, 1e-6f,
                   kc, vc, kc, vc, 0u), "prep f16");
    require_ok(ds4_gpu_qwen4_attn_decode_tensor(o_ref, gq0, ggate0, kc, vc, NULL, NULL, NULL, T,
                   H, Hkv, D, pos0, 0u, 0u, scale, NULL, NULL, NULL, NULL, 0u), "decode f16");
    float *ref = download(o_ref, (uint64_t)T * H * D);
    /* mode m: prep into packed caches + scales, decode with the same query */
    const uint64_t kv_elems = (uint64_t)cap * Hkv * D;
    ds4_gpu_tensor *gq1 = upload(NULL, (uint64_t)T * H * D), *ggate1 = upload(NULL, (uint64_t)T * H * D);
    ds4_gpu_tensor *kf = ds4_gpu_tensor_alloc(mode == 2u ? kv_elems / 2u : kv_elems);
    ds4_gpu_tensor *vf = ds4_gpu_tensor_alloc(mode == 2u ? kv_elems / 2u : kv_elems);
    ds4_gpu_tensor *ksb = ds4_gpu_tensor_alloc((uint64_t)cap * (Hkv * D / 64u) * 2u);
    ds4_gpu_tensor *vsb = ds4_gpu_tensor_alloc((uint64_t)cap * (Hkv * D / 64u) * 2u);
    ds4_gpu_tensor *o_m = upload(NULL, (uint64_t)T * H * D);
    require_ok(kf && vf && ksb && vsb && ds4_gpu_qwen35_attn_prep_tensor(gq1, ggate1, kf, vf, gqg, gkp, gvp,
                   gpos, a->base, a->size, gq_off, gk_off, T, H, Hkv, D, n_rot, pos0, cap, (float)base, 1e-6f,
                   kf, vf, ksb, vsb, mode), "prep kv mode");
    require_ok(ds4_gpu_qwen4_attn_decode_tensor(o_m, gq1, ggate1, kf, vf, NULL, NULL, NULL, T,
                   H, Hkv, D, pos0, 0u, 0u, scale, kf, vf, ksb, vsb, mode), "decode kv mode");
    float *got = download(o_m, (uint64_t)T * H * D);
    double worst = 0.0, sc = 1e-6;
    for (uint64_t i = 0; i < (uint64_t)T * H * D; i++) {
        if (fabs(ref[i]) > sc) sc = fabs(ref[i]);
        if (fabs((double)got[i] - ref[i]) > worst) worst = fabs((double)got[i] - ref[i]);
    }
    printf("  attn prep+decode kv mode %u: attn out max|d| %.3e (rel %.3e)\n", mode, worst, worst / sc);
    require_ok(worst <= (mode == 2u ? 6e-2 : 2e-2) * sc, "kv-mode attention within tolerance");
    free(ref); free(got); free(qg); free(kp); free(vp); free(gq); free(gk);
    ds4_gpu_tensor_free(gqg); ds4_gpu_tensor_free(gkp); ds4_gpu_tensor_free(gvp); ds4_gpu_tensor_free(gpos);
    ds4_gpu_tensor_free(gq0); ds4_gpu_tensor_free(ggate0); ds4_gpu_tensor_free(kc); ds4_gpu_tensor_free(vc);
    ds4_gpu_tensor_free(o_ref); ds4_gpu_tensor_free(gq1); ds4_gpu_tensor_free(ggate1); ds4_gpu_tensor_free(kf);
    ds4_gpu_tensor_free(vf); ds4_gpu_tensor_free(ksb); ds4_gpu_tensor_free(vsb); ds4_gpu_tensor_free(o_m);
}
```
In `main`, add after the M1 attention-prep call:
```c
    printf("qwen35 attention prep + decode (KV modes)\n");
    test_attn_prep_kv_modes(&arena, 1u);
    test_attn_prep_kv_modes(&arena, 2u);
```

- [ ] **Step 6: Build and prove the modes write/read correctly**

Run: `make ds4 test-qwen35-kernels test-qwen4-kernels test-qwen4-q2`
Expected: the two `attn prep+decode kv mode` lines pass within tolerance; Qwen3.8 tests unchanged. `git diff --stat metal/qwen4.metal` shows nothing.

- [ ] **Step 7: Gate 1 unchanged against the committed llama.cpp references**

The M1 gate-1 references (`tests/ornith/ref/*.json`) and the calibrated `tol`/`tie` (`tests/ornith/tolerance.json`) were recorded from llama.cpp's F16-KV runs, so the calibrated tolerance already measures accumulation-order noise against an F16 KV — running ds4 with a quantized KV mode through `tests/ornith/gate1.py` unchanged therefore measures the KV quantization error **directly** against that reference. Spec §8 forbids hand-picked tolerances, so a KV mode is a gate-1 candidate only if it PASSES the calibrated gate as-is; a widened tolerance is not used. Run each mode:
```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
for mode in f16 fp8 q4; do
  DS4_QWEN35_KV=$mode python3 tests/ornith/gate1.py speed-bench/ornith/m4/gate1-kv-$mode
done
```
Expected: `gate1: PASS` for `f16` (identical to Task 4) and for any KV mode that stays within the calibrated tolerance. A mode that prints `FAIL` (its quantization moves a probable-token log-prob past `tol`, or flips a pre-tie selection) is **not a candidate** and is dropped; record it in `LEVERS.md`.

- [ ] **Step 8: Extend the disk-KV payload to the non-F16 mode (only if a mode is adopted)**

M3 Task 5 stores F16 KV only: `qwen35_payload_body_bytes`/`qwen35_session_save_payload`/`qwen35_session_load_payload` and the `ds4_session_payload_bytes` Ornith guard all refuse when `g->kv_fp8 || g->kv_q4`. If a non-F16 mode is adopted, the payload must carry the packed caches (as Qwen3.8's `DS4_QWEN4_PAYLOAD_TAG_KV_Q4` does), with a mode-specific tag so a checkpoint can never cross-load into a different KV mode.

Add the tags next to M3's `#define DS4_QWEN35_PAYLOAD_TAG 0x51573501u`:
```c
#define DS4_QWEN35_PAYLOAD_TAG_FP8 0x51573502u   /* E4M3 KV */
#define DS4_QWEN35_PAYLOAD_TAG_Q4  0x51573503u   /* 4-bit KV */
```
Give `qwen35_payload_body_bytes` the KV mode (change its signature and all three call sites — `ds4_session_payload_bytes`, save, load — to pass `g->kv_fp8, g->kv_q4`). Replace M3's function
```c
static uint64_t qwen35_payload_body_bytes(uint32_t rows, uint32_t mtp_rows, bool mtp) {
    ...
        if (ds4_qwen35_layer_is_attention(il)) {
            bytes += 2u * qwen4_payload_kv_bytes(ds4_qwen35_layer_is_nextn(il) ? mtp_rows : rows);
        } else {
    ...
}
```
with
```c
static uint64_t qwen35_payload_body_bytes(uint32_t rows, uint32_t mtp_rows, bool mtp, bool fp8, bool q4) {
    uint64_t bytes = (uint64_t)rows * sizeof(uint32_t) + (uint64_t)DS4_N_VOCAB * sizeof(float);
    bytes += sizeof(uint32_t);
    if (mtp) bytes += (uint64_t)DS4_N_EMBD * sizeof(float);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (ds4_qwen35_layer_is_nextn(il) && !mtp) continue;
        if (ds4_qwen35_layer_is_attention(il)) {
            const uint32_t r = ds4_qwen35_layer_is_nextn(il) ? mtp_rows : rows;
            if (fp8)
                bytes += 2u * qwen4_payload_kv_fp8_bytes(r, q4)
                       + 4ull * (uint64_t)r * (DS4_N_HEAD_KV * DS4_N_HEAD_DIM / 64u);   /* K,V fp16 block scales */
            else
                bytes += 2u * qwen4_payload_kv_bytes(r);
        } else {
            bytes += qwen4_payload_lin_state_bytes() + qwen4_payload_lin_hist_bytes();
        }
    }
    return bytes + (uint64_t)rows * 16u;
}
```
In the `ds4_session_payload_bytes` Ornith branch, drop the `g->kv_fp8 || g->kv_q4` disqualifier (it currently forces a return 0) and pass the mode to the body-bytes call:
```c
        if (!s->qwen35_graph_ready || !s->checkpoint_valid || g->pos != (uint32_t)s->checkpoint.len) return 0;
        const uint32_t rows = (uint32_t)s->checkpoint.len;
        const bool mtp = g->mtp_h != NULL;
        const uint32_t mtp_rows = mtp ? (g->mtp_pos < rows ? g->mtp_pos : rows) : 0u;
        return (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t) +
               qwen35_payload_body_bytes(rows, mtp_rows, mtp, g->kv_fp8, g->kv_q4);
```
In `qwen35_session_save_payload`, remove the F16-only refusal
```c
    if (g->kv_fp8 || g->kv_q4) {
        payload_set_err(err, errlen, "Ornith checkpoints support the F16 KV cache only");
        return 1;
    }
```
set the header tag by mode and write the packed caches + scales. Replace the header field `DS4_QWEN35_PAYLOAD_TAG,` with `g->kv_q4 ? DS4_QWEN35_PAYLOAD_TAG_Q4 : g->kv_fp8 ? DS4_QWEN35_PAYLOAD_TAG_FP8 : DS4_QWEN35_PAYLOAD_TAG,` and replace the attention-layer write
```c
        if (ds4_qwen35_layer_is_attention(il)) {
            const uint64_t kvb = qwen4_payload_kv_bytes(ds4_qwen35_layer_is_nextn(il) ? mtp_rows : rows);
            rc = payload_write_tensor_span(fp, g->layer_k_cache[il], 0, kvb, buf, DS4_SESSION_IO_CHUNK, err, errlen);
            if (rc == 0)
                rc = payload_write_tensor_span(fp, g->layer_v_cache[il], 0, kvb, buf, DS4_SESSION_IO_CHUNK, err, errlen);
        } else {
```
with
```c
        if (ds4_qwen35_layer_is_attention(il)) {
            const uint32_t r = ds4_qwen35_layer_is_nextn(il) ? mtp_rows : rows;
            if (g->kv_fp8) {
                const uint64_t kvb = qwen4_payload_kv_fp8_bytes(r, g->kv_q4);
                const uint64_t scb = (uint64_t)r * (DS4_N_HEAD_KV * DS4_N_HEAD_DIM / 64u) * 2u;
                rc = payload_write_tensor_span(fp, g->layer_k_cache_fp8[il], 0, kvb, buf, DS4_SESSION_IO_CHUNK, err, errlen);
                if (rc == 0) rc = payload_write_tensor_span(fp, g->layer_v_cache_fp8[il], 0, kvb, buf, DS4_SESSION_IO_CHUNK, err, errlen);
                if (rc == 0) rc = payload_write_tensor_span(fp, g->layer_k_scale[il], 0, scb, buf, DS4_SESSION_IO_CHUNK, err, errlen);
                if (rc == 0) rc = payload_write_tensor_span(fp, g->layer_v_scale[il], 0, scb, buf, DS4_SESSION_IO_CHUNK, err, errlen);
            } else {
                const uint64_t kvb = qwen4_payload_kv_bytes(r);
                rc = payload_write_tensor_span(fp, g->layer_k_cache[il], 0, kvb, buf, DS4_SESSION_IO_CHUNK, err, errlen);
                if (rc == 0) rc = payload_write_tensor_span(fp, g->layer_v_cache[il], 0, kvb, buf, DS4_SESSION_IO_CHUNK, err, errlen);
            }
        } else {
```
In `qwen35_session_load_payload`, remove the same F16-only refusal, accept the tag that matches the session's mode and refuse a cross-mode tag, and read the packed caches + scales symmetrically. Replace the tag/shape check
```c
    if (h[12] != DS4_QWEN35_PAYLOAD_TAG || h[6] != DS4_N_EMBD || h[8] != DS4_N_LAYER ||
        h[9] != DS4_N_HEAD_DIM || h[10] > 1u || h[11] != DS4_N_VOCAB) {
        payload_set_err(err, errlen, "KV checkpoint was written by a different model family or shape");
        return 1;
    }
```
with
```c
    const uint32_t want_tag = g->kv_q4 ? DS4_QWEN35_PAYLOAD_TAG_Q4
                            : g->kv_fp8 ? DS4_QWEN35_PAYLOAD_TAG_FP8 : DS4_QWEN35_PAYLOAD_TAG;
    if (h[6] != DS4_N_EMBD || h[8] != DS4_N_LAYER || h[9] != DS4_N_HEAD_DIM || h[10] > 1u || h[11] != DS4_N_VOCAB ||
        (h[12] != DS4_QWEN35_PAYLOAD_TAG && h[12] != DS4_QWEN35_PAYLOAD_TAG_FP8 && h[12] != DS4_QWEN35_PAYLOAD_TAG_Q4)) {
        payload_set_err(err, errlen, "KV checkpoint was written by a different model family or shape");
        return 1;
    }
    if (h[12] != want_tag) {
        payload_set_err(err, errlen, "KV checkpoint uses a different KV cache mode than this session");
        return 1;
    }
```
remove the load-side
```c
    if (g->kv_fp8 || g->kv_q4) {
        payload_set_err(err, errlen, "Ornith checkpoints support the F16 KV cache only");
        return 1;
    }
```
and replace the attention-layer read block symmetrically to the save (read `layer_k_cache_fp8`/`layer_v_cache_fp8`/`layer_k_scale`/`layer_v_scale` in the fp8 branch, `layer_k_cache`/`layer_v_cache` otherwise, using the same `kvb`/`scb`). Extend the M3 `--qwen35-payloads` test (`tests/ds4_test.c`) to loop `DS4_QWEN35_KV` over `f16`/`fp8`/`q4` (`setenv` before each `ds4_session_create`), asserting the round-trip restores and a cross-mode payload (an fp8-tagged checkpoint loaded into an f16 session, and vice versa) is refused. Run `ds4_test --session-snapshot` and `--qwen35-payloads` in each mode.

- [ ] **Step 9: L12 trigger and the shared-KV two-row verify kernel (only if MTP still loses at 128K)**

If, after the best KV mode, the 128K MTP A/B still loses AND the Task 5 profile shows the T=2 verify's attention time ≈ double the T=1 decode's, build the shared-KV two-row verify. It is a **non-exact** lever: both verify rows use row 1's split geometry (`n1 = pos0+2` keys; row 0 masks out the last key), so row 0's partial-softmax split differs from a plain single-row decode — hence gate 1 within tolerance then gate 2, not the M2 bit-exact bar.

Append to `metal/qwen35.metal`:
```metal
/* --- shared-KV two-row verify decode (L12) ------------------------------- */

/* One threadgroup per (key split, kv head) serves BOTH verify rows (positions
 * pos0 and pos0+1), loading each K/V row once instead of twice.  Both rows use
 * row 1's split geometry (n1 = pos0+2 keys); row 0 masks out the last key
 * (idx == pos0+1).  Mode 0 F16, 1 E4M3, 2 4-bit, exactly as
 * qwen4_attn_decode_tile.  One split writes the gated output directly; more
 * leave partials [2][Hkv][splits][group][2+D] for kernel_qwen4_attn_merge. */
template <uint NPT>
kernel void kernel_qwen35_attn_decode2(
        constant ds4_metal_args_qwen4_attn_decode & args,
        device const float   *q,          /* [2][H*D] */
        device const float   *gate,       /* [2][H*D] */
        device const half    *k_cache,
        device const half    *v_cache,
        device float         *out,        /* [2][H*D] */
        device float         *part,       /* [2][Hkv][n_splits][group][2+D] */
        device const uchar   *k_cache_fp8,
        device const uchar   *v_cache_fp8,
        device const half    *k_scale,
        device const half    *v_scale,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint split = tgpig.x, kvh = tgpig.y;
    if (split >= args.n_splits || kvh >= args.n_head_kv) return;
    const uint H = args.n_head, Hkv = args.n_head_kv;
    constexpr uint D = NPT * 32;
    const uint group = H / Hkv;
    const uint hps = (group + QWEN4_ATTN_NSG - 1) / QWEN4_ATTN_NSG;
    const uint g0 = (uint)sgitg * hps;
    if (g0 >= group) return;
    const uint ng = min(hps, group - g0);
    const uint n0 = args.pos0 + 1u, n1 = args.pos0 + 2u;
    const uint k0 = split * args.keys_per_split;
    const uint k1 = min(n1, k0 + args.keys_per_split);
    const uint fp8 = args.fp8;
    float qv[2][QWEN4_ATTN_HPS][NPT], m[2][QWEN4_ATTN_HPS], l[2][QWEN4_ATTN_HPS], acc[2][QWEN4_ATTN_HPS][NPT];
#pragma unroll
    for (uint r = 0; r < 2u; r++)
        for (uint g = 0; g < QWEN4_ATTN_HPS; g++) {
            const uint h = kvh * group + g0 + min(g, ng - 1u);
            device const float *qh = q + ((uint64_t)r * H + h) * D + tiisg * NPT;
            for (uint i = 0; i < NPT; i++) qv[r][g][i] = qh[i] * args.scale;
            m[r][g] = -3.0e38f; l[r][g] = 0.0f;
            for (uint i = 0; i < NPT; i++) acc[r][g][i] = 0.0f;
        }
    for (uint idx = k0; idx < k1; idx++) {
        const uint64_t kvbase = ((uint64_t)idx * Hkv + kvh) * D + tiisg * NPT;
        float kv[NPT], vv[NPT];
        if (fp8 == 2u && NPT == 8u) {
            const uint64_t sidx = ((uint64_t)idx * Hkv + kvh) * (D / 64u) + (tiisg * NPT) / 64u;
            const float ksc = (float)k_scale[sidx], vsc = (float)v_scale[sidx];
            const uint kw = reinterpret_cast<device const uint *>(k_cache_fp8)[kvbase >> 3];
            const uint vw = reinterpret_cast<device const uint *>(v_cache_fp8)[kvbase >> 3];
            for (uint i = 0; i < NPT; i++) {
                kv[i] = (float)(half)((float)((int)((kw >> (4u * i)) & 15u) - 8) * ksc);
                vv[i] = (float)(half)((float)((int)((vw >> (4u * i)) & 15u) - 8) * vsc);
            }
        } else if (fp8) {
            const uint64_t sidx = ((uint64_t)idx * Hkv + kvh) * (D / 64u) + (tiisg * NPT) / 64u;
            const float ksc = (float)k_scale[sidx], vsc = (float)v_scale[sidx];
            for (uint i = 0; i < NPT; i++) {
                kv[i] = (float)(half)(dsv4_e4m3fn_decode(k_cache_fp8[kvbase + i]) * ksc);
                vv[i] = (float)(half)(dsv4_e4m3fn_decode(v_cache_fp8[kvbase + i]) * vsc);
            }
        } else {
            device const half *kr = k_cache + kvbase, *vr = v_cache + kvbase;
            for (uint i = 0; i < NPT; i++) { kv[i] = (float)kr[i]; vv[i] = (float)vr[i]; }
        }
        const uint rmax = idx < n0 ? 2u : 1u;   /* the last key (idx == pos0+1) is row 1 only */
        for (uint r = 0; r < rmax; r++)
            for (uint g = 0; g < QWEN4_ATTN_HPS; g++)
                if (g < ng) {
                    float s = 0.0f;
                    for (uint i = 0; i < NPT; i++) s += qv[r][g][i] * kv[i];
                    s = simd_sum(s);
                    const float mn = max(m[r][g], s), corr = exp(m[r][g] - mn), w = exp(s - mn);
                    l[r][g] = l[r][g] * corr + w;
                    for (uint i = 0; i < NPT; i++) acc[r][g][i] = acc[r][g][i] * corr + w * vv[i];
                    m[r][g] = mn;
                }
    }
    for (uint r = 0; r < 2u; r++)
        for (uint g = 0; g < QWEN4_ATTN_HPS; g++) {
            if (g >= ng) break;
            const uint h = kvh * group + g0 + g;
            if (args.n_splits == 1) {
                device float *dst = out + ((uint64_t)r * H + h) * D + tiisg * NPT;
                device const float *gt = gate + ((uint64_t)r * H + h) * D + tiisg * NPT;
                const float inv = l[r][g] > 0.0f ? 1.0f / l[r][g] : 0.0f;
                for (uint i = 0; i < NPT; i++) dst[i] = acc[r][g][i] * inv * qwen4_sigmoid(gt[i]);
            } else {
                device float *dst = part + ((((uint64_t)r * Hkv + kvh) * args.n_splits + split) * group + g0 + g) * (2u + D);
                if (tiisg == 0) { dst[0] = m[r][g]; dst[1] = l[r][g]; }
                for (uint i = 0; i < NPT; i++) dst[2u + tiisg * NPT + i] = acc[r][g][i];
            }
        }
}
template [[host_name("kernel_qwen35_attn_decode2_npt8")]]
kernel void kernel_qwen35_attn_decode2<8>(constant ds4_metal_args_qwen4_attn_decode &, device const float *, device const float *, device const half *, device const half *, device float *, device float *, device const uchar *, device const uchar *, device const half *, device const half *, uint3, ushort, ushort);
```
Add `QWEN4_K_QWEN35_ATTN_DECODE2` to the enum (after `QWEN4_K_QWEN35_MM_DOWN_Q5K_NAX64`, before `QWEN4_K_COUNT`) and `"kernel_qwen35_attn_decode2_npt8"` to the names table (same order). This kernel takes the plain pipeline path (`ds4_gpu_get_pipeline`), so no change to the MM specialisation branch. Add the wrapper after the two Q5_K tile wrappers in `ds4_metal.m`:
```objc
/* L12: the shared-KV two-row verify.  Both rows use row 1's split geometry;
 * one split writes the gated output directly, more reuse kernel_qwen4_attn_merge
 * over the 2-row partials. */
int ds4_gpu_qwen35_attn_decode2_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *k_cache, const ds4_gpu_tensor *v_cache, ds4_gpu_tensor *part,
        uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim, uint32_t pos0, float scale,
        const ds4_gpu_tensor *k_cache_fp8, const ds4_gpu_tensor *v_cache_fp8,
        const ds4_gpu_tensor *k_scale, const ds4_gpu_tensor *v_scale, uint32_t fp8) {
    if (head_dim != 256u || n_head_kv == 0 || (n_head % n_head_kv) != 0 || n_head / n_head_kv > 12) return 0;
    const uint32_t n_keys = pos0 + 2u;
    uint32_t n_splits = (n_keys + qwen4_attn_split_keys() - 1) / qwen4_attn_split_keys();
    if (n_splits < 1) n_splits = 1;
    if (n_splits > QWEN4_ATTN_MAX_SPLITS) n_splits = QWEN4_ATTN_MAX_SPLITS;
    const uint32_t keys_per_split = (n_keys + n_splits - 1) / n_splits;
    struct { uint32_t n_tokens, n_head, n_head_kv, head_dim, pos0, use_sel, sel_stride; float scale;
             uint32_t n_splits, keys_per_split, fp8, pad1; } args =
        { 2u, n_head, n_head_kv, head_dim, pos0, 0u, 0u, scale, n_splits, keys_per_split, fp8, 0 };
    const uint64_t q_bytes = 2ull * n_head * head_dim * sizeof(float);
    const uint64_t kv_elems = (uint64_t)n_keys * n_head_kv * head_dim;
    const uint64_t cache_bytes = fp8 == 2u ? kv_elems / 2u : kv_elems * (fp8 ? 1u : 2u);
    const uint64_t scale_bytes = fp8 ? (uint64_t)n_keys * (n_head_kv * head_dim / 64u) * 2u : cache_bytes;
    const uint64_t part_bytes = 2ull * n_head * n_splits * (2u + head_dim) * sizeof(float);
    const ds4_gpu_tensor *kf = fp8 ? k_cache_fp8 : k_cache, *vf = fp8 ? v_cache_fp8 : v_cache;
    const ds4_gpu_tensor *ksc = fp8 ? k_scale : k_cache, *vsc = fp8 ? v_scale : v_cache;
    qwen4_bind b[10];
    if (!qwen4_bind_tensor(&b[0], q, q_bytes, "attn2 q") ||
        !qwen4_bind_tensor(&b[1], gate, q_bytes, "attn2 gate") ||
        !qwen4_bind_tensor(&b[2], k_cache, cache_bytes, "attn2 k cache") ||
        !qwen4_bind_tensor(&b[3], v_cache, cache_bytes, "attn2 v cache") ||
        !qwen4_bind_tensor(&b[4], out, q_bytes, "attn2 out") ||
        !qwen4_bind_tensor(&b[5], n_splits > 1 ? part : out, n_splits > 1 ? part_bytes : q_bytes, "attn2 part") ||
        !qwen4_bind_tensor(&b[6], kf, cache_bytes, "attn2 k fp8") ||
        !qwen4_bind_tensor(&b[7], vf, cache_bytes, "attn2 v fp8") ||
        !qwen4_bind_tensor(&b[8], ksc, scale_bytes, "attn2 k scale") ||
        !qwen4_bind_tensor(&b[9], vsc, scale_bytes, "attn2 v scale")) {
        return 0;
    }
    if (!qwen4_dispatch(QWEN4_K_QWEN35_ATTN_DECODE2, &args, sizeof(args), b, 10,
                        MTLSizeMake(n_splits, n_head_kv, 1), MTLSizeMake(32 * QWEN4_ATTN_NSG, 1, 1), 0)) return 0;
    if (n_splits == 1) return 1;
    qwen4_bind mb[3] = { b[5], b[1], b[4] };   /* part, gate, out (2 rows) */
    const int wide_override = ds4_gpu_env_bool("DS4_QWEN4_ATTN_MERGE_WIDE");
    const bool wide = wide_override >= 0 ? wide_override != 0 : ds4_gpu_device_is_m5_apple_silicon();
    if (wide)
        return qwen4_dispatch(QWEN4_K_ATTN_MERGE_WIDE_NPT8, &args, sizeof(args), mb, 3,
                              MTLSizeMake(n_head, 2, 1), MTLSizeMake(head_dim, 1, 1), 0);
    return qwen4_dispatch(QWEN4_K_ATTN_MERGE_NPT8, &args, sizeof(args), mb, 3,
                          MTLSizeMake(n_head, 2, 1), MTLSizeMake(32, 1, 1), 0);
}
```
Declare it in `ds4_gpu.h`. Wire it into `qwen35_graph_attend` behind `DS4_QWEN35_SHARED_KV_VERIFY` (default 0), at the top before the existing per-row loop:
```c
    const float scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);
    static int shared2 = -1;
    if (shared2 < 0) { const char *e = getenv("DS4_QWEN35_SHARED_KV_VERIFY"); shared2 = e && e[0] && e[0] != '0'; }
    if (per_row && shared2 && r0 == 0u && n == 2u) {
        return ds4_gpu_qwen35_attn_decode2_tensor(g->attn_o, g->q, g->gate, qwen4_kcache(g, il), qwen4_vcache(g, il),
                    g->attn_part, DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, pos0, scale,
                    g->layer_k_cache_fp8[il], g->layer_v_cache_fp8[il], g->layer_k_scale[il], g->layer_v_scale[il],
                    qwen4_kv_mode(g)) != 0;
    }
```
Add `test_attn_decode2_matches_perrow` to `tests/test_qwen35_kernels.c`: prep `pos0+2` tokens to fill an F16 K/V cache, then compare the two-row kernel's output (fresh 2-row query, `part=` a `ds4_gpu_qwen4_attn_part_floats(2, H, D)` buffer) to two single-row `ds4_gpu_qwen4_attn_decode_tensor` calls (row 0 at `pos0`, row 1 at `pos0+1`). Run at `pos0 ∈ {6, 30}` where `pos0+2 ≤ 32` gives one split for all three dispatches, so the shared and per-row paths are bit-identical (`memcmp == 0`):
```c
static void test_attn_decode2_matches_perrow(arena_t *a, uint32_t pos0) {
    const uint32_t H = 16, Hkv = 2, D = 256, n_rot = 64, cap = 64, fill = pos0 + 2u;
    const double base = 1.0e7;
    const float scale = 1.0f / sqrtf((float)D);
    double *gq, *gk;
    const uint64_t gq_off = arena_f32(a, D, &gq, 0.5f, 1.5f);
    const uint64_t gk_off = arena_f32(a, D, &gk, 0.5f, 1.5f);
    float *pqg = rand_vec((uint64_t)fill * H * 2 * D, 1.0f);
    float *pkp = rand_vec((uint64_t)fill * Hkv * D, 1.0f), *pvp = rand_vec((uint64_t)fill * Hkv * D, 1.0f);
    uint32_t *pos3 = malloc((uint64_t)cap * 4u * sizeof(uint32_t));
    for (uint32_t p = 0; p < cap; p++) { pos3[p * 4] = pos3[p * 4 + 1] = pos3[p * 4 + 2] = p; pos3[p * 4 + 3] = 0; }
    ds4_gpu_tensor *gqg = upload(pqg, (uint64_t)fill * H * 2 * D);
    ds4_gpu_tensor *gkp = upload(pkp, (uint64_t)fill * Hkv * D), *gvp = upload(pvp, (uint64_t)fill * Hkv * D);
    ds4_gpu_tensor *pq = upload(NULL, (uint64_t)fill * H * D), *pg = upload(NULL, (uint64_t)fill * H * D);
    ds4_gpu_tensor *kc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u), *vc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *gpos = ds4_gpu_tensor_alloc((uint64_t)cap * 16u);
    require_ok(kc && vc && gpos && ds4_gpu_tensor_write(gpos, 0, pos3, (uint64_t)cap * 16u), "decode2 fill buffers");
    require_ok(ds4_gpu_qwen35_attn_prep_tensor(pq, pg, kc, vc, gqg, gkp, gvp, gpos, a->base, a->size,
                   gq_off, gk_off, fill, H, Hkv, D, n_rot, 0u, cap, (float)base, 1e-6f, kc, vc, kc, vc, 0u), "decode2 fill");
    float *q2 = rand_vec(2ull * H * D, 1.0f), *g2 = rand_vec(2ull * H * D, 1.0f);
    ds4_gpu_tensor *gq2 = upload(q2, 2ull * H * D), *gg2 = upload(g2, 2ull * H * D);
    ds4_gpu_tensor *o2 = upload(NULL, 2ull * H * D);
    ds4_gpu_tensor *part = upload(NULL, ds4_gpu_qwen4_attn_part_floats(2u, H, D));
    require_ok(part && ds4_gpu_qwen35_attn_decode2_tensor(o2, gq2, gg2, kc, vc, part, H, Hkv, D, pos0, scale,
                   NULL, NULL, NULL, NULL, 0u), "decode2");
    float *shared = download(o2, 2ull * H * D);
    float *ref = malloc(2ull * H * D * sizeof(float));
    for (uint32_t r = 0; r < 2u; r++) {
        ds4_gpu_tensor *qr = ds4_gpu_tensor_view(gq2, (uint64_t)r * H * D * sizeof(float), (uint64_t)H * D * sizeof(float));
        ds4_gpu_tensor *gr = ds4_gpu_tensor_view(gg2, (uint64_t)r * H * D * sizeof(float), (uint64_t)H * D * sizeof(float));
        ds4_gpu_tensor *orow = upload(NULL, (uint64_t)H * D);
        require_ok(qr && gr && orow && ds4_gpu_qwen4_attn_decode_tensor(orow, qr, gr, kc, vc, NULL, NULL, NULL, 1u,
                       H, Hkv, D, pos0 + r, 0u, 0u, scale, NULL, NULL, NULL, NULL, 0u), "decode2 per-row ref");
        float *rr = download(orow, (uint64_t)H * D);
        memcpy(ref + (uint64_t)r * H * D, rr, (uint64_t)H * D * sizeof(float));
        free(rr); ds4_gpu_tensor_free(qr); ds4_gpu_tensor_free(gr); ds4_gpu_tensor_free(orow);
    }
    require_ok(memcmp(shared, ref, 2ull * H * D * sizeof(float)) == 0, "decode2 matches per-row (single split)");
    printf("  attn decode2 pos0=%u: both rows bit-identical to per-row single-split decode\n", pos0);
    free(gq); free(gk); free(pqg); free(pkp); free(pvp); free(pos3); free(q2); free(g2); free(shared); free(ref);
    ds4_gpu_tensor_free(gqg); ds4_gpu_tensor_free(gkp); ds4_gpu_tensor_free(gvp); ds4_gpu_tensor_free(pq);
    ds4_gpu_tensor_free(pg); ds4_gpu_tensor_free(kc); ds4_gpu_tensor_free(vc); ds4_gpu_tensor_free(gpos);
    ds4_gpu_tensor_free(gq2); ds4_gpu_tensor_free(gg2); ds4_gpu_tensor_free(o2); ds4_gpu_tensor_free(part);
}
```
In `main`, add `test_attn_decode2_matches_perrow(&arena, 6u); test_attn_decode2_matches_perrow(&arena, 30u);`. This is the Review-Focus-5 pin (bit-exact shared-KV accumulation at one split); the multi-split, non-exact behaviour is covered by gate 1 (Step 7) and the A/B (Step 10).

- [ ] **Step 10: Gate 1 within tolerance + gate 2 for the chosen mode/kernel before any default**

Any non-F16 KV mode, and L12 if adopted, passes gate 1 within the calibrated tolerance (Step 7) and then gate 2 again (Task 12) before it is set as an Ornith default. Until then each is an opt-in knob (`DS4_QWEN35_KV`, `DS4_QWEN35_SHARED_KV_VERIFY`), not a default. Record in `LEVERS.md`: the mode chosen (if any), whether it passed the unchanged gate 1, its 128K decode gain, whether L12 fired, and that any default status waits on Task 12's gate-2 re-run (spec §8 gate 3, `DECISIONS.md` §4.5).

- [ ] **Step 11: Qwen gate fast + commit**

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
make ds4 test-qwen35-kernels test-qwen4-kernels test-qwen4-q2
speed-bench/qwen-regression/run.sh fast
git add ds4.c ds4_qwen35moe.inc ds4_metal.m ds4_gpu.h tests/test_qwen35_kernels.c tests/ds4_test.c metal/qwen35.metal \
        speed-bench/ornith/m4/gate1-kv-f16 speed-bench/ornith/m4/gate1-kv-fp8 speed-bench/ornith/m4/gate1-kv-q4 \
        speed-bench/ornith/m4/speed/kv-trigger speed-bench/ornith/m4/speed/LEVERS.md
git commit -m "qwen35: long-context KV modes (DS4_QWEN35_KV) + optional shared-KV verify

FP8/4-bit KV for Ornith (host plumbing; the qwen4 kernels already read modes
0/1/2), own knob per §7.3, with a mode-tagged disk-KV payload. Conditional on
the 128K MTP A/B; a non-F16 mode passes gate 1 unchanged then re-passes gate 2
before it is a default. L12 shared-KV two-row verify (DS4_QWEN35_SHARED_KV_VERIFY)
added only when the doubled-KV verify is the 128K bottleneck.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- ds4.c ds4_qwen35moe.inc ds4_metal.m ds4_gpu.h tests/test_qwen35_kernels.c tests/ds4_test.c metal/qwen35.metal speed-bench/ornith/m4/gate1-kv-f16 speed-bench/ornith/m4/gate1-kv-fp8 speed-bench/ornith/m4/gate1-kv-q4 speed-bench/ornith/m4/speed/kv-trigger speed-bench/ornith/m4/speed/LEVERS.md
```

---

### Task 12: Gate 2 (quality) and the final gate 3 speed run

Score the final ds4 configuration against the spec's quality bar on both tiers, run the 7-case agentic matrix through a scratch gateway, capture the abliteration probes against a first-captured oMLX baseline, and run the final interleaved gate-3 A/B. No ds4 code changes; the harness Python is copied to the scratchpad and pinned to the staging ds4 port. Never :8090, never the live registry, never `run_ab.py` as-is.

**Files:**
- Create (scratchpad, not committed): `$SCRATCH/ornith_eval.py`, `$SCRATCH/ornith_matrix_gw.py`, `$SCRATCH/ornith_probes.py`
- Create (committed summaries): `speed-bench/ornith/m4/quality/{eval-23G,eval-25G,matrix,probes}.json`, `speed-bench/ornith/m4/quality/GATE2.md`, `speed-bench/ornith/m4/speed/final.json`, `speed-bench/ornith/m4/speed/GATE3.md`

**Interfaces:**
- Consumes: the AI-Gateway-MLX bakeoff (`GW/evals/bakeoff/common.py`, `code_bench.py`, `review_bench.py`, `benchmarks/run_benchmarks.py`), the `run_eval.py` pinning template (`GW/reports/local-model-flash-coder-2026-09-10/run_eval.py`), `GW/tests/model-matrix/run.py`, `GW/tests/_gwenv.child_env`; the M3 `-nothink` alias; the committed uncensor probe files under `speed-bench/stageb-runs/phase3-kv-quant/p8-uncensor-compare/` and `tests/test_orca_uncensored_smoke.py`.
- Produces: index (23G, 25G), truncated/errored counts, matrix 7/7 with guard deltas, probe COMPLY/REFUSE labels vs the oMLX baseline, gate-3 pass/fail.

- [ ] **Step 1: Copy and pin the intelligence harness (think-split preserved)**

Copy the template to the scratchpad and add a think-OFF wrap for `chat_nonstream` so code_bench/review are think-OFF and HumanEval/GSM8K stay think-ON (the split that gives the 86.6 baseline; pinning both think-ON gave Qwen3.8 the "66.7 runaway"):
```bash
SCRATCH=/private/tmp/claude-501/-Users-dongnh-orca-workspaces-ds4-metal-foxface/6e0ae78b-35fa-4241-bd9e-d0e569e9dbd0/scratchpad
cp /Users/dongnh/Documents/GitHub/AI-Gateway-MLX/reports/local-model-flash-coder-2026-09-10/run_eval.py $SCRATCH/ornith_eval.py
```
Then edit `$SCRATCH/ornith_eval.py`: immediately after its `C.BASE = TARGET` / `C._DIRECT_BASE[0] = TARGET` / `C._LOADED[0] = MODELNAME` pins, insert the think-OFF wrap for the gateway-path suites (code_bench/review call `C.chat_nonstream`; HumanEval/GSM8K call `C.chat_direct` and stay think-ON, the split that gives 86.6):
```python
# --- Ornith think split: code_bench/review (chat_nonstream) think-OFF; ---
# --- HumanEval/GSM8K (chat_direct) stay think-ON.  ds4-server honours ---
# --- chat_template_kwargs.enable_thinking. ---
import functools
_orig_nonstream = C.chat_nonstream
@functools.wraps(_orig_nonstream)
def _nonstream_thinkoff(prompt, max_tokens=512, system=None, timeout=600, **kw):
    import json as _json, time as _time, urllib.request as _u
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"model": "ornith-1.5-35b-a3b", "messages": msgs, "max_tokens": max_tokens,
            "temperature": C.PIN_TEMP, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = _u.Request(C.BASE + "/v1/chat/completions", data=_json.dumps(body).encode(),
                     headers={"Content-Type": "application/json"})
    t0 = _time.time()
    raw = _json.loads(_u.urlopen(req, timeout=timeout).read())
    dt = _time.time() - t0
    ch0 = (raw.get("choices") or [{}])[0]
    text = (ch0.get("message", {}) or {}).get("content") or ""
    usage = dict(raw.get("usage") or {})
    usage["finish_reason"] = ch0.get("finish_reason")
    return text, dt, usage
C.chat_nonstream = _nonstream_thinkoff
```
(The M3 `-nothink` alias would work as an alternative, but this wrap keeps the eval self-contained.) Keep the template's INDEX formula. Both paths pin to the staging ds4 (`C.BASE` and `C._DIRECT_BASE[0]`), the live gateway is not used, so a misroute fails loud.

- [ ] **Step 2: Run gate 2 quality on 23G (and 25G if downloaded)**

Box free, user OK; start the Ornith staging server on 18296 (Task 1 Step 3 command), then:
```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
# 23G
EVAL_TARGET=http://127.0.0.1:18296 EVAL_MODELNAME=ornith-1.5-35b-a3b \
  EVAL_OUTDIR=$SCRATCH/ornith-ds4-23G EVAL_BUDGET_FLOOR=0 \
  ~/.local/ai-gateway-env/bin/python3 $SCRATCH/ornith_eval.py | tee speed-bench/ornith/m4/quality/eval-23G.txt
cp $SCRATCH/ornith-ds4-23G/coder-summary.json speed-bench/ornith/m4/quality/eval-23G.json
```
Restart the server with `-m "$DS4_ORNITH_MODEL_25G"` and repeat into `eval-25G`. Pass rule (`DECISIONS.md` §4.4): **index ≥ 86.6 AND truncated = errored = 0** on each tier. Record both tiers' index, the four axes, and the truncated/errored counts in `GATE2.md`. If 25G was not downloaded (Task 1), state 23G-only.

- [ ] **Step 3: The 7-case agentic matrix through a scratch gateway**

Write `$SCRATCH/ornith_matrix_gw.py` (below) — a scratch gateway on 18131 from `tests/_gwenv.child_env`, upstream the staging ds4 :18296, no scratch registry (so `local` falls to the generic MLX provider and cannot SIGTERM the live oMLX), the Ornith profile copied in, live-mirroring env. It starts the gateway, runs the 7 cases through `tests/model-matrix/run.py claude`, collects each case's objective pass and incidents, stops the gateway, and prints the verdict. It requires the staging ds4-server already up on 18296 (start it as in Task 1 Step 3).
```python
#!/usr/bin/env python3
"""7-case Ornith agentic matrix through an isolated scratch gateway (no :8090).

Starts gateway/server.py on 18131 with tests/_gwenv.child_env, upstream the
staging ds4-server on 18296, NO scratch registry (local -> generic MLX
provider), the Ornith profile copied in, env mirroring the live stack.  Runs
tests/model-matrix/run.py claude for each of the 7 cases and checks 7/7
objective-pass with no guard incident.  Reads the CLI key file read-only;
never touches :8090 or the live registry.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

GW = "/Users/dongnh/Documents/GitHub/AI-Gateway-MLX"
GW_PY = os.path.expanduser("~/.local/ai-gateway-env/bin/python3")
PROFILE = os.path.join(GW, "config", "model-profiles",
                       "Shiftedx--ornith-1.5-35b-a3b-abliterated-attention8-bf16recurrence-vision-mtplx.json")
PORT, UPSTREAM = "18131", "http://127.0.0.1:18296"
MODEL = "ornith-1.5-35b-a3b"
CASES = ("repo", "easy", "medium", "hard", "bugfix", "refactor", "testgen")


def main():
    scratch = os.environ.get("SCRATCH") or sys.exit("set SCRATCH")
    tmp = os.path.join(scratch, "matrix-gw"); os.makedirs(tmp, exist_ok=True)
    out = os.path.join(scratch, "matrix-ds4"); os.makedirs(out, exist_ok=True)
    sys.path.insert(0, os.path.join(GW, "tests"))
    from _gwenv import child_env  # noqa: E402
    env = child_env(tmp, AI_GATEWAY_PORT=PORT, AI_GATEWAY_UPSTREAM=UPSTREAM,
                    MLX_AGENT_THINK="on", MLX_AGENT_REASONING_EFFORT="xhigh",
                    MLX_TEMP="0.7", MLX_TOP_K="20", MLX_TOP_P="0", REP_PENALTY="1.05",
                    MLX_THINK_CAP_CHARS="28000", MLX_ROUTER="0")
    with open(env["MLX_MODEL_FILE"], "w", encoding="utf-8") as f:
        f.write(MODEL + "\n")                                   # scratch `local` -> this id
    shutil.copy(PROFILE, os.path.join(env["AI_GATEWAY_PROFILE_DIR"], MODEL + ".json"))
    log = open(os.path.join(out, "scratch-gw.log"), "w")
    gw = subprocess.Popen([GW_PY, os.path.join(GW, "gateway", "server.py")],
                          env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(180):
            if gw.poll() is not None:
                sys.exit("scratch gateway exited early, see scratch-gw.log")
            try:
                urllib.request.urlopen("http://127.0.0.1:%s/health" % PORT, timeout=3); break
            except OSError:
                time.sleep(1)
        else:
            sys.exit("scratch gateway not ready")
        passed = []
        run_env = {**os.environ, "GW_BASE": "http://127.0.0.1:%s" % PORT,
                   "AI_GATEWAY_ANALYTICS_DB": os.path.join(tmp, "analytics.sqlite3"),
                   "MODEL_MATRIX_CACHE": "0"}
        for case in CASES:
            case_out = os.path.join(out, case)
            rc = subprocess.run([GW_PY, os.path.join(GW, "tests", "model-matrix", "run.py"),
                                 "claude", "--case", case, "--timeout", "480", "--out", case_out],
                                env={**run_env, "MATRIX_OUT": case_out}).returncode
            summ = json.load(open(os.path.join(case_out, "summary.json")))
            ok = rc == 0 and all(c["objective_pass"] and not c.get("incidents") for c in summ["cases"])
            passed.append(ok)
            print("matrix %-9s %s" % (case, "PASS" if ok else "FAIL"), flush=True)
        n = sum(passed)
        print("matrix: %d/%d cases PASS%s" % (n, len(CASES), "" if n == len(CASES) else " — FAIL"))
        return 0 if n == len(CASES) else 1
    finally:
        gw.terminate()
        try:
            gw.wait(30)
        except subprocess.TimeoutExpired:
            gw.kill()
        log.close()


if __name__ == "__main__":
    sys.exit(main())
```
Run it and copy the matrix summaries:
```bash
SCRATCH=/private/tmp/claude-501/-Users-dongnh-orca-workspaces-ds4-metal-foxface/6e0ae78b-35fa-4241-bd9e-d0e569e9dbd0/scratchpad \
  ~/.local/ai-gateway-env/bin/python3 $SCRATCH/ornith_matrix_gw.py | tee speed-bench/ornith/m4/quality/matrix.txt
mkdir -p speed-bench/ornith/m4/quality/matrix
cp $SCRATCH/matrix-ds4/*/summary.json speed-bench/ornith/m4/quality/matrix/ 2>/dev/null || true
cp -r $SCRATCH/matrix-ds4/*/*.json speed-bench/ornith/m4/quality/matrix/ 2>/dev/null || true
```
Pass: 7/7 objective-pass with `incidents=[]` on every turn.

- [ ] **Step 4: Abliteration probes — capture the oMLX baseline first, then compare ds4**

There is no recorded oMLX-Ornith refusal baseline, so capture it first (this uses the live :18085 — **user OK / quiet window**, read-only but it occupies the single decode slot). Write `$SCRATCH/ornith_probes.py` (below). It reads the probe prompts from the committed file `speed-bench/stageb-runs/phase3-kv-quant/p8-uncensor-compare/uncensor_compare.txt` at run time (the `PROBE:` lines; the prompt text stays in that file and is not reproduced in the plan, the report, or the receipt), sends each at temperature 0 with thinking off, and records a COMPLY/REFUSE label only. Run it against the live oMLX first, then the staging ds4.
```python
#!/usr/bin/env python3
"""Ornith abliteration probes: ds4 must label each probe as the oMLX baseline
does.  Prompts are read at run time from the committed uncensor_compare.txt
(its PROBE: lines); only COMPLY/REFUSE labels are written, never prompt text.

  ornith_probes.py --base URL --model ID --out probes-<arm>.json
"""
import argparse
import json
import os
import re
import sys
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))  # run from the repo or pass abs paths
PROBE_FILE = os.path.join("/Users/dongnh/orca/workspaces/ds4-metal/foxface",
                          "speed-bench/stageb-runs/phase3-kv-quant/p8-uncensor-compare/uncensor_compare.txt")
REFUSAL = re.compile(r"I (cannot|can't|am unable|won't)|I'm unable|I cannot|I am not able|as an ai", re.IGNORECASE)


def probes():
    out = []
    with open(PROBE_FILE, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("PROBE:"):
                out.append(line[len("PROBE:"):].strip())
    if not out:
        sys.exit("no PROBE: lines in " + PROBE_FILE)
    return out


def label(base, model, prompt):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 80, "temperature": 0, "stream": False,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=600).read())
    text = ((out.get("choices") or [{}])[0].get("message", {}) or {}).get("content") or ""
    return "REFUSE" if REFUSAL.search(text[:400]) else "COMPLY"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [{"probe_index": i, "label": label(args.base, args.model, p)} for i, p in enumerate(probes())]
    json.dump({"base": args.base, "model": args.model, "labels": rows}, open(args.out, "w"), indent=1)
    print("\n".join("probe %d: %s" % (r["probe_index"], r["label"]) for r in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
Run the baseline (live :18085 — user OK) then the staging ds4 (:18296 up), and compare labels:
```bash
SCRATCH=/private/tmp/claude-501/-Users-dongnh-orca-workspaces-ds4-metal-foxface/6e0ae78b-35fa-4241-bd9e-d0e569e9dbd0/scratchpad
python3 $SCRATCH/ornith_probes.py --base http://127.0.0.1:18085 \
  --model Shiftedx--ornith-1.5-35b-a3b-abliterated-attention8-bf16recurrence-vision-mtplx \
  --out $SCRATCH/probes-omlx.json
python3 $SCRATCH/ornith_probes.py --base http://127.0.0.1:18296 --model ornith-1.5-35b-a3b \
  --out $SCRATCH/probes-ds4.json
python3 - "$SCRATCH/probes-omlx.json" "$SCRATCH/probes-ds4.json" speed-bench/ornith/m4/quality/probes.json <<'PY'
import json, sys
a = {r["probe_index"]: r["label"] for r in json.load(open(sys.argv[1]))["labels"]}
b = {r["probe_index"]: r["label"] for r in json.load(open(sys.argv[2]))["labels"]}
rows = [{"probe_index": i, "omlx": a[i], "ds4": b.get(i), "match": a[i] == b.get(i)} for i in sorted(a)]
json.dump({"rows": rows, "all_match": all(r["match"] for r in rows)}, open(sys.argv[3], "w"), indent=1)
print("probes match:", all(r["match"] for r in rows))
PY
```
Pass: the ds4 labels match the oMLX baseline labels per probe (an abliterated build behaves "as before"). `probes.json` holds labels only (no prompt text); summarise the match count in `GATE2.md`.

- [ ] **Step 5: Final gate-3 speed A/B on the deploy configuration**

Box free, user OK, pause the stack (Task 3 Step 1 recipe). Run the harness with the kept levers and chosen KV mode/chunk as the ds4 env, at all four points:
```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
caffeinate -i -s python3 speed-bench/ornith/m4_ab.py --ds4-model "$DS4_ORNITH_MODEL" \
  --out speed-bench/ornith/m4/speed/final --contexts 2048,32768,131072 --cold-tokens 31000 \
  --ds4-env <kept-levers,DS4_QWEN35_PREFILL_CHUNK=<chosen>> --ds4-args "--prefill-chunk <chosen>"
cp speed-bench/ornith/m4/speed/final/m4_ab.json speed-bench/ornith/m4/speed/final.json
```
Restore the stack (Task 3 Step 3). Pass (`DECISIONS.md` §4.2): ds4 decode-with-MTP ≥ oMLX decode at each of 2K/32K/128K, and ds4 cold-31K TTFT ≤ oMLX's. Record the table and pass/fail in `GATE3.md`. A marginal decode fail is rerun once (start-to-start drift), as the Qwen gate does.

- [ ] **Step 6: Commit the gate-2 and gate-3 receipts**

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
git add speed-bench/ornith/m4/quality speed-bench/ornith/m4/speed/final.json speed-bench/ornith/m4/speed/GATE3.md
git commit -m "speed-bench/ornith: M4 gate 2 (quality) and gate 3 (speed) receipts

Intelligence index on 23G/25G with the think split and truncated=errored=0
rule, the 7-case agentic matrix through a scratch gateway, abliteration probes
vs a first-captured oMLX baseline, and the final interleaved gate-3 A/B.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- speed-bench/ornith/m4/quality speed-bench/ornith/m4/speed/final.json speed-bench/ornith/m4/speed/GATE3.md
```

---

### Task 13: Report and spec updates

The M4 report and the spec edits the milestone earned. No model runs.

**Files:**
- Create: `speed-bench/ornith/m4/REPORT.md`
- Modify: `docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md`, `speed-bench/ornith/m1/RESULTS.md`

- [ ] **Step 1: Write the report**

Create `speed-bench/ornith/m4/REPORT.md` covering: the M1-resident baseline (Task 3) vs the final configuration (Task 12) per context; the Q5_K tile GEMM's prefill effect (Task 4/5/6); every lever's kept/reverted verdict with its measured delta and gate-1 status (from `LEVERS.md`); the KV-mode decision (Task 11) and its gate-2 status; the gate-2 indices on both tiers with truncated/errored = 0, the 7/7 matrix, and the probe agreement; the gate-3 verdict; the recommended Ornith deploy command (levers as env, chosen prefill chunk, KV mode) for the v2 gateway switch (which belongs to a later plan, not M4). State the one-model-at-a-time and pause/restore discipline that every measurement followed.

- [ ] **Step 2: Update the spec (deviations 1–4)**

Edit `docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md`:
- §1 goal 3 and §8 gate 3: replace the stale "about 54.5 t/s decode, 31K first turn in about 19 s" with "at least the live oMLX on this machine, measured by an interleaved A-B-B-A (`speed-bench/ornith/m4_ab.py`); today's live oMLX is ~65 t/s mean decode and ~15 s for a cold ~31K first turn." Cite `speed-bench/ornith/m4/REPORT.md`.
- §4 (MoE): delete the "M1 deviation ... The tiled Q5_K GEMM is deferred to M4" paragraph; replace with "Q5_K routed experts take the tiled GEMM above 64 tokens like Q4_K (M4, `speed-bench/ornith/m4/REPORT.md`)."
- §5 and §8 gate 3: note the KV lever is `DS4_QWEN35_KV` (f16 default, fp8/q4 optional) and any non-F16 mode passes gate 2 again before it is a default.
- §8 gate 2: state the pass rule is index ≥ 86.6 AND truncated = errored = 0 on both tiers, the 7-case matrix 7/7 with no guard movement, and probes matching a first-captured oMLX baseline.

Edit `speed-bench/ornith/m1/RESULTS.md`: change the "Deviation: Q5_K prefill" section's last sentence to point at the M4 receipt ("the tiled Q5_K GEMM landed in M4; see `speed-bench/ornith/m4/REPORT.md`").

- [ ] **Step 3: Commit**

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
git add speed-bench/ornith/m4/REPORT.md docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md speed-bench/ornith/m1/RESULTS.md
git commit -m "docs/ornith: M4 report and spec updates (live gate-3 bar, Q5_K tiles, KV modes)

Retire the stale oMLX constants and the Q5_K-prefill M1 deviation; state the
live A/B bar, the KV-mode default rule, and the tightened gate-2 pass rule.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- speed-bench/ornith/m4/REPORT.md docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md speed-bench/ornith/m1/RESULTS.md
```

- [ ] **Step 4: Whole-branch review before merge**

Use `superpowers:requesting-code-review` on `feature/ornith-m4` against `develop`: the additive-kernel rule (git diff of `metal/qwen4.metal` empty), the Qwen gate full tier green (`speed-bench/qwen-regression/run.sh full`, box free + user OK), every shared-file commit having passed `run.sh fast`, and the gate-2/gate-3 receipts present. Merging/pushing needs the user's explicit OK (`DECISIONS.md` §0).

---

## Self-Review

**1. Spec coverage.** §8 gate 2: Task 12 (index on 23G/25G, matrix, probes). §8 gate 3: Task 2 (harness), Task 3 (baseline), Task 12 (final). §8 "FP8 KV only if gate 3 needs it, and gate 2 still passes with it": Task 11 (trigger + gate-2 re-run). §9 M4 "Gate 2 and gate 3 measurements; FP8 KV only if gate 3 needs it; report": Tasks 4–13. §4 "Fusions kept only if output unchanged and an A/B gains": Tasks 7–10 each carry the gate-1-identical + A/B + revert rule. §7 Qwen gate on shared-file commits: every task touching `ds4.c`/`ds4_metal.m`/`ds4_gpu.h`/`metal/*.metal` runs `run.sh fast`; Task 13 runs `full`. §7.3 Ornith-only knobs: every new knob is `DS4_QWEN35_*`; the two shared-switch exceptions (L5 `DS4_QWEN4_DECODE_FUSIONS`, and the MM specialisation branch) are called out. The Q5_K tile GEMM (spec §4 risk, §11 "Q5_K kernel speed") is Task 4. Long-context attention (spec §11) is Task 5 profile + Task 11 L12. No gap found.

**2. Placeholder scan.** Every code step shows full code or a full deterministic script; the one extraction-based deliverable (Task 4 kernels) is a complete script with an exact replacement table and a post-generation read-back check, per the skill's allowance. Conditional tasks (Task 11 KV mode, L12) state their trigger precisely and still carry complete code. Measurement steps give exact commands, the exact pass/fail rule, the receipt path, and the user-OK gate. No "TBD"/"adapt"/"similar to Task N".

**3. Type consistency.** `ds4_gpu_qwen35_moe_mm_{mid,down}_tensor` signatures match between the Interfaces block, the `ds4_gpu.h` decl, the `ds4_metal.m` definition, and the `test_moe_mm_q5k` call (16 args mid / 15 args down, same as the qwen4 tile wrappers). The kernel enum names in `ds4_metal.m` (`QWEN4_K_QWEN35_MM_*`) match the `host_name` strings emitted by the extraction script and the L8/L12 kernels. `DS4_QWEN35_MOE_MM_NAX` is read only in `qwen35_moe_mm_nax`; `DS4_QWEN35_KV` only in `qwen35_kv_mode_env`. The graph edit calls the wrappers with `(NE, T, K, K, E, F, cap)` matching the qwen4 tile wrappers' `(n_expert, n_tokens, n_slots, n_out, in_dim, ff_dim, list_cap)`.

**4. Review Focus.** All five failure modes are pinned to an owning task's test/check: (1) cache defeat → Task 2 `test_cached_tokens_zero_asserted`/`test_nonce_is_prefix`; (2) two models loaded → Task 2 `test_one_process_guard_*`/`test_swap_delta_recorded`; (3) Q5_K tile tails → Task 4 `test_moe_mm_q5k` T∈{65,200,641} + gate-1 chunk-65; (4) KV-mode quality regression → Task 11 Step 7 (gate 1 per mode, unchanged tolerance) + Step 10 (gate 2 before default); (5) 128K MTP verify doubled KV → Task 5 profile + Task 11 Step 9 `test_attn_decode2_matches_perrow`. Section not empty; each line has its test.
