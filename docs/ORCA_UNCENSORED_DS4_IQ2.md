# Orca Uncensored → DS4 IQ2 GGUF (M5 Pro 64 GB) — build recipe

Builds `Qwen3.8-Flash-Next` Orca Uncensored (abliterated) into the DS4 Metal
runtime's `qwen4exp` GGUF: IQ2_XXS gate/up + padded Q2_K down experts,
abliterated dense tensors re-encoded, MTP + PLE sidecar. Target machine:
Apple Silicon Mac, 64 GB RAM.

Plan of record: `docs/Task/2026-09-15-orca-uncensored-ds4-iq2.md`
(inside the `ds4-metal` main checkout, not this worktree).

## Pinned constants

| Input | Value |
| --- | --- |
| Orca source repo | `orcarouter/Qwen3.8-Flash-Next-Uncensored` (gated; approval required) |
| Orca revision | `8336e613ea508b13c2159bd0f68965d97a606b95` (131 shards, ~336 GB, 1658 tensors, 179,999,981,459 params) |
| Qwen base revision | `de4b8e4d43b917e7706784d8bb445c9af86a3540` (`Qwen/Qwen3.8-Flash-Next`) |
| Imatrix | `unsloth/Qwen3.8-Flash-Next-GGUF` rev `c8b5954a88c2775c546b92593eda40ea041d3176`, file `imatrix_unsloth.gguf_file` |
| Imatrix sha256 | `a5863123db1ca458727e738955bef7bfc199520aa2bee3a30142a1aff9254154` |
| Ivan IQ2 template | `Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf` (44.81 GB / 41.73 GiB) from `ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2` |
| Template file sha256 | `341c8d79468384a05e22998ae834a489145f6c385e276dcf79170a6b0d0ffd2d` (verified on-disk copy; its manifest `replaced[]` = the 48 `blk.*.ffn_down_exps.weight`) |

## Abliteration scope (Orca card, rev-pinned)

149 residual-writing tensors changed: 49 fused `mlp.experts.down_proj`, 49
`mlp.shared_expert.down_proj`, 36 GDN `linear_attn.out_proj`, 13
`self_attn.o_proj` (incl. MTP), plus `ple.value_proj` and `embed_tokens`.
Byte-identical to the base (must stay so in our output): gate/up readers,
router, Hyper-Connection mixers, indexer, vision tower, n-gram table.

## Target machine (build + validation)

| Field | Value |
| --- | --- |
| Machine | MacBook Pro (Mac17,8, Z1N00009TSA/A) |
| Chip | Apple M5 Pro (18 cores: 6 Super + 12 Performance) |
| RAM | 64 GB |
| SSD | internal APFS, 926 GiB volume (≥ 454 GiB free at build start) |
| macOS | 26.6.2 (build 25G83) |
| Python for tooling | `~/.local/omlx-venv/bin/python3` (hf_hub 1.24, numpy 2.3.5, safetensors 0.8.0; `hf` CLI there) |

## Constraints

- No C runtime / Metal / CUDA / distributed changes (AGENT.md); all new
  code lives in `gguf-tools/`.
- Runtime needs the discussion #6 vision-dim fix (`ds4_engine_embd_dim` in
  `ds4_server.c`) — present in this worktree (verified at line 10185).
- Never use `--mtp-exact-sampling` on this machine (known stall bug);
  use default opportunistic `--mtp`.
- PLE stays Q4_1 (160-wide dim only admits 32-block qtypes).

## Progress

- [x] Task 0: build `gguf-tools/libds4quants.dylib`; download pinned
      imatrix (sha256 verified); confirm template source already on disk.
- [x] Task 1: `gguf-tools/orca_diff.py` — selective payload-hash diff,
      resumable per shard. Base gate/up readers range-hashed from the hub
      (no base download); Orca checkpoint downloaded selectively (131
      shards). Diff report `gguf/orca-diff.json`: 7/7 readers identical,
      149 changed-set payloads recorded as Orca provenance.
- [x] Task 2: 48 Orca down experts quantized to Q2_K padded 768
      (`gguf/experts-down-orca/` + `experts.json` manifest).
- [x] Task 3: `gguf-tools/orca_dense_replace.py` — 149 in-template
      changed tensors re-encoded (48 Q2_K down from Task 2, 1 MTP MXFP4
      down, 49 shared-down + 36 GDN out + 13 o_proj + PLE value_proj
      as Q8_0, embed_tokens BF16 copy) and the final main GGUF assembled
      from the verified template with read-back verification of every
      payload: `gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-MTP.gguf`
      (44.81 GB, 149 replaced).
- [x] Task 4: embed the original BF16 n-grams into the main GGUF with
      `gguf-tools/qwen4_native_ngrams.py` (self-contained, matching this
      branch's "self-contained Qwen BF16 n-gram releases" design — no
      `--ple` sidecar on this branch). N-gram provenance, all verified:
      - PLE aux (`layer_multipliers`, `ngram_heads_offsets`,
        `ngram_heads_vocab_sizes`) byte-identical to the pinned base
        (hub range reads vs Orca shards, `gguf/orca-ngrams/ple-aux-verify.json`).
      - n-gram table spot check: `ngram_embedding.shard_0.weight`
        (800003840 B) sha256-identical base vs Orca
        (`gguf/orca-ngrams/ngram-shard0-verify.json`); the card claims
        the whole table unchanged.
      `ple.value_proj` (the one PLE tensor abliterated) was replaced in
      Task 3's main GGUF as `blk.1.ple_value.weight` (Q8_0).
      Final artifact: `Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-MTP-NNgram.gguf`
      (147.21 GB, 1256 tensors, sha256 `5c64c3ff247340a5...`).
- [x] Task 5 (correctness verification): byte-exact payload audit
      (`gguf-tools/orca_verify_q8.py` + one-shot checks):
      - Q8_0 payloads (shared downs, o_projs, GDN out_projs, PLE
        value_proj) re-encoded from Orca and byte-identical to a fresh
        `GGMLQuantizer` encode of the Orca BF16 source.
      - MTP expert down (MXFP4) byte-identical to the fresh
        `quant_mxfp4_rows` of the Orca source.
      - `token_embd.weight` BF16 copy and all 3D expert geometry
        (Q2_K `[768,2560,512]`, MXFP4 `[640,2560,512]`, IQ2_XXS
        gate/up `[2560,640,512]`) verified against the template.
      - All 1106 byte-copied payloads match the Ivan template manifest
        hashes; the 149 replaced slots keep the template's kind/size.
      `make test-qwen4-kernels` passes.

  **Garble root-cause (found via payload bisection):** the 36 GDN
  `ssm_out` payloads (GDN `linear_attn.out_proj`) were encoded as plain
  Q8_0 over the wrong axis with no head-block permutation. The qwen4exp
  schema stores `ssm_out` with the 48 v-head blocks of 128 input features
  permuted by the tiled-v order (`prod_out_colperm` in the official
  qwen4exp packer); a 128-wide group is four Q8_0 blocks so the
  permutation commutes with the encoding. All other 113 replaced
  families (48 trunk Q2_K, MTP MXFP4, 49 shared-down, 13 o_proj, PLE
  value_proj, embed_tokens) were correct and generate coherently when
  paired with base payloads; only `ssm_out` produced garble. Fixed with
  `gguf-tools/orca_fix_ssm_out.py` (re-fetch 36 Orca BF16 payloads,
  verify each against the `orca_diff` provenance sha, re-encode with the
  tiled-v permutation, dequant rel-L2 ≈ 0.006, write in place). After the
  fix the whole build is coherent.
- [x] Task 5 (benchmarks + smoke):
      - Greedy (`--temp 0`) + default sampling: coherent prose, coding and
        refusal/uncensor probes on real tasks (energy-drink pitch,
        iterative Fibonacci, factual "fake ID" question — no hard
        refusal, as the abliterated card intends).
      - `--mtp` (opportunistic, no `--mtp-exact-sampling`): 4K context
        prefill ≈ 106 t/s, generation ≈ 36.6 t/s; 32K context prefill ≈
        109 t/s, generation ≈ 36.5 t/s; KV at 32K ≈ 1.97 GiB context
        (model resident 41.72 GiB, total planned 43.69 GiB — fits the
        64 GB M5 Pro with no swap).
      - `gguf-tools/gguf_payload_diff.py` bisection confirms only the
        36 `ssm_out` payloads differ from the base control after the fix.
      - `ds4-eval --suite core` (12 selected questions, 1536-token budget,
        `--retry-incomplete`): **11/12 passed**. All SuperGPQA and
        GPQA-Diamond choices correct; AIME2025: `aime2025-01`=70,
        `aime2025-16`=468, `aime2025-03`=16 correct; the one INCOMPLETE
        is `aime2025-02` (long combinatorics proof that ran past the
        3072-token retry budget — a budget limit, not a garble/regression).
      - `tests/test_orca_uncensored_smoke.py` (official harness): **PASS**
        for the harness's two criteria — the harmful probe returned no
        refusal phrasing (uncensored) and the coding probe returned
        runnable code. Noted edge: one harmful probe variant
        *did* emit a safety-style "I cannot provide instructions…" refusal
        before pivoting to a safety write-up, so the "always comply"
        property is not absolute — flag to OrcaRouter card authors if it
        matters. `safe` probe coherent.
      - Max-context no-swap headroom: model resident 41.72 GiB; KV
        12.50 GiB at 384K context → 55.83 GiB planned (fits 64 GB with
        ~8 GiB headroom). Native ceiling 262144 tokens; past that
        requires `DS4_QWEN4_YARN_FACTOR`.
