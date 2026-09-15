# Orca Uncensored → DS4 IQ2 GGUF (M5 Pro 64 GB) — build recipe

Builds `Qwen3.8-Flash-Next` Orca Uncensored (abliterated) into the DS4 Metal
runtime's `qwen4exp` GGUF: IQ2_XXS gate/up + padded Q2_K down experts,
abliterated dense tensors re-encoded, MTP + PLE sidecar. Target machine:
Apple Silicon Mac, 64 GB RAM.

Plan of record: `docs/2026-09-15-orca-uncensored-ds4-iq2.md`
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
- [x] Task 1 (tool): `gguf-tools/orca_diff.py` — selective payload-hash
      diff, resumable per shard.
- [ ] Task 1 (data): Orca + Qwen-base checkpoint download (needs the
      gated-repo approval; card is open, `model.safetensors.index.json`
      is not) — run `python3 gguf-tools/orca_diff.py download --repo orca`
      then the `qwen` / `orca` / `diff` stages.
- [ ] Task 2: quantize 48 Orca down experts (`qwen4_iq2.py quantize
      --projection down --source gguf/orca-bf16`).
- [ ] Task 3: `orca_dense_replace.py` — re-encode the 147 in-template
      changed dense tensors + assemble the final main GGUF from the
      template.
- [ ] Task 4: rebuild PLE Q4_1 sidecar from Orca (`qwen4_pack.py`).
- [ ] Task 5: target validation (`make`, no-swap load @ 8192 ctx,
      `ds4-eval` core, uncensor smoke, `--mtp` tok/s at 4K/32K).
