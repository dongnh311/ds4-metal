# Orca Uncensored → DS4 IQ2 (M5 Pro 64 GB) — Design

## Target
GGUF `qwen4exp` của Qwen3.8-Flash-Next **uncensored (abliterated)** chạy trên
ds4-metal Metal, M5 Pro 64 GB: IQ2_XXS gate/up (reuse Ivan, Orca giữ nguyên phần
này so với Qwen base) + Q2_K Pad768 down experts **quant từ Orca BF16** +
98 dense tensors abliterated **re-encode từ Orca BF16** + MTP unchanged +
PLE Q4_1 sidecar rebuild từ Orca. RAM resident ~41.7 GiB; PLE SSD-mapped
(không chiếm RAM). Không đụng code C runtime / Metal / CUDA / distributed —
chỉ tools Python trong `gguf-tools/` + build GGUF + chạy test.

## Sources (pinned)
- Orca: `orcarouter/Qwen3.8-Flash-Next-Uncensored` @ `8336e613ea508b13c2159bd0f68965d97a606b95`
  (1658 tensors, 179,999,981,459 params; 149 tensor changed — abliteration
  `W' = W − r(rᵀW)` trên 149 residual-writing matrices: 49 expert-down fused,
  49 shared-down, 36 GDN out_proj, 13 o_proj (incl. MTP), ple.value_proj,
  embed_tokens. Gate/up readers, router, mixers, indexer, vision, n-grams:
  byte-identical Qwen base.)
- Imatrix: `unsloth/Qwen3.8-Flash-Next-GGUF` @ `c8b5954…`
  (sha256 `a5863123…`, pinned trong `qwen4_pack.py`).
- Template: Ivan `Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf`
  (44.81 GB; 1.255 tensors, manifest kèm bên cạnh).

## tok/s levers (không sửa runtime)
1. Expert Q2_K down (Ivan recipe): −5.16 GiB RAM, speed ~không đổi, NLL +4.4%.
2. `--mtp` opportunistic (KHÔNG dùng `--mtp-exact-sampling` — bug open
   discussion #5: rewind reset recurrent graph → full replay, stall tới 457s).
3. Context budgeting: đo max ctx không swap trên M5 Pro (M1 Max fail 8K,
   plan 54.17 GiB — M5 Pro 64 GB cần verify thực tế).
4. Runtime kernel/MTP-depth tuning: **ngoài scope** (AGENT.md: không đụng
   Metal/CUDA/distributed paths; cần benchmark infra riêng).

## Discussion status (checked 2026-09-15)
- #6 (SIGSEGV image cache 4096→2560): **đã fix** trên branch halfbeak
  (`ds4_server.c:10185` dùng `ds4_engine_embd_dim`).
- #5 (MTP exact-sampling rewind): open; tránh flag này.
- #1 (M1 Max: 22-24 t/s @4K, swap 14.6/15.4 GiB, 8K fail): động lực giảm RAM.
- #2 (M4 Max 64GB: 38.7-38.9 t/s @32-64K, zero swap-in): reference.

## Disk budget (M5 Pro, peak)
Orca 336 + template 44.8 + PLE out 32 + main out 45 ≈ 458 GB peak trước khi
xóa Orca; yêu cầu ≥ ~470 GB free từ đầu (hoặc chạy task theo thứ tự và xóa
dần như plan; plan hiện tại giữ Orca đến hết Task 4).

## Status (2026-09-16) — Plan 1 XONG
- Artifact cuối: `gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-MTP-NNgram.gguf`
  (sha head `ed238d8d…` — n-grams self-contained, coherent, verified).
- Fix ssm_out head-block permutation garble: commit `9edb69d`.
- Benchmark: ds4-eval core 11/12 (INCOMPLETE duy nhất: aime2025-02 — budget
  3072-token, không phải lỗi); smoke test PASS; MTP prefill/gen ≈ 106/36.6
  t/s @4K, ≈109/36.5 @32K.
- Max-context no-swap: resident 41.72 GiB; KV 12.50 GiB @384K → 55.83 GiB
  (còn ~8 GiB headroom trên 64 GB). Trần native 262144 token; vượt trần cần
  `DS4_QWEN4_YARN_FACTOR`.
- ⚠️ Edge-case: 1 biến thể harmful probe phát "I cannot provide instructions…"
  rồi chuyển sang viết an toàn — uncensor không tuyệt đối; báo tác giả
  OrcaRouter card nếu quan trọng.
