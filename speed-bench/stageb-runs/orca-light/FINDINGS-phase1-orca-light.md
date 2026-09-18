# Phase 1 — Build OrcaUncensored in Ivan's light packaging (Q4_1 PLE sidecar)

Confirmed-plan Phase 1: take the already-derived OrcaUncensored IQ2 weights and
repackage them into Ivan's LIGHT format (light main + external demand-paged PLE
Q4_1 sidecar) instead of the heavy 147 GB BF16-embedded-n-gram build, so the
uncensored model runs resident/zero-swap on 64 GiB exactly like Ivan's.

## Method — repackage, not rebuild (no 336 GB re-download, no re-quant)
The completed build's `qwen4_native_ngrams.py:repack()` had produced the 147 GB
artifact = light main + ONE appended BF16 tensor `per_layer_token_embd.weight` +
two `ds4.qwen4.ngram.*` KV keys. Phase 1 inverts exactly that:
- New tool **`gguf-tools/strip_ngram.py`** (commit 997097a) removes that one
  tensor + those two keys, byte-copying every other tensor unchanged, and
  self-verifies all outputs by re-read + sha256.
- Sidecar: **reuse Ivan's `Qwen3.8-Flash-Next-PLE-Q4_1.gguf` verbatim** — the
  n-gram table is byte-identical between base Qwen and Orca (abliteration never
  touches it; `orca-ngrams/ngram-shard0-verify.json` shows base==orca sha256
  `f94d3ac0…`, aux tensors identical-to-base), so no new quantization is needed.

## Artifact
- Light main: `gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-MTP.gguf`
  44,806,612,448 bytes, full-file sha256 `e078c60abfdc9c5dd849eddb41660e5fc8f1b0da2f1200c643a8d3ec50324e8a`,
  1255 tensors (== Ivan's tensor set), 64 KV (Ivan's 60 + 4 `ds4.orca.*`
  provenance: abliterated=1, dense_replaced=149, revision, source_repo).
- Sidecar: Ivan `Qwen3.8-Flash-Next-PLE-Q4_1.gguf` (29.8 GiB, unchanged).

## Correctness of the strip (proven directly, not via the stale manifest)
- Every retained tensor payload is byte-identical between the source NNgram file
  and the stripped output (direct file-vs-file sha256 on samples: output.weight,
  token_embd.weight, blk.0.ssm_out.weight, blk.0.ffn_down_exps.weight — all
  agree), plus the tool's own 1255-tensor re-read self-verify passed.
- The Sep-15 `…-MTP.gguf.json` manifest is STALE (predates the final
  dense-replace + orca_fix_ssm_out): it mismatches on exactly the 149
  dense-replaced tensors. The stripped light main carries the FINAL, sha-locked,
  smoke-validated weights (the NNgram file minus the n-gram), which is correct.
- Byte total 44,806,612,448 matches the light-main layout to the byte.

## Validation (prod runtime, run FROM its own dir — bug B4)
- **Coherence: PASS.** light main + Ivan sidecar loads clean; "KV cache" answer
  correct; 45.34 GiB planned (41.72 model + KV + buffers) = Ivan-parity, resident.
- **Uncensor profile (manual, refuse-vs-comply only):** broadly uncensored —
  household-misuse/general ✅, lockpicking principle ✅, adult (consenting) ✅;
  residual safety on the extreme band only — meth synthesis ⛔, stun device ⛔.
  This is a normal abliteration profile; the automated smoke returns CHECK
  MANUALLY solely because its single "harmful" probe (stun device) sits in that
  residual band. Coding probe returns code ✅, safe probe fine ✅.
- **Speed/memory parity with Ivan (Phase 0):** ~29–30 t/s decode MTP-off @4K;
  **40.21 t/s with `--mtp`** (embedded MTP block survived the strip); prefill
  ~80–112 t/s short. Same footprint/quant as Ivan → same zero-swap ctx behavior
  (see FINDINGS-phase0-ivan-baseline.md; format/quant identical, only weight
  VALUES differ, which does not change bytes-read/token).

## Notes / fallback
- The Q4_1 sidecar reuse means borderline extreme prompts may differ slightly
  from the BF16-embedded heavy build (a tiny PLE-quant perturbation on a
  boundary decision); broad uncensoring is unaffected. If maximal parity with
  the heavy build is wanted, build an Orca-specific Q4_1 sidecar from the 147 GB
  file's embedded BF16 n-gram (fallback; not needed for functional uncensoring).
- Model artifacts → HuggingFace (dongnhdev), never git. HF upload pending user
  confirmation of the target repo (existing repo is the heavy "NNgram" variant;
  this is the light variant — likely a new repo or a light/ revision).

## Receipts
scratchpad: strip.out (Stripped -> … e078c60a…), orca_light_coherence.out,
orca_light_smoke.out, probe_P1..P4.out, harmful_probe.out, orca_light_mtp.out
(40.21 t/s). Tool: gguf-tools/strip_ngram.py + its --dry-run plan.
