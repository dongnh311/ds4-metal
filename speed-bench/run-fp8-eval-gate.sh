#!/bin/sh
# FP8 KV quality gate (spec §25 ground rule 6): ds4-eval core subset, BF16 baseline
# (resident; == paged-BF16 bit-identical per gate #KV) vs FP8 (paged + fp8sim).
# 12-case subset matches the documented 11/12 baseline. Usage: run-fp8-eval-gate.sh [N]
set -u
N=${1:-12}
M=gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-MTP-NNgram.gguf
SP="$(dirname "$0")"
echo "===== BASELINE BF16 (resident), $N cases $(date -u +%FT%TZ) =====" >&2
./ds4-eval -m "$M" --suite core --questions "$N" --retry-incomplete > "$SP/evalgate_bf16.out" 2>&1
echo "BF16_DONE rc=$?" >&2
grep -aE "passed, runtime|^ *[0-9]+ (PASSED|FAILED|INCOMPLETE)" "$SP/evalgate_bf16.out" | tail -20 >&2
echo "===== FP8 E4M3 (paged + fp8sim), $N cases $(date -u +%FT%TZ) =====" >&2
DS4_QWEN4_KV_PAGED=1 DS4_QWEN4_KV_FP8SIM=1 ./ds4-eval -m "$M" --suite core --questions "$N" --retry-incomplete > "$SP/evalgate_fp8.out" 2>&1
echo "FP8_DONE rc=$?" >&2
grep -aE "passed, runtime|^ *[0-9]+ (PASSED|FAILED|INCOMPLETE)" "$SP/evalgate_fp8.out" | tail -20 >&2
echo "===== GATE SUMMARY =====" >&2
echo "BF16: $(grep -aE 'passed, runtime' "$SP/evalgate_bf16.out" | tail -1)" >&2
echo "FP8 : $(grep -aE 'passed, runtime' "$SP/evalgate_fp8.out" | tail -1)" >&2
echo "GATE_COMPLETE" >&2
