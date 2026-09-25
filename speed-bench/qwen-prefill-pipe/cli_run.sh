#!/bin/bash
# One CLI prefill + decode run with the PROD ds4 flags (gateway registry, 2026-09-24).
# usage: cli_run.sh OUT_DIR TAG PROMPT_FILE MODE [NGEN]
# MODE is DS4_QWEN4_PREFILL_MODE (off|safe|max); other env (DS4_METAL_GPU_IDLE, ...)
# passes through. Prints "TAG prefill P gen G md5 M". Needs the machine free.
set -euo pipefail
out=$1 tag=$2 prompt=$3 mode=$4 ngen=${5:-300}
root=$(cd "$(dirname "$0")/../.." && pwd)
models=${DS4_MODELS_DIR:-$HOME/.local/share/ai-gateway/ds4-models}
mkdir -p "$out"
cd "$root"
env DS4_QWEN4_STREAM_FULL_LAYERS=32 DS4_QWEN4_PLE_PREFETCH_FULL=0 \
    DS4_QWEN4_MTP_DRAFT_VOCAB="$models/Qwen3.8-Flash-Next-draft-vocab-vi-en-code-64k.txt" \
    DS4_QWEN4_KV_GROW=1 DS4_QWEN4_PREFILL_MODE="$mode" \
    ./ds4 --metal \
    -m "$models/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q4KDownPad768-DenseQ4Kselimat-MTP.gguf" \
    --ple "$models/Qwen3.8-Flash-Next-PLE-Q4_1.gguf" \
    -c 262144 --prefill-chunk 2048 --mtp --ssd-streaming --ssd-streaming-cache-experts 6GB \
    --temp 0 -n "$ngen" --prompt-file "$prompt" \
    > "$out/$tag.out" 2> "$out/$tag.err"
speeds=$(grep -a 'prefill:' "$out/$tag.err" | tail -1 |
         sed -E 's/.*prefill: ([0-9.]+) t\/s, generation: ([0-9.]+) t\/s.*/prefill \1 gen \2/')
echo "$tag $speeds md5 $(md5 -q "$out/$tag.out")"
