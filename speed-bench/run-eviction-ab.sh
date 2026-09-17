#!/bin/sh
# Eviction A/B receipt: E1 (LRU) vs E2 (scored) at a context where the
# resident cache (fixed to the floor by ds4_engine_configure_streaming_auto_cache)
# is well under the routed working set, so eviction policy actually bites.
#
# Usage: run-eviction-ab.sh OUT.csv [CTX] [GEN_TOKENS]
set -eu

OUT=${1:-speed-bench/eviction-ab.csv}
CTX=${2:-8192}
GEN=${3:-32}

if pgrep -x ds4-bench >/dev/null 2>&1 || pgrep -x ds4 >/dev/null 2>&1; then
    echo "run-eviction-ab: an instance is already running; refusing to interleave" >&2
    exit 1
fi
TMPD=$(mktemp -d)
trap 'rm -rf "$TMPD"' EXIT

rows_only() {
    grep -v '^#' "$1" | awk -F, 'NF>10 && $2 ~ /^[0-9]+$/'
}

MODEL=gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-MTP-NNgram.gguf
BUNDLE=qwen38-experts.bin
INDEX=qwen38-experts.index.json
PROMPT=speed-bench/promessi_sposi.txt
WANT_SHA=ed238d8d50c6fb4b504c2796c3bc82b2a9f6c21566ad29bdbfdbadfe01e28d7a

have=$(shasum -a 256 "$MODEL" | cut -d' ' -f1)
if [ "$have" != "$WANT_SHA" ]; then
    echo "run-eviction-ab: model sha256 $have, expected $WANT_SHA" >&2
    exit 1
fi

{
    echo "# stage=stageB-eviction-ab"
    echo "# git_commit=$(git rev-parse HEAD)"
    echo "# git_dirty_files=$(git status --porcelain | wc -l | tr -d ' ')"
    echo "# model_sha256=$have"
    echo "# bundle_bytes=$(wc -c < "$BUNDLE" | tr -d ' ')"
    echo "# prompt=$PROMPT"
    echo "# ctx=$CTX gen=$GEN"
} > "$OUT"

for policy in lru scored; do
    echo "== policy=$policy ctx=$CTX ==" >&2
    # shellcheck disable=SC2086
    DS4_QWEN4_EXPERT_EVICT="$policy" DS4_QWEN4_PAGER_STATS=1 \
        ./ds4-bench -m "$MODEL" --prompt-file "$PROMPT" \
        --ctx-start "$CTX" --ctx-max "$CTX" --gen-tokens "$GEN" \
        --ssd-streaming --qwen4-expert-bundle "$BUNDLE" --qwen4-expert-index "$INDEX" \
        --csv "$TMPD/$policy.csv" 2>&1 | tee "$TMPD/$policy.log" | \
        grep -E "peak footprint|PAGER_STATS|resident cache|evict" >&2 || true
    rows_only "$TMPD/$policy.csv" | sed "s/^/$policy,/" >> "$OUT"
    # Carry the pager's own summary (hit rate, evictions) into the receipt as
    # a provenance comment line, since the CSV row itself does not have an
    # eviction-count column.
    grep -E "^DS4_PAGER_STATS|Cache:" "$TMPD/$policy.log" | sed "s/^/# $policy: /" >> "$OUT"
done

echo "wrote $OUT" >&2
