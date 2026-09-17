#!/bin/sh
# Stage B / E receipt runner.
#
# Walks the frontier ladder in resident mode and with the SSD expert pager, so
# every row carries the read volume, hit rate, resident set and swap that
# ground rule 4 is checked against. Both modes use the locked artefact; the
# recorded sha256 is asserted, not assumed.
#
# The two modes get different ladders on purpose. The never-cached pager reads
# the whole expert working set once per token, so its read volume grows with
# the square of the frontier: 1.24 TiB for a 2048-token prefill. Extending that
# ladder to 8192 would read another ~22 TiB and confirm what the smallest
# frontier already rejects by 36x, at the cost of the drive. The ladder is
# therefore a sanity sweep in resident mode and a single measured frontier for
# the pager; the extrapolation is arithmetic, not measurement.
#
# Usage: run-stageb.sh OUT.csv [PAGED_CTX] [GEN_TOKENS]

set -eu

OUT=${1:-speed-bench/stageB-real.csv}
PAGED_CTX=${2:-2048}
GEN=${3:-16}
RES_LO=2048
RES_HI=8192

# ds4 refuses to run a second instance, and it writes its CSV header before it
# gets that far -- so a concurrent invocation silently truncates the other's
# output and leaves header rows that read like completed runs. Refuse early,
# and keep the per-mode CSVs in a private directory.
if pgrep -x ds4-bench >/dev/null 2>&1; then
    echo "run-stageb: a ds4-bench is already running; refusing to interleave" >&2
    exit 1
fi
TMPD=$(mktemp -d)
trap 'rm -rf "$TMPD"' EXIT

# CSV rows only: a header or a stray blank line must not be read as a run.
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
    echo "run-stageb: model sha256 $have, expected $WANT_SHA" >&2
    exit 1
fi

# Provenance header, so a row can never be read without its build and artefact.
{
    echo "# stage=stageB"
    echo "# git_commit=$(git rev-parse HEAD)"
    echo "# git_dirty_files=$(git status --porcelain | wc -l | tr -d ' ')"
    echo "# model_sha256=$have"
    echo "# bundle_bytes=$(wc -c < "$BUNDLE" | tr -d ' ')"
    echo "# prompt=$PROMPT"
    echo "# resident_ladder=$RES_LO..$RES_HI step 2048 gen=$GEN"
    echo "# paged_frontier=$PAGED_CTX gen=$GEN"
} > "$OUT"

echo "== resident ($RES_LO..$RES_HI) ==" >&2
# shellcheck disable=SC2086
./ds4-bench -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-start "$RES_LO" --ctx-max "$RES_HI" --gen-tokens "$GEN" \
    --csv "$TMPD/resident.csv" 2>&1 | grep -E "peak footprint" >&2 || true
rows_only "$TMPD/resident.csv" | sed 's/^/resident,/' >> "$OUT"

echo "== paged (ssd streaming) at $PAGED_CTX ==" >&2
# shellcheck disable=SC2086
DS4_QWEN4_PAGER_STATS=1 ./ds4-bench -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-start "$PAGED_CTX" --ctx-max "$PAGED_CTX" --gen-tokens "$GEN" \
    --ssd-streaming --qwen4-expert-bundle "$BUNDLE" --qwen4-expert-index "$INDEX" \
    --csv "$TMPD/paged.csv" 2>&1 | grep -E "peak footprint|PAGER_STATS" >&2 || true
rows_only "$TMPD/paged.csv" | sed 's/^/paged,/' >> "$OUT"

echo "wrote $OUT" >&2