#!/bin/sh
# Gate #5: logit dump resident vs paged, bit-identical, BEFORE any bench that
# trusts the paged expert path's numbers. The pager changes where expert
# bytes come from, not the math; if the two paths disagree, that is a
# correctness bug, not a policy choice, and the gate must fail the stage.
#
# Usage: run-logit-gate.sh [CTX]
set -eu

CTX=${1:-512}
# Prefill chunk cap. Gate #5 tests the PAGER (SSD-streamed expert bytes vs
# mmap'd resident bytes) -- it must hold the compute KERNEL constant so the
# only variable is the weight source. At chunk T>8 the resident prefill takes
# the tiled "mm" GEMM, whose dot-product accumulation order differs from the
# per-token decode kernel by floating-point non-associativity; the paged path
# structurally cannot use "mm" (its staging buffer holds one token's experts),
# so it always runs per-token. Comparing mm-resident vs per-token-paged mixes
# a kernel-ordering difference into the pager test (~1.88 max logit diff, all
# lanes, argmax stable -- inherent to any tiled GEMM, NOT a pager bug). Capping
# to 8 forces resident onto the same per-token kernel the pager uses, so the
# comparison isolates the pager and is bit-identical when the pager is correct.
# This is the same per-token path the paged run uses at any bench ctx (decode
# and streamed prefill are per-token), so the gate stays representative.
CHUNK=${2:-8}
OUTDIR=speed-bench/logit-gate
RESIDENT_DIR="$OUTDIR/resident"
PAGED_DIR="$OUTDIR/paged"

if pgrep -x ds4-bench >/dev/null 2>&1 || pgrep -x ds4 >/dev/null 2>&1; then
    echo "run-logit-gate: an instance is already running; refusing to interleave" >&2
    exit 1
fi

rm -rf "$OUTDIR"
mkdir -p "$RESIDENT_DIR" "$PAGED_DIR"

MODEL=gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-MTP-NNgram.gguf
BUNDLE=qwen38-experts.bin
INDEX=qwen38-experts.index.json
PROMPT=speed-bench/promessi_sposi.txt
WANT_SHA=ed238d8d50c6fb4b504c2796c3bc82b2a9f6c21566ad29bdbfdbadfe01e28d7a

have=$(shasum -a 256 "$MODEL" | cut -d' ' -f1)
if [ "$have" != "$WANT_SHA" ]; then
    echo "run-logit-gate: model sha256 $have, expected $WANT_SHA" >&2
    exit 1
fi

echo "== resident, ctx=$CTX prefill-chunk=$CHUNK ==" >&2
./ds4-bench -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-start "$CTX" --ctx-max "$CTX" --gen-tokens 0 --prefill-chunk "$CHUNK" \
    --dump-frontier-logits-dir "$RESIDENT_DIR" \
    --csv "$OUTDIR/resident.csv" >&2

echo "== paged (ssd streaming), ctx=$CTX prefill-chunk=$CHUNK ==" >&2
./ds4-bench -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-start "$CTX" --ctx-max "$CTX" --gen-tokens 0 --prefill-chunk "$CHUNK" \
    --ssd-streaming --qwen4-expert-bundle "$BUNDLE" --qwen4-expert-index "$INDEX" \
    --dump-frontier-logits-dir "$PAGED_DIR" \
    --csv "$OUTDIR/paged.csv" >&2

RES_FILE=$(ls "$RESIDENT_DIR"/frontier_*.logits.json | head -1)
PAGED_FILE=$(ls "$PAGED_DIR"/frontier_*.logits.json | head -1)
if [ -z "$RES_FILE" ] || [ -z "$PAGED_FILE" ]; then
    echo "run-logit-gate: missing frontier dump(s)" >&2
    exit 1
fi

set +e
python3 - "$RES_FILE" "$PAGED_FILE" "$CTX" << 'PYEOF'
import json, sys, hashlib, re

res_path, paged_path, ctx = sys.argv[1], sys.argv[2], sys.argv[3]

def load_relaxed(path):
    # The C side writes non-finite floats as bare `nan`/`inf` tokens (via
    # %.9g), which json.load rejects outright. A dump that is not valid
    # JSON is itself gate evidence -- a fully-NaN frontier -- not a script
    # bug, so patch those tokens to null rather than crashing before we
    # can even report it.
    with open(path) as f:
        raw = f.read()
    patched = re.sub(r'(?<=[:,\[])\s*-?(?:nan|inf)\b', ' null', raw)
    return json.loads(patched)

res = load_relaxed(res_path)
paged = load_relaxed(paged_path)

for label, doc in (("resident", res), ("paged", paged)):
    if doc.get("argmax_logit") is None or not isinstance(doc.get("argmax_logit"), (int, float)):
        print(f"WARNING: {label} argmax_logit is non-finite (dumped as nan/null)")

rl = res["logits"]
pl = paged["logits"]
if len(rl) != len(pl):
    print(f"GATE FAIL: vocab length mismatch resident={len(rl)} paged={len(pl)}")
    sys.exit(1)

max_diff = 0.0
max_idx = -1
n_mismatch_bits = 0
for i, (a, b) in enumerate(zip(rl, pl)):
    # A non-finite logit is dumped as JSON null (parsed as None). Any
    # resident/paged disagreement over which lanes are finite is itself a
    # mismatch -- do not let it crash the diff.
    if a is None or b is None:
        if a is not b:
            n_mismatch_bits += 1
            max_diff = float("inf")
            max_idx = i
        continue
    d = abs(a - b)
    if d > max_diff:
        max_diff = d
        max_idx = i
    if a != b:
        n_mismatch_bits += 1

bit_identical = (n_mismatch_bits == 0)
with open(res_path, "rb") as f:
    res_sha = hashlib.sha256(f.read()).hexdigest()
with open(paged_path, "rb") as f:
    paged_sha = hashlib.sha256(f.read()).hexdigest()

print(f"ctx={ctx} vocab={len(rl)}")
print(f"max_abs_diff={max_diff!r} at index={max_idx}")
print(f"value_mismatches={n_mismatch_bits}/{len(rl)}")
print(f"resident_dump_sha256={res_sha}")
print(f"paged_dump_sha256={paged_sha}")
if bit_identical:
    print("GATE PASS: bit-identical")
    sys.exit(0)
else:
    print("GATE FAIL: not bit-identical")
    sys.exit(1)
PYEOF
GATE_RC=$?
set -e
echo "gate_rc=$GATE_RC" >&2
exit $GATE_RC
