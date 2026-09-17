#!/bin/sh
# Gate #KV (spec §25 item 2.2): logit dump DS4_QWEN4_KV_PAGED=1 vs resident,
# bit-identical. Both runs use resident experts (no --ssd-streaming); the ONLY
# variable is whether K/V flows through the paged store (Design A materialize).
# Usage: run-kv-gate.sh [CTX]
set -eu
CTX=${1:-512}
CHUNK=${2:-0}
OUTDIR=speed-bench/kv-gate
RESIDENT_DIR="$OUTDIR/resident"
PAGED_DIR="$OUTDIR/paged"
if pgrep -x ds4-bench >/dev/null 2>&1; then echo "ds4-bench already running; refuse" >&2; exit 1; fi
rm -rf "$RESIDENT_DIR" "$PAGED_DIR" "$OUTDIR/resident.csv" "$OUTDIR/paged.csv"
mkdir -p "$RESIDENT_DIR" "$PAGED_DIR"
MODEL=gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-MTP-NNgram.gguf
PROMPT=speed-bench/promessi_sposi.txt
WANT_SHA=ed238d8d50c6fb4b504c2796c3bc82b2a9f6c21566ad29bdbfdbadfe01e28d7a
KNOWN_SIZE=147207127040
sz=$(stat -f '%z' "$MODEL"); [ "$sz" = "$KNOWN_SIZE" ] && echo "model size OK; assert sha $WANT_SHA (skip re-hash)" >&2 || { echo "size changed $sz; full hash"; have=$(shasum -a 256 "$MODEL"|cut -d' ' -f1); [ "$have" = "$WANT_SHA" ] || { echo "sha mismatch $have"; exit 1; }; }

echo "== resident (KV_PAGED unset), ctx=$CTX ==" >&2
./ds4-bench -m "$MODEL" --prompt-file "$PROMPT" --ctx-start "$CTX" --ctx-max "$CTX" \
    --gen-tokens 0 $([ "$CHUNK" -gt 0 ] && echo "--prefill-chunk $CHUNK") --dump-frontier-logits-dir "$RESIDENT_DIR" --csv "$OUTDIR/resident.csv" >&2

echo "== paged (DS4_QWEN4_KV_PAGED=1), ctx=$CTX ==" >&2
DS4_QWEN4_KV_PAGED=1 ./ds4-bench -m "$MODEL" --prompt-file "$PROMPT" --ctx-start "$CTX" --ctx-max "$CTX" \
    --gen-tokens 0 $([ "$CHUNK" -gt 0 ] && echo "--prefill-chunk $CHUNK") --dump-frontier-logits-dir "$PAGED_DIR" --csv "$OUTDIR/paged.csv" >&2

RES_FILE=$(ls "$RESIDENT_DIR"/frontier_*.logits.json | head -1)
PAGED_FILE=$(ls "$PAGED_DIR"/frontier_*.logits.json | head -1)
[ -n "$RES_FILE" ] && [ -n "$PAGED_FILE" ] || { echo "missing dump(s)" >&2; exit 1; }
set +e
python3 - "$RES_FILE" "$PAGED_FILE" "$CTX" << 'PYEOF'
import json,sys,hashlib,re
def load(p):
    raw=open(p).read()
    return json.loads(re.sub(r'(?<=[:,\[])\s*-?(?:nan|inf)\b',' null',raw))
res=load(sys.argv[1]); paged=load(sys.argv[2]); ctx=sys.argv[3]
rl=res["logits"]; pl=paged["logits"]
if len(rl)!=len(pl): print(f"GATE FAIL: len {len(rl)} vs {len(pl)}"); sys.exit(1)
md=0.0; mi=-1; nm=0
for i,(a,b) in enumerate(zip(rl,pl)):
    if a is None or b is None:
        if a is not b: nm+=1; md=float('inf'); mi=i
        continue
    d=abs(a-b)
    if d>md: md=d; mi=i
    if a!=b: nm+=1
rs=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()
ps=hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest()
print(f"ctx={ctx} vocab={len(rl)}")
print(f"max_abs_diff={md!r} at index={mi}")
print(f"value_mismatches={nm}/{len(rl)}")
print(f"resident_dump_sha256={rs}")
print(f"paged_dump_sha256={ps}")
if nm==0: print("GATE PASS: bit-identical"); sys.exit(0)
else: print("GATE FAIL: not bit-identical"); sys.exit(1)
PYEOF
RC=$?; set -e; echo "gate_rc=$RC" >&2; exit $RC
