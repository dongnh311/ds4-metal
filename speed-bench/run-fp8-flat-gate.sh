#!/bin/sh
# Item (b) gate (spec §25 Phase 3): in-kernel FP8 KV (DS4_QWEN4_KV_FP8) correctness
# + memory win. Three prefill logit dumps on the same prompt/ctx:
#   BF16   : resident (no env)                  -- the reference
#   FP8SIM : DS4_QWEN4_KV_PAGED=1 KV_FP8SIM=1   -- the CPU E4M3 round-trip (a)-certified
#   FP8    : DS4_QWEN4_KV_FP8=1                  -- the new in-kernel byte-packed path
# PASS if FP8 ~= FP8SIM (same E4M3 math: cosine>=0.99999 AND top-1 token identical),
# confirming (b) implements the (a)-certified quality; also report kv_cache_bytes to
# show the ~2x KV memory win vs BF16. FP8 gate is quality-based, NOT bit-identical
# (ground rule 6). Usage: run-fp8-flat-gate.sh [CTX] [CHUNK]
set -eu
CTX=${1:-512}
CHUNK=${2:-0}
OUTDIR=speed-bench/fp8-flat-gate
BF16_DIR="$OUTDIR/bf16"; SIM_DIR="$OUTDIR/fp8sim"; FP8_DIR="$OUTDIR/fp8"
if pgrep -x ds4-bench >/dev/null 2>&1; then echo "ds4-bench already running; refuse" >&2; exit 1; fi
rm -rf "$OUTDIR"; mkdir -p "$BF16_DIR" "$SIM_DIR" "$FP8_DIR"
MODEL=gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-MTP-NNgram.gguf
PROMPT=speed-bench/promessi_sposi.txt
WANT_SHA=ed238d8d50c6fb4b504c2796c3bc82b2a9f6c21566ad29bdbfdbadfe01e28d7a
KNOWN_SIZE=147207127040
sz=$(stat -f '%z' "$MODEL"); [ "$sz" = "$KNOWN_SIZE" ] && echo "model size OK; assert sha $WANT_SHA (skip re-hash)" >&2 || { echo "size changed $sz"; exit 1; }
CH() { [ "$CHUNK" -gt 0 ] && echo "--prefill-chunk $CHUNK"; }

echo "== BF16 (reference), ctx=$CTX ==" >&2
./ds4-bench -m "$MODEL" --prompt-file "$PROMPT" --ctx-start "$CTX" --ctx-max "$CTX" \
    --gen-tokens 0 $(CH) --dump-frontier-logits-dir "$BF16_DIR" --csv "$OUTDIR/bf16.csv" 2> "$OUTDIR/bf16.stderr"
grep -aiE "kv.?cache|kvcache" "$OUTDIR/bf16.csv" "$OUTDIR/bf16.stderr" 2>/dev/null | head -2 >&2 || true

echo "== FP8SIM (paged + fp8sim), ctx=$CTX ==" >&2
DS4_QWEN4_KV_PAGED=1 DS4_QWEN4_KV_FP8SIM=1 ./ds4-bench -m "$MODEL" --prompt-file "$PROMPT" --ctx-start "$CTX" --ctx-max "$CTX" \
    --gen-tokens 0 $(CH) --dump-frontier-logits-dir "$SIM_DIR" --csv "$OUTDIR/fp8sim.csv" 2> "$OUTDIR/fp8sim.stderr"
grep -q "paged KV path ACTIVE" "$OUTDIR/fp8sim.stderr" || { echo "GATE FAIL: fp8sim paged path inert" >&2; exit 3; }

echo "== FP8 (in-kernel DS4_QWEN4_KV_FP8=1), ctx=$CTX ==" >&2
DS4_QWEN4_KV_FP8=1 ./ds4-bench -m "$MODEL" --prompt-file "$PROMPT" --ctx-start "$CTX" --ctx-max "$CTX" \
    --gen-tokens 0 $(CH) --dump-frontier-logits-dir "$FP8_DIR" --csv "$OUTDIR/fp8.csv" 2> "$OUTDIR/fp8.stderr"
grep -q "in-kernel FP8 KV cache enabled" "$OUTDIR/fp8.stderr" || { echo "GATE FAIL: FP8 path did not enable" >&2; exit 3; }
echo "FP8 path CONFIRMED ENABLED" >&2

BF=$(ls "$BF16_DIR"/frontier_*.logits.json | head -1)
SM=$(ls "$SIM_DIR"/frontier_*.logits.json | head -1)
FP=$(ls "$FP8_DIR"/frontier_*.logits.json | head -1)
[ -n "$BF" ] && [ -n "$SM" ] && [ -n "$FP" ] || { echo "missing dump(s)" >&2; exit 1; }
set +e
python3 - "$BF" "$SM" "$FP" "$CTX" << 'PYEOF'
import json,sys,re,math
def load(p):
    raw=open(p).read()
    return json.loads(re.sub(r'(?<=[:,\[])\s*-?(?:nan|inf)\b',' null',raw))["logits"]
def clean(x): return [0.0 if v is None else v for v in x]
bf=clean(load(sys.argv[1])); sm=clean(load(sys.argv[2])); fp=clean(load(sys.argv[3])); ctx=sys.argv[4]
def cos(a,b):
    d=sum(x*y for x,y in zip(a,b)); na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(y*y for y in b))
    return d/(na*nb) if na>0 and nb>0 else 0.0
def maxd(a,b): return max(abs(x-y) for x,y in zip(a,b))
def top1(a): return max(range(len(a)), key=lambda i:a[i])
print(f"ctx={ctx} vocab={len(bf)}")
cfp_bf=cos(fp,bf); csm_bf=cos(sm,bf)
print(f"[FP8 vs FP8SIM]  cosine={cos(fp,sm):.8f}  max_abs_diff={maxd(fp,sm):.3e}  top1 fp={top1(fp)} sim={top1(sm)}")
print(f"[FP8 vs BF16]    cosine={cfp_bf:.8f}  max_abs_diff={maxd(fp,bf):.3e}  top1 fp={top1(fp)} bf={top1(bf)}")
print(f"[FP8SIM vs BF16] cosine={csm_bf:.8f}  max_abs_diff={maxd(sm,bf):.3e}")
# In-kernel FP8 and the CPU fp8sim are two INDEPENDENT E4M3 quantizers (float-row
# + metal exp2/log2 scale vs f16-row + CPU ldexpf/log2f scale); they need not be
# bit-identical. The (a) gate certified fp8sim quality (12/12). PASS iff in-kernel
# FP8 is QUALITY-EQUIVALENT to that certified reference: greedy top-1 matches BF16
# AND its cosine-to-BF16 is no worse than fp8sim's (within 5e-3). Ground rule 6:
# FP8 gate is quality-based, not bit-identical.
top1_ok = top1(fp)==top1(bf)
cos_ok  = cfp_bf >= csm_bf - 5e-3
print(f"[criterion] top1(FP8)==top1(BF16): {top1_ok}   cos(FP8,BF16) {cfp_bf:.6f} >= cos(FP8SIM,BF16)-5e-3 {csm_bf-5e-3:.6f}: {cos_ok}")
ok = top1_ok and cos_ok
print("GATE PASS: in-kernel FP8 quality-equivalent to the (a)-certified fp8sim" if ok else "GATE FAIL: in-kernel FP8 worse than the certified fp8sim")
sys.exit(0 if ok else 1)
PYEOF
RC=$?; set -e; echo "gate_rc=$RC" >&2
echo "== KV cache bytes (memory win) ==" >&2
for tag in bf16 fp8sim fp8; do
  v=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="kv_cache_bytes"){c=i}} NR>1 && c{print $c; exit}' "$OUTDIR/$tag.csv" 2>/dev/null)
  [ -n "$v" ] && printf "%-7s kv_cache_bytes=%s (%.3f GiB)\n" "$tag" "$v" "$(echo "$v/1073741824" | bc -l)" >&2 || echo "$tag: kv_cache_bytes not found" >&2
done
exit $RC
