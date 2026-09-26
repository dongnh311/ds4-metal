#!/bin/sh
# Loader checks for Ornith (qwen35moe): the 23G GGUF inspects cleanly, and a
# wrong metadata value, an unsupported expert type (trunk or MTP) or a type
# mismatch between fused tensors fails with a message naming the key, the
# tier or the tensors.  The Metal-only open gate also
# refuses a non-Metal backend, --batched-session, a context above the
# native 262144, --mtp-model and --mtp-exact-sampling, and ds4-server /
# ds4-agent refuse Ornith until milestone M3.
# Needs a built ./ds4, ./ds4-server and ./ds4-agent and DS4_ORNITH_MODEL.
set -eu
model=${DS4_ORNITH_MODEL:?set DS4_ORNITH_MODEL to the 23G ICE GGUF}
tmp=$(mktemp -d "${TMPDIR:-/tmp}/ornith-loader.XXXXXX")
trap 'rm -rf "$tmp"' EXIT

# expect_refused NAME OUT CMD...: run CMD in the background with no input and
# require a non-zero exit.  A regression that starts serving or waits for
# input must not hang the test, so the wait is bounded: SIGTERM (never -9)
# after 120s, then fail.
expect_refused() {
    name=$1
    out=$2
    shift 2
    "$@" < /dev/null > "$out" 2>&1 &
    pid=$!
    rc=""
    i=0
    while [ "$i" -lt 120 ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            rc=0
            wait "$pid" || rc=$?
            break
        fi
        sleep 1
        i=$((i + 1))
    done
    if [ -z "$rc" ]; then
        kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        cat "$out"
        echo "$name did not exit within 120s"
        exit 1
    fi
    if [ "$rc" -eq 0 ]; then
        cat "$out"
        echo "$name accepted Ornith"
        exit 1
    fi
}
./ds4 --inspect -m "$model" > "$tmp/ok.txt" 2>&1 || { cat "$tmp/ok.txt"; exit 1; }
grep -q 'Ornith-1.5-35B-A3B: 40 layers + MTP' "$tmp/ok.txt" || { cat "$tmp/ok.txt"; exit 1; }
python3 tests/ornith/make_bad_gguf.py "$model" "$tmp"
if ./ds4 --inspect -m "$tmp/bad_embd.gguf" > "$tmp/embd.txt" 2>&1; then
    echo "bad metadata accepted"; exit 1
fi
grep -q 'expected embedding_length=2048' "$tmp/embd.txt" || { cat "$tmp/embd.txt"; exit 1; }
if ./ds4 --inspect -m "$tmp/bad_tier.gguf" > "$tmp/tier.txt" 2>&1; then
    echo "IQ4_XS experts accepted"; exit 1
fi
grep -q 'not supported; use the 23G or 25G tier' "$tmp/tier.txt" || { cat "$tmp/tier.txt"; exit 1; }
if ./ds4 --inspect -m "$tmp/bad_mtp.gguf" > "$tmp/mtp.txt" 2>&1; then
    echo "IQ4_XS MTP experts accepted"; exit 1
fi
grep -q 'not supported; use the 23G or 25G tier' "$tmp/mtp.txt" || { cat "$tmp/mtp.txt"; exit 1; }
# Tensors fused into one kernel slot must share a quant type.
if ./ds4 --inspect -m "$tmp/bad_gdn_pair.gguf" > "$tmp/gdn_pair.txt" 2>&1; then
    echo "ssm_alpha/ssm_beta type mismatch accepted"; exit 1
fi
grep -q 'layer 0: blk.0.ssm_beta.weight is f32 but blk.0.ssm_alpha.weight is f16' "$tmp/gdn_pair.txt" ||
    { cat "$tmp/gdn_pair.txt"; exit 1; }
if ./ds4 --inspect -m "$tmp/bad_shexp_pair.gguf" > "$tmp/shexp_pair.txt" 2>&1; then
    echo "ffn_gate_shexp/ffn_up_shexp type mismatch accepted"; exit 1
fi
grep -q 'layer 0: blk.0.ffn_up_shexp.weight is q4_0 but blk.0.ffn_gate_shexp.weight is q8_0' "$tmp/shexp_pair.txt" ||
    { cat "$tmp/shexp_pair.txt"; exit 1; }

# Metal-only open gate: a non-Metal backend is refused before any model load.
if ./ds4 --cpu -m "$model" -p hi -n 1 > "$tmp/cpu.txt" 2>&1; then
    echo "CPU backend accepted for Ornith"; exit 1
fi
grep -q 'not supported' "$tmp/cpu.txt" || { cat "$tmp/cpu.txt"; exit 1; }

# Metal-only open gate: a context above the native 262144 (no YaRN) is
# refused before any model load.
if ./ds4 -m "$model" --raw -c 262145 -p hi -n 1 > "$tmp/ctx.txt" 2>&1; then
    echo "context 262145 accepted for Ornith"; exit 1
fi
grep -q 'Ornith supports up to 262144 tokens of context (no YaRN)' "$tmp/ctx.txt" || { cat "$tmp/ctx.txt"; exit 1; }

# Metal-only open gate: --batched-session > 1 is refused too (port 18191,
# never a gateway port).
expect_refused "ds4-server --batched-session 2" "$tmp/srv.txt" \
    ./ds4-server -m "$model" --batched-session 2 --port 18191
grep -q 'not supported' "$tmp/srv.txt" || { cat "$tmp/srv.txt"; exit 1; }

# ds4-server and ds4-agent refuse Ornith until milestone M3.
expect_refused "ds4-server" "$tmp/srv_plain.txt" ./ds4-server -m "$model" --port 18191
grep -q 'ds4-server: Ornith-1.5-35B-A3B serving arrives in milestone M3' "$tmp/srv_plain.txt" ||
    { cat "$tmp/srv_plain.txt"; exit 1; }
expect_refused "ds4-agent" "$tmp/agent.txt" ./ds4-agent -m "$model"
grep -q 'ds4-agent: Ornith-1.5-35B-A3B agent mode arrives in milestone M3' "$tmp/agent.txt" ||
    { cat "$tmp/agent.txt"; exit 1; }

# M2: --mtp runs the embedded blk.40 head; an external MTP model and exact
# speculative sampling stay refused.
if ./ds4 -m "$model" --raw --mtp --mtp-exact-sampling -p hi -n 1 > "$tmp/mtp_exact.txt" 2>&1; then
    echo "--mtp-exact-sampling accepted for Ornith"; exit 1
fi
grep -q -- '--mtp-exact-sampling is not supported' "$tmp/mtp_exact.txt" || { cat "$tmp/mtp_exact.txt"; exit 1; }
if ./ds4 -m "$model" --raw --mtp-model "$model" -p hi -n 1 > "$tmp/mtp_model.txt" 2>&1; then
    echo "--mtp-model accepted for Ornith"; exit 1
fi
grep -q -- '--mtp-model is not supported' "$tmp/mtp_model.txt" || { cat "$tmp/mtp_model.txt"; exit 1; }

echo "ornith loader: ok"
