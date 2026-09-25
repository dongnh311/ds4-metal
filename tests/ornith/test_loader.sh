#!/bin/sh
# Loader checks for Ornith (qwen35moe): the 23G GGUF inspects cleanly, and a
# wrong metadata value or an unsupported expert type (trunk or MTP) fails
# with a message naming the key or the tier.  The Metal-only open gate also
# refuses a non-Metal backend and --batched-session.  Needs a built ./ds4
# (and ./ds4-server for the batched-session check) and DS4_ORNITH_MODEL.
set -eu
model=${DS4_ORNITH_MODEL:?set DS4_ORNITH_MODEL to the 23G ICE GGUF}
tmp=$(mktemp -d "${TMPDIR:-/tmp}/ornith-loader.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
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

# Metal-only open gate: a non-Metal backend is refused before any model load.
if ./ds4 --cpu -m "$model" -p hi -n 1 > "$tmp/cpu.txt" 2>&1; then
    echo "CPU backend accepted for Ornith"; exit 1
fi
grep -q 'not supported' "$tmp/cpu.txt" || { cat "$tmp/cpu.txt"; exit 1; }

# Metal-only open gate: --batched-session > 1 is refused too.  ds4-server
# returns before starting the accept loop when ds4_engine_open fails, but
# bound the wait so a regression that does start serving cannot hang the
# test: SIGTERM (never -9) after 60s, then fail.
./ds4-server -m "$model" --batched-session 2 --port 18191 > "$tmp/srv.txt" 2>&1 &
srv_pid=$!
srv_rc=""
i=0
while [ "$i" -lt 60 ]; do
    if ! kill -0 "$srv_pid" 2>/dev/null; then
        srv_rc=0
        wait "$srv_pid" || srv_rc=$?
        break
    fi
    sleep 1
    i=$((i + 1))
done
if [ -z "$srv_rc" ]; then
    kill -TERM "$srv_pid" 2>/dev/null || true
    wait "$srv_pid" 2>/dev/null || true
    cat "$tmp/srv.txt"
    echo "ds4-server --batched-session 2 did not exit within 60s"
    exit 1
fi
if [ "$srv_rc" -eq 0 ]; then
    cat "$tmp/srv.txt"
    echo "ds4-server accepted --batched-session 2 for Ornith"
    exit 1
fi
grep -q 'not supported' "$tmp/srv.txt" || { cat "$tmp/srv.txt"; exit 1; }

echo "ornith loader: ok"
