#!/bin/sh
# Loader checks for Ornith (qwen35moe): the 23G GGUF inspects cleanly, and a
# wrong metadata value or an unsupported expert type fails with a message
# naming the key or the tier.  Needs a built ./ds4 and DS4_ORNITH_MODEL.
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
echo "ornith loader: ok"
