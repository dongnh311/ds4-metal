#!/bin/sh
# Ornith CLI chat rendering equals the embedded template token for token:
# ./ds4 --dump-tokens renders --system/-p through encode_chat_prompt, and the
# golden text of the same conversation, tokenized as a rendered prompt
# (--raw), must give the same ids.  Reads only the GGUF header.
# Needs a built ./ds4 and DS4_ORNITH_MODEL.
set -eu
model=${DS4_ORNITH_MODEL:?set DS4_ORNITH_MODEL to the 23G ICE GGUF}
golden=tests/ornith/chat/golden
tmp=$(mktemp -d "${TMPDIR:-/tmp}/ornith-cli-chat.XXXXXX")
trap 'rm -rf "$tmp"' EXIT

# check NAME GOLDEN CLI-ARGS...
check() {
    name=$1
    case_name=$2
    shift 2
    if ! ./ds4 -m "$model" --dump-tokens "$@" > "$tmp/$name.cli" 2> "$tmp/$name.err"; then
        cat "$tmp/$name.err"
        echo "$name: ./ds4 --dump-tokens failed"
        exit 1
    fi
    ./ds4 -m "$model" --dump-tokens --raw --prompt-file "$golden/$case_name.txt" > "$tmp/$name.golden" \
        2> "$tmp/$name.golden.err" || { cat "$tmp/$name.golden.err"; exit 1; }
    if [ "$(head -n 1 "$tmp/$name.cli")" != "$(head -n 1 "$tmp/$name.golden")" ]; then
        echo "$name: CLI tokens differ from golden $case_name"
        diff "$tmp/$name.cli" "$tmp/$name.golden" | head -20
        exit 1
    fi
    echo "$name: ok"
}

check default chat_system_user -sys "You are a helpful assistant." -p "What is the capital of France?"
check nothink chat_think_off --nothink -sys "You are a helpful assistant." -p "What is the capital of France?"
# ./ds4 has a built-in default system prompt; -sys "" removes it
check nosystem chat_no_system -sys "" -p "What is the capital of France?"
check think_max effort_max --think-max -sys "You are a helpful assistant." -p "What is the capital of France?"
echo "ornith cli chat: ok"
