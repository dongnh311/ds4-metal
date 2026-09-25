#!/bin/bash
# Byte-identical check across DS4_QWEN4_PREFILL_MODE=off|safe|max.
# usage: cli_exact.sh OUT_DIR [PROMPT...]   (default: chat p5k p36k p134k)
# Exits non-zero if any prompt's replies differ between modes. Needs the machine free.
set -euo pipefail
out=$1
shift
here=$(cd "$(dirname "$0")" && pwd)
python3 "$here/make_prompts.py" "$out/prompts"
prompts=("$@")
[ ${#prompts[@]} -eq 0 ] && prompts=(chat p5k p36k p134k)
fail=0
for p in "${prompts[@]}"; do
    sums=()
    for mode in off safe max; do
        line=$("$here/cli_run.sh" "$out" "$p-$mode" "$out/prompts/$p.txt" "$mode" 300)
        echo "$line"
        sums+=("${line##* }")
    done
    if [ "$(printf '%s\n' "${sums[@]}" | sort -u | wc -l)" -ne 1 ]; then
        echo "FAIL $p: replies differ between modes"
        fail=1
    else
        echo "PASS $p"
    fi
done
exit $fail
