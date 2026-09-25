#!/bin/bash
# Qwen3.8 regression gate (docs/V41_64GB_BUILD.md §4).
#   run.sh fast [BASELINE_DIR]   unit kernels + byte-identical PROD replies
#   run.sh full [BASELINE_DIR]   fast + decode t/s, steady wired GiB, long needle
# Needs the machine free: no other ds4 process may run.
set -euo pipefail
tier=${1:?usage: run.sh fast|full [BASELINE_DIR]}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
baseline=${2:-$here/baseline}
case "$tier" in
    fast) full_flag="" ;;
    full) full_flag="--full" ;;
    *) echo "usage: run.sh fast|full [BASELINE_DIR]" >&2; exit 2 ;;
esac
cd "$root"
make ds4-server test-qwen4-kernels test-qwen4-q2 test-qwen4-prefill-pipe
python3 "$here/qwen_gate.py" check --bin "$root" --baseline "$baseline" \
    --out "$here/last-$tier" $full_flag
