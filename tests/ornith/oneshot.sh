#!/bin/sh
# One-shot greedy generation on Ornith against llama.cpp's generated text.
# ds4 writes 24 tokens; the reference holds more, so ds4's text must be a
# prefix of it.  A mismatch may be a near tie: Task 8 decides at id level.
set -eu
model=${DS4_ORNITH_MODEL:?}
status=0
for n in en_capital code_py vi_hanoi; do
  python3 - "$n" > "/tmp/ornith-$n.txt" <<'EOF'
import sys
sys.path.insert(0, "tests/ornith")
import ornith_ref as r
p = next(x for x in r.load_prompts("tests/ornith/prompts.json") if x["name"] == sys.argv[1])
sys.stdout.write(r.prompt_text(p, "."))
EOF
  ./ds4 -m "$model" --metal --raw --prompt-file "/tmp/ornith-$n.txt" -c 4096 --temp 0 -n 24 \
      > "/tmp/ornith-$n.out" 2> "/tmp/ornith-$n.err" || { tail -5 "/tmp/ornith-$n.err"; exit 1; }
  python3 - "$n" <<'EOF' || status=1
import json, sys
n = sys.argv[1]
ref = json.load(open(f"tests/ornith/ref/{n}.json"))["content"]
got = open(f"/tmp/ornith-{n}.out", encoding="utf-8", errors="replace").read().rstrip("\n")
same = 0
while same < min(len(got), len(ref)) and got[same] == ref[same]:
    same += 1
print(f"{n}: {'match' if same == len(got) else 'DIFF'} ({same}/{len(got)} chars)")
if same != len(got):
    print("  ds4:", repr(got[:160]))
    print("  ref:", repr(ref[:160]))
    sys.exit(1)
EOF
done
exit $status
