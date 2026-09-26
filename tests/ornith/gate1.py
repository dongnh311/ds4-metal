#!/usr/bin/env python3
"""M1 gate 1: ds4 against the llama.cpp references for every prompt.

  gate1.py [--ds4-arg ARG ...] OUT_DIR

For each prompt in tests/ornith/prompts.json: write the raw text, check that
ds4's prompt tokens equal llama.cpp's, run `ds4 --dump-logprobs` greedily for
the reference's step count, and compare with tests/ornith/tolerance.json.
A prompt whose comparison checked no probable-token pair fails as vacuous.
Extra ds4 arguments (for example --prefill-chunk 64) are passed through.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import ornith_ref as r

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main(argv):
    extra = []
    while len(argv) >= 2 and argv[0] == "--ds4-arg":
        extra.append(argv[1])
        argv = argv[2:]
    if len(argv) != 1:
        sys.exit(__doc__)
    out = argv[0]
    model = os.environ["DS4_ORNITH_MODEL"]
    os.makedirs(os.path.join(out, "ds4"), exist_ok=True)
    tol = json.load(open(os.path.join(ROOT, "tests/ornith/tolerance.json")))
    failed = compared = checked = 0
    for p in r.load_prompts(os.path.join(ROOT, "tests/ornith/prompts.json")):
        ref = json.load(open(os.path.join(ROOT, "tests/ornith/ref", p["name"] + ".json")))
        txt = os.path.join(out, "ds4", p["name"] + ".txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write(r.prompt_text(p, ROOT))
        ids_line = subprocess.run([os.path.join(ROOT, "ds4"), "-m", model, "--raw", "--prompt-file", txt,
                                   "--dump-tokens"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
        if json.loads(ids_line) != ref["prompt_ids"]:
            print(f"{p['name']}: FAIL tokenization differs")
            failed += 1
            continue
        dump = os.path.join(out, "ds4", p["name"] + ".json")
        log = open(os.path.join(out, "ds4", p["name"] + ".log"), "w")
        cmd = [os.path.join(ROOT, "ds4"), "-m", model, "--metal", "--raw", "--prompt-file", txt,
               "-c", "16384", "-n", str(len(ref["steps"])), "--dump-logprobs", dump] + extra
        if subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode != 0:
            print(f"{p['name']}: FAIL ds4 exited non-zero (see {log.name})")
            failed += 1
            continue
        res = r.compare(ref["steps"], r.ds4_steps(json.load(open(dump))), tol["tol"], tol["tie"])
        status = r.verdict(res)
        failed += 0 if status == "ok" else 1
        compared += res["compared"]
        checked += res["checked"]
        reason = f" {res['reason']}" if res["reason"] else ""
        print(f"{p['name']}: {status} compared={res['compared']} checked={res['checked']} "
              f"tie_at={res['stopped_at_tie']} max_delta={res['max_delta']:.4f}{reason}")
    print(f"gate1: {'PASS' if failed == 0 else f'FAIL ({failed})'} compared={compared} checked={checked} "
          f"tol={tol['tol']:.4f} tie={tol['tie']:.4f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
