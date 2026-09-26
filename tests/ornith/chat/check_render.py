#!/usr/bin/env python3
"""Compare ds4-server's Ornith rendering with the committed jinja2 goldens.

  check_render.py [--ds4-test PATH] [CASE ...]

For every tests/ornith/chat/golden/<case>.input.json (or the named cases),
writes the request body to a temporary file, runs
`ds4_test --qwen35-render BODY [--anthropic]` (the server's own parser and
renderer under the Ornith flavor, no model) and requires stdout to equal
<case>.txt byte for byte.  Prints a unified diff per mismatch.  Needs no
jinja2 and no model.
"""
import difflib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
GOLDEN = os.path.join(HERE, "golden")


def case_names():
    return sorted(f[:-len(".input.json")] for f in os.listdir(GOLDEN) if f.endswith(".input.json"))


def render(ds4_test, case):
    with open(os.path.join(GOLDEN, case + ".input.json"), encoding="utf-8") as f:
        spec = json.load(f)
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(spec["body"], f, ensure_ascii=False)
        cmd = [ds4_test, "--qwen35-render", path]
        if spec["api"] == "anthropic":
            cmd.append("--anthropic")
        run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finally:
        os.unlink(path)
    if run.returncode != 0:
        return None, run.stderr.decode("utf-8", "replace").strip()
    return run.stdout.decode("utf-8"), ""


def main(argv):
    ds4_test = os.path.join(ROOT, "ds4_test")
    if len(argv) >= 2 and argv[0] == "--ds4-test":
        ds4_test, argv = argv[1], argv[2:]
    cases = argv or case_names()
    failed = 0
    for case in cases:
        with open(os.path.join(GOLDEN, case + ".txt"), encoding="utf-8", newline="") as f:
            want = f.read()
        got, err = render(ds4_test, case)
        if got == want:
            print(f"{case}: ok")
            continue
        failed += 1
        print(f"{case}: FAIL")
        if got is None:
            print("  ds4_test: " + err)
        else:
            sys.stdout.writelines(difflib.unified_diff(want.splitlines(True), got.splitlines(True),
                                                       "golden (jinja2)", "ds4-server", n=2))
    verdict = "PASS" if not failed else f"FAIL ({failed}/{len(cases)})"
    print(f"ornith render: {verdict}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
