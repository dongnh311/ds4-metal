"""Ornith --mtp must not change greedy output (spec section 8, gate 1 item 3).

For each (prompt, prefill chunk) the one-shot plain generator, the session
path without MTP (DS4_CLI_FORCE_SESSION=1) and the session path with --mtp
must print byte-identical text.  DS4_QWEN35_SPEC_TRACE counts the verify
outcomes: the whole run must see accepted and rejected drafts, and every
long_copy run (9,371 prompt tokens, past the 2048-key attention split
change) must verify at least once.

A final smoke run at --temp 0.7 exercises the Ornith branch of the sampled
speculative path (ds4_session_eval_speculative, ds4.c ~88240): with greedy
decoding, --mtp always resolves to temperature <= 0 and takes the argmax
path instead, so that branch is otherwise never exercised.  It cannot be
compared against plain decoding (acceptance there is opportunistic, so a
different token stream is expected), so this only checks the run finishes
cleanly with output and at least one verify cycle.

Usage: python3 tests/ornith/test_mtp_cli.py OUT_DIR
Needs DS4_ORNITH_MODEL and a built ./ds4.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ornith_ref as r  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHORT = ["en_capital", "code_py", "vi_hanoi", "en_contributing", "code_c_pack"]
RUNS = [(p, c) for p in SHORT for c in (1, 2, 64, 0)] + [("long_copy", 0), ("long_copy", 64)]
N_PREDICT = 48
MODES = (
    ("plain", [], {}),
    ("session", [], {"DS4_CLI_FORCE_SESSION": "1"}),
    ("mtp", ["--mtp"], {"DS4_QWEN35_SPEC_TRACE": "1"}),
)


def temp_smoke(model, prompts, out):
    """Smoke check only (see module docstring): --mtp --temp 0.7 --seed 1
    must run to completion, print output and verify at least once."""
    name = "en_capital"
    txt = os.path.join(out, name + "-temp.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(r.prompt_text(prompts[name], ROOT))
    cmd = [os.path.join(ROOT, "ds4"), "-m", model, "--metal", "--raw", "--prompt-file", txt,
           "-c", "16384", "--temp", "0.7", "--seed", "1", "-n", str(N_PREDICT), "--mtp"]
    env = dict(os.environ)
    env["DS4_QWEN35_SPEC_TRACE"] = "1"
    env.pop("DS4_QWEN35_SPEC_FORCE_ACCEPT", None)
    env.pop("DS4_CLI_FORCE_SESSION", None)
    tag = "temp-smoke"
    res = r.run_ds4(cmd, cwd=ROOT, env=env, timeout=1800)
    with open(os.path.join(out, tag + ".stdout"), "wb") as f:
        f.write(res.stdout)
    with open(os.path.join(out, tag + ".stderr"), "wb") as f:
        f.write(res.stderr)
    cycles = res.stderr.count(b" accept\n") + res.stderr.count(b" reject\n")
    ok = res.returncode == 0 and b"failed" not in res.stderr.lower() and len(res.stdout) > 0 and cycles >= 1
    print(f"{tag}: {'ok' if ok else 'FAIL'} (exit {res.returncode}, {len(res.stdout)} bytes, "
          f"{cycles} verify cycles)")
    return 0 if ok else 1


def main(argv):
    if len(argv) != 1:
        sys.exit(__doc__)
    out = argv[0]
    os.makedirs(out, exist_ok=True)
    model = os.environ.get("DS4_ORNITH_MODEL")
    if not model:
        sys.exit("set DS4_ORNITH_MODEL to the 23G ICE GGUF")
    prompts = {p["name"]: p for p in r.load_prompts(os.path.join(ROOT, "tests/ornith/prompts.json"))}
    accepts = rejects = failed = 0
    for name, chunk in RUNS:
        txt = os.path.join(out, name + ".txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write(r.prompt_text(prompts[name], ROOT))
        base = [os.path.join(ROOT, "ds4"), "-m", model, "--metal", "--raw", "--prompt-file", txt,
                "-c", "16384", "--temp", "0", "-n", str(N_PREDICT)]
        if chunk:
            base += ["--prefill-chunk", str(chunk)]
        outs = {}
        for mode, extra, env_extra in MODES:
            env = dict(os.environ)
            env.pop("DS4_QWEN35_SPEC_FORCE_ACCEPT", None)
            env.pop("DS4_CLI_FORCE_SESSION", None)
            env.update(env_extra)
            res = r.run_ds4(base + extra, cwd=ROOT, env=env, timeout=1800)
            tag = f"{name}-c{chunk}-{mode}"
            with open(os.path.join(out, tag + ".stdout"), "wb") as f:
                f.write(res.stdout)
            with open(os.path.join(out, tag + ".stderr"), "wb") as f:
                f.write(res.stderr)
            if res.returncode != 0 or b"failed" in res.stderr.lower():
                print(f"{tag}: FAIL exit {res.returncode} (see {tag}.stderr)")
                failed += 1
                continue
            outs[mode] = res.stdout
            if mode == "mtp":
                a = res.stderr.count(b" accept\n")
                j = res.stderr.count(b" reject\n")
                accepts += a
                rejects += j
                if name == "long_copy" and a + j == 0:
                    print(f"{tag}: FAIL no verify cycle")
                    failed += 1
        same = len(outs) == 3 and outs["plain"] == outs["session"] == outs["mtp"]
        print(f"{name} chunk {chunk or 'default'}: {'ok' if same else 'FAIL output differs'}")
        failed += 0 if same else 1
    failed += temp_smoke(model, prompts, out)
    print(f"mtp cli: {accepts} accepted, {rejects} rejected drafts")
    if accepts == 0 or rejects == 0:
        print("FAIL: both verify outcomes must occur")
        failed += 1
    print(f"test_mtp_cli: {'PASS' if failed == 0 else f'FAIL ({failed})'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
