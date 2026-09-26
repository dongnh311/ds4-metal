"""MTP acceptance of ds4 against llama.cpp's draft-mtp on the M1 prompts.

Greedy output cannot show a wrong draft, only a lower acceptance, so this
is the check on the MTP catch-up and draft head.  Both sides draft one
token per cycle at temperature 0 from the same prompt token ids.  One model
process at a time: every ds4 run first, then one fresh llama-server per
prompt (llama.cpp carries its MTP seed across requests in a slot).

Usage: python3 tests/ornith/mtp_accept.py OUT_DIR [--n 128]
Needs DS4_ORNITH_MODEL, ./ds4 and llama-server (0.5.0) on PATH.
Writes OUT_DIR/accept.json and OUT_DIR/accept.md; exits 1 when ds4's
overall acceptance is more than 5 points below llama.cpp's.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ornith_ref as r  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PORT = 18190
DS4_RE = re.compile(r"Ornith mtp: (\d+) verify cycles, (\d+) drafts accepted")


def run_ds4(model, prompt, n, out):
    txt = os.path.join(out, prompt["name"] + ".txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(r.prompt_text(prompt, ROOT))
    cmd = [os.path.join(ROOT, "ds4"), "-m", model, "--metal", "--raw", "--prompt-file", txt,
           "-c", "16384", "--temp", "0", "-n", str(n), "--mtp", "--mtp-timing"]
    res = r.run_ds4(cmd, cwd=ROOT, timeout=1800, text=True)
    with open(os.path.join(out, prompt["name"] + ".ds4.stderr"), "w") as f:
        f.write(res.stderr)
    m = DS4_RE.search(res.stderr)
    if res.returncode != 0 or not m:
        sys.exit(f"{prompt['name']}: ds4 failed or printed no acceptance (see {out})")
    return int(m.group(1)), int(m.group(2))


def http_json(url, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def run_llama(model, name, ids, n, out):
    log = open(os.path.join(out, name + ".llama.log"), "w")
    srv = subprocess.Popen(
        ["llama-server", "-m", model, "--spec-type", "draft-mtp", "--spec-draft-n-max", "1",
         "--spec-draft-p-min", "0", "-np", "1", "-c", "16384", "-b", "2048", "-ub", "512",
         "-ngl", "all", "-fa", "on", "--temp", "0", "--top-k", "1",
         "--host", "127.0.0.1", "--port", str(PORT)],
        stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 600
        while True:
            if srv.poll() is not None:
                sys.exit(f"{name}: llama-server exited (see {log.name})")
            try:
                if http_json(f"http://127.0.0.1:{PORT}/health", timeout=5).get("status") == "ok":
                    break
            except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
                pass
            if time.time() > deadline:
                sys.exit(f"{name}: llama-server did not become ready")
            time.sleep(2)
        body = {"prompt": ids, "n_predict": n, "temperature": 0, "top_k": 1, "cache_prompt": False}
        t = http_json(f"http://127.0.0.1:{PORT}/completion", body)["timings"]
        return int(t.get("draft_n", 0)), int(t.get("draft_n_accepted", 0))
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=120)
        except subprocess.TimeoutExpired:
            sys.exit(f"{name}: llama-server did not stop after SIGTERM; do not kill -9, check it by hand")
        log.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = os.environ.get("DS4_ORNITH_MODEL") or sys.exit("set DS4_ORNITH_MODEL")
    prompts = r.load_prompts(os.path.join(ROOT, "tests/ornith/prompts.json"))
    rows = []
    for p in prompts:
        cycles, acc = run_ds4(model, p, args.n, args.out)
        rows.append({"name": p["name"], "ds4_drafts": cycles, "ds4_accepted": acc})
    for row in rows:
        ids = json.load(open(os.path.join(ROOT, "tests/ornith/ref", row["name"] + ".json")))["prompt_ids"]
        drafted, acc = run_llama(model, row["name"], ids, args.n, args.out)
        row["llama_drafts"], row["llama_accepted"] = drafted, acc
    tot = {k: sum(x[k] for x in rows) for k in ("ds4_drafts", "ds4_accepted", "llama_drafts", "llama_accepted")}
    ds4_rate = tot["ds4_accepted"] / max(tot["ds4_drafts"], 1)
    llama_rate = tot["llama_accepted"] / max(tot["llama_drafts"], 1)
    json.dump({"rows": rows, "total": tot, "ds4_rate": ds4_rate, "llama_rate": llama_rate},
              open(os.path.join(args.out, "accept.json"), "w"), indent=1)
    lines = ["| prompt | ds4 accepted/drafts | ds4 rate | llama.cpp accepted/drafts | llama.cpp rate |",
             "|---|---|---|---|---|"]
    for x in rows:
        dr = x["ds4_accepted"] / max(x["ds4_drafts"], 1)
        lr = x["llama_accepted"] / max(x["llama_drafts"], 1)
        lines.append(f"| {x['name']} | {x['ds4_accepted']}/{x['ds4_drafts']} | {dr:.3f} | "
                     f"{x['llama_accepted']}/{x['llama_drafts']} | {lr:.3f} |")
    lines.append(f"| **total** | {tot['ds4_accepted']}/{tot['ds4_drafts']} | **{ds4_rate:.3f}** | "
                 f"{tot['llama_accepted']}/{tot['llama_drafts']} | **{llama_rate:.3f}** |")
    open(os.path.join(args.out, "accept.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    ok = ds4_rate >= llama_rate - 0.05
    print(f"mtp_accept: {'PASS' if ok else 'FAIL'} ds4 {ds4_rate:.3f} vs llama.cpp {llama_rate:.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
