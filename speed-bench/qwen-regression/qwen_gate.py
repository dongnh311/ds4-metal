#!/usr/bin/env python3
"""Qwen3.8 regression gate (docs/V41_64GB_BUILD.md §4).

record  run the PROD gateway command (registry binary) and save a reference.
check   run the same command with ds4-server from --bin and compare.

Fast tier: byte-identical vi/code replies and an unchanged registry command.
--full adds steady wired GiB (<= baseline + 0.5) and a long-context needle that
must be found. With --bin it also runs a paired speed check: PROD, branch,
branch, PROD on fresh servers, branch mean of per-server median t/s >= 97 % of
PROD's (without --bin it compares against the stored median of 3).
"""
import argparse
import contextlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "speed-bench", "lib"))
import machine  # noqa: E402
import wired  # noqa: E402

REGISTRY = os.path.expanduser(os.environ.get(
    "DS4_GATEWAY_REGISTRY", "~/.local/ai-gateway/runtime-registry.json"))
# Same prompts and request body as deploy-ai-gateway.sh smoke.
PROMPTS = {
    "vi": "Giải thích ngắn gọn cách bộ nhớ đệm KV giúp mô hình ngôn ngữ sinh văn bản nhanh hơn.",
    "code": "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and two doctests.",
}
NEEDLE = "The secret passphrase is VIOLET-HARBOR-2719."
NEEDLE_KEY = "VIOLET-HARBOR-2719"
QUESTION = "\n\nWhat is the secret passphrase stated in the text above? Answer with the passphrase only."
NEEDLE_SOURCE = "speed-bench/promessi_sposi.txt"
TPS_FLOOR = 0.97
# The same binary's decode t/s varies 6-12 % from one server start to the next
# (the first start after idle measured slowest), so decode speed is judged against
# PROD measured interleaved in the same run, never against a stored number.
AB_ORDER = ("prod", "branch", "branch", "prod")
WIRED_SLACK_GIB = 0.5


def registry_command(registry, bin_dir, port, kv_dir):
    """PROD ds4 command from the gateway registry, retargeted like the smoke."""
    with open(registry) as fp:
        models = json.load(fp)["models"]
    entry = next((m["runtimes"]["ds4"] for m in models.values()
                  if m.get("runtimes", {}).get("ds4", {}).get("enabled")), None)
    if entry is None:
        raise SystemExit("qwen_gate: no enabled ds4 runtime in " + registry)
    cmd = list(entry["process_command"])
    for i, arg in enumerate(cmd):
        if arg == "--port":
            cmd[i + 1] = str(port)
        elif arg == "--kv-disk-dir":
            cmd[i + 1] = kv_dir
        elif bin_dir and os.path.basename(arg) == "ds4-server":
            cmd[i] = os.path.join(os.path.abspath(bin_dir), "ds4-server")
    cwd = os.path.abspath(bin_dir) if bin_dir else entry.get("process_cwd")
    return cmd, cwd


def compare_replies(ref_dir, replies):
    """Names whose reply differs from ref_dir/<name>.txt or has no reference."""
    bad = []
    for name, text in replies.items():
        try:
            with open(os.path.join(ref_dir, name + ".txt"), encoding="utf-8") as fp:
                if fp.read() != text:
                    bad.append(name)
        except OSError:
            bad.append(name)
    return bad


def make_needle(filler, chars):
    """filler[:chars] with NEEDLE on its own line near the middle, QUESTION at the end."""
    if len(filler) < chars:
        raise ValueError(f"needle filler has {len(filler)} chars, need {chars}")
    body = filler[:chars]
    mid = body.rfind("\n", 0, chars // 2) + 1
    return body[:mid] + NEEDLE + "\n" + body[mid:] + QUESTION


def speed_ab_failures(ab):
    """Paired speed verdict: the branch's mean of per-server medians must reach
    TPS_FLOOR of PROD's, both measured interleaved in the same gate run."""
    for side in ("prod", "branch"):
        if not ab.get(side):
            return [f"paired A/B has no {side} measurement"]
    prod = statistics.fmean(ab["prod"])
    branch = statistics.fmean(ab["branch"])
    floor = prod * TPS_FLOOR
    if branch < floor:
        per = "/".join(f"{x:.2f}" for x in ab["branch"])
        per_prod = "/".join(f"{x:.2f}" for x in ab["prod"])
        return [f"decode {branch:.2f} t/s ({per}) < {floor:.2f} (97% of PROD {prod:.2f} "
                f"({per_prod}), paired A/B; rerun once before calling it a regression)"]
    return []


def paired_speed(bin_dir, out, port, measure=None):
    """Interleaved PROD/branch servers: PROD runs the registry binary, the branch --bin."""
    measure = measure or measure_server
    tags = iter(range(len(AB_ORDER)))
    return speed_ab(lambda which: measure(None if which == "prod" else bin_dir, out,
                                          f"{next(tags)}-{which}", port))


def require_idle(read=None, timeout=wired.IDLE_SETTLE_S, interval=2.0):
    """Wait for the previous server's Metal wiring to drain; refuse if it does not."""
    idle = wired.wait_idle_gib(read=read, timeout=timeout, interval=interval)
    if idle > wired.IDLE_WIRED_LIMIT_GIB:
        raise SystemExit(f"qwen_gate: {idle:.1f} GiB wired before the run; the machine is not idle")


def speed_ab(measure, order=AB_ORDER):
    """Run `measure(which)` in the interleaved order; which is "prod" or "branch" and
    measure returns the timed t/s list of one fresh server. Returns per-server medians."""
    ab = {"prod": [], "branch": [], "runs": []}
    for which in order:
        tps = measure(which)
        ab[which].append(statistics.median(tps))
        ab["runs"].append({"bin": which, "tps": tps})
    return ab


def evaluate(baseline, current):
    """Failure messages; an empty list means the gate passes."""
    failures = []
    if baseline.get("registry_command") != current.get("registry_command"):
        failures.append("registry command changed; re-record the baseline")
        return failures
    if (current.get("tps_median") is not None or current.get("wired")) and \
            (not baseline.get("tps_median") or not baseline.get("wired")):
        failures.append("baseline lacks full-tier data; re-record with --full")
        return failures
    if current.get("speed_ab"):
        failures += speed_ab_failures(current["speed_ab"])
    elif baseline.get("tps_median") and current.get("tps_median") is not None:
        floor = baseline["tps_median"] * TPS_FLOOR
        if current["tps_median"] < floor:
            failures.append(f"decode {current['tps_median']:.2f} t/s < {floor:.2f} (97% of baseline)")
    if baseline.get("wired") and current.get("wired"):
        limit = baseline["wired"]["steady_gib"] + WIRED_SLACK_GIB
        if current["wired"]["steady_gib"] > limit:
            failures.append(f"steady wired {current['wired']['steady_gib']:.2f} GiB > {limit:.2f}")
    if "needle_hit" in current and not current["needle_hit"]:
        failures.append("needle MISS")
    return failures


@contextlib.contextmanager
def server(cmd, cwd, port, log_path):
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(900):
            if proc.poll() is not None:
                raise SystemExit("qwen_gate: ds4-server exited during startup, see " + log_path)
            try:
                urllib.request.urlopen(base + "/v1/models", timeout=2)
                break
            except OSError:
                time.sleep(1)
        else:
            raise SystemExit("qwen_gate: ds4-server did not come up in 900 s")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def chat(base, text, max_tokens):
    """(reply, completion_tokens, prompt_tokens, seconds), reply built like the smoke."""
    body = json.dumps({"model": "ds4", "messages": [{"role": "user", "content": text}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    seconds = time.time() - t0
    msg = out["choices"][0]["message"]
    reply = (msg.get("reasoning_content") or "") + "\n---\n" + (msg.get("content") or "")
    usage = out.get("usage", {})
    return reply, usage.get("completion_tokens", 0), usage.get("prompt_tokens", 0), seconds


def timed_tps(base):
    """Three timed runs of the code prompt; completion tokens per wall second."""
    tps = []
    for _ in range(3):
        _, n, _, seconds = chat(base, PROMPTS["code"], 300)
        tps.append(n / seconds)
    return tps


def measure_server(bin_dir, out, tag, port):
    """One fresh server (PROD when bin_dir is None): vi + code warm-up, then timed_tps."""
    kv = os.path.join(out, "kv-ab")
    shutil.rmtree(kv, ignore_errors=True)
    os.makedirs(kv)
    require_idle()
    cmd, cwd = registry_command(REGISTRY, bin_dir, port, kv)
    with server(cmd, cwd, port, os.path.join(out, f"server-ab-{tag}.log")) as base:
        for text in PROMPTS.values():
            chat(base, text, 300)
        tps = timed_tps(base)
    shutil.rmtree(kv, ignore_errors=True)
    return tps


def check_preflight(out, baseline):
    """Refusal message for `check`, or None when it is safe to start a server.
    Both checks run before any server is started."""
    if os.path.abspath(out) == os.path.abspath(baseline):
        return "qwen_gate: --out and --baseline resolve to the same path"
    if not os.path.exists(os.path.join(baseline, "result.json")):
        return f"qwen_gate: no result.json in baseline {baseline}; record it first"
    return None


def run(bin_dir, out, full, needle_chars, ds4_running=machine.ds4_running, idle_read=None,
        idle_timeout=wired.IDLE_SETTLE_S, idle_interval=2.0):
    # Servers run in PROD's cwd (or --bin's): a relative KV dir would land there.
    out = os.path.abspath(out)
    busy = ds4_running()
    if busy:
        raise SystemExit("qwen_gate: ds4 is running; the machine must be free:\n" + busy)
    require_idle(idle_read, idle_timeout, idle_interval)
    os.makedirs(out, exist_ok=True)
    port = int(os.environ.get("DS4_GATE_PORT", "18298"))
    kv = os.path.join(out, "kv")
    shutil.rmtree(kv, ignore_errors=True)
    os.makedirs(kv)
    cmd, cwd = registry_command(REGISTRY, bin_dir, port, kv)
    result = {"registry_command": registry_command(REGISTRY, None, "PORT", "KV")[0],
              "replies": {}}
    with server(cmd, cwd, port, os.path.join(out, "server.log")) as base:
        for name, text in PROMPTS.items():
            reply, n, _, _ = chat(base, text, 300)
            if n == 0:
                raise SystemExit(f"qwen_gate: empty reply for {name}")
            with open(os.path.join(out, name + ".txt"), "w", encoding="utf-8") as fp:
                fp.write(reply)
            result["replies"][name] = reply
        if full:
            with wired.WiredSampler() as ws:
                tps = timed_tps(base)
            result["tps"] = tps
            result["tps_median"] = statistics.median(tps)
            result["wired"] = ws.summary()
            with open(os.path.join(ROOT, NEEDLE_SOURCE), encoding="utf-8", errors="replace") as fp:
                prompt = make_needle(fp.read(), needle_chars)
            reply, _, prompt_tokens, _ = chat(base, prompt, 512)
            result["needle_prompt_tokens"] = prompt_tokens
            result["needle_hit"] = NEEDLE_KEY in reply
    if full and bin_dir is not None:
        result["speed_ab"] = paired_speed(bin_dir, out, port)
    still = ds4_running()
    if still:
        raise SystemExit("qwen_gate: another ds4 process ran during the gate; "
                          "results are not trustworthy")
    shutil.rmtree(kv, ignore_errors=True)
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=1, ensure_ascii=False)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    rec = sub.add_parser("record", help="save a reference from the PROD binary")
    rec.add_argument("--out", required=True)
    chk = sub.add_parser("check", help="compare a build against a reference")
    chk.add_argument("--bin", required=True, help="directory holding the ds4-server to test")
    chk.add_argument("--baseline", required=True)
    chk.add_argument("--out", required=True)
    for p in (rec, chk):
        p.add_argument("--full", action="store_true")
        p.add_argument("--needle-chars", type=int, default=730_000)
    args = ap.parse_args()
    if args.mode == "record":
        run(None, args.out, args.full, args.needle_chars)
        print("qwen_gate: reference recorded in", args.out)
        return 0
    msg = check_preflight(args.out, args.baseline)
    if msg:
        raise SystemExit(msg)
    current = run(args.bin, args.out, args.full, args.needle_chars)
    failures = [f"{name}: reply differs from the reference"
                for name in compare_replies(args.baseline, current["replies"])]
    with open(os.path.join(args.baseline, "result.json"), encoding="utf-8") as fp:
        failures += evaluate(json.load(fp), current)
    for failure in failures:
        print("qwen_gate: FAIL", failure)
    print("qwen_gate:", "FAIL" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
