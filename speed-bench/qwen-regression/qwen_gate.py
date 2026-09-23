#!/usr/bin/env python3
"""Qwen3.8 regression gate (docs/V41_64GB_BUILD.md §4).

record  run the PROD gateway command (registry binary) and save a reference.
check   run the same command with ds4-server from --bin and compare.

Fast tier: byte-identical vi/code replies and an unchanged registry command.
--full adds decode t/s (median of 3, >= 97% of baseline), steady wired GiB
(<= baseline + 0.5) and a long-context needle that must be found.
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
NEEDLE_SOURCE = "ds4_server.c"
TPS_FLOOR = 0.97
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


def evaluate(baseline, current):
    """Failure messages; an empty list means the gate passes."""
    failures = []
    if baseline.get("registry_command") != current.get("registry_command"):
        failures.append("registry command changed; re-record the baseline")
        return failures
    if baseline.get("tps_median") and current.get("tps_median") is not None:
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


def run(bin_dir, out, full, needle_chars):
    running = machine.ds4_running()
    if running:
        raise SystemExit("qwen_gate: ds4 is running; the machine must be free:\n" + running)
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
            tps = []
            with wired.WiredSampler() as ws:
                for _ in range(3):
                    _, n, _, seconds = chat(base, PROMPTS["code"], 300)
                    tps.append(n / seconds)
            result["tps"] = tps
            result["tps_median"] = statistics.median(tps)
            result["wired"] = ws.summary()
            with open(os.path.join(ROOT, NEEDLE_SOURCE), encoding="utf-8", errors="replace") as fp:
                prompt = make_needle(fp.read(), needle_chars)
            reply, _, prompt_tokens, _ = chat(base, prompt, 512)
            result["needle_prompt_tokens"] = prompt_tokens
            result["needle_hit"] = NEEDLE_KEY in reply
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
        p.add_argument("--needle-chars", type=int, default=650_000)
    args = ap.parse_args()
    if args.mode == "record":
        run(None, args.out, args.full, args.needle_chars)
        print("qwen_gate: reference recorded in", args.out)
        return 0
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
