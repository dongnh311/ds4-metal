"""Choose --think-budget N: thinking-token distribution and code-graded quality per N.

python3 speed-bench/think-budget/measure.py --budgets 0,8192,4096,2048,1024

Runs the PROD ds4 command from the gateway registry, but with this checkout's ds4-server, a scratch port and
KV dir, and --think-budget N (0 = off). Sends GSM8K-mini and HumanEval-mini (vendored in AI-Gateway-MLX) plus
five Vietnamese prompts at temperature 0 with thinking on, and reads the server log for the thinking length.
"""

import argparse
import json
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
REGISTRY = pathlib.Path.home() / ".local/ai-gateway/runtime-registry.json"
PORT = 18299
VI_PROMPTS = [
    "Giải thích ngắn gọn cách bộ nhớ đệm KV giúp mô hình ngôn ngữ sinh văn bản nhanh hơn.",
    "Viết một đoạn văn khoảng 120 chữ giới thiệu vịnh Hạ Long cho khách du lịch.",
    "So sánh ưu và nhược điểm của Python và Go khi viết dịch vụ web.",
    "Tóm tắt ý chính của định luật Ohm và cho một ví dụ tính toán.",
    "Một cửa hàng giảm giá 20% rồi giảm thêm 10% trên giá đã giảm. Tổng cộng giảm bao nhiêu phần trăm?",
]


def prod_command(port, kv_dir, budget):
    reg = json.loads(REGISTRY.read_text())
    rt = next(m["runtimes"]["ds4"] for m in reg["models"].values()
              if m.get("runtimes", {}).get("ds4", {}).get("enabled"))
    cmd = list(rt["process_command"])
    for i, a in enumerate(cmd):
        if a == "--port":
            cmd[i + 1] = str(port)
        elif a == "--kv-disk-dir":
            cmd[i + 1] = str(kv_dir)
        elif os.path.basename(a) == "ds4-server":
            cmd[i] = str(ROOT / "ds4-server")
    if budget > 0:
        cmd += ["--think-budget", str(budget)]
    return cmd


def ask(base, prompt):
    body = {"model": "ds4", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16384, "temperature": 0, "stream": False}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    o = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    msg = o["choices"][0]["message"]
    return msg.get("content") or "", time.time() - t0


def run_budget(budget, bench, out_dir):
    kv = pathlib.Path(tempfile.mkdtemp(prefix="kv-", dir=out_dir))
    log_path = out_dir / f"server-{budget}.log"
    fh = open(log_path, "w")
    proc = subprocess.Popen(prod_command(PORT, kv, budget), cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{PORT}"
    try:
        for _ in range(900):
            if proc.poll() is not None:
                raise RuntimeError(f"server exited, see {log_path}")
            try:
                urllib.request.urlopen(base + "/v1/models", timeout=2).read()
                break
            except OSError:
                time.sleep(1)
        cases = ([("gsm8k", p, bench["gsm8k"]) for p in bench["gsm8k"].PROBLEMS] +
                 [("humaneval", p, bench["humaneval"]) for p in bench["humaneval"].PROBLEMS] +
                 [("vi", p, None) for p in VI_PROMPTS])
        rows = []
        mark = 0
        for kind, problem, mod in cases:
            prompt = mod.build_prompt(problem) if mod else problem
            content, secs = ask(base, prompt)
            text = log_path.read_text(errors="replace")
            new, mark = text[mark:], len(text)
            m = re.findall(r"thinking closed after (\d+) tokens( \(budget\))?", new)
            think = int(m[-1][0]) if m else None
            forced = bool(m and m[-1][1])
            passed = mod.grade(problem, content)[0] if mod else None
            rows.append({"kind": kind, "passed": passed, "think_tokens": think,
                         "forced": forced, "seconds": round(secs, 1),
                         "answer": content if kind == "vi" else None})
            print(f"N={budget} {kind} passed={passed} think={think} forced={forced} {secs:.1f}s", flush=True)
        return rows
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        fh.close()
        shutil.rmtree(kv, ignore_errors=True)
        time.sleep(30)  # let the previous server's wired memory drain before the next start


def summarize(budget, rows):
    think = [r["think_tokens"] for r in rows if r["think_tokens"] is not None]
    graded = [r for r in rows if r["passed"] is not None]
    return {
        "budget": budget,
        "gsm8k_pass": sum(1 for r in rows if r["kind"] == "gsm8k" and r["passed"]),
        "humaneval_pass": sum(1 for r in rows if r["kind"] == "humaneval" and r["passed"]),
        "graded": len(graded),
        "forced": sum(1 for r in rows if r["forced"]),
        "think_median": statistics.median(think) if think else None,
        "think_p90": sorted(think)[int(0.9 * (len(think) - 1))] if think else None,
        "think_max": max(think) if think else None,
        "seconds_total": round(sum(r["seconds"] for r in rows), 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budgets", default="0,8192,4096,2048,1024")
    ap.add_argument("--gateway-repo", default=str(pathlib.Path.home() / "Documents/GitHub/AI-Gateway-MLX"))
    ap.add_argument("--out", default=str(ROOT / "speed-bench/think-budget/runs"))
    args = ap.parse_args()
    sys.path.insert(0, str(pathlib.Path(args.gateway_repo) / "evals/bakeoff/benchmarks"))
    import gsm8k_mini
    import humaneval_mini
    bench = {"gsm8k": gsm8k_mini, "humaneval": humaneval_mini}
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for budget in [int(b) for b in args.budgets.split(",")]:
        rows = run_budget(budget, bench, out_dir)
        (out_dir / f"budget-{budget}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        summaries.append(summarize(budget, rows))
        print(json.dumps(summaries[-1]), flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=1))


if __name__ == "__main__":
    main()
