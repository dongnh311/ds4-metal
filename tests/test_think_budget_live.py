"""Live thinking-budget check; needs the Qwen3.8 GGUF, its PLE sidecar and a free GPU.

python3 tests/test_think_budget_live.py --model MODEL --ple PLE
"""

import argparse
import json
import os
import pathlib
import re
import socket
import subprocess
import tempfile
import time
import urllib.request

PROMPT = "Prove that there are infinitely many prime numbers, then list the first ten primes."
MESSAGE = ("Considering the limited time by the user, I have to give the solution "
           "based on the thinking directly now.")
BUDGET = 64


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Server:
    def __init__(self, root, args, extra_env=None):
        self.out = pathlib.Path(tempfile.mkdtemp(prefix="ds4-think-budget-"))
        self.log = self.out / "server.log"
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        env = dict(os.environ, DS4_QWEN4_STREAM_FULL_LAYERS="32", DS4_QWEN4_PLE_PREFETCH_FULL="0")
        env.update(extra_env or {})
        cmd = [str(root / "ds4-server"), "--metal", "-m", args.model, "--ple", args.ple,
               "-c", "32768", "--prefill-chunk", "2048", "--mtp", "--ssd-streaming",
               "--ssd-streaming-cache-experts", "6GB", "--think-budget", str(BUDGET),
               "--kv-disk-dir", str(self.out / "kv"), "--host", "127.0.0.1",
               "--port", str(self.port)]
        self.fh = open(self.log, "w")
        self.proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=self.fh, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 600
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited {self.proc.returncode}, see {self.log}")
            try:
                urllib.request.urlopen(self.base + "/v1/models", timeout=2).read()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("startup timeout")
                time.sleep(1)
        self.mark = 0

    def new_log(self):
        text = self.log.read_text(errors="replace")
        chunk, self.mark = text[self.mark:], len(text)
        return chunk

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.fh.close()


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=1800)


def chat(base, messages, stream=False, **extra):
    body = {"model": "ds4", "messages": messages, "max_tokens": 400, "temperature": 0,
            "stream": stream}
    body.update(extra)
    resp = post(base, "/v1/chat/completions", body)
    if not stream:
        o = json.loads(resp.read())
        msg = o["choices"][0]["message"]
        return (msg.get("reasoning_content") or "", msg.get("content") or "",
                o["choices"][0].get("finish_reason"))
    reasoning, content, finish = [], [], None
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        choices = json.loads(line[6:]).get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        reasoning.append(delta.get("reasoning_content") or "")
        content.append(delta.get("content") or "")
        finish = choice.get("finish_reason") or finish
    return "".join(reasoning), "".join(content), finish


def anthropic(base, budget):
    body = {"model": "ds4", "max_tokens": 400, "temperature": 0,
            "thinking": {"type": "enabled", "budget_tokens": budget},
            "messages": [{"role": "user", "content": PROMPT}]}
    o = json.loads(post(base, "/v1/messages", body).read())
    thinking = "".join(b.get("thinking", "") for b in o["content"] if b["type"] == "thinking")
    text = "".join(b.get("text", "") for b in o["content"] if b["type"] == "text")
    return thinking, text


def check(cond, what):
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    if not cond:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ple", required=True)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    user = [{"role": "user", "content": PROMPT}]

    srv = Server(root, args)
    try:
        # 0. warm-up: the first request after a server start can decode differently
        chat(srv.base, user)
        srv.new_log()
        # 1. forced close, non-stream
        r1 = chat(srv.base, user)
        log = srv.new_log()
        check(f"thinking budget reached {BUDGET} tokens" in log, "1 budget log line")
        check(MESSAGE in r1[0], "1 forced sentence in reasoning")
        check(r1[1].strip() != "", "1 answer after the forced close")
        # 2. stream equals non-stream
        r2 = chat(srv.base, user, stream=True)
        srv.new_log()
        norm = lambda r: (r[0].strip(), r[1].strip())
        check(norm(r2) == norm(r1), "2 stream text == non-stream text (whitespace-trimmed)")
        # 3. next turn continues from the live state (prefix cached)
        turn2 = user + [{"role": "assistant", "content": r2[1]},
                        {"role": "user", "content": "Now list the next five primes."}]
        chat(srv.base, turn2)
        log = srv.new_log()
        starts = re.findall(r"chat ctx=(\d+)\.\.\d+:\d+(?: [A-Z_]+)* prompt start", log)
        check(bool(starts) and int(starts[0]) > 0 and "live kv cache miss" not in log,
              "3 next turn reuses the live prefix")
        # 4. request budget lower than the server cap (Anthropic)
        thinking, text = anthropic(srv.base, 32)
        log = srv.new_log()
        check("thinking budget reached 32 tokens" in log, "4 anthropic budget_tokens lowers the cap")
        check(MESSAGE in thinking and text.strip() != "", "4 anthropic forced close + answer")
        # 5. max_tokens below the budget: plain length finish, no forced close
        r5 = chat(srv.base, user, max_tokens=40)
        log = srv.new_log()
        check(r5[2] == "length" and "thinking budget reached" not in log, "5 max_tokens wins")
        # 6. thinking disabled: the budget changes nothing
        a = chat(srv.base, user, think=False, thinking_budget=16)
        b = chat(srv.base, user, think=False)
        log = srv.new_log()
        check(a == b and "thinking budget reached" not in log, "6 no-think request unaffected")
    finally:
        srv.stop()

    # 7. MTP speculation off: the forced close still works
    srv = Server(root, args, {"DS4_MTP_SPEC_DISABLE": "1"})
    try:
        r7 = chat(srv.base, user)
        log = srv.new_log()
        check(f"thinking budget reached {BUDGET} tokens" in log and MESSAGE in r7[0]
              and r7[1].strip() != "", "7 forced close with MTP off")
    finally:
        srv.stop()
    print("think-budget live: OK")


if __name__ == "__main__":
    main()
