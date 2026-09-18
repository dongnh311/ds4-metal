#!/usr/bin/env python3
"""Sustained agentic-throughput bench for ds4-server (OpenAI-compatible).

Drives a realistic multi-turn coding-agent task through tool calls, feeding
back synthetic-but-realistic tool results so context grows turn over turn (like
OpenCode / a coding agent). Records per-turn TTFT, decode t/s (server-side),
tokens, prefix-cache reuse, wall latency, and swap stability.
"""
import json, time, re, subprocess, urllib.request, sys

BASE = "http://127.0.0.1:8091/v1/chat/completions"
SERR = "/private/tmp/claude-501/-Users-dongnh-orca-workspaces-ds4-metal-scallop/86acdcd4-3d34-4b3b-9383-6dfe77d07097/scratchpad/server_agent.err"

TOOLS = [
 {"type":"function","function":{"name":"list_directory","description":"List files in a directory","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
 {"type":"function","function":{"name":"read_file","description":"Read a source file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
 {"type":"function","function":{"name":"write_file","description":"Write/overwrite a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
 {"type":"function","function":{"name":"run_command","description":"Run a shell command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}},
]

# Realistic synthetic tool results (sized like real agent I/O: files ~1-2KB, output ~0.5KB)
LOGIN_PY = '''from flask import Flask, request, jsonify
from .db import get_user
from .auth import verify_password, issue_token

app = Flask(__name__)

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data["username"]
    password = data["password"]
    user = get_user(username)
    if user is None:
        return jsonify({"error": "no such user"}), 404
    if not verify_password(password, user.pw_hash):
        return jsonify({"error": "bad password"}), 401
    token = issue_token(user.id)
    return jsonify({"token": token})
''' * 2
LS = "auth.py\ndb.py\nlogin.py\n__init__.py\ntests/\nrequirements.txt\nREADME.md\n"
TEST_FAIL = "collected 7 items\ntests/test_login.py ...F..  [ 71%]\nFAILED tests/test_login.py::test_login_missing_field - KeyError: 'password'\n5 passed, 1 failed in 0.42s\n"
TEST_PASS = "collected 8 items\ntests/test_login.py ......  [100%]\n8 passed in 0.51s\n"

def tool_result(name, args):
    if name == "list_directory": return LS
    if name == "read_file": return LOGIN_PY if "login" in str(args) else "# (module source)\n"
    if name == "run_command":
        return TEST_PASS if "test" in str(args) and _state.get("fixed") else (TEST_FAIL if "test" in str(args) else "ok\n")
    if name == "write_file":
        _state["fixed"] = True
        return "wrote %d bytes\n" % len(str(args.get("content","")))
    return "ok\n"

_state = {}

def last_server_tps():
    try:
        txt = open(SERR, errors="ignore").read()
        m = re.findall(r"prefill: ([\d.]+) t/s, generation: ([\d.]+) t/s", txt)
        return (float(m[-1][0]), float(m[-1][1])) if m else (None, None)
    except Exception: return (None, None)

def swap_used():
    out = subprocess.run(["sysctl","-n","vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out); return float(m.group(1)) if m else 0.0

def post(messages):
    body = json.dumps({"model":"qwen3.8-flash-next","temperature":0.3,"max_tokens":400,
                       "messages":messages,"tools":TOOLS,"tool_choice":"auto"}).encode()
    req = urllib.request.Request(BASE, data=body, headers={"Content-Type":"application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.load(r)
    return resp, time.time()-t0

SYSTEM = ("You are a coding agent. Use the provided tools to inspect and fix the "
          "repository. Call one tool at a time. When the tests pass, reply with a "
          "short summary and do not call more tools.")
USER = ("There is a bug: POST /login crashes with KeyError when the request JSON "
        "is missing a field. Investigate the repo, add input validation to the "
        "login endpoint so missing fields return HTTP 400, and make the tests pass.")

def main():
    messages = [{"role":"system","content":SYSTEM},{"role":"user","content":USER}]
    swap0 = swap_used()
    rows = []
    max_turns = 12
    t_session = time.time()
    for turn in range(1, max_turns+1):
        resp, wall = post(messages)
        ch = resp["choices"][0]; msg = ch["message"]; usage = resp.get("usage",{})
        pf_tps, dec_tps = last_server_tps()
        cached = usage.get("prompt_tokens_details",{}).get("cached_tokens", 0)
        rows.append(dict(turn=turn, finish=ch.get("finish_reason"), wall=round(wall,2),
                         prompt=usage.get("prompt_tokens"), completion=usage.get("completion_tokens"),
                         cached=cached, pf_tps=pf_tps, dec_tps=dec_tps,
                         swapM=round(swap_used()-swap0,1)))
        tc = msg.get("tool_calls")
        # append assistant message
        am = {"role":"assistant","content":msg.get("content") or ""}
        if tc: am["tool_calls"] = tc
        messages.append(am)
        if tc:
            for c in tc:
                fn = c["function"]["name"]
                try: args = json.loads(c["function"]["arguments"] or "{}")
                except Exception: args = {}
                res = tool_result(fn, args)
                messages.append({"role":"tool","tool_call_id":c.get("id","call_%d"%turn),"content":res})
            print(f"turn {turn}: TOOL {[c['function']['name'] for c in tc]} wall={wall:.2f}s dec={dec_tps} cached={cached} ctx~{usage.get('prompt_tokens')}")
        else:
            print(f"turn {turn}: FINAL wall={wall:.2f}s dec={dec_tps} tokens={usage.get('completion_tokens')} content={ (msg.get('content') or '')[:80]!r}")
            break
    total = time.time()-t_session
    # aggregate
    decs = [r["dec_tps"] for r in rows if r["dec_tps"]]
    comp = sum(r["completion"] or 0 for r in rows)
    print("\n=== SUMMARY ===")
    print(f"turns={len(rows)} total_wall={total:.1f}s total_completion_tokens={comp}")
    if decs: print(f"decode t/s: min={min(decs):.1f} mean={sum(decs)/len(decs):.1f} max={max(decs):.1f}")
    print(f"final ctx (prompt_tokens last turn)={rows[-1]['prompt']} cached_last={rows[-1]['cached']}")
    print(f"swap growth over session={rows[-1]['swapM']} MiB")
    json.dump({"rows":rows,"total_wall":total,"total_completion":comp},
              open("/private/tmp/claude-501/-Users-dongnh-orca-workspaces-ds4-metal-scallop/86acdcd4-3d34-4b3b-9383-6dfe77d07097/scratchpad/agent_bench_result.json","w"), indent=2)

if __name__ == "__main__":
    main()
