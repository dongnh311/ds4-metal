#!/usr/bin/env python3
"""Convergent agent trace: drive ds4-server through a coding task until it
FINISHES (tests pass -> the model stops calling tools and summarizes).

Improvements over agent_bench.py: an explicit inspect->edit->test->done workflow
in the system prompt, tool results that actually guide the fix, fix-detection on
the write_file content (validation must be present before tests pass), and full
transcript capture. Records per-turn throughput + the whole message trace.
"""
import json, time, re, subprocess, urllib.request

BASE = "http://127.0.0.1:8091/v1/chat/completions"
OUT_TRACE = "/private/tmp/claude-501/-Users-dongnh-orca-workspaces-ds4-metal-scallop/86acdcd4-3d34-4b3b-9383-6dfe77d07097/scratchpad/agent_trace.json"

TOOLS = [
 {"type":"function","function":{"name":"list_directory","description":"List files in a directory","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
 {"type":"function","function":{"name":"read_file","description":"Read a source file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
 {"type":"function","function":{"name":"write_file","description":"Overwrite a file with new content","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
 {"type":"function","function":{"name":"run_tests","description":"Run the test suite and return results","parameters":{"type":"object","properties":{},"required":[]}}},
]

LOGIN_PY = '''from flask import Flask, request, jsonify
from .db import get_user
from .auth import verify_password, issue_token

app = Flask(__name__)

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data["username"]      # BUG: KeyError if the field is missing
    password = data["password"]      # BUG: KeyError if the field is missing
    user = get_user(username)
    if user is None:
        return jsonify({"error": "no such user"}), 404
    if not verify_password(password, user.pw_hash):
        return jsonify({"error": "bad password"}), 401
    return jsonify({"token": issue_token(user.id)})
'''
LS = "app/\n  __init__.py\n  auth.py\n  db.py\n  login.py\ntests/\n  test_login.py\nrequirements.txt\n"

_state = {"fixed": False, "wrote_login": False}

def is_valid_fix(content):
    c = content.lower()
    # accept a fix that guards missing fields and returns 400
    guards = ("data.get(" in c or "not in data" in c or "if not data" in c
              or "keyerror" in c or ".get(" in c or "required" in c)
    return guards and "400" in c

def tool_result(name, args):
    if name == "list_directory":
        return LS
    if name == "read_file":
        p = str(args.get("path",""))
        if "login" in p: return LOGIN_PY
        if "test" in p:
            return ("def test_login_ok(client): ...\n"
                    "def test_login_missing_field(client):\n"
                    "    r = client.post('/login', json={'username':'a'})\n"
                    "    assert r.status_code == 400\n")
        return "# module source\n"
    if name == "write_file":
        p = str(args.get("path","")); content = str(args.get("content",""))
        if "login" in p:
            _state["wrote_login"] = True
            if is_valid_fix(content):
                _state["fixed"] = True
                return "wrote %d bytes to %s\n" % (len(content), p)
            return ("wrote %d bytes to %s (note: still no missing-field validation "
                    "returning HTTP 400)\n" % (len(content), p))
        return "wrote %d bytes to %s\n" % (len(content), p)
    if name == "run_tests":
        if _state["fixed"]:
            return "collected 2 items\ntests/test_login.py ..  [100%]\n2 passed in 0.19s\n"
        return ("collected 2 items\ntests/test_login.py .F  [100%]\n"
                "FAILED tests/test_login.py::test_login_missing_field - "
                "KeyError: 'password'\n1 passed, 1 failed in 0.21s\n")
    return "ok\n"

def swap_used():
    out = subprocess.run(["sysctl","-n","vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out); return float(m.group(1)) if m else 0.0

def post(messages):
    body = json.dumps({"model":"qwen3.8-flash-next","temperature":0.0,"max_tokens":1200,
                       "messages":messages,"tools":TOOLS,"tool_choice":"auto"}).encode()
    req = urllib.request.Request(BASE, data=body, headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(req, timeout=300) as r: resp=json.load(r)
    return resp, time.time()-t0

SYSTEM = ("You are an autonomous coding agent working in a Python (Flask) repo. "
          "Workflow: inspect the code, make the minimal edit that fixes the bug, "
          "then run the tests. Call exactly ONE tool per step. As soon as the "
          "tests all pass, STOP calling tools and reply with a one-paragraph "
          "summary of the fix. Do not keep inspecting after tests pass.")
USER = ("Bug: POST /login raises KeyError and returns HTTP 500 when the request "
        "JSON is missing 'username' or 'password'. Fix app/login.py so a missing "
        "field returns HTTP 400 with a clear error, and make the test suite pass.")

def main():
    messages=[{"role":"system","content":SYSTEM},{"role":"user","content":USER}]
    swap0=swap_used(); rows=[]; t0=time.time(); converged=False
    MAXT=20
    for turn in range(1,MAXT+1):
        resp,wall=post(messages); ch=resp["choices"][0]; msg=ch["message"]; usage=resp.get("usage",{})
        cached=usage.get("prompt_tokens_details",{}).get("cached_tokens",0)
        tc=msg.get("tool_calls")
        rows.append(dict(turn=turn,finish=ch.get("finish_reason"),wall=round(wall,2),
                         prompt=usage.get("prompt_tokens"),completion=usage.get("completion_tokens"),
                         cached=cached,tools=[c["function"]["name"] for c in tc] if tc else [],
                         swapM=round(swap_used()-swap0,1)))
        am={"role":"assistant","content":msg.get("content") or ""}
        if tc: am["tool_calls"]=tc
        messages.append(am)
        if tc:
            for c in tc:
                fn=c["function"]["name"]
                try: args=json.loads(c["function"]["arguments"] or "{}")
                except Exception: args={}
                res=tool_result(fn,args)
                messages.append({"role":"tool","tool_call_id":c.get("id","call_%d"%turn),"content":res})
            print(f"turn {turn}: {[c['function']['name'] for c in tc]} wall={wall:.2f}s ctx={usage.get('prompt_tokens')} cached={cached} fixed={_state['fixed']}",flush=True)
        else:
            converged=True
            print(f"turn {turn}: FINAL wall={wall:.2f}s tokens={usage.get('completion_tokens')} fixed={_state['fixed']}",flush=True)
            print("FINAL MESSAGE:", (msg.get('content') or '')[:400],flush=True)
            break
    total=time.time()-t0
    comp=sum(r["completion"] or 0 for r in rows)
    decs=[(r["completion"] or 0)/r["wall"] for r in rows if r["wall"] and r["completion"]]
    print("\n=== TRACE SUMMARY ===",flush=True)
    print(f"converged={converged} (fix_applied={_state['fixed']}) turns={len(rows)} total_wall={total:.1f}s completion={comp}",flush=True)
    if decs: print(f"tok/s(wall): mean={sum(decs)/len(decs):.1f} max={max(decs):.1f}",flush=True)
    print(f"final ctx={rows[-1]['prompt']} cached={rows[-1]['cached']} swap_growth={rows[-1]['swapM']}MiB",flush=True)
    json.dump({"converged":converged,"fix_applied":_state["fixed"],"rows":rows,
               "total_wall":total,"completion":comp,"transcript":messages},
              open(OUT_TRACE,"w"),indent=2)
    print("TRACEDONE",flush=True)

if __name__=="__main__": main()
