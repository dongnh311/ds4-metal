#!/usr/bin/env python3
"""Record llama.cpp references for the Ornith gate, or calibrate tolerances.

  llama_ref.py record    MODEL OUT_DIR [--cpu]
  llama_ref.py calibrate METAL_DIR CPU_DIR TOLERANCE_JSON

`record` starts llama-server on port 18190 (never a gateway port), tokenizes
every prompt with the server (no BOS, like ds4's --raw), and stores the
prompt ids, the generated text and, per greedy step, the selected id and
top-20 log-probabilities of the unsampled distribution.  A prompt with
"llama_ubatch" is recorded by its own server with that physical batch
(-ub): long_copy uses 1, because llama.cpp's batched Metal prefill moves its
contested step by more than the tolerance while its per-token path agrees
with the CPU backend.  --cpu runs the same model on llama.cpp's CPU backend
for calibration and skips the file-backed prompts.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import ornith_ref as r

PORT = 18190
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as resp:
        return json.loads(resp.read())


def wait_ready(proc):
    for _ in range(900):
        if proc.poll() is not None:
            sys.exit(f"llama-server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    sys.exit("llama-server did not become ready")


def record(model, out_dir, cpu):
    os.makedirs(out_dir, exist_ok=True)
    prompts = r.load_prompts(os.path.join(ROOT, "tests/ornith/prompts.json"))
    for ubatch, group in r.llama_server_groups(prompts, cpu):
        record_group(model, out_dir, cpu, ubatch, group)


def record_group(model, out_dir, cpu, ubatch, prompts):
    cmd = ["llama-server", "-m", model, "--host", "127.0.0.1", "--port", str(PORT),
           "-c", "16384", "-np", "1"]
    cmd += ["-ngl", "0", "--device", "none"] if cpu else ["-ngl", "99"]
    cmd += ["-ub", str(ubatch)] if ubatch is not None else []
    log_name = "llama-server.log" if ubatch is None else f"llama-server-ub{ubatch}.log"
    log = open(os.path.join(out_dir, log_name), "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_ready(proc)
        for p in prompts:
            text = r.prompt_text(p, ROOT)
            ids = post("/tokenize", {"content": text, "add_special": False})["tokens"]
            comp = post("/completion", {"prompt": ids, "n_predict": p["n_predict"], "temperature": 0.0,
                                        "n_probs": 20, "cache_prompt": False,
                                        "post_sampling_probs": False})
            with open(os.path.join(out_dir, p["name"] + ".json"), "w") as f:
                json.dump({"name": p["name"], "prompt_ids": ids, "content": comp.get("content", ""),
                           "steps": r.llama_steps(comp)}, f)
            print(f"{p['name']}: {len(ids)} prompt tokens, {len(comp['completion_probabilities'])} steps")
    finally:
        proc.terminate()
        proc.wait(timeout=120)


def calibrate(metal_dir, cpu_dir, out_path):
    metal, cpu = [], []
    for p in r.load_prompts(os.path.join(ROOT, "tests/ornith/prompts.json")):
        cpu_path = os.path.join(cpu_dir, p["name"] + ".json")
        if not os.path.exists(cpu_path):
            continue  # the long prompt is not run on the CPU backend
        with open(os.path.join(metal_dir, p["name"] + ".json")) as f:
            metal.append(json.load(f)["steps"])
        with open(cpu_path) as f:
            cpu.append(json.load(f)["steps"])
    cal = r.calibrate(metal, cpu)
    cal["prompts"] = len(metal)
    with open(out_path, "w") as f:
        json.dump(cal, f, indent=2)
    print(json.dumps(cal, indent=2))


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "record":
        record(sys.argv[2], sys.argv[3], "--cpu" in sys.argv[4:])
    elif len(sys.argv) == 5 and sys.argv[1] == "calibrate":
        calibrate(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit(__doc__)
