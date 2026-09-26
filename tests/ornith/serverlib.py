"""Staging ds4-server helpers for the Ornith live tests (M3).

Port 18296 is the Ornith staging port, never a gateway port.  One model
process at a time: stop a server before starting the next.  A server is
stopped with SIGTERM and waited for; one that does not exit is reported and
left alone, never killed with SIGKILL (a killed Metal process can wedge the
GGUF until reboot).

check() is for structure: finish reasons, tool names and argument objects,
cached-token counts, byte identity between runs that must agree.
soft_check() is for what the model says: a miss prints WARN with the saved
request and the gate goes on; a WARN is investigated by sending that request
to llama-server on the same GGUF before calling it a ds4 bug.
"""
import json
import os
import pathlib
import subprocess
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
PORT = 18296
BASE = f"http://127.0.0.1:{PORT}"
BUDGET_MESSAGE = ("Considering the limited time by the user, I have to give the solution "
                  "based on the thinking directly now.")
WARNINGS = []


def model_path():
    path = os.environ.get("DS4_ORNITH_MODEL")
    if not path:
        raise SystemExit("set DS4_ORNITH_MODEL to the 23G ICE GGUF")
    return path


class Server:
    def __init__(self, log_path, args, env=None):
        self.log = pathlib.Path(log_path)
        self.fh = open(self.log, "a", encoding="utf-8")
        cmd = [str(ROOT / "ds4-server"), "--metal", "-m", model_path(), "--host", "127.0.0.1",
               "--port", str(PORT)] + [str(a) for a in args]
        self.fh.write("\n$ " + " ".join(cmd) + "\n")
        self.fh.flush()
        self.mark = self.log.stat().st_size
        self.proc = subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ, **(env or {})),
                                     stdout=self.fh, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 600
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"ds4-server exited {self.proc.returncode}; see {self.log}")
            try:
                with urllib.request.urlopen(BASE + "/v1/models", timeout=2) as resp:
                    self.models = json.load(resp)
                return
            except OSError:
                if time.monotonic() > deadline:
                    self.stop()
                    raise RuntimeError(f"ds4-server did not answer within 600 s; see {self.log}")
                time.sleep(1)

    def new_log(self):
        """Server output since the previous call (or since start)."""
        with open(self.log, "rb") as f:
            f.seek(self.mark)
            data = f.read()
        self.mark += len(data)
        return data.decode("utf-8", "replace")

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"ds4-server pid {self.proc.pid} ignored SIGTERM for 180 s; check "
                                   f"`ps -o pid,stat -p {self.proc.pid}` and never kill -9 a Metal process")
        self.fh.close()


def post(path, body, timeout=1800, save=None):
    """POST a JSON body; with save (a path stem) keep the request and the reply."""
    if save is not None:
        save = pathlib.Path(save)
        save.with_name(save.name + ".request.json").write_text(json.dumps(body, ensure_ascii=False, indent=1))
    req = urllib.request.Request(BASE + path, json.dumps(body).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.load(resp)
    if save is not None:
        save.with_name(save.name + ".reply.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return result


def check(cond, what):
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    if not cond:
        raise SystemExit(1)


def soft_check(cond, what, request=None):
    """What the model says: a miss is logged as WARN and does not fail the gate."""
    if cond:
        print("PASS " + what, flush=True)
        return
    note = f" (request: {request})" if request else ""
    print("WARN " + what + note, flush=True)
    WARNINGS.append(what + note)


def report(name):
    print(f"{name}: OK ({len(WARNINGS)} WARN)", flush=True)
