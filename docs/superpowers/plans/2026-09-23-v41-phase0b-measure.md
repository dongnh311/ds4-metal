# V4.1 Phase 0(b) — Measure, Then Set the Target — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure DeepSeek-V4.1-Flash Q2 on the M5 Pro 64 GB box with the existing
`--ssd-streaming` path unchanged, and turn the numbers into a measured roofline,
a routing-locality curve and a tok/s target the user picks (decision "C").

**Architecture:**
- **Tooling** (Python, stdlib only) lives under `speed-bench/`:
  - a machine guard and wired-memory sampler;
  - the Qwen3.8 regression gate;
  - a GGUF per-token byte accountant;
  - an offline router-locality simulator;
  - a Phase-0 run driver and report.
- **Two diagnostics** go into the existing V4.1 decode step (`ds41_graph_step` in
  `ds4.c`): a router log and a per-token decode profile. Both are env-gated and
  off by default.
- **GPU- and disk-heavy work** is grouped into three gated tasks (8–10). Each one
  runs only after the user says the machine is free.

**Tech Stack:** C / Objective-C (ds4 runtime, Metal), Python 3.9 stdlib, `ds4-bench`,
`download_model.sh`, `vm_stat`, `sysctl`.

**Spec:** `docs/V41_64GB_BUILD.md` (v2, commit f395b24), sections §2, §3.1, §4, §7, §9, §10.

## Global Constraints

- **GPU- or disk-heavy steps (Tasks 8, 9, 10) need the user's go-ahead first.** The
  GPU and SSD are shared with other sessions' benchmarks and the PROD gateway.
  Never stop the gateway or other sessions' processes yourself.
- **Protect Qwen.** Never change Qwen-specific code. New runtime diagnostics must be
  env-gated and off by default. With the env unset, generated code paths must be
  byte-identical.
- **Qwen gate before merging.** Every commit touching `ds4.c`, `ds4_metal.m` or
  `ds4_gpu.h` must pass the Qwen fast tier before it reaches `develop`. Runs are
  batched into Task 8. If the Qwen gate fails, STOP the DS4.1 work
  (spec §4) and report.
- **Data locations:**
  - Model files: `~/orca/workspaces/ds4-metal-data/gguf/`.
  - Raw run output: `~/orca/workspaces/ds4-metal-data/v41-phase0/<YYYYMMDD>/`.
  - Git gets only source and small summaries (CSV / JSON / Markdown).
- **Measure wired memory with `vm_stat`** ("Pages wired down" × page size),
  never RSS (spec §3.1).
- **Model:** `DeepSeek-V4.1-Flash-Q2.gguf`, 365 713 686 528 bytes, SHA-256
  `1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42`
  (pinned in `download_model.sh`).
- **Language:** code, comments, docs and commit messages in English.
- **Commits:** every message ends with
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- **Branch:** work on `feature/ds4.1-flash`. Do not push unless the user asks.
- **Python tests:** `python3 -m unittest discover -s speed-bench/tests -v`
  (system python3 3.9, stdlib only).

## Review Focus

1. **Failed or killed run.** A bench run that fails or is killed must leave no
   result file, so a rerun redoes it. Task 7 tests a fake failing `ds4-bench`.
2. **Swap.** Swap during a run makes its speed and wired numbers meaningless. The
   report must mark rows with `swap_delta_mib > 256` as swapped. Tested in Task 7.
3. **Another ds4 process mid-run.** A ds4 process that starts during a run
   contaminates it. A post-run guard marks the row `contaminated`. Tested in
   Task 7 with an injected `pgrep`.
4. **Registry change.** If the gateway registry command changes between baseline
   and check, the Qwen gate must fail with "registry command changed" and not
   compare the wrong model. Tested in Task 2.
5. **Unwritable log path.** `DS4_V41_ROUTER_LOG` pointing at an unwritable path
   must disable the logger with one error while decode continues. Tested in
   Task 4.

---

### Task 1: Machine guard and wired-memory sampler

**Files:**
- Create: `speed-bench/lib/machine.py`
- Create: `speed-bench/lib/wired.py`
- Test: `speed-bench/tests/test_machine.py`, `speed-bench/tests/test_wired.py`

**Interfaces:**
- Produces (used by Tasks 2 and 7):
  - `machine.ds4_running(pgrep=None) -> str`: `pgrep -fl` lines for ds4
    binaries, `''` when none. `pgrep` is an injectable callable returning text.
  - `machine.parse_swap_used_mib(text) -> float`
  - `machine.swap_used_mib() -> float`
  - `wired.GIB = 1024**3`
  - `wired.parse_vm_stat(text) -> int` (wired bytes)
  - `wired.summarize(samples) -> {"steady_gib", "peak_gib", "n"}`
  - `wired.WiredSampler(interval=0.5, read=None)`: a context manager with
    `.samples` and `.summary()`.

- [ ] **Step 1: Write the failing tests**

`speed-bench/tests/test_machine.py`:
```python
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import machine  # noqa: E402


class MachineTest(unittest.TestCase):
    def test_ds4_running_reports_lines(self):
        self.assertEqual(machine.ds4_running(pgrep=lambda: "123 ds4-server --metal\n"),
                         "123 ds4-server --metal")

    def test_ds4_running_empty_when_free(self):
        self.assertEqual(machine.ds4_running(pgrep=lambda: ""), "")

    def test_parse_swap_used(self):
        text = "vm.swapusage: total = 2048.00M  used = 938.19M  free = 1109.81M  (encrypted)"
        self.assertEqual(machine.parse_swap_used_mib(text), 938.19)

    def test_parse_swap_rejects_garbage(self):
        with self.assertRaises(ValueError):
            machine.parse_swap_used_mib("nothing")


if __name__ == "__main__":
    unittest.main()
```

`speed-bench/tests/test_wired.py`:
```python
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import wired  # noqa: E402

SAMPLE = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                    10623.
Pages wired down:                             262144.
Pages purgeable:                                9311.
"""


class WiredTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(wired.parse_vm_stat(SAMPLE), 262144 * 16384)

    def test_parse_rejects_garbage(self):
        with self.assertRaises(ValueError):
            wired.parse_vm_stat("nothing here")

    def test_summary_uses_median_of_second_half(self):
        g = wired.GIB
        s = wired.summarize([1 * g, 9 * g, 4 * g, 5 * g, 6 * g])
        self.assertEqual(s["steady_gib"], 5.0)
        self.assertEqual(s["peak_gib"], 9.0)
        self.assertEqual(s["n"], 5)

    def test_summary_rejects_empty(self):
        with self.assertRaises(ValueError):
            wired.summarize([])

    def test_sampler_collects(self):
        with wired.WiredSampler(interval=0.01, read=lambda: SAMPLE) as ws:
            time.sleep(0.05)
        self.assertGreaterEqual(len(ws.samples), 2)
        self.assertEqual(ws.summary()["peak_gib"], 262144 * 16384 / wired.GIB)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest discover -s speed-bench/tests -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'machine'` (and `'wired'`).

- [ ] **Step 3: Write the minimal implementation**

`speed-bench/lib/machine.py`:
```python
"""Guards for benchmarks on the shared 64 GB box (docs/V41_64GB_BUILD.md §4)."""
import re
import subprocess

DS4_PATTERN = r"(^|/)ds4(-server|-agent|-bench|-eval)?( |$)"
_SWAP_RE = re.compile(r"used = ([\d.]+)M")


def ds4_running(pgrep=None):
    """pgrep -fl lines for running ds4 binaries; '' when the machine is free."""
    if pgrep is None:
        out = subprocess.run(["pgrep", "-fl", DS4_PATTERN],
                             capture_output=True, text=True).stdout
    else:
        out = pgrep()
    return out.strip()


def parse_swap_used_mib(text):
    m = _SWAP_RE.search(text)
    if not m:
        raise ValueError("unexpected vm.swapusage output")
    return float(m.group(1))


def swap_used_mib():
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True,
                         text=True, check=True).stdout
    return parse_swap_used_mib(out)
```

`speed-bench/lib/wired.py`:
```python
"""System-wide wired memory from vm_stat.

RSS and phys_footprint miss Metal residency; the wired page count does not
(docs/V41_64GB_BUILD.md §3.1)."""
import re
import statistics
import subprocess
import threading

GIB = 1024 ** 3
_PAGE_RE = re.compile(r"page size of (\d+) bytes")
_WIRED_RE = re.compile(r"^Pages wired down:\s+(\d+)\.", re.M)


def parse_vm_stat(text):
    """Wired bytes from one vm_stat output."""
    page = _PAGE_RE.search(text)
    pages = _WIRED_RE.search(text)
    if not page or not pages:
        raise ValueError("vm_stat output lacks the page size or wired pages")
    return int(page.group(1)) * int(pages.group(1))


def summarize(samples):
    """Steady state = median of the second half of the samples; peak = max. GiB."""
    if not samples:
        raise ValueError("no wired samples")
    tail = samples[len(samples) // 2:]
    return {"steady_gib": statistics.median(tail) / GIB,
            "peak_gib": max(samples) / GIB,
            "n": len(samples)}


def _read_vm_stat():
    return subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout


class WiredSampler:
    """Samples vm_stat every `interval` seconds while inside a `with` block."""

    def __init__(self, interval=0.5, read=None):
        self.interval = interval
        self.read = read or _read_vm_stat
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(parse_vm_stat(self.read()))
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        return False

    def summary(self):
        return summarize(self.samples)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest discover -s speed-bench/tests -v`
Expected: 9 tests, all `ok`.

- [ ] **Step 5: Smoke against the real machine** (read-only, no GPU)

Run: `python3 -c 'import sys; sys.path.insert(0, "speed-bench/lib"); import machine, wired; print(machine.ds4_running() or "free", machine.swap_used_mib(), wired.parse_vm_stat(wired._read_vm_stat()) / wired.GIB)'`
Expected: prints `free` or a pgrep line, a swap number in MiB and a wired GiB number. No exception.

- [ ] **Step 6: Commit**

```bash
git add speed-bench/lib/machine.py speed-bench/lib/wired.py speed-bench/tests/test_machine.py speed-bench/tests/test_wired.py
git commit -m "speed-bench: machine guard and vm_stat wired-memory sampler

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Qwen3.8 regression gate harness

**Files:**
- Create: `speed-bench/qwen-regression/qwen_gate.py`
- Create: `speed-bench/qwen-regression/run.sh`
- Create: `speed-bench/qwen-regression/README.md`
- Modify: `.gitignore`: add `/speed-bench/qwen-regression/last-*/` and `/speed-bench/qwen-regression/baseline/server.log`
- Test: `speed-bench/tests/test_qwen_gate.py`

**Interfaces:**
- Consumes: `machine.ds4_running()`, `wired.WiredSampler` (Task 1).
- Produces: the CLIs `qwen_gate.py record --out DIR [--full]` and
  `qwen_gate.py check --bin DIR --baseline DIR --out DIR [--full]`, and
  `run.sh fast|full [BASELINE_DIR]`. Each run writes `<out>/{vi.txt,code.txt,result.json,server.log}`.
  `result.json` keys:
  - always: `registry_command`, `replies`;
  - `--full` only: `tps`, `tps_median`, `wired`, `needle_prompt_tokens`, `needle_hit`.

The prompts and request body match `deploy-ai-gateway.sh smoke` byte for byte, so its
references stay interchangeable. The server command comes from the gateway registry
(`~/.local/ai-gateway/runtime-registry.json`, the enabled `ds4` runtime). That makes the
gate test the exact PROD configuration.

- [ ] **Step 1: Write the failing tests**

`speed-bench/tests/test_qwen_gate.py`:
```python
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "qwen-regression"))
import qwen_gate  # noqa: E402

ENTRY = {"enabled": True, "process_cwd": "/prod/ds4-metal",
         "process_command": ["/usr/bin/env", "DS4_X=1", "/prod/ds4-metal/ds4-server", "--metal",
                             "-m", "/m.gguf", "--kv-disk-dir", "/prod/kv", "--port", "18086"]}


def registry(tmp, entry=ENTRY):
    path = os.path.join(tmp, "registry.json")
    with open(path, "w") as fp:
        json.dump({"models": {"a": {"runtimes": {"llama": {"enabled": True}}},
                              "b": {"runtimes": {"ds4": entry}}}}, fp)
    return path


class RegistryTest(unittest.TestCase):
    def test_retargets_binary_port_and_kv(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd, cwd = qwen_gate.registry_command(registry(tmp), "/work", 1234, "/scratch/kv")
        self.assertIn("/work/ds4-server", cmd)
        self.assertEqual(cmd[cmd.index("--port") + 1], "1234")
        self.assertEqual(cmd[cmd.index("--kv-disk-dir") + 1], "/scratch/kv")
        self.assertEqual(cwd, "/work")

    def test_prod_binary_when_no_bin(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd, cwd = qwen_gate.registry_command(registry(tmp), None, 1, "/kv")
        self.assertIn("/prod/ds4-metal/ds4-server", cmd)
        self.assertEqual(cwd, "/prod/ds4-metal")

    def test_no_enabled_ds4_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                qwen_gate.registry_command(registry(tmp, {"enabled": False}), None, 1, "/kv")


class CompareTest(unittest.TestCase):
    def test_differs_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "vi.txt"), "w", encoding="utf-8") as fp:
                fp.write("same")
            self.assertEqual(qwen_gate.compare_replies(tmp, {"vi": "same"}), [])
            self.assertEqual(qwen_gate.compare_replies(tmp, {"vi": "other"}), ["vi"])
            self.assertEqual(qwen_gate.compare_replies(tmp, {"code": "x"}), ["code"])


class NeedleTest(unittest.TestCase):
    def test_needle_in_middle_question_at_end(self):
        filler = "".join(f"line {i}\n" for i in range(100))
        text = qwen_gate.make_needle(filler, 400)
        self.assertIn("\n" + qwen_gate.NEEDLE + "\n", text)
        self.assertTrue(text.endswith(qwen_gate.QUESTION))
        at = text.index(qwen_gate.NEEDLE)
        self.assertTrue(100 < at < 300)

    def test_short_filler_rejected(self):
        with self.assertRaises(ValueError):
            qwen_gate.make_needle("tiny", 400)


class EvaluateTest(unittest.TestCase):
    BASE = {"registry_command": ["a"], "tps_median": 10.0, "wired": {"steady_gib": 40.0}}

    def test_pass(self):
        cur = {"registry_command": ["a"], "tps_median": 9.8, "wired": {"steady_gib": 40.4},
               "needle_hit": True}
        self.assertEqual(qwen_gate.evaluate(self.BASE, cur), [])

    def test_speed_wired_needle_failures(self):
        cur = {"registry_command": ["a"], "tps_median": 9.6, "wired": {"steady_gib": 40.6},
               "needle_hit": False}
        failures = qwen_gate.evaluate(self.BASE, cur)
        self.assertEqual(len(failures), 3)

    def test_fast_tier_only_checks_command(self):
        self.assertEqual(qwen_gate.evaluate(self.BASE, {"registry_command": ["a"]}), [])

    def test_registry_command_changed(self):
        failures = qwen_gate.evaluate(self.BASE, {"registry_command": ["b"]})
        self.assertEqual(len(failures), 1)
        self.assertIn("registry command changed", failures[0])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest discover -s speed-bench/tests -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qwen_gate'`.

- [ ] **Step 3: Write the implementation**

`speed-bench/qwen-regression/qwen_gate.py`:
```python
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
```

`speed-bench/qwen-regression/run.sh`:
```bash
#!/bin/bash
# Qwen3.8 regression gate (docs/V41_64GB_BUILD.md §4).
#   run.sh fast [BASELINE_DIR]   unit kernels + byte-identical PROD replies
#   run.sh full [BASELINE_DIR]   fast + decode t/s, steady wired GiB, long needle
# Needs the machine free: no other ds4 process may run.
set -euo pipefail
tier=${1:?usage: run.sh fast|full [BASELINE_DIR]}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
baseline=${2:-$here/baseline}
case "$tier" in
    fast) full_flag="" ;;
    full) full_flag="--full" ;;
    *) echo "usage: run.sh fast|full [BASELINE_DIR]" >&2; exit 2 ;;
esac
cd "$root"
make ds4-server test-qwen4-kernels test-qwen4-q2
python3 "$here/qwen_gate.py" check --bin "$root" --baseline "$baseline" \
    --out "$here/last-$tier" $full_flag
```

`speed-bench/qwen-regression/README.md`:
````markdown
# Qwen3.8 regression gate

Spec: `docs/V41_64GB_BUILD.md` §4. It protects the PROD Qwen3.8 configuration,
read from the gateway registry (`~/.local/ai-gateway/runtime-registry.json`).

| Tier | When | Checks |
| --- | --- | --- |
| fast | every commit touching shared runtime code | `make test-qwen4-kernels test-qwen4-q2`; vi/code replies byte-identical to `baseline/`; registry command unchanged |
| full | end of every phase, before merging to `develop` | fast + decode t/s median of 3 ≥ 97 % of baseline, steady wired ≤ baseline + 0.5 GiB (`vm_stat`), long-context needle found |

Both tiers need the machine free: the PROD gateway's ds4 backend and every
other ds4 process must be stopped, and the user must agree to the run.

```sh
speed-bench/qwen-regression/run.sh fast
speed-bench/qwen-regression/run.sh full
# re-record the reference from the PROD binary (only after the PROD deploy changes):
python3 speed-bench/qwen-regression/qwen_gate.py record --out speed-bench/qwen-regression/baseline --full
```

A failure stops DS4.1 work until it is understood (spec §4).
````

Append to `.gitignore`:
```
/speed-bench/qwen-regression/last-*/
/speed-bench/qwen-regression/baseline/server.log
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `chmod +x speed-bench/qwen-regression/run.sh speed-bench/qwen-regression/qwen_gate.py && python3 -m unittest discover -s speed-bench/tests -v`
Expected: all tests `ok` (9 from Task 1 + 10 new).

- [ ] **Step 5: Check the CLI without a model** (no GPU)

Run: `python3 speed-bench/qwen-regression/qwen_gate.py --help && python3 speed-bench/qwen-regression/qwen_gate.py check --help`
Expected: usage text for `record` and `check`. No exception.

- [ ] **Step 6: Commit**

```bash
git add .gitignore speed-bench/qwen-regression/ speed-bench/tests/test_qwen_gate.py
git commit -m "speed-bench: Qwen3.8 regression gate (fast and full tiers)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: GGUF per-token byte accountant

**Files:**
- Create: `speed-bench/v41/gguf_bytes.py`
- Test: `speed-bench/tests/test_gguf_bytes.py`

**Interfaces:**
- Produces (used by Tasks 6, 7, 9 and 11):
  - `read_header(path) -> (meta: dict, tensors: [(name, dims, type, offset)], data_start: int)`
  - `tensor_sizes(tensors, data_start, file_bytes) -> {name: bytes}`
  - `role(name) -> str`
  - `account(meta, tensors, data_start, file_bytes) -> dict` with keys:
    `n_layer`, `n_expert`, `k`, `file_bytes`, `n_tensors`, `roles`,
    `engram_tables`, `resident_floor`, `per_token{resident, embedding_row, routed,
    engram_rows, ram_total}`, `expert_unit_bytes{str(layer): bytes}`.
  - `roofline(acct, gbps) -> {resident_ms, embedding_row_ms, routed_ms, total_ms, tps_ceiling}`
  - CLI: `gguf_bytes.py MODEL [--gbps 290] [--out bytes.json]`.

Assumptions (from the spec §2.1):
- Every non-routed, non-embedding, non-disk tensor is read once per decoded token.
- `token_embd` contributes one row per token.
- Routed experts contribute `k / n_expert` of their bytes.
- Engram disk tables contribute `DS4_ENGRAM_COLS` rows of `DS4_ENGRAM_ROW_BYTES`
  each (`ds4_engram.h`).

The Phase-0 GPU-busy measurement (Task 10) checks these assumptions.

- [ ] **Step 1: Write the failing tests**

`speed-bench/tests/test_gguf_bytes.py`:
```python
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v41"))
import gguf_bytes  # noqa: E402


def w_str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def kv_u32(key, value):
    return w_str(key) + struct.pack("<II", 4, value)


def kv_str(key, value):
    return w_str(key) + struct.pack("<I", 8) + w_str(value)


def kv_str_array(key, items):
    return (w_str(key) + struct.pack("<I", 9) + struct.pack("<IQ", 8, len(items))
            + b"".join(w_str(i) for i in items))


def tinfo(name, dims, ttype, offset):
    return (w_str(name) + struct.pack("<I", len(dims))
            + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", ttype, offset))


# name, dims, type, offset: sizes follow from consecutive offsets; 1888 = end of data
TENSORS = [
    ("token_embd.weight", [8, 10], 1, 0),                 # 160
    ("blk.0.attn_q_a.weight", [8, 8], 8, 160),             # 96
    ("blk.0.ffn_gate_exps.weight", [8, 4, 4], 16, 256),    # 128
    ("blk.0.ffn_up_exps.weight", [8, 4, 4], 16, 384),      # 128
    ("blk.0.ffn_down_exps.weight", [4, 8, 4], 10, 512),    # 256
    ("blk.1.engram_embd.weight", [264, 100], 24, 768),     # 1024
    ("output.weight", [8, 10], 8, 1792),                   # 96
]
DATA_BYTES = 1888


def write_gguf(path):
    kvs = [kv_str("general.architecture", "deepseek41"), kv_u32("general.alignment", 32),
           kv_u32("deepseek41.num_hidden_layers", 2), kv_u32("deepseek41.n_routed_experts", 4),
           kv_u32("deepseek41.num_experts_per_tok", 2),
           kv_str_array("tokenizer.ggml.tokens", ["a", "b", "c"])]
    head = (b"GGUF" + struct.pack("<IQQ", 3, len(TENSORS), len(kvs)) + b"".join(kvs)
            + b"".join(tinfo(*t) for t in TENSORS))
    pad = (-len(head)) % 32
    with open(path, "wb") as fp:
        fp.write(head + b"\0" * pad + b"\0" * DATA_BYTES)
    return len(head) + pad


class HeaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "m.gguf")
        self.data_start = write_gguf(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_header(self):
        meta, tensors, data_start = gguf_bytes.read_header(self.path)
        self.assertEqual(meta["deepseek41.n_routed_experts"], 4)
        self.assertEqual(meta["tokenizer.ggml.tokens"], ["a", "b", "c"])
        self.assertEqual(len(tensors), 7)
        self.assertEqual(data_start, self.data_start)

    def test_account(self):
        meta, tensors, data_start = gguf_bytes.read_header(self.path)
        acct = gguf_bytes.account(meta, tensors, data_start, os.path.getsize(self.path))
        self.assertEqual(acct["roles"], {"embedding": 160, "attention": 96, "routed": 512,
                                         "engram_disk": 1024, "output": 96})
        self.assertEqual(acct["per_token"], {"resident": 192, "embedding_row": 16, "routed": 256,
                                             "engram_rows": 24 * 264, "ram_total": 464})
        self.assertEqual(acct["resident_floor"], 352)
        self.assertEqual(acct["engram_tables"], 1)
        self.assertEqual(acct["expert_unit_bytes"], {"0": 128})
        self.assertEqual((acct["n_layer"], acct["n_expert"], acct["k"], acct["n_tensors"]),
                         (2, 4, 2, 7))

    def test_rejects_non_gguf(self):
        bad = os.path.join(self.tmp.name, "bad.bin")
        with open(bad, "wb") as fp:
            fp.write(b"NOPE" + b"\0" * 64)
        with self.assertRaises(ValueError):
            gguf_bytes.read_header(bad)


class RoleTest(unittest.TestCase):
    def test_roles(self):
        cases = {
            "blk.3.indexer.attn_q_b.weight": "indexer",
            "blk.3.attn_compressor_norm.weight": "compressor",
            "blk.3.hc_attn_fn.weight": "hc",
            "blk.3.ffn_gate_shexp.weight": "shared",
            "blk.3.ffn_exp_probs_b.bias": "router",
            "blk.3.ffn_gate_inp.weight": "router",
            "blk.3.engram_kv.weight": "engram",
            "blk.3.attn_q_a_norm.weight": "attention",
            "blk.3.ffn_norm.weight": "norm",
            "output_norm.weight": "output",
        }
        for name, want in cases.items():
            self.assertEqual(gguf_bytes.role(name), want, name)


class RooflineTest(unittest.TestCase):
    def test_one_ms_at_290(self):
        acct = {"per_token": {"resident": 290_000_000, "embedding_row": 0, "routed": 0}}
        r = gguf_bytes.roofline(acct, 290.0)
        self.assertAlmostEqual(r["total_ms"], 1.0)
        self.assertAlmostEqual(r["tps_ceiling"], 1000.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest discover -s speed-bench/tests -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gguf_bytes'`.

- [ ] **Step 3: Write the implementation**

`speed-bench/v41/gguf_bytes.py`:
```python
#!/usr/bin/env python3
"""Per-token byte accounting for a DeepSeek-V4.1 GGUF (docs/V41_64GB_BUILD.md §7 step 5).

Reads only the GGUF header. Tensor sizes come from consecutive data offsets
(the last tensor runs to the end of the file), so alignment padding is counted.
"""
import argparse
import json
import os
import re
import struct
import sys

ENGRAM_COLS = 24          # ds4_engram.h DS4_ENGRAM_COLS: rows per table per token
ENGRAM_ROW_BYTES = 264    # ds4_engram.h DS4_ENGRAM_ROW_BYTES
GIB = 1024 ** 3
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
           10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
ROUTED = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


class _Reader:
    def __init__(self, fp):
        self.fp = fp

    def take(self, fmt):
        size = struct.calcsize(fmt)
        data = self.fp.read(size)
        if len(data) != size:
            raise ValueError("truncated GGUF header")
        return struct.unpack(fmt, data)[0]

    def string(self):
        n = self.take("<Q")
        data = self.fp.read(n)
        if len(data) != n:
            raise ValueError("truncated GGUF string")
        return data.decode("utf-8", errors="replace")

    def value(self, vtype):
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            etype, n = self.take("<I"), self.take("<Q")
            return [self.value(etype) for _ in range(n)]
        if vtype not in _SCALAR:
            raise ValueError(f"unknown GGUF value type {vtype}")
        return self.take(_SCALAR[vtype])


def read_header(path):
    """(metadata, [(name, dims, type, offset)], data_start)."""
    with open(path, "rb") as fp:
        if fp.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        r = _Reader(fp)
        version = r.take("<I")
        if version != 3:
            raise ValueError(f"{path}: GGUF version {version}, expected 3")
        n_tensors, n_kv = r.take("<Q"), r.take("<Q")
        meta = {}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = r.value(r.take("<I"))
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.take("<Q") for _ in range(r.take("<I"))]
            ttype, offset = r.take("<I"), r.take("<Q")
            tensors.append((name, dims, ttype, offset))
        align = meta.get("general.alignment", 32)
        data_start = (fp.tell() + align - 1) // align * align
    return meta, tensors, data_start


def tensor_sizes(tensors, data_start, file_bytes):
    """name -> bytes from consecutive offsets; the last tensor ends at EOF."""
    ordered = sorted(tensors, key=lambda t: t[3])
    sizes = {}
    for i, (name, _, _, offset) in enumerate(ordered):
        end = ordered[i + 1][3] if i + 1 < len(ordered) else file_bytes - data_start
        sizes[name] = end - offset
    return sizes


def role(name):
    if name == "token_embd.weight":
        return "embedding"
    if ROUTED.match(name):
        return "routed"
    if name.endswith(".engram_embd.weight"):
        return "engram_disk"
    if name.startswith("output"):
        return "output"
    for key, label in (("_shexp", "shared"), ("ffn_gate_inp", "router"), ("exp_probs", "router"),
                       ("indexer", "indexer"), ("compressor", "compressor"), (".hc_", "hc"),
                       (".engram_", "engram"), (".attn", "attention"), ("norm", "norm")):
        if key in name:
            return label
    return "other"


def account(meta, tensors, data_start, file_bytes):
    arch = meta.get("general.architecture", "deepseek41")
    n_layer = meta[f"{arch}.num_hidden_layers"]
    n_expert = meta[f"{arch}.n_routed_experts"]
    k = meta[f"{arch}.num_experts_per_tok"]
    sizes = tensor_sizes(tensors, data_start, file_bytes)
    dims = {name: d for name, d, _, _ in tensors}
    roles, units = {}, {}
    for name, size in sizes.items():
        label = role(name)
        roles[label] = roles.get(label, 0) + size
        m = ROUTED.match(name)
        if m:
            layer = int(m.group(1))
            units[layer] = units.get(layer, 0) + size // n_expert
    engram_tables = sum(1 for name in sizes if role(name) == "engram_disk")
    resident = sum(b for label, b in roles.items()
                   if label not in ("routed", "engram_disk", "embedding"))
    per_token = {
        "resident": resident,
        "embedding_row": roles.get("embedding", 0) // dims["token_embd.weight"][1],
        "routed": roles.get("routed", 0) * k // n_expert,
        "engram_rows": engram_tables * ENGRAM_COLS * ENGRAM_ROW_BYTES,
    }
    per_token["ram_total"] = per_token["resident"] + per_token["embedding_row"] + per_token["routed"]
    return {
        "n_layer": n_layer, "n_expert": n_expert, "k": k,
        "file_bytes": file_bytes, "n_tensors": len(tensors),
        "roles": roles,
        "engram_tables": engram_tables,
        "resident_floor": resident + roles.get("embedding", 0),
        "per_token": per_token,
        "expert_unit_bytes": {str(layer): b for layer, b in sorted(units.items())},
    }


def roofline(acct, gbps):
    """Byte-floor milliseconds per decoded token at `gbps` GB/s of memory bandwidth."""
    pt = acct["per_token"]
    out = {f"{key}_ms": pt[key] / (gbps * 1e9) * 1e3
           for key in ("resident", "embedding_row", "routed")}
    out["total_ms"] = sum(out.values())
    out["tps_ceiling"] = 1e3 / out["total_ms"] if out["total_ms"] else 0.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--gbps", type=float, default=290.0,
                    help="memory bandwidth for the roofline (measured Q8_0 gemv: 290)")
    ap.add_argument("--out")
    args = ap.parse_args()
    meta, tensors, data_start = read_header(args.model)
    acct = account(meta, tensors, data_start, os.path.getsize(args.model))
    acct["roofline"] = roofline(acct, args.gbps)
    acct["gbps"] = args.gbps
    text = json.dumps(acct, indent=1)
    if args.out:
        with open(args.out, "w") as fp:
            fp.write(text + "\n")
    pt = acct["per_token"]
    print(f"tensors={acct['n_tensors']} layers={acct['n_layer']} experts={acct['n_expert']} k={acct['k']}")
    for label, b in sorted(acct["roles"].items(), key=lambda kv: -kv[1]):
        print(f"  {label:12s} {b / GIB:9.2f} GiB")
    print(f"resident floor {acct['resident_floor'] / GIB:.2f} GiB; per token: resident "
          f"{pt['resident'] / GIB:.2f} GiB, routed {pt['routed'] / GIB:.2f} GiB, "
          f"engram {pt['engram_rows']} B from {acct['engram_tables']} tables")
    r = acct["roofline"]
    print(f"roofline @{args.gbps:g} GB/s: {r['total_ms']:.1f} ms/token = {r['tps_ceiling']:.1f} t/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest discover -s speed-bench/tests -v`
Expected: all tests `ok` (5 new).

- [ ] **Step 5: Commit**

```bash
git add speed-bench/v41/gguf_bytes.py speed-bench/tests/test_gguf_bytes.py
git commit -m "speed-bench/v41: GGUF per-token byte accountant and roofline

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: V4.1 router log (`DS4_V41_ROUTER_LOG`)

**Files:**
- Modify: `ds4.c`. Add the logger just before `ds41_graph_step` (currently
  around line 41558). Call it inside `ds41_graph_step`: capture after the
  per-layer `ds41_graph_layer` / `ds41_graph_decode_layer` call, flush after the
  final `ds4_gpu_end_commands()`.
- Test: `tests/test_deepseek41_graph.c`. Add `check_router_log_format`
  (model-free) and `check_router_log` (needs the model, run in Task 9), and
  dispatch both from `main`.

**Interfaces:**
- Produces (the file format Task 6 parses): the env `DS4_V41_ROUTER_LOG=<path>`.
  Line 1 is `# pos layer e0..e5`. Then, for each decoded token and each layer in
  order, one line `"<pos> <layer> <e0> ... <e{k-1}>"`.
- C functions:
  - `static void ds41_router_log_write(FILE *fp, uint32_t pos, const int32_t *ids, uint32_t n_layer, uint32_t k)`
  - `static bool ds41_router_log_on(void)`
  - `static void ds41_router_log_capture(ds41_gpu_graph *g, uint32_t il)`
  - `static void ds41_router_log_flush(uint32_t pos)`

How it works: each layer's `g->selected` (k int32 ids, allocated as k × 4 bytes by
`DS41_ALLOC`) is copied GPU→GPU with `ds4_gpu_tensor_copy`. That blit is encoded
into the active command buffer (`ds4_metal.m:9518`), so it runs in order and adds no
synchronization. The log buffer is read once per token after the final drain.
With the env unset, both calls return immediately.

- [ ] **Step 1: Write the failing model-free test**

In `tests/test_deepseek41_graph.c`, add above `int main`:
```c
static int check_router_log_format(void) {
    int rc = 1;
    char line[128];
    const int32_t ids[6] = {5, 17, 383, 0, 1, 2};
    FILE *fp = tmpfile();
    REQUIRE(fp);
    ds41_router_log_write(fp, 42, ids, 2, 3);
    rewind(fp);
    REQUIRE(fgets(line, sizeof(line), fp) && !strcmp(line, "42 0 5 17 383\n"));
    REQUIRE(fgets(line, sizeof(line), fp) && !strcmp(line, "42 1 0 1 2\n"));
    REQUIRE(!fgets(line, sizeof(line), fp));
    /* An unwritable path disables the logger once; decode must not care. */
    setenv("DS4_V41_ROUTER_LOG", "/nonexistent-dir/router.log", 1);
    REQUIRE(!ds41_router_log_on());
    REQUIRE(!ds41_router_log_on());
    unsetenv("DS4_V41_ROUTER_LOG");
    fprintf(stderr, "V4.1 router log format PASS\n");
    rc = 0;
done:
    unsetenv("DS4_V41_ROUTER_LOG");
    if (fp) fclose(fp);
    return rc;
}
```
and at the start of `main` (before the `#ifdef __APPLE__` block):
```c
    if (argc == 2 && !strcmp(argv[1], "--router-log-format"))
        return check_router_log_format();
```

- [ ] **Step 2: Build to verify it fails**

Run: `make tests/test_deepseek41_graph 2>&1 | grep -E "error" | head -3`
Expected: `error: call to undeclared function 'ds41_router_log_write'` (or "implicit declaration").

- [ ] **Step 3: Implement the logger in `ds4.c`**

Insert immediately before `static DS4_MAYBE_UNUSED bool ds41_graph_step(`:
```c
/* Diagnostic (V4.1 Phase 0): DS4_V41_ROUTER_LOG=<path> writes one line per
 * decoded token and layer, "pos layer e0 .. e{k-1}", for offline routing
 * locality analysis. Off by default. Each layer's `selected` is blitted into
 * a log buffer inside the active command buffer and read once per token after
 * the final drain, so the log adds no synchronization and cannot change
 * outputs. An unwritable path disables it after one error. */
static FILE *g_ds41_router_log_fp;
static ds4_gpu_tensor *g_ds41_router_log_buf;
static int32_t *g_ds41_router_log_host;
static uint32_t g_ds41_router_log_layers;
static bool g_ds41_router_log_failed;

static void ds41_router_log_write(FILE *fp, uint32_t pos, const int32_t *ids,
                                  uint32_t n_layer, uint32_t k) {
    for (uint32_t il = 0; il < n_layer; il++) {
        fprintf(fp, "%u %u", pos, il);
        for (uint32_t i = 0; i < k; i++) fprintf(fp, " %d", ids[il * k + i]);
        fputc('\n', fp);
    }
}

static bool ds41_router_log_on(void) {
    const char *path = getenv("DS4_V41_ROUTER_LOG");
    if (!path || !path[0] || g_ds41_router_log_failed) return false;
    if (!g_ds41_router_log_fp) {
        g_ds41_router_log_fp = fopen(path, "w");
        if (!g_ds41_router_log_fp) {
            fprintf(stderr, "ds4: cannot open DS4_V41_ROUTER_LOG=%s; router log disabled\n", path);
            g_ds41_router_log_failed = true;
            return false;
        }
        fprintf(g_ds41_router_log_fp, "# pos layer e0..e%u\n", DS4_N_EXPERT_USED - 1u);
    }
    return true;
}

static void ds41_router_log_capture(ds41_gpu_graph *g, uint32_t il) {
    if (!ds41_router_log_on()) return;
    const uint64_t row = (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t);
    if (!g_ds41_router_log_buf) {
        g_ds41_router_log_buf = ds4_gpu_tensor_alloc((uint64_t)DS4_N_LAYER * row);
        g_ds41_router_log_host = malloc((size_t)DS4_N_LAYER * row);
        if (!g_ds41_router_log_buf || !g_ds41_router_log_host) {
            fprintf(stderr, "ds4: V4.1 router log buffers unavailable; router log disabled\n");
            g_ds41_router_log_failed = true;
            return;
        }
    }
    if (ds4_gpu_tensor_copy(g_ds41_router_log_buf, il * row, g->selected, 0, row))
        g_ds41_router_log_layers = il + 1u;
}

static void ds41_router_log_flush(uint32_t pos) {
    if (!ds41_router_log_on() || !g_ds41_router_log_buf || !g_ds41_router_log_layers) return;
    const uint64_t row = (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t);
    if (ds4_gpu_tensor_read(g_ds41_router_log_buf, 0, g_ds41_router_log_host,
                            g_ds41_router_log_layers * row))
        ds41_router_log_write(g_ds41_router_log_fp, pos, g_ds41_router_log_host,
                              g_ds41_router_log_layers, DS4_N_EXPERT_USED);
    fflush(g_ds41_router_log_fp);
    g_ds41_router_log_layers = 0;
}
```

In `ds41_graph_step`, directly after the block
```c
        if (ok) {
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
            ok = ds41_graph_decode_layer(g, m, l, il, token);
#else
            ok = ds41_graph_layer(g, m, l, il, token);
#endif
        }
```
add:
```c
        if (ok) ds41_router_log_capture(g, il);
```
and directly after `if (ds4_gpu_commands_active() && !ds4_gpu_end_commands()) ok = false;`
(the line after the layer loop) add:
```c
    if (ok) ds41_router_log_flush(g->pos);
```

- [ ] **Step 4: Build and run the model-free test**

Run: `make tests/test_deepseek41_graph && ./tests/test_deepseek41_graph --router-log-format`
Expected: `ds4: cannot open DS4_V41_ROUTER_LOG=/nonexistent-dir/router.log; router log disabled`
printed once, then `V4.1 router log format PASS`, exit 0. It is CPU-only and needs no GPU gate.

- [ ] **Step 5: Add the model-based check** (compiled now, run in Task 9)

In `tests/test_deepseek41_graph.c`, add above `int main`:
```c
#ifdef __APPLE__
static int check_router_log(const char *path, const char *prompt_path) {
    ds4_engine *engine = NULL;
    ds4_session *control = NULL, *candidate = NULL;
    ds4_tokens tokens = {0};
    char *prompt = NULL, err[256] = "", log_path[] = "/tmp/ds41_router_log_XXXXXX";
    size_t prompt_bytes;
    FILE *fp = NULL;
    uint32_t lines = 0;
    int rc = 1;
    enum { PREFIX = 511, STEPS = 32 };
    const int fd = mkstemp(log_path);
    ds4_engine_options opt = {.model_path = path, .backend = DS4_BACKEND_METAL,
        .context_size = 4096, .power_percent = 100, .ssd_streaming = true,
        .ssd_streaming_cache_bytes = UINT64_C(8) << 30};
    REQUIRE(fd >= 0);
    close(fd);
    unsetenv("DS4_V41_ROUTER_LOG");
    REQUIRE(imatrix_read_text_file(prompt_path, &prompt, &prompt_bytes));
    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    ds4_tokenize_text(engine, prompt, &tokens);
    REQUIRE(tokens.len > PREFIX + STEPS);
    REQUIRE(ds4_session_create(&control, engine, 4096) == 0);
    REQUIRE(ds4_session_create(&candidate, engine, 4096) == 0);
    ds4_tokens input = {.v = tokens.v, .len = PREFIX, .cap = PREFIX};
    REQUIRE(ds4_session_sync(control, &input, err, sizeof(err)) == 0);
    REQUIRE(ds4_session_sync(candidate, &input, err, sizeof(err)) == 0);
    for (int step = 0; step < STEPS; step++) {
        const int token = tokens.v[PREFIX + step];
        unsetenv("DS4_V41_ROUTER_LOG");
        REQUIRE(ds4_session_eval(control, token, err, sizeof(err)) == 0);
        setenv("DS4_V41_ROUTER_LOG", log_path, 1);
        REQUIRE(ds4_session_eval(candidate, token, err, sizeof(err)) == 0);
        REQUIRE(!memcmp(control->logits, candidate->logits, DS4_N_VOCAB * sizeof(float)));
    }
    unsetenv("DS4_V41_ROUTER_LOG");
    REQUIRE((fp = fopen(log_path, "r")) != NULL);
    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#') continue;
        unsigned pos, layer;
        int e[6];
        const int n = sscanf(line, "%u %u %d %d %d %d %d %d", &pos, &layer,
                             &e[0], &e[1], &e[2], &e[3], &e[4], &e[5]);
        REQUIRE(n == 2 + (int)DS4_N_EXPERT_USED && layer == lines % DS4_N_LAYER);
        for (int i = 0; i < n - 2; i++) {
            REQUIRE(e[i] >= 0 && (uint32_t)e[i] < DS4_N_EXPERT);
            for (int j = 0; j < i; j++) REQUIRE(e[i] != e[j]);
        }
        lines++;
    }
    REQUIRE(lines == (uint32_t)STEPS * DS4_N_LAYER);
    fprintf(stderr, "V4.1 router log: %u lines; logits identical with and without it PASS\n", lines);
    rc = 0;
done:
    if (err[0]) fprintf(stderr, "%s\n", err);
    unsetenv("DS4_V41_ROUTER_LOG");
    if (fp) fclose(fp);
    unlink(log_path);
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_session_free(candidate); ds4_session_free(control); ds4_engine_close(engine);
    ds4_tokens_free(&tokens); free(prompt);
    return rc;
}
#endif
```
and in `main`, inside the existing `#ifdef __APPLE__` block at the top:
```c
    if (argc == 4 && !strcmp(argv[2], "--router-log"))
        return check_router_log(argv[1], argv[3]);
```

- [ ] **Step 6: Build everything and rerun the model-free test**

Run: `make all tests/test_deepseek41_graph && ./tests/test_deepseek41_graph --router-log-format`
Expected: build succeeds with no new warnings in the touched lines, and prints `V4.1 router log format PASS`.

- [ ] **Step 7: Commit** (the Qwen fast tier for this commit runs in Task 8)

```bash
git add ds4.c tests/test_deepseek41_graph.c
git commit -m "V4.1: env-gated router log for Phase-0 locality analysis

DS4_V41_ROUTER_LOG=<path> records each decoded token's per-layer expert ids.
Off by default; a GPU-side blit per layer and one read per token, so outputs
cannot change.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: V4.1 decode profile (`DS4_V41_DECODE_PROFILE`)

**Files:**
- Modify: `ds4_metal.m`. Add two read-only accessors after
  `ds4_gpu_stream_expert_cache_note_pread` (currently around line 13572).
- Modify: `ds4_gpu.h`. Declare them next to `ds4_gpu_print_memory_report`
  (currently around line 450).
- Modify: `ds4.c`. Add the profile struct, print function and accumulation in
  `ds41_graph_step`.
- Test: `tests/test_deepseek41_graph.c`. Add `check_decode_profile_format`
  (model-free).

**Interfaces:**
- Produces (the line format Task 7 parses): with `DS4_V41_DECODE_PROFILE=1`, every
  64 decoded tokens, one cumulative per-token average line:
  `ds4: V4.1 decode profile: tokens=<n> step_ms=<f> engram_ms=<f> gpu_busy_ms=<f> pread_ms=<f> pread_mib=<f> hits=<f> misses=<f>`.
  `gpu_busy_ms` needs `DS4_METAL_GPU_BUSY_PROFILE=1`, which feeds the existing
  `g_gpu_busy_accum` in `ds4_gpu_wait_command_buffer`. All command buffers,
  including pending ones, pass through that function.
- C:
  - `double ds4_gpu_busy_accum_ms(void)`
  - `void ds4_gpu_stream_expert_cache_counters(uint64_t *hits, uint64_t *misses, uint64_t *pread_bytes, double *pread_ms)`
  - `static void ds41_decode_profile_print(FILE *fp, const ds41_decode_profile *p)`

Decomposition used later:
`host_gap_ms = step_ms − gpu_busy_ms − pread_ms − engram_ms`. Pread and Engram
reads happen while the GPU is idle in the streaming solo path.

- [ ] **Step 1: Write the failing model-free test**

In `tests/test_deepseek41_graph.c`, add above `int main`:
```c
static int check_decode_profile_format(void) {
    int rc = 1;
    char line[256];
    const ds41_decode_profile p = {.tokens = 2, .step_ms = 200.0, .engram_ms = 2.0,
        .gpu_ms = 120.0, .pread_ms = 30.0, .pread_bytes = UINT64_C(6) << 20,
        .hits = 100, .misses = 20};
    FILE *fp = tmpfile();
    REQUIRE(fp);
    ds41_decode_profile_print(fp, &p);
    rewind(fp);
    REQUIRE(fgets(line, sizeof(line), fp));
    REQUIRE(!strcmp(line, "ds4: V4.1 decode profile: tokens=2 step_ms=100.000 engram_ms=1.000 "
                          "gpu_busy_ms=60.000 pread_ms=15.000 pread_mib=3.000 hits=50.00 misses=10.00\n"));
    fprintf(stderr, "V4.1 decode profile format PASS\n");
    rc = 0;
done:
    if (fp) fclose(fp);
    return rc;
}
```
and at the start of `main`:
```c
    if (argc == 2 && !strcmp(argv[1], "--decode-profile-format"))
        return check_decode_profile_format();
```

- [ ] **Step 2: Build to verify it fails**

Run: `make tests/test_deepseek41_graph 2>&1 | grep -E "error" | head -3`
Expected: `error: unknown type name 'ds41_decode_profile'` (or an undeclared-identifier error).

- [ ] **Step 3: Add the accessors**

`ds4_gpu.h`, next to `void ds4_gpu_print_memory_report(const char *label);`:
```c
/* Diagnostics: cumulative GPU busy time of waited command buffers (only while
 * DS4_METAL_GPU_BUSY_PROFILE is set) and streaming expert-cache counters. */
double ds4_gpu_busy_accum_ms(void);
void ds4_gpu_stream_expert_cache_counters(uint64_t *hits, uint64_t *misses,
                                          uint64_t *pread_bytes, double *pread_ms);
```

`ds4_metal.m`, directly after the closing brace of `ds4_gpu_stream_expert_cache_note_pread`:
```objc
double ds4_gpu_busy_accum_ms(void) {
    return g_gpu_busy_accum * 1000.0;
}

void ds4_gpu_stream_expert_cache_counters(uint64_t *hits, uint64_t *misses,
                                          uint64_t *pread_bytes, double *pread_ms) {
    if (hits) *hits = g_stream_expert_cache_hits;
    if (misses) *misses = g_stream_expert_cache_misses;
    if (pread_bytes) *pread_bytes = g_stream_expert_cache_pread_bytes;
    if (pread_ms) *pread_ms = g_stream_expert_cache_pread_ms;
}
```

- [ ] **Step 4: Add the profile to `ds4.c`**

Insert just before `static DS4_MAYBE_UNUSED bool ds41_graph_step(` (after the Task 4 logger):
```c
/* Diagnostic (V4.1 Phase 0): DS4_V41_DECODE_PROFILE=1 prints, every 64
 * decoded tokens, per-token averages since the first one: step wall time,
 * Engram read time, GPU busy time (with DS4_METAL_GPU_BUSY_PROFILE=1) and the
 * streaming expert cache's pread time, bytes, hits and misses. */
typedef struct {
    uint64_t tokens, pread_bytes, hits, misses;
    double step_ms, engram_ms, gpu_ms, pread_ms;
} ds41_decode_profile;

static ds41_decode_profile g_ds41_decode_profile;

static void ds41_decode_profile_print(FILE *fp, const ds41_decode_profile *p) {
    const double n = p->tokens ? (double)p->tokens : 1.0;
    fprintf(fp, "ds4: V4.1 decode profile: tokens=%llu step_ms=%.3f engram_ms=%.3f "
            "gpu_busy_ms=%.3f pread_ms=%.3f pread_mib=%.3f hits=%.2f misses=%.2f\n",
            (unsigned long long)p->tokens, p->step_ms / n, p->engram_ms / n,
            p->gpu_ms / n, p->pread_ms / n, (double)p->pread_bytes / n / 1048576.0,
            (double)p->hits / n, (double)p->misses / n);
}
```

In `ds41_graph_step`, right after `if (!ds41_hash_tokens(g, &next_history, &token, 1, &ids[0][0])) return false;`:
```c
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    const bool profile = getenv("DS4_V41_DECODE_PROFILE") != NULL;
    uint64_t hits0 = 0, misses0 = 0, pread_bytes0 = 0;
    double pread_ms0 = 0.0;
    const double step_t0 = profile ? now_sec() : 0.0;
    const double gpu_ms0 = profile ? ds4_gpu_busy_accum_ms() : 0.0;
    if (profile) ds4_gpu_stream_expert_cache_counters(&hits0, &misses0, &pread_bytes0, &pread_ms0);
#endif
```
Put the Engram read in brackets. Directly before `#ifdef __APPLE__` / `const bool parallel = ...`:
```c
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    const double engram_t0 = profile ? now_sec() : 0.0;
#endif
```
and directly after the closing `}` of the `for (uint32_t i = 0; !parallel && ...` Engram loop:
```c
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    const double engram_s = profile ? now_sec() - engram_t0 : 0.0;
#endif
```
At the end of the function, directly before `g->history = next_history;`:
```c
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    if (profile) {
        ds41_decode_profile *p = &g_ds41_decode_profile;
        uint64_t hits = 0, misses = 0, pread_bytes = 0;
        double pread_ms = 0.0;
        ds4_gpu_stream_expert_cache_counters(&hits, &misses, &pread_bytes, &pread_ms);
        p->tokens++;
        p->step_ms += (now_sec() - step_t0) * 1e3;
        p->engram_ms += engram_s * 1e3;
        p->gpu_ms += ds4_gpu_busy_accum_ms() - gpu_ms0;
        p->pread_ms += pread_ms - pread_ms0;
        p->pread_bytes += pread_bytes - pread_bytes0;
        p->hits += hits - hits0;
        p->misses += misses - misses0;
        if (p->tokens % 64u == 0u) ds41_decode_profile_print(stderr, p);
    }
#endif
```

- [ ] **Step 5: Build and run both model-free tests**

Run: `make all tests/test_deepseek41_graph && ./tests/test_deepseek41_graph --decode-profile-format && ./tests/test_deepseek41_graph --router-log-format`
Expected: build succeeds, then `V4.1 decode profile format PASS` and `V4.1 router log format PASS`.

- [ ] **Step 6: Commit** (the Qwen fast tier for this commit runs in Task 8)

```bash
git add ds4.c ds4_metal.m ds4_gpu.h tests/test_deepseek41_graph.c
git commit -m "V4.1: env-gated per-token decode profile for Phase 0

DS4_V41_DECODE_PROFILE=1 reports step, Engram, GPU busy and expert-cache
pread time plus hits and misses per decoded token. Adds two read-only Metal
accessors; nothing changes with the env unset.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Router-locality simulator

**Files:**
- Create: `speed-bench/v41/router_locality.py`
- Test: `speed-bench/tests/test_router_locality.py`

**Interfaces:**
- Consumes: the Task 4 log format and `expert_unit_bytes` from Task 3.
- Produces (used by Task 7 `report`):
  - `parse_router_log(lines) -> tokens[t][layer] = tuple(ids)`
  - `lru_hit_rates(tokens, unit_bytes, budgets, warmup=0) -> {budget_bytes: rate}`
  - `overlap_stats(tokens, ks=(1,2,4,8)) -> {"pair_overlap", "union_cover": {"1".."8"}}`
  - `new_experts_per_token(tokens) -> float`
  - CLI `router_locality.py LOG --bytes bytes.json [--budgets-gib ...] [--warmup 200] [--out F]`,
    which writes JSON keys `tokens, layers, warmup, lru_hit{"<GiB>": rate}, new_per_token, pair_overlap, union_cover`.

- [ ] **Step 1: Write the failing tests**

`speed-bench/tests/test_router_locality.py`:
```python
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v41"))
import router_locality as rl  # noqa: E402

LOG = """# pos layer e0..e1
10 0 1 2
11 0 1 2
12 0 3 4
13 0 1 2
"""


class ParseTest(unittest.TestCase):
    def test_parse(self):
        tokens = rl.parse_router_log(LOG.splitlines())
        self.assertEqual(tokens, [[(1, 2)], [(1, 2)], [(3, 4)], [(1, 2)]])

    def test_layers_out_of_order(self):
        with self.assertRaises(ValueError):
            rl.parse_router_log(["5 1 1 2", "5 0 3 4"])

    def test_two_layers(self):
        tokens = rl.parse_router_log(["1 0 7 8", "1 1 9 10", "2 0 7 8", "2 1 9 11"])
        self.assertEqual(tokens, [[(7, 8), (9, 10)], [(7, 8), (9, 11)]])


class LruTest(unittest.TestCase):
    def test_hit_rate_by_hand(self):
        tokens = rl.parse_router_log(LOG.splitlines())
        # budget 2 units: t1 hits both, t2 evicts 1 and 2, t3 misses both -> 2/8
        self.assertEqual(rl.lru_hit_rates(tokens, [1], [2]), {2: 0.25})
        # budget 4 units holds everything: only the 4 first-time loads miss -> 4/8
        self.assertEqual(rl.lru_hit_rates(tokens, [1], [4]), {4: 0.5})

    def test_warmup_excludes_early_tokens(self):
        tokens = rl.parse_router_log(LOG.splitlines())
        # count only t2, t3 with budget 4: t2 misses 3,4; t3 hits 1,2 -> 2/4
        self.assertEqual(rl.lru_hit_rates(tokens, [1], [4], warmup=2), {4: 0.5})


class OverlapTest(unittest.TestCase):
    def test_overlap_and_cover(self):
        tokens = rl.parse_router_log(LOG.splitlines())
        s = rl.overlap_stats(tokens, ks=(1, 2))
        self.assertAlmostEqual(s["pair_overlap"], 1 / 3)
        self.assertAlmostEqual(s["union_cover"]["1"], 1 / 3)
        self.assertAlmostEqual(s["union_cover"]["2"], 2 / 3)

    def test_new_per_token(self):
        tokens = rl.parse_router_log(LOG.splitlines())
        # fresh counts 2,0,2,0 -> second half [2,0] -> 1.0
        self.assertEqual(rl.new_experts_per_token(tokens), 1.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest discover -s speed-bench/tests -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'router_locality'`.

- [ ] **Step 3: Write the implementation**

`speed-bench/v41/router_locality.py`:
```python
#!/usr/bin/env python3
"""Routing locality from a DS4_V41_ROUTER_LOG file (docs/V41_64GB_BUILD.md §7 step 4).

LRU expert-cache hit rate versus budget, and how much of the next token's
experts the previous tokens already selected (the ceiling for prefetch).
"""
import argparse
import collections
import json
import statistics
import sys

GIB = 1024 ** 3


def parse_router_log(lines):
    """tokens[t][layer] -> tuple of expert ids, in log order."""
    tokens, current, cur_pos = [], [], None
    for raw in lines:
        if not raw.strip() or raw.startswith("#"):
            continue
        fields = [int(x) for x in raw.split()]
        pos, layer, ids = fields[0], fields[1], tuple(fields[2:])
        if pos != cur_pos:
            if current:
                tokens.append(current)
            current, cur_pos = [], pos
        if layer != len(current):
            raise ValueError(f"pos {pos}: layer {layer} out of order")
        current.append(ids)
    if current:
        tokens.append(current)
    if tokens and any(len(t) != len(tokens[0]) for t in tokens):
        raise ValueError("tokens have different layer counts")
    return tokens


def lru_hit_rates(tokens, unit_bytes, budgets, warmup=0):
    """budget bytes -> hit rate of a global LRU over (layer, expert), after `warmup` tokens."""
    out = {}
    for budget in budgets:
        cache, used, hits, total = collections.OrderedDict(), 0, 0, 0
        for t, layers in enumerate(tokens):
            for layer, ids in enumerate(layers):
                for expert in ids:
                    key = (layer, expert)
                    hit = key in cache
                    if hit:
                        cache.move_to_end(key)
                    else:
                        cache[key] = unit_bytes[layer]
                        used += unit_bytes[layer]
                        while used > budget and cache:
                            used -= cache.popitem(last=False)[1]
                    if t >= warmup:
                        total += 1
                        hits += hit
        out[budget] = hits / total if total else 0.0
    return out


def overlap_stats(tokens, ks=(1, 2, 4, 8)):
    """Same-layer overlap with the previous token, and coverage by the union of the last K."""
    pair, cover = [], {k: [] for k in ks}
    n_layer = len(tokens[0]) if tokens else 0
    for t in range(1, len(tokens)):
        for layer in range(n_layer):
            now = set(tokens[t][layer])
            pair.append(len(now & set(tokens[t - 1][layer])) / len(now))
            for k in ks:
                seen = set()
                for back in tokens[max(0, t - k):t]:
                    seen.update(back[layer])
                cover[k].append(len(now & seen) / len(now))

    def mean(xs):
        return statistics.fmean(xs) if xs else 0.0

    return {"pair_overlap": mean(pair), "union_cover": {str(k): mean(v) for k, v in cover.items()}}


def new_experts_per_token(tokens):
    """Mean number of never-seen (layer, expert) pairs per token over the second half."""
    seen, fresh = set(), []
    for layers in tokens:
        n = 0
        for layer, ids in enumerate(layers):
            for expert in ids:
                if (layer, expert) not in seen:
                    seen.add((layer, expert))
                    n += 1
        fresh.append(n)
    tail = fresh[len(fresh) // 2:]
    return statistics.fmean(tail) if tail else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log")
    ap.add_argument("--bytes", required=True, help="gguf_bytes.py JSON (expert_unit_bytes)")
    ap.add_argument("--budgets-gib", default="2,4,8,12,16,24,32,48")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--out")
    args = ap.parse_args()
    with open(args.log) as fp:
        tokens = parse_router_log(fp)
    if not tokens:
        raise SystemExit("router_locality: empty log")
    with open(args.bytes) as fp:
        units = json.load(fp)["expert_unit_bytes"]
    unit_bytes = [units[str(layer)] for layer in range(len(tokens[0]))]
    budgets = [float(x) * GIB for x in args.budgets_gib.split(",")]
    rates = lru_hit_rates(tokens, unit_bytes, budgets, args.warmup)
    result = {"tokens": len(tokens), "layers": len(tokens[0]), "warmup": args.warmup,
              "lru_hit": {f"{b / GIB:g}": r for b, r in rates.items()},
              "new_per_token": new_experts_per_token(tokens), **overlap_stats(tokens)}
    text = json.dumps(result, indent=1)
    if args.out:
        with open(args.out, "w") as fp:
            fp.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest discover -s speed-bench/tests -v`
Expected: all tests `ok` (7 new).

- [ ] **Step 5: Commit**

```bash
git add speed-bench/v41/router_locality.py speed-bench/tests/test_router_locality.py
git commit -m "speed-bench/v41: offline router-locality and LRU cache simulator

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Phase-0 driver, collector and report

**Files:**
- Create: `speed-bench/v41/phase0.py`
- Create: `speed-bench/v41/README.md`
- Test: `speed-bench/tests/test_phase0.py`

**Interfaces:**
- Consumes:
  - `machine`, `wired` (Task 1);
  - the profile line (Task 5);
  - the `ds4-bench` CSV header `ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes` (`ds4_bench.c:789`);
  - the cleanup cache report line `ds4:   streaming expert cache budget=<n> experts ... hit_rate=<f> ...` (`ds4_metal.m:4502`, printed with `DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1`);
  - `gguf_bytes.roofline` (Task 3) and `router_locality` JSON (Task 6).
- Produces the CLI:
  - `phase0.py prompts --out DIR`
  - `phase0.py run --plan speed|locality --model M --prompts DIR --out DIR [--bin DIR] [--dry-run]`
  - `phase0.py table --out DIR`
  - `phase0.py report --out DIR --bytes bytes.json [--locality F ...]`
- Produces files:
  - per run `<tag>.result.json`, `<tag>.csv`, `<tag>.stderr` and optionally `<tag>.router.log`;
  - `results.csv`;
  - `RESULTS.md` (the generated part).

Plans (spec §7 0(b) steps 2 and 4; sized to about 2–3 h of machine time):

| Plan | Workloads | Context × cache | Decode |
| --- | --- | --- | --- |
| `speed` | `switch` | ctx {4096, 8192, 32768} × cache {4, 8, 16, 24} GB | 512 teacher-forced tokens |
| `locality` | `code`, `docs`, `it`, `switch` | ctx 4096, cache 8 GB | 2000 teacher-forced tokens, router log on |

Teacher-forced decode (`ds4-bench --teacher-forced-decode`) walks the prompt's own
next tokens. The routing therefore comes from real text and does not depend on
sampling. `switch` interleaves 2 000-character chunks of the other three
workloads, so every decode window crosses task switches.

- [ ] **Step 1: Write the failing tests**

`speed-bench/tests/test_phase0.py`:
```python
import json
import os
import stat
import sys
import tempfile
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "v41"))
import phase0  # noqa: E402

PROFILE = ("ds4: V4.1 decode profile: tokens=64 step_ms=100.000 engram_ms=1.000 "
           "gpu_busy_ms=60.000 pread_ms=15.000 pread_mib=3.000 hits=50.00 misses=10.00\n")
PROFILE_LATER = PROFILE.replace("tokens=64", "tokens=128").replace("step_ms=100.000", "step_ms=90.000")
CACHE = ("ds4:   streaming expert cache budget=7000 experts entries=6990 expert=1.42 MiB "
         "target=9.70 GiB live=9.69 GiB, hits=900 misses=100 hit_rate=0.900 wraps=0\n")
CSV = ("ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,"
       "gen_steady_tps,kvcache_bytes\n4096,4096,120.5,512,10.2,150.0,511,10.4,123456\n")
WIRED = {"steady_gib": 30.0, "peak_gib": 32.0, "n": 10}


class PromptTest(unittest.TestCase):
    def test_prompts(self):
        texts = phase0.build_prompts(phase0.ROOT)
        self.assertEqual(sorted(texts), ["code", "docs", "it", "switch"])
        for name, text in texts.items():
            self.assertEqual(len(text), phase0.PROMPT_CHARS, name)
        c = phase0.CHUNK
        self.assertEqual(texts["switch"][:c], texts["code"][:c])
        self.assertEqual(texts["switch"][c:2 * c], texts["docs"][:c])
        self.assertEqual(texts["switch"][2 * c:3 * c], texts["it"][:c])


class PlanTest(unittest.TestCase):
    def test_plans(self):
        self.assertEqual(len(phase0.PLANS["speed"]), 12)
        self.assertEqual(len(phase0.PLANS["locality"]), 4)
        self.assertTrue(all(spec[4] for spec in phase0.PLANS["locality"]))

    def test_bench_cmd(self):
        cmd = phase0.bench_cmd("/b", "/m.gguf", "/p.txt", 8192, 512, 16, "/o.csv")
        self.assertEqual(cmd[0], "/b/ds4-bench")
        for flag in ("--teacher-forced-decode", "--ssd-streaming", "16GB", "8192", "/o.csv"):
            self.assertIn(flag, cmd)


class ParseTest(unittest.TestCase):
    def test_profile_takes_last_line_and_host_gap(self):
        p = phase0.parse_profile(PROFILE + PROFILE_LATER)
        self.assertEqual(p["tokens"], 128)
        self.assertAlmostEqual(p["host_gap_ms"], 90.0 - 60.0 - 15.0 - 1.0)

    def test_profile_missing(self):
        self.assertIsNone(phase0.parse_profile("nothing"))

    def test_cache(self):
        self.assertEqual(phase0.parse_cache(CACHE), {"cache_experts": 7000, "cache_hit_rate": 0.9})

    def test_bench_csv(self):
        b = phase0.parse_bench_csv(CSV)
        self.assertEqual(b["gen_steady_tps"], 10.4)
        self.assertEqual(b["prefill_tps"], 120.5)

    def test_combine(self):
        spec = ("switch", 4096, 8, 512, False)
        row = phase0.combine(spec, phase0.parse_bench_csv(CSV), phase0.parse_profile(PROFILE),
                             phase0.parse_cache(CACHE), WIRED, 12.0, contaminated=False)
        self.assertAlmostEqual(row["decode_hit_rate"], 50 / 60)
        self.assertEqual(row["wired_steady_gib"], 30.0)
        self.assertFalse(row["contaminated"])


def fake_bin(tmp, body):
    path = os.path.join(tmp, "ds4-bench")
    with open(path, "w") as fp:
        fp.write("#!/bin/sh\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return tmp


class RunTest(unittest.TestCase):
    SPEC = ("switch", 4096, 8, 512, False)

    def run_one(self, tmp, body, running=lambda: ""):
        os.makedirs(os.path.join(tmp, "prompts"), exist_ok=True)
        open(os.path.join(tmp, "prompts", "switch.txt"), "w").close()
        return phase0.run_one(fake_bin(tmp, body), "/m.gguf", os.path.join(tmp, "prompts"), tmp,
                              self.SPEC, running=running, swap=lambda: 0.0,
                              sampler=lambda: _FakeSampler())

    def test_failed_run_leaves_no_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                self.run_one(tmp, "exit 3\n")
            self.assertFalse(any(n.endswith(".result.json") for n in os.listdir(tmp)))

    def test_refuses_when_busy(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                self.run_one(tmp, "exit 0\n", running=lambda: "9 ds4-server")

    def test_success_and_contamination(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_line = CSV.replace("\n", "\\n")
            body = (f'for a; do last=$a; done\nprintf "{csv_line}" > "$last"\n'
                    f'printf "%s" "{PROFILE.strip()}" >&2\n')
            calls = iter(["", "7 ds4-server"])   # free before, busy after
            row = self.run_one(tmp, body, running=lambda: next(calls))
            self.assertTrue(row["contaminated"])
            self.assertEqual(row["gen_steady_tps"], 10.4)
            self.assertIsNone(self.run_one(tmp, "exit 9\n"))   # done -> skipped

    def test_dry_run_touches_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = phase0.run_one("/b", "/m.gguf", "/p", tmp, self.SPEC, dry_run=True)
            self.assertIsNone(out)
            self.assertEqual(os.listdir(tmp), [])


class _FakeSampler:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def summary(self):
        return WIRED


class ReportTest(unittest.TestCase):
    def test_report_marks_swapped_and_contaminated(self):
        rows = [
            {"workload": "switch", "ctx": 4096, "cache_gb": 8, "gen_steady_tps": 10.0,
             "step_ms": 100.0, "gpu_busy_ms": 60.0, "pread_ms": 15.0, "engram_ms": 1.0,
             "host_gap_ms": 24.0, "decode_hit_rate": 0.9, "wired_steady_gib": 30.0,
             "swap_delta_mib": 900.0, "contaminated": False},
            {"workload": "switch", "ctx": 8192, "cache_gb": 8, "gen_steady_tps": 11.0,
             "step_ms": 91.0, "gpu_busy_ms": 60.0, "pread_ms": 10.0, "engram_ms": 1.0,
             "host_gap_ms": 20.0, "decode_hit_rate": 0.92, "wired_steady_gib": 30.5,
             "swap_delta_mib": 0.0, "contaminated": True},
        ]
        bytes_json = {"per_token": {"resident": 8_000_000_000, "embedding_row": 0,
                                    "routed": 2_000_000_000}, "gbps": 290.0}
        locality = [{"name": "code", "lru_hit": {"8": 0.9}, "pair_overlap": 0.3,
                     "union_cover": {"1": 0.3, "4": 0.6}, "new_per_token": 2.0}]
        text = phase0.report(rows, bytes_json, locality)
        self.assertIn("swapped", text)
        self.assertIn("contaminated", text)
        self.assertIn("34.5 ms/token", text)   # (8e9 + 2e9) / 290e9
        self.assertIn("| code |", text)
        self.assertIn("Best clean run: none", text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest discover -s speed-bench/tests -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'phase0'`.

- [ ] **Step 3: Write the implementation**

`speed-bench/v41/phase0.py`:
```python
#!/usr/bin/env python3
"""Phase 0(b) driver for DeepSeek-V4.1-Flash Q2 on M5 Pro 64 GB (docs/V41_64GB_BUILD.md §7).

prompts  build the workload prompt files
run      run a plan (speed | locality) with ds4-bench, one result JSON per run
table    merge result JSONs into results.csv
report   render RESULTS.md tables from results, byte accounting and locality

Runs refuse to start while any ds4 process is running: the machine must be free.
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "speed-bench", "lib"))
sys.path.insert(0, HERE)
import machine  # noqa: E402
import wired  # noqa: E402
import gguf_bytes  # noqa: E402

PROMPT_CHARS = 200_000
CHUNK = 2_000
SOURCES = {
    "code": ["ds4_server.c"],
    "docs": ["README.md", "docs/*.md", "QA_BEFORE_RELEASES.md", "MODEL_CARD.md", "EVAL_DATA.md"],
    "it": ["speed-bench/promessi_sposi.txt"],
}
# (workload, ctx, cache_gb, gen_tokens, router_log)
PLANS = {
    "speed": [("switch", ctx, gb, 512, False)
              for ctx in (4096, 8192, 32768) for gb in (4, 8, 16, 24)],
    "locality": [(w, 4096, 8, 2000, True) for w in ("code", "docs", "it", "switch")],
}
ENV = {"DS4_V41_DECODE_PROFILE": "1", "DS4_METAL_GPU_BUSY_PROFILE": "1",
       "DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY": "1"}
SWAP_LIMIT_MIB = 256.0
FIELDS = ["workload", "ctx", "cache_gb", "gen", "prefill_tps", "gen_tps", "gen_steady_tps",
          "gen_first_ms", "step_ms", "gpu_busy_ms", "pread_ms", "engram_ms", "host_gap_ms",
          "pread_mib", "hits", "misses", "decode_hit_rate", "cache_experts", "cache_hit_rate",
          "wired_steady_gib", "wired_peak_gib", "swap_delta_mib", "contaminated", "router_log"]
PROFILE_RE = re.compile(
    r"ds4: V4\.1 decode profile: tokens=(\d+) step_ms=([\d.]+) engram_ms=([\d.]+) "
    r"gpu_busy_ms=([\d.]+) pread_ms=([\d.]+) pread_mib=([\d.]+) hits=([\d.]+) misses=([\d.]+)")
CACHE_RE = re.compile(r"streaming expert cache budget=(\d+) experts .*?hit_rate=([\d.]+)")


def build_prompts(root, chars=PROMPT_CHARS, chunk=CHUNK):
    """name -> text; 'switch' interleaves chunk-sized pieces of code, docs and it."""
    texts = {}
    for name, patterns in SOURCES.items():
        paths = [p for pat in patterns for p in sorted(glob.glob(os.path.join(root, pat)))]
        text = ""
        for path in paths:
            with open(path, encoding="utf-8", errors="replace") as fp:
                text += fp.read() + "\n\n"
        if len(text) < chars:
            raise ValueError(f"{name}: sources have {len(text)} chars, need {chars}")
        texts[name] = text[:chars]
    parts, i = [], 0
    while sum(len(p) for p in parts) < chars:
        for name in ("code", "docs", "it"):
            parts.append(texts[name][i:i + chunk])
        i += chunk
    texts["switch"] = "".join(parts)[:chars]
    return texts


def bench_cmd(bin_dir, model, prompt, ctx, gen, cache_gb, csv_path):
    return [os.path.join(bin_dir, "ds4-bench"), "--metal", "-m", model,
            "--prompt-file", prompt, "--ctx-start", str(ctx), "--ctx-max", str(ctx),
            "--gen-tokens", str(gen), "--teacher-forced-decode",
            "--ssd-streaming", "--ssd-streaming-cache-experts", f"{cache_gb}GB",
            "--csv", csv_path]


def parse_profile(text):
    rows = PROFILE_RE.findall(text)
    if not rows:
        return None
    keys = ("tokens", "step_ms", "engram_ms", "gpu_busy_ms", "pread_ms", "pread_mib",
            "hits", "misses")
    p = dict(zip(keys, (float(x) for x in rows[-1])))
    p["tokens"] = int(p["tokens"])
    p["host_gap_ms"] = p["step_ms"] - p["gpu_busy_ms"] - p["pread_ms"] - p["engram_ms"]
    return p


def parse_cache(text):
    rows = CACHE_RE.findall(text)
    if not rows:
        return None
    return {"cache_experts": int(rows[-1][0]), "cache_hit_rate": float(rows[-1][1])}


def parse_bench_csv(text):
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError("empty ds4-bench CSV")
    return {k: float(rows[-1][k]) for k in ("prefill_tps", "gen_tps", "gen_steady_tps",
                                             "gen_first_ms")}


def combine(spec, bench, profile, cache, wired_summary, swap_delta_mib, contaminated):
    workload, ctx, gb, gen, _ = spec
    row = {"workload": workload, "ctx": ctx, "cache_gb": gb, "gen": gen, **bench,
           "wired_steady_gib": wired_summary["steady_gib"],
           "wired_peak_gib": wired_summary["peak_gib"],
           "swap_delta_mib": swap_delta_mib, "contaminated": contaminated}
    if profile:
        for key in ("step_ms", "engram_ms", "gpu_busy_ms", "pread_ms", "pread_mib",
                    "hits", "misses", "host_gap_ms"):
            row[key] = profile[key]
        lookups = profile["hits"] + profile["misses"]
        row["decode_hit_rate"] = profile["hits"] / lookups if lookups else None
    if cache:
        row.update(cache)
    return row


def run_one(bin_dir, model, prompts_dir, out_dir, spec, dry_run=False,
            running=machine.ds4_running, swap=machine.swap_used_mib,
            sampler=wired.WiredSampler):
    workload, ctx, gb, gen, log_router = spec
    tag = f"{workload}-c{ctx}-g{gb}-n{gen}"
    result_path = os.path.join(out_dir, tag + ".result.json")
    csv_path = os.path.join(out_dir, tag + ".csv")
    cmd = bench_cmd(bin_dir, model, os.path.join(prompts_dir, workload + ".txt"),
                    ctx, gen, gb, csv_path)
    if dry_run:
        print("phase0:", " ".join(cmd))
        return None
    if os.path.exists(result_path):
        print("phase0: skip (done)", tag)
        return None
    busy = running()
    if busy:
        raise SystemExit("phase0: ds4 is running; the machine must be free:\n" + busy)
    env = dict(os.environ, **ENV)
    router_log = os.path.join(out_dir, tag + ".router.log") if log_router else None
    if router_log:
        env["DS4_V41_ROUTER_LOG"] = router_log
    stderr_path = os.path.join(out_dir, tag + ".stderr")
    swap0 = swap()
    with open(stderr_path, "w") as err, sampler() as ws:
        rc = subprocess.run(cmd, env=env, stdout=err, stderr=err).returncode
    if rc != 0:
        raise SystemExit(f"phase0: {tag} failed (rc={rc}), see {stderr_path}")
    contaminated = bool(running())
    with open(stderr_path) as fp:
        stderr = fp.read()
    with open(csv_path) as fp:
        bench = parse_bench_csv(fp.read())
    row = combine(spec, bench, parse_profile(stderr), parse_cache(stderr), ws.summary(),
                  swap() - swap0, contaminated)
    row["router_log"] = router_log
    with open(result_path, "w") as fp:
        json.dump(row, fp, indent=1)
    print(f"phase0: done {tag} {row['gen_steady_tps']:.2f} t/s"
          + (" CONTAMINATED" if contaminated else ""))
    return row


def table(out_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.result.json"))):
        with open(path) as fp:
            rows.append(json.load(fp))
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return rows


def _flag(row):
    flags = []
    if (row.get("swap_delta_mib") or 0.0) > SWAP_LIMIT_MIB:
        flags.append("swapped")
    if row.get("contaminated"):
        flags.append("contaminated")
    return ", ".join(flags)


def _fmt(value, spec=".2f"):
    return "—" if value is None else format(value, spec)


def report(rows, bytes_json, locality):
    """Markdown tables for RESULTS.md; `locality` entries carry a 'name' key."""
    roof = gguf_bytes.roofline(bytes_json, bytes_json.get("gbps", 290.0))
    lines = ["## Measured decode (per token)", "",
             "| workload | ctx | cache GB | t/s | step ms | GPU busy | pread | Engram | host gaps "
             "| hit rate | wired GiB | flags |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in sorted(rows, key=lambda r: (r["workload"], r["ctx"], r["cache_gb"])):
        lines.append(
            f"| {r['workload']} | {r['ctx']} | {r['cache_gb']} | {_fmt(r.get('gen_steady_tps'))} "
            f"| {_fmt(r.get('step_ms'), '.1f')} | {_fmt(r.get('gpu_busy_ms'), '.1f')} "
            f"| {_fmt(r.get('pread_ms'), '.1f')} | {_fmt(r.get('engram_ms'), '.1f')} "
            f"| {_fmt(r.get('host_gap_ms'), '.1f')} | {_fmt(r.get('decode_hit_rate'), '.3f')} "
            f"| {_fmt(r.get('wired_steady_gib'), '.1f')} | {_flag(r)} |")
    clean = [r for r in rows if not _flag(r)]
    best = max(clean, key=lambda r: r["gen_steady_tps"]) if clean else None
    lines += ["", "Best clean run: " + (
        f"{best['gen_steady_tps']:.2f} t/s ({best['workload']}, ctx {best['ctx']}, "
        f"cache {best['cache_gb']} GB)" if best else "none"), ""]
    lines += ["## Roofline from byte accounting", "",
              f"Byte floor at {bytes_json.get('gbps', 290.0):g} GB/s: resident "
              f"{roof['resident_ms']:.1f} + routed {roof['routed_ms']:.1f} ms = "
              f"{roof['total_ms']:.1f} ms/token → {roof['tps_ceiling']:.1f} t/s ceiling "
              f"(no I/O, no sync).", ""]
    if locality:
        budgets = sorted({b for loc in locality for b in loc["lru_hit"]}, key=float)
        ks = sorted({k for loc in locality for k in loc["union_cover"]}, key=int)
        lines += ["## Routing locality (decode, LRU)", "",
                  "| workload | " + " | ".join(f"{b} GiB" for b in budgets)
                  + " | pair overlap | " + " | ".join(f"cover K={k}" for k in ks)
                  + " | new/token |",
                  "| --- |" + " ---: |" * (len(budgets) + 2 + len(ks))]
        for loc in locality:
            lines.append(f"| {loc['name']} | "
                         + " | ".join(_fmt(loc["lru_hit"].get(b), ".3f") for b in budgets)
                         + f" | {loc['pair_overlap']:.3f} | "
                         + " | ".join(_fmt(loc["union_cover"].get(k), ".3f") for k in ks)
                         + f" | {loc['new_per_token']:.2f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--plan", choices=sorted(PLANS), required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--prompts", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--bin", default=ROOT)
    r.add_argument("--dry-run", action="store_true")
    t = sub.add_parser("table")
    t.add_argument("--out", required=True)
    rep = sub.add_parser("report")
    rep.add_argument("--out", required=True)
    rep.add_argument("--bytes", required=True)
    rep.add_argument("--locality", nargs="*", default=[])
    args = ap.parse_args()
    if args.mode == "prompts":
        os.makedirs(args.out, exist_ok=True)
        for name, text in build_prompts(ROOT).items():
            with open(os.path.join(args.out, name + ".txt"), "w", encoding="utf-8") as fp:
                fp.write(text)
        print("phase0: prompts in", args.out)
    elif args.mode == "run":
        os.makedirs(args.out, exist_ok=True)
        for spec in PLANS[args.plan]:
            run_one(args.bin, args.model, args.prompts, args.out, spec, dry_run=args.dry_run)
    elif args.mode == "table":
        print(f"phase0: {len(table(args.out))} rows -> results.csv")
    else:
        rows = table(args.out)
        with open(args.bytes) as fp:
            bytes_json = json.load(fp)
        locality = []
        for path in args.locality:
            with open(path) as fp:
                loc = json.load(fp)
            loc["name"] = os.path.basename(path).split("-")[0]
            locality.append(loc)
        with open(os.path.join(args.out, "RESULTS.generated.md"), "w") as fp:
            fp.write(report(rows, bytes_json, locality))
        print("phase0: wrote", os.path.join(args.out, "RESULTS.generated.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`speed-bench/v41/README.md`:
````markdown
# V4.1 Phase 0(b) tools

Spec: `docs/V41_64GB_BUILD.md` §7. Plan: `docs/superpowers/plans/2026-09-23-v41-phase0b-measure.md`.

| Tool | Purpose |
| --- | --- |
| `gguf_bytes.py MODEL` | per-token byte accounting and byte-floor roofline from the GGUF header |
| `router_locality.py LOG --bytes bytes.json` | LRU hit rate vs cache budget, next-token overlap, from `DS4_V41_ROUTER_LOG` |
| `phase0.py prompts/run/table/report` | run the speed and locality plans with `ds4-bench` and summarize |

Runtime diagnostics (off by default, outputs unchanged):
`DS4_V41_ROUTER_LOG=<path>`, `DS4_V41_DECODE_PROFILE=1` (with
`DS4_METAL_GPU_BUSY_PROFILE=1`), plus the existing
`DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1`.

Runs need the machine free (no other ds4 process, the user's go-ahead). Raw
output goes to `~/orca/workspaces/ds4-metal-data/v41-phase0/<YYYYMMDD>/`; only
`results.csv`, `bytes.json`, the locality JSONs and `RESULTS.md` are committed,
under `speed-bench/v41/phase0-<YYYYMMDD>/`.
````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest discover -s speed-bench/tests -v`
Expected: all tests `ok` (13 new).

- [ ] **Step 5: Dry-run both plans** (no GPU)

Run: `python3 speed-bench/v41/phase0.py run --plan speed --model /m.gguf --prompts /p --out "$(mktemp -d)" --dry-run | wc -l && python3 speed-bench/v41/phase0.py run --plan locality --model /m.gguf --prompts /p --out "$(mktemp -d)" --dry-run | head -1`
Expected: `12`, then one `phase0: .../ds4-bench --metal -m /m.gguf --prompt-file /p/code.txt ... --teacher-forced-decode --ssd-streaming --ssd-streaming-cache-experts 8GB ...` line.

- [ ] **Step 6: Commit**

```bash
git add speed-bench/v41/phase0.py speed-bench/v41/README.md speed-bench/tests/test_phase0.py
git commit -m "speed-bench/v41: Phase-0 driver, collector and report

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: [GATE A] Verify the upstream sync and record the Qwen baseline

**⚠ Machine-free gate:** ask the user first. Stop and wait for a yes. The PROD
gateway's ds4 backend and every other session's ds4 process must be stopped by the
user. Do not stop them yourself.

**Files:**
- Create: `speed-bench/qwen-regression/baseline/{vi.txt,code.txt,result.json}` (from `record`)
- Modify: `speed-bench/qwen-regression/qwen_gate.py` (only the `--needle-chars` default, if Step 4 says so)

- [ ] **Step 1: Confirm the machine is free**

Run: `python3 -c 'import sys; sys.path.insert(0, "speed-bench/lib"); import machine; print(machine.ds4_running() or "free", machine.swap_used_mib())'`
Expected: `free` and a swap figure. If anything is listed, stop and tell the user.

- [ ] **Step 2: Run the sync's pending unit tests** (from memory `ds41-sync-pending-tests`)

Run: `make test-qwen4-kernels test-qwen4-q2 test-deepseek41-metal test-metal-ssd-experts test-metal-moe-prefill`
Expected: every target exits 0. On failure, STOP (spec §4) and use superpowers:systematic-debugging.

- [ ] **Step 3: Record the baseline from the PROD binary**

Run: `python3 speed-bench/qwen-regression/qwen_gate.py record --out speed-bench/qwen-regression/baseline --full`
Expected: `qwen_gate: reference recorded in speed-bench/qwen-regression/baseline`, and
`baseline/result.json` holding `tps_median`, `wired.steady_gib` and `"needle_hit": true`.

- [ ] **Step 4: Check the needle length**

Run: `python3 -c 'import json; r = json.load(open("speed-bench/qwen-regression/baseline/result.json")); print(r["needle_prompt_tokens"], r["needle_hit"])'`
Expected: a token count between 200000 and 228000, and `True`. If it falls outside that
range, scale the `--needle-chars` default in `qwen_gate.py` by `215000 / tokens`, then
repeat Step 3. If `needle_hit` is false on PROD, stop and tell the user: the baseline
itself fails the needle.

- [ ] **Step 5: Run the full tier on this branch** (sync merges + Tasks 4–5)

Run: `speed-bench/qwen-regression/run.sh full`
Expected: `qwen_gate: PASS`. On `FAIL`, STOP DS4.1 work (spec §4). Report which check
failed. Invoke superpowers:systematic-debugging and bisect across `ec56a05`, `d3bf293`
and the Task 4/5 commits.

- [ ] **Step 6: Commit the baseline and close the pending-test note**

```bash
git add speed-bench/qwen-regression/baseline/vi.txt speed-bench/qwen-regression/baseline/code.txt speed-bench/qwen-regression/baseline/result.json speed-bench/qwen-regression/qwen_gate.py
git commit -m "speed-bench: Qwen3.8 regression baseline from the PROD binary

Sync merges ec56a05/d3bf293 and the V4.1 diagnostics pass the full tier.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
Update memory `ds41-sync-pending-tests.md`: tests done, with the result and date.

---

### Task 9: [GATE B] Download the pinned Q2, byte accounting, model checks

**⚠ Machine-free gate:** ask the user first. The download writes 341 GiB to the SSD
and would disturb other sessions' SSD-streaming benchmarks. It can share one window
with Task 8.

**Files:**
- Create: `speed-bench/v41/phase0-<YYYYMMDD>/bytes.json`

- [ ] **Step 1: Check free disk**

Run: `df -g ~ | awk 'NR==2 {print $4 " GiB free"}'`
Expected: at least 360 GiB free. If it is lower, stop and ask the user what to remove.

- [ ] **Step 2: Download and verify**

Run: `DS4_GGUF_DIR=$HOME/orca/workspaces/ds4-metal-data/gguf ./download_model.sh ds41f-q2`
Expected: ends after `Verifying SHA-256: .../DeepSeek-V4.1-Flash-Q2.gguf` with no
mismatch error. `ds4flash.gguf` in the worktree now links to the file (gitignored).
The script checks size 365713686528 and SHA `1ce6a8f8…6f42`. If it is interrupted,
rerun the same command to resume.

- [ ] **Step 3: Byte accounting on the real file**

Run:
```bash
M=$HOME/orca/workspaces/ds4-metal-data/gguf/DeepSeek-V4.1-Flash-Q2.gguf
D=speed-bench/v41/phase0-$(date +%Y%m%d)
mkdir -p "$D" && python3 speed-bench/v41/gguf_bytes.py "$M" --out "$D/bytes.json"
```
Expected:
- `tensors=1046 layers=40 experts=384 k=6`.
- Resident floor ≈ 8.79 GiB, routed ≈ 142.38 GiB and engram_disk ≈ 189 GiB (spec §3).
- A roofline line near the spec §2.1 estimate (≈ 34–36 ms/token at 290 GB/s).
- If `engram_tables` is not 2, note it for Task 11: `ds41_graph_step` reads two
  tables per token.

- [ ] **Step 4: Model checks** (GPU; watch `sysctl vm.swapusage` in a second shell and abort if swap grows by more than 2 GiB)

Run:
```bash
make tests/test_deepseek41_graph
./tests/test_deepseek41_graph "$M" --router-log speed-bench/promessi_sposi.txt
./tests/test_deepseek41_graph "$M" --engram-parallel-ssd speed-bench/promessi_sposi.txt
```
Expected:
- `V4.1 router log: 1280 lines; logits identical with and without it PASS`
  (32 steps × 40 layers).
- `V4.1 decode control DS4_METAL_DISABLE_V41_ENGRAM_PARALLEL SSD prefix=... PASS`
  for both prefixes. This validates the Engram conflict resolution in `d3bf293`.
- If the Engram check fails on memory (its upstream fixture asks for a 64 GiB
  cache), report the error to the user. Do not edit the upstream test.

- [ ] **Step 5: Commit**

```bash
git add speed-bench/v41/phase0-*/bytes.json
git commit -m "speed-bench/v41: byte accounting of the pinned V4.1 Q2 GGUF

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: [GATE C] Run the Phase-0 plans

**⚠ Machine-free gate:** ask the user first. It takes about 2–3 h of exclusive GPU
and SSD time. The driver refuses to start a run while any ds4 process exists, and
marks a run `contaminated` if one appears during it.

**Files:**
- Create: `~/orca/workspaces/ds4-metal-data/v41-phase0/<YYYYMMDD>/` (raw, not in git)
- Create: `speed-bench/v41/phase0-<YYYYMMDD>/{results.csv,*.locality.json}`

- [ ] **Step 1: Prompts and dry run**

Run:
```bash
M=$HOME/orca/workspaces/ds4-metal-data/gguf/DeepSeek-V4.1-Flash-Q2.gguf
R=$HOME/orca/workspaces/ds4-metal-data/v41-phase0/$(date +%Y%m%d)
make ds4-bench
python3 speed-bench/v41/phase0.py prompts --out "$R/prompts"
python3 speed-bench/v41/phase0.py run --plan locality --model "$M" --prompts "$R/prompts" --out "$R" --dry-run
```
Expected: four prompt files of 200 000 chars, then four `ds4-bench` command lines.

- [ ] **Step 2: Locality plan** (4 runs, 2000 decode tokens each)

Run: `python3 speed-bench/v41/phase0.py run --plan locality --model "$M" --prompts "$R/prompts" --out "$R"`
Expected: four `phase0: done <tag> <t/s>` lines. Each `$R/<tag>.router.log` holds
2000 × 40 lines plus the header. If a run fails, read `$R/<tag>.stderr`. Report
out-of-memory failures to the user instead of shrinking the plan silently.

- [ ] **Step 3: Speed plan** (12 runs)

Run: `python3 speed-bench/v41/phase0.py run --plan speed --model "$M" --prompts "$R/prompts" --out "$R"`
Expected: twelve `phase0: done` lines. Rerunning the command skips finished runs.

- [ ] **Step 4: Tables and locality**

Run:
```bash
python3 speed-bench/v41/phase0.py table --out "$R"
D=speed-bench/v41/phase0-$(date +%Y%m%d)
for log in "$R"/*.router.log; do
  tag=$(basename "$log" .router.log)
  python3 speed-bench/v41/router_locality.py "$log" --bytes "$D/bytes.json" --out "$D/$tag.locality.json" > /dev/null
done
cp "$R/results.csv" "$D/results.csv"
python3 speed-bench/v41/phase0.py report --out "$R" --bytes "$D/bytes.json" --locality "$D"/*.locality.json
cp "$R/RESULTS.generated.md" "$D/RESULTS.generated.md"
```
Expected:
- `results.csv` has 16 rows.
- There are four locality JSONs.
- `RESULTS.generated.md` has the decode, roofline and locality tables, and its "Best
  clean run" line is not `none`. If every row is flagged, report that to the user.

- [ ] **Step 5: Commit the summaries**

```bash
git add speed-bench/v41/phase0-*/results.csv speed-bench/v41/phase0-*/*.locality.json speed-bench/v41/phase0-*/RESULTS.generated.md
git commit -m "speed-bench/v41: Phase 0(b) measurements on M5 Pro 64 GB

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Results, measured roofline and the target decision

**Files:**
- Create: `speed-bench/v41/phase0-<YYYYMMDD>/RESULTS.md`
- Modify: `docs/V41_64GB_BUILD.md` (§0 status, §2.1 measured column, DoD 2, §7 output)

- [ ] **Step 1: Write `RESULTS.md`**

Start from `RESULTS.generated.md` and add these sections, each built from the committed
files:
1. **Setup:**
   - model SHA from Task 9;
   - `git rev-parse HEAD`;
   - date;
   - diagnostics env;
   - cache sizes as `ds4-bench` reported them (`cache_experts`).
2. **Per-token decomposition vs the spec §2.1 estimate.** For the best clean `switch`
   row, compare `gpu_busy_ms` with the byte floor (`resident_ms + routed_ms`), and
   `pread_ms + host_gap_ms` with the estimated 12 ms sync + miss term. Give the ratio
   of `gpu_busy_ms` to the byte floor: that is the kernel-efficiency factor.
3. **Cache size:**
   - Does 4 → 24 GB reproduce the #810 shape (small cache as fast or faster)?
   - Quote `decode_hit_rate` and `wired_steady_gib` per cache size.
4. **Locality:**
   - LRU hit rate at 8 / 16 GB per workload vs the spec §7 thresholds (≥93 %, ~85 %, <80 %);
   - `union_cover` K=1..8 as the predictive-prefetch ceiling.
5. **Largest term.** Say which term dominates per-token time. That is the first lever
   for spec §8 step 3 onward.
6. **Target options.** Three candidate DoD-2 targets:
   - (a) the measured best clean t/s rounded down to its band;
   - (b) the band reachable if the largest term shrinks by its measured headroom;
   - (c) what needs levers above the roofline (DSpark/MTP, cheaper non-routed weights).

   Put the numbers behind each one in a table.

- [ ] **Step 2: Present the options to the user and wait**

Summarize RESULTS.md in chat (in Vietnamese) and ask the user to pick the target
(a/b/c or a custom number). This is decision "C" from the spec. Do not pick for them.

- [ ] **Step 3: Record the decision in the spec**

In `docs/V41_64GB_BUILD.md`:
- DoD 2: replace "the target chosen at the end of Phase 0" with the chosen t/s and
  band, citing RESULTS.md.
- §2.1: add a "Measured (Phase 0)" column next to the estimate.
- §0: set "Phase 0(b) DONE <date>".
- §7 Output: link RESULTS.md.

- [ ] **Step 4: Verify and commit**

Run: `python3 -m unittest discover -s speed-bench/tests -v && ./tests/test_deepseek41_graph --router-log-format && ./tests/test_deepseek41_graph --decode-profile-format`
Expected: all pass.

```bash
git add speed-bench/v41/phase0-*/RESULTS.md docs/V41_64GB_BUILD.md
git commit -m "docs: V4.1 Phase 0(b) results and the chosen tok/s target

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
Update the project memory (`ds41-upstream-work-and-64gb-gap.md`) with the measured
numbers and the chosen target.
