# Ornith qwen35moe M1: Plain Inference Correct — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load the Ornith-1.5-35B-A3B `qwen35moe` ICE GGUF in ds4 and run plain (non-speculative) Metal inference whose greedy tokens and top-20 log-probabilities match llama.cpp on the same file, without changing the Qwen3.8 production path.

**Architecture:** A new model family `DS4_MODEL_FAMILY_QWEN35_MOE` with its own shape, validator and weight binding in `ds4.c`, and its own plain-residual graph in `ds4_qwen35moe.inc`. The graph reuses the qwen4 graph struct, GEMVs, GDN, router, MoE and attention kernels. The new GPU pieces are additive: a Q5_K expert row-dot with mid/down kernels, a silu-gated GDN output kernel, and an attention-prep wrapper that dispatches only the query and KV slots of the existing qwen4 prep kernel. They live in a new `metal/qwen35.metal`, appended after `metal/qwen4.metal` in the single Metal library. llama.cpp (Homebrew) is the correctness oracle.

**Tech Stack:** C (ds4.c), Objective-C (ds4_metal.m), Metal Shading Language, Python 3 stdlib (oracle and comparison scripts), llama.cpp `llama-server` / `llama-perplexity`.

**Spec:** `docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md` (this plan is milestone M1 of §9; M2 MTP, M3 serving and M4 acceptance get their own plans).

## Global Constraints

- The Qwen3.8 production path stays byte-identical and as fast as today (spec §1, §7).
- Metal only for Ornith. SSD streaming, CUDA/ROCm/CPU inference, distributed/TP, `--batched-session` above 1, directional steering, `--ple`, `--vision` and `--mtp` are refused at open with a message naming the option (spec §6; `--mtp` is added in M2).
- Accepted tensor types: routed experts Q4_K or Q5_K with gate and up of the same type; dense projections and shared experts Q8_0; `attn_k`/`attn_v` F16 or Q8_0; norms, router, `ssm_*` scalars and conv F32. Anything else fails the load with the tensor name (spec §4).
- `DS4_QWEN4_*` environment variables are read only on the qwen4exp path; Ornith uses the `DS4_QWEN35_*` prefix (spec §7).
- KV cache F16 (spec §5).
- Kernel changes are additive: new kernels or entry points only; `metal/qwen4.metal` is not edited (spec §7).
- Code, comments, docs and commit messages in English. Model files never go into git; GGUFs live under `~/orca/workspaces/ds4-metal-data/gguf/ornith/`.
- One model process at a time on this 64 GB machine. Never `kill -9` a hung Metal process (see memory `metal-kill9-wedges-gguf-vnode`). Run long jobs under `caffeinate -i -s`. Check `ps -axo pid,stat,comm | awk '$2 ~ /E|U/'` before starting a model run.
- No C++. Follow AGENT.md: small, readable code; comments explain why.
- Every commit ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2
  ```

Two deviations from the spec wording, decided while planning:

1. Spec §3 puts weight binding in the `.inc`. The binding and layout validation are called from `weights_bind()` around `ds4.c:7925`, long before the qwen4 graph block where the `.inc` is included (`ds4.c:60273`). They stay beside their qwen4 counterparts in `ds4.c`; the `.inc` holds the graph, one-shot generation and session helpers. Task 5 updates the spec sentence.
2. `DS4_HAS_QWEN4_GPU` is also defined for CUDA builds. All Ornith GPU code, including the `.inc` include and every session branch, sits under `#ifdef DS4_HAS_QWEN4_METAL` so CUDA and ROCm builds do not reference Metal-only wrappers.

## Review Focus

1. **A prompt longer than one prefill chunk, then extended by later requests.** The live session must equal a fresh session that replays the same sequence of syncs (max |Δ| < 1e-3, same argmax). Covered by Task 7, `tests/test_qwen35_session.c`, with lengths crossing the 512-token chunk boundaries. A one-pass prefill of the same prompt chunks differently and can flip a near tie, so its difference is only reported, as `tests/test_qwen4_prefill.c` does.
2. **Q4_K layers exactly at the tiled-GEMM threshold** (64 rows per-token kernels, 65 rows tile GEMM). Both must match llama.cpp. Covered by Task 8, Step 4, runs with `--prefill-chunk 64` and `--prefill-chunk 65`.
3. **A second, unrelated prompt in the same session.** The recurrent state must reset; logits must be bit-identical to a fresh session. Covered by Task 7, `test_qwen35_session.c` "divergent prompt" case.
4. **Context exhausted during generation.** Generation stops cleanly and a further eval returns an error, with no crash or out-of-range write. Covered by Task 7, "context full" case.
5. **An unsupported tier or wrong metadata.** Loading fails with a message naming the key or the tier, never a silent fallback into DeepSeek code. Covered by Task 5, `tests/ornith/test_loader.sh`.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `speed-bench/qwen-regression/` | Import from local `develop` | Qwen3.8 regression gate (`run.sh fast|full`) |
| `tests/ornith/prompts.json` | Create | The M1 prompt set |
| `tests/ornith/ornith_ref.py` | Create | Shared helpers: prompt text, llama.cpp and ds4 JSON normalisation, comparison, calibration |
| `tests/ornith/llama_ref.py` | Create | Record llama.cpp references and calibrate tolerances |
| `tests/ornith/gate1.py` | Create | Run ds4 on every prompt and compare with the references |
| `tests/ornith/test_ornith_ref.py` | Create | Unit tests for the comparison logic |
| `tests/ornith/make_bad_gguf.py` | Create | Sparse copies of the real GGUF with one metadata value or one expert type patched |
| `tests/ornith/test_loader.sh` | Create | Loader acceptance and refusal checks |
| `tests/ornith/ref/*.json`, `tests/ornith/tolerance.json`, `tests/ornith/ref/ORACLE.txt` | Generate + commit | llama.cpp references, calibrated tolerances, oracle version |
| `metal/qwen35.metal` | Create | Q5_K row-dot, `kernel_qwen35_moe_mid/down`, `kernel_qwen35_gdn_out` |
| `ds4_metal.m` | Modify | Source list entry, kernel enum/names, four `ds4_gpu_qwen35_*` wrappers |
| `ds4_gpu.h` | Modify | Declarations of the four wrappers |
| `tests/test_qwen35_kernels.c` | Create | Kernel tests against double-precision references |
| `ds4.c` | Modify | Family enum, shape, predicates, validator, binding, layout, output head, tokenizer/vocab sites, memory estimate, open gate, session dispatch, generate dispatch, `.inc` include, `gdn_silu` hook |
| `ds4.h` | Modify | `ds4_engine_is_qwen35moe()` |
| `ds4_qwen35moe.inc` | Create | Ornith graph: alloc/reset, inputs, attention, MoE, layer loop, forward, one-shot generate |
| `tests/test_qwen35_session.c` | Create | Real-model session checks (chunk replay, divergent prompt, context full) |
| `Makefile` | Modify | `test-qwen35-kernels`, `test-qwen35-session`, `.inc` dependency, clean |
| `speed-bench/ornith/oracle/RESULTS.md`, `speed-bench/ornith/m1/RESULTS.md` | Create | Receipts |
| `.gitignore` | Modify | Ignore per-run ds4 dumps and logs under `speed-bench/ornith/` |

---

### Task 1: Tooling, model file and oracle self-check

No production code. The deliverable is a verified oracle: llama.cpp reproduces the PPL the GGUF repo publishes for the 23G tier.

**Files:**
- Import: `speed-bench/qwen-regression/` (from local `develop`)
- Create: `speed-bench/ornith/oracle/RESULTS.md`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `$DS4_ORNITH_MODEL` = `~/orca/workspaces/ds4-metal-data/gguf/ornith/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf`; `llama-server` and `llama-perplexity` on `PATH`; `speed-bench/qwen-regression/run.sh`.

- [ ] **Step 1: Import the Qwen3.8 regression gate from local `develop`**

The gate lives only on local `develop` (commits 2dd77de, 1f8912f, 258690d); `origin/develop` 3673192 does not have it.

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/foxface
git checkout develop -- speed-bench/qwen-regression
git status --short speed-bench/qwen-regression | head
```

Expected: `A` lines for `README.md`, `qwen_gate.py`, `run.sh` and the `baseline/` files.

- [ ] **Step 2: Ignore per-run ornith dumps**

Append to `.gitignore`:

```gitignore
# Ornith (qwen35moe) run artefacts: keep RESULTS.md and compare output only
speed-bench/ornith/*/ds4/
speed-bench/ornith/*/logs/
speed-bench/ornith/oracle/ref-cpu/
```

- [ ] **Step 3: Commit the gate import**

```bash
git add speed-bench/qwen-regression .gitignore
git commit -m "speed-bench: import the Qwen3.8 regression gate for the Ornith branch

The gate lives on local develop only; the Ornith work is based on
origin/develop and needs it for every shared-code change.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- speed-bench/qwen-regression .gitignore
git show --stat HEAD | head -20
```

- [ ] **Step 4: Install llama.cpp and record its version**

```bash
brew install llama.cpp
llama-server --version 2>&1 | head -3
llama-perplexity --version 2>&1 | head -3
```

Expected: both print a build line (for example `version: NNNN (abcdef0)`). Keep that line for Step 7.

- [ ] **Step 5: Download the 23G tier and the code corpus**

Disk had 96 GiB free on 2026-09-25; the file is 22.84 GB. Check `df -h /Users/dongnh` first and stop if under 40 GiB free.

```bash
mkdir -p ~/orca/workspaces/ds4-metal-data/gguf/ornith
cd ~/orca/workspaces/ds4-metal-data/gguf/ornith
R=gbuzhf/Ornith-1.5-35B-A3B-Abliterated-CyberTiel-Calibrated-MTPv2-ICE-GGUF
~/.local/omlx-venv/bin/hf download "$R" \
  Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf \
  corpora/code.test.raw --local-dir .
ls -l Ornith-*23G-ICE.gguf corpora/code.test.raw
```

Expected size: `22836518208` bytes for the GGUF. Do not `pip install` into `~/.local/omlx-venv` (it is the PROD oMLX environment).

- [ ] **Step 6: Oracle self-check with llama-perplexity**

Stop the gateway's model backends first (oMLX and ds4 slots) and check that nothing else holds the GPU.

```bash
ps -axo pid,stat,comm | awk 'NR==1 || /ds4|omlx|llama/' | grep -v awk
ps -axo pid,stat,comm | awk '$2 ~ /^(E|U)/'
cd ~/orca/workspaces/ds4-metal-data/gguf/ornith
caffeinate -i -s llama-perplexity \
  -m Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf \
  -f corpora/code.test.raw -c 2048 --chunks 64 -ngl 99 -fa on 2>&1 | tee /tmp/ornith-ppl.log | tail -3
```

Expected: `Final estimate: PPL = 2.19...`. The published values give 2.194208 x 1.0018 = 2.1982 (repo `measurements/code/kld_23G-ICE.txt`). Accept 2.187..2.209 (±0.5 %; builds differ). Outside that range: stop and report. The oracle is not trusted until this passes.

- [ ] **Step 7: Write the oracle receipt and commit**

Create `speed-bench/ornith/oracle/RESULTS.md`:

```markdown
# Ornith 23G ICE: llama.cpp oracle self-check

- Model: `Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf`, 22,836,518,208 bytes,
  from gbuzhf/Ornith-1.5-35B-A3B-Abliterated-CyberTiel-Calibrated-MTPv2-ICE-GGUF.
- llama.cpp: <paste the Step 4 version line>, Homebrew, Metal (`-ngl 99 -fa on`).
- Corpus: repo `corpora/code.test.raw`, 64 chunks, n_ctx 2048.
- Published: PPL(BF16) 2.194208, 23G PPL ratio 1.0018 -> 2.1982.
- Measured: <paste the Final estimate line>.
- Verdict: <PASS if within 2.187..2.209, else FAIL>.
```

```bash
git add speed-bench/ornith/oracle/RESULTS.md
git commit -m "speed-bench/ornith: llama.cpp oracle self-check on the 23G tier

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- speed-bench/ornith/oracle/RESULTS.md
```

---

### Task 2: Reference recording and comparison scripts

Python stdlib only. The comparison logic is unit-tested before any model run.

**Files:**
- Create: `tests/ornith/prompts.json`, `tests/ornith/ornith_ref.py`, `tests/ornith/llama_ref.py`, `tests/ornith/test_ornith_ref.py`
- Generate: `tests/ornith/ref/*.json`, `tests/ornith/ref/ORACLE.txt`, `tests/ornith/tolerance.json`

**Interfaces:**
- Produces (module `tests/ornith/ornith_ref.py`):
  - `load_prompts(path) -> list[dict]`
  - `prompt_text(prompt: dict, root: str) -> str`
  - `llama_steps(completion: dict) -> list[dict]`: each step is `{"selected": int, "top": [[id, logprob], ...]}`
  - `ds4_steps(dump: dict) -> list[dict]`: same form
  - `compare(ref_steps, got_steps, tol: float, tie: float) -> dict` with keys `ok`, `compared`, `stopped_at_tie`, `max_delta`, `first_mismatch`, `reason`
  - `calibrate(metal: list[list[dict]], cpu: list[list[dict]]) -> dict` with keys `tol`, `tie`, `observed_max_delta`, `observed_divergence_gap`
- Produces (files): `tests/ornith/ref/<name>.json` = `{"name", "prompt_ids", "steps"}`; `tests/ornith/tolerance.json` = `{"tol", "tie", ...}`.

- [ ] **Step 1: Write the prompt set**

Create `tests/ornith/prompts.json`:

```json
[
  {"name": "en_capital", "text": "The capital of France is", "n_predict": 64},
  {"name": "en_story", "text": "Once upon a time, in a small village by the sea, there lived", "n_predict": 64},
  {"name": "en_explain", "text": "Explain in two sentences why the sky is blue:", "n_predict": 64},
  {"name": "vi_greeting", "text": "Xin chào! Hôm nay tôi muốn kể cho bạn nghe về", "n_predict": 64},
  {"name": "vi_hanoi", "text": "Hà Nội là thủ đô của Việt Nam. Thành phố này", "n_predict": 64},
  {"name": "code_py", "text": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n", "n_predict": 64},
  {"name": "code_c", "text": "#include <stdio.h>\n\nint main(void) {\n", "n_predict": 64},
  {"name": "code_rust", "text": "fn main() {\n    let v: Vec<i32> = (1..=10).collect();\n", "n_predict": 64},
  {"name": "json_obj", "text": "{\"name\": \"Alice\", \"age\": 30, \"city\":", "n_predict": 32},
  {"name": "math_mul", "text": "Q: What is 17 * 23?\nA:", "n_predict": 32},
  {"name": "list_tips", "text": "Three tips for writing clean code:\n1.", "n_predict": 64},
  {"name": "long_it", "file": "speed-bench/promessi_sposi.txt", "chars": 32000, "n_predict": 16}
]
```

- [ ] **Step 2: Write the failing unit tests**

Create `tests/ornith/test_ornith_ref.py`:

```python
"""Unit tests for the Ornith reference comparison (python3 -m unittest)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import ornith_ref as r


def step(sel, *pairs):
    return {"selected": sel, "top": [list(p) for p in pairs]}


class CompareTest(unittest.TestCase):
    def test_identical_passes(self):
        ref = [step(5, (5, -0.1), (7, -3.0)), step(9, (9, -0.2), (2, -2.5))]
        res = r.compare(ref, ref, tol=0.05, tie=0.05)
        self.assertTrue(res["ok"])
        self.assertEqual(res["compared"], 2)
        self.assertEqual(res["max_delta"], 0.0)

    def test_selection_mismatch_fails(self):
        ref = [step(5, (5, -0.1), (7, -3.0))]
        got = [step(7, (7, -0.1), (5, -3.0))]
        res = r.compare(ref, got, tol=0.05, tie=0.05)
        self.assertFalse(res["ok"])
        self.assertEqual(res["first_mismatch"], 0)

    def test_mismatch_after_near_tie_is_allowed(self):
        ref = [step(5, (5, -0.10), (7, -3.0)), step(1, (1, -0.69), (2, -0.70)), step(4, (4, -0.1))]
        got = [step(5, (5, -0.10), (7, -3.0)), step(2, (2, -0.69), (1, -0.70)), step(8, (8, -0.1))]
        res = r.compare(ref, got, tol=0.05, tie=0.05)
        self.assertTrue(res["ok"])
        self.assertEqual(res["stopped_at_tie"], 1)
        self.assertEqual(res["compared"], 1)

    def test_logprob_delta_above_tolerance_fails(self):
        ref = [step(5, (5, -0.1), (7, -3.0))]
        got = [step(5, (5, -0.1), (7, -2.5))]
        res = r.compare(ref, got, tol=0.05, tie=0.05)
        self.assertFalse(res["ok"])
        self.assertAlmostEqual(res["max_delta"], 0.5)

    def test_shorter_output_fails_unless_stopped(self):
        ref = [step(5, (5, -0.1), (7, -3.0)), step(9, (9, -0.2), (2, -2.5))]
        got = [step(5, (5, -0.1), (7, -3.0))]
        res = r.compare(ref, got, tol=0.05, tie=0.05)
        self.assertFalse(res["ok"])

    def test_calibrate_uses_floors(self):
        metal = [[step(5, (5, -0.1), (7, -3.0))]]
        cpu = [[step(5, (5, -0.101), (7, -3.002))]]
        cal = r.calibrate(metal, cpu)
        self.assertAlmostEqual(cal["tol"], 0.05)
        self.assertAlmostEqual(cal["tie"], 0.05)
        self.assertAlmostEqual(cal["observed_max_delta"], 0.002, places=6)


class NormaliseTest(unittest.TestCase):
    def test_llama_steps(self):
        completion = {"completion_probabilities": [
            {"id": 5, "token": "a", "logprob": -0.1,
             "top_logprobs": [{"id": 5, "token": "a", "logprob": -0.1},
                              {"id": 7, "token": "b", "logprob": -3.0}]}]}
        self.assertEqual(r.llama_steps(completion), [step(5, (5, -0.1), (7, -3.0))])

    def test_ds4_steps(self):
        dump = {"steps": [{"step": 0, "selected": {"id": 5, "text": "a", "bytes": [97]},
                           "top_logprobs": [{"token": {"id": 5}, "logit": 9.0, "logprob": -0.1},
                                            {"token": {"id": 7}, "logit": 6.1, "logprob": -3.0}]}]}
        self.assertEqual(r.ds4_steps(dump), [step(5, (5, -0.1), (7, -3.0))])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `python3 -m unittest tests/ornith/test_ornith_ref.py -v`
Expected: `ModuleNotFoundError: No module named 'ornith_ref'`.

- [ ] **Step 4: Implement `tests/ornith/ornith_ref.py`**

```python
"""Shared helpers for the Ornith (qwen35moe) correctness gate.

References come from llama.cpp's llama-server on the same GGUF; ds4 output
comes from `ds4 --dump-logprobs`.  Both are reduced to the same step form,
{"selected": id, "top": [[id, logprob], ...]}, and compared greedily: the
selected tokens must match and every token in both top-k lists must agree
within a tolerance, until the reference reaches a near-tie (top-1/top-2 gap
below `tie`), after which a different but equally valid continuation is
allowed.  The tolerances are calibrated from llama.cpp's own Metal-versus-CPU
spread instead of being picked by hand.
"""
import json
import os

FLOOR = 0.05  # log-prob; the smallest tolerance and tie gap the gate uses


def load_prompts(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def prompt_text(prompt, root):
    if "text" in prompt:
        return prompt["text"]
    with open(os.path.join(root, prompt["file"]), encoding="utf-8") as f:
        return f.read()[: prompt["chars"]]


def llama_steps(completion):
    steps = []
    for s in completion["completion_probabilities"]:
        tops = s.get("top_logprobs") or []
        steps.append({"selected": s["id"], "top": [[t["id"], t["logprob"]] for t in tops]})
    return steps


def ds4_steps(dump):
    steps = []
    for s in dump["steps"]:
        steps.append({"selected": s["selected"]["id"],
                      "top": [[t["token"]["id"], t["logprob"]] for t in s["top_logprobs"]]})
    return steps


def _gap(step):
    lps = sorted((lp for _, lp in step["top"]), reverse=True)
    return lps[0] - lps[1] if len(lps) >= 2 else float("inf")


def compare(ref_steps, got_steps, tol, tie):
    res = {"ok": True, "compared": 0, "stopped_at_tie": None, "max_delta": 0.0,
           "first_mismatch": None, "reason": ""}
    for i, ref in enumerate(ref_steps):
        if _gap(ref) < tie:
            res["stopped_at_tie"] = i
            return res
        if i >= len(got_steps):
            res.update(ok=False, first_mismatch=i, reason=f"ds4 stopped after {len(got_steps)} steps")
            return res
        got = got_steps[i]
        mine = dict((tid, lp) for tid, lp in got["top"])
        for tid, lp in ref["top"]:
            if tid in mine:
                res["max_delta"] = max(res["max_delta"], abs(mine[tid] - lp))
        if got["selected"] != ref["selected"]:
            res.update(ok=False, first_mismatch=i,
                       reason=f"step {i}: selected {got['selected']} != {ref['selected']}")
            return res
        if res["max_delta"] > tol:
            res.update(ok=False, first_mismatch=i,
                       reason=f"step {i}: logprob delta {res['max_delta']:.4f} > {tol:.4f}")
            return res
        res["compared"] = i + 1
    return res


def calibrate(metal, cpu):
    delta, gap = 0.0, 0.0
    for m_steps, c_steps in zip(metal, cpu):
        res = compare(m_steps, c_steps, tol=float("inf"), tie=0.0)
        delta = max(delta, res["max_delta"])
        if res["first_mismatch"] is not None and res["first_mismatch"] < len(m_steps):
            gap = max(gap, _gap(m_steps[res["first_mismatch"]]))
    return {"tol": max(FLOOR, 3.0 * delta), "tie": max(FLOOR, 2.0 * gap),
            "observed_max_delta": delta, "observed_divergence_gap": gap}
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `python3 -m unittest tests/ornith/test_ornith_ref.py -v`
Expected: `Ran 8 tests ... OK`.

- [ ] **Step 6: Write `tests/ornith/llama_ref.py`**

```python
#!/usr/bin/env python3
"""Record llama.cpp references for the Ornith gate, or calibrate tolerances.

  llama_ref.py record    MODEL OUT_DIR [--cpu]
  llama_ref.py calibrate METAL_DIR CPU_DIR TOLERANCE_JSON

`record` starts llama-server on port 18190 (never a gateway port), tokenizes
every prompt with the server (no BOS, like ds4's --raw), and stores the
prompt ids, the generated text and, per greedy step, the selected id and
top-20 log-probabilities of the unsampled distribution.  --cpu runs the same
model on llama.cpp's CPU backend for calibration and skips the long
file-backed prompt, which would take hours there.
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
    cmd = ["llama-server", "-m", model, "--host", "127.0.0.1", "--port", str(PORT),
           "-c", "16384", "-np", "1"]
    cmd += ["-ngl", "0", "--device", "none"] if cpu else ["-ngl", "99"]
    log = open(os.path.join(out_dir, "llama-server.log"), "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_ready(proc)
        for p in r.load_prompts(os.path.join(ROOT, "tests/ornith/prompts.json")):
            if cpu and "file" in p:
                continue
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
```

- [ ] **Step 7: Record the Metal references**

```bash
export DS4_ORNITH_MODEL=~/orca/workspaces/ds4-metal-data/gguf/ornith/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf
caffeinate -i -s python3 tests/ornith/llama_ref.py record "$DS4_ORNITH_MODEL" tests/ornith/ref
ls tests/ornith/ref | wc -l
```

Expected: 12 lines of `name: N prompt tokens, M steps`, and 13 files in `tests/ornith/ref` (12 JSON + `llama-server.log`). If `completion_probabilities` is missing from the response, the llama.cpp build uses a different field; print one raw response, fix `llama_steps`, add a unit test for the new shape, and rerun.

- [ ] **Step 8: Record CPU references for the short prompts and calibrate**

The CPU run is slow; `--cpu` skips the long file-backed prompt.

```bash
caffeinate -i -s python3 tests/ornith/llama_ref.py record "$DS4_ORNITH_MODEL" speed-bench/ornith/oracle/ref-cpu --cpu
grep -i -E 'cpu|metal' speed-bench/ornith/oracle/ref-cpu/llama-server.log | head -5
python3 tests/ornith/llama_ref.py calibrate tests/ornith/ref speed-bench/ornith/oracle/ref-cpu tests/ornith/tolerance.json
```

Expected: 11 recorded prompts; the server log shows the CPU backend only (no Metal device lines); `tolerance.json` prints `tol` and `tie`, each at least 0.05. If `--device none` is not accepted by this llama.cpp build, use `-ngl 0` alone and confirm in the log that no layer was offloaded.

- [ ] **Step 9: Record the oracle version and commit**

```bash
llama-server --version 2>&1 | head -1 > tests/ornith/ref/ORACLE.txt
rm tests/ornith/ref/llama-server.log
git add tests/ornith
git commit -m "tests/ornith: llama.cpp references and calibrated tolerances for Ornith

Greedy steps with top-20 log-probabilities for 12 raw prompts, recorded
from llama-server on the 23G ICE GGUF; tolerances come from llama.cpp's own
Metal-versus-CPU spread.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- tests/ornith
```

---

### Task 3: Q5_K routed-expert kernels

**Files:**
- Create: `metal/qwen35.metal`, `tests/test_qwen35_kernels.c`
- Modify: `ds4_metal.m` (source list ~4926, enum ~48658, names ~48780, wrappers after `ds4_gpu_qwen4_moe_down_tensor`), `ds4_gpu.h`, `Makefile`

**Interfaces:**
- Produces (C, `ds4_gpu.h`); same parameter lists as `ds4_gpu_qwen4_moe_mid_tensor` / `ds4_gpu_qwen4_moe_down_tensor`:
  ```c
  int ds4_gpu_qwen35_moe_mid_tensor(
          ds4_gpu_tensor *mid, const ds4_gpu_tensor *x, const ds4_gpu_tensor *selected,
          const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset,
          uint32_t weight_type, uint32_t n_total_expert, uint32_t n_tokens, uint32_t n_slots,
          uint32_t in_dim, uint32_t ff_dim,
          uint64_t shared_gate_offset, uint64_t shared_up_offset, uint32_t shared_type);
  int ds4_gpu_qwen35_moe_down_tensor(
          ds4_gpu_tensor *part, const ds4_gpu_tensor *mid, const ds4_gpu_tensor *selected,
          const void *model_map, uint64_t model_size, uint64_t down_offset,
          uint32_t weight_type, uint32_t n_total_expert, uint32_t n_tokens, uint32_t n_slots,
          uint32_t ff_dim, uint32_t out_dim,
          uint64_t shared_down_offset, uint32_t shared_type);
  ```
- Produces (Metal): `qwen35_row_dot(row, x, weight_type, in_dim, tiisg)`. It handles type 13 (Q5_K) itself and delegates every other type to `qwen4_row_dot`.

- [ ] **Step 1: Write the failing kernel test**

Create `tests/test_qwen35_kernels.c`. Copy these helpers verbatim from `tests/test_qwen4_kernels.c`; they are test-local statics, so copying is the project pattern:
- lines 22-96: `g_rng`, `frand`, `require_ok`, `check_close`, `f32_to_f16`, `f16_to_f32`, `sigmoid_d`, `silu_d`;
- lines 99-128: `arena_t`, `arena_alloc`, `arena_f32`;
- lines 451-493: `arena_q8_0`, `upload`, `download`;
- lines 501-505: `rand_vec`.

Leave out `softplus_d` (line 97), `arena_f16` and `check_tensor`: this test does not use them, and `-Wall -Wextra` would warn.

```c
/* GPU kernel tests for Ornith-1.5-35B-A3B (qwen35moe): the kernels in
 * metal/qwen35.metal against double-precision references.
 * Build: make test-qwen35-kernels */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "ds4.h"
#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

/* ... helpers copied verbatim from tests/test_qwen4_kernels.c:
 * g_rng/frand, require_ok, check_close, f32_to_f16, f16_to_f32,
 * sigmoid_d, silu_d, arena_t, arena_alloc, arena_f32, arena_q8_0,
 * upload, download, rand_vec ... */

/* q5_K rows: 176-byte super-blocks of 256 (d, dmin, 12 packed 6-bit
 * scale/min bytes, 32 high-bit bytes, 128 nibble bytes; llama.cpp layout).
 * Element j of 32-group g stores its low nibble in qs[(g/2)*32 + j] (low
 * half for even g) and its fifth bit in bit g of qh[j].  The shadow holds the
 * dequantized weights. */
static uint64_t arena_q5_K(arena_t *a, uint64_t rows, uint64_t cols, double **shadow, float scale) {
    const uint64_t blocks = cols / 256;
    const uint64_t off = arena_alloc(a, rows * blocks * 176u);
    uint8_t *w = a->base + off;
    *shadow = malloc(rows * cols * sizeof(double));
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t b = 0; b < blocks; b++) {
            uint8_t *blk = w + (r * blocks + b) * 176u;
            const float d = scale / 63.0f / 31.0f, dmin = d;
            const uint16_t dh = f32_to_f16(d), mh = f32_to_f16(dmin);
            const float dq = f16_to_f32(dh), mq = f16_to_f32(mh);
            memcpy(blk, &dh, 2);
            memcpy(blk + 2, &mh, 2);
            uint8_t sc[8], mn[8];
            for (int g = 0; g < 8; g++) {
                sc[g] = (uint8_t)(1 + (int)(62.0f * (0.5f * frand() + 0.5f)));
                mn[g] = (uint8_t)((int)(63.0f * (0.5f * frand() + 0.5f)));
            }
            uint8_t *s = blk + 4;
            memset(s, 0, 12);
            for (int g = 0; g < 4; g++) { s[g] = sc[g] & 63; s[g + 4] = mn[g] & 63; }
            for (int g = 4; g < 8; g++) {
                s[g + 4] = (uint8_t)((sc[g] & 0xF) | ((mn[g] & 0xF) << 4));
                s[g - 4] |= (uint8_t)((sc[g] >> 4) << 6);
                s[g] |= (uint8_t)((mn[g] >> 4) << 6);
            }
            memset(blk + 16, 0, 160);
            for (int g = 0; g < 8; g++) {
                for (int j = 0; j < 32; j++) {
                    const int q = (int)(31.0f * (0.5f * frand() + 0.5f));
                    blk[48 + (g >> 1) * 32 + j] |= (uint8_t)((q & 15) << ((g & 1) * 4));
                    if (q & 16) blk[16 + j] |= (uint8_t)(1u << g);
                    (*shadow)[r * cols + b * 256 + g * 32 + j] = (double)dq * sc[g] * q - (double)mq * mn[g];
                }
            }
        }
    }
    return off;
}

/* Q5_K routed experts with a Q8_0 shared expert slot: mid and down
 * against the double reference, for decode (T=1), a verify (T=2) and a
 * small prefill batch (T=5). */
static void test_moe_q5k(arena_t *a, uint32_t NE, uint32_t slots, uint32_t E, uint32_t F, uint32_t T) {
    double *gate_w, *up_w, *down_w, *sg_w, *su_w, *sd_w;
    const uint64_t gate_off = arena_q5_K(a, (uint64_t)NE * F, E, &gate_w, 0.05f);
    const uint64_t up_off = arena_q5_K(a, (uint64_t)NE * F, E, &up_w, 0.05f);
    const uint64_t down_off = arena_q5_K(a, (uint64_t)NE * E, F, &down_w, 0.05f);
    const uint64_t sg_off = arena_q8_0(a, F, E, &sg_w, 0.05f);
    const uint64_t su_off = arena_q8_0(a, F, E, &su_w, 0.05f);
    const uint64_t sd_off = arena_q8_0(a, E, F, &sd_w, 0.05f);
    const uint32_t n_out = slots + 1;
    float *x = rand_vec((uint64_t)T * E, 1.0f);
    int32_t *sel = malloc((uint64_t)T * slots * 4);
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t s = 0; s < slots; s++) sel[t * slots + s] = (int32_t)((t * 7u + s * 3u) % NE);
    double *mid = malloc((uint64_t)T * n_out * F * sizeof(double));
    double *part = malloc((uint64_t)T * n_out * E * sizeof(double));
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t s = 0; s < n_out; s++) {
            const bool shared = s == slots;
            const uint32_t e = shared ? 0 : (uint32_t)sel[t * slots + s];
            const double *gw = shared ? sg_w : gate_w + (uint64_t)e * F * E;
            const double *uw = shared ? su_w : up_w + (uint64_t)e * F * E;
            const double *dw = shared ? sd_w : down_w + (uint64_t)e * E * F;
            for (uint32_t f = 0; f < F; f++) {
                double g = 0.0, u = 0.0;
                for (uint32_t i = 0; i < E; i++) {
                    g += gw[(uint64_t)f * E + i] * x[t * E + i];
                    u += uw[(uint64_t)f * E + i] * x[t * E + i];
                }
                mid[((uint64_t)t * n_out + s) * F + f] = silu_d(g) * u;
            }
            for (uint32_t d = 0; d < E; d++) {
                double acc = 0.0;
                for (uint32_t f = 0; f < F; f++) acc += dw[(uint64_t)d * F + f] * mid[((uint64_t)t * n_out + s) * F + f];
                part[((uint64_t)t * n_out + s) * E + d] = acc;
            }
        }
    }
    ds4_gpu_tensor *gx = upload(x, (uint64_t)T * E);
    ds4_gpu_tensor *gsel = ds4_gpu_tensor_alloc((uint64_t)T * slots * 4);
    require_ok(ds4_gpu_tensor_write(gsel, 0, sel, (uint64_t)T * slots * 4), "sel write");
    ds4_gpu_tensor *gmid = upload(NULL, (uint64_t)T * n_out * F);
    ds4_gpu_tensor *gpart = upload(NULL, (uint64_t)T * n_out * E);
    require_ok(ds4_gpu_qwen35_moe_mid_tensor(gmid, gx, gsel, a->base, a->size, gate_off, up_off, 13u, NE, T, slots,
                                             E, F, sg_off, su_off, 8u), "qwen35 moe mid");
    /* The down pass reads the reference mid, so a mid error cannot mask a down error. */
    float *mid_f = malloc((uint64_t)T * n_out * F * sizeof(float));
    for (uint64_t i = 0; i < (uint64_t)T * n_out * F; i++) mid_f[i] = (float)mid[i];
    ds4_gpu_tensor *gmid_ref = upload(mid_f, (uint64_t)T * n_out * F);
    require_ok(ds4_gpu_qwen35_moe_down_tensor(gpart, gmid_ref, gsel, a->base, a->size, down_off, 13u, NE, T, slots,
                                              F, E, sd_off, 8u), "qwen35 moe down");
    char what[64];
    snprintf(what, sizeof(what), "q5_K moe mid T=%u", T);
    {
        float *got = download(gmid, (uint64_t)T * n_out * F);
        check_close(what, got, mid, (uint64_t)T * n_out * F, 2e-4);
        free(got);
    }
    snprintf(what, sizeof(what), "q5_K moe down T=%u", T);
    {
        float *got = download(gpart, (uint64_t)T * n_out * E);
        check_close(what, got, part, (uint64_t)T * n_out * E, 2e-4);
        free(got);
    }
    ds4_gpu_tensor_free(gx); ds4_gpu_tensor_free(gsel); ds4_gpu_tensor_free(gmid);
    ds4_gpu_tensor_free(gmid_ref); ds4_gpu_tensor_free(gpart);
    free(x); free(sel); free(mid); free(part); free(mid_f);
    free(gate_w); free(up_w); free(down_w); free(sg_w); free(su_w); free(sd_w);
}

int main(void) {
    arena_t arena;
    arena.size = (uint64_t)512 << 20;
    arena.base = mmap(NULL, arena.size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    arena.used = 0;
    if (arena.base == MAP_FAILED) { perror("mmap"); return 1; }
    require_ok(ds4_gpu_init(), "GPU initialization");
    require_ok(ds4_gpu_set_model_map(arena.base, arena.size), "model map registration");

    printf("qwen35 moe (q5_K experts, q8_0 shared slot)\n");
    test_moe_q5k(&arena, 16, 8, 512, 256, 1);
    test_moe_q5k(&arena, 16, 8, 512, 256, 2);
    test_moe_q5k(&arena, 16, 8, 512, 512, 5);
    printf("qwen35 kernels: ok\n");
    return 0;
}
```

Add to `Makefile`, inside the same Darwin block that holds the `$(QWEN4_KERNEL_TEST)` rule (after `Makefile:627-628`):

```make
tests/test_qwen35_kernels.o: tests/test_qwen35_kernels.c ds4_gpu.h ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_qwen35_kernels.c

tests/test_qwen35_kernels: tests/test_qwen35_kernels.o ds4_metal.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
```

Next to the `test-qwen4-kernels` target (`Makefile:642-644`):

```make
.PHONY: test-qwen35-kernels
test-qwen35-kernels: tests/test_qwen35_kernels
	./tests/test_qwen35_kernels
```

Add `tests/test_qwen35_kernels` to the `rm -f` line of the clean target (`Makefile:1091`).

- [ ] **Step 2: Build and confirm the test fails**

Run: `make test-qwen35-kernels`
Expected: link error `Undefined symbols ... _ds4_gpu_qwen35_moe_mid_tensor`.

- [ ] **Step 3: Write `metal/qwen35.metal` (Q5_K part)**

```metal
// Ornith-1.5-35B-A3B (llama.cpp qwen35moe) kernels.  This file is appended
// after qwen4.metal in the single Metal library, so the qwen4 helpers and
// argument structs (qwen4_row_dot, qwen4_silu, qwen4_sigmoid,
// ds4_metal_args_qwen4_moe, ds4_metal_args_qwen4_gdn_out) are in scope.
// Everything here is a new entry point: the Qwen3.8 kernels are not touched.

/* --- Q5_K routed experts ------------------------------------------------ */

/* q5_K: 176-byte super-blocks of 256 (d, dmin, 12 packed 6-bit scale/min
 * pairs, 32 high-bit bytes, 128 nibble bytes).  The lane mapping and the
 * accumulation order follow the q4_K case of qwen4_row_dot: group = lane/4,
 * l = (lane%4)*8, eight consecutive elements per lane per block.  Element j
 * of group g takes its fifth bit from bit g of qh[j]. */
static inline float qwen35_row_dot(device const char *row, device const float *x,
                                   uint weight_type, uint in_dim, ushort tiisg) {
    if (weight_type != 13u) return qwen4_row_dot(row, x, weight_type, in_dim, tiisg);
    float acc = 0.0f;
    const uint nb = in_dim / 256u;
    const uint group = tiisg / 4, l = (tiisg % 4) * 8;
    for (uint ib = 0; ib < nb; ib++) {
        device const uchar *blk = (device const uchar *)(row + (uint64_t)ib * 176);
        const float d = (float)(*(device const half *)blk);
        const float dmin = (float)(*(device const half *)(blk + 2));
        device const uchar *sc = blk + 4;
        uint s, mn;
        if (group < 4) { s = sc[group] & 63u; mn = sc[group + 4] & 63u; }
        else { s = (sc[group + 4] & 0xFu) | ((sc[group - 4] & 0xC0u) >> 2); mn = (sc[group + 4] >> 4) | ((sc[group] & 0xC0u) >> 2); }
        const float ds = d * (float)s, dm = dmin * (float)mn;
        device const uchar *qh = blk + 16 + l;
        device const uchar *qs = blk + 48 + (group >> 1) * 32 + l;
        const uint shift = (group & 1u) * 4u;
        device const float *y = x + ib * 256 + group * 32 + l;
        for (uint i = 0; i < 8; i++) {
            const uint q = ((qs[i] >> shift) & 0xFu) | (((qh[i] >> group) & 1u) << 4);
            acc += (ds * (float)q - dm) * y[i];
        }
    }
    return simd_sum(acc);
}

/* kernel_qwen4_moe_mid with qwen35_row_dot: mid[t][s][r] =
 * silu(gate_row . x) * (up_row . x); slot n_slots (when has_shared) is the
 * shared expert from its own bases.  Two rows per SIMD group. */
kernel void kernel_qwen35_moe_mid(
        constant ds4_metal_args_qwen4_moe & args,
        device const char    *gate_base,
        device const char    *up_base,
        device const int32_t *selected,   /* [T][n_slots] */
        device const float   *x,          /* [T][in_dim] */
        device float         *mid,        /* [T][n_slots+has_shared][out_rows] */
        device const char    *sh_gate,
        device const char    *sh_up,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint tok = tgpig.z;
    const uint n_out = args.n_slots + args.has_shared;
    const uint nr = 2u;
    const uint row0 = (tgpig.x * (ntg.x / 32u) + (uint)sgitg) * nr;
    if (row0 >= args.out_rows || slot >= n_out || tok >= args.n_tokens) return;
    const bool shared = slot == args.n_slots;
    const uint type = shared ? args.shared_type : args.weight_type;
    const uint row_bytes = shared ? args.shared_row_bytes : args.row_bytes;
    device const char *gb = shared ? sh_gate : gate_base;
    device const char *ub = shared ? sh_up : up_base;
    const uint64_t ebase = shared ? 0 : (uint64_t)(uint)selected[(uint64_t)tok * args.n_slots + slot] * args.expert_bytes;
    device const float *xt = x + (uint64_t)tok * args.in_dim;
    for (uint r = row0; r < row0 + nr && r < args.out_rows; r++) {
        const uint64_t off = ebase + (uint64_t)r * row_bytes;
        const float g = qwen35_row_dot(gb + off, xt, type, args.in_dim, tiisg);
        const float u = qwen35_row_dot(ub + off, xt, type, args.in_dim, tiisg);
        if (tiisg == 0) mid[((uint64_t)tok * n_out + slot) * args.out_rows + r] = qwen4_silu(g) * u;
    }
}

/* kernel_qwen4_moe_down with qwen35_row_dot. */
kernel void kernel_qwen35_moe_down(
        constant ds4_metal_args_qwen4_moe & args,
        device const char    *down_base,
        device const int32_t *selected,   /* [T][n_slots] */
        device const float   *mid,        /* [T][n_slots+has_shared][in_dim] */
        device float         *part,       /* [T][n_slots+has_shared][out_rows] */
        device const char    *sh_down,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint tok = tgpig.z;
    const uint n_out = args.n_slots + args.has_shared;
    const uint nr = 2u;
    const uint row0 = (tgpig.x * (ntg.x / 32u) + (uint)sgitg) * nr;
    if (row0 >= args.out_rows || slot >= n_out || tok >= args.n_tokens) return;
    const bool shared = slot == args.n_slots;
    const uint type = shared ? args.shared_type : args.weight_type;
    const uint row_bytes = shared ? args.shared_row_bytes : args.row_bytes;
    device const char *db = shared ? sh_down : down_base;
    const uint64_t pair = (uint64_t)tok * n_out + slot;
    const uint64_t ebase = shared ? 0 : (uint64_t)(uint)selected[(uint64_t)tok * args.n_slots + slot] * args.expert_bytes;
    device const float *m = mid + pair * args.in_dim;
    for (uint r = row0; r < row0 + nr && r < args.out_rows; r++) {
        const float v = qwen35_row_dot(db + ebase + (uint64_t)r * row_bytes, m, type, args.in_dim, tiisg);
        if (tiisg == 0) part[pair * args.out_rows + r] = v;
    }
}
```

- [ ] **Step 4: Register the source file and kernels in `ds4_metal.m`**

In the `required_sources` array (`ds4_metal.m:4926`), directly after the qwen4 entry. It must come after qwen4.metal because it uses qwen4 helpers:

```objc
        @[@"DS4_METAL_QWEN4_SOURCE",      @"metal/qwen4.metal"],
        @[@"DS4_METAL_QWEN35_SOURCE",     @"metal/qwen35.metal"],
```

In the qwen4 kernel enum, before `QWEN4_K_COUNT` (`ds4_metal.m:48658`):

```objc
    QWEN4_K_VIS_BIAS_ACT,
    QWEN4_K_QWEN35_MOE_MID,
    QWEN4_K_QWEN35_MOE_DOWN,
    QWEN4_K_QWEN35_GDN_OUT,
    QWEN4_K_COUNT,
```

In `qwen4_kernel_names`, after `"kernel_qwen4_vis_bias_act",` (`ds4_metal.m:48780`). The table is positional, so the order must match the enum:

```objc
    "kernel_qwen4_vis_bias_act",
    "kernel_qwen35_moe_mid",
    "kernel_qwen35_moe_down",
    "kernel_qwen35_gdn_out",
```

`kernel_qwen35_gdn_out` is written in Task 4. Until then, its pipeline is only created lazily on first dispatch, so an unused enum entry is harmless.

- [ ] **Step 5: Add the host wrappers**

In `ds4_metal.m`, directly after `ds4_gpu_qwen4_moe_down_tensor`:

```objc
/* Ornith (qwen35moe) routed experts: the qwen4 row geometry (two rows per
 * SIMD group, four groups per threadgroup) with Q5_K handled by
 * qwen35_row_dot; every other type falls through to qwen4_row_dot. */
static uint32_t qwen35_expert_row_bytes(uint32_t weight_type, uint32_t in_dim) {
    if (weight_type == 13u) return (in_dim % 256u) ? 0u : (in_dim / 256u) * 176u;   /* q5_K */
    return qwen4_expert_row_bytes(weight_type, in_dim);
}

int ds4_gpu_qwen35_moe_mid_tensor(
        ds4_gpu_tensor *mid, const ds4_gpu_tensor *x, const ds4_gpu_tensor *selected,
        const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset,
        uint32_t weight_type, uint32_t n_total_expert, uint32_t n_tokens, uint32_t n_slots,
        uint32_t in_dim, uint32_t ff_dim,
        uint64_t shared_gate_offset, uint64_t shared_up_offset, uint32_t shared_type) {
    const uint32_t row_bytes = qwen35_expert_row_bytes(weight_type, in_dim);
    const uint64_t expert_bytes = (uint64_t)row_bytes * ff_dim;
    const bool has_shared = shared_type != UINT32_MAX;
    const uint32_t sh_row_bytes = has_shared ? qwen4_expert_row_bytes(shared_type, in_dim) : 0u;
    const uint32_t n_out = n_slots + (has_shared ? 1u : 0u);
    const uint64_t shared_bytes = (uint64_t)sh_row_bytes * ff_dim;
    qwen4_moe_args args = { n_tokens, n_slots, in_dim, ff_dim, weight_type, row_bytes, expert_bytes,
                            has_shared ? 1u : 0u, has_shared ? shared_type : 0u, sh_row_bytes, n_total_expert, 0u, 0u };
    qwen4_bind b[7];
    if (n_tokens == 0 || n_slots == 0 || row_bytes == 0 || ff_dim == 0 || (has_shared && sh_row_bytes == 0) ||
        !qwen4_bind_weight(&b[0], model_map, model_size, gate_offset, expert_bytes * n_total_expert, "moe gate experts") ||
        !qwen4_bind_weight(&b[1], model_map, model_size, up_offset, expert_bytes * n_total_expert, "moe up experts") ||
        !qwen4_bind_tensor(&b[2], selected, (uint64_t)n_tokens * n_slots * sizeof(int32_t), "moe selected") ||
        !qwen4_bind_tensor(&b[3], x, (uint64_t)n_tokens * in_dim * sizeof(float), "moe input") ||
        !qwen4_bind_tensor(&b[4], mid, (uint64_t)n_tokens * n_out * ff_dim * sizeof(float), "moe mid")) {
        return 0;
    }
    if (has_shared) {
        if (!qwen4_bind_weight(&b[5], model_map, model_size, shared_gate_offset, shared_bytes, "shared gate") ||
            !qwen4_bind_weight(&b[6], model_map, model_size, shared_up_offset, shared_bytes, "shared up")) {
            return 0;
        }
    } else {
        b[5] = b[0];
        b[6] = b[1];
    }
    return qwen4_dispatch(QWEN4_K_QWEN35_MOE_MID, &args, sizeof(args), b, 7,
                          MTLSizeMake((ff_dim + 7u) / 8u, n_out, n_tokens), MTLSizeMake(128, 1, 1), 0);
}

int ds4_gpu_qwen35_moe_down_tensor(
        ds4_gpu_tensor *part, const ds4_gpu_tensor *mid, const ds4_gpu_tensor *selected,
        const void *model_map, uint64_t model_size, uint64_t down_offset,
        uint32_t weight_type, uint32_t n_total_expert, uint32_t n_tokens, uint32_t n_slots,
        uint32_t ff_dim, uint32_t out_dim,
        uint64_t shared_down_offset, uint32_t shared_type) {
    const uint32_t row_bytes = qwen35_expert_row_bytes(weight_type, ff_dim);
    const uint64_t expert_bytes = (uint64_t)row_bytes * out_dim;
    const bool has_shared = shared_type != UINT32_MAX;
    const uint32_t sh_row_bytes = has_shared ? qwen4_expert_row_bytes(shared_type, ff_dim) : 0u;
    const uint32_t n_out = n_slots + (has_shared ? 1u : 0u);
    qwen4_moe_args args = { n_tokens, n_slots, ff_dim, out_dim, weight_type, row_bytes, expert_bytes,
                            has_shared ? 1u : 0u, has_shared ? shared_type : 0u, sh_row_bytes, n_total_expert, 0u, 0u };
    qwen4_bind b[5];
    if (n_tokens == 0 || n_slots == 0 || row_bytes == 0 || out_dim == 0 || (has_shared && sh_row_bytes == 0) ||
        !qwen4_bind_weight(&b[0], model_map, model_size, down_offset, expert_bytes * n_total_expert, "moe down experts") ||
        !qwen4_bind_tensor(&b[1], selected, (uint64_t)n_tokens * n_slots * sizeof(int32_t), "moe selected") ||
        !qwen4_bind_tensor(&b[2], mid, (uint64_t)n_tokens * n_out * ff_dim * sizeof(float), "moe mid") ||
        !qwen4_bind_tensor(&b[3], part, (uint64_t)n_tokens * n_out * out_dim * sizeof(float), "moe partial")) {
        return 0;
    }
    if (has_shared) {
        if (!qwen4_bind_weight(&b[4], model_map, model_size, shared_down_offset,
                               (uint64_t)sh_row_bytes * out_dim, "shared down")) {
            return 0;
        }
    } else {
        b[4] = b[0];
    }
    return qwen4_dispatch(QWEN4_K_QWEN35_MOE_DOWN, &args, sizeof(args), b, 5,
                          MTLSizeMake((out_dim + 7u) / 8u, n_out, n_tokens), MTLSizeMake(128, 1, 1), 0);
}
```

Declare both in `ds4_gpu.h`, directly after the `ds4_gpu_qwen4_moe_down_tensor` prototype (`ds4_gpu.h:3489`), using the signatures from this task's Interfaces block.

- [ ] **Step 6: Run the kernel test**

Run: `make test-qwen35-kernels`
Expected:
```
qwen35 moe (q5_K experts, q8_0 shared slot)
  q5_K moe mid T=1                              ok  max|d|=...
  q5_K moe down T=1                             ok  ...
  ... (T=2, T=5)
qwen35 kernels: ok
```

- [ ] **Step 7: Confirm the Qwen3.8 kernel tests still pass**

Run: `make test-qwen4-kernels test-qwen4-q2`
Expected: both end without a failure line (same output as before this task).

- [ ] **Step 8: Commit**

```bash
git add metal/qwen35.metal ds4_metal.m ds4_gpu.h tests/test_qwen35_kernels.c Makefile
git commit -m "qwen35: Q5_K routed-expert row kernels for Ornith

New metal/qwen35.metal with qwen35_row_dot (Q5_K, other types delegate to
qwen4_row_dot) and the mid/down row kernels, appended after qwen4.metal;
qwen4 kernels are unchanged. Tested against a double reference.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- metal/qwen35.metal ds4_metal.m ds4_gpu.h tests/test_qwen35_kernels.c Makefile
```

---

### Task 4: Silu-gated GDN output and indexer-free attention prep

**Files:**
- Modify: `metal/qwen35.metal`, `ds4_metal.m`, `ds4_gpu.h`, `tests/test_qwen35_kernels.c`

**Interfaces:**
- Consumes: `QWEN4_K_QWEN35_GDN_OUT` (Task 3 enum), `QWEN4_K_ATTN_PREP` (existing), `qwen4_rope_fill` (existing, `ds4_metal.m`).
- Produces:
  ```c
  /* Same parameters as ds4_gpu_qwen4_gdn_out_tensor; gate silu(z). */
  int ds4_gpu_qwen35_gdn_out_tensor(
          ds4_gpu_tensor *o, const ds4_gpu_tensor *z,
          const void *model_map, uint64_t model_size, uint64_t weight_offset,
          uint32_t n_tokens, uint32_t n_head, uint32_t head_dim, float eps);
  /* q/k RMSNorm, NEOX RoPE on n_rot dims, F16 KV append; no indexer. */
  int ds4_gpu_qwen35_attn_prep_tensor(
          ds4_gpu_tensor *q_out, ds4_gpu_tensor *gate_out, ds4_gpu_tensor *k_cache, ds4_gpu_tensor *v_cache,
          const ds4_gpu_tensor *qg, const ds4_gpu_tensor *kproj, const ds4_gpu_tensor *vproj,
          const ds4_gpu_tensor *pos3, const void *model_map, uint64_t model_size,
          uint64_t g_q_offset, uint64_t g_k_offset,
          uint32_t n_tokens, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim, uint32_t n_rot,
          uint32_t pos0, uint32_t cache_cap, float rope_base, float eps);
  ```

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_qwen35_kernels.c` before `main`:

```c
/* GDN output: per-head RMSNorm of the scan output times ssm_norm, gated by
 * silu(z) (Qwen3.5 RMSNormGated). */
static void test_gdn_out_silu(arena_t *a, uint32_t H, uint32_t D, uint32_t T) {
    double *w;
    const uint64_t w_off = arena_f32(a, D, &w, 0.5f, 1.5f);
    const uint64_t n = (uint64_t)T * H * D;
    float *o = rand_vec(n, 1.0f), *z = rand_vec(n, 2.0f);
    double *ref = malloc(n * sizeof(double));
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t h = 0; h < H; h++) {
            const uint64_t base = ((uint64_t)t * H + h) * D;
            double ss = 0.0;
            for (uint32_t i = 0; i < D; i++) ss += (double)o[base + i] * o[base + i];
            const double r = 1.0 / sqrt(ss / D + 1e-6);
            for (uint32_t i = 0; i < D; i++) ref[base + i] = o[base + i] * r * w[i] * silu_d(z[base + i]);
        }
    }
    ds4_gpu_tensor *go = upload(o, n), *gz = upload(z, n);
    require_ok(ds4_gpu_qwen35_gdn_out_tensor(go, gz, a->base, a->size, w_off, T, H, D, 1e-6f), "qwen35 gdn out");
    float *got = download(go, n);
    check_close("gdn out silu", got, ref, n, 1e-5);
    free(got); free(o); free(z); free(ref); free(w);
    ds4_gpu_tensor_free(go); ds4_gpu_tensor_free(gz);
}

/* NEOX partial RoPE on the first n_rot dims (pairs i, i + n_rot/2). */
static void rope_ref(double *x, uint32_t n_rot, uint32_t pos, double base) {
    const uint32_t nh = n_rot / 2;
    for (uint32_t i = 0; i < nh; i++) {
        const double th = (double)pos * pow(base, -2.0 * i / n_rot);
        const double c = cos(th), s = sin(th), x0 = x[i], x1 = x[i + nh];
        x[i] = x0 * c - x1 * s;
        x[i + nh] = x0 * s + x1 * c;
    }
}

/* Attention prep without the indexer: q (from the interleaved [q | gate]
 * rows) and k get RMSNorm, their weight and RoPE; the gate passes through; k
 * and v are appended to the F16 caches at pos0.. . */
static void test_attn_prep_noindexer(arena_t *a) {
    const uint32_t T = 3, H = 16, Hkv = 2, D = 256, n_rot = 64, pos0 = 5, cap = 16;
    const double base = 1.0e7;
    double *gq, *gk;
    const uint64_t gq_off = arena_f32(a, D, &gq, 0.5f, 1.5f);
    const uint64_t gk_off = arena_f32(a, D, &gk, 0.5f, 1.5f);
    float *qg = rand_vec((uint64_t)T * H * 2 * D, 1.0f);
    float *kp = rand_vec((uint64_t)T * Hkv * D, 1.0f), *vp = rand_vec((uint64_t)T * Hkv * D, 1.0f);
    uint32_t pos3[16 * 4];
    for (uint32_t p = 0; p < cap; p++) { pos3[p * 4] = pos3[p * 4 + 1] = pos3[p * 4 + 2] = p; pos3[p * 4 + 3] = 0; }
    double *q_ref = malloc((uint64_t)T * H * D * sizeof(double));
    double *g_ref = malloc((uint64_t)T * H * D * sizeof(double));
    double *k_ref = malloc((uint64_t)T * Hkv * D * sizeof(double));
    double row[256];
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t h = 0; h < H; h++) {
            const float *src = qg + ((uint64_t)t * H + h) * 2 * D;
            double ss = 0.0;
            for (uint32_t i = 0; i < D; i++) ss += (double)src[i] * src[i];
            const double r = 1.0 / sqrt(ss / D + 1e-6);
            for (uint32_t i = 0; i < D; i++) row[i] = src[i] * r * gq[i];
            rope_ref(row, n_rot, pos0 + t, base);
            for (uint32_t i = 0; i < D; i++) {
                q_ref[((uint64_t)t * H + h) * D + i] = row[i];
                g_ref[((uint64_t)t * H + h) * D + i] = src[D + i];
            }
        }
        for (uint32_t h = 0; h < Hkv; h++) {
            const float *src = kp + ((uint64_t)t * Hkv + h) * D;
            double ss = 0.0;
            for (uint32_t i = 0; i < D; i++) ss += (double)src[i] * src[i];
            const double r = 1.0 / sqrt(ss / D + 1e-6);
            for (uint32_t i = 0; i < D; i++) row[i] = src[i] * r * gk[i];
            rope_ref(row, n_rot, pos0 + t, base);
            for (uint32_t i = 0; i < D; i++) k_ref[((uint64_t)t * Hkv + h) * D + i] = row[i];
        }
    }
    ds4_gpu_tensor *gqg = upload(qg, (uint64_t)T * H * 2 * D);
    ds4_gpu_tensor *gkp = upload(kp, (uint64_t)T * Hkv * D), *gvp = upload(vp, (uint64_t)T * Hkv * D);
    ds4_gpu_tensor *gq_out = upload(NULL, (uint64_t)T * H * D), *ggate = upload(NULL, (uint64_t)T * H * D);
    ds4_gpu_tensor *kc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *vc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *gpos = ds4_gpu_tensor_alloc(sizeof(pos3));
    require_ok(kc && vc && gpos && ds4_gpu_tensor_write(gpos, 0, pos3, sizeof(pos3)), "prep buffers");
    require_ok(ds4_gpu_qwen35_attn_prep_tensor(gq_out, ggate, kc, vc, gqg, gkp, gvp, gpos, a->base, a->size,
                                               gq_off, gk_off, T, H, Hkv, D, n_rot, pos0, cap,
                                               (float)base, 1e-6f), "qwen35 attn prep");
    float *got = download(gq_out, (uint64_t)T * H * D);
    check_close("attn prep q", got, q_ref, (uint64_t)T * H * D, 2e-5);
    free(got);
    got = download(ggate, (uint64_t)T * H * D);
    check_close("attn prep gate", got, g_ref, (uint64_t)T * H * D, 0.0);
    free(got);
    uint16_t *kh = malloc((uint64_t)cap * Hkv * D * 2u), *vh = malloc((uint64_t)cap * Hkv * D * 2u);
    require_ok(ds4_gpu_tensor_read(kc, 0, kh, (uint64_t)cap * Hkv * D * 2u), "k cache read");
    require_ok(ds4_gpu_tensor_read(vc, 0, vh, (uint64_t)cap * Hkv * D * 2u), "v cache read");
    float *kf = malloc((uint64_t)T * Hkv * D * sizeof(float)), *vf = malloc((uint64_t)T * Hkv * D * sizeof(float));
    double *v_ref = malloc((uint64_t)T * Hkv * D * sizeof(double));
    for (uint64_t i = 0; i < (uint64_t)T * Hkv * D; i++) {
        kf[i] = f16_to_f32(kh[(uint64_t)pos0 * Hkv * D + i]);
        vf[i] = f16_to_f32(vh[(uint64_t)pos0 * Hkv * D + i]);
        v_ref[i] = vp[i];
    }
    check_close("attn prep k cache (f16)", kf, k_ref, (uint64_t)T * Hkv * D, 2e-3);
    check_close("attn prep v cache (f16)", vf, v_ref, (uint64_t)T * Hkv * D, 2e-3);
    free(kh); free(vh); free(kf); free(vf); free(v_ref);
    free(qg); free(kp); free(vp); free(q_ref); free(g_ref); free(k_ref); free(gq); free(gk);
    ds4_gpu_tensor_free(gqg); ds4_gpu_tensor_free(gkp); ds4_gpu_tensor_free(gvp);
    ds4_gpu_tensor_free(gq_out); ds4_gpu_tensor_free(ggate); ds4_gpu_tensor_free(kc);
    ds4_gpu_tensor_free(vc); ds4_gpu_tensor_free(gpos);
}
```

`check_close` with tolerance 0.0 requires an exact copy for the gate pass-through.

Ornith also calls the existing qwen4 MoE reduce without a residual (`R = NULL`, `n_hc = 0`), a path Qwen3.8 never takes. Pin it with a test of existing code; no new code:

```c
/* MoE reduce without the residual combine (R = NULL, n_hc = 0): the shared
 * expert comes from a part slot (single rows) or a separate buffer (batches). */
static void test_reduce_nohc(uint32_t T, uint32_t K, uint32_t E, bool shared_slot) {
    const uint32_t stride = K + (shared_slot ? 1u : 0u);
    float *part = rand_vec((uint64_t)T * stride * E, 1.0f);
    float *w = rand_vec((uint64_t)T * K, 1.0f);
    float *sg = rand_vec(T, 2.0f);
    float *sh = rand_vec((uint64_t)T * E, 1.0f);
    double *ref = malloc((uint64_t)T * E * sizeof(double));
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t d = 0; d < E; d++) {
            double acc = 0.0;
            for (uint32_t s = 0; s < K; s++) acc += (double)w[t * K + s] * part[((uint64_t)t * stride + s) * E + d];
            const double shared = shared_slot ? part[((uint64_t)t * stride + K) * E + d] : sh[(uint64_t)t * E + d];
            ref[(uint64_t)t * E + d] = acc + sigmoid_d(sg[t]) * shared;
        }
    }
    ds4_gpu_tensor *gp = upload(part, (uint64_t)T * stride * E), *gw = upload(w, (uint64_t)T * K);
    ds4_gpu_tensor *gsg = upload(sg, T), *gsh = upload(sh, (uint64_t)T * E), *gout = upload(NULL, (uint64_t)T * E);
    require_ok(ds4_gpu_qwen4_moe_reduce_tensor(gout, gp, gw, gsg, shared_slot ? NULL : gsh, NULL, NULL,
                                               T, K, stride, E, 0u), "moe reduce (no hc)");
    float *got = download(gout, (uint64_t)T * E);
    check_close(shared_slot ? "moe reduce, shared slot" : "moe reduce, shared buffer", got, ref, (uint64_t)T * E, 1e-5);
    free(got); free(part); free(w); free(sg); free(sh); free(ref);
    ds4_gpu_tensor_free(gp); ds4_gpu_tensor_free(gw); ds4_gpu_tensor_free(gsg);
    ds4_gpu_tensor_free(gsh); ds4_gpu_tensor_free(gout);
}
```

In `main`, before the final `printf`:

```c
    printf("qwen35 gdn out (silu gate)\n");
    test_gdn_out_silu(&arena, 32, 128, 3);
    printf("qwen35 attention prep (no indexer)\n");
    test_attn_prep_noindexer(&arena);
    printf("qwen4 moe reduce without residual (as Ornith calls it)\n");
    test_reduce_nohc(1, 8, 2048, true);
    test_reduce_nohc(12, 8, 2048, false);
```

- [ ] **Step 2: Build and confirm the tests fail**

Run: `make test-qwen35-kernels`
Expected: link error for `_ds4_gpu_qwen35_gdn_out_tensor` and `_ds4_gpu_qwen35_attn_prep_tensor`.

- [ ] **Step 3: Add the silu GDN output kernel to `metal/qwen35.metal`**

```metal
/* --- Gated DeltaNet output ---------------------------------------------- */

/* Qwen3.5 RMSNormGated: per-head RMSNorm of the scan output, scaled by
 * ssm_norm and gated by silu(z).  Qwen3.8 gates with sigmoid
 * (kernel_qwen4_gdn_out); the arithmetic is otherwise the same. */
kernel void kernel_qwen35_gdn_out(
        constant ds4_metal_args_qwen4_gdn_out & args,
        device float       *o,        /* [T][H*D], in place */
        device const float *z,        /* [T][H*D] */
        device const float *weight,   /* [D] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint h = tgpig.x;
    const uint tok = tgpig.y;
    if (h >= args.n_head || tok >= args.n_tokens) return;
    const uint D = args.head_dim;
    const uint npt = D / 32;
    const uint64_t base = ((uint64_t)tok * args.n_head + h) * D + tiisg * npt;
    float ss = 0.0f;
    for (uint i = 0; i < npt; i++) ss += o[base + i] * o[base + i];
    ss = simd_sum(ss);
    const float r = rsqrt(ss / (float)D + args.eps);
    for (uint i = 0; i < npt; i++) {
        o[base + i] = o[base + i] * r * weight[tiisg * npt + i] * qwen4_silu(z[base + i]);
    }
}
```

- [ ] **Step 4: Add the two wrappers to `ds4_metal.m`**

Directly after `ds4_gpu_qwen4_gdn_out_tensor`:

```objc
int ds4_gpu_qwen35_gdn_out_tensor(
        ds4_gpu_tensor *o, const ds4_gpu_tensor *z,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n_tokens, uint32_t n_head, uint32_t head_dim, float eps) {
    struct { uint32_t n_tokens, n_head, head_dim; float eps; } args = { n_tokens, n_head, head_dim, eps };
    qwen4_bind b[3];
    const uint64_t bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    if (n_tokens == 0 || head_dim < 32 || (head_dim % 32) != 0 ||
        !qwen4_bind_tensor(&b[0], o, bytes, "gdn out") ||
        !qwen4_bind_tensor(&b[1], z, bytes, "gdn gate") ||
        !qwen4_bind_weight(&b[2], model_map, model_size, weight_offset, (uint64_t)head_dim * sizeof(float),
                           "ssm_norm")) {
        return 0;
    }
    return qwen4_dispatch(QWEN4_K_QWEN35_GDN_OUT, &args, sizeof(args), b, 3,
                          MTLSizeMake(n_head, n_tokens, 1), MTLSizeMake(32, 1, 1), 0);
}
```

Directly after `ds4_gpu_qwen4_attn_prep_tensor`:

```objc
/* Ornith attention prep: the qwen4 prep kernel with only its query and KV
 * slots dispatched (grid.x = n_head + n_head_kv), so the QSA indexer slots
 * never run and their buffers can alias bound ones.  F16 KV only. */
int ds4_gpu_qwen35_attn_prep_tensor(
        ds4_gpu_tensor *q_out, ds4_gpu_tensor *gate_out, ds4_gpu_tensor *k_cache, ds4_gpu_tensor *v_cache,
        const ds4_gpu_tensor *qg, const ds4_gpu_tensor *kproj, const ds4_gpu_tensor *vproj,
        const ds4_gpu_tensor *pos3, const void *model_map, uint64_t model_size,
        uint64_t g_q_offset, uint64_t g_k_offset,
        uint32_t n_tokens, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim, uint32_t n_rot,
        uint32_t pos0, uint32_t cache_cap, float rope_base, float eps) {
    struct {
        uint32_t n_tokens, n_head, n_head_kv, head_dim, n_rot, n_idx_head, idx_dim, pos0, cache_cap;
        float rope_base, eps; uint32_t fp8; float rope_mscale; float rope_freq[32]; uint32_t ik_ring, kv_simq;
    } args = { n_tokens, n_head, n_head_kv, head_dim, n_rot, 0u, 0u, pos0, cache_cap,
               rope_base, eps, 0u, 1.0f, { 0 }, 0u, 0u };
    qwen4_rope_fill(args.rope_freq, &args.rope_mscale, n_rot, rope_base);
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)n_tokens * n_head_kv * head_dim * sizeof(float);
    const uint64_t cache_bytes = (uint64_t)cache_cap * n_head_kv * head_dim * 2u;
    qwen4_bind b[19];
    if (n_tokens == 0 || head_dim < 32 || head_dim > 256 || (head_dim % 32) != 0 || n_rot > 64 ||
        (n_rot % 2) != 0 || (uint64_t)pos0 + n_tokens > cache_cap || n_head_kv == 0 || (n_head % n_head_kv) != 0 ||
        !qwen4_bind_tensor(&b[0], qg, 2u * q_bytes, "attn q/gate projection") ||
        !qwen4_bind_tensor(&b[1], kproj, kv_bytes, "attn k projection") ||
        !qwen4_bind_tensor(&b[2], vproj, kv_bytes, "attn v projection") ||
        !qwen4_bind_weight(&b[5], model_map, model_size, g_q_offset, (uint64_t)head_dim * sizeof(float),
                           "attn q_norm") ||
        !qwen4_bind_weight(&b[6], model_map, model_size, g_k_offset, (uint64_t)head_dim * sizeof(float),
                           "attn k_norm") ||
        !qwen4_bind_tensor(&b[8], q_out, q_bytes, "attn q") ||
        !qwen4_bind_tensor(&b[9], gate_out, q_bytes, "attn gate") ||
        !qwen4_bind_tensor(&b[10], k_cache, cache_bytes, "k cache") ||
        !qwen4_bind_tensor(&b[11], v_cache, cache_bytes, "v cache") ||
        !qwen4_bind_tensor(&b[14], pos3, (uint64_t)cache_cap * 16u, "rope positions")) {
        return 0;
    }
    b[3] = b[0];  b[4] = b[0];  b[7] = b[5];              /* indexer inputs, never read */
    b[12] = b[8]; b[13] = b[8];                           /* indexer outputs, never written */
    b[15] = b[10]; b[16] = b[11]; b[17] = b[10]; b[18] = b[11];   /* FP8 slots, fp8 = 0 */
    return qwen4_dispatch(QWEN4_K_ATTN_PREP, &args, sizeof(args), b, 19,
                          MTLSizeMake(n_head + n_head_kv, n_tokens, 1), MTLSizeMake(32, 1, 1), 0);
}
```

Declare both in `ds4_gpu.h`, next to their qwen4 counterparts (`ds4_gpu.h:3402`, `:3434`).

- [ ] **Step 5: Run the kernel tests**

Run: `make test-qwen35-kernels`
Expected: all lines `ok`, ending with `qwen35 kernels: ok`.

- [ ] **Step 6: Confirm the Qwen3.8 kernel tests still pass**

Run: `make test-qwen4-kernels test-qwen4-q2`
Expected: no failure line.

- [ ] **Step 7: Commit**

```bash
git add metal/qwen35.metal ds4_metal.m ds4_gpu.h tests/test_qwen35_kernels.c
git commit -m "qwen35: silu-gated GDN output and indexer-free attention prep

The GDN output is a new kernel; attention prep reuses the qwen4 prep kernel
with only its query and KV slots dispatched, so the QSA indexer never runs.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- metal/qwen35.metal ds4_metal.m ds4_gpu.h tests/test_qwen35_kernels.c
```

---

### Task 5: Family, loader and tokenizer

**Files:**
- Modify: `ds4.c`, `ds4.h`, `docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md` (§3 sentence)
- Create: `tests/ornith/make_bad_gguf.py`, `tests/ornith/test_loader.sh`

**Interfaces:**
- Produces (in `ds4.c`):
  - `DS4_MODEL_FAMILY_QWEN35_MOE = 4`, `DS4_VARIANT_QWEN35_MOE = 7`, `DS4_SHAPE_QWEN35_MOE`
  - `static bool ds4_model_is_qwen35moe(void)`
  - `static bool ds4_model_uses_qwen35_text(void)`
  - `static bool ds4_qwen35_layer_is_nextn(uint32_t il)`
  - `static bool ds4_qwen35_layer_is_attention(uint32_t il)`: true for `(il + 1) % 4 == 0` and for the MTP block
  - `static void ds4_qwen35_not_reached(const char *where)`: dies if the Ornith family reaches a DeepSeek default path
- Produces (public, `ds4.h`): `bool ds4_engine_is_qwen35moe(ds4_engine *e);`
- Ornith layer mapping onto `ds4_layer_weights`: `attn_norm` = `blk.N.attn_norm`, `ffn_norm` = `blk.N.post_attention_norm`, the `lin_*` fields as qwen4, the `attn_*` fields as qwen4 minus the indexer, and `nextn_eh_proj`/`nextn_enorm`/`nextn_hnorm`/`nextn_shared_head_norm` on `blk.40`.

- [ ] **Step 1: Write the loader test (failing)**

Create `tests/ornith/make_bad_gguf.py`:

```python
#!/usr/bin/env python3
"""Write sparse copies of an Ornith GGUF with one field patched.

  make_bad_gguf.py MODEL OUT_DIR

bad_embd.gguf: qwen35moe.embedding_length = 64.
bad_tier.gguf: blk.20 gate/up experts typed IQ4_XS (ggml type 23).
Only the header is copied; the tensor data is a hole of the original size,
so each file costs a few MB on disk.
"""
import os
import struct
import sys

SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_header(path):
    with open(path, "rb") as f:
        buf = f.read(64 << 20)
    pos = 0

    def take(n):
        nonlocal pos
        out = buf[pos:pos + n]
        pos += n
        return out

    def string():
        (n,) = struct.unpack("<Q", take(8))
        return take(n).decode("utf-8", "replace")

    def skip_value(t):
        if t in SCALAR:
            take(SCALAR[t])
        elif t == 8:
            string()
        elif t == 9:
            (et,) = struct.unpack("<I", take(4))
            (n,) = struct.unpack("<Q", take(8))
            for _ in range(n):
                skip_value(et)
        else:
            sys.exit(f"unknown GGUF value type {t}")

    assert take(4) == b"GGUF"
    take(4)
    n_tensors, n_kv = struct.unpack("<QQ", take(16))
    kv_pos = {}
    for _ in range(n_kv):
        key = string()
        (t,) = struct.unpack("<I", take(4))
        kv_pos[key] = (pos, t)
        skip_value(t)
    type_pos = {}
    for _ in range(n_tensors):
        name = string()
        (nd,) = struct.unpack("<I", take(4))
        take(8 * nd)
        type_pos[name] = pos
        take(4 + 8)
    return buf, pos, kv_pos, type_pos


def write(out, header, size):
    with open(out, "wb") as f:
        f.write(header)
        f.truncate(size)


def main(model, out_dir):
    size = os.path.getsize(model)
    buf, end, kv_pos, type_pos = read_header(model)
    header = bytearray(buf[: end + 4096])
    p, t = kv_pos["qwen35moe.embedding_length"]
    assert t == 4
    bad = bytearray(header)
    struct.pack_into("<I", bad, p, 64)
    write(os.path.join(out_dir, "bad_embd.gguf"), bad, size)
    bad = bytearray(header)
    for name in ("blk.20.ffn_gate_exps.weight", "blk.20.ffn_up_exps.weight"):
        struct.pack_into("<I", bad, type_pos[name], 23)
    write(os.path.join(out_dir, "bad_tier.gguf"), bad, size)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
```

Create `tests/ornith/test_loader.sh` (make it executable):

```sh
#!/bin/sh
# Loader checks for Ornith (qwen35moe): the 23G GGUF inspects cleanly, and a
# wrong metadata value or an unsupported expert type fails with a message
# naming the key or the tier.  Needs a built ./ds4 and DS4_ORNITH_MODEL.
set -eu
model=${DS4_ORNITH_MODEL:?set DS4_ORNITH_MODEL to the 23G ICE GGUF}
tmp=$(mktemp -d "${TMPDIR:-/tmp}/ornith-loader.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
./ds4 --inspect -m "$model" > "$tmp/ok.txt" 2>&1 || { cat "$tmp/ok.txt"; exit 1; }
grep -q 'Ornith-1.5-35B-A3B: 40 layers + MTP' "$tmp/ok.txt" || { cat "$tmp/ok.txt"; exit 1; }
python3 tests/ornith/make_bad_gguf.py "$model" "$tmp"
if ./ds4 --inspect -m "$tmp/bad_embd.gguf" > "$tmp/embd.txt" 2>&1; then
    echo "bad metadata accepted"; exit 1
fi
grep -q 'expected embedding_length=2048' "$tmp/embd.txt" || { cat "$tmp/embd.txt"; exit 1; }
if ./ds4 --inspect -m "$tmp/bad_tier.gguf" > "$tmp/tier.txt" 2>&1; then
    echo "IQ4_XS experts accepted"; exit 1
fi
grep -q 'not supported; use the 23G or 25G tier' "$tmp/tier.txt" || { cat "$tmp/tier.txt"; exit 1; }
echo "ornith loader: ok"
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `make ds4 && tests/ornith/test_loader.sh`
Expected: exit 1. The first `./ds4 --inspect` fails because `general.architecture = qwen35moe` falls through to the DeepSeek V4 validator (a message about missing `deepseek4.*` keys).

- [ ] **Step 3: Add the enums, shape and predicates**

In `ds4.c:525-540`:

```c
typedef enum {
    DS4_MODEL_FAMILY_DEEPSEEK4 = 0,
    DS4_MODEL_FAMILY_GLM_DSA   = 1,
    DS4_MODEL_FAMILY_DEEPSEEK41 = 2,
    DS4_MODEL_FAMILY_QWEN4_EXP = 3,
    DS4_MODEL_FAMILY_QWEN35_MOE = 4,
} ds4_model_family;
```

Add `DS4_VARIANT_QWEN35_MOE = 7,` after `DS4_VARIANT_QWEN4_MINI = 6,`.

After `DS4_SHAPE_QWEN4_MINI` (`ds4.c:878`):

```c
/* Ornith-1.5-35B-A3B (llama.cpp qwen35moe): 40 trunk layers repeating
 * 3 Gated DeltaNet + 1 gated full attention on a plain residual stream,
 * plus one MTP block.  256 routed experts, 8 used, and one shared expert.
 * No hyper-connections, PLE or QSA indexer: those shape fields stay 0 and
 * only the qwen4exp path reads them. */
static const ds4_shape DS4_SHAPE_QWEN35_MOE = {
    .name = "Ornith-1.5-35B-A3B",
    .family = DS4_MODEL_FAMILY_QWEN35_MOE,
    .variant = DS4_VARIANT_QWEN35_MOE,
    .n_layer = 41,
    .n_embd = 2048,
    .n_vocab = 248320,
    .n_head = 16,
    .n_head_kv = 2,
    .n_head_dim = 256,
    .n_value_dim = 256,
    .n_rot = 64,
    .n_expert = 256,
    .n_expert_used = 8,
    .n_expert_shared = 1,
    .n_ff_exp = 512,
    .n_nextn_predict = 1,
    .n_lin_k_head = 16,
    .n_lin_v_head = 32,
    .n_lin_head_dim = 128,
    .n_lin_conv = 4,
    .n_full_attn_interval = 4,
    .rms_eps = 1.0e-6f,
    .rope_freq_base = 10000000.0f,
    .rope_orig_ctx = 262144,
};
```

After `ds4_qwen4_layer_is_nextn` (`ds4.c:1030`):

```c
static bool ds4_model_is_qwen35moe(void) {
    return DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_QWEN35_MOE;
}

/* Qwen3.5 tokenizer, ChatML turns and XML tool calls: Qwen3.8 and Ornith. */
static bool ds4_model_uses_qwen35_text(void) {
    return ds4_model_is_qwen4() || ds4_model_is_qwen35moe();
}

static bool ds4_qwen35_layer_is_nextn(uint32_t il) {
    return ds4_model_is_qwen35moe() && DS4_N_NEXTN_PREDICT != 0 &&
           il + DS4_N_NEXTN_PREDICT >= DS4_N_LAYER;
}

/* Full attention at (il + 1) % 4 == 0, as llama.cpp's qwen35moe loader; the
 * MTP block is an attention layer too. */
static bool ds4_qwen35_layer_is_attention(uint32_t il) {
    return ds4_model_is_qwen35moe() &&
           (ds4_qwen35_layer_is_nextn(il) || (il + 1u) % DS4_N_FULL_ATTN_INTERVAL == 0u);
}

/* The Ornith family has its own branch at every graph, session and memory
 * dispatch point; reaching a DeepSeek default instead is a missed branch. */
static void ds4_qwen35_not_reached(const char *where) {
    if (!ds4_model_is_qwen35moe()) return;
    fprintf(stderr, "ds4: internal error: Ornith reached the DeepSeek path in %s\n", where);
    exit(1);
}
```

- [ ] **Step 4: Add the validator and hook it into `config_validate_model`**

After `config_validate_qwen4_model` (`ds4.c:7173`):

```c
/* Ornith-1.5-35B-A3B: every shape value is fixed; a mismatch names the key.
 * RoPE is plain NEOX partial rotation at the native context (the qwen4 prep
 * kernel computes base^(-2i/n_rot) when no YaRN table is set). */
static void config_validate_qwen35moe_model(const ds4_model *m) {
    g_ds4_shape = DS4_SHAPE_QWEN35_MOE;
    memset(g_ds4_compress_ratios, 0, sizeof(g_ds4_compress_ratios));
    config_expect_u32("embedding_length", required_u32(m, "qwen35moe.embedding_length"), DS4_N_EMBD);
    config_expect_u32("block_count", required_u32(m, "qwen35moe.block_count"), DS4_N_LAYER);
    config_expect_u32("nextn_predict_layers", required_u32(m, "qwen35moe.nextn_predict_layers"),
                      DS4_N_NEXTN_PREDICT);
    config_expect_u32("context_length", required_u32(m, "qwen35moe.context_length"),
                      (uint32_t)DS4_ROPE_ORIG_CTX);
    config_expect_u32("attention.head_count", required_u32(m, "qwen35moe.attention.head_count"), DS4_N_HEAD);
    config_expect_u32("attention.head_count_kv", required_u32(m, "qwen35moe.attention.head_count_kv"),
                      DS4_N_HEAD_KV);
    config_expect_u32("attention.key_length", required_u32(m, "qwen35moe.attention.key_length"),
                      DS4_N_HEAD_DIM);
    config_expect_u32("attention.value_length", required_u32(m, "qwen35moe.attention.value_length"),
                      DS4_N_VALUE_DIM);
    config_expect_u32("rope.dimension_count", required_u32(m, "qwen35moe.rope.dimension_count"), DS4_N_ROT);
    config_expect_f32("rope.freq_base", required_f32(m, "qwen35moe.rope.freq_base"), DS4_ROPE_FREQ_BASE);
    config_expect_epsilon("attention.layer_norm_rms_epsilon",
                          required_f32(m, "qwen35moe.attention.layer_norm_rms_epsilon"), DS4_RMS_EPS);
    config_expect_u32("expert_count", required_u32(m, "qwen35moe.expert_count"), DS4_N_EXPERT);
    config_expect_u32("expert_used_count", required_u32(m, "qwen35moe.expert_used_count"), DS4_N_EXPERT_USED);
    config_expect_u32("expert_feed_forward_length",
                      required_u32(m, "qwen35moe.expert_feed_forward_length"), DS4_N_FF_EXP);
    config_expect_u32("expert_shared_feed_forward_length",
                      required_u32(m, "qwen35moe.expert_shared_feed_forward_length"), DS4_N_FF_EXP);
    config_expect_u32("ssm.conv_kernel", required_u32(m, "qwen35moe.ssm.conv_kernel"), DS4_N_LIN_CONV);
    config_expect_u32("ssm.state_size", required_u32(m, "qwen35moe.ssm.state_size"), DS4_N_LIN_HEAD_DIM);
    config_expect_u32("ssm.group_count", required_u32(m, "qwen35moe.ssm.group_count"), DS4_N_LIN_K_HEAD);
    config_expect_u32("ssm.time_step_rank", required_u32(m, "qwen35moe.ssm.time_step_rank"), DS4_N_LIN_V_HEAD);
    config_expect_u32("ssm.inner_size", required_u32(m, "qwen35moe.ssm.inner_size"),
                      DS4_N_LIN_V_HEAD * DS4_N_LIN_HEAD_DIM);
    config_expect_u32("full_attention_interval",
                      required_u32(m, "qwen35moe.full_attention_interval"), DS4_N_FULL_ATTN_INTERVAL);
}
```

In `config_validate_model` (`ds4.c:7188-7192`), after the `qwen4exp` branch:

```c
        if (ds4_streq(arch, "qwen35moe")) {
            config_validate_qwen35moe_model(m);
            return;
        }
```

- [ ] **Step 5: Bind the layers, the output head and the MTP block**

After `weights_bind_qwen4_layer` (`ds4.c:7845`):

```c
/* llama.cpp qwen35moe names.  attn_norm / post_attention_norm land in the
 * attn_norm / ffn_norm fields the other families use for their pre-mixer and
 * pre-FFN norms; their weights already carry llama.cpp's +1. */
static void weights_bind_qwen35moe_layer(ds4_layer_weights *l, const ds4_model *m, uint32_t il) {
    l->attn_norm = required_tensorf(m, "blk.%u.attn_norm.weight", il);
    l->ffn_norm  = required_tensorf(m, "blk.%u.post_attention_norm.weight", il);
    if (ds4_qwen35_layer_is_attention(il)) {
        l->attn_q      = required_tensorf(m, "blk.%u.attn_q.weight", il);
        l->attn_k      = required_tensorf(m, "blk.%u.attn_k.weight", il);
        l->attn_v      = required_tensorf(m, "blk.%u.attn_v.weight", il);
        l->attn_output = required_tensorf(m, "blk.%u.attn_output.weight", il);
        l->attn_q_norm = required_tensorf(m, "blk.%u.attn_q_norm.weight", il);
        l->attn_k_norm = required_tensorf(m, "blk.%u.attn_k_norm.weight", il);
    } else {
        l->lin_qkv     = required_tensorf(m, "blk.%u.attn_qkv.weight", il);
        l->lin_gate    = required_tensorf(m, "blk.%u.attn_gate.weight", il);
        l->lin_conv    = required_tensorf(m, "blk.%u.ssm_conv1d.weight", il);
        l->lin_dt_bias = required_tensorf(m, "blk.%u.ssm_dt.bias", il);
        l->lin_a       = required_tensorf(m, "blk.%u.ssm_a", il);
        l->lin_beta    = required_tensorf(m, "blk.%u.ssm_beta.weight", il);
        l->lin_alpha   = required_tensorf(m, "blk.%u.ssm_alpha.weight", il);
        l->lin_norm    = required_tensorf(m, "blk.%u.ssm_norm.weight", il);
        l->lin_out     = required_tensorf(m, "blk.%u.ssm_out.weight", il);
    }
    l->ffn_gate_inp       = required_tensorf(m, "blk.%u.ffn_gate_inp.weight", il);
    l->ffn_gate_exps      = required_tensorf(m, "blk.%u.ffn_gate_exps.weight", il);
    l->ffn_up_exps        = required_tensorf(m, "blk.%u.ffn_up_exps.weight", il);
    l->ffn_down_exps      = required_tensorf(m, "blk.%u.ffn_down_exps.weight", il);
    l->ffn_gate_inp_shexp = required_tensorf(m, "blk.%u.ffn_gate_inp_shexp.weight", il);
    l->ffn_gate_shexp     = required_tensorf(m, "blk.%u.ffn_gate_shexp.weight", il);
    l->ffn_up_shexp       = required_tensorf(m, "blk.%u.ffn_up_shexp.weight", il);
    l->ffn_down_shexp     = required_tensorf(m, "blk.%u.ffn_down_shexp.weight", il);
    if (ds4_qwen35_layer_is_nextn(il)) {
        l->nextn_eh_proj          = required_tensorf(m, "blk.%u.nextn.eh_proj.weight", il);
        l->nextn_enorm            = required_tensorf(m, "blk.%u.nextn.enorm.weight", il);
        l->nextn_hnorm            = required_tensorf(m, "blk.%u.nextn.hnorm.weight", il);
        l->nextn_shared_head_norm = required_tensorf(m, "blk.%u.nextn.shared_head_norm.weight", il);
    }
}
```

In `weights_bind_layer` (`ds4.c:7848`), after the qwen4 branch:

```c
    if (ds4_model_is_qwen35moe()) {
        weights_bind_qwen35moe_layer(l, m, il);
        return;
    }
```

In `weights_bind` (`ds4.c:7936` and `:7971`), extend both family conditions:

```c
    if ((DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA || ds4_model_is_qwen4() || ds4_model_is_qwen35moe()) &&
        DS4_N_LAYER > DS4_N_NEXTN_PREDICT) {
```

```c
        (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA || ds4_model_is_qwen4() || ds4_model_is_qwen35moe()) &&
```

In `weights_bind_output` (`ds4.c:7684`), `weights_have_output_head` (`ds4.c:5348`) and `weights_have_partial_output_head` (`ds4.c:5364`), make the plain `output_norm` + `output` branch include Ornith:

```c
    } else if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA ||
        DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK41 ||
        DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_QWEN35_MOE) {
```

(Same three-way condition in the two `weights_have_*` functions.)

- [ ] **Step 6: Add the layout validation**

Before `weights_validate_layout` (`ds4.c:5890`):

```c
static bool weights_qwen35moe_layer_has_required(const ds4_layer_weights *l, uint32_t il) {
    if (!l || !l->attn_norm || !l->ffn_norm) return false;
    if (ds4_qwen35_layer_is_attention(il)) {
        if (!l->attn_q || !l->attn_k || !l->attn_v || !l->attn_output || !l->attn_q_norm || !l->attn_k_norm)
            return false;
    } else if (!l->lin_qkv || !l->lin_gate || !l->lin_conv || !l->lin_dt_bias || !l->lin_a ||
               !l->lin_beta || !l->lin_alpha || !l->lin_norm || !l->lin_out) {
        return false;
    }
    if (!l->ffn_gate_inp || !l->ffn_gate_exps || !l->ffn_up_exps || !l->ffn_down_exps ||
        !l->ffn_gate_inp_shexp || !l->ffn_gate_shexp || !l->ffn_up_shexp || !l->ffn_down_shexp)
        return false;
    if (ds4_qwen35_layer_is_nextn(il) &&
        (!l->nextn_eh_proj || !l->nextn_enorm || !l->nextn_hnorm || !l->nextn_shared_head_norm))
        return false;
    return true;
}

/* Ornith routed experts: the ICE 23G/25G tiers use Q4_K and Q5_K; the 19G
 * and 21G tiers (IQ4_XS / IQ3_S) have no Metal kernels here. */
static void tensor_expect_qwen35_expert_layout(const ds4_tensor *t, uint64_t d0, uint64_t d1, uint64_t d2) {
    if (!t) ds4_die("internal error: missing tensor while validating layout");
    if (t->type != DS4_TENSOR_Q4_K && t->type != DS4_TENSOR_Q5_K) {
        fprintf(stderr, "ds4: %.*s: expert type %s not supported; use the 23G or 25G tier (Q4_K/Q5_K experts)\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, 3, d0, d1, d2);
}

static void weights_validate_qwen35moe_layout(const ds4_weights *w, uint32_t layer_start, uint32_t layer_end,
                                              bool require_token_embd, bool require_output) {
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t kv_dim = (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    const uint64_t lin_k_dim = (uint64_t)DS4_N_LIN_K_HEAD * DS4_N_LIN_HEAD_DIM;
    const uint64_t lin_v_dim = (uint64_t)DS4_N_LIN_V_HEAD * DS4_N_LIN_HEAD_DIM;
    if (!w) ds4_die("internal error: missing weights while validating Ornith layout");
    if (layer_end == UINT32_MAX) layer_end = DS4_N_LAYER - 1u;
    if (layer_start > layer_end || layer_end >= DS4_N_LAYER) ds4_die("invalid Ornith layer range");
    if (require_token_embd && !w->token_embd) ds4_die("required token embedding tensor is missing");
    if (w->token_embd) tensor_expect_qwen4_dense_layout(w->token_embd, 2, DS4_N_EMBD, DS4_N_VOCAB, 0);
    const bool have_output = weights_have_output_head(w);
    if (require_output && !have_output) ds4_die("required output head tensors are missing");
    if (have_output) {
        tensor_expect_layout(w->output_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
        tensor_expect_qwen4_dense_layout(w->output, 2, DS4_N_EMBD, DS4_N_VOCAB, 0);
    }
    uint32_t n_q4k = 0, n_q5k = 0;
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        if (!weights_qwen35moe_layer_has_required(l, il)) {
            fprintf(stderr, "ds4: required Ornith tensors for layer %u are missing\n", il);
            exit(1);
        }
        tensor_expect_layout(l->attn_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
        tensor_expect_layout(l->ffn_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
        if (ds4_qwen35_layer_is_attention(il)) {
            tensor_expect_qwen4_dense_layout(l->attn_q, 2, DS4_N_EMBD, 2u * q_dim, 0);
            tensor_expect_qwen4_dense_layout(l->attn_k, 2, DS4_N_EMBD, kv_dim, 0);
            tensor_expect_qwen4_dense_layout(l->attn_v, 2, DS4_N_EMBD, kv_dim, 0);
            tensor_expect_qwen4_dense_layout(l->attn_output, 2, q_dim, DS4_N_EMBD, 0);
            tensor_expect_layout(l->attn_q_norm, DS4_TENSOR_F32, 1, DS4_N_HEAD_DIM, 0, 0);
            tensor_expect_layout(l->attn_k_norm, DS4_TENSOR_F32, 1, DS4_N_HEAD_DIM, 0, 0);
        } else {
            tensor_expect_qwen4_dense_layout(l->lin_qkv, 2, DS4_N_EMBD, 2u * lin_k_dim + lin_v_dim, 0);
            tensor_expect_qwen4_dense_layout(l->lin_gate, 2, DS4_N_EMBD, lin_v_dim, 0);
            tensor_expect_layout(l->lin_conv, DS4_TENSOR_F32, 2, DS4_N_LIN_CONV, 2u * lin_k_dim + lin_v_dim, 0);
            tensor_expect_layout(l->lin_dt_bias, DS4_TENSOR_F32, 1, DS4_N_LIN_V_HEAD, 0, 0);
            tensor_expect_layout(l->lin_a, DS4_TENSOR_F32, 1, DS4_N_LIN_V_HEAD, 0, 0);
            tensor_expect_qwen4_dense_layout(l->lin_beta, 2, DS4_N_EMBD, DS4_N_LIN_V_HEAD, 0);
            tensor_expect_qwen4_dense_layout(l->lin_alpha, 2, DS4_N_EMBD, DS4_N_LIN_V_HEAD, 0);
            tensor_expect_layout(l->lin_norm, DS4_TENSOR_F32, 1, DS4_N_LIN_HEAD_DIM, 0, 0);
            tensor_expect_qwen4_dense_layout(l->lin_out, 2, lin_v_dim, DS4_N_EMBD, 0);
        }
        tensor_expect_layout(l->ffn_gate_inp, DS4_TENSOR_F32, 2, DS4_N_EMBD, DS4_N_EXPERT, 0);
        tensor_expect_qwen35_expert_layout(l->ffn_gate_exps, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
        tensor_expect_qwen35_expert_layout(l->ffn_up_exps, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
        tensor_expect_qwen35_expert_layout(l->ffn_down_exps, DS4_N_FF_EXP, DS4_N_EMBD, DS4_N_EXPERT);
        if (l->ffn_gate_exps->type != l->ffn_up_exps->type) {
            fprintf(stderr, "ds4: routed gate/up experts use different quant types in layer %u\n", il);
            exit(1);
        }
        if (l->ffn_gate_exps->type == DS4_TENSOR_Q5_K) n_q5k++; else n_q4k++;
        tensor_expect_layout(l->ffn_gate_inp_shexp, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
        tensor_expect_qwen4_dense_layout(l->ffn_gate_shexp, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
        tensor_expect_qwen4_dense_layout(l->ffn_up_shexp, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
        tensor_expect_qwen4_dense_layout(l->ffn_down_shexp, 2, DS4_N_FF_EXP, DS4_N_EMBD, 0);
        if (ds4_qwen35_layer_is_nextn(il)) {
            tensor_expect_qwen4_dense_layout(l->nextn_eh_proj, 2, 2u * DS4_N_EMBD, DS4_N_EMBD, 0);
            tensor_expect_layout(l->nextn_enorm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
            tensor_expect_layout(l->nextn_hnorm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
            tensor_expect_layout(l->nextn_shared_head_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
        }
    }
    fprintf(stderr, "ds4: %s: %u layers + MTP, routed experts Q5_K x%u / Q4_K x%u\n",
            DS4_MODEL_SHAPE_NAME, DS4_N_LAYER - DS4_N_NEXTN_PREDICT, n_q5k, n_q4k);
}
```

In `weights_validate_layout`, after the qwen4 branch (`ds4.c:5906`):

```c
    if (ds4_model_is_qwen35moe()) {
        weights_validate_qwen35moe_layout(w, layer_start, layer_end, require_token_embd, require_output);
        return;
    }
```

In `weights_layer_has_required` (`ds4.c:5646`), add the Ornith check next to the qwen4 one:

```c
    if (ds4_model_is_qwen35moe()) return weights_qwen35moe_layer_has_required(l, il);
```

- [ ] **Step 7: Switch the two tokenizer sites to the shared predicate**

`ds4.c:43316` (`bpe_tokenize_text`) and `ds4.c:43443` (`vocab_load`): replace `if (ds4_model_is_qwen4())` with `if (ds4_model_uses_qwen35_text())`. The other six SHARED sites in Appendix B belong to M3 (chat rendering, server and agent).

- [ ] **Step 8: Open gate, inspect path and memory estimate**

After the qwen4 open gate (`ds4.c:71682`, the closing brace of `if (ds4_model_is_qwen4() && !opt->inspect_only) { ... }`):

```c
    if (ds4_model_is_qwen35moe() && !opt->inspect_only) {
        const char *bad =
            e->backend != DS4_BACKEND_METAL ? "a non-Metal backend" :
            opt->ssd_streaming ? "--ssd-streaming" :
            (opt->tp.role != DS4_TP_NONE || opt->cuda_tensor_parallel) ? "tensor parallelism" :
            (gpu_cfg && gpu_cfg->n_gpus > 1) ? "--gpu placement" :
            opt->distributed.role != DS4_DISTRIBUTED_NONE ? "distributed inference" :
            load_slice ? "a layer slice" :
            opt->dspark ? "--dspark" :
            opt->glm_mtp ? "--mtp" :
            (opt->mtp_path && opt->mtp_path[0]) ? "--mtp-model" :
            (opt->vision_path && opt->vision_path[0]) ? "--vision" :
            (opt->ple_path && opt->ple_path[0]) ? "--ple" :
            ((opt->directional_steering_file && opt->directional_steering_file[0]) ||
             opt->directional_steering_attn != 0.0f || opt->directional_steering_ffn != 0.0f) ?
                "directional steering" :
            e->power_percent != 100 ? "--power" :
            (opt->first_token_test || opt->metal_graph_test) ? "legacy diagnostics" : NULL;
        if (bad) {
            fprintf(stderr, "ds4: Ornith-1.5-35B-A3B runs on single-host Metal only; %s is not supported\n", bad);
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
    }
```

All `opt->` fields above exist in `ds4_engine_options` (`ds4.h:133-173`, checked while planning). `gpu_cfg` and `load_slice` are the locals the qwen4 gate uses at the same point.

Next to the qwen4 inspect early return (`ds4.c:71962`):

```c
    if (ds4_model_is_qwen35moe() && opt->inspect_only) {
        *out = e;
        return 0;
    }
```

In `ds4_context_memory_estimate_with_prefill_mode` (`ds4.c:39756`), after the qwen4 block:

```c
    if (ds4_backend_uses_graph(backend) && ds4_model_is_qwen35moe()) {
        /* F16 K/V per full-attention trunk layer, rope positions, fixed GDN
         * state and conv history; transients scale with the prefill chunk. */
        const uint64_t T = prefill_chunk ? prefill_chunk : (ctx < 2048u ? ctx : 2048u);
        const uint64_t E = DS4_N_EMBD, kv_row = 2ull * DS4_N_HEAD_KV * DS4_N_HEAD_DIM * 2u;
        uint32_t n_attn = 0;
        for (uint32_t il = 0; il + DS4_N_NEXTN_PREDICT < DS4_N_LAYER; il++) n_attn += ds4_qwen35_layer_is_attention(il);
        const uint32_t n_lin = DS4_N_LAYER - DS4_N_NEXTN_PREDICT - n_attn;
        m.prefill_cap = (uint32_t)T;
        m.raw_cap = ctx;
        m.raw_bytes = (uint64_t)n_attn * ctx * kv_row + (uint64_t)ctx * 16u;
        m.scratch_bytes = T * (8u * E + 4u * DS4_N_LIN_CONV_DIM + 6u * (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM +
                               DS4_N_EXPERT + (uint64_t)(DS4_N_EXPERT_USED + 1u) * (E + DS4_N_FF_EXP)) * 4u +
                          (uint64_t)n_lin * ((uint64_t)DS4_N_LIN_V_HEAD * DS4_N_LIN_HEAD_DIM * DS4_N_LIN_HEAD_DIM +
                                             (uint64_t)(DS4_N_LIN_CONV - 1u) * DS4_N_LIN_CONV_DIM) * sizeof(float);
        m.total_bytes = m.raw_bytes + m.scratch_bytes;
        return m;
    }
    ds4_qwen35_not_reached("context memory estimate");
```

Add the public predicate. In `ds4.h`, next to `ds4_engine_is_qwen4`:

```c
bool ds4_engine_is_qwen35moe(ds4_engine *e);
```

In `ds4.c`, next to the definition of `ds4_engine_is_qwen4` (`ds4.c:72991`):

```c
bool ds4_engine_is_qwen35moe(ds4_engine *e) {
    return e && ds4_model_is_qwen35moe();
}
```

- [ ] **Step 9: Build and run the loader test**

```bash
make ds4 && DS4_ORNITH_MODEL=$DS4_ORNITH_MODEL tests/ornith/test_loader.sh
```

Expected: `ornith loader: ok`. If `bad_tier.gguf` is rejected earlier by `model_open` (a size or alignment check), keep the patched types but also shrink the patched tensors' byte counts consistently. Record in the commit message what `model_open` checks.

- [ ] **Step 10: Check the tokenizer against llama.cpp**

```bash
for n in en_capital vi_greeting code_py; do
  python3 - "$n" > /tmp/ornith-$n.txt <<'EOF'
import json, sys
sys.path.insert(0, "tests/ornith")
import ornith_ref as r
p = next(x for x in r.load_prompts("tests/ornith/prompts.json") if x["name"] == sys.argv[1])
sys.stdout.write(r.prompt_text(p, "."))
EOF
  ./ds4 -m "$DS4_ORNITH_MODEL" --raw --prompt-file /tmp/ornith-$n.txt --dump-tokens 2>/dev/null | head -1 > /tmp/ornith-$n.ids
  python3 -c "import json,sys; a=json.load(open('/tmp/ornith-$n.ids')); b=json.load(open('tests/ornith/ref/$n.json'))['prompt_ids']; print('$n', 'same' if a==b else ('DIFF', a, b))"
done
```

Expected: three `same` lines.

- [ ] **Step 11: Build every target that compiles `ds4.c`, and rerun the Qwen3.8 kernel tests**

```bash
make ds4 ds4-server ds4-bench ds4-eval ds4-agent && make cpu && make test-qwen4-kernels test-qwen4-q2
```

Expected: no errors or new warnings; kernel tests pass.

- [ ] **Step 12: Update the spec sentence and commit**

In the spec §3 "Where the code lives", replace the first bullet's opening sentence with: "The graph, one-shot generation and session helpers for the family go in a new `ds4_qwen35moe.inc`; the validator, weight binding and layout validation stay beside their qwen4 counterparts in `ds4.c` because `weights_bind()` runs before the graph block."

```bash
git add ds4.c ds4.h tests/ornith/make_bad_gguf.py tests/ornith/test_loader.sh docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md
git commit -m "qwen35: Ornith family, validator, weight binding and tokenizer

qwen35moe GGUFs select DS4_MODEL_FAMILY_QWEN35_MOE; every metadata value and
tensor is checked, unsupported expert tiers fail by name, and the Qwen3.5
tokenizer sites take the shared predicate. Metal-only open gate.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- ds4.c ds4.h tests/ornith/make_bad_gguf.py tests/ornith/test_loader.sh docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md
```

---

### Task 6: Graph and one-shot generation

**Files:**
- Create: `ds4_qwen35moe.inc`
- Modify: `ds4.c` (graph struct field, `qwen4_graph_linear`, include, `generate_metal_graph_raw_swa`), `Makefile` (dependency)

**Interfaces:**
- Consumes: Task 3/4 wrappers, Task 5 predicates and binding.
- Produces (static, in `ds4_qwen35moe.inc`):
  ```c
  static uint32_t qwen35_prefill_chunk_tokens(uint32_t ctx);
  static bool qwen35_graph_alloc(ds4_qwen4_gpu_graph *g, uint32_t ctx_cap, uint32_t cap_tokens);
  static void qwen35_graph_reset(ds4_qwen4_gpu_graph *g);
  static bool qwen35_graph_forward_tokens(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                          const int *tokens, uint32_t T, float *logits_out);
  static int generate_qwen35_metal_argmax(const ds4_model *model, const ds4_vocab *vocab,
                                          const ds4_weights *weights, const token_vec *prompt,
                                          int n_predict, int ctx_size, uint32_t prefill_chunk,
                                          ds4_token_emit_fn emit, ds4_generation_done_fn done, void *emit_ud,
                                          ds4_session_progress_fn progress, void *progress_ud);
  ```
  Graphs are freed with the existing `qwen4_graph_free`.
- New field in `ds4_qwen4_gpu_graph`: `bool gdn_silu;` (false for Qwen3.8 through the `memset` in `qwen4_graph_alloc`).

- [ ] **Step 1: Write the failing end-to-end check**

This task's test is the CLI's one-shot greedy path against the text llama.cpp generated for three prompts (the `content` field Task 2 recorded). Create `tests/ornith/oneshot.sh` (executable):

```sh
#!/bin/sh
# One-shot greedy generation on Ornith against llama.cpp's generated text.
# ds4 writes 24 tokens; the reference holds more, so ds4's text must be a
# prefix of it.  A mismatch may be a near tie: Task 8 decides at id level.
set -eu
model=${DS4_ORNITH_MODEL:?}
status=0
for n in en_capital code_py vi_hanoi; do
  python3 - "$n" > "/tmp/ornith-$n.txt" <<'EOF'
import sys
sys.path.insert(0, "tests/ornith")
import ornith_ref as r
p = next(x for x in r.load_prompts("tests/ornith/prompts.json") if x["name"] == sys.argv[1])
sys.stdout.write(r.prompt_text(p, "."))
EOF
  ./ds4 -m "$model" --metal --raw --prompt-file "/tmp/ornith-$n.txt" -c 4096 --temp 0 -n 24 \
      > "/tmp/ornith-$n.out" 2> "/tmp/ornith-$n.err" || { tail -5 "/tmp/ornith-$n.err"; exit 1; }
  python3 - "$n" <<'EOF' || status=1
import json, sys
n = sys.argv[1]
ref = json.load(open(f"tests/ornith/ref/{n}.json"))["content"]
got = open(f"/tmp/ornith-{n}.out", encoding="utf-8", errors="replace").read().rstrip("\n")
same = 0
while same < min(len(got), len(ref)) and got[same] == ref[same]:
    same += 1
print(f"{n}: {'match' if same == len(got) else 'DIFF'} ({same}/{len(got)} chars)")
if same != len(got):
    print("  ds4:", repr(got[:160]))
    print("  ref:", repr(ref[:160]))
    sys.exit(1)
EOF
done
exit $status
```

Run it now: `make ds4 && tests/ornith/oneshot.sh`.
Expected: failure with `internal error: Ornith reached the DeepSeek path in one-shot generation`, or earlier in the open path.

- [ ] **Step 2: Add the `gdn_silu` hook**

In `ds4_qwen4_gpu_graph` (`ds4.c:57984`, before `} ds4_qwen4_gpu_graph;`):

```c
    /* Ornith (qwen35moe) gates the GDN output with silu(z); Qwen3.8 with
     * sigmoid(z).  qwen4_graph_alloc's memset leaves it false. */
    bool gdn_silu;
```

In `qwen4_graph_linear` (`ds4.c:58972-58975`), replace the `gdn_out` call:

```c
    if (ok) {
#ifdef DS4_HAS_QWEN4_METAL
        if (g->gdn_silu)
            ok = ds4_gpu_qwen35_gdn_out_tensor(g->lin_o, g->z, m->map, m->size, l->lin_norm->abs_offset, T,
                                               DS4_N_LIN_V_HEAD, DS4_N_LIN_HEAD_DIM, DS4_RMS_EPS) != 0;
        else
#endif
        ok = ds4_gpu_qwen4_gdn_out_tensor(g->lin_o, g->z, m->map, m->size, l->lin_norm->abs_offset, T,
                                          DS4_N_LIN_V_HEAD, DS4_N_LIN_HEAD_DIM, DS4_RMS_EPS) != 0;
    }
```

- [ ] **Step 3: Write `ds4_qwen35moe.inc`**

```c
/* Ornith-1.5-35B-A3B (llama.cpp qwen35moe) on Metal.
 *
 * Included by ds4.c right after the qwen4 Metal graph, so the family reuses
 * the qwen4 graph struct, GEMV and kernels.  GDN layers call
 * qwen4_graph_linear with the silu output gate (g->gdn_silu).  Attention is
 * the qwen4 gated attention without the QSA indexer: every position is
 * attended densely.  The MoE reuses the router and the Q4_K expert kernels,
 * with the qwen35 row kernels for Q5_K layers.  The residual stream is one
 * n_embd row per token, with no hyper-connections and no PLE layer.  The
 * llama.cpp converter already folded +1 into the norm weights, stores
 * ssm_a = -exp(A_log) and tiles the GDN value heads, which is what the qwen4
 * kernels expect. */

static uint32_t qwen35_prefill_chunk_tokens(uint32_t ctx) {
    const char *env = getenv("DS4_QWEN35_PREFILL_CHUNK");
    const unsigned long v = env && env[0] ? strtoul(env, NULL, 10) : 2048ul;
    const uint32_t chunk = v == 0 || v > 65536ul ? 2048u : (uint32_t)v;
    return chunk > ctx ? ctx : chunk;
}

static bool qwen35_graph_alloc(ds4_qwen4_gpu_graph *g, uint32_t ctx_cap, uint32_t cap_tokens) {
    memset(g, 0, sizeof(*g));
    const uint64_t E = DS4_N_EMBD, T = cap_tokens;
    const uint64_t conv_dim = DS4_N_LIN_CONV_DIM;
    const uint64_t v_dim = (uint64_t)DS4_N_LIN_V_HEAD * DS4_N_LIN_HEAD_DIM;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t kv_dim = (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    g->ctx_cap = ctx_cap;
    g->alloc_cap = ctx_cap;
    g->cap_tokens = cap_tokens;
    g->n_logit_rows = 1u;
    g->owns_scratch = true;
    g->gdn_silu = true;
    bool ok = true;
#define QWEN35_ALLOC(field_, n_) do { g->field_ = qwen4_graph_alloc_f32(n_); ok = ok && g->field_; } while (0)
    QWEN35_ALLOC(R, T * E);
    QWEN35_ALLOC(mixed, T * E);
    QWEN35_ALLOC(blk, T * E);
    QWEN35_ALLOC(qkv, T * conv_dim);
    QWEN35_ALLOC(z, T * v_dim);
    QWEN35_ALLOC(ga, T * DS4_N_LIN_V_HEAD);
    QWEN35_ALLOC(gb, T * DS4_N_LIN_V_HEAD);
    QWEN35_ALLOC(lin_o, T * v_dim);
    QWEN35_ALLOC(qg, T * 2u * q_dim);
    QWEN35_ALLOC(kp, T * kv_dim);
    QWEN35_ALLOC(vp, T * kv_dim);
    QWEN35_ALLOC(q, T * q_dim);
    QWEN35_ALLOC(gate, T * q_dim);
    QWEN35_ALLOC(attn_o, T * q_dim);
    QWEN35_ALLOC(attn_part, ds4_gpu_qwen4_attn_part_floats(3u, DS4_N_HEAD, DS4_N_HEAD_DIM));
    QWEN35_ALLOC(router, T * DS4_N_EXPERT);
    QWEN35_ALLOC(selected, T * DS4_N_EXPERT_USED);
    QWEN35_ALLOC(weights, T * DS4_N_EXPERT_USED);
    QWEN35_ALLOC(mid, T * (DS4_N_EXPERT_USED + 1u) * DS4_N_FF_EXP);
    QWEN35_ALLOC(part, T * (DS4_N_EXPERT_USED + 1u) * E);
    QWEN35_ALLOC(sh_gate_logit, T);
    QWEN35_ALLOC(moe_lists, (uint64_t)DS4_N_EXPERT * T);
    QWEN35_ALLOC(moe_counts, DS4_N_EXPERT);
    QWEN35_ALLOC(sh_gate, T * DS4_N_FF_EXP);
    QWEN35_ALLOC(sh_up, T * DS4_N_FF_EXP);
    QWEN35_ALLOC(sh_mid, T * DS4_N_FF_EXP);
    QWEN35_ALLOC(sh_out, T * E);
    QWEN35_ALLOC(logits, (uint64_t)DS4_N_VOCAB);
    QWEN35_ALLOC(pos3, (uint64_t)ctx_cap * 4u);
#undef QWEN35_ALLOC
    /* F16 K/V per full-attention trunk layer; recurrent state and conv
     * history per GDN layer.  The MTP block's caches arrive with M2. */
    const uint64_t kv_bytes = (uint64_t)ctx_cap * kv_dim * 2u;
    for (uint32_t il = 0; il + DS4_N_NEXTN_PREDICT < DS4_N_LAYER && ok; il++) {
        if (ds4_qwen35_layer_is_attention(il)) {
            g->layer_k_cache[il] = ds4_gpu_tensor_alloc(kv_bytes);
            g->layer_v_cache[il] = ds4_gpu_tensor_alloc(kv_bytes);
            ok = g->layer_k_cache[il] && g->layer_v_cache[il];
        } else {
            g->layer_lin_state[il] = qwen4_graph_alloc_f32(v_dim * DS4_N_LIN_HEAD_DIM);
            g->layer_lin_hist[il] = qwen4_graph_alloc_f32((uint64_t)(DS4_N_LIN_CONV - 1u) * conv_dim);
            ok = g->layer_lin_state[il] && g->layer_lin_hist[il];
        }
    }
    g->host_row = xmalloc(T * E * sizeof(float));
    g->host_pos3 = xmalloc(T * 4u * sizeof(uint32_t));
    g->host_logits = xmalloc((uint64_t)DS4_N_VOCAB * sizeof(float));
    if (!ok) {
        fprintf(stderr, "ds4: Ornith graph allocation failed (ctx %u)\n", ctx_cap);
        qwen4_graph_free(g);
        return false;
    }
    return true;
}

static void qwen35_graph_reset(ds4_qwen4_gpu_graph *g) {
    (void)ds4_gpu_synchronize();
    const uint64_t conv_dim = DS4_N_LIN_CONV_DIM;
    const uint64_t v_dim = (uint64_t)DS4_N_LIN_V_HEAD * DS4_N_LIN_HEAD_DIM;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (g->layer_lin_state[il]) ds4_gpu_tensor_fill_f32(g->layer_lin_state[il], 0.0f, v_dim * DS4_N_LIN_HEAD_DIM);
        if (g->layer_lin_hist[il])
            ds4_gpu_tensor_fill_f32(g->layer_lin_hist[il], 0.0f, (uint64_t)(DS4_N_LIN_CONV - 1u) * conv_dim);
    }
    g->pos = 0;
}

/* Embedding rows (dequantized on the host from Q8_0) and text rope
 * positions (t, h, w) = (p, p, p). */
static bool qwen35_graph_stage_inputs(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                      const int *tokens, uint32_t T) {
    const uint32_t E = DS4_N_EMBD;
    for (uint32_t t = 0; t < T; t++) {
        qwen4_ref_row(m, w->token_embd, (uint64_t)tokens[t], g->host_row + (uint64_t)t * E);
        uint32_t *p4 = g->host_pos3 + (uint64_t)t * 4u;
        p4[0] = p4[1] = p4[2] = g->pos + t;
        p4[3] = 0;
    }
    return ds4_gpu_tensor_write(g->R, 0, g->host_row, (uint64_t)T * E * sizeof(float)) &&
           ds4_gpu_tensor_write(g->pos3, (uint64_t)g->pos * 16u, g->host_pos3, (uint64_t)T * 16u);
}

/* Gated full attention over every cached position: q/gate, k and v
 * projections, indexer-free prep (norms, RoPE, F16 KV append), the qwen4
 * dense attention kernels (which apply sigmoid(gate)), output projection. */
static bool qwen35_graph_attention(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                                   uint32_t il, uint32_t pos0, uint32_t T) {
    const float scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);
    return qwen4_gemv(g->qg, m, l->attn_q, g->mixed, T) &&
           qwen4_gemv(g->kp, m, l->attn_k, g->mixed, T) &&
           qwen4_gemv(g->vp, m, l->attn_v, g->mixed, T) &&
           ds4_gpu_qwen35_attn_prep_tensor(g->q, g->gate, g->layer_k_cache[il], g->layer_v_cache[il],
                                           g->qg, g->kp, g->vp, g->pos3, m->map, m->size,
                                           l->attn_q_norm->abs_offset, l->attn_k_norm->abs_offset,
                                           T, DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT,
                                           pos0, g->alloc_cap, DS4_ROPE_FREQ_BASE, DS4_RMS_EPS) != 0 &&
           ds4_gpu_qwen4_attn_decode_tensor(g->attn_o, g->q, g->gate, g->layer_k_cache[il], g->layer_v_cache[il],
                                            NULL, NULL, T <= 2u ? g->attn_part : NULL, T,
                                            DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, pos0, 0u, 0u, scale,
                                            NULL, NULL, NULL, NULL, 0u) != 0 &&
           qwen4_gemv(g->blk, m, l->attn_output, g->attn_o, T);
}

/* Router (softmax over 256, top 8, renormalized, sigmoid shared gate), the
 * routed experts and the shared expert, reduced into blk without a residual
 * (n_hc = 0); the caller adds blk to R.  Q4_K layers take the tiled expert
 * GEMMs above one tile of tokens, as qwen4 prefill does; Q5_K layers keep
 * the per-token row kernels.  Batches above 8 rows run the shared expert as
 * dense projections, single rows and verifies keep it as an extra slot. */
static bool qwen35_graph_moe(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                             uint32_t T) {
    const uint32_t E = DS4_N_EMBD, F = DS4_N_FF_EXP, NE = DS4_N_EXPERT, K = DS4_N_EXPERT_USED;
    const uint32_t xt = l->ffn_gate_exps->type, dt = l->ffn_down_exps->type;
    bool ok = qwen4_gemv(g->router, m, l->ffn_gate_inp, g->mixed, T) &&
        ds4_gpu_qwen4_router_topk_tensor(g->selected, g->weights, g->router, g->mixed, m->map, m->size,
                                         l->ffn_gate_inp_shexp->abs_offset, l->ffn_gate_inp_shexp->type,
                                         E, g->sh_gate_logit, T, NE, K) != 0;
    const bool mm = T > 64u && xt == DS4_TENSOR_Q4_K && dt == DS4_TENSOR_Q4_K;
    const bool shared_dense = mm || T > 8u;
    if (ok && shared_dense) {
        ok = qwen4_gemv(g->sh_gate, m, l->ffn_gate_shexp, g->mixed, T) &&
             qwen4_gemv(g->sh_up, m, l->ffn_up_shexp, g->mixed, T) &&
             ds4_gpu_swiglu_tensor(g->sh_mid, g->sh_gate, g->sh_up, T * F, 0.0f, 1.0f) &&
             qwen4_gemv(g->sh_out, m, l->ffn_down_shexp, g->sh_mid, T);
    }
    if (ok && mm) {
        ok = ds4_gpu_qwen4_moe_build_lists_tensor(g->moe_lists, g->moe_counts, g->selected, T, K, NE,
                                                  g->cap_tokens) &&
             ds4_gpu_qwen4_moe_mm_mid_tensor(g->mid, g->mixed, g->moe_lists, g->moe_counts, m->map, m->size,
                                             l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset, xt,
                                             NE, T, K, K, E, F, g->cap_tokens) &&
             ds4_gpu_qwen4_moe_mm_down_tensor(g->part, g->mid, g->moe_lists, g->moe_counts, m->map, m->size,
                                              l->ffn_down_exps->abs_offset, dt, NE, T, K, K, F, E, g->cap_tokens);
    } else if (ok) {
        const uint64_t sg = shared_dense ? 0u : l->ffn_gate_shexp->abs_offset;
        const uint64_t su = shared_dense ? 0u : l->ffn_up_shexp->abs_offset;
        const uint64_t sd = shared_dense ? 0u : l->ffn_down_shexp->abs_offset;
        const uint32_t st = shared_dense ? UINT32_MAX : l->ffn_gate_shexp->type;
        const uint32_t sdt = shared_dense ? UINT32_MAX : l->ffn_down_shexp->type;
        if (xt == DS4_TENSOR_Q5_K) {
            ok = ds4_gpu_qwen35_moe_mid_tensor(g->mid, g->mixed, g->selected, m->map, m->size,
                                               l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset, xt,
                                               NE, T, K, E, F, sg, su, st) != 0;
        } else {
            ok = ds4_gpu_qwen4_moe_mid_tensor(g->mid, g->mixed, g->selected, m->map, m->size,
                                              l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset, xt,
                                              NE, T, K, E, F, sg, su, st) != 0;
        }
        if (ok && dt == DS4_TENSOR_Q5_K) {
            ok = ds4_gpu_qwen35_moe_down_tensor(g->part, g->mid, g->selected, m->map, m->size,
                                                l->ffn_down_exps->abs_offset, dt, NE, T, K, F, E, sd, sdt) != 0;
        } else if (ok) {
            ok = ds4_gpu_qwen4_moe_down_tensor(g->part, g->mid, g->selected, m->map, m->size,
                                               l->ffn_down_exps->abs_offset, dt, NE, T, K, F, E, sd, sdt) != 0;
        }
    }
    if (ok) {
        ok = ds4_gpu_qwen4_moe_reduce_tensor(g->blk, g->part, g->weights, g->sh_gate_logit,
                                             shared_dense ? g->sh_out : NULL, NULL, NULL, T, K,
                                             K + (shared_dense ? 0u : 1u), E, 0u) != 0;
    }
    return ok;
}

/* One trunk layer: pre-norm mixer and pre-norm MoE, each added to R. */
static bool qwen35_graph_layer(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                               uint32_t il, uint32_t pos0, uint32_t T) {
    const ds4_layer_weights *l = &w->layer[il];
    const uint32_t n = T * DS4_N_EMBD;
    bool ok = ds4_gpu_rms_norm_weight_rows_tensor(g->mixed, g->R, m->map, m->size, l->attn_norm->abs_offset,
                                                  DS4_N_EMBD, T, DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_qwen35_layer_is_attention(il) ? qwen35_graph_attention(g, m, l, il, pos0, T)
                                                    : qwen4_graph_linear(g, m, l, il, T);
    if (ok) ok = ds4_gpu_add_tensor(g->R, g->R, g->blk, n) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->mixed, g->R, m->map, m->size, l->ffn_norm->abs_offset,
                                                     DS4_N_EMBD, T, DS4_RMS_EPS) != 0;
    if (ok) ok = qwen35_graph_moe(g, m, l, T);
    if (ok) ok = ds4_gpu_add_tensor(g->R, g->R, g->blk, n) != 0;
    return ok;
}

/* Forward T tokens at g->pos..; logits_out (optional) receives the last
 * row.  Causal by construction: the recurrent kernels walk tokens in order
 * and attention reads the caches written for the same chunk. */
static bool qwen35_graph_forward_tokens(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                        const int *tokens, uint32_t T, float *logits_out) {
    if (!g || T == 0 || T > g->cap_tokens || g->pos + T > g->ctx_cap) return false;
    for (uint32_t t = 0; t < T; t++) {
        if (tokens[t] < 0 || tokens[t] >= (int)DS4_N_VOCAB) {
            fprintf(stderr, "ds4: Ornith token id %d is outside the vocabulary\n", tokens[t]);
            return false;
        }
    }
    const uint32_t pos0 = g->pos;
    if (!qwen35_graph_stage_inputs(g, m, w, tokens, T)) return false;
    if (!glm_graph_begin_commands_if_needed()) return false;
    bool ok = true;
    for (uint32_t il = 0; il + DS4_N_NEXTN_PREDICT < DS4_N_LAYER && ok; il++) {
        ok = qwen35_graph_layer(g, m, w, il, pos0, T);
    }
    if (ok && logits_out) {
        ds4_gpu_tensor *last = ds4_gpu_tensor_view(g->R, (uint64_t)(T - 1u) * DS4_N_EMBD * sizeof(float),
                                                   (uint64_t)DS4_N_EMBD * sizeof(float));
        ok = last && ds4_gpu_rms_norm_weight_rows_tensor(g->mixed, last, m->map, m->size, w->output_norm->abs_offset,
                                                         DS4_N_EMBD, 1u, DS4_RMS_EPS) != 0 &&
             qwen4_gemv(g->logits, m, w->output, g->mixed, 1u);
        ds4_gpu_tensor_free(last);
    }
    if (!ds4_gpu_end_commands()) ok = false;
    if (ok && logits_out) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits_out, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (ok) g->pos += T;
    return ok;
}

static int generate_qwen35_metal_argmax(
        const ds4_model *model, const ds4_vocab *vocab, const ds4_weights *weights,
        const token_vec *prompt, int n_predict, int ctx_size, uint32_t prefill_chunk,
        ds4_token_emit_fn emit, ds4_generation_done_fn done, void *emit_ud,
        ds4_session_progress_fn progress, void *progress_ud) {
    if (!prompt || prompt->len <= 0 || prompt->len >= ctx_size) {
        fprintf(stderr, "ds4: prompt is empty or exceeds context size\n");
        return 1;
    }
    const uint32_t cap = prefill_chunk && prefill_chunk < (uint32_t)ctx_size ?
        prefill_chunk : qwen35_prefill_chunk_tokens((uint32_t)ctx_size);
    ds4_qwen4_gpu_graph *g = xcalloc(1, sizeof(*g));
    if (!qwen35_graph_alloc(g, (uint32_t)ctx_size, cap)) {
        free(g);
        return 1;
    }
    qwen35_graph_reset(g);
    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    const double t_prefill0 = now_sec();
    bool ok = true;
    for (int i = 0; i < prompt->len && ok;) {
        uint32_t chunk = (uint32_t)(prompt->len - i);
        if (chunk > g->cap_tokens) chunk = g->cap_tokens;
        ok = qwen35_graph_forward_tokens(g, model, weights, prompt->v + i, chunk,
                                         i + (int)chunk == prompt->len ? logits : NULL);
        i += (int)chunk;
        if (progress) progress(progress_ud, "prefill_chunk", i, prompt->len);
    }
    const double t_prefill1 = now_sec();
    int n_generated = 0;
    const double t_decode0 = now_sec();
    for (int i = 0; ok && i < n_predict && g->pos < g->ctx_cap; i++) {
        const int token = sample_argmax(logits, DS4_N_VOCAB);
        if (vocab_token_is_generation_stop(vocab, token)) break;
        if (emit) emit(emit_ud, token);
        n_generated++;
        if (i == n_predict - 1 || g->pos + 1u >= g->ctx_cap) break;
        ok = qwen35_graph_forward_tokens(g, model, weights, &token, 1, logits);
    }
    const double t_decode1 = now_sec();
    if (!ok) fprintf(stderr, "ds4: Ornith forward failed at position %u\n", g->pos);
    if (ok && done) done(emit_ud);
    ds4_log(stderr, DS4_LOG_TIMING, "ds4: Ornith prefill: %.2f t/s, generation: %.2f t/s\n",
            t_prefill1 > t_prefill0 ? (double)prompt->len / (t_prefill1 - t_prefill0) : 0.0,
            t_decode1 > t_decode0 ? (double)n_generated / (t_decode1 - t_decode0) : 0.0);
    free(logits);
    qwen4_graph_free(g);
    free(g);
    return ok ? 0 : 1;
}
```

- [ ] **Step 4: Include the file and dispatch the one-shot path**

In `ds4.c`, immediately before the `#endif` at `ds4.c:60273` that closes `#ifdef DS4_HAS_QWEN4_GPU` (after `generate_qwen4_metal_argmax`):

```c
#ifdef DS4_HAS_QWEN4_METAL
#include "ds4_qwen35moe.inc"
#endif
```

In `generate_metal_graph_raw_swa` (`ds4.c:60308`), after the qwen4 block:

```c
#ifdef DS4_HAS_QWEN4_METAL
    if (ds4_model_is_qwen35moe()) {
        return generate_qwen35_metal_argmax(model, vocab, weights, prompt, n_predict, ctx_size, prefill_chunk,
                                            emit, done, emit_ud, progress, progress_ud);
    }
#endif
    ds4_qwen35_not_reached("one-shot generation");
```

In `Makefile:1095`, add the `.inc` next to the unicode one:

```make
ds4.o ds4_cpu.o ds4_cpu_test_hooks.o: ds4_qwen4_unicode.inc ds4_qwen35moe.inc
```

- [ ] **Step 5: Run the one-shot check**

```bash
make ds4 && tests/ornith/oneshot.sh
```

Expected: three `match` lines.

- **The normal (non-inspect) open stops at a DeepSeek-only step.** Task 5 only exercised `--inspect`. Find the step and exclude Ornith there the way the qwen4 family is excluded at that site. Add the exclusion with `!ds4_model_is_qwen35moe()`, and never widen `ds4_model_is_qwen4()`.
- **The text is garbage, or `DIFF` shows early.** Stop and debug with `superpowers:systematic-debugging` before going on. Compare per-layer hidden states against llama.cpp (`llama-eval-callback`) on `en_capital`, starting from layer 0.
- **A `DIFF` appears late in a line.** It may be a near tie; carry on and let Task 8 decide.

- [ ] **Step 6: Build all targets and the Qwen3.8 kernel tests**

```bash
make ds4 ds4-server ds4-bench ds4-eval ds4-agent && make cpu && make test-qwen4-kernels test-qwen4-q2 test-qwen35-kernels
```

Expected: clean build, all kernel tests pass.

- [ ] **Step 7: Commit**

```bash
chmod +x tests/ornith/oneshot.sh
git add ds4_qwen35moe.inc ds4.c Makefile tests/ornith/oneshot.sh
git commit -m "qwen35: Ornith graph and one-shot greedy generation on Metal

Plain-residual layer loop over the qwen4 GDN, dense gated attention and MoE
kernels; the GDN output takes the silu gate through a graph flag that stays
false for Qwen3.8.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- ds4_qwen35moe.inc ds4.c Makefile tests/ornith/oneshot.sh
```

---

### Task 7: Session core

**Files:**
- Modify: `ds4.c` (session struct ~61218, predicate ~62242, create ~73955/74029, free ~74470, sync ~76618, eval ~78692)
- Create: `tests/test_qwen35_session.c`
- Modify: `Makefile`

**Interfaces:**
- Consumes: Task 6 graph functions.
- Produces: `static bool ds4_session_is_qwen35(const ds4_session *s)`; field `bool qwen35_graph_ready;` in `ds4_session`. `ds4_session_create/sync/eval/free` then work for Ornith through the public API in `ds4.h`.

- [ ] **Step 1: Write the failing session test**

Create `tests/test_qwen35_session.c`:

```c
/* Real-model Ornith session checks.
 * Usage: test_qwen35_session MODEL
 *  - prefix extension across chunk boundaries equals a fresh replay of the same
 *    syncs (< 1e-3, same argmax); the one-pass difference is reported only;
 *  - a divergent prompt resets the recurrent state (logits bit-identical to a fresh session);
 *  - a full context stops eval with an error. */
#define _POSIX_C_SOURCE 200809L
#include "../ds4.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void sync_len(ds4_session *s, const ds4_tokens *tokens, int n) {
    ds4_tokens prefix = *tokens;
    prefix.len = n;
    char err[256] = {0};
    const int rc = ds4_session_sync(s, &prefix, err, sizeof(err));
    if (rc) fprintf(stderr, "sync %d: %s\n", n, err);
    assert(rc == 0 && ds4_session_pos(s) == n);
}

static float max_diff(const float *a, const float *b, int n) {
    float worst = 0.0f;
    for (int i = 0; i < n; i++) {
        assert(isfinite(a[i]) && isfinite(b[i]));
        worst = fmaxf(worst, fabsf(a[i] - b[i]));
    }
    return worst;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL\n", argv[0]);
        return 1;
    }
    const int ctx = 4096 + 64;
    ds4_engine_options opt = {.model_path = argv[1], .context_size = ctx, .prefill_chunk = 512,
                              .backend = DS4_BACKEND_METAL};
    ds4_engine *engine = NULL;
    assert(ds4_engine_open(&engine, &opt) == 0 && ds4_engine_is_qwen35moe(engine));
    FILE *fp = fopen("speed-bench/promessi_sposi.txt", "rb");
    assert(fp);
    char *text = calloc(40001, 1);
    assert(fread(text, 1, 40000, fp) > 0);
    fclose(fp);
    ds4_tokens tokens = {0};
    ds4_tokenize_text(engine, text, &tokens);
    free(text);
    assert(tokens.len >= ctx - 2);
    const int vocab = ds4_engine_vocab_size(engine);
    float *a = malloc((size_t)vocab * 4), *b = malloc((size_t)vocab * 4);
    ds4_session *live = NULL, *control = NULL;
    assert(ds4_session_create(&live, engine, ctx) == 0);
    assert(ds4_session_create(&control, engine, ctx) == 0);

    /* 1. prefix extension across chunk boundaries: the live session equals a
     * fresh session replaying the same syncs; a one-pass prefill chunks
     * differently, so its difference is reported only */
    const int lengths[] = {128, 129, 511, 512, 513, 1024, 2049, 4096};
    const size_t n_lengths = sizeof(lengths) / sizeof(*lengths);
    for (size_t j = 0; j < n_lengths; j++) {
        sync_len(live, &tokens, lengths[j]);
        assert(ds4_session_copy_logits(live, a, vocab) == vocab);
        ds4_session_invalidate(control);
        for (size_t k = 0; k <= j; k++) sync_len(control, &tokens, lengths[k]);
        assert(ds4_session_copy_logits(control, b, vocab) == vocab);
        const float d = max_diff(a, b, vocab);
        ds4_session_invalidate(control);
        sync_len(control, &tokens, lengths[j]);
        assert(ds4_session_copy_logits(control, b, vocab) == vocab);
        printf("  extend to %5d: replay max|d| %.2e, one-pass max|d| %.2e argmax %d/%d\n", lengths[j], d,
               max_diff(a, b, vocab), ds4_session_argmax(live), ds4_session_argmax(control));
        assert(d < 1e-3f);
    }

    /* 2. a divergent prompt resets the state: bit-identical to a fresh session */
    ds4_tokens other = {0};
    ds4_tokenize_text(engine, "Xin chào! Hôm nay tôi muốn kể cho bạn nghe về Hà Nội.", &other);
    sync_len(live, &other, other.len);
    assert(ds4_session_copy_logits(live, a, vocab) == vocab);
    ds4_session *fresh = NULL;
    assert(ds4_session_create(&fresh, engine, ctx) == 0);
    sync_len(fresh, &other, other.len);
    assert(ds4_session_copy_logits(fresh, b, vocab) == vocab);
    assert(memcmp(a, b, (size_t)vocab * 4) == 0);
    printf("  divergent prompt: bit-identical to a fresh session\n");

    /* 3. context full: eval stops with an error, no crash */
    ds4_session_invalidate(control);
    sync_len(control, &tokens, ctx - 2);
    char err[256] = {0};
    assert(ds4_session_eval(control, 11, err, sizeof(err)) == 0);
    assert(ds4_session_eval(control, 11, err, sizeof(err)) == 0);
    assert(ds4_session_eval(control, 11, err, sizeof(err)) != 0);
    printf("  context full: '%s'\n", err);

    ds4_session_free(fresh);
    ds4_session_free(control);
    ds4_session_free(live);
    ds4_tokens_free(&other);
    ds4_tokens_free(&tokens);
    free(a);
    free(b);
    ds4_engine_close(engine);
    printf("qwen35 session: ok\n");
    return 0;
}
```

Check the exact public names used (`ds4_engine_options` fields, `ds4_tokens_free`, `ds4_session_invalidate`, `ds4_engine_close`) against `ds4.h` before building. `tests/test_qwen4_prefill.c` uses the same API.

`Makefile`, next to the `tests/test_qwen4_prefill` rules (`Makefile:747-755`):

```make
tests/test_qwen35_session.o: tests/test_qwen35_session.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_qwen35_session.c

tests/test_qwen35_session: tests/test_qwen35_session.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

.PHONY: test-qwen35-session
test-qwen35-session: tests/test_qwen35_session
	./tests/test_qwen35_session "$(DS4_ORNITH_MODEL)"
```

Add `tests/test_qwen35_session` to the clean target.

- [ ] **Step 2: Run it and confirm it fails**

Run: `make test-qwen35-session DS4_ORNITH_MODEL=$DS4_ORNITH_MODEL`
Expected: failure at `ds4_session_create` (session falls into the DeepSeek graph path) or at the `ds4_qwen35_not_reached` guard.

- [ ] **Step 3: Session struct field and predicate**

In `ds4_session` (`ds4.c:61219`, after `int qwen4_slot;`):

```c
    /* Ornith (qwen35moe) sessions reuse qwen4_graph for their graph but never
     * set qwen4_graph_ready, so qwen4-only paths that test it stay off. */
    bool qwen35_graph_ready;
```

After `ds4_session_is_qwen4` (`ds4.c:62244`):

```c
static bool ds4_session_is_qwen35(const ds4_session *s) {
    return s && s->engine && ds4_model_is_qwen35moe();
}
```

- [ ] **Step 4: Create and free**

In the CPU branch of `ds4_session_create` (`ds4.c:73955`), next to the qwen4 refusal:

```c
        if (ds4_model_is_qwen35moe()) {
            fprintf(stderr, "ds4: Ornith sessions require Metal\n");
            return 1;
        }
```

In the graph branch, directly before `#ifdef DS4_HAS_QWEN4_GPU` / `if (ds4_model_is_qwen4()) {` (`ds4.c:74028`):

```c
#ifdef DS4_HAS_QWEN4_METAL
    if (ds4_model_is_qwen35moe()) {
        if (e->backend != DS4_BACKEND_METAL || e->distributed.role != DS4_DISTRIBUTED_NONE) {
            fprintf(stderr, "ds4: Ornith sessions require single-host Metal\n");
            free(s);
            return 1;
        }
        const uint32_t cap_tokens = e->prefill_chunk && e->prefill_chunk < (uint32_t)ctx_size ?
            e->prefill_chunk : qwen35_prefill_chunk_tokens((uint32_t)ctx_size);
        s->qwen4_slot = -1;
        if (!qwen35_graph_alloc(&s->qwen4_graph, (uint32_t)ctx_size, cap_tokens)) {
            free(s);
            return 1;
        }
        qwen35_graph_reset(&s->qwen4_graph);
        s->qwen35_graph_ready = true;
        s->prefill_cap = (uint32_t)ctx_size;
        s->logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
        s->sample_probs = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->sample_probs[0]));
        *out = s;
        return 0;
    }
#endif
    ds4_qwen35_not_reached("session create");
```

In `ds4_session_free` (`ds4.c:74469`), ahead of `#ifdef DS4_HAS_QWEN4_GPU` / `if (ds4_session_is_qwen4(s)) {`:

```c
#ifdef DS4_HAS_QWEN4_METAL
        if (s->qwen35_graph_ready) {
            qwen4_graph_free(&s->qwen4_graph);
        } else
#endif
```

- [ ] **Step 5: Sync and eval**

In `ds4_session_sync_internal`, directly before `#ifdef DS4_HAS_QWEN4_GPU` / `if (ds4_session_is_qwen4(s)) {` (`ds4.c:76617`):

```c
#ifdef DS4_HAS_QWEN4_METAL
    if (ds4_session_is_qwen35(s)) {
        ds4_engine *e = s->engine;
        ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
        int start = 0;
        if (s->checkpoint_valid && g->pos == (uint32_t)s->checkpoint.len &&
            prompt->len >= s->checkpoint.len && ds4_tokens_starts_with(prompt, &s->checkpoint)) {
            start = s->checkpoint.len;
        } else {
            /* A different prompt, a rewind or a failed forward: the recurrent
             * state cannot be trimmed, so replay from the start. */
            qwen35_graph_reset(g);
            s->checkpoint.len = 0;
            s->checkpoint_valid = false;
        }
        if ((uint32_t)prompt->len > g->ctx_cap) {
            snprintf(err, errlen, "prompt of %d tokens exceeds the %u-token context", prompt->len, g->ctx_cap);
            return 1;
        }
        for (int i = start; i < prompt->len; i++) {
            if (prompt->v[i] < 0 || prompt->v[i] >= (int)DS4_N_VOCAB) {
                snprintf(err, errlen, "token id %d at position %d is outside the vocabulary", prompt->v[i], i);
                return 1;
            }
        }
        for (int i = start; i < prompt->len;) {
            if (ds4_session_cancelled(s)) {
                snprintf(err, errlen, "interrupted");
                s->checkpoint_valid = s->checkpoint.len > 0;
                return DS4_SESSION_SYNC_INTERRUPTED;
            }
            uint32_t chunk = (uint32_t)(prompt->len - i);
            if (chunk > g->cap_tokens) chunk = g->cap_tokens;
            s->checkpoint_valid = false;
            if (!qwen35_graph_forward_tokens(g, &e->model, &e->weights, prompt->v + i, chunk, s->logits)) {
                snprintf(err, errlen, "Ornith prefill failed at token %d", i);
                return 1;
            }
            for (uint32_t j = 0; j < chunk; j++) token_vec_push(&s->checkpoint, prompt->v[i + (int)j]);
            i += (int)chunk;
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
            if (s->progress) s->progress(s->progress_ud, "prefill_chunk", i, prompt->len);
        }
        s->checkpoint_valid = true;
        return 0;
    }
#endif
```

In `ds4_session_eval_internal`, directly before `#ifdef DS4_HAS_QWEN4_GPU` / `if (ds4_session_is_qwen4(s)) {` (`ds4.c:78691`):

```c
#ifdef DS4_HAS_QWEN4_METAL
    if (ds4_session_is_qwen35(s)) {
        ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
        if (!s->qwen35_graph_ready || !s->checkpoint_valid || g->pos != (uint32_t)s->checkpoint.len) {
            if (errlen) snprintf(err, errlen, "Ornith session is not synchronized");
            return 1;
        }
        if (g->pos >= g->ctx_cap) {
            if (errlen) snprintf(err, errlen, "context is full");
            return 1;
        }
        if (!qwen35_graph_forward_tokens(g, &e->model, &e->weights, &token, 1, s->logits)) {
            if (errlen) snprintf(err, errlen, "Ornith decode failed at position %u", g->pos);
            s->checkpoint_valid = false;
            return 1;
        }
        token_vec_push(&s->checkpoint, token);
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        return 0;
    }
#endif
```

Add `ds4_qwen35_not_reached("session sync")` and `ds4_qwen35_not_reached("session eval")` at the start of the DeepSeek default path in each function, directly after the last family branch (before the `ds4_session_is_cpu(s)` / `ds4_session_is_glm(s)` blocks that follow).

- [ ] **Step 6: Run the session test and the logprob dump smoke**

```bash
make ds4 test-qwen35-session DS4_ORNITH_MODEL=$DS4_ORNITH_MODEL
printf 'The capital of France is' > /tmp/ornith-cap.txt
./ds4 -m "$DS4_ORNITH_MODEL" --metal --raw --prompt-file /tmp/ornith-cap.txt -c 4096 -n 8 \
    --dump-logprobs /tmp/ornith-cap.json && python3 -c "import json; d=json.load(open('/tmp/ornith-cap.json')); print(len(d['steps']), d['steps'][0]['selected'])"
./ds4 -m "$DS4_ORNITH_MODEL" --metal --raw --prompt-file /tmp/ornith-cap.txt -c 16 -n 200 --temp 0 > /dev/null && echo "small ctx: clean stop"
```

Expected: the test prints eight `extend` lines with the replay difference under 1e-3, then `divergent prompt: bit-identical`, `context full: 'context is full'` and `qwen35 session: ok`; the dump has 8 steps; the last command prints `small ctx: clean stop`. Record the one-pass differences in the commit message. If one is large, it is a near tie; Task 8 checks correctness against llama.cpp.

- [ ] **Step 7: Build all targets and the Qwen3.8 kernel tests**

```bash
make ds4 ds4-server ds4-bench ds4-eval ds4-agent && make cpu && make test-qwen4-kernels test-qwen4-q2
```

- [ ] **Step 8: Commit**

```bash
git add ds4.c Makefile tests/test_qwen35_session.c
git commit -m "qwen35: Ornith sessions (create, sync with prefix reuse, eval, free)

Sessions reuse the qwen4 graph struct under their own ready flag; a
divergent prompt or failed forward replays from the start, and DeepSeek
fallbacks refuse the family explicitly.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- ds4.c Makefile tests/test_qwen35_session.c
```

---

### Task 8: Gate 1 correctness run against llama.cpp

**Files:**
- Create: `tests/ornith/gate1.py`, `speed-bench/ornith/m1/RESULTS.md`, `speed-bench/ornith/m1/compare-*.txt`

**Interfaces:**
- Consumes: `tests/ornith/ref/*.json`, `tests/ornith/tolerance.json`, `ornith_ref.compare`, the built `./ds4`.
- Produces: `gate1.py [--ds4-arg ARG ...] OUT_DIR`. It exits 0 only when every prompt passes and prints one line per prompt.

- [ ] **Step 1: Write `tests/ornith/gate1.py`**

```python
#!/usr/bin/env python3
"""M1 gate 1: ds4 against the llama.cpp references for every prompt.

  gate1.py [--ds4-arg ARG ...] OUT_DIR

For each prompt in tests/ornith/prompts.json: write the raw text, check that
ds4's prompt tokens equal llama.cpp's, run `ds4 --dump-logprobs` greedily for
the reference's step count, and compare with tests/ornith/tolerance.json.
Extra ds4 arguments (for example --prefill-chunk 64) are passed through.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import ornith_ref as r

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main(argv):
    extra = []
    while len(argv) >= 2 and argv[0] == "--ds4-arg":
        extra.append(argv[1])
        argv = argv[2:]
    if len(argv) != 1:
        sys.exit(__doc__)
    out = argv[0]
    model = os.environ["DS4_ORNITH_MODEL"]
    os.makedirs(os.path.join(out, "ds4"), exist_ok=True)
    tol = json.load(open(os.path.join(ROOT, "tests/ornith/tolerance.json")))
    failed = 0
    for p in r.load_prompts(os.path.join(ROOT, "tests/ornith/prompts.json")):
        ref = json.load(open(os.path.join(ROOT, "tests/ornith/ref", p["name"] + ".json")))
        txt = os.path.join(out, "ds4", p["name"] + ".txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write(r.prompt_text(p, ROOT))
        ids_line = subprocess.run([os.path.join(ROOT, "ds4"), "-m", model, "--raw", "--prompt-file", txt,
                                   "--dump-tokens"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
        if json.loads(ids_line) != ref["prompt_ids"]:
            print(f"{p['name']}: FAIL tokenization differs")
            failed += 1
            continue
        dump = os.path.join(out, "ds4", p["name"] + ".json")
        log = open(os.path.join(out, "ds4", p["name"] + ".log"), "w")
        cmd = [os.path.join(ROOT, "ds4"), "-m", model, "--metal", "--raw", "--prompt-file", txt,
               "-c", "16384", "-n", str(len(ref["steps"])), "--dump-logprobs", dump] + extra
        if subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode != 0:
            print(f"{p['name']}: FAIL ds4 exited non-zero (see {log.name})")
            failed += 1
            continue
        res = r.compare(ref["steps"], r.ds4_steps(json.load(open(dump))), tol["tol"], tol["tie"])
        status = "ok" if res["ok"] else "FAIL"
        failed += 0 if res["ok"] else 1
        print(f"{p['name']}: {status} compared={res['compared']} tie_at={res['stopped_at_tie']} "
              f"max_delta={res['max_delta']:.4f} {res['reason']}")
    print(f"gate1: {'PASS' if failed == 0 else f'FAIL ({failed})'} tol={tol['tol']:.4f} tie={tol['tie']:.4f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 2: Run the unit tests for the shared module again**

Run: `python3 -m unittest tests/ornith/test_ornith_ref.py -v`
Expected: `OK`.

- [ ] **Step 3: Run gate 1 with the default prefill chunk**

```bash
caffeinate -i -s python3 tests/ornith/gate1.py speed-bench/ornith/m1 | tee speed-bench/ornith/m1/compare-default.txt
```

Expected: 12 `ok` lines and `gate1: PASS`. On any FAIL, stop and use `superpowers:systematic-debugging`:
1. Localize the first bad step.
2. Compare the same prompt with `--prefill-chunk 1` (pure decode) to split prefill kernels from decode kernels.
3. Compare per-layer hidden states against `llama-eval-callback` for that prompt.

Fix, re-run Tasks 3-7 tests, then rerun this step. Do not widen `tolerance.json` to make a failure pass.

- [ ] **Step 4: Review Focus 2 (tile-GEMM threshold) runs**

```bash
caffeinate -i -s python3 tests/ornith/gate1.py --ds4-arg --prefill-chunk --ds4-arg 64 speed-bench/ornith/m1-chunk64 | tee speed-bench/ornith/m1/compare-chunk64.txt
caffeinate -i -s python3 tests/ornith/gate1.py --ds4-arg --prefill-chunk --ds4-arg 65 speed-bench/ornith/m1-chunk65 | tee speed-bench/ornith/m1/compare-chunk65.txt
```

Expected: `gate1: PASS` for both (per-token Q4_K kernels at 64 rows, tile GEMM at 65).

- [ ] **Step 5: Write the receipt and commit**

Create `speed-bench/ornith/m1/RESULTS.md`:

```markdown
# Ornith M1 gate 1 (plain inference against llama.cpp)

- Branch commit: <git rev-parse --short HEAD>
- Model: 23G ICE (`speed-bench/ornith/oracle/RESULTS.md`), llama.cpp oracle `tests/ornith/ref/ORACLE.txt`.
- Tolerances (`tests/ornith/tolerance.json`): tol <x>, tie <y> (from llama.cpp Metal vs CPU).
- Kernel tests: `make test-qwen35-kernels` ok; session test: `make test-qwen35-session` ok.

| run | result | notes |
|---|---|---|
| default prefill chunk | <PASS/FAIL> | `compare-default.txt` |
| --prefill-chunk 64 (per-token Q4_K) | <PASS/FAIL> | `compare-chunk64.txt` |
| --prefill-chunk 65 (tile GEMM Q4_K) | <PASS/FAIL> | `compare-chunk65.txt` |

Per-prompt compared steps and max delta: see the compare files.
Decode speed is not a gate in M1 (M4); for reference, the one-shot log line on `en_capital`: <paste>.
```

```bash
chmod +x tests/ornith/gate1.py
git add tests/ornith/gate1.py speed-bench/ornith/m1/RESULTS.md speed-bench/ornith/m1/compare-*.txt
git commit -m "tests/ornith: gate 1 runner and M1 correctness receipt

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- tests/ornith/gate1.py speed-bench/ornith/m1/RESULTS.md speed-bench/ornith/m1/compare-default.txt speed-bench/ornith/m1/compare-chunk64.txt speed-bench/ornith/m1/compare-chunk65.txt
```

---

### Task 9: Qwen3.8 regression gate and M1 close-out

**Files:**
- Create: `speed-bench/ornith/m1/QWEN_GATE.md`
- Modify: memory `ornith-qwen35moe-port-research.md` (outside the repo)

- [ ] **Step 1: Agree the run window with the user**

The gate needs the machine free: stop the gateway's ds4 and oMLX backends, and no V4.1 session may be running a model. Ask the user and wait for the go-ahead. The Qwen gate README requires this.

- [ ] **Step 2: Run the full Qwen3.8 gate**

```bash
ps -axo pid,stat,comm | awk '$2 ~ /^(E|U)/'
caffeinate -i -s speed-bench/qwen-regression/run.sh full 2>&1 | tee /tmp/qwen-gate-full.log | tail -30
```

Expected: the fast checks (kernel tests, byte-identical vi/code replies to `baseline/`, registry unchanged) pass, and the full checks (paired decode ≥ 97 % of PROD, steady wired ≤ baseline + 0.5 GiB, needle found) pass. A paired-speed failure alone is rerun once (README). Any other failure: stop, bisect over this branch's commits (Tasks 3-7 touch shared files), fix, rerun.

- [ ] **Step 3: Write the receipt and commit**

Create `speed-bench/ornith/m1/QWEN_GATE.md`:

```markdown
# Qwen3.8 regression gate after Ornith M1

- Branch commit: <git rev-parse --short HEAD>
- Command: `speed-bench/qwen-regression/run.sh full`
- Result: <PASS/FAIL>; replies byte-identical <yes/no>; paired decode <branch/PROD %>; wired <GiB>; needle <HIT/MISS>.
- Log tail:

<paste the last lines of /tmp/qwen-gate-full.log>
```

```bash
git add speed-bench/ornith/m1/QWEN_GATE.md
git commit -m "speed-bench/ornith: Qwen3.8 regression gate after M1

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- speed-bench/ornith/m1/QWEN_GATE.md
```

- [ ] **Step 4: Restore services and update memory**

Restart the gateway backends that Step 1 stopped (per memory `prod-ai-gateway-stop-start`). Update the memory file `ornith-qwen35moe-port-research.md`:
- M1 done: commit, gate 1 result, Qwen gate result.
- The one-shot decode t/s line as a first speed hint.
- Next: M2 plan (MTP).

- [ ] **Step 5: Hand over for review**

M1 is complete when Tasks 1-9 are checked. The next step is `superpowers:requesting-code-review` on the branch diff against `origin/develop`, then the M2 plan.
