# Ornith qwen35moe M3: Serving — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve Ornith-1.5-35B-A3B through `ds4-server` and `ds4-agent`: prompts rendered exactly like the GGUF's embedded chat template (gate 1 item 5), disk KV checkpoints that restore and continue like an uninterrupted run with and without `--mtp` (gate 1 item 4), MTP rewinds that keep the session, Ornith model ids, and no M1 refusals, while Qwen3.8 stays byte-identical.

**Architecture:** Rendering keeps `SERVER_MODEL_SYNTAX_QWEN` (parser, XML tool calls, live tails) and adds an Ornith render flavor chosen once at server start: new `render_ornith_*` functions produce the template's system turn (terse block, tool instructions, effort default) and conversation body (trimmed tool results with the tool-error warning, trimmed history, `preserve_thinking`, in-place later system messages), checked byte for byte against jinja2 goldens rendered from the GGUF's own template. The engine-side renderers (CLI, agent) switch to the shared Qwen text predicate and get an Ornith system turn. Disk KV gets an Ornith payload (`DS4_QWEN35_PAYLOAD_TAG`) that carries GDN state, trunk and MTP KV rows and the MTP hidden-state carry; rewind restores M2's after-row-0 verify snapshot. Every Ornith branch sits beside the Qwen3.8 one and is off for Qwen3.8 sessions and servers.

**Tech Stack:** C (`ds4.c`, `ds4_qwen35moe.inc`, `ds4_server.c`, `ds4_agent.c`, `tests/ds4_test.c`), Python 3 stdlib (test scripts; system `python3` 3.9), jinja2 3.1 from `~/.local/ai-gateway-env/bin/python3` (golden generation only), POSIX shell.

**Spec:** `docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md` (milestone M3 of §9). Binding decisions: the controller's `DECISIONS.md` (§1 Global Constraints, §2 M2 interfaces, §3 M3 decisions). M2 plan: `docs/superpowers/plans/2026-09-26-ornith-m2-mtp.md` (lands before M3).

## Global Constraints

- The Qwen3.8 production path stays byte-identical and as fast as today (spec §1, §7). Every commit that touches a shared file (`ds4.c` outside Ornith-only functions, `ds4_metal.m`, `ds4_gpu.h`, `metal/*.metal`, `ds4_server.c`, `ds4_agent.c`, `ds4_kvstore.c`) passes `make test-qwen4-kernels test-qwen4-q2` and `speed-bench/qwen-regression/run.sh fast`; the branch passes `run.sh full` before merge.
- Metal only for Ornith; refusals from M1 stay unless the plan lifts one explicitly.
- Ornith knobs use the `DS4_QWEN35_*` prefix; Ornith code reads no family-level `DS4_QWEN4_*` knob (spec §7.3; the kernel A/B switches inside shared qwen4 helpers are the documented exception).
- Kernel changes are additive: new kernels/entry points in `metal/qwen35.metal`; `metal/qwen4.metal` is not edited.
- Code, comments, docs and commit messages in English. Model files never go into git. The 23G GGUF: `DS4_ORNITH_MODEL=$HOME/orca/workspaces/ds4-metal-data/gguf/ornith/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf`. Every model test reads `DS4_ORNITH_MODEL` and fails with a message when it is unset.
- One model process at a time on this 64 GB machine. Never `kill -9` a Metal process. Long runs under `caffeinate -i -s`; monitors poll process liveness, not only a log pattern.
- GPU windows: pausing the live stack needs the user's OK once per execution window. Pause = `launchctl unload ~/Library/LaunchAgents/dev.dongnh.gateway-watchdog.plist ~/Library/LaunchAgents/dev.dongnh.gateway-eval.plist ~/Library/LaunchAgents/dev.dongnh.ai-proxy.plist`, then `kill -TERM $(cat ~/.local/share/ai-gateway/omlx.pid)`; restore = `launchctl load` of the three plists, then wait for `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8090/status` = 200.
- Never add a staging entry to the live gateway registry; never run AI-Gateway `run_ab.py`/`run_ab.sh` as they are (they rewrite live aliases and hot-swap oMLX). Staging ds4-server ports: 18296 (Ornith staging), 18190/18191 stay the oracle/loader test ports.
- No C++. Follow AGENT.md: small readable code; comments explain why.
- Every commit ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2
  ```

Plan-specific constraints:

- Branch `feature/ornith-m3`, cut from `develop` after the M2 merge. Merging or pushing needs the user's explicit OK.
- M2 interfaces are consumed as DECISIONS §2 lists them, with one controller change: M2 does not define `qwen35_graph_h_last` / `qwen35_graph_set_h_last`; this plan defines them (Task 5).
- Existing unit tests are never edited. `./ds4_test --server` and `./ds4_agent_test` pass unchanged after every task that touches `ds4_server.c` or `ds4_agent.c`. With the Qwen gate they are the proof that Qwen3.8 rendering did not change: `test_render_qwen_chat_prompt_text`, `test_qwen_reasoning_effort_levels`, `test_render_qwen_tool_round_trip`, `test_qwen_tool_visible_checkpoint_boundary`, `test_qwen_checkpoint_suffix_matches_render`, `test_qwen_thinking_visible_text_matches_render`, `test_qwen_plain_answer_keeps_trailing_whitespace`, `test_qwen_sampled_tool_text_after_think_renders_exactly`, `test_parse_qwen_tool_call_message`, `test_reasoning_effort_mapping`, `test_api_thinking_controls_parse`, `test_model_alias_thinking_controls` (server) and `test_qwen_tool_syntax`, `test_tool_contracts` (agent).
- The golden generator runs only with `~/.local/ai-gateway-env/bin/python3` (it has jinja2). Never `pip install` into `~/.local/omlx-venv`.
- `--mtp-exact-sampling` stays refused for Ornith (DECISIONS §2).
- Live tests: port 18296; the loader test keeps 18191. Servers are stopped with SIGTERM and waited for; a server that ignores SIGTERM is reported, never killed.

## Deviations from the spec wording, decided while planning

1. **Effort text (spec §5).** The spec says `ds4_qwen4_reasoning_effort_text()` gains a family check and returns NULL for Ornith. The embedded template emits the same xhigh and low lines as Qwen3.8 for explicit efforts; only its default (medium) emits nothing. `ds4_qwen4_reasoning_effort_text()` stays unchanged. The server makes an absent or null effort MEDIUM for Ornith (the parser default, Task 3). The engine-side renderers use a new `ds4_engine_reasoning_effort_text()`: the frontends' default `DS4_THINK_HIGH` renders Ornith's medium, `--think-max` the xhigh line, low the low line (Task 2). Task 4 edits spec §5.
2. **Ornith template differences (spec §5).** "Thinking off rendered as `<think>\n\n</think>\n\n`" is already what ds4 renders; it is not a difference. The list misses the terse block, the tool instruction text, `tojson` spacing, leading-only system merging, tool-result trimming, the tool-error warning, assistant-content trimming and `preserve_thinking`. The server gets an Ornith flavor (DECISIONS §3.1-3.2). Task 4 edits spec §5.
3. **Template branches not rendered like the template, by decision:** `<|think_*|>` tags inside messages (passed through as text), unknown roles (dropped, as for Qwen3.8), `max_tool_arg_chars`/`max_tool_response_chars`/`suppress_tool_instructions`/`auto_disable_thinking_with_tools` (ignored), `tool_call_format` json (HTTP 400), think tags inside assistant `content` (copied verbatim as for Qwen3.8), the `thinking`/`reasoning` history fields (only `reasoning_content` is parsed, as today), closing-sentinel escaping in tool bodies and parameters (kept from Qwen3.8), sampled tool-text replay (kept, for live-KV alignment), a generation prompt only when an assistant turn is pending (ds4's prefill semantics), ASCII-only trimming and lowercasing, and float spellings copied as written by `tojson` (`1e5` stays `1e5`; Python would print `100000.0`). Anthropic tools keep ds4's existing mapping, `{"type": "function", "function": <tool as given>}` (`input_schema` is not renamed to `parameters`). Task 4 records them in spec §5.
4. **Agent and CLI rendering (DECISIONS §3.4).** The agent keeps the Qwen3.8 tools prompt and gets no terse block; the CLI's `encode_chat_prompt` path gets the terse block, its `ds4_chat_append_message` path does not. Task 4 records it in spec §5.
5. **`ds4_think_mode_for_context`** no longer turns `DS4_THINK_MAX` into `DS4_THINK_HIGH` for Ornith: the clamp exists for DeepSeek's max-effort prefix, and Ornith's `--think-max` must reach the xhigh line (Task 2).
6. **No silent fallthrough (spec §3)** lists context estimate, generate, session create/sync/eval, speculative cycle and rewind. Disk KV payload size/save/load are added (they fall into DeepSeek code today); rewind needs no die path because an Ornith rewind restores or invalidates. Task 5 edits spec §3.
7. **Disk KV (spec §5).** The payload also records MTP presence and carries the MTP hidden-state carry and the rope positions; a payload whose MTP presence differs from the session's is refused and the server prefills. The KV-cache file's model id (7) already separates the families. `ds4_engine_routed_quant_bits` stays 2 for both Ornith tiers (Q5_K in layer 0), so a 25G server accepts a 23G checkpoint of the same text, as the Qwen3.8 IQ2 tiers do today. Task 5 edits spec §5.
8. **Rewind (spec §6).** "Otherwise it resets the recurrent state and conv history together with the checkpoint": the plan invalidates the checkpoint and the next sync resets and replays (same effect, no eager reset). Task 6 edits spec §6.
9. **Gate 1 item 4, "the other way round" (spec §8).** The Ornith side is tested with a Qwen3.8-tagged payload on a real Ornith session. The Qwen3.8 side rests on the Qwen3.8 loader's exact tag check (`h[12]`) and on the KV-cache model id, which a model-free unit test pins; M3 runs no Qwen3.8 model test for it. Task 5 edits spec §8.
10. **Server aliases (spec §5).** `-no-think` is accepted besides `-nothink`; `/v1/models` lists the base id, `-chat` and `-reasoner`, as for Qwen3.8. Task 7 edits spec §5.
11. **Anthropic system prompt position.** ds4 appends the Anthropic `system` field as the last message (only V4.1 moves it first). The Ornith flavor merges only leading system messages, so for Ornith it moves first too (Task 3).

## Review Focus

1. **The gateway's think-cap replay flips `enable_thinking` to false** (and drops `preserve_thinking`) on a conversation with tools and historical `reasoning_content`. Expected: the think-off terse lead and think-off tool instructions, historical think blocks kept (Ornith's `preserve_thinking` defaults to true, unlike Qwen3.6), the closed generation prompt, and an answer with no reasoning. Pinned by golden `think_cap_replay` (Task 4, `make test-ornith-render`) and part 5 of `tests/ornith/test_server_live.py` (Task 8).
2. **Tool error outputs** (a JSON error body, a shell failure, a traceback, two failures across an assistant retry, and a failure answered through the live tool tail). Expected: the template's exact warning text inside the `<tool_response>`, the count continuing across the assistant retry, and a live tail that is byte for byte the suffix of the full render. Pinned by goldens `tool_error`, `tool_errors_consecutive` and `test_ornith_tool_error_heuristic`, `test_ornith_live_tail_continues_full_render` (Task 4).
3. **Anthropic `/v1/messages` tool_result turns**, including a `tool_result` block and a trailing text block in one user message and the top-level `system`. Expected: the same bytes as the OpenAI-shaped conversation (system first, tool message, then a user turn). Pinned by golden `anthropic_tool_result` and `test_ornith_anthropic_tool_results_match_openai` (Task 4), and part 4 of `test_server_live.py` (Task 8).
4. **Disk KV restore across a server restart with `--mtp`, and a restart without `--mtp` on the same cache.** Expected: the restored checkpoint continues with >3000 cached tokens and correct answers; a checkpoint whose MTP presence differs is refused with a clear log line and the prompt is prefilled. Pinned by `ds4_test --session-snapshot --qwen35-payloads` with and without `DS4_TEST_GLM_MTP=1` (Task 5) and `tests/ornith/test_server_kv.py` (Task 8).
5. **A reasoning effort sent vs absent** through OpenAI `reasoning_effort`, `chat_template_kwargs.reasoning_effort`, Anthropic `output_config.effort`, Responses `reasoning.effort`, `null`, `none`, `off`, `max`. Expected: absent/null/medium = no line, high/xhigh/max/extreme = the xhigh line, low/minimal = the low line, none/off = thinking off; Qwen3.8 keeps xhigh for an absent effort and rejects `off`. Pinned by `test_ornith_effort_sent_vs_absent` and the `effort_*` goldens (Task 3).

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `tests/ornith/chat/render_golden.py` | Create | Stdlib GGUF template reader, golden cases, jinja2 render in the transformers environment, writer and `--check` |
| `tests/ornith/chat/test_render_golden.py` | Create | Stdlib unit tests for the reader, the request-to-template mapping and the case list |
| `tests/ornith/chat/golden/<case>.txt`, `<case>.input.json`, `TEMPLATE.txt` | Generate + commit | jinja2 renders, the request body ds4 parses, the template inputs, template hash |
| `tests/ornith/chat/check_render.py` | Create | Runs `ds4_test --qwen35-render` per case and diffs against the golden (no jinja2, no model) |
| `tests/ornith/test_cli_chat.sh` | Create | CLI chat tokens (`./ds4 --dump-tokens`) equal the golden text's tokens |
| `ds4.h` | Modify | `ds4_engine_uses_qwen35_text`, `ds4_engine_reasoning_effort_text`, `ds4_qwen35_terse_text` |
| `ds4.c` | Modify | Engine chat sites, think prefix, think-mode clamp, Ornith payload bytes/save/load + guards, rewind branch, batch guard |
| `ds4_qwen35moe.inc` | Modify | `qwen35_graph_h_last`, `qwen35_graph_set_h_last` |
| `ds4_server.c` | Modify | Ornith flavor, template kwargs, effort default, Ornith renderer and live tail, ids and aliases, startup flavor, refusal removal, unit tests |
| `ds4_agent.c` | Modify | Tool syntax and system tokens through the shared predicate, refusal removal |
| `tests/ds4_test.c` | Modify | `--qwen35-render` harness, `--qwen35-payloads`, `--qwen35-rewind` |
| `tests/test_qwen35_session.c` | Modify | Case 5: a decode batch of two Ornith sessions takes the serialized path |
| `tests/ornith/test_loader.sh` | Modify | Server and agent serve Ornith instead of refusing it |
| `tests/ornith/serverlib.py`, `tests/ornith/test_server_kv.py`, `tests/ornith/test_server_live.py` | Create | Live staging-server gate |
| `Makefile` | Modify | `test-ornith-render` |
| `docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md` | Modify | Deviations 1-10 |
| `speed-bench/ornith/m3/RESULTS.md`, `speed-bench/ornith/m3/QWEN_GATE.md` | Create | Receipts |

---

### Task 1: Branch, M2 interface check, golden render generator and fixtures

No production code. The deliverable is the committed golden set that gate 1 item 5 compares against, rendered from the template stored in the GGUF.

**Files:**
- Create: `tests/ornith/chat/render_golden.py`, `tests/ornith/chat/test_render_golden.py`
- Generate + commit: `tests/ornith/chat/golden/*.txt`, `tests/ornith/chat/golden/*.input.json`, `tests/ornith/chat/golden/TEMPLATE.txt`

**Interfaces:**
- Consumes (M2, verified in Step 2): `qwen35_graph_alloc(g, ctx_cap, cap_tokens, bool mtp)`, `qwen35_graph_reset(g)`, `qwen35_graph_state_swap(g)`, `qwen35_graph_mtp(...)`, struct fields `mtp_h`, `mtp_h_pos0`, `mtp_h_rows` (plus the existing `mtp_pos`, `snap_valid`, `snap_pos`), `ds4_session_qwen35_spec_cycle(...)`, session fields `glm_mtp_have`, `glm_mtp_have2`, `qwen4_verify_logits`, knob `DS4_QWEN35_SPEC_FORCE_ACCEPT`, `ds4_engine_mtp_draft_tokens()` = 2 for Ornith with `--mtp`.
- Produces (module `tests/ornith/chat/render_golden.py`):
  - `read_chat_template(path: str) -> str`
  - `template_view(body: dict) -> dict` with keys `messages`, `tools`, `kwargs`
  - `case_spec(case: dict) -> dict` with keys `api` (`"chat"` or `"anthropic"`), `body`, `template`
  - `CASES: dict[str, dict]`
- Produces (files): `golden/<case>.input.json` = `{"api", "body", "template"}`; `golden/<case>.txt` = the jinja2 render; `golden/TEMPLATE.txt`.

- [ ] **Step 1: Cut the branch**

```bash
cd /path/to/execution-worktree            # created with superpowers:using-git-worktrees
git log --oneline -20 develop | grep -i 'ornith.*m2\|feature/ornith-m2'
git switch -c feature/ornith-m3 develop
git log --oneline -1
```

Expected: the first command prints the M2 merge commit. If it prints nothing, M2 has not landed: stop and report to the controller.

- [ ] **Step 2: Re-verify every consumed M2 symbol**

```bash
for fn in qwen35_graph_alloc qwen35_graph_reset qwen35_graph_state_swap qwen35_graph_mtp; do
  grep -n -A2 "^static [a-z]* $fn(" ds4_qwen35moe.inc
done
grep -n -A3 '^static int ds4_session_qwen35_spec_cycle(' ds4.c
grep -n 'ds4_gpu_tensor \*mtp_h;\|uint32_t mtp_h_pos0;\|uint32_t mtp_h_rows;' ds4.c
grep -n 'DS4_QWEN35_SPEC_FORCE_ACCEPT' ds4.c ds4_qwen35moe.inc
grep -n 'qwen35_graph_alloc(&s->qwen4_graph' ds4.c
grep -n 'qwen4_verify_logits' ds4.c | head -40
grep -c 'qwen35_graph_h_last\|qwen35_graph_set_h_last' ds4.c ds4_qwen35moe.inc
grep -n 'ds4_qwen35_not_reached("' ds4.c
```

Expected:
- the loop prints the four graph functions with the parameter lists of DECISIONS §2 (`qwen35_graph_alloc(g, uint32_t ctx_cap, uint32_t cap_tokens, bool mtp)`, `qwen35_graph_reset(g)`, `qwen35_graph_state_swap(g)` returning `bool`, `qwen35_graph_mtp(g, m, w, const int *tokens, uint32_t T, ...)`);
- `ds4_session_qwen35_spec_cycle` has the qwen4 cycle's parameter list;
- one match each for the three `mtp_h` fields and for the force-accept knob;
- the Ornith session create passes the engine's MTP flag as the fourth `qwen35_graph_alloc` argument;
- `ds4_session_qwen35_spec_cycle` allocates `qwen4_verify_logits` (4 x vocab floats, rows 0 and 1 = the verify rows) before its first verify, and the Ornith free path releases it; the rewind in Task 6 tests the pointer before use;
- `0` for both files in the `grep -c` line (M3 defines these two functions in Task 5);
- the M1 guards plus M2's in the two speculative implementations.

Also read the Ornith branch of `ds4_session_sync_internal` (`grep -n 'ds4_session_is_qwen35(s)' ds4.c`, the first match): the MTP catch-up for a chunk (`qwen35_graph_mtp` with `want_draft` false) must run before the `progress(..., "prefill_chunk", ...)` call, so a continued checkpoint saved from the progress callback carries MTP rows up to its position (Task 5 asserts it).

If any signature, field name or knob differs, stop: report the difference to the controller for a ledger ruling. Do not adapt later tasks on your own.

- [ ] **Step 3: Write the failing generator unit tests**

Create `tests/ornith/chat/test_render_golden.py`:

```python
"""Unit tests for the Ornith golden generator (python3 -m unittest; no jinja2)."""
import io
import json
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import render_golden as rg


def gguf_bytes(kvs):
    """A header-only GGUF v3 file: no tensors, the given (key, type, value) entries."""
    out = io.BytesIO()
    out.write(b"GGUF")
    out.write(struct.pack("<I", 3))
    out.write(struct.pack("<QQ", 0, len(kvs)))
    for key, vtype, value in kvs:
        kb = key.encode("utf-8")
        out.write(struct.pack("<Q", len(kb)))
        out.write(kb)
        out.write(struct.pack("<I", vtype))
        if vtype == 8:
            vb = value.encode("utf-8")
            out.write(struct.pack("<Q", len(vb)))
            out.write(vb)
        elif vtype == 4:
            out.write(struct.pack("<I", value))
        elif vtype == 9:
            etype, items = value
            out.write(struct.pack("<IQ", etype, len(items)))
            for item in items:
                if etype == 8:
                    ib = item.encode("utf-8")
                    out.write(struct.pack("<Q", len(ib)))
                    out.write(ib)
                else:
                    out.write(struct.pack("<i", item))
    return out.getvalue()


class ReaderTest(unittest.TestCase):
    def write(self, data):
        fd, path = tempfile.mkstemp(suffix=".gguf")
        os.write(fd, data)
        os.close(fd)
        self.addCleanup(os.unlink, path)
        return path

    def test_finds_template_after_arrays(self):
        path = self.write(gguf_bytes([
            ("general.architecture", 8, "qwen35moe"),
            ("tokenizer.ggml.tokens", 9, (8, ["a", "bb", "ccc"])),
            ("tokenizer.ggml.token_type", 9, (5, [1, 3, 4])),
            ("x.count", 4, 7),
            ("tokenizer.chat_template", 8, "{{ messages }}"),
        ]))
        self.assertEqual(rg.read_chat_template(path), "{{ messages }}")

    def test_missing_template_fails(self):
        path = self.write(gguf_bytes([("x.count", 4, 7)]))
        with self.assertRaises(SystemExit):
            rg.read_chat_template(path)

    def test_not_a_gguf_fails(self):
        path = self.write(b"NOPE" + b"\0" * 28)
        with self.assertRaises(SystemExit):
            rg.read_chat_template(path)


class TemplateViewTest(unittest.TestCase):
    def test_string_arguments_become_objects_in_order(self):
        body = {"messages": [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "f", "arguments": "{\"b\": 1, \"a\": [2]}"}}]}]}
        view = rg.template_view(body)
        args = view["messages"][0]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(args, {"b": 1, "a": [2]})
        self.assertEqual(list(args), ["b", "a"])
        self.assertIsInstance(body["messages"][0]["tool_calls"][0]["function"]["arguments"], str)

    def test_non_object_arguments_stay_strings(self):
        body = {"messages": [{"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "f", "arguments": "not json"}},
            {"function": {"name": "g", "arguments": "[1, 2]"}}]}]}
        calls = rg.template_view(body)["messages"][0]["tool_calls"]
        self.assertEqual(calls[0]["function"]["arguments"], "not json")
        self.assertEqual(calls[1]["function"]["arguments"], "[1, 2]")

    def test_effort_and_kwargs(self):
        view = rg.template_view({"messages": [], "reasoning_effort": "high",
                                 "chat_template_kwargs": {"terse": False}})
        self.assertEqual(view["kwargs"], {"reasoning_effort": "high", "terse": False})
        self.assertIsNone(view["tools"])


class CasesTest(unittest.TestCase):
    def test_cases_cover_the_decided_set(self):
        required = {
            "chat_system_user", "chat_think_off", "chat_no_system", "chat_terse_off",
            "developer_head_merge", "effort_none", "effort_medium", "effort_high", "effort_low",
            "effort_max", "effort_low_think_off", "tools", "tools_think_off", "tools_unicode",
            "tool_round_trip", "tool_error", "tool_errors_consecutive", "tool_output_trailing_nl",
            "tool_results_grouped", "history_reasoning", "preserve_thinking_off",
            "preserve_thinking_off_tool_loop", "assistant_trailing_ws", "midconv_system",
            "think_cap_replay", "anthropic_tool_result",
        }
        self.assertEqual(required - set(rg.CASES), set())

    def test_case_specs_are_complete(self):
        for name, case in rg.CASES.items():
            spec = rg.case_spec(case)
            self.assertIn(spec["api"], ("chat", "anthropic"), name)
            self.assertIn("messages", spec["template"], name)
            json.dumps(spec, ensure_ascii=False)
            if spec["api"] == "anthropic":
                self.assertIn("template", case, name)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: Run the tests and confirm they fail**

Run: `python3 -m unittest tests/ornith/chat/test_render_golden.py -v`
Expected: `ModuleNotFoundError: No module named 'render_golden'`.

- [ ] **Step 5: Write `tests/ornith/chat/render_golden.py`**

```python
#!/usr/bin/env python3
"""Golden renders of Ornith's embedded chat template (gate 1 item 5).

  render_golden.py [--check]

Reads tokenizer.chat_template from $DS4_ORNITH_MODEL (GGUF header only, stdlib
reader; the template is not committed), renders every case in CASES with jinja2
in the environment transformers.apply_chat_template builds, and writes
tests/ornith/chat/golden/<case>.txt plus <case>.input.json:
  api       "chat" (POST /v1/chat/completions) or "anthropic" (POST /v1/messages)
  body      the HTTP body ds4-server parses (OpenAI tool arguments are JSON strings)
  template  what jinja saw: messages, tools, kwargs
OpenAI string tool arguments are parsed into objects before rendering, as
llama.cpp does for templates that accept object arguments.  Anthropic cases
carry a hand-written OpenAI-shaped template input; tools keep ds4's
Anthropic mapping, {"type": "function", "function": <tool as given>}.
--check re-renders and compares with the committed files instead of writing.

jinja2 is not in the system python3; run with
  ~/.local/ai-gateway-env/bin/python3 tests/ornith/chat/render_golden.py
(never pip install into ~/.local/omlx-venv).
"""
import copy
import hashlib
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "golden")
TEMPLATE_VERSION = "qwen3.8-froggeric-v22.4.1"
MODEL_ID = "ornith-1.5-35b-a3b"

GGUF_STRING, GGUF_ARRAY = 8, 9
GGUF_SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _read_exact(f, n):
    data = f.read(n)
    if len(data) != n:
        raise SystemExit("render_golden.py: truncated GGUF header")
    return data


def _read_string(f):
    n, = struct.unpack("<Q", _read_exact(f, 8))
    return _read_exact(f, n).decode("utf-8")


def _skip_value(f, vtype):
    if vtype == GGUF_STRING:
        n, = struct.unpack("<Q", _read_exact(f, 8))
        f.seek(n, 1)
    elif vtype == GGUF_ARRAY:
        etype, count = struct.unpack("<IQ", _read_exact(f, 12))
        if etype in GGUF_SCALAR_BYTES:
            f.seek(GGUF_SCALAR_BYTES[etype] * count, 1)
        else:
            for _ in range(count):
                _skip_value(f, etype)
    elif vtype in GGUF_SCALAR_BYTES:
        f.seek(GGUF_SCALAR_BYTES[vtype], 1)
    else:
        raise SystemExit(f"render_golden.py: unknown GGUF value type {vtype}")


def read_chat_template(path):
    """tokenizer.chat_template from a GGUF file, reading only the metadata."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"render_golden.py: {path} is not a GGUF file")
        _read_exact(f, 4)                                   # version
        _n_tensors, n_kv = struct.unpack("<QQ", _read_exact(f, 16))
        for _ in range(n_kv):
            key = _read_string(f)
            vtype, = struct.unpack("<I", _read_exact(f, 4))
            if key == "tokenizer.chat_template":
                if vtype != GGUF_STRING:
                    raise SystemExit("render_golden.py: tokenizer.chat_template is not a string")
                return _read_string(f)
            _skip_value(f, vtype)
    raise SystemExit(f"render_golden.py: {path} has no tokenizer.chat_template")


def template_view(body):
    """The template inputs llama.cpp derives from an OpenAI chat body."""
    messages = copy.deepcopy(body.get("messages") or [])
    for message in messages:
        for call in message.get("tool_calls") or []:
            fn = call.get("function", call)
            args = fn.get("arguments")
            if not isinstance(args, str):
                continue
            try:
                parsed = json.loads(args)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                fn["arguments"] = parsed
    kwargs = {}
    if "reasoning_effort" in body:
        kwargs["reasoning_effort"] = body["reasoning_effort"]
    kwargs.update(body.get("chat_template_kwargs") or {})
    return {"messages": messages, "tools": copy.deepcopy(body.get("tools")), "kwargs": kwargs}


def case_spec(case):
    api = case.get("api", "chat")
    template = case["template"] if "template" in case else template_view(case["body"])
    return {"api": api, "body": case["body"], "template": template}


def make_renderer(template_text):
    """jinja2 environment of transformers' apply_chat_template."""
    try:
        import jinja2
        import jinja2.ext
        from jinja2.exceptions import TemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError:
        raise SystemExit("render_golden.py: jinja2 missing; run with ~/.local/ai-gateway-env/bin/python3")

    def raise_exception(message):
        raise TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                          sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    compiled = env.from_string(template_text)

    def render(view):
        ctx = {"messages": view["messages"], "add_generation_prompt": True,
               "bos_token": "<|endoftext|>", "eos_token": "<|im_end|>"}
        if view.get("tools"):
            ctx["tools"] = view["tools"]
        ctx.update(view.get("kwargs") or {})
        return compiled.render(**ctx)

    render.jinja2_version = jinja2.__version__
    return render


SYSTEM = "You are a helpful assistant."
USER = "What is the capital of France?"
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string", "description": "City name"},
                                      "unit": {"type": "string", "enum": ["c", "f"]}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from disk.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "max_lines": {"type": "integer"}},
                       "required": ["path"]}}},
]
RUN_TOOL = {"type": "function", "function": {
    "name": "run",
    "description": "Run a shell command.",
    "parameters": {"type": "object",
                   "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
                   "required": ["command"]}}}
UNICODE_TOOL = {"type": "function", "function": {
    "name": "translate",
    "description": "Dịch \"văn bản\" sang tiếng Việt — nhanh.\tTab.",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    "strict": True}}
ANTHROPIC_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
WEATHER_OK = '{"temp_c": 18, "sky": "cloudy"}'


def call(cid, name, args):
    """An OpenAI wire tool call: arguments travel as a JSON string."""
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def chat(messages, **extra):
    body = {"model": MODEL_ID, "messages": messages}
    body.update(extra)
    return {"api": "chat", "body": body}


def sys_user(**extra):
    return chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}], **extra)


def weather_turns(tool_content):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": "", "reasoning_content": "The user wants weather. Call the tool.",
         "tool_calls": [call("call_1", "get_weather", {"city": "Paris", "unit": "c"})]},
        {"role": "tool", "tool_call_id": "call_1", "content": tool_content},
        {"role": "assistant", "content": "It is 18 C and cloudy in Paris.",
         "reasoning_content": "Tool returned 18 C cloudy."},
        {"role": "user", "content": "And should I bring an umbrella?"},
    ]


HISTORY = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": USER},
    {"role": "assistant", "content": "The capital of France is Paris.",
     "reasoning_content": "Simple geography question.\nAnswer: Paris."},
    {"role": "user", "content": "And of Italy?"},
]

CASES = {
    # system turn
    "chat_system_user": sys_user(),
    "chat_think_off": sys_user(chat_template_kwargs={"enable_thinking": False}),
    "chat_no_system": chat([{"role": "user", "content": USER}]),
    "chat_terse_off": sys_user(chat_template_kwargs={"terse": False}),
    "developer_head_merge": chat([
        {"role": "system", "content": "  You are a helpful assistant.  "},
        {"role": "developer", "content": "Prefer metric units.\n"},
        {"role": "user", "content": USER}]),
    "effort_none": sys_user(reasoning_effort="none"),
    "effort_medium": sys_user(reasoning_effort="medium"),
    "effort_high": sys_user(reasoning_effort="high"),
    "effort_low": sys_user(reasoning_effort="low"),
    "effort_max": sys_user(reasoning_effort="max"),
    "effort_low_think_off": sys_user(chat_template_kwargs={"reasoning_effort": "low",
                                                           "enable_thinking": False}),
    "tools": sys_user(tools=TOOLS),
    "tools_think_off": sys_user(tools=TOOLS, chat_template_kwargs={"enable_thinking": False}),
    "tools_unicode": sys_user(tools=[UNICODE_TOOL]),
    # conversation body
    "tool_round_trip": chat(weather_turns(WEATHER_OK), tools=TOOLS),
    "tool_error": chat(weather_turns("Error: city not found"), tools=TOOLS),
    "tool_errors_consecutive": chat([
        {"role": "user", "content": "Run the build."},
        {"role": "assistant", "content": "", "reasoning_content": "Run make.",
         "tool_calls": [call("c1", "run", {"command": "make"})]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"error": "make: not found", "code": 127}'},
        {"role": "assistant", "content": "Retrying with the full path.", "reasoning_content": "Try /usr/bin/make.",
         "tool_calls": [call("c2", "run", {"command": "/usr/bin/make", "timeout": 30})]},
        {"role": "tool", "tool_call_id": "c2",
         "content": "Traceback (most recent call last):\n  File \"build.py\", line 3, in <module>\n"
                    "ImportError: no module named x\n"}],
        tools=[RUN_TOOL]),
    "tool_output_trailing_nl": chat([
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": "", "tool_calls": [call("c1", "read_file", {"path": "/tmp"})]},
        {"role": "tool", "tool_call_id": "c1", "content": "a.txt\nb.txt\n"}], tools=TOOLS),
    "tool_results_grouped": chat([
        {"role": "user", "content": "read it"},
        {"role": "assistant", "content": "Reading the file now.", "tool_calls": [
            call("c1", "read_file", {"path": "/tmp/a.txt", "max_lines": 20,
                                     "opts": {"a": [1, 2], "b": {"deep": True}}}),
            call("c2", "get_weather", {"city": "Paris", "unit": "c"})]},
        {"role": "tool", "tool_call_id": "c1", "content": "line1\nline2"},
        {"role": "tool", "tool_call_id": "c2", "content": '{"temp_c": 18, "error": null}'}], tools=TOOLS),
    "history_reasoning": chat(HISTORY),
    "preserve_thinking_off": chat(HISTORY, chat_template_kwargs={"preserve_thinking": False}),
    "preserve_thinking_off_tool_loop": chat([
        {"role": "user", "content": "Run the build."},
        {"role": "assistant", "content": "", "reasoning_content": "Run make.",
         "tool_calls": [call("c1", "run", {"command": "make"})]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"}],
        tools=[RUN_TOOL], chat_template_kwargs={"preserve_thinking": False}),
    "assistant_trailing_ws": chat([
        {"role": "user", "content": USER},
        {"role": "assistant", "content": "Paris.\n\n", "reasoning_content": "easy"},
        {"role": "user", "content": "And of Italy?"}]),
    "midconv_system": chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
        {"role": "assistant", "content": "Paris."},
        {"role": "system", "content": "Reminder: be brief."},
        {"role": "user", "content": "And of Italy?"}]),
    # the gateway's think-cap replay: enable_thinking false, preserve_thinking removed
    "think_cap_replay": chat(weather_turns(WEATHER_OK), tools=TOOLS,
                             chat_template_kwargs={"enable_thinking": False}),
    "anthropic_tool_result": {
        "api": "anthropic",
        "body": {"model": MODEL_ID, "max_tokens": 256, "system": SYSTEM, "tools": [ANTHROPIC_TOOL],
                 "messages": [
                     {"role": "user", "content": "What's the weather in Paris?"},
                     {"role": "assistant", "content": [
                         {"type": "thinking", "thinking": "The user wants weather. Call the tool.",
                          "signature": "sig"},
                         {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                          "input": {"city": "Paris"}}]},
                     {"role": "user", "content": [
                         {"type": "tool_result", "tool_use_id": "toolu_1",
                          "content": "Error: city not found\n"},
                         {"type": "text", "text": "Try Lyon instead."}]}]},
        "template": {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "What's the weather in Paris?"},
                {"role": "assistant", "content": "", "reasoning_content": "The user wants weather. Call the tool.",
                 "tool_calls": [{"id": "toolu_1", "type": "function",
                                 "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
                {"role": "tool", "tool_call_id": "toolu_1", "content": "Error: city not found\n"},
                {"role": "user", "content": "Try Lyon instead."}],
            "tools": [{"type": "function", "function": ANTHROPIC_TOOL}],
            "kwargs": {}},
    },
}


def main(argv):
    check = argv == ["--check"]
    if argv and not check:
        sys.exit(__doc__)
    model = os.environ.get("DS4_ORNITH_MODEL")
    if not model:
        sys.exit("render_golden.py: set DS4_ORNITH_MODEL to the 23G ICE GGUF")
    template = read_chat_template(model)
    first_line = template.splitlines()[0] if template else ""
    if TEMPLATE_VERSION not in first_line:
        sys.exit(f"render_golden.py: expected template {TEMPLATE_VERSION}, found {first_line!r}")
    render = make_renderer(template)
    info = (f"template_version: {TEMPLATE_VERSION}\n"
            f"sha256: {hashlib.sha256(template.encode('utf-8')).hexdigest()}\n"
            f"source: {os.path.basename(model)}\n"
            f"jinja2: {render.jinja2_version}\n")
    os.makedirs(GOLDEN, exist_ok=True)
    outputs = {"TEMPLATE.txt": info}
    for name, case in CASES.items():
        spec = case_spec(case)
        outputs[name + ".txt"] = render(spec["template"])
        outputs[name + ".input.json"] = json.dumps(spec, ensure_ascii=False, indent=1) + "\n"
    stale = []
    for fname, text in outputs.items():
        path = os.path.join(GOLDEN, fname)
        if check:
            try:
                with open(path, encoding="utf-8", newline="") as f:
                    if f.read() != text:
                        stale.append(fname)
            except FileNotFoundError:
                stale.append(fname)
        else:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(text)
    if check and stale:
        sys.exit("render_golden.py: out of date: " + ", ".join(sorted(stale)))
    print(f"render_golden.py: {'checked' if check else 'wrote'} {len(CASES)} cases in {GOLDEN}")


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 6: Run the unit tests and confirm they pass**

Run: `python3 -m unittest tests/ornith/chat/test_render_golden.py -v`
Expected: `Ran 8 tests ... OK`.

- [ ] **Step 7: Generate the goldens and check them twice**

This reads only the GGUF header; no GPU is used.

```bash
export DS4_ORNITH_MODEL=$HOME/orca/workspaces/ds4-metal-data/gguf/ornith/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf
~/.local/ai-gateway-env/bin/python3 tests/ornith/chat/render_golden.py
~/.local/ai-gateway-env/bin/python3 tests/ornith/chat/render_golden.py --check
ls tests/ornith/chat/golden | wc -l
cat tests/ornith/chat/golden/TEMPLATE.txt
grep -c 'SYSTEM WARNING' tests/ornith/chat/golden/tool_error.txt tests/ornith/chat/golden/tool_errors_consecutive.txt tests/ornith/chat/golden/anthropic_tool_result.txt
grep -c '2 consecutive tool errors detected' tests/ornith/chat/golden/tool_errors_consecutive.txt
```

Expected: `wrote 26 cases`, then `checked 26 cases`; 53 files; `template_version: qwen3.8-froggeric-v22.4.1`; warning counts 1, 2, 1; the consecutive count 1.

If the planning research renders still exist, compare the overlapping cases byte for byte (they came from the same template and inputs):

```bash
R=/private/tmp/claude-501/-Users-dongnh-orca-workspaces-ds4-metal-foxface/6e0ae78b-35fa-4241-bd9e-d0e569e9dbd0/scratchpad/plan-research
G=tests/ornith/chat/golden
if [ -d "$R" ]; then
  for pair in chat_system_user:a chat_think_off:b tools:c tool_round_trip:d history_reasoning:e \
              chat_no_system:a_nosystem chat_terse_off:a_terse_off effort_high:a_effort_high \
              effort_low:a_effort_low tools_think_off:b_tools_nothink tool_error:d_tool_error \
              tool_output_trailing_nl:d_tool_trailing_nl assistant_trailing_ws:e_assistant_trailing_ws \
              preserve_thinking_off:e_preserve_off midconv_system:f_midconv_system; do
    cmp "$G/${pair%%:*}.txt" "$R/render_${pair#*:}.txt" && echo "${pair%%:*}: same as research"
  done
fi
```

Expected: 15 `same as research` lines when the directory exists. A difference means the generator's environment or inputs drifted: stop and compare.

- [ ] **Step 8: Commit**

```bash
git add tests/ornith/chat/render_golden.py tests/ornith/chat/test_render_golden.py tests/ornith/chat/golden
git commit -m "tests/ornith: golden renders of the embedded chat template

render_golden.py reads tokenizer.chat_template from the GGUF header and
renders 26 conversations with jinja2 in the transformers environment:
thinking on/off, tools on/off, every effort, tool errors, trailing
newlines, grouped results, history reasoning, preserve_thinking false,
a later system message, the gateway think-cap replay and an Anthropic
tool_result turn.  The renders and their inputs are committed fixtures.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- tests/ornith/chat
```

---

### Task 2: Shared Qwen text predicate and the CLI's Ornith chat rendering

`encode_chat_prompt`, `ds4_chat_append_message` and `ds4_chat_append_assistant_prefix` still test `ds4_model_is_qwen4()`, so Ornith chat prompts die in the DeepSeek branch. This task switches them to the shared predicate, gives Ornith its template's system turn and removes the DeepSeek max prefix for Ornith.

**Files:**
- Modify: `ds4.h`, `ds4.c` (`chat_push_think_prefix`, the `DS4_QWEN4_REASONING_*` strings, `qwen4_chat_system`, `encode_chat_prompt`, `ds4_chat_append_message`, `ds4_chat_append_assistant_prefix`, `ds4_think_mode_for_context`, `ds4_engine_is_qwen35moe`)
- Create: `tests/ornith/test_cli_chat.sh`

**Interfaces:**
- Consumes: Task 1 goldens `chat_system_user`, `chat_think_off`, `chat_no_system`, `effort_max`.
- Produces (public, `ds4.h`):
  - `bool ds4_engine_uses_qwen35_text(ds4_engine *e);` (ignores `e`, like `ds4_engine_is_qwen4`)
  - `const char *ds4_engine_reasoning_effort_text(ds4_engine *e, ds4_think_mode mode);`
  - `const char *ds4_qwen35_terse_text(bool think);`

- [ ] **Step 1: Record the Qwen3.8 CLI tokens before any change**

`--dump-tokens` reads only the GGUF header. The Qwen3.8 GGUF path comes from the live registry (read-only).

```bash
make ds4
mkdir -p /tmp/ornith-m3-qwen38-cli
python3 - > /tmp/ornith-m3-qwen38-cli/model.txt <<'EOF'
import json, os
reg = json.load(open(os.path.expanduser("~/.local/ai-gateway/runtime-registry.json")))["models"]
entry = next(m["runtimes"]["ds4"] for m in reg.values()
             if m.get("runtimes", {}).get("ds4", {}).get("enabled"))
cmd = entry["process_command"]
i = next(i for i, a in enumerate(cmd) if a in ("-m", "--model"))
path = cmd[i + 1]
print(path if os.path.isabs(path) else os.path.join(entry.get("process_cwd") or "", path))
EOF
QWEN38_MODEL=$(cat /tmp/ornith-m3-qwen38-cli/model.txt)
echo "$QWEN38_MODEL"
for mode in "" --nothink --think-max; do
  ./ds4 -m "$QWEN38_MODEL" --dump-tokens $mode -sys "You are terse." -p "Hello" \
    > "/tmp/ornith-m3-qwen38-cli/before${mode:-_default}.txt" 2>&1
done
head -c 200 /tmp/ornith-m3-qwen38-cli/before_default.txt
```

Expected: a token-id list starting with `[248045` (`<|im_start|>`). If the dump fails for Qwen3.8, record the error text; Step 9 then compares the same failure.

- [ ] **Step 2: Write the failing CLI chat test**

Create `tests/ornith/test_cli_chat.sh`:

```sh
#!/bin/sh
# Ornith CLI chat rendering equals the embedded template token for token:
# ./ds4 --dump-tokens renders --system/-p through encode_chat_prompt, and the
# golden text of the same conversation, tokenized as a rendered prompt
# (--raw), must give the same ids.  Reads only the GGUF header.
# Needs a built ./ds4 and DS4_ORNITH_MODEL.
set -eu
model=${DS4_ORNITH_MODEL:?set DS4_ORNITH_MODEL to the 23G ICE GGUF}
golden=tests/ornith/chat/golden
tmp=$(mktemp -d "${TMPDIR:-/tmp}/ornith-cli-chat.XXXXXX")
trap 'rm -rf "$tmp"' EXIT

# check NAME GOLDEN CLI-ARGS...
check() {
    name=$1
    case_name=$2
    shift 2
    if ! ./ds4 -m "$model" --dump-tokens "$@" > "$tmp/$name.cli" 2> "$tmp/$name.err"; then
        cat "$tmp/$name.err"
        echo "$name: ./ds4 --dump-tokens failed"
        exit 1
    fi
    ./ds4 -m "$model" --dump-tokens --raw --prompt-file "$golden/$case_name.txt" > "$tmp/$name.golden" \
        2> "$tmp/$name.golden.err" || { cat "$tmp/$name.golden.err"; exit 1; }
    if [ "$(head -n 1 "$tmp/$name.cli")" != "$(head -n 1 "$tmp/$name.golden")" ]; then
        echo "$name: CLI tokens differ from golden $case_name"
        diff "$tmp/$name.cli" "$tmp/$name.golden" | head -20
        exit 1
    fi
    echo "$name: ok"
}

check default chat_system_user -sys "You are a helpful assistant." -p "What is the capital of France?"
check nothink chat_think_off --nothink -sys "You are a helpful assistant." -p "What is the capital of France?"
# ./ds4 has a built-in default system prompt; -sys "" removes it
check nosystem chat_no_system -sys "" -p "What is the capital of France?"
check think_max effort_max --think-max -sys "You are a helpful assistant." -p "What is the capital of France?"
echo "ornith cli chat: ok"
```

- [ ] **Step 3: Run it and confirm it fails**

```bash
chmod +x tests/ornith/test_cli_chat.sh
./tests/ornith/test_cli_chat.sh
```

Expected: `this tokenizer does not provide the DeepSeek chat markers; use raw prompt tokenization` and `default: ./ds4 --dump-tokens failed`.

- [ ] **Step 4: Public declarations**

In `ds4.h`, directly after `bool ds4_engine_is_qwen35moe(ds4_engine *e);`:

```c
/* Qwen3.5 tokenizer, ChatML turns and XML tool calls: Qwen3.8 and Ornith. */
bool ds4_engine_uses_qwen35_text(ds4_engine *e);
/* Reasoning-effort system line the engine-side renderers (CLI, agent) put in
 * the system turn, or NULL.  Qwen3.8: ds4_qwen4_reasoning_effort_text().
 * Ornith: the template defaults to medium (no line), and the frontends'
 * default DS4_THINK_HIGH stands for "no effort given"; DS4_THINK_MAX gives
 * the xhigh line and DS4_THINK_LOW the low line. */
const char *ds4_engine_reasoning_effort_text(ds4_engine *e, ds4_think_mode mode);
/* Ornith's "terse" system block (template kwarg terse, default true); the
 * lead line depends on whether thinking is on. */
const char *ds4_qwen35_terse_text(bool think);
```

- [ ] **Step 5: Ornith text helpers and system turn in `ds4.c`**

Directly after the definition of `DS4_QWEN4_REASONING_LOW` (the line `"moving directly to the conclusion without unnecessary elaboration.";`), before `static void qwen4_chat_open(`:

```c
/* Ornith's embedded template (froggeric v22.4.1) appends this block to the
 * system turn unless the caller passes terse=false. */
#define DS4_QWEN35_TERSE_CORE \
    "Never: open with preamble or pleasantries; restate the question; add filler transitions; " \
    "hedge with niceties; or repeat a point you've already made.\n" \
    "Always: keep essential steps, caveats, uncertainties, and specifics \xe2\x80\x94 never drop " \
    "correctness or a needed warning for brevity. Keep the final answer lean. Use the least structure " \
    "that conveys it (plain prose when short; lists or code only when they earn their place). If " \
    "genuinely uncertain, say so and explain why \xe2\x80\x94 never omit uncertainty for the sake of " \
    "brevity.\n" \
    "If a user request is genuinely ambiguous, ask a sharp question, don't guess."

const char *ds4_qwen35_terse_text(bool think) {
    return think ?
        "Answer directly, after thinking. Lead with the answer, then only what it needs to be correct "
        "and usable.\n" DS4_QWEN35_TERSE_CORE :
        "Answer directly and concisely. Give the answer with only what it needs to be correct and "
        "usable.\n" DS4_QWEN35_TERSE_CORE;
}

/* The CLI and the agent have no "effort was given" signal and default to
 * DS4_THINK_HIGH, which therefore renders Ornith's default medium effort
 * (no line); --think-max gives the xhigh line, a low level the low line. */
static const char *qwen35_engine_effort_text(ds4_think_mode mode) {
    if (mode == DS4_THINK_MAX) return DS4_QWEN4_REASONING_XHIGH;
    if (mode == DS4_THINK_LOW) return DS4_QWEN4_REASONING_LOW;
    return NULL;
}
```

Directly after the closing brace of `qwen4_chat_system`, before `static void encode_chat_prompt(`:

```c
/* Ornith's system turn: [effort "\n\n"] [trimmed system "\n\n"] terse block.
 * The template emits it even without a system prompt.  The content is one
 * BPE call so it tokenizes like the server's rendered text. */
static void qwen35_chat_system(const ds4_vocab *vocab, const char *system, ds4_think_mode think_mode,
                               token_vec *out) {
    const bool think = ds4_think_mode_enabled(think_mode);
    const char *effort = think ? qwen35_engine_effort_text(think_mode) : NULL;
    const char *terse = ds4_qwen35_terse_text(think);
    const char *s = system ? system : "";
    while (*s && isspace((unsigned char)*s)) s++;
    size_t n = strlen(s);
    while (n > 0 && isspace((unsigned char)s[n - 1])) n--;
    const size_t cap = (effort ? strlen(effort) + 2u : 0u) + n + 2u + strlen(terse) + 1u;
    char *text = xmalloc(cap);
    snprintf(text, cap, "%s%s%.*s%s%s", effort ? effort : "", effort ? "\n\n" : "",
             (int)n, s, n ? "\n\n" : "", terse);
    qwen4_chat_open(vocab, "system", out);
    bpe_tokenize_text(vocab, text, out);
    qwen4_chat_close(vocab, out);
    free(text);
}
```

- [ ] **Step 6: Switch the three shared chat sites and the think prefix**

In `encode_chat_prompt`, replace

```c
    if (ds4_model_is_qwen4()) {
        if (vocab->im_start_id < 0 || vocab->im_end_id < 0 ||
            vocab->think_start_id < 0 || vocab->think_end_id < 0) {
            ds4_die("this tokenizer does not provide the Qwen chat markers; use raw prompt tokenization");
        }
        qwen4_chat_system(vocab, system, think_mode, out);
```

with

```c
    if (ds4_model_uses_qwen35_text()) {
        if (vocab->im_start_id < 0 || vocab->im_end_id < 0 ||
            vocab->think_start_id < 0 || vocab->think_end_id < 0) {
            ds4_die("this tokenizer does not provide the Qwen chat markers; use raw prompt tokenization");
        }
        if (ds4_model_is_qwen35moe()) qwen35_chat_system(vocab, system, think_mode, out);
        else qwen4_chat_system(vocab, system, think_mode, out);
```

In `ds4_chat_append_message`, replace the first `    if (ds4_model_is_qwen4()) {` in the function (it comes a blank line after `    if (!content) content = "";`) with `    if (ds4_model_uses_qwen35_text()) {`.

In `ds4_chat_append_assistant_prefix`, replace

```c
    if (ds4_model_is_qwen4()) {
        qwen4_chat_assistant_prefix(&e->vocab, think_mode, tokens);
```

with

```c
    if (ds4_model_uses_qwen35_text()) {
        qwen4_chat_assistant_prefix(&e->vocab, think_mode, tokens);
```

In `chat_push_think_prefix`, replace

```c
    } else if (think_mode == DS4_THINK_MAX) {
        bpe_tokenize_text(vocab, DS4_REASONING_EFFORT_MAX_PREFIX, out);
    }
```

with

```c
    } else if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_QWEN35_MOE) {
        /* Ornith renders its effort inside the ChatML system turn; the
         * DeepSeek max prefix is not part of its template. */
    } else if (think_mode == DS4_THINK_MAX) {
        bpe_tokenize_text(vocab, DS4_REASONING_EFFORT_MAX_PREFIX, out);
    }
```

In `ds4_think_mode_for_context`, replace

```c
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_DEEPSEEK41 &&
        mode == DS4_THINK_MAX && (uint32_t)(ctx_size > 0 ? ctx_size : 0) < DS4_THINK_MAX_MIN_CONTEXT) {
```

with

```c
    /* The clamp protects DeepSeek's max-effort prefix; Ornith's max is its
     * template's xhigh line, valid at any context. */
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_DEEPSEEK41 &&
        DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_QWEN35_MOE &&
        mode == DS4_THINK_MAX && (uint32_t)(ctx_size > 0 ? ctx_size : 0) < DS4_THINK_MAX_MIN_CONTEXT) {
```

- [ ] **Step 7: Public predicate and effort function**

Directly after

```c
bool ds4_engine_is_qwen35moe(ds4_engine *e) {
    return e && ds4_model_is_qwen35moe();
}
```

add

```c
bool ds4_engine_uses_qwen35_text(ds4_engine *e) {
    (void)e;
    return ds4_model_uses_qwen35_text();
}

const char *ds4_engine_reasoning_effort_text(ds4_engine *e, ds4_think_mode mode) {
    (void)e;
    return ds4_model_is_qwen35moe() ? qwen35_engine_effort_text(mode) : ds4_qwen4_reasoning_effort_text(mode);
}
```

- [ ] **Step 8: Build and run the CLI test**

```bash
make ds4 && ./tests/ornith/test_cli_chat.sh
```

Expected: four `ok` lines and `ornith cli chat: ok`. A token difference in `default` or `nothink` points at `qwen35_chat_system` (text) or at piecewise tokenization (`qwen4_chat_open` tokenizes the role and `\n` separately; the golden tokenizes `system\n...` in one segment). Print both lines with the text column (`sed -n '2,40p'`) to find the first differing token before changing anything.

- [ ] **Step 9: Qwen3.8 CLI tokens are unchanged**

```bash
QWEN38_MODEL=$(cat /tmp/ornith-m3-qwen38-cli/model.txt)
for mode in "" --nothink --think-max; do
  ./ds4 -m "$QWEN38_MODEL" --dump-tokens $mode -sys "You are terse." -p "Hello" \
    > "/tmp/ornith-m3-qwen38-cli/after${mode:-_default}.txt" 2>&1
  cmp "/tmp/ornith-m3-qwen38-cli/before${mode:-_default}.txt" "/tmp/ornith-m3-qwen38-cli/after${mode:-_default}.txt" &&
    echo "qwen3.8 ${mode:-default}: unchanged"
done
```

Expected: three `unchanged` lines.

- [ ] **Step 10: Build every target, run the model-free suites and the Qwen fast gate**

The fast gate needs a GPU window (see Global Constraints).

```bash
make ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_test ds4_agent_test && make cpu
# make cpu relinks ./ds4, ./ds4-server, ./ds4-agent, ./ds4-bench and ./ds4-eval as CPU-only
# binaries; rebuild the Metal ones before any model run or run.sh (it only runs make ds4-server)
rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent && make ds4 ds4-server ds4-bench ds4-eval ds4-agent
./ds4_test --server && ./ds4_agent_test
make test-qwen4-kernels test-qwen4-q2
caffeinate -i -s speed-bench/qwen-regression/run.sh fast 2>&1 | tail -5
```

Expected: `ds4 tests: ok`, the agent test's final `ok` line, the kernel tests passing, `qwen_gate: PASS`.

- [ ] **Step 11: Commit**

```bash
git add ds4.h ds4.c tests/ornith/test_cli_chat.sh
git commit -m "ds4: Ornith CLI chat through the shared Qwen text predicate

encode_chat_prompt, ds4_chat_append_message and the assistant prefix now
cover both Qwen3.5-tokenizer families.  Ornith's system turn carries the
template's terse block and its default medium effort (no line); only
--think-max adds the xhigh line, so the MAX clamp no longer applies to
Ornith, and the DeepSeek max prefix is never pushed for it.
tests/ornith/test_cli_chat.sh matches the CLI tokens to the goldens.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- ds4.h ds4.c tests/ornith/test_cli_chat.sh
```

---

### Task 3: Server Ornith flavor: system turn, effort default, template kwargs, render harness

**Files:**
- Modify: `ds4_server.c` (flavor block before the `request` struct, `request`, `request_init`, `parse_reasoning_effort_name`, `parse_chat_template_kwargs`, `server_model_syntax_for_engine`, the four request parsers, `request_tokenize_multimodal_prompt`, the renderer block around `render_qwen_chat_prompt_text`, unit tests)
- Modify: `tests/ds4_test.c` (`--qwen35-render`)
- Create: `tests/ornith/chat/check_render.py`
- Modify: `Makefile` (`test-ornith-render`)

**Interfaces:**
- Consumes: `ds4_engine_uses_qwen35_text`, `ds4_qwen35_terse_text` (Task 2); goldens (Task 1).
- Produces (static, `ds4_server.c`):
  - `typedef enum { SERVER_QWEN_FLAVOR_QWEN38 = 0, SERVER_QWEN_FLAVOR_ORNITH } server_qwen_flavor;`
  - `static server_qwen_flavor g_server_qwen_flavor;` and `static bool server_qwen_is_ornith(void);`
  - `typedef struct { bool terse; bool preserve_thinking; bool json_tool_format; } chat_template_opts;` and `static const chat_template_opts CHAT_TEMPLATE_DEFAULTS;`; field `chat_template_opts tmpl;` in `request`
  - `static bool parse_chat_template_kwargs_ex(const char **p, bool *thinking_enabled, bool *got_thinking, ds4_think_mode *effort, int *think_budget, chat_template_opts *opts);`
  - `static char *pyjson_from_raw(const char *raw);` (Python `json.dumps` spacing)
  - `static char *render_ornith_chat_prompt_text(const chat_msgs *msgs, const char *tool_schemas, const tool_schema_orders *tool_orders, ds4_think_mode think_mode, const chat_template_opts *opts);`
  - `static char *render_chat_prompt_text_opts(server_model_syntax syntax, const chat_msgs *msgs, const char *tool_schemas, const tool_schema_orders *tool_orders, ds4_think_mode think_mode, const chat_template_opts *opts);`
  - Test helpers: `static char *test_ornith_prompt(const char *api, const char *body, ds4_think_mode *mode);`, `static bool ornith_test_ends_with(const char *s, const char *suffix);`
- Produces (CLI): `ds4_test --qwen35-render BODY.json [--anthropic]` prints the rendered prompt text.
- Produces (script): `check_render.py [--ds4-test PATH] [CASE ...]`, exit 0 only when every case matches.

- [ ] **Step 1: Write the failing unit tests**

In `ds4_server.c`, directly above `static void ds4_server_unit_tests_run(void) {`:

```c
/* Parse one request body with the server parser, no engine (render only),
 * under the current flavor; returns a copy of the prompt text. */
static char *test_ornith_prompt(const char *api, const char *body, ds4_think_mode *mode) {
    request r;
    char err[160] = {0};
    bool ok;
    if (!strcmp(api, "anthropic"))
        ok = parse_anthropic_request(NULL, NULL, body, 64, 262144, &r, err, sizeof(err));
    else if (!strcmp(api, "responses"))
        ok = parse_responses_request(NULL, NULL, body, 64, 262144, &r, err, sizeof(err));
    else
        ok = parse_chat_request(NULL, NULL, body, 64, 262144, &r, err, sizeof(err));
    if (!ok) {
        fprintf(stderr, "ornith test parse (%s): %s\n", api, err);
        return NULL;
    }
    if (mode) *mode = r.think_mode;
    char *text = xstrdup(r.prompt_text ? r.prompt_text : "");
    request_free(&r);
    return text;
}

static bool ornith_test_ends_with(const char *s, const char *suffix) {
    const size_t n = strlen(s), m = strlen(suffix);
    return n >= m && !memcmp(s + n - m, suffix, m);
}

/* The Ornith flavor is off by default, so every Qwen3.8 render is unchanged;
 * on, the same messages get the template's system turn. */
static void test_ornith_render_flavor_is_opt_in(void) {
    chat_msgs msgs = {0};
    chat_msg sys = {0};
    sys.role = xstrdup("system");
    sys.content = xstrdup("You are terse.");
    chat_msgs_push(&msgs, sys);
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("Hello");
    chat_msgs_push(&msgs, user);

    TEST_ASSERT(server_model_syntax_for_engine(NULL) != SERVER_MODEL_SYNTAX_QWEN);
    char *qwen = render_chat_prompt_text_for_syntax(SERVER_MODEL_SYNTAX_QWEN, &msgs, NULL, NULL, DS4_THINK_HIGH);
    g_server_qwen_flavor = SERVER_QWEN_FLAVOR_ORNITH;
    TEST_ASSERT(server_model_syntax_for_engine(NULL) == SERVER_MODEL_SYNTAX_QWEN);
    char *ornith = render_chat_prompt_text_for_syntax(SERVER_MODEL_SYNTAX_QWEN, &msgs, NULL, NULL,
                                                      DS4_THINK_MEDIUM);
    g_server_qwen_flavor = SERVER_QWEN_FLAVOR_QWEN38;
    char *again = render_chat_prompt_text_for_syntax(SERVER_MODEL_SYNTAX_QWEN, &msgs, NULL, NULL, DS4_THINK_HIGH);
    TEST_ASSERT(qwen && again && !strcmp(qwen, again));
    TEST_ASSERT(qwen && strstr(qwen, "Answer directly") == NULL);
    buf want = {0};
    buf_puts(&want, "<|im_start|>system\nYou are terse.\n\n");
    buf_puts(&want, ds4_qwen35_terse_text(true));
    buf_puts(&want, "<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n");
    TEST_ASSERT(ornith && !strcmp(ornith, want.ptr));
    buf_free(&want);
    free(qwen);
    free(ornith);
    free(again);
    chat_msgs_free(&msgs);
}

/* Review Focus 5: an absent or null effort is Ornith's medium (no line);
 * explicit names map like the template; Qwen3.8 keeps its defaults. */
static void test_ornith_effort_sent_vs_absent(void) {
    static const char xhigh[] = "<|im_start|>system\nReasoning effort is set to xhigh.";
    static const char low[] = "<|im_start|>system\nReasoning effort is set to low.";
    static const char think_on[] = "<|im_start|>assistant\n<think>\n";
    static const char think_off[] = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
#define ORNITH_HI "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]"
    static const struct {
        const char *api;
        const char *body;
        int mode;              /* expected ds4_think_mode; -1 = not checked */
        const char *prefix;    /* expected effort line; NULL = none */
        bool think;
    } cases[] = {
        {"chat", "{" ORNITH_HI "}", DS4_THINK_MEDIUM, NULL, true},
        {"chat", "{\"reasoning_effort\":null," ORNITH_HI "}", DS4_THINK_MEDIUM, NULL, true},
        {"chat", "{\"reasoning_effort\":\"medium\"," ORNITH_HI "}", DS4_THINK_MEDIUM, NULL, true},
        {"chat", "{\"reasoning_effort\":\"high\"," ORNITH_HI "}", DS4_THINK_HIGH, xhigh, true},
        {"chat", "{\"reasoning_effort\":\"xhigh\"," ORNITH_HI "}", DS4_THINK_HIGH, xhigh, true},
        {"chat", "{\"reasoning_effort\":\"extreme\"," ORNITH_HI "}", DS4_THINK_HIGH, xhigh, true},
        {"chat", "{\"reasoning_effort\":\"max\"," ORNITH_HI "}", -1, xhigh, true},
        {"chat", "{\"reasoning_effort\":\"low\"," ORNITH_HI "}", DS4_THINK_LOW, low, true},
        {"chat", "{\"reasoning_effort\":\"minimal\"," ORNITH_HI "}", DS4_THINK_LOW, low, true},
        {"chat", "{\"reasoning_effort\":\"none\"," ORNITH_HI "}", DS4_THINK_NONE, NULL, false},
        {"chat", "{\"reasoning_effort\":\"off\"," ORNITH_HI "}", DS4_THINK_NONE, NULL, false},
        {"chat", "{\"chat_template_kwargs\":{\"reasoning_effort\":\"low\"}," ORNITH_HI "}",
         DS4_THINK_LOW, low, true},
        {"chat", "{\"chat_template_kwargs\":{\"enable_thinking\":false,\"reasoning_effort\":\"high\"},"
                 ORNITH_HI "}", DS4_THINK_NONE, NULL, false},
        {"anthropic", "{\"max_tokens\":8," ORNITH_HI "}", DS4_THINK_MEDIUM, NULL, true},
        {"anthropic", "{\"max_tokens\":8,\"output_config\":{\"effort\":\"high\"}," ORNITH_HI "}",
         DS4_THINK_HIGH, xhigh, true},
        {"responses", "{\"input\":\"hi\"}", DS4_THINK_MEDIUM, NULL, true},
        {"responses", "{\"input\":\"hi\",\"reasoning\":{\"effort\":\"low\"}}", DS4_THINK_LOW, low, true},
    };
    g_server_qwen_flavor = SERVER_QWEN_FLAVOR_ORNITH;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        ds4_think_mode mode = DS4_THINK_HIGH;
        char *p = test_ornith_prompt(cases[i].api, cases[i].body, &mode);
        TEST_ASSERT(p != NULL);
        if (!p) continue;
        const bool mode_ok = cases[i].mode < 0 || (int)mode == cases[i].mode;
        const bool prefix_ok = cases[i].prefix ? !strncmp(p, cases[i].prefix, strlen(cases[i].prefix))
                                               : strstr(p, "Reasoning effort") == NULL;
        const bool tail_ok = ornith_test_ends_with(p, cases[i].think ? think_on : think_off);
        const bool terse_ok = strstr(p, cases[i].think ? "Answer directly, after thinking."
                                                      : "Answer directly and concisely.") != NULL;
        if (!mode_ok || !prefix_ok || !tail_ok || !terse_ok)
            fprintf(stderr, "ornith effort case %zu (%s %s): mode %d\n%s\n", i, cases[i].api, cases[i].body,
                    (int)mode, p);
        TEST_ASSERT(mode_ok && prefix_ok && tail_ok && terse_ok);
        free(p);
    }
    g_server_qwen_flavor = SERVER_QWEN_FLAVOR_QWEN38;
#undef ORNITH_HI
    ds4_think_mode mode = DS4_THINK_NONE;
    char *p = test_ornith_prompt("chat", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", &mode);
    TEST_ASSERT(p && mode == DS4_THINK_HIGH);
    free(p);
    ds4_think_mode off = DS4_THINK_HIGH;
    TEST_ASSERT(!parse_reasoning_effort_name("off", &off) && off == DS4_THINK_HIGH);
}

/* terse, preserve_thinking / preserve_reasoning and tool_call_format reach the
 * request; anything but a boolean keeps the default, as a skipped key did. */
static void test_ornith_template_kwargs(void) {
    bool enabled = true, got = false;
    ds4_think_mode mode = DS4_THINK_HIGH;
    chat_template_opts opts = CHAT_TEMPLATE_DEFAULTS;
    const char *kw = "{\"terse\": false, \"preserve_thinking\": false, \"tool_call_format\": \"json\"}";
    TEST_ASSERT(parse_chat_template_kwargs_ex(&kw, &enabled, &got, &mode, NULL, &opts));
    TEST_ASSERT(!opts.terse && !opts.preserve_thinking && opts.json_tool_format && !got);
    opts = CHAT_TEMPLATE_DEFAULTS;
    kw = "{\"preserve_thinking\": false, \"preserve_reasoning\": true, \"terse\": \"no\", \"tool_call_format\": 7}";
    TEST_ASSERT(parse_chat_template_kwargs_ex(&kw, &enabled, &got, &mode, NULL, &opts));
    TEST_ASSERT(opts.terse && opts.preserve_thinking && !opts.json_tool_format);
    opts = CHAT_TEMPLATE_DEFAULTS;
    kw = "{\"preserve_thinking\": null}";
    TEST_ASSERT(parse_chat_template_kwargs_ex(&kw, &enabled, &got, &mode, NULL, &opts));
    TEST_ASSERT(opts.preserve_thinking);

    static const char json_format[] =
        "{\"chat_template_kwargs\":{\"tool_call_format\":\"json\"},"
        "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    request r;
    char err[160] = {0};
    /* Qwen3.8 ignores the kwarg as before */
    TEST_ASSERT(parse_chat_request(NULL, NULL, json_format, 64, 262144, &r, err, sizeof(err)));
    request_free(&r);
    g_server_qwen_flavor = SERVER_QWEN_FLAVOR_ORNITH;
    TEST_ASSERT(!parse_chat_request(NULL, NULL, json_format, 64, 262144, &r, err, sizeof(err)));
    TEST_ASSERT(strstr(err, "tool_call_format") != NULL);
    char *p = test_ornith_prompt("chat", "{\"chat_template_kwargs\":{\"terse\":false},"
                                         "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", NULL);
    TEST_ASSERT(p && !strcmp(p, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"));
    free(p);
    g_server_qwen_flavor = SERVER_QWEN_FLAVOR_QWEN38;
}

/* tojson as Python's json.dumps(ensure_ascii=False) prints it; numbers are
 * copied as written. */
static void test_ornith_pyjson(void) {
    char *s = pyjson_from_raw("{\"a\":[1,2,{\"b\":null}],\"c\":\"\\u00e9\\n\\\"q\\\"\\/\",\"d\":true,\"e\":-1.5e3}");
    TEST_ASSERT(s && !strcmp(s, "{\"a\": [1, 2, {\"b\": null}], \"c\": \"\xc3\xa9\\n\\\"q\\\"/\", "
                                "\"d\": true, \"e\": -1.5e3}"));
    free(s);
    s = pyjson_from_raw("\"\\b\\f\\u0001\\t\"");
    TEST_ASSERT(s && !strcmp(s, "\"\\b\\f\\u0001\\t\""));
    free(s);
    s = pyjson_from_raw(" [ ] ");
    TEST_ASSERT(s && !strcmp(s, "[]"));
    free(s);
    s = pyjson_from_raw("{}");
    TEST_ASSERT(s && !strcmp(s, "{}"));
    free(s);
    TEST_ASSERT(pyjson_from_raw("{\"a\":") == NULL);
}
```

At the end of `ds4_server_unit_tests_run`, after `    test_kv_cache_eviction_keeps_aligned_continued_frontiers();`:

```c
    test_ornith_render_flavor_is_opt_in();
    test_ornith_effort_sent_vs_absent();
    test_ornith_template_kwargs();
    test_ornith_pyjson();
```

- [ ] **Step 2: Write the golden comparer and the Makefile target**

Create `tests/ornith/chat/check_render.py`:

```python
#!/usr/bin/env python3
"""Compare ds4-server's Ornith rendering with the committed jinja2 goldens.

  check_render.py [--ds4-test PATH] [CASE ...]

For every tests/ornith/chat/golden/<case>.input.json (or the named cases),
writes the request body to a temporary file, runs
`ds4_test --qwen35-render BODY [--anthropic]` (the server's own parser and
renderer under the Ornith flavor, no model) and requires stdout to equal
<case>.txt byte for byte.  Prints a unified diff per mismatch.  Needs no
jinja2 and no model.
"""
import difflib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
GOLDEN = os.path.join(HERE, "golden")


def case_names():
    return sorted(f[:-len(".input.json")] for f in os.listdir(GOLDEN) if f.endswith(".input.json"))


def render(ds4_test, case):
    with open(os.path.join(GOLDEN, case + ".input.json"), encoding="utf-8") as f:
        spec = json.load(f)
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(spec["body"], f, ensure_ascii=False)
        cmd = [ds4_test, "--qwen35-render", path]
        if spec["api"] == "anthropic":
            cmd.append("--anthropic")
        run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finally:
        os.unlink(path)
    if run.returncode != 0:
        return None, run.stderr.decode("utf-8", "replace").strip()
    return run.stdout.decode("utf-8"), ""


def main(argv):
    ds4_test = os.path.join(ROOT, "ds4_test")
    if len(argv) >= 2 and argv[0] == "--ds4-test":
        ds4_test, argv = argv[1], argv[2:]
    cases = argv or case_names()
    failed = 0
    for case in cases:
        with open(os.path.join(GOLDEN, case + ".txt"), encoding="utf-8", newline="") as f:
            want = f.read()
        got, err = render(ds4_test, case)
        if got == want:
            print(f"{case}: ok")
            continue
        failed += 1
        print(f"{case}: FAIL")
        if got is None:
            print("  ds4_test: " + err)
        else:
            sys.stdout.writelines(difflib.unified_diff(want.splitlines(True), got.splitlines(True),
                                                       "golden (jinja2)", "ds4-server", n=2))
    verdict = "PASS" if not failed else f"FAIL ({failed}/{len(cases)})"
    print(f"ornith render: {verdict}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

In `Makefile`, directly after the `test-qwen35-session` recipe (`	./tests/test_qwen35_session "$(DS4_ORNITH_MODEL)"`), add (the recipe line starts with a tab):

```make

.PHONY: test-ornith-render
test-ornith-render: ds4_test
	python3 tests/ornith/chat/check_render.py
```

- [ ] **Step 3: Confirm the tests fail**

```bash
chmod +x tests/ornith/chat/check_render.py
make ds4_test 2>&1 | grep -m3 'error:'
python3 tests/ornith/chat/check_render.py chat_system_user
```

Expected: the build stops with `use of undeclared identifier 'g_server_qwen_flavor'` (and similar); the comparer prints `chat_system_user: FAIL` (the old `ds4_test` binary reports `unknown test switch: --qwen35-render`, or there is no binary).

- [ ] **Step 4: Flavor, template options and the request field**

Directly above the `typedef struct {` whose first member is `    req_kind kind;` (the `request` struct), add:

```c
/* SERVER_MODEL_SYNTAX_QWEN renders one of two ChatML dialects.  Qwen3.8 and
 * Ornith share the parser, the XML tool calls and the live tails; Ornith's
 * embedded template (froggeric v22.4.1) adds a terse system block, its own
 * tool instructions and effort default, and different tool-result and
 * history rules (spec §5, tests/ornith/chat/).  main() sets the flavor once
 * from the loaded engine; model-less unit tests switch it around Ornith cases
 * and restore Qwen3.8. */
typedef enum {
    SERVER_QWEN_FLAVOR_QWEN38 = 0,
    SERVER_QWEN_FLAVOR_ORNITH,
} server_qwen_flavor;

static server_qwen_flavor g_server_qwen_flavor = SERVER_QWEN_FLAVOR_QWEN38;

static bool server_qwen_is_ornith(void) {
    return g_server_qwen_flavor == SERVER_QWEN_FLAVOR_ORNITH;
}

/* Ornith template kwargs a request sets through chat_template_kwargs; the
 * Qwen3.8 renderer ignores them. */
typedef struct {
    bool terse;              /* terse, default true */
    bool preserve_thinking;  /* preserve_reasoning, else preserve_thinking; default true */
    bool json_tool_format;   /* tool_call_format "json": refused for Ornith */
} chat_template_opts;

static const chat_template_opts CHAT_TEMPLATE_DEFAULTS = {
    .terse = true,
    .preserve_thinking = true,
    .json_tool_format = false,
};
```

In the `request` struct, directly after `    int think_budget;       /* request thinking cap in tokens; 0 = none */`:

```c
    chat_template_opts tmpl;  /* Ornith template kwargs (chat completions) */
```

In `request_init`, directly after `    r->think_mode = DS4_THINK_HIGH;`:

```c
    r->tmpl = CHAT_TEMPLATE_DEFAULTS;
```

Directly above `static bool parse_ignore_eos_value(const char **p, request *r) {`:

```c
/* The effort a request renders with when it names none (or null): Qwen3.8's
 * template default is xhigh, Ornith's is medium, which adds no line. */
static ds4_think_mode request_default_reasoning_effort(const request *r) {
    return r->model_syntax == SERVER_MODEL_SYNTAX_QWEN && server_qwen_is_ornith() ?
           DS4_THINK_MEDIUM : DS4_THINK_HIGH;
}
```

In all four parsers (`parse_chat_request`, `parse_anthropic_request`, `parse_responses_request`, `parse_completion_request`) replace every

```c
    ds4_think_mode reasoning_effort = DS4_THINK_HIGH;
```

with

```c
    ds4_think_mode reasoning_effort = request_default_reasoning_effort(r);
```

(exactly four occurrences; each follows the `r->model_syntax = server_model_syntax_for_engine(e);` line of its parser).

- [ ] **Step 5: Effort names, template kwargs and the syntax for the flavor**

In `parse_reasoning_effort_name`, replace

```c
static bool parse_reasoning_effort_name(const char *s, ds4_think_mode *out) {
    if (!s) return false;
```

with

```c
static bool parse_reasoning_effort_name(const char *s, ds4_think_mode *out) {
    if (!s) return false;
    if (server_qwen_is_ornith()) {
        /* names Ornith's template also accepts */
        if (!strcmp(s, "off")) {
            *out = DS4_THINK_NONE;
            return true;
        }
        if (!strcmp(s, "ultracode") || !strcmp(s, "extreme")) {
            *out = DS4_THINK_HIGH;
            return true;
        }
    }
```

Replace the whole of `parse_chat_template_kwargs` (from its comment `/* chat_template_kwargs as the Qwen3.8 model card documents them: enable_thinking` to its closing brace) with:

```c
/* A template kwarg only a JSON boolean sets: null, strings and numbers keep
 * the default, as the key was skipped before Ornith read it. */
static bool parse_template_bool_hint(const char **p, int *out) {
    json_ws(p);
    if (**p == 't' || **p == 'f') {
        bool v = false;
        if (!json_bool(p, &v)) return false;
        *out = v ? 1 : 0;
        return true;
    }
    return json_skip_value(p);
}

/* chat_template_kwargs: enable_thinking and reasoning_effort as the Qwen3.8
 * model card documents them, thinking_budget sets the request's thinking cap;
 * with opts, Ornith's terse, preserve_thinking, preserve_reasoning (read
 * first, as the template does) and tool_call_format; other keys are ignored */
static bool parse_chat_template_kwargs_ex(const char **p, bool *thinking_enabled, bool *got_thinking,
                                          ds4_think_mode *effort, int *think_budget,
                                          chat_template_opts *opts) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    if (**p != '{') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    int terse = -1, preserve_thinking = -1, preserve_reasoning = -1;
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        json_ws(p);
        if (**p != ':') {
            free(key);
            return false;
        }
        (*p)++;
        json_ws(p);
        bool ok;
        if (!strcmp(key, "enable_thinking")) {
            ok = json_bool(p, thinking_enabled);
            if (ok) *got_thinking = true;
        } else if (!strcmp(key, "reasoning_effort")) {
            ok = parse_reasoning_effort_value(p, effort);
        } else if (!strcmp(key, "thinking_budget") && think_budget) {
            ok = json_int(p, think_budget);
        } else if (opts && !strcmp(key, "terse")) {
            ok = parse_template_bool_hint(p, &terse);
        } else if (opts && !strcmp(key, "preserve_thinking")) {
            ok = parse_template_bool_hint(p, &preserve_thinking);
        } else if (opts && !strcmp(key, "preserve_reasoning")) {
            ok = parse_template_bool_hint(p, &preserve_reasoning);
        } else if (opts && !strcmp(key, "tool_call_format") && **p == '"') {
            char *format = NULL;
            ok = json_string(p, &format);
            if (ok) opts->json_tool_format = !strcmp(format, "json");
            free(format);
        } else {
            ok = json_skip_value(p);
        }
        free(key);
        if (!ok) return false;
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    if (opts) {
        if (terse >= 0) opts->terse = terse != 0;
        if (preserve_reasoning >= 0) opts->preserve_thinking = preserve_reasoning != 0;
        else if (preserve_thinking >= 0) opts->preserve_thinking = preserve_thinking != 0;
    }
    return true;
}

static DS4_SERVER_MAYBE_UNUSED bool parse_chat_template_kwargs(const char **p, bool *thinking_enabled,
                                                               bool *got_thinking, ds4_think_mode *effort,
                                                               int *think_budget) {
    return parse_chat_template_kwargs_ex(p, thinking_enabled, got_thinking, effort, think_budget, NULL);
}
```

In `server_model_syntax_for_engine`, replace `    if (ds4_engine_is_qwen4(engine)) return SERVER_MODEL_SYNTAX_QWEN;` with:

```c
    /* the Ornith flavor implies ChatML; model-less unit tests rely on it */
    if (ds4_engine_uses_qwen35_text(engine) || server_qwen_is_ornith()) return SERVER_MODEL_SYNTAX_QWEN;
```

- [ ] **Step 6: Parser wiring**

In `parse_chat_request`, replace

```c
            if (!parse_chat_template_kwargs(&p, &thinking_enabled, &got_thinking, &reasoning_effort,
                                            &r->think_budget)) {
```

with

```c
            if (!parse_chat_template_kwargs_ex(&p, &thinking_enabled, &got_thinking, &reasoning_effort,
                                               &r->think_budget, &r->tmpl)) {
```

In `parse_chat_request`, directly after the block

```c
    if (!request_validate_ignore_eos(r, err, errlen)) {
        chat_msgs_free(&msgs);
        free(tool_schemas);
        request_free(r);
        return false;
    }
```

add

```c
    if (r->tmpl.json_tool_format && r->model_syntax == SERVER_MODEL_SYNTAX_QWEN && server_qwen_is_ornith()) {
        snprintf(err, errlen, "tool_call_format \"json\" is not supported; Ornith tool calls use the xml format");
        chat_msgs_free(&msgs);
        free(tool_schemas);
        request_free(r);
        return false;
    }
```

In `parse_chat_request`, replace

```c
    tool_memory_attach_to_messages(s, &msgs, &r->tool_replay);
    const char *active_tool_schemas = r->has_tools ? tool_schemas : NULL;
    r->prompt_preserves_reasoning =
        chat_history_uses_tool_context(&msgs, active_tool_schemas);
    r->prompt_text = render_chat_prompt_text_for_syntax(
        r->model_syntax, &msgs, active_tool_schemas,
        &r->tool_orders, r->think_mode);
```

with

```c
    tool_memory_attach_to_messages(s, &msgs, &r->tool_replay);
    const char *active_tool_schemas = r->has_tools ? tool_schemas : NULL;
    r->prompt_preserves_reasoning =
        chat_history_uses_tool_context(&msgs, active_tool_schemas);
    r->prompt_text = render_chat_prompt_text_opts(
        r->model_syntax, &msgs, active_tool_schemas,
        &r->tool_orders, r->think_mode, &r->tmpl);
```

In `parse_anthropic_request`, replace

```c
        if (r->model_syntax == SERVER_MODEL_SYNTAX_DEEPSEEK41) {
            /* This is the API's initial system prompt, not a later system
             * turn, which V4.1 preserves at its original history position. */
```

with

```c
        if (r->model_syntax == SERVER_MODEL_SYNTAX_DEEPSEEK41 ||
            (r->model_syntax == SERVER_MODEL_SYNTAX_QWEN && server_qwen_is_ornith())) {
            /* This is the API's initial system prompt, not a later system
             * turn, which V4.1 and Ornith render at its history position. */
```

In `request_tokenize_multimodal_prompt`, replace

```c
    if (count == 0) {
        ds4_tokenize_rendered_chat(e, r->prompt_text, &r->prompt);
        return true;
    }
```

with

```c
    if (count == 0) {
        /* model-less unit tests parse without an engine and check the text */
        if (e) ds4_tokenize_rendered_chat(e, r->prompt_text, &r->prompt);
        return true;
    }
```

- [ ] **Step 7: The Python-style JSON printer**

Directly above the comment `/* Parameter values render as the template does: strings verbatim, other` (before `append_qwen_tool_calls_text`):

```c
/* Ornith's template prints tool schemas and non-string tool arguments with
 * Python's json.dumps(ensure_ascii=False): ", " and ": " at every depth, keys
 * in the given order, strings decoded and re-escaped.  Numbers are copied as
 * written. */
static void append_pyjson_string(buf *b, const char *s) {
    buf_putc(b, '"');
    for (const unsigned char *u = (const unsigned char *)(s ? s : ""); *u; u++) {
        switch (*u) {
        case '"': buf_puts(b, "\\\""); break;
        case '\\': buf_puts(b, "\\\\"); break;
        case '\n': buf_puts(b, "\\n"); break;
        case '\r': buf_puts(b, "\\r"); break;
        case '\t': buf_puts(b, "\\t"); break;
        case '\b': buf_puts(b, "\\b"); break;
        case '\f': buf_puts(b, "\\f"); break;
        default:
            if (*u < 0x20) buf_printf(b, "\\u%04x", (unsigned)*u);
            else buf_putc(b, (char)*u);
        }
    }
    buf_putc(b, '"');
}

static bool append_pyjson_value(buf *b, const char **p, int depth) {
    json_ws(p);
    const char open = **p;
    if (open == '{' || open == '[') {
        if (depth >= 128) return false;
        const char close = open == '{' ? '}' : ']';
        (*p)++;
        buf_putc(b, open);
        json_ws(p);
        for (bool first = true; **p && **p != close; first = false) {
            if (!first) buf_puts(b, ", ");
            if (open == '{') {
                char *key = NULL;
                if (!json_string(p, &key)) return false;
                append_pyjson_string(b, key);
                free(key);
                json_ws(p);
                if (**p != ':') return false;
                (*p)++;
                buf_puts(b, ": ");
            }
            if (!append_pyjson_value(b, p, depth + 1)) return false;
            json_ws(p);
            if (**p == ',') (*p)++;
            json_ws(p);
        }
        if (**p != close) return false;
        (*p)++;
        buf_putc(b, close);
        return true;
    }
    if (open == '"') {
        char *s = NULL;
        if (!json_string(p, &s)) return false;
        append_pyjson_string(b, s);
        free(s);
        return true;
    }
    const char *start = *p;
    while (**p && !strchr(",]} \t\r\n", **p)) (*p)++;
    if (*p == start) return false;
    buf_append(b, start, (size_t)(*p - start));
    return true;
}

static char *pyjson_from_raw(const char *raw) {
    buf b = {0};
    const char *p = raw ? raw : "";
    if (!append_pyjson_value(&b, &p, 0)) {
        buf_free(&b);
        return NULL;
    }
    return buf_take(&b);
}
```

- [ ] **Step 8: The Ornith system turn and its dispatch**

Directly after the closing brace of `render_qwen_chat_prompt_text`, before `static char *render_chat_prompt_text_for_syntax(`:

```c
/* A tool the client marked defer_loading: true stays out of the prompt, as
 * the Qwen3.8 renderer leaves it out. */
static bool tool_schema_is_deferred(const char *raw) {
    json_args args = {0};
    if (!json_args_parse(raw, &args)) return false;
    bool deferred = false;
    for (int i = 0; i < args.len; i++) {
        if (!strcmp(args.v[i].key, "defer_loading") && !args.v[i].is_string &&
            !strcmp(args.v[i].value, "true")) deferred = true;
    }
    json_args_free(&args);
    return deferred;
}

/* The template's tools block (xml tool_call_format, lines 195-210): each
 * tool as {"type": "function", "function": <schema>} through tojson, then
 * the instructions, whose example and first bullets depend on thinking. */
static void append_ornith_tools_text(buf *b, const char *tool_schemas, bool think) {
    buf_puts(b, "# Tools\n\nYou have access to the following functions:\n\n<tools>");
    const char *p = tool_schemas ? tool_schemas : "";
    json_ws(&p);
    while (*p) {
        char *raw = NULL;
        if (!json_raw_value(&p, &raw)) break;
        if (!tool_schema_is_deferred(raw)) {
            char *fn = pyjson_from_raw(raw);
            buf_puts(b, "\n{\"type\": \"function\", \"function\": ");
            buf_puts(b, fn ? fn : raw);
            buf_putc(b, '}');
            free(fn);
        }
        free(raw);
        json_ws(&p);
    }
    buf_puts(b, "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n");
    if (think) buf_puts(b, "<think>\nBrief explanation of tool call\n</think>\n");
    buf_puts(b,
        "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
        "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
        "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n");
    if (think) {
        buf_puts(b,
            "- You can use the <think></think> block to plan your next tool call OR to synthesize data and "
            "formulate your final response to the user.\n"
            "- ALL explanation and reasoning MUST be placed strictly inside the <think></think> block.\n");
    }
    buf_puts(b,
        "- Function calls MUST follow the specified format: an inner <function=...></function> block must be "
        "nested within <tool_call></tool_call> XML tags.\n"
        "- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY");
    if (think) buf_puts(b, " after thinking");
    buf_puts(b,
        ", with NO conversational text before it.\n"
        "- The <tool_call> and <function> tags MUST be at the very beginning of a new line, with NO spaces or "
        "indentation before them.\n"
        "- To call multiple functions, output a separate, completely closed <tool_call></tool_call> block for "
        "EACH function. Do NOT nest <tool_call> blocks.\n"
        "- If you have all necessary data, provide your final answer directly to the user without any tool call.\n"
        "</IMPORTANT>");
}

/* Ornith's system turn (template lines 132-221): effort line, tools block,
 * then the leading system/developer messages (each trimmed, joined by a
 * blank line) and the terse block.  Later system messages stay in place. */
static char *render_ornith_chat_prompt_text(const chat_msgs *msgs, const char *tool_schemas,
                                            const tool_schema_orders *tool_orders,
                                            ds4_think_mode think_mode, const chat_template_opts *opts) {
    const bool think = ds4_think_mode_enabled(think_mode);
    const char *effort = think ? ds4_qwen4_reasoning_effort_text(think_mode) : NULL;
    const bool have_tools = tool_schemas && tool_schemas[0];
    int head = 0;
    buf sc = {0};
    for (; msgs && head < msgs->len && role_is_system(msgs->v[head].role); head++) {
        buf part = {0};
        append_trimmed_text(&part, msgs->v[head].content);
        if (part.len) {
            if (sc.len) buf_puts(&sc, "\n\n");
            buf_append(&sc, part.ptr, part.len);
        }
        buf_free(&part);
    }
    if (opts->terse) {
        if (sc.len) buf_puts(&sc, "\n\n");
        buf_puts(&sc, ds4_qwen35_terse_text(think));
    }
    buf out = {0};
    if (have_tools || sc.len || effort) {
        buf_puts(&out, "<|im_start|>system\n");
        if (effort) {
            buf_puts(&out, effort);
            if (have_tools || sc.len) buf_puts(&out, "\n\n");
        }
        if (have_tools) {
            append_ornith_tools_text(&out, tool_schemas, think);
            if (sc.len) buf_puts(&out, "\n\n");
        }
        if (sc.len) buf_append(&out, sc.ptr, sc.len);
        buf_puts(&out, "<|im_end|>\n");
    }
    buf_free(&sc);
    /* Task 4 replaces this Qwen3.8 body with the Ornith conversation */
    append_qwen_conversation(&out, msgs, head, tool_orders, think);
    return buf_take(&out);
}
```

In `render_chat_prompt_text_for_syntax`, replace

```c
    if (syntax == SERVER_MODEL_SYNTAX_QWEN) {
        return render_qwen_chat_prompt_text(msgs, tool_schemas, tool_orders, think_mode);
    }
```

with

```c
    if (syntax == SERVER_MODEL_SYNTAX_QWEN) {
        if (server_qwen_is_ornith())
            return render_ornith_chat_prompt_text(msgs, tool_schemas, tool_orders, think_mode,
                                                  &CHAT_TEMPLATE_DEFAULTS);
        return render_qwen_chat_prompt_text(msgs, tool_schemas, tool_orders, think_mode);
    }
```

Directly after the closing brace of `render_chat_prompt_text_for_syntax`, before `static DS4_SERVER_MAYBE_UNUSED char *render_chat_prompt_text(`:

```c
/* Chat completions pass the request's template kwargs; every other caller
 * renders Ornith with the template defaults. */
static char *render_chat_prompt_text_opts(server_model_syntax syntax, const chat_msgs *msgs,
                                          const char *tool_schemas, const tool_schema_orders *tool_orders,
                                          ds4_think_mode think_mode, const chat_template_opts *opts) {
    if (syntax == SERVER_MODEL_SYNTAX_QWEN && server_qwen_is_ornith())
        return render_ornith_chat_prompt_text(msgs, tool_schemas, tool_orders, think_mode,
                                              opts ? opts : &CHAT_TEMPLATE_DEFAULTS);
    return render_chat_prompt_text_for_syntax(syntax, msgs, tool_schemas, tool_orders, think_mode);
}
```

- [ ] **Step 9: The render harness in `tests/ds4_test.c`**

Directly above `int main(int argc, char **argv) {`:

```c
/* Model-free Ornith render harness for tests/ornith/chat/check_render.py:
 * parse one request body with the server's own parser under the Ornith
 * flavor and print the rendered prompt text. */
static int test_qwen35_render_main(const char *path, const char *api) {
    const bool anthropic = api && !strcmp(api, "--anthropic");
    if (api && !anthropic) {
        fprintf(stderr, "ds4_test: unknown --qwen35-render option %s\n", api);
        return 2;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror(path);
        return 2;
    }
    buf body = {0};
    char chunk[4096];
    size_t n;
    while ((n = fread(chunk, 1, sizeof(chunk), fp)) > 0) buf_append(&body, chunk, n);
    fclose(fp);
    g_server_qwen_flavor = SERVER_QWEN_FLAVOR_ORNITH;
    request r;
    char err[256] = {0};
    const char *text = body.ptr ? body.ptr : "";
    const bool ok = anthropic ?
        parse_anthropic_request(NULL, NULL, text, 256, 262144, &r, err, sizeof(err)) :
        parse_chat_request(NULL, NULL, text, 256, 262144, &r, err, sizeof(err));
    g_server_qwen_flavor = SERVER_QWEN_FLAVOR_QWEN38;
    buf_free(&body);
    if (!ok) {
        fprintf(stderr, "ds4_test: %s\n", err);
        return 1;
    }
    fputs(r.prompt_text, stdout);
    request_free(&r);
    return 0;
}
```

In `main`, directly above `    bool run_all = argc == 1;`:

```c
    if ((argc == 3 || argc == 4) && !strcmp(argv[1], "--qwen35-render")) {
        return test_qwen35_render_main(argv[2], argc == 4 ? argv[3] : NULL);
    }
```

- [ ] **Step 10: Run the unit tests and the system-turn goldens**

```bash
make ds4_test ds4_agent_test
./ds4_test --server
./ds4_agent_test
python3 tests/ornith/chat/check_render.py chat_system_user chat_think_off chat_no_system chat_terse_off \
    developer_head_merge effort_none effort_medium effort_high effort_low effort_max effort_low_think_off \
    tools tools_think_off tools_unicode
```

Expected: `ds4 tests: ok` (all existing Qwen3.8 tests unchanged plus the four new ones), the agent tests pass, and 14 `ok` lines with `ornith render: PASS`. The conversation-body cases are Task 4.

- [ ] **Step 11: Build every target and run the Qwen fast gate**

```bash
make ds4 ds4-server ds4-bench ds4-eval ds4-agent && make cpu
# make cpu relinks ./ds4, ./ds4-server, ./ds4-agent, ./ds4-bench and ./ds4-eval as CPU-only
# binaries; rebuild the Metal ones before any model run or run.sh (it only runs make ds4-server)
rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent && make ds4 ds4-server ds4-bench ds4-eval ds4-agent
make test-qwen4-kernels test-qwen4-q2
caffeinate -i -s speed-bench/qwen-regression/run.sh fast 2>&1 | tail -5
```

Expected: clean builds, kernel tests pass, `qwen_gate: PASS`.

- [ ] **Step 12: Commit**

```bash
git add ds4_server.c tests/ds4_test.c tests/ornith/chat/check_render.py Makefile
git commit -m "ds4-server: Ornith render flavor for the system turn

An Ornith flavor of SERVER_MODEL_SYNTAX_QWEN renders the embedded
template's system turn: effort line, the template's tool instructions
with tojson spacing, the leading system messages and the terse block.
An absent effort is medium for Ornith (no line); chat_template_kwargs
terse, preserve_thinking/preserve_reasoning and tool_call_format reach
the request, and json tool calls are refused.  The flavor is off unless
set, so Qwen3.8 renders are unchanged.  ds4_test --qwen35-render and
tests/ornith/chat/check_render.py compare the server with the goldens.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- ds4_server.c tests/ds4_test.c tests/ornith/chat/check_render.py Makefile
```

---

### Task 4: Ornith conversation body, live tails and the full golden set

**Files:**
- Modify: `ds4_server.c` (Python-JSON block from Task 3, `append_qwen_tool_calls_text`, Ornith renderer block, `render_ornith_chat_prompt_text`, `render_qwen_live_tool_tail`, unit tests)
- Modify: `docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md` (§5 Chat rendering, §8 gate 1 item 5)

**Interfaces:**
- Consumes: Task 3 flavor, `pyjson_from_raw`, `render_ornith_chat_prompt_text`, `test_ornith_prompt`, `ornith_test_ends_with`.
- Produces (static):
  - `static void append_ornith_tool_calls_text(buf *b, const tool_calls *calls, bool has_content);`
  - `static bool ornith_tool_result_failed(const char *trimmed);`
  - `static void append_ornith_conversation(buf *out, const chat_msgs *msgs, int first, int start, bool think, bool preserve_thinking);` (`first` = first non-leading-system message, `start` = first message to emit)

- [ ] **Step 1: Write the failing unit tests**

Directly above `static void ds4_server_unit_tests_run(void) {`:

```c
/* The template's tool-error heuristic (lines 413-425) on trimmed results. */
static void test_ornith_tool_error_heuristic(void) {
    static const struct { const char *text; bool failed; } cases[] = {
        {"Error: city not found", true},
        {"{\"error\": \"file not found\", \"path\": \"/tmp/x\"}", true},
        {"{\"temp_c\": 18, \"error\": null}", false},
        {"{\"status\": \"error\", \"detail\": \"quota\"}", true},
        {"Traceback (most recent call last):\n  File \"x.py\", line 1", true},
        {"bash: foo: command not found", true},
        {"fatal: not a git repository", true},
        {"Process exited with code 1", true},
        {"Exit code: 0\nok", false},
        {"exception: boom", true},
        {"Failed to open /tmp/x", true},
        {"$ make\nerror: missing target", false},
        {"Build took 3s; error: none", false},
        {"def f():\n    raise ValueError('error: x')", false},
        {"src/a.c:12: logger.error(\"x\")", false},
        {"line1\nline2", false},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const bool got = ornith_tool_result_failed(cases[i].text);
        if (got != cases[i].failed) fprintf(stderr, "ornith tool error case %zu: '%s' -> %d\n", i, cases[i].text, got);
        TEST_ASSERT(got == cases[i].failed);
    }
    /* a weak error in output of 600 characters or more does not count */
    buf big = {0};
    buf_puts(&big, "error: first line\n");
    while (big.len < 700) buf_puts(&big, "more output ");
    TEST_ASSERT(!ornith_tool_result_failed(big.ptr));
    buf_free(&big);
    /* the head is 120 characters, not bytes: 100 two-byte characters, then a
     * strong signal inside the first 120 characters */
    buf wide = {0};
    for (int i = 0; i < 100; i++) buf_puts(&wide, "\xc3\xa9");
    buf_puts(&wide, " fatal: x");
    TEST_ASSERT(ornith_tool_result_failed(wide.ptr));
    buf_free(&wide);
}

/* Review Focus 2: a live tool tail counts the failures before it and is the
 * exact suffix of the full render, so live KV and a later replay agree. */
static void test_ornith_live_tail_continues_full_render(void) {
    static const char *roles[] = {"user", "assistant", "tool", "assistant", "tool"};
    static const char *texts[] = {
        "Run the build.", "", "make: *** No rule to make target 'all'.\nerror: build failed",
        "Retrying.", "Traceback (most recent call last):\n  File \"build.py\", line 3"};
    chat_msgs msgs = {0};
    for (int i = 0; i < 5; i++) {
        chat_msg m = {0};
        m.role = xstrdup(roles[i]);
        m.content = xstrdup(texts[i]);
        if (!strcmp(roles[i], "assistant")) {
            tool_call tc = {0};
            tc.id = xstrdup(i == 1 ? "c1" : "c2");
            tc.name = xstrdup("run");
            tc.arguments = xstrdup(i == 1 ? "{\"command\":\"make\"}" : "{\"command\":\"make all\",\"timeout\":30}");
            tool_calls_push(&m.calls, tc);
            m.reasoning = xstrdup("Use the tool.");
        }
        chat_msgs_push(&msgs, m);
    }
    static const char tail_head[] = "<|im_end|>\n<|im_start|>user\n<tool_response>\nTraceback";
    g_server_qwen_flavor = SERVER_QWEN_FLAVOR_ORNITH;
    char *full = render_chat_prompt_text_for_syntax(SERVER_MODEL_SYNTAX_QWEN, &msgs, NULL, NULL, DS4_THINK_MEDIUM);
    char *tail = render_live_tool_tail_for_syntax(SERVER_MODEL_SYNTAX_QWEN, &msgs, 4, NULL, DS4_THINK_MEDIUM);
    g_server_qwen_flavor = SERVER_QWEN_FLAVOR_QWEN38;
    TEST_ASSERT(full && strstr(full, "The previous tool call returned an error.") != NULL);
    TEST_ASSERT(tail && strstr(tail, "2 consecutive tool errors detected") != NULL);
    TEST_ASSERT(tail && !strncmp(tail, tail_head, strlen(tail_head)));
    TEST_ASSERT(full && tail && ornith_test_ends_with(full, tail));
    TEST_ASSERT(full && strstr(full, "<parameter=timeout>\n30\n</parameter>") != NULL);
    free(full);
    free(tail);
    chat_msgs_free(&msgs);
}

/* Review Focus 3: an Anthropic tool_result turn (a result block and a text
 * block in one user message, system at the top level) renders the bytes of
 * the OpenAI-shaped conversation. */
static void test_ornith_anthropic_tool_results_match_openai(void) {
    static const char anthropic[] =
        "{\"max_tokens\":64,\"system\":\"You are a helpful assistant.\","
        "\"tools\":[{\"name\":\"get_weather\",\"description\":\"Get the weather.\","
        "\"input_schema\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}],"
        "\"messages\":[{\"role\":\"user\",\"content\":\"Weather in Paris?\"},"
        "{\"role\":\"assistant\",\"content\":[{\"type\":\"thinking\",\"thinking\":\"Call the tool.\"},"
        "{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{\"city\":\"Paris\"}}]},"
        "{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_1\","
        "\"content\":\"Error: city not found\\n\"},{\"type\":\"text\",\"text\":\"Try Lyon.\"}]}]}";
    static const char openai[] =
        "{\"max_tokens\":64,"
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get the weather.\","
        "\"input_schema\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}}],"
        "\"messages\":[{\"role\":\"system\",\"content\":\"You are a helpful assistant.\"},"
        "{\"role\":\"user\",\"content\":\"Weather in Paris?\"},"
        "{\"role\":\"assistant\",\"content\":\"\",\"reasoning_content\":\"Call the tool.\","
        "\"tool_calls\":[{\"id\":\"toolu_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
        "\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"toolu_1\",\"content\":\"Error: city not found\\n\"},"
        "{\"role\":\"user\",\"content\":\"Try Lyon.\"}]}";
    static const char warning[] =
        "<tool_response>\nError: city not found\n\n\xe2\x9a\xa0\xef\xb8\x8f SYSTEM WARNING: "
        "The previous tool call returned an error.";
    static const char after[] =
        "</tool_response><|im_end|>\n<|im_start|>user\nTry Lyon.<|im_end|>\n<|im_start|>assistant\n<think>\n";
    static const char head[] = "<|im_start|>system\n# Tools\n";
    g_server_qwen_flavor = SERVER_QWEN_FLAVOR_ORNITH;
    char *a = test_ornith_prompt("anthropic", anthropic, NULL);
    char *o = test_ornith_prompt("chat", openai, NULL);
    g_server_qwen_flavor = SERVER_QWEN_FLAVOR_QWEN38;
    if (a && o && strcmp(a, o)) fprintf(stderr, "anthropic:\n%s\nopenai:\n%s\n", a, o);
    TEST_ASSERT(a && o && !strcmp(a, o));
    TEST_ASSERT(a && strstr(a, warning) != NULL);
    TEST_ASSERT(a && strstr(a, after) != NULL);
    TEST_ASSERT(a && !strncmp(a, head, strlen(head)));
    free(a);
    free(o);
}

/* The OpenAI tool-turn visible key stays a prefix of the next Ornith render,
 * so a tool loop continues from live KV. */
static void test_ornith_tool_turn_visible_text_prefixes_next_render(void) {
    g_server_qwen_flavor = SERVER_QWEN_FLAVOR_ORNITH;
    for (int with_content = 0; with_content < 2; with_content++) {
        chat_msgs msgs = {0};
        chat_msg user = {0};
        user.role = xstrdup("user");
        user.content = xstrdup("run it");
        chat_msgs_push(&msgs, user);
        request r = {0};
        r.kind = REQ_CHAT;
        r.model_syntax = SERVER_MODEL_SYNTAX_QWEN;
        r.think_mode = DS4_THINK_MEDIUM;
        r.prompt_text = render_chat_prompt_text_for_syntax(SERVER_MODEL_SYNTAX_QWEN, &msgs, NULL, NULL, r.think_mode);
        chat_msg assistant = {0};
        assistant.role = xstrdup("assistant");
        assistant.content = xstrdup(with_content ? "Running." : "");
        tool_call call = {0};
        call.name = xstrdup("bash");
        call.arguments = xstrdup("{}");
        tool_calls_push(&assistant.calls, call);
        assistant.calls.raw_tool_text = xstrdup("\n\n<tool_call>\n<function=bash>\n</function>\n</tool_call>");
        char *visible = build_qwen_tool_turn_visible_text(&r, "tool_calls", false, assistant.content,
                                                          &assistant.calls);
        chat_msgs_push(&msgs, assistant);
        chat_msg tool = {0};
        tool.role = xstrdup("tool");
        tool.content = xstrdup("ok\n");
        chat_msgs_push(&msgs, tool);
        char *next = render_chat_prompt_text_for_syntax(SERVER_MODEL_SYNTAX_QWEN, &msgs, NULL, NULL, r.think_mode);
        TEST_ASSERT(visible && next && !strncmp(next, visible, strlen(visible)));
        if (visible && next && !strncmp(next, visible, strlen(visible))) {
            static const char boundary[] =
                "<|im_end|>\n<|im_start|>user\n<tool_response>\nok\n</tool_response><|im_end|>\n";
            TEST_ASSERT(!strncmp(next + strlen(visible), boundary, strlen(boundary)));
        }
        free(next);
        free(visible);
        free(r.prompt_text);
        chat_msgs_free(&msgs);
    }
    g_server_qwen_flavor = SERVER_QWEN_FLAVOR_QWEN38;
}
```

At the end of `ds4_server_unit_tests_run`, after `    test_ornith_pyjson();`:

```c
    test_ornith_tool_error_heuristic();
    test_ornith_live_tail_continues_full_render();
    test_ornith_anthropic_tool_results_match_openai();
    test_ornith_tool_turn_visible_text_prefixes_next_render();
```

- [ ] **Step 2: Confirm the failures**

```bash
make ds4_test 2>&1 | grep -m2 'error:'
python3 tests/ornith/chat/check_render.py | tail -15
```

Expected: clang reports `call to undeclared function 'ornith_tool_result_failed'`; the comparer (old `ds4_test` binary from Task 3) prints 18 `ok` and 8 `FAIL`: `anthropic_tool_result`, `assistant_trailing_ws`, `midconv_system`, `preserve_thinking_off`, `tool_error`, `tool_errors_consecutive`, `tool_output_trailing_nl`, `tool_results_grouped`. (`history_reasoning`, `preserve_thinking_off_tool_loop`, `think_cap_replay` and `tool_round_trip` already pass with the Qwen3.8 body.)

- [ ] **Step 3: Ornith tool calls**

Directly after `pyjson_from_raw` (Task 3), before the comment `/* Parameter values render as the template does: strings verbatim, other`:

```c
/* Tool calls as the Ornith template renders them: arguments in the order the
 * call gives them (no schema reordering), non-string values through tojson.
 * Sampled tool text replays verbatim, as for Qwen3.8, so an echoed turn
 * matches the live KV. */
static void append_ornith_tool_calls_text(buf *b, const tool_calls *calls, bool has_content) {
    if (!calls || calls->len == 0) return;
    if (calls->raw_tool_text && calls->raw_tool_text[0]) {
        buf_puts(b, calls->raw_tool_text);
        return;
    }
    for (int i = 0; i < calls->len; i++) {
        const tool_call *tc = &calls->v[i];
        if (i == 0) buf_puts(b, has_content ? "\n\n<tool_call>\n<function=" : "<tool_call>\n<function=");
        else buf_puts(b, "\n<tool_call>\n<function=");
        buf_puts(b, tc->name ? tc->name : "");
        buf_puts(b, ">\n");
        json_args args = {0};
        if (json_args_parse(tc->arguments, &args)) {
            for (int k = 0; k < args.len; k++) {
                const json_arg *arg = &args.v[k];
                buf_puts(b, "<parameter=");
                append_glm_tag_body_text(b, arg->key, ">");
                buf_puts(b, ">\n");
                char *py = arg->is_string ? NULL : pyjson_from_raw(arg->value);
                append_glm_tag_body_text(b, arg->is_string ? arg->value : (py ? py : arg->value), "</parameter>");
                free(py);
                buf_puts(b, "\n</parameter>\n");
            }
            json_args_free(&args);
        } else if (tc->arguments && tc->arguments[0]) {
            /* not a JSON object: the template prints the raw text */
            append_glm_tag_body_text(b, tc->arguments, "</function>");
        }
        buf_puts(b, "</function>\n</tool_call>");
    }
}
```

In `append_qwen_tool_calls_text`, replace

```c
    if (!calls || calls->len == 0) return;
    if (calls->raw_tool_text && calls->raw_tool_text[0]) {
        buf_puts(b, calls->raw_tool_text);
        return;
    }
    for (int i = 0; i < calls->len; i++) {
        const tool_call *tc = &calls->v[i];
        const tool_schema_order *order = tool_schema_orders_find(tool_orders, tc->name);
```

with

```c
    if (server_qwen_is_ornith()) {
        append_ornith_tool_calls_text(b, calls, has_content);
        return;
    }
    if (!calls || calls->len == 0) return;
    if (calls->raw_tool_text && calls->raw_tool_text[0]) {
        buf_puts(b, calls->raw_tool_text);
        return;
    }
    for (int i = 0; i < calls->len; i++) {
        const tool_call *tc = &calls->v[i];
        const tool_schema_order *order = tool_schema_orders_find(tool_orders, tc->name);
```

- [ ] **Step 4: The Ornith conversation**

Directly above the comment `/* Ornith's system turn (template lines 132-221): effort line, tools block,` (Task 3):

```c
/* ---- Ornith conversation body (template lines 222-459) ----------------- */

typedef enum {
    ORNITH_ITEM_SYSTEM,
    ORNITH_ITEM_USER,
    ORNITH_ITEM_ASSISTANT,
    ORNITH_ITEM_TOOL,
} ornith_item_kind;

typedef struct {
    ornith_item_kind kind;
    int msg;            /* source message index */
    const char *text;   /* system/user/tool text before trimming */
    char *owned;        /* text assembled here, freed with the list */
} ornith_item;

typedef struct {
    ornith_item *v;
    int len;
    int cap;
} ornith_items;

static void ornith_items_push(ornith_items *items, ornith_item_kind kind, int msg, const char *text, char *owned) {
    if (items->len == items->cap) {
        items->cap = items->cap ? items->cap * 2 : 16;
        items->v = xrealloc(items->v, (size_t)items->cap * sizeof(items->v[0]));
    }
    items->v[items->len++] = (ornith_item){.kind = kind, .msg = msg, .text = owned ? owned : text, .owned = owned};
}

static void ornith_items_free(ornith_items *items) {
    for (int i = 0; i < items->len; i++) free(items->v[i].owned);
    free(items->v);
    memset(items, 0, sizeof(*items));
}

static bool ornith_text_is_blank(const char *s) {
    for (; s && *s; s++) {
        if (!isspace((unsigned char)*s)) return false;
    }
    return true;
}

/* The template's message list from msgs[first..].  OpenAI tool messages are
 * tool items.  An Anthropic user message with tool_result blocks becomes one
 * tool item per block, then a user item with the text outside the blocks:
 * what the OpenAI conversion of that message renders.  Roles the template
 * would show as "[role]: ..." are dropped, as the Qwen3.8 renderer drops them. */
static void ornith_items_build(ornith_items *items, const chat_msgs *msgs, int first) {
    for (int i = first; msgs && i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        const char *content = m->content ? m->content : "";
        if (role_is_system(m->role)) {
            ornith_items_push(items, ORNITH_ITEM_SYSTEM, i, content, NULL);
        } else if (!strcmp(m->role, "tool") || !strcmp(m->role, "function")) {
            ornith_items_push(items, ORNITH_ITEM_TOOL, i, content, NULL);
        } else if (!strcmp(m->role, "user") && m->tool_results_len > 0) {
            const size_t len = strlen(content);
            buf rest = {0};
            size_t at = 0;
            for (int k = 0; k < m->tool_results_len; k++) {
                const tool_result_span *sp = &m->tool_results[k];
                ornith_items_push(items, ORNITH_ITEM_TOOL, i, sp->content ? sp->content : "", NULL);
                if (sp->begin > at && sp->begin <= len) buf_append(&rest, content + at, sp->begin - at);
                if (sp->end > at) at = sp->end <= len ? sp->end : len;
            }
            if (at < len) buf_append(&rest, content + at, len - at);
            if (rest.len && !ornith_text_is_blank(rest.ptr))
                ornith_items_push(items, ORNITH_ITEM_USER, i, NULL, buf_take(&rest));
            else
                buf_free(&rest);
        } else if (!strcmp(m->role, "user")) {
            ornith_items_push(items, ORNITH_ITEM_USER, i, content, NULL);
        } else if (!strcmp(m->role, "assistant")) {
            ornith_items_push(items, ORNITH_ITEM_ASSISTANT, i, NULL, NULL);
        }
    }
}

/* last_query_index: the last user item that is not a bare <tool_response>
 * wrapper; without one, 0, or the last index once the list has more than 51
 * items (template lines 222-240). */
static int ornith_last_query(const ornith_items *items) {
    static const char open[] = "<tool_response>";
    static const char close[] = "</tool_response>";
    for (int k = items->len - 1; k >= 0; k--) {
        if (items->v[k].kind != ORNITH_ITEM_USER) continue;
        buf t = {0};
        append_trimmed_text(&t, items->v[k].text);
        const bool wrapped = t.len >= strlen(open) + strlen(close) &&
                             !strncmp(t.ptr, open, strlen(open)) &&
                             !strcmp(t.ptr + t.len - strlen(close), close);
        buf_free(&t);
        if (!wrapped) return k;
    }
    const int last = items->len - 1;
    return last > 50 ? last : 0;
}

static bool ornith_has(const char *hay, const char *needle) {
    return strstr(hay, needle) != NULL;
}

/* The template's tool-error test (lines 413-425) on the trimmed result.  The
 * head is the first 120 characters (code points, as Python counts them) of
 * the lowercased text; only ASCII is lowercased, and every signal is ASCII. */
static bool ornith_tool_result_failed(const char *text) {
    const size_t n = strlen(text);
    char *lower = xmalloc(n + 1);
    size_t head_end = n, chars = 0;
    for (size_t i = 0; i < n; i++) {
        const unsigned char c = (unsigned char)text[i];
        lower[i] = (char)(c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c);
        if ((c & 0xC0u) != 0x80u) {
            if (chars == 120 && head_end == n) head_end = i;
            chars++;
        }
    }
    lower[n] = '\0';
    char *head = xstrndup(lower, head_end);
    const bool code_or_grep =
        ornith_has(lower, "throw new ") || ornith_has(lower, "throw error") ||
        ornith_has(lower, "console.error") || ornith_has(lower, "logger.error") ||
        ornith_has(lower, "logging.error") || ornith_has(head, "import ") ||
        ornith_has(head, "def ") || ornith_has(head, "function ");
    const bool exit_code_zero = ornith_has(head, "exit code: 0") || ornith_has(head, "process exited with code 0");
    const bool error_field_ok =
        ornith_has(head, "\"error\": null") || ornith_has(head, "\"error\":null") ||
        ornith_has(head, "\"error\": false") || ornith_has(head, "\"error\":false") ||
        ornith_has(head, "\"error\": \"\"") || ornith_has(head, "\"error\":\"\"");
    const bool strong =
        (ornith_has(head, "\"error\":") && !error_field_ok) ||
        ornith_has(head, "\"status\": \"error\"") || ornith_has(head, "\"status\":\"error\"") ||
        ornith_has(head, "traceback (most recent call last):") || ornith_has(head, "command not found") ||
        ornith_has(head, "invalid syntax") || ornith_has(head, "fatal:") ||
        ((ornith_has(head, "exit code: ") || ornith_has(head, "process exited with code")) && !exit_code_zero) ||
        !strncmp(head, "exception:", 10) || !strncmp(head, "failed to ", 10);
    const bool weak = ornith_has(head, "error:") || ornith_has(head, "err!");
    const bool weak_suppressed = ornith_has(head, "$ ") || ornith_has(head, "took ") || chars >= 600;
    free(head);
    free(lower);
    return !code_or_grep && (strong || (weak && !weak_suppressed));
}

/* One tool result inside the open user turn: trimmed body, then the
 * template's warning after one failure or a run of failures. */
static void append_ornith_tool_response(buf *out, const char *trimmed, int failures) {
    buf_puts(out, "\n<tool_response>\n");
    append_glm_tag_body_text(out, trimmed, "</tool_response>");
    if (failures >= 2) {
        buf_printf(out, "\n\n\xe2\x9a\xa0\xef\xb8\x8f SYSTEM WARNING: %d consecutive tool errors detected. "
                        "Your previous approach is incorrect. You MUST use a fundamentally different "
                        "approach or corrected arguments.", failures);
    } else if (failures == 1) {
        buf_puts(out, "\n\n\xe2\x9a\xa0\xef\xb8\x8f SYSTEM WARNING: The previous tool call returned an error. "
                      "Diagnose the failure and retry with completely corrected arguments.");
    }
    buf_puts(out, "\n</tool_response>");
}

/* One assistant turn: content trimmed (no trailing-whitespace carve-out),
 * the reasoning block unless preserve_thinking is false and the turn comes
 * before the last user query.  Content that already starts with a think tag
 * is copied as Qwen3.8 copies it (spec §5), and sampled tool text replays. */
static void append_ornith_assistant_message(buf *out, const chat_msg *m, bool keep_think) {
    const char *content = m->content ? m->content : "";
    const char *reasoning = m->reasoning ? m->reasoning : "";
    buf body = {0};
    append_trimmed_text(&body, content);
    buf_puts(out, "<|im_start|>assistant\n");
    if (keep_think && !text_starts_with_think_tag(content)) {
        buf_puts(out, "<think>\n");
        append_trimmed_text(out, reasoning);
        /* sampled tool text already carries what followed the model's </think> */
        const bool sampled_after_think = body.len == 0 && reasoning[0] &&
            m->calls.raw_tool_text && m->calls.raw_tool_text[0];
        buf_puts(out, sampled_after_think ? "\n</think>" : "\n</think>\n\n");
    }
    buf_append(out, body.ptr ? body.ptr : "", body.len);
    append_ornith_tool_calls_text(out, &m->calls, body.len > 0);
    buf_puts(out, "<|im_end|>\n");
    buf_free(&body);
}

/* The conversation from msgs[first..], emitting only messages at index
 * >= start.  The failure count and the last user query are computed from
 * `first`, so a live tail renders the same bytes as the full prompt's end. */
static void append_ornith_conversation(buf *out, const chat_msgs *msgs, int first, int start,
                                       bool think, bool preserve_thinking) {
    ornith_items items = {0};
    ornith_items_build(&items, msgs, first);
    const int last_query = preserve_thinking ? -1 : ornith_last_query(&items);
    int failures = 0;
    bool tool_open = false;
    bool pending_assistant = false;
    for (int k = 0; k < items.len; k++) {
        const ornith_item *it = &items.v[k];
        const bool emit = it->msg >= start;
        if (it->kind == ORNITH_ITEM_TOOL) {
            buf body = {0};
            append_trimmed_text(&body, it->text);
            const char *trimmed = body.ptr ? body.ptr : "";
            failures = ornith_tool_result_failed(trimmed) ? failures + 1 : 0;
            if (emit) {
                if (!tool_open) buf_puts(out, "<|im_start|>user");
                append_ornith_tool_response(out, trimmed, failures);
                tool_open = true;
                pending_assistant = true;
            }
            buf_free(&body);
            continue;
        }
        if (tool_open) {
            buf_puts(out, "<|im_end|>\n");
            tool_open = false;
        }
        if (it->kind == ORNITH_ITEM_USER) failures = 0;
        if (!emit) continue;
        if (it->kind == ORNITH_ITEM_ASSISTANT) {
            append_ornith_assistant_message(out, &msgs->v[it->msg], preserve_thinking || k > last_query);
            pending_assistant = false;
        } else {
            buf_puts(out, it->kind == ORNITH_ITEM_SYSTEM ? "<|im_start|>system\n" : "<|im_start|>user\n");
            append_trimmed_text(out, it->text);
            buf_puts(out, "<|im_end|>\n");
            if (it->kind == ORNITH_ITEM_USER) pending_assistant = true;
        }
    }
    if (tool_open) buf_puts(out, "<|im_end|>\n");
    if (pending_assistant) append_qwen_generation_prompt(out, think);
    ornith_items_free(&items);
}
```

In `render_ornith_chat_prompt_text`, replace

```c
    buf_free(&sc);
    /* Task 4 replaces this Qwen3.8 body with the Ornith conversation */
    append_qwen_conversation(&out, msgs, head, tool_orders, think);
    return buf_take(&out);
```

with

```c
    buf_free(&sc);
    (void)tool_orders;   /* Ornith keeps the call's argument order */
    append_ornith_conversation(&out, msgs, head, head, think, opts->preserve_thinking);
    return buf_take(&out);
```

- [ ] **Step 5: The Ornith live tool tail**

In `render_qwen_live_tool_tail`, replace

```c
    buf out = {0};
    buf_puts(&out, "<|im_end|>\n");
    append_qwen_conversation(&out, msgs, start, tool_orders, ds4_think_mode_enabled(think_mode));
    return buf_take(&out);
```

with

```c
    buf out = {0};
    buf_puts(&out, "<|im_end|>\n");
    if (server_qwen_is_ornith()) {
        /* the tool-error count and the last user query depend on the turns
         * before the tail: walk from the first non-system message, emit from
         * start (template defaults; live tails carry no kwargs) */
        int head = 0;
        while (msgs && head < msgs->len && role_is_system(msgs->v[head].role)) head++;
        append_ornith_conversation(&out, msgs, head, start > head ? start : head,
                                   ds4_think_mode_enabled(think_mode), true);
        return buf_take(&out);
    }
    append_qwen_conversation(&out, msgs, start, tool_orders, ds4_think_mode_enabled(think_mode));
    return buf_take(&out);
```

- [ ] **Step 6: Run the unit tests and the full golden set**

```bash
make ds4_test ds4_agent_test
./ds4_test --server
./ds4_agent_test
make test-ornith-render
```

Expected: `ds4 tests: ok`, the agent tests pass, 26 `ok` lines and `ornith render: PASS`. On a mismatch, read the unified diff: the golden is the authority; fix the renderer, never the golden.

- [ ] **Step 7: Build every target and run the Qwen fast gate**

```bash
make ds4 ds4-server ds4-bench ds4-eval ds4-agent && make cpu
# make cpu relinks ./ds4, ./ds4-server, ./ds4-agent, ./ds4-bench and ./ds4-eval as CPU-only
# binaries; rebuild the Metal ones before any model run or run.sh (it only runs make ds4-server)
rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent && make ds4 ds4-server ds4-bench ds4-eval ds4-agent
make test-qwen4-kernels test-qwen4-q2
caffeinate -i -s speed-bench/qwen-regression/run.sh fast 2>&1 | tail -5
```

Expected: clean builds, kernel tests pass, `qwen_gate: PASS`.

- [ ] **Step 8: Update the spec (deviations 1-4)**

In `docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md` §5, replace the block

```
- **Chat rendering.**
  - Reuse the Qwen3.8 ChatML and XML tool-call renderer.
  - Ornith differences, gated by family: thinking on by default with the
    generation prompt opened by `<think>\n`, thinking off rendered as
    `<think>\n\n</think>\n\n`, and the tool schema placement of the embedded
    template.
  - The Qwen3.8 "Reasoning effort is set to xhigh" system line is not added.
    It is added in three places (`encode_chat_prompt` through
    `qwen4_chat_system`, the agent's system tokens, and the server's
    `render_qwen_chat_prompt_text`). All three get the text from
    `ds4_qwen4_reasoning_effort_text()`, which gains a family check and returns
    NULL for Ornith. For Qwen3.8 the output does not change.
```

with

```
- **Chat rendering.**
  - Reuse the Qwen3.8 ChatML turns, XML tool-call syntax, parser and live
    continuation tails (`SERVER_MODEL_SYNTAX_QWEN`). The generation prompt
    (`<think>\n`, or `<think>\n\n</think>\n\n` with thinking off) already
    equals the embedded template's.
  - The embedded template (froggeric v22.4.1) differs in the system turn,
    tool results and assistant history. ds4-server renders an Ornith flavor,
    chosen once at startup from the engine, that equals the template on the
    golden set in `tests/ornith/chat/golden/` (`make test-ornith-render`):
    - a "terse" block appended to the system turn (kwarg `terse`, default
      true; its lead line depends on thinking);
    - the template's tool instructions (thinking-dependent) and `tool | tojson`
      spacing for tool schemas and non-string arguments, in the order given;
      Anthropic tools keep ds4's mapping `{"type": "function", "function":
      <tool as given>}` (`input_schema` is not renamed);
    - reasoning effort: an absent or null effort is the template's medium (no
      line); high/xhigh/max give the existing xhigh line and low/minimal the
      low line, the same strings Qwen3.8 uses; none/off turn thinking off.
      `ds4_qwen4_reasoning_effort_text()` is unchanged; the Ornith default
      comes from the request parsers;
    - only leading system/developer messages merge into the system turn;
      later ones render in place; Anthropic's `system` comes first;
    - tool results trimmed, with the template's tool-error warning (the
      count runs across assistant turns and into live tails);
    - assistant history trimmed; `preserve_thinking` (or
      `preserve_reasoning`) false drops reasoning before the last user query.
  - Not rendered like the template, by decision: `<|think_*|>` tags inside
    messages (plain text), unknown roles (dropped), the truncation and
    tool-suppression kwargs (ignored), `tool_call_format` json (HTTP 400),
    think tags inside assistant content (copied), the `thinking`/`reasoning`
    history fields (only `reasoning_content`), closing-sentinel escaping and
    sampled tool-text replay (kept from Qwen3.8), a generation prompt only
    when an assistant turn is pending, ASCII-only trimming, and numbers
    printed as written by tojson.
  - CLI and agent: `encode_chat_prompt` renders the Ornith system turn with
    the terse block; the frontends' default think mode stands for "no effort
    given" (medium), `--think-max` gives the xhigh line
    (`ds4_engine_reasoning_effort_text()`), and `ds4_think_mode_for_context`
    does not clamp max for Ornith. The agent keeps the Qwen3.8 tools prompt
    and adds no terse block.
```

In §8 gate 1, replace

```
5. **Chat.** ds4 renderings match jinja2 renderings of the embedded template
   for a fixed conversation set: system prompt, tools, multi-turn with tool
   results, thinking on and off.
```

with

```
5. **Chat.** ds4 renderings match jinja2 renderings of the embedded template
   for a fixed conversation set: system prompt, tools, multi-turn with tool
   results, thinking on and off (`tests/ornith/chat/`: 26 goldens rendered
   from the GGUF's template, compared by `make test-ornith-render`).
```

- [ ] **Step 9: Commit**

```bash
git add ds4_server.c docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md
git commit -m "ds4-server: Ornith conversation body and live tails

Tool results are trimmed and carry the template's tool-error warning,
counted across assistant turns and into live tool tails; assistant
history is trimmed and preserve_thinking false drops reasoning before
the last user query; later system messages render in place; tool-call
arguments keep their order with tojson spacing; Anthropic tool_result
blocks render like the OpenAI tool messages they stand for.  All 26
goldens match.  The spec's chat section now lists the real template
differences and the effort rule.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- ds4_server.c docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md
```

---

### Task 5: Disk KV payload with MTP state

**Files:**
- Modify: `ds4_qwen35moe.inc` (after `qwen35_graph_state_swap`)
- Modify: `ds4.c` (`ds4_session_payload_bytes`, the qwen4 payload block, `ds4_session_save_payload`, `ds4_session_load_payload`)
- Modify: `tests/ds4_test.c` (`--qwen35-payloads`)
- Modify: `ds4_server.c` (KV-cache model-id unit test)
- Modify: spec §3 (no silent fallthrough), §5 (disk KV), §8 (gate 1 item 4)

**Interfaces:**
- Consumes (M2): `g->mtp_h`, `g->mtp_h_pos0`, `g->mtp_h_rows`, `g->mtp_pos`, `g->layer_k_cache[DS4_N_LAYER - 1]` / `layer_v_cache[...]` (allocated only with `--mtp`), `g->snap_valid`, `qwen35_graph_reset`; session fields `glm_mtp_have`, `glm_mtp_have2`, `mtp_draft_valid`, `qwen35_graph_ready` (M1). Task 2's `ds4_encode_chat_prompt` for Ornith.
- Produces (`ds4_qwen35moe.inc`):
  - `static bool qwen35_graph_h_last(ds4_qwen4_gpu_graph *g, float *out);` reads h_{g->pos-1} = mtp_h row (g->pos - g->mtp_h_pos0); fails when `!g->mtp_h`, `g->pos < g->mtp_h_pos0` or the row index exceeds `g->mtp_h_rows`; synchronizes the GPU first.
  - `static bool qwen35_graph_set_h_last(ds4_qwen4_gpu_graph *g, const float *h);` writes row 0 := h, then `g->mtp_h_pos0 = g->pos`, `g->mtp_h_rows = 0` (the caller sets `g->pos` first).
- Produces (`ds4.c`): `#define DS4_QWEN35_PAYLOAD_TAG 0x51573501u`; payload header `h[6] = DS4_N_EMBD`, `h[7] = rows`, `h[8] = DS4_N_LAYER`, `h[9] = DS4_N_HEAD_DIM`, `h[10] = MTP present (0/1)`, `h[11] = DS4_N_VOCAB`, `h[12] = tag`; body: tokens, logits, `u32 mtp_rows`, [h carry, E floats, iff MTP], per layer 0..40 (GDN state + history, or trunk K/V rows [0, rows), or MTP K/V rows [0, mtp_rows) iff MTP), pos3 rows (16 bytes each). Rule: the carry and MTP rows are written iff the session has MTP allocated (`g->mtp_h != NULL`).

- [ ] **Step 1: Write the failing payload test**

In `tests/ds4_test.c`, directly above `static void test_session_snapshot_roundtrip(void) {`:

```c
/* Ornith disk-KV payloads (M3).  A payload staged from the prefill progress
 * callback at a chunk boundary restores into another session with the logits
 * and recurrent state of an independent prefill, also after a cancelled
 * prefill; a Qwen3.8-tagged header and a payload whose MTP presence differs
 * from the session's are refused before anything is written; a truncated
 * payload leaves no sampleable checkpoint.  Run it with DS4_TEST_GLM_MTP=1
 * too: the MTP rows and the hidden-state carry travel with the payload. */
static void test_qwen35_payloads(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine || !ds4_engine_is_qwen35moe(engine)) {
        puts("qwen35-payloads: Ornith model required, skipped");
        return;
    }
    char *saved_chunk = test_save_env("DS4_QWEN35_PREFILL_CHUNK");
    setenv("DS4_QWEN35_PREFILL_CHUNK", "128", 1);
    ds4_session *live = NULL, *reference = NULL, *restored = NULL;
    ds4_tokens prompt = {0};
    ds4_session_snapshot snap = {0};
    buf text = {0};
    char err[192] = {0};
    for (int i = 0; i < 200; i++) buf_puts(&text, "The harbor records the weather and shipping schedules. ");
    ds4_encode_chat_prompt(engine, NULL, text.ptr, DS4_THINK_NONE, &prompt);
    buf_free(&text);
    TEST_ASSERT(prompt.len > 512);
    TEST_ASSERT(ds4_session_create(&live, engine, 1024) == 0);
    TEST_ASSERT(ds4_session_create(&reference, engine, 1024) == 0);
    TEST_ASSERT(ds4_session_create(&restored, engine, 1024) == 0);
    if (!live || !reference || !restored || prompt.len <= 512) goto cleanup;

    /* 1. chunk-boundary payloads, as the server's continued checkpoints */
    for (int round = 0; round < 3; round++) {
        test_qwen_prefill_checkpoint capture = {.session = live, .cancel = round == 2};
        if (round == 2) ds4_session_invalidate(live);
        ds4_tokens target = prompt;
        target.len = round == 1 ? 512 : 256;
        ds4_session_set_progress(live, test_qwen_capture_prefill, &capture);
        ds4_session_set_cancel(live, test_qwen_cancel_after_checkpoint, &capture);
        const int rc = ds4_session_sync(live, &target, err, sizeof(err));
        ds4_session_set_progress(live, NULL, NULL);
        ds4_session_set_cancel(live, NULL, NULL);
        TEST_ASSERT(rc == (capture.cancel ? DS4_SESSION_SYNC_INTERRUPTED : 0));
        TEST_ASSERT(capture.captured && capture.payload.path);
        TEST_ASSERT(capture.pos == (round == 1 ? 384 : 128));
        if (!capture.payload.path) continue;
        ds4_session_invalidate(reference);
        ds4_tokens prefix = prompt;
        prefix.len = capture.pos;
        TEST_ASSERT(ds4_session_sync(reference, &prefix, err, sizeof(err)) == 0);
        FILE *fp = fopen(capture.payload.path, "rb");
        TEST_ASSERT(fp != NULL);
        if (fp) {
            /* header word 10 = MTP presence; with MTP, the chunk's MTP rows
             * were written before the progress report (mtp_rows == pos) */
            uint32_t h[13], mtp_rows = 0;
            const bool mtp = ds4_engine_mtp_draft_tokens(engine) > 1;
            TEST_ASSERT(fread(h, sizeof(uint32_t), 13, fp) == 13);
            TEST_ASSERT(h[10] == (mtp ? 1u : 0u));
            TEST_ASSERT(fseeko(fp, (off_t)(13 + capture.pos + ds4_engine_vocab_size(engine)) * 4, SEEK_SET) == 0);
            TEST_ASSERT(fread(&mtp_rows, sizeof(mtp_rows), 1, fp) == 1);
            TEST_ASSERT(mtp_rows == (mtp ? (uint32_t)capture.pos : 0u));
            rewind(fp);
            const int load = ds4_session_load_payload(restored, fp, capture.payload.bytes, err, sizeof(err));
            if (load) fprintf(stderr, "ds4-test: Ornith payload load: %s\n", err);
            TEST_ASSERT(load == 0);
            fclose(fp);
            TEST_ASSERT(ds4_session_pos(restored) == capture.pos);
            test_qwen_prefill_scores_equal(restored, reference);
            if (capture.cancel) test_qwen_prefill_scores_equal(live, reference);
            TEST_ASSERT(ds4_session_sync(restored, &target, err, sizeof(err)) == 0);
            if (capture.cancel) TEST_ASSERT(ds4_session_sync(live, &target, err, sizeof(err)) == 0);
            test_qwen_prefill_scores_equal(restored, live);
        }
        ds4_session_payload_file_free(&capture.payload);
    }

    /* 2. refusals before any write leave the session as it was */
    TEST_ASSERT(ds4_session_save_snapshot(reference, &snap, err, sizeof(err)) == 0);
    if (!snap.ptr) goto cleanup;
    TEST_ASSERT(snap.len == ds4_session_payload_bytes(reference));
    const int before = ds4_session_argmax(restored);
    TEST_ASSERT(before >= 0);
    for (int variant = 0; variant < 2; variant++) {
        uint8_t *copy = malloc(snap.len);
        TEST_ASSERT(copy != NULL);
        if (!copy) break;
        memcpy(copy, snap.ptr, snap.len);
        const size_t off = (variant == 0 ? 12u : 10u) * sizeof(uint32_t);
        uint32_t word;
        memcpy(&word, copy + off, sizeof(word));
        word = variant == 0 ? 0x51573802u : (word ^ 1u);   /* a Qwen3.8 tag; the other MTP presence */
        memcpy(copy + off, &word, sizeof(word));
        FILE *fp = tmpfile();
        TEST_ASSERT(fp != NULL);
        if (fp) {
            TEST_ASSERT(fwrite(copy, 1, snap.len, fp) == snap.len);
            rewind(fp);
            TEST_ASSERT(ds4_session_load_payload(restored, fp, snap.len, err, sizeof(err)) != 0);
            fclose(fp);
            fprintf(stderr, "ds4-test: refused variant %d: %s\n", variant, err);
            TEST_ASSERT(strstr(err, variant == 0 ? "different model family" : "--mtp") != NULL);
            TEST_ASSERT(ds4_session_argmax(restored) == before);
        }
        free(copy);
    }

    /* 3. a truncated payload leaves no sampleable checkpoint */
    {
        const size_t bytes = 13u * sizeof(uint32_t) + (size_t)ds4_session_pos(reference) * sizeof(uint32_t) + 17u;
        TEST_ASSERT(bytes < snap.len);
        FILE *fp = tmpfile();
        TEST_ASSERT(fp != NULL);
        if (fp) {
            TEST_ASSERT(fwrite(snap.ptr, 1, bytes, fp) == bytes);
            rewind(fp);
            TEST_ASSERT(ds4_session_load_payload(restored, fp, snap.len, err, sizeof(err)) != 0);
            fclose(fp);
            TEST_ASSERT(ds4_session_argmax(restored) == -1);
        }
    }
cleanup:
    ds4_session_snapshot_free(&snap);
    ds4_session_free(restored);
    ds4_session_free(reference);
    ds4_session_free(live);
    ds4_tokens_free(&prompt);
    test_restore_env("DS4_QWEN35_PREFILL_CHUNK", saved_chunk);
}
```

In `test_entries`, directly after the `{"--qwen-kv-grow", ...}` line:

```c
    {"--qwen35-payloads", "qwen35-payloads", "Ornith disk-KV payloads restore; foreign, MTP-mismatched and truncated ones are refused", test_qwen35_payloads},
```

In `ds4_server.c`, directly above `static void ds4_server_unit_tests_run(void) {`:

```c
/* A KV-cache file carries the model id: Qwen3.8 (5) and Ornith (7)
 * checkpoints of the same rendered text never match each other. */
static void test_kv_cache_lookup_separates_qwen38_and_ornith(void) {
    char tmpl[] = "/tmp/ds4-kv-qwen-ornith-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;
    const char *text = "<|im_start|>system\nshared rendered prefix";
    const char *prompt = "<|im_start|>system\nshared rendered prefix and tail";
    test_kv_text_stub_file_model(dir, text, 7, KV_REASON_COLD, 512, 0);
    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    TEST_ASSERT(ds4_kvstore_find_text_prefix(&kc, prompt, 5, 2, 32768) < 0);
    const int idx = ds4_kvstore_find_text_prefix(&kc, prompt, 7, 2, 32768);
    TEST_ASSERT(idx >= 0 && kc.entry[idx].model_id == 7);
    kv_cache_close(&kc);
    char sha[41];
    sha1_bytes_hex(text, strlen(text), sha);
    char name[44];
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    char *path = path_join(dir, name);
    unlink(path);
    free(path);
    rmdir(dir);
}
```

At the end of `ds4_server_unit_tests_run`, after `    test_ornith_tool_turn_visible_text_prefixes_next_render();`:

```c
    test_kv_cache_lookup_separates_qwen38_and_ornith();
```

- [ ] **Step 2: Run the tests and confirm the failure**

GPU window. One model process at a time.

```bash
make ds4_test
./ds4_test --server 2>&1 | tail -2
export DS4_ORNITH_MODEL=$HOME/orca/workspaces/ds4-metal-data/gguf/ornith/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf
ps -axo pid,stat,comm | awk '$2 ~ /^(E|U)/'
caffeinate -i -s env DS4_TEST_MODEL=${DS4_ORNITH_MODEL:?set DS4_ORNITH_MODEL} ./ds4_test --qwen35-payloads 2>&1 | tail -5
```

Expected: `ds4 tests: ok` (the model-id test passes already: the store layer separates families); the payload test fails, typically with a crash or an exit in the DeepSeek payload code (`raw_cap` division) during the first `ds4_session_stage_payload`.

- [ ] **Step 3: The MTP hidden-state carry in `ds4_qwen35moe.inc`**

Directly after the closing brace of `qwen35_graph_state_swap` (M2):

```c
/* h_{g->pos - 1}: the post-output_norm trunk row the next MTP row pairs
 * with, held in mtp_h row (g->pos - g->mtp_h_pos0).  The disk payload saves
 * it so a restored session drafts exactly like the uninterrupted one. */
static bool qwen35_graph_h_last(ds4_qwen4_gpu_graph *g, float *out) {
    if (!g->mtp_h || g->pos < g->mtp_h_pos0) return false;
    const uint32_t row = g->pos - g->mtp_h_pos0;
    if (row > g->mtp_h_rows) return false;
    if (ds4_gpu_synchronize() == 0) return false;
    return ds4_gpu_tensor_read(g->mtp_h, (uint64_t)row * DS4_N_EMBD * sizeof(float), out,
                               (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
}

/* Install a restored carry: row 0 := h with mtp_h_pos0 = g->pos (the caller
 * sets g->pos first) and no forward rows yet, so the next forward pairs its
 * first MTP row with h. */
static bool qwen35_graph_set_h_last(ds4_qwen4_gpu_graph *g, const float *h) {
    if (!g->mtp_h) return false;
    if (ds4_gpu_tensor_write(g->mtp_h, 0, h, (uint64_t)DS4_N_EMBD * sizeof(float)) == 0) return false;
    g->mtp_h_pos0 = g->pos;
    g->mtp_h_rows = 0;
    return true;
}
```

- [ ] **Step 4: Payload size, save and load in `ds4.c`**

Directly after the forward declaration block

```c
#ifdef DS4_HAS_QWEN4_GPU
static uint64_t qwen4_payload_tensor_bytes(uint32_t rows, uint32_t mtp_rows, bool fp8, bool q4);
#endif
```

add

```c
#ifdef DS4_HAS_QWEN4_METAL
static uint64_t qwen35_payload_body_bytes(uint32_t rows, uint32_t mtp_rows, bool mtp);
#endif
```

At the top of `ds4_session_payload_bytes`, directly after `uint64_t ds4_session_payload_bytes(ds4_session *s) {`:

```c
#ifdef DS4_HAS_QWEN4_METAL
    if (s && !s->distributed && ds4_session_is_qwen35(s)) {
        const ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
        if (!s->qwen35_graph_ready || !s->checkpoint_valid || g->pos != (uint32_t)s->checkpoint.len ||
            g->kv_fp8 || g->kv_q4) return 0;
        const uint32_t rows = (uint32_t)s->checkpoint.len;
        const bool mtp = g->mtp_h != NULL;
        const uint32_t mtp_rows = mtp ? (g->mtp_pos < rows ? g->mtp_pos : rows) : 0u;
        return (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t) +
               qwen35_payload_body_bytes(rows, mtp_rows, mtp);
    }
#endif
```

In `ds4_session_payload_bytes`, replace

```c
#else
    const ds4_gpu_graph *g = &s->graph;
    uint64_t bytes = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
```

with

```c
#else
    ds4_qwen35_not_reached("session payload bytes");
    const ds4_gpu_graph *g = &s->graph;
    uint64_t bytes = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
```

After `qwen4_session_load_payload`, replace

```c
    s->qwen4_rewound = false;
    return 0;
}
#endif

int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen) {
```

with

```c
    s->qwen4_rewound = false;
    return 0;
}

#ifdef DS4_HAS_QWEN4_METAL
/* Ornith session payload.  The header mirrors Qwen3.8's with its own tag;
 * h[6] is the embedding width and h[10] whether the session carries the MTP
 * block (1) or not (0).  Body: tokens, logits, u32 mtp_rows, then with MTP
 * the trunk hidden-state carry h_{rows-1} (E floats); per layer the GDN state
 * and conv history, the trunk attention K/V rows [0, rows), with MTP the MTP
 * block's K/V rows [0, mtp_rows); last the rope positions of the rows
 * (16 bytes each), which a later MTP pass below g->pos may read.  F16 KV
 * only: an FP8/Q4 KV mode needs its own layout before it saves. */
#define DS4_QWEN35_PAYLOAD_TAG 0x51573501u

static uint64_t qwen35_payload_body_bytes(uint32_t rows, uint32_t mtp_rows, bool mtp) {
    uint64_t bytes = (uint64_t)rows * sizeof(uint32_t) + (uint64_t)DS4_N_VOCAB * sizeof(float);
    bytes += sizeof(uint32_t);
    if (mtp) bytes += (uint64_t)DS4_N_EMBD * sizeof(float);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (ds4_qwen35_layer_is_nextn(il) && !mtp) continue;
        if (ds4_qwen35_layer_is_attention(il)) {
            bytes += 2u * qwen4_payload_kv_bytes(ds4_qwen35_layer_is_nextn(il) ? mtp_rows : rows);
        } else {
            bytes += qwen4_payload_lin_state_bytes() + qwen4_payload_lin_hist_bytes();
        }
    }
    return bytes + (uint64_t)rows * 16u;
}

static int qwen35_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen) {
    ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
    const uint32_t rows = (uint32_t)s->checkpoint.len;
    if (!s->qwen35_graph_ready || g->pos != rows) {
        payload_set_err(err, errlen, "Ornith snapshot requires a synchronized session");
        return 1;
    }
    if (g->kv_fp8 || g->kv_q4) {
        payload_set_err(err, errlen, "Ornith checkpoints support the F16 KV cache only");
        return 1;
    }
    if (ds4_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator before Ornith snapshot");
        return 1;
    }
    const bool mtp = g->mtp_h != NULL;
    const uint32_t mtp_rows = mtp ? (g->mtp_pos < rows ? g->mtp_pos : rows) : 0u;
    float *h_last = NULL;
    if (mtp) {
        h_last = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
        if (!qwen35_graph_h_last(g, h_last)) {
            free(h_last);
            payload_set_err(err, errlen, "failed to read the Ornith MTP hidden-state carry");
            return 1;
        }
    }
    const uint32_t header[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC,
        DS4_SESSION_PAYLOAD_VERSION,
        (uint32_t)s->ctx_size,
        s->prefill_cap,
        g->ctx_cap,
        g->ctx_cap,
        (uint32_t)DS4_N_EMBD,
        rows,
        DS4_N_LAYER,
        DS4_N_HEAD_DIM,
        mtp ? 1u : 0u,
        DS4_N_VOCAB,
        DS4_QWEN35_PAYLOAD_TAG,
    };
    int rc = 0;
    for (uint32_t i = 0; rc == 0 && i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++)
        rc = payload_write_u32(fp, header[i], err, errlen);
    for (int i = 0; rc == 0 && i < s->checkpoint.len; i++)
        rc = payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen);
    if (rc == 0) rc = payload_write_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float), err, errlen);
    if (rc == 0) rc = payload_write_u32(fp, mtp_rows, err, errlen);
    if (rc == 0 && mtp) rc = payload_write_bytes(fp, h_last, (uint64_t)DS4_N_EMBD * sizeof(float), err, errlen);
    free(h_last);
    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
        if (ds4_qwen35_layer_is_nextn(il) && !mtp) continue;
        if (ds4_qwen35_layer_is_attention(il)) {
            const uint64_t kvb = qwen4_payload_kv_bytes(ds4_qwen35_layer_is_nextn(il) ? mtp_rows : rows);
            rc = payload_write_tensor_span(fp, g->layer_k_cache[il], 0, kvb, buf, DS4_SESSION_IO_CHUNK, err, errlen);
            if (rc == 0)
                rc = payload_write_tensor_span(fp, g->layer_v_cache[il], 0, kvb, buf, DS4_SESSION_IO_CHUNK, err, errlen);
        } else {
            rc = payload_write_tensor_span(fp, g->layer_lin_state[il], 0, qwen4_payload_lin_state_bytes(),
                                           buf, DS4_SESSION_IO_CHUNK, err, errlen);
            if (rc == 0)
                rc = payload_write_tensor_span(fp, g->layer_lin_hist[il], 0, qwen4_payload_lin_hist_bytes(),
                                               buf, DS4_SESSION_IO_CHUNK, err, errlen);
        }
    }
    if (rc == 0)
        rc = payload_write_tensor_span(fp, g->pos3, 0, (uint64_t)rows * 16u, buf, DS4_SESSION_IO_CHUNK, err, errlen);
    free(buf);
    return rc;
}

static int qwen35_session_load_payload(ds4_session *s, FILE *fp, const uint32_t *h, uint64_t *remaining,
                                       char *err, size_t errlen) {
    ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
    if (!s->qwen35_graph_ready) {
        payload_set_err(err, errlen, "Ornith graph is not ready for restore");
        return 1;
    }
    const bool mtp = g->mtp_h != NULL;
    const uint32_t rows = h[7];
    if (h[12] != DS4_QWEN35_PAYLOAD_TAG || h[6] != DS4_N_EMBD || h[8] != DS4_N_LAYER ||
        h[9] != DS4_N_HEAD_DIM || h[10] > 1u || h[11] != DS4_N_VOCAB) {
        payload_set_err(err, errlen, "KV checkpoint was written by a different model family or shape");
        return 1;
    }
    if (h[10] != (mtp ? 1u : 0u)) {
        payload_set_err(err, errlen, mtp ?
            "KV checkpoint was saved without --mtp; this Ornith session keeps MTP state" :
            "KV checkpoint was saved with --mtp; this Ornith session has no MTP state");
        return 1;
    }
    if (g->kv_fp8 || g->kv_q4) {
        payload_set_err(err, errlen, "Ornith checkpoints support the F16 KV cache only");
        return 1;
    }
    if (rows > g->ctx_cap || rows > (uint32_t)s->ctx_size) {
        payload_set_err(err, errlen, "KV checkpoint is longer than this session's context");
        return 1;
    }
    token_vec new_checkpoint = {0};
    for (uint32_t i = 0; i < rows; i++) {
        uint32_t tok;
        if (payload_read_u32(fp, &tok, remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (tok >= DS4_N_VOCAB) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint token id is outside the vocabulary");
            return 1;
        }
        token_vec_push(&new_checkpoint, (int)tok);
    }
    /* From the first write on, a failure leaves no reusable checkpoint, and
     * verifier state belongs to the old transcript even at the same row. */
    s->checkpoint_valid = false;
    s->mtp_draft_valid = false;
    s->glm_mtp_have = 0;
    s->glm_mtp_have2 = false;
    g->snap_valid = false;
    int rc = payload_read_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float), remaining, err, errlen);
    if (rc == 0 && ds4_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator before Ornith restore");
        rc = 1;
    }
    uint32_t mtp_rows = 0;
    if (rc == 0) rc = payload_read_u32(fp, &mtp_rows, remaining, err, errlen);
    if (rc == 0 && (mtp_rows > rows || (!mtp && mtp_rows != 0))) {
        payload_set_err(err, errlen, "KV checkpoint MTP rows are invalid");
        rc = 1;
    }
    float *h_last = mtp ? xmalloc((size_t)DS4_N_EMBD * sizeof(float)) : NULL;
    if (rc == 0 && mtp)
        rc = payload_read_bytes(fp, h_last, (uint64_t)DS4_N_EMBD * sizeof(float), remaining, err, errlen);
    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
        if (ds4_qwen35_layer_is_nextn(il) && !mtp) continue;
        if (ds4_qwen35_layer_is_attention(il)) {
            const uint64_t kvb = qwen4_payload_kv_bytes(ds4_qwen35_layer_is_nextn(il) ? mtp_rows : rows);
            rc = payload_read_tensor_span(fp, g->layer_k_cache[il], 0, kvb, buf, DS4_SESSION_IO_CHUNK,
                                          remaining, err, errlen);
            if (rc == 0)
                rc = payload_read_tensor_span(fp, g->layer_v_cache[il], 0, kvb, buf, DS4_SESSION_IO_CHUNK,
                                              remaining, err, errlen);
        } else {
            rc = payload_read_tensor_span(fp, g->layer_lin_state[il], 0, qwen4_payload_lin_state_bytes(),
                                          buf, DS4_SESSION_IO_CHUNK, remaining, err, errlen);
            if (rc == 0)
                rc = payload_read_tensor_span(fp, g->layer_lin_hist[il], 0, qwen4_payload_lin_hist_bytes(),
                                              buf, DS4_SESSION_IO_CHUNK, remaining, err, errlen);
        }
    }
    if (rc == 0)
        rc = payload_read_tensor_span(fp, g->pos3, 0, (uint64_t)rows * 16u, buf, DS4_SESSION_IO_CHUNK,
                                      remaining, err, errlen);
    free(buf);
    if (rc == 0) {
        g->pos = rows;
        if (mtp) {
            if (!qwen35_graph_set_h_last(g, h_last)) {
                payload_set_err(err, errlen, "failed to restore the Ornith MTP hidden-state carry");
                rc = 1;
            } else {
                g->mtp_pos = mtp_rows;
            }
        }
    }
    free(h_last);
    if (rc != 0) {
        token_vec_free(&new_checkpoint);
        qwen35_graph_reset(g);
        s->checkpoint.len = 0;
        return 1;
    }
    token_vec_free(&s->checkpoint);
    s->checkpoint = new_checkpoint;
    s->checkpoint_valid = true;
    return 0;
}
#endif
#endif

int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen) {
```

In `ds4_session_save_payload`, replace

```c
    if (ds4_session_is_qwen4(s)) {
#ifndef DS4_HAS_QWEN4_GPU
        payload_set_err(err, errlen, "graph backend support is not compiled in");
        return 1;
#else
        return qwen4_session_save_payload(s, fp, err, errlen);
```

with

```c
#ifdef DS4_HAS_QWEN4_METAL
    if (ds4_session_is_qwen35(s)) return qwen35_session_save_payload(s, fp, err, errlen);
#endif
    if (ds4_session_is_qwen4(s)) {
#ifndef DS4_HAS_QWEN4_GPU
        payload_set_err(err, errlen, "graph backend support is not compiled in");
        return 1;
#else
        return qwen4_session_save_payload(s, fp, err, errlen);
```

and replace

```c
#else
    if (ds4_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator before snapshot");
```

with

```c
#else
    ds4_qwen35_not_reached("session payload save");
    if (ds4_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator before snapshot");
```

In `ds4_session_load_payload`, replace

```c
    if (ds4_session_is_qwen4(s)) {
#ifndef DS4_HAS_QWEN4_GPU
        payload_set_err(err, errlen, "graph backend support is not compiled in");
        return 1;
#else
        return qwen4_session_load_payload(s, fp, h, &remaining, err, errlen);
```

with

```c
#ifdef DS4_HAS_QWEN4_METAL
    if (ds4_session_is_qwen35(s)) return qwen35_session_load_payload(s, fp, h, &remaining, err, errlen);
#endif
    if (ds4_session_is_qwen4(s)) {
#ifndef DS4_HAS_QWEN4_GPU
        payload_set_err(err, errlen, "graph backend support is not compiled in");
        return 1;
#else
        return qwen4_session_load_payload(s, fp, h, &remaining, err, errlen);
```

and replace

```c
#else
    ds4_gpu_graph *g = &s->graph;
    const uint32_t saved_ctx = h[2];
```

with

```c
#else
    ds4_qwen35_not_reached("session payload load");
    ds4_gpu_graph *g = &s->graph;
    const uint32_t saved_ctx = h[2];
```

- [ ] **Step 5: Run the payload and snapshot tests, without and with MTP**

GPU window.

```bash
make ds4_test
caffeinate -i -s env DS4_TEST_MODEL=${DS4_ORNITH_MODEL:?set DS4_ORNITH_MODEL} ./ds4_test --session-snapshot --qwen35-payloads 2>&1 | tail -8
caffeinate -i -s env DS4_TEST_MODEL=${DS4_ORNITH_MODEL:?set DS4_ORNITH_MODEL} DS4_TEST_GLM_MTP=1 ./ds4_test --session-snapshot --qwen35-payloads 2>&1 | tail -8
./ds4_test --server 2>&1 | tail -1
```

Expected: `session-snapshot: OK` and `qwen35-payloads: OK` in both runs; the MTP run prints `GLM MTP snapshot cycles=16 ...` (restored speculative cycles equal the reference) and the refused-variant lines name `different model family` and `--mtp`; `ds4 tests: ok`.

- [ ] **Step 6: Build every target, M1 session test, Qwen fast gate**

```bash
make ds4 ds4-server ds4-bench ds4-eval ds4-agent && make cpu
# make cpu relinks ./ds4, ./ds4-server, ./ds4-agent, ./ds4-bench and ./ds4-eval as CPU-only
# binaries; rebuild the Metal ones before any model run or run.sh (it only runs make ds4-server)
rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent && make ds4 ds4-server ds4-bench ds4-eval ds4-agent
make test-qwen35-session DS4_ORNITH_MODEL=${DS4_ORNITH_MODEL:?set DS4_ORNITH_MODEL} 2>&1 | tail -2
make test-qwen4-kernels test-qwen4-q2
caffeinate -i -s speed-bench/qwen-regression/run.sh fast 2>&1 | tail -5
```

Expected: `qwen35 session: ok`, kernel tests pass, `qwen_gate: PASS`.

- [ ] **Step 7: Update the spec (deviations 6, 7, 9)**

In §3, replace

```
- **No silent fallthrough.** Where a qwen4 branch is followed by a DeepSeek/GLM
  default (context estimate, one-shot generate, session create/sync/eval,
  speculative cycle, rewind), the default path dies with a clear message if it
  ever sees the Ornith family, so a missed DISPATCH branch cannot silently run
  DeepSeek code.
```

with

```
- **No silent fallthrough.** Where a qwen4 branch is followed by a DeepSeek/GLM
  default (context estimate, one-shot generate, session create/sync/eval,
  speculative cycle, disk KV payload size/save/load), the default path dies
  with a clear message if it ever sees the Ornith family, so a missed DISPATCH
  branch cannot silently run DeepSeek code. Rewind has no DeepSeek default to
  fall into: an Ornith rewind restores a verify snapshot or invalidates the
  checkpoint (section 6).
```

In §5, replace

```
- **Disk KV checkpoints.**
  - A new payload tag, `DS4_QWEN35_PAYLOAD_TAG`, covers the 30 GDN recurrent
    states and conv histories plus the KV rows of the 10 attention layers and
    the MTP block.
  - The checkpoint header records `qwen35moe`, so Qwen3.8 and Ornith
    checkpoints can never load into each other. Existing Qwen3.8 checkpoints
    are unaffected.
```

with

```
- **Disk KV checkpoints.**
  - A new payload tag, `DS4_QWEN35_PAYLOAD_TAG`, covers the 30 GDN recurrent
    states and conv histories, the KV rows of the 10 attention layers, the
    rope positions and, with `--mtp`, the MTP block's KV rows and the trunk
    hidden-state carry the next draft pairs with. The payload records whether
    it carries MTP state; a payload whose MTP presence differs from the
    session's is refused and the server prefills instead. F16 KV only.
  - The KV-cache file header records the model id (`qwen35moe` = 7) and the
    payload its own tag, so Qwen3.8 and Ornith checkpoints can never load
    into each other. Existing Qwen3.8 checkpoints are unaffected. The routed
    quant byte stays 2 for both Ornith tiers (Q5_K in layer 0), so a 25G
    server accepts a 23G checkpoint of the same text, as the Qwen3.8 IQ2
    tiers do.
```

In §8 gate 1, replace

```
4. **Sessions.** Save to disk KV, restore and continue: the output equals an
   uninterrupted run. A Qwen3.8 checkpoint is refused by Ornith and the other
   way round.
```

with

```
4. **Sessions.** Save to disk KV, restore and continue: the output equals an
   uninterrupted run. A Qwen3.8 checkpoint is refused by Ornith and the other
   way round. (M3 tests the Ornith side with a Qwen3.8-tagged payload; the
   other direction rests on the Qwen3.8 loader's exact tag check and the
   KV-cache model id, tested without a model.)
```

- [ ] **Step 8: Commit**

```bash
git add ds4_qwen35moe.inc ds4.c tests/ds4_test.c ds4_server.c docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md
git commit -m "qwen35: Ornith disk-KV payload with MTP state

DS4_QWEN35_PAYLOAD_TAG saves tokens, logits, GDN state and history, the
trunk KV rows, the rope positions and, with --mtp, the MTP KV rows and
the trunk hidden-state carry (qwen35_graph_h_last/set_h_last).  A
payload of the other family or MTP presence is refused before anything
is written; payload size, save and load now die on a missed Ornith
branch instead of running DeepSeek code.  ds4_test --qwen35-payloads and
--session-snapshot pass with and without --mtp.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- ds4_qwen35moe.inc ds4.c tests/ds4_test.c ds4_server.c docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md
```

---

### Task 6: Rewind by verify snapshot, and the batch guard

**Files:**
- Modify: `ds4.c` (`ds4_session_rewind`, `ds4_sessions_eval_batch_metal_supported`)
- Modify: `tests/ds4_test.c` (`--qwen35-rewind`), `tests/test_qwen35_session.c` (case 5)
- Modify: spec §6

**Interfaces:**
- Consumes (M2): `g->snap_valid`, `g->snap_pos` (verify pos0 + 1 after a verify, until the next forward), `qwen35_graph_state_swap(g)` (sets `g->pos = g->snap_pos`, `snap_valid = false`), `s->qwen4_verify_logits` (row 0 = after the first verify row), `g->mtp_pos`, knob `DS4_QWEN35_SPEC_FORCE_ACCEPT`.
- Produces: `ds4_session_rewind` on an Ornith session: one token back right after an accepted verify keeps the checkpoint valid with row 0's logits; any other rewind leaves `ds4_session_checkpoint_valid()` false (the caller re-syncs the kept prefix). `ds4_sessions_eval_batch` serializes Ornith sessions.

- [ ] **Step 1: Write the failing tests**

In `tests/ds4_test.c`, directly above `static void test_session_snapshot_roundtrip(void) {`:

```c
/* Ornith rewind (M3).  One token back right after an accepted verify
 * restores the after-row-0 snapshot: the checkpoint stays valid and the
 * logits and the next eval equal a fresh session's.  Any other rewind
 * invalidates, and a sync of the kept prefix replays it.  The snapshot half
 * needs DS4_TEST_GLM_MTP=1; without MTP every rewind invalidates. */
static void test_qwen35_rewind(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine || !ds4_engine_is_qwen35moe(engine)) {
        puts("qwen35-rewind: Ornith model required, skipped");
        return;
    }
    const bool mtp = ds4_engine_mtp_draft_tokens(engine) > 1;
    char *saved_force = test_save_env("DS4_QWEN35_SPEC_FORCE_ACCEPT");
    if (mtp) setenv("DS4_QWEN35_SPEC_FORCE_ACCEPT", "1", 1);
    ds4_session *live = NULL, *fresh = NULL;
    ds4_tokens prompt = {0}, replay = {0};
    char err[192] = {0};
    ds4_token_score got[8], want[8];
    ds4_chat_begin(engine, &prompt);
    ds4_chat_append_message(engine, &prompt, "user", "Count from one to ten.");
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
    TEST_ASSERT(ds4_session_create(&live, engine, 1024) == 0);
    TEST_ASSERT(ds4_session_create(&fresh, engine, 1024) == 0);
    if (!live || !fresh) goto cleanup;
    TEST_ASSERT(ds4_session_sync(live, &prompt, err, sizeof(err)) == 0);
    for (int i = 0; i < prompt.len; i++) ds4_tokens_push(&replay, prompt.v[i]);
    int last_n = 1;
    for (int step = 0; step < 3; step++) {
        const int first = ds4_session_argmax(live);
        if (mtp) {
            int acc[2] = {0, 0};
            last_n = ds4_session_eval_speculative_argmax(live, first, 2, -1, acc, 2, err, sizeof(err));
            TEST_ASSERT(last_n == 1 || last_n == 2);
            if (last_n < 1) goto cleanup;
            for (int i = 0; i < last_n; i++) ds4_tokens_push(&replay, acc[i]);
        } else {
            TEST_ASSERT(ds4_session_eval(live, first, err, sizeof(err)) == 0);
            ds4_tokens_push(&replay, first);
        }
    }
    if (mtp) TEST_ASSERT(last_n == 2);
    TEST_ASSERT(ds4_session_pos(live) == replay.len);
    const int last = replay.v[replay.len - 1];
    ds4_tokens prefix = replay;

    /* 1. one token back: the snapshot with MTP, a replay without */
    ds4_session_rewind(live, replay.len - 1);
    TEST_ASSERT(ds4_session_pos(live) == replay.len - 1);
    TEST_ASSERT(ds4_session_checkpoint_valid(live) == mtp);
    prefix.len = replay.len - 1;
    if (!mtp) TEST_ASSERT(ds4_session_sync(live, &prefix, err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_sync(fresh, &prefix, err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_top_logprobs(live, got, 8) == 8);
    TEST_ASSERT(ds4_session_top_logprobs(fresh, want, 8) == 8);
    TEST_ASSERT(got[0].id == want[0].id);
    for (int i = 0; i < 8; i++) TEST_ASSERT(fabsf(got[i].logprob - want[i].logprob) < 2e-3f);
    TEST_ASSERT(ds4_session_eval(live, last, err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_eval(fresh, last, err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_top_logprobs(live, got, 8) == 8);
    TEST_ASSERT(ds4_session_top_logprobs(fresh, want, 8) == 8);
    TEST_ASSERT(got[0].id == want[0].id);
    for (int i = 0; i < 8; i++) TEST_ASSERT(fabsf(got[i].logprob - want[i].logprob) < 2e-3f);

    /* 2. two tokens back: invalidated; the kept prefix replays on sync */
    ds4_session_rewind(live, replay.len - 2);
    TEST_ASSERT(ds4_session_pos(live) == replay.len - 2);
    TEST_ASSERT(!ds4_session_checkpoint_valid(live));
    prefix.len = replay.len - 2;
    TEST_ASSERT(ds4_session_sync(live, &prefix, err, sizeof(err)) == 0);
    ds4_session_invalidate(fresh);
    TEST_ASSERT(ds4_session_sync(fresh, &prefix, err, sizeof(err)) == 0);
    test_qwen_prefill_scores_equal(live, fresh);
    fprintf(stderr, "ds4-test: Ornith rewind checked (mtp=%d)\n", mtp);
cleanup:
    ds4_tokens_free(&replay);
    ds4_tokens_free(&prompt);
    ds4_session_free(fresh);
    ds4_session_free(live);
    test_restore_env("DS4_QWEN35_SPEC_FORCE_ACCEPT", saved_force);
}
```

In `test_entries`, directly after the `{"--qwen35-payloads", ...}` line (Task 5):

```c
    {"--qwen35-rewind", "qwen35-rewind", "Ornith rewind by verify snapshot, otherwise invalidate and replay", test_qwen35_rewind},
```

In `tests/test_qwen35_session.c`, replace the header comment's last line

```c
 *  4. a session above the native 262144-token context is refused (no YaRN). */
```

with

```c
 *  4. a session above the native 262144-token context is refused (no YaRN);
 *  5. a decode batch of two Ornith sessions takes the serialized path and
 *     equals two single evals bit for bit. */
```

and add, directly before `    ds4_session_free(fresh);` near the end of `main`:

```c
    /* 5. a decode batch of Ornith sessions is serialized: no native batch
     * path knows the family, and the result equals two single evals */
    {
        ds4_session *x = NULL, *y = NULL, *xr = NULL, *yr = NULL;
        assert(ds4_session_create(&x, engine, ctx) == 0 && ds4_session_create(&y, engine, ctx) == 0);
        assert(ds4_session_create(&xr, engine, ctx) == 0 && ds4_session_create(&yr, engine, ctx) == 0);
        sync_len(x, &tokens, 200);
        sync_len(xr, &tokens, 200);
        sync_len(y, &other, other.len);
        sync_len(yr, &other, other.len);
        ds4_decode_item items[2] = {{x, ds4_session_argmax(x)}, {y, ds4_session_argmax(y)}};
        assert(ds4_sessions_eval_batch(items, 2, err, sizeof(err)) == 0);
        assert(ds4_session_eval(xr, items[0].token, err, sizeof(err)) == 0);
        assert(ds4_session_eval(yr, items[1].token, err, sizeof(err)) == 0);
        assert(ds4_session_copy_logits(x, a, vocab) == vocab && ds4_session_copy_logits(xr, b, vocab) == vocab);
        assert(memcmp(a, b, (size_t)vocab * 4) == 0);
        assert(ds4_session_copy_logits(y, a, vocab) == vocab && ds4_session_copy_logits(yr, b, vocab) == vocab);
        assert(memcmp(a, b, (size_t)vocab * 4) == 0);
        printf("  decode batch of two: serialized, equal to single evals\n");
        ds4_session_free(yr);
        ds4_session_free(xr);
        ds4_session_free(y);
        ds4_session_free(x);
    }
```

- [ ] **Step 2: Run them and confirm the failures**

GPU window.

```bash
make ds4_test tests/test_qwen35_session
caffeinate -i -s env DS4_TEST_MODEL=${DS4_ORNITH_MODEL:?set DS4_ORNITH_MODEL} DS4_TEST_GLM_MTP=1 ./ds4_test --qwen35-rewind 2>&1 | tail -4
caffeinate -i -s ./tests/test_qwen35_session "${DS4_ORNITH_MODEL:?set DS4_ORNITH_MODEL}" 2>&1 | tail -3
```

Expected: the rewind test fails at `ds4_session_checkpoint_valid(live) == mtp` (Ornith rewinds always invalidate today). The session test either crashes in the DeepSeek batch path or passes case 5; if case 5 passes, the DeepSeek checks happened to refuse the sessions: keep the guard of Step 3 anyway (it removes the dependency on that accident) and note it in the commit message.

- [ ] **Step 3: The rewind branch and the batch guard**

In `ds4_session_rewind`, replace

```c
        s->qwen4_rewound = logit_row < 0;
    }
#endif
    if (s->checkpoint_valid && ds4_session_is_glm(s)) {
```

with

```c
        s->qwen4_rewound = logit_row < 0;
    }
#endif
#ifdef DS4_HAS_QWEN4_METAL
    if (s->checkpoint_valid && ds4_session_is_qwen35(s)) {
        /* One token back after an accepted verify: the after-row-0 GDN
         * snapshot is the state at pos and verify row 0 holds its logits.
         * Recurrent state cannot be trimmed otherwise, so any other rewind
         * leaves state_ok false and the caller re-syncs the kept prefix. */
        ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
        if (s->qwen4_verify_logits && g->snap_valid && g->snap_pos == (uint32_t)pos &&
            qwen35_graph_state_swap(g)) {
            memcpy(s->logits, s->qwen4_verify_logits, (size_t)DS4_N_VOCAB * sizeof(float));
            if (g->mtp_pos > (uint32_t)pos) g->mtp_pos = (uint32_t)pos;
            state_ok = true;
        }
    }
#endif
    if (s->checkpoint_valid && ds4_session_is_glm(s)) {
```

In `ds4_sessions_eval_batch_metal_supported`, replace

```c
        getenv("DS4_METAL_DECODE_STAGE_PROFILE") != NULL) {
        return false;
    }
#if defined(__APPLE__)
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK41)
```

with

```c
        getenv("DS4_METAL_DECODE_STAGE_PROFILE") != NULL) {
        return false;
    }
    /* Ornith sessions decode one at a time: batching is refused at open and
     * no native batch path knows the family, so never reach the DeepSeek
     * checks below. */
    if (ds4_model_is_qwen35moe()) return false;
#if defined(__APPLE__)
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK41)
```

- [ ] **Step 4: Run the tests**

GPU window.

```bash
make ds4_test tests/test_qwen35_session
caffeinate -i -s env DS4_TEST_MODEL=${DS4_ORNITH_MODEL:?set DS4_ORNITH_MODEL} DS4_TEST_GLM_MTP=1 ./ds4_test --qwen35-rewind 2>&1 | tail -3
caffeinate -i -s env DS4_TEST_MODEL=${DS4_ORNITH_MODEL:?set DS4_ORNITH_MODEL} ./ds4_test --qwen35-rewind 2>&1 | tail -3
caffeinate -i -s ./tests/test_qwen35_session "${DS4_ORNITH_MODEL:?set DS4_ORNITH_MODEL}" 2>&1 | tail -3
```

Expected: `Ornith rewind checked (mtp=1)` and `qwen35-rewind: OK`; then `(mtp=0)` and `OK`; `decode batch of two: serialized, equal to single evals` and `qwen35 session: ok`.

- [ ] **Step 5: Build every target, re-run the payload tests, Qwen fast gate**

```bash
make ds4 ds4-server ds4-bench ds4-eval ds4-agent && make cpu
# make cpu relinks ./ds4, ./ds4-server, ./ds4-agent, ./ds4-bench and ./ds4-eval as CPU-only
# binaries; rebuild the Metal ones before any model run or run.sh (it only runs make ds4-server)
rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent && make ds4 ds4-server ds4-bench ds4-eval ds4-agent
caffeinate -i -s env DS4_TEST_MODEL=${DS4_ORNITH_MODEL:?set DS4_ORNITH_MODEL} DS4_TEST_GLM_MTP=1 ./ds4_test --session-snapshot --qwen35-payloads 2>&1 | tail -3
make test-qwen4-kernels test-qwen4-q2
caffeinate -i -s speed-bench/qwen-regression/run.sh fast 2>&1 | tail -5
```

Expected: both Ornith tests `OK`, kernel tests pass, `qwen_gate: PASS`.

- [ ] **Step 6: Update the spec (deviation 8)**

In §6, replace

```
- **Rewind.** A rewind on an Ornith session restores the GDN snapshot when one
  covers the target position. Otherwise it resets the recurrent state and
  conv history together with the checkpoint, so a later sync can never reuse
  a stale prefix.
```

with

```
- **Rewind.** A rewind on an Ornith session restores the after-row-0 verify
  snapshot when it covers the target position (one token back after an
  accepted draft, which a stop token inside a verify block produces), with
  that row's logits. Otherwise it invalidates the checkpoint; the next sync
  resets the recurrent state and conv history and replays the kept prefix,
  so a stale prefix is never reused.
```

- [ ] **Step 7: Commit**

```bash
git add ds4.c tests/ds4_test.c tests/test_qwen35_session.c docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md
git commit -m "qwen35: rewind to the verify snapshot; serialize Ornith batches

One token back after an accepted verify restores M2's after-row-0 GDN
snapshot with verify row 0's logits and keeps the checkpoint; other
rewinds invalidate, and the caller's sync replays the kept prefix.  The
native batch check refuses Ornith before the DeepSeek checks.  Tests:
ds4_test --qwen35-rewind (with and without --mtp) and case 5 of
test_qwen35_session.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- ds4.c tests/ds4_test.c tests/test_qwen35_session.c docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md
```

---

### Task 7: Server and agent serve Ornith: ids, startup flavor, refusal removal

**Files:**
- Modify: `ds4_server.c` (`model_alias_disables_thinking`, `model_alias_enables_thinking`, `server_model_id_from_engine`, `server_model_alias_known`, `send_models`, `main`, unit test)
- Modify: `ds4_agent.c` (`agent_tool_syntax_for_engine`, `agent_worker_build_system_tokens`, `main`)
- Modify: `tests/ornith/test_loader.sh`
- Modify: spec §5 Server

**Interfaces:**
- Consumes: Task 2 `ds4_engine_uses_qwen35_text`, `ds4_engine_reasoning_effort_text`; Task 3 flavor and `test_ornith_prompt`, `ornith_test_ends_with`; Task 5 payload (the agent's session files and the server's `--kv-disk-dir` use it).
- Produces: model id `ornith-1.5-35b-a3b` (default `model` of responses), aliases `-chat`, `-nothink`, `-no-think` (thinking off when the request sets no thinking field), `-reasoner` (thinking on); `/v1/models` = base, `-chat`, `-reasoner`; `g_server_qwen_flavor` set at startup.

- [ ] **Step 1: Write the failing unit test and the loader checks**

In `ds4_server.c`, directly above `static void ds4_server_unit_tests_run(void) {`:

```c
/* Ornith ids: the aliases are known, toggle thinking like Qwen3.8's, and an
 * explicit thinking field wins over an alias. */
static void test_ornith_model_ids(void) {
    static const char *ids[] = {"ornith-1.5-35b-a3b", "ornith-1.5-35b-a3b-chat", "ornith-1.5-35b-a3b-reasoner",
                                "ornith-1.5-35b-a3b-nothink", "ornith-1.5-35b-a3b-no-think"};
    for (size_t i = 0; i < sizeof(ids) / sizeof(ids[0]); i++) TEST_ASSERT(server_model_alias_known(ids[i]));
    TEST_ASSERT(!model_alias_disables_thinking("ornith-1.5-35b-a3b"));
    TEST_ASSERT(model_alias_disables_thinking("ornith-1.5-35b-a3b-chat"));
    TEST_ASSERT(model_alias_disables_thinking("ornith-1.5-35b-a3b-nothink"));
    TEST_ASSERT(model_alias_disables_thinking("ornith-1.5-35b-a3b-no-think"));
    TEST_ASSERT(model_alias_enables_thinking("ornith-1.5-35b-a3b-reasoner"));
    TEST_ASSERT(!model_alias_enables_thinking("ornith-1.5-35b-a3b"));
    g_server_qwen_flavor = SERVER_QWEN_FLAVOR_ORNITH;
    ds4_think_mode mode = DS4_THINK_HIGH;
    char *p = test_ornith_prompt("chat", "{\"model\":\"ornith-1.5-35b-a3b-chat\","
                                         "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", &mode);
    TEST_ASSERT(p && mode == DS4_THINK_NONE && ornith_test_ends_with(p, "<think>\n\n</think>\n\n"));
    free(p);
    p = test_ornith_prompt("chat", "{\"model\":\"ornith-1.5-35b-a3b-chat\",\"think\":true,"
                                   "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", &mode);
    TEST_ASSERT(p && mode == DS4_THINK_MEDIUM);
    free(p);
    g_server_qwen_flavor = SERVER_QWEN_FLAVOR_QWEN38;
}
```

At the end of `ds4_server_unit_tests_run`, after `    test_kv_cache_lookup_separates_qwen38_and_ornith();`:

```c
    test_ornith_model_ids();
```

In `tests/ornith/test_loader.sh`, replace the header comment (from `# Loader checks for Ornith (qwen35moe): the 23G GGUF inspects cleanly, and a` through `# Needs a built ./ds4, ./ds4-server and ./ds4-agent and DS4_ORNITH_MODEL.`; M2 already added `--mtp-model` and `--mtp-exact-sampling` to it) with:

```sh
# Loader checks for Ornith (qwen35moe): the 23G GGUF inspects cleanly, and a
# wrong metadata value, an unsupported expert type (trunk or MTP) or a type
# mismatch between fused tensors fails with a message naming the key, the
# tier or the tensors.  The Metal-only open gate also refuses a non-Metal
# backend, --batched-session, a context above the native 262144,
# --mtp-model and --mtp-exact-sampling.  Since milestone M3, ds4-server
# serves Ornith under its own ids and ds4-agent runs a short
# non-interactive turn.
# Needs a built ./ds4, ./ds4-server and ./ds4-agent and DS4_ORNITH_MODEL.
```

Directly after the `expect_refused` function (after its closing `}`), add:

```sh
# expect_ok NAME OUT CMD...: run CMD in the background with no input and
# require exit 0 within 600s; SIGTERM (never -9) on a hang, then fail.
expect_ok() {
    name=$1
    out=$2
    shift 2
    "$@" < /dev/null > "$out" 2>&1 &
    pid=$!
    rc=""
    i=0
    while [ "$i" -lt 600 ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            rc=0
            wait "$pid" || rc=$?
            break
        fi
        sleep 1
        i=$((i + 1))
    done
    if [ -z "$rc" ]; then
        kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        cat "$out"
        echo "$name did not exit within 600s"
        exit 1
    fi
    if [ "$rc" -ne 0 ]; then
        cat "$out"
        echo "$name failed ($rc)"
        exit 1
    fi
}
```

Replace the block that starts with `# ds4-server and ds4-agent refuse Ornith until milestone M3.` (it ends with the `ds4-agent ... arrives in milestone M3` grep; M2's `# M2: --mtp runs the embedded blk.40 head` checks after it stay)

```sh
# ds4-server and ds4-agent refuse Ornith until milestone M3.
expect_refused "ds4-server" "$tmp/srv_plain.txt" ./ds4-server -m "$model" --port 18191
grep -q 'ds4-server: Ornith-1.5-35B-A3B serving arrives in milestone M3' "$tmp/srv_plain.txt" ||
    { cat "$tmp/srv_plain.txt"; exit 1; }
expect_refused "ds4-agent" "$tmp/agent.txt" ./ds4-agent -m "$model"
grep -q 'ds4-agent: Ornith-1.5-35B-A3B agent mode arrives in milestone M3' "$tmp/agent.txt" ||
    { cat "$tmp/agent.txt"; exit 1; }
```

with

```sh
# M3: ds4-server serves Ornith under its own ids (port 18191, never a gateway
# port).  The server is stopped with SIGTERM and waited for, never killed.
./ds4-server -m "$model" -c 4096 --port 18191 > "$tmp/srv_plain.txt" 2>&1 &
srv=$!
stop_server() {
    kill -TERM "$srv" 2>/dev/null || true
    j=0
    while kill -0 "$srv" 2>/dev/null; do
        j=$((j + 1))
        if [ "$j" -ge 180 ]; then
            echo "ds4-server pid $srv ignored SIGTERM for 180s; not killing a Metal process"
            exit 1
        fi
        sleep 1
    done
    wait "$srv" 2>/dev/null || true
}
i=0
until curl -sf http://127.0.0.1:18191/v1/models > "$tmp/models.json" 2>/dev/null; do
    if ! kill -0 "$srv" 2>/dev/null; then
        cat "$tmp/srv_plain.txt"
        echo "ds4-server exited during startup"
        exit 1
    fi
    i=$((i + 1))
    if [ "$i" -ge 300 ]; then
        stop_server
        cat "$tmp/srv_plain.txt"
        echo "ds4-server did not answer within 300s"
        exit 1
    fi
    sleep 1
done
python3 - "$tmp/models.json" <<'EOF' || { stop_server; exit 1; }
import json, sys
ids = [m["id"] for m in json.load(open(sys.argv[1]))["data"]]
want = ["ornith-1.5-35b-a3b", "ornith-1.5-35b-a3b-chat", "ornith-1.5-35b-a3b-reasoner"]
if ids != want:
    sys.exit(f"/v1/models lists {ids}, expected {want}")
EOF
curl -sf http://127.0.0.1:18191/v1/models/ornith-1.5-35b-a3b-nothink > /dev/null ||
    { stop_server; echo "alias ornith-1.5-35b-a3b-nothink is unknown"; exit 1; }
curl -sf http://127.0.0.1:18191/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Say ok."}],"max_tokens":8,"temperature":0,"think":false}' \
    > "$tmp/chat.json" || { stop_server; cat "$tmp/srv_plain.txt"; echo "chat request failed"; exit 1; }
python3 - "$tmp/chat.json" <<'EOF' || { stop_server; exit 1; }
import json, sys
r = json.load(open(sys.argv[1]))
if r.get("model") != "ornith-1.5-35b-a3b":
    sys.exit(f"default model id {r.get('model')!r}")
if not (r["choices"][0]["message"].get("content") or "").strip():
    sys.exit("empty reply")
EOF
stop_server

# M3: ds4-agent runs one short non-interactive turn (scratch directory, so a
# tool call cannot touch the repository).
root=$(pwd)
mkdir -p "$tmp/agent-cwd"
( cd "$tmp/agent-cwd" && expect_ok "ds4-agent" "$tmp/agent.txt" "$root/ds4-agent" -m "$model" \
    --non-interactive --nothink -n 64 -p 'Reply with the single word: ready. Do not call any tool.' ) || exit 1
if grep -q 'arrives in milestone M3' "$tmp/agent.txt"; then
    cat "$tmp/agent.txt"
    echo "ds4-agent still refuses Ornith"
    exit 1
fi
# the answer's wording is the model's business: a soft check, logged only
grep -qi 'ready' "$tmp/agent.txt" || { cat "$tmp/agent.txt"; echo "WARN: ds4-agent did not answer 'ready'"; }
```

- [ ] **Step 2: Confirm the failures**

```bash
make ds4_test && ./ds4_test --server 2>&1 | tail -2
make ds4 ds4-server ds4-agent
caffeinate -i -s ./tests/ornith/test_loader.sh 2>&1 | tail -3
```

Expected: `server: ERR` and `ds4 tests: 10 failure(s)` (every failed `TEST_ASSERT` of `test_ornith_model_ids` counts); the loader test ends with `ds4-server exited during startup` after the M1 refusal message.

- [ ] **Step 3: Ids and aliases in `ds4_server.c`**

In `model_alias_disables_thinking`, replace the line

```c
            !strcmp(model, "qwen/qwen3.8-flash-next-chat") ||
```

with

```c
            !strcmp(model, "qwen/qwen3.8-flash-next-chat") ||
            !strcmp(model, "ornith-1.5-35b-a3b-chat") ||
            !strcmp(model, "ornith-1.5-35b-a3b-no-think") ||
            !strcmp(model, "ornith-1.5-35b-a3b-nothink") ||
```

In `model_alias_enables_thinking`, replace the line

```c
            !strcmp(model, "qwen/qwen3.8-flash-next-reasoner") ||
```

with

```c
            !strcmp(model, "qwen/qwen3.8-flash-next-reasoner") ||
            !strcmp(model, "ornith-1.5-35b-a3b-reasoner") ||
```

In `server_model_id_from_engine`, replace `    if (ds4_engine_is_qwen4(engine)) return "qwen3.8-flash-next";` with:

```c
    if (ds4_engine_is_qwen35moe(engine)) return "ornith-1.5-35b-a3b";
    if (ds4_engine_is_qwen4(engine)) return "qwen3.8-flash-next";
```

In `server_model_alias_known`, replace the line

```c
            !strcmp(id, "qwen/qwen3.8-flash-next-reasoner") ||
```

with

```c
            !strcmp(id, "qwen/qwen3.8-flash-next-reasoner") ||
            !strcmp(id, "ornith-1.5-35b-a3b") ||
            !strcmp(id, "ornith-1.5-35b-a3b-chat") ||
            !strcmp(id, "ornith-1.5-35b-a3b-no-think") ||
            !strcmp(id, "ornith-1.5-35b-a3b-nothink") ||
            !strcmp(id, "ornith-1.5-35b-a3b-reasoner") ||
```

In `send_models`, replace

```c
    } else if (ds4_engine_is_qwen4(s->engine)) {
        append_model_json(&b, s, "qwen3.8-flash-next");
```

with

```c
    } else if (ds4_engine_is_qwen35moe(s->engine)) {
        append_model_json(&b, s, "ornith-1.5-35b-a3b");
        buf_putc(&b, ',');
        append_model_json(&b, s, "ornith-1.5-35b-a3b-chat");
        buf_putc(&b, ',');
        append_model_json(&b, s, "ornith-1.5-35b-a3b-reasoner");
    } else if (ds4_engine_is_qwen4(s->engine)) {
        append_model_json(&b, s, "qwen3.8-flash-next");
```

In `main`, replace

```c
    /* The server's chat ids, tool syntax and disk KV are not wired for
     * Ornith yet. */
    if (ds4_engine_is_qwen35moe(engine)) {
        fprintf(stderr, "ds4-server: Ornith-1.5-35B-A3B serving arrives in milestone M3; use ./ds4 for now\n");
        ds4_engine_close(engine);
        return 1;
    }
```

with

```c
    /* Ornith renders its own ChatML dialect (spec §5); every other engine
     * keeps the Qwen3.8 flavor. */
    g_server_qwen_flavor = ds4_engine_is_qwen35moe(engine) ?
        SERVER_QWEN_FLAVOR_ORNITH : SERVER_QWEN_FLAVOR_QWEN38;
```

- [ ] **Step 4: The agent**

In `ds4_agent.c`, replace `    if (ds4_engine_is_qwen4(engine)) return AGENT_TOOL_SYNTAX_QWEN;` (in `agent_tool_syntax_for_engine`) with:

```c
    if (ds4_engine_uses_qwen35_text(engine)) return AGENT_TOOL_SYNTAX_QWEN;
```

In `agent_worker_build_system_tokens`, replace

```c
    if (ds4_engine_is_qwen4(w->engine)) {
        const char *effort = ds4_qwen4_reasoning_effort_text(think_mode);
        if (effort) ds4_chat_append_message(w->engine, out, "system", effort);
```

with

```c
    if (ds4_engine_uses_qwen35_text(w->engine)) {
        /* Qwen3.8: its xhigh/low lines; Ornith: its template default medium
         * for the agent's default think mode (ds4_engine_reasoning_effort_text) */
        const char *effort = ds4_engine_reasoning_effort_text(w->engine, think_mode);
        if (effort) ds4_chat_append_message(w->engine, out, "system", effort);
```

In `main`, delete

```c
    /* The agent's tool syntax and session files are not wired for Ornith
     * yet. */
    if (ds4_engine_is_qwen35moe(engine)) {
        fprintf(stderr, "ds4-agent: Ornith-1.5-35B-A3B agent mode arrives in milestone M3; use ./ds4 for now\n");
        ds4_engine_close(engine);
        return 1;
    }
```

- [ ] **Step 5: Run the unit tests and the loader test**

GPU window.

```bash
make ds4 ds4-server ds4-agent ds4_test ds4_agent_test
./ds4_test --server && ./ds4_agent_test
make test-ornith-render
caffeinate -i -s ./tests/ornith/test_loader.sh 2>&1 | tail -3
```

Expected: `ds4 tests: ok`, the agent tests pass, `ornith render: PASS`, `ornith loader: ok`.

- [ ] **Step 6: Build every target and run the Qwen fast gate**

```bash
make ds4-bench ds4-eval && make cpu
# make cpu relinks ./ds4, ./ds4-server, ./ds4-agent, ./ds4-bench and ./ds4-eval as CPU-only
# binaries; rebuild the Metal ones before any model run or run.sh (it only runs make ds4-server)
rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent && make ds4 ds4-server ds4-bench ds4-eval ds4-agent
make test-qwen4-kernels test-qwen4-q2
caffeinate -i -s speed-bench/qwen-regression/run.sh fast 2>&1 | tail -5
```

Expected: clean builds, kernel tests pass, `qwen_gate: PASS` (the registry command check included).

- [ ] **Step 7: Update the spec (deviation 10)**

In §5, replace

```
- **Server.** Model id `ornith-1.5-35b-a3b` with the `-chat`, `-reasoner` and
  `-nothink` aliases, following the Qwen3.8 pattern.
  `SERVER_MODEL_SYNTAX_QWEN` for tool calls. Gateway registry changes belong to
  the deploy step after v2.
```

with

```
- **Server.** Model id `ornith-1.5-35b-a3b` with the `-chat`, `-reasoner`,
  `-nothink` and `-no-think` aliases, following the Qwen3.8 pattern: -chat,
  -nothink and -no-think turn thinking off and -reasoner on when the request
  sets no thinking field; `/v1/models` lists the base id, `-chat` and
  `-reasoner`. `SERVER_MODEL_SYNTAX_QWEN` for tool calls with the Ornith
  render flavor. Gateway registry changes belong to the deploy step after v2.
```

- [ ] **Step 8: Commit**

```bash
git add ds4_server.c ds4_agent.c tests/ornith/test_loader.sh docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md
git commit -m "ds4-server, ds4-agent: serve Ornith

The server answers as ornith-1.5-35b-a3b with -chat, -nothink,
-no-think and -reasoner aliases and sets the Ornith render flavor at
startup; the agent uses the Qwen tool syntax and the engine's effort
line for Ornith.  The M1 refusals are gone; test_loader.sh now checks
the model list, a chat reply and a short agent turn.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- ds4_server.c ds4_agent.c tests/ornith/test_loader.sh docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md
```

---

### Task 8: Live staging-server gate

Test scripts and a receipt only. Every run needs a GPU window (live stack paused with the user's OK) and runs one model process at a time.

**Files:**
- Create: `tests/ornith/serverlib.py`, `tests/ornith/test_server_kv.py`, `tests/ornith/test_server_live.py`, `speed-bench/ornith/m3/RESULTS.md`

**Interfaces:**
- Consumes: Tasks 1-7 (a built Metal `./ds4-server`), `DS4_ORNITH_MODEL`.
- Produces (module `tests/ornith/serverlib.py`): `ROOT`, `PORT = 18296`, `BASE`, `BUDGET_MESSAGE`, `WARNINGS: list[str]`, `model_path() -> str`, `class Server(log_path, args, env=None)` with `.models`, `.new_log() -> str`, `.stop()`; `post(path, body, timeout=1800, save=None) -> dict` (with `save`, a path stem, it writes `<stem>.request.json` and `<stem>.reply.json`); `check(cond, what)` (hard: prints PASS/FAIL, exits 1 on FAIL); `soft_check(cond, what, request=None)` (prints PASS or WARN, never fails); `report(name)`.
- Produces (scripts): `test_server_kv.py OUT_DIR`, `test_server_live.py OUT_DIR`; each exits 0 only when every hard check passes, prints `PASS`/`FAIL`/`WARN`/`INFO` lines and ends with `<name>: OK (N WARN)`.

Hard checks are structural: finish and stop reasons, tool names and parsed argument objects, cached-token counts, byte identity between runs that must agree. What the model says (an arithmetic result, a city, a temperature) is a soft check: a miss prints `WARN` with the saved request and does not fail the gate; Step 5 says how a WARN is investigated.

- [ ] **Step 1: Write `tests/ornith/serverlib.py`**

```python
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
```

- [ ] **Step 2: Write `tests/ornith/test_server_kv.py`**

```python
#!/usr/bin/env python3
"""Ornith disk-KV checkpoints across server restarts (gate 1 item 4, live).

  python3 tests/ornith/test_server_kv.py OUT_DIR

Needs DS4_ORNITH_MODEL, a built Metal ./ds4-server and a free GPU.

Phase 1, --mtp: three conversations x three turns (tool-less, tools with the
reasoning omitted, tools with the reasoning echoed); the server restarts
before turn 3.  Every turn ends with stop and no tool call; turns 2 and 3
reuse more than 3000 cached tokens (turn 3 from disk).
Phase 2, the same cache without --mtp, replays the last turn of tools-echo
(its checkpoints are the newest; the 512 MB budget evicts older ones): the
--mtp checkpoint is refused with a clear log line and the prompt is
prefilled and answered.
Phase 3, a fresh cache: a plain server answers the same replay and stores
checkpoints without MTP, then a --mtp server refuses them the same way.
Hard: the three cold answers of phases 2-3 are byte-identical (greedy, same
prefill chunks, M2's --mtp output equals plain).  Soft: the arithmetic
answers, and the cold answer equal to phase 1's, which came through a
restored checkpoint whose KV was partly written by decode (rounding there
can differ at a near tie).
"""
import copy
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import serverlib as sl

TOOLS = [{"type": "function", "function": {
    "name": "lookup", "description": "Look up information only when explicitly requested.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]
FILLER = "The town archive records weather and routine shipping schedules. " * 300
ANSWERS = ["51", "53", "55"]


def server_args(cache, mtp):
    args = ["-c", "16384", "--prefill-chunk", "1024", "--kv-disk-dir", str(cache),
            "--kv-disk-space-mb", "512", "--kv-cache-min-tokens", "128",
            "--kv-cache-cold-max-tokens", "0", "--kv-cache-continued-interval-tokens", "1024",
            "--kv-cache-boundary-align-tokens", "128"]
    return (["--mtp"] if mtp else []) + args


def content(result):
    return result["choices"][0]["message"].get("content") or ""


def refused(srv, body, saved, save):
    """The best checkpoint for body is refused; the prompt is prefilled and answered."""
    result = sl.post("/v1/chat/completions", body, save=save)
    log = srv.new_log()
    sl.check("kv cache load failed" in log and saved in log, f"a checkpoint {saved} is refused")
    sl.check(result["choices"][0]["finish_reason"] == "stop" and content(result).strip() != "",
             "after the refusal the prompt is prefilled and answered")
    return content(result)


def main(argv):
    if len(argv) != 1:
        sys.exit(__doc__)
    out = pathlib.Path(argv[0]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "kv"
    cache.mkdir(exist_ok=True)
    log = out / "server.log"
    replay = None
    phase1_answer = None
    srv = sl.Server(log, server_args(cache, True))
    try:
        for name, has_tools, echo in [("tool-less-omit", False, False),
                                      ("tools-omit", True, False),
                                      ("tools-echo", True, True)]:
            history = [
                {"role": "system", "content": "You are a helpful assistant. Answer arithmetic yourself; "
                                              "do not call tools. Think briefly."},
                {"role": "user", "content": name + "\n" + FILLER +
                 "\nWhat is 17 multiplied by 3? Answer with just the number."}]
            for turn in range(3):
                if turn == 2:
                    srv.stop()
                    srv = sl.Server(log, server_args(cache, True))
                body = {"messages": history, "temperature": 0, "max_tokens": 512,
                        "reasoning_effort": "low", "stream": False}
                if has_tools:
                    body["tools"] = TOOLS
                stem = out / f"{name}-{turn + 1}"
                result = sl.post("/v1/chat/completions", body, save=stem)
                if name == "tools-echo" and turn == 2:
                    replay = copy.deepcopy(body)
                    phase1_answer = content(result)
                message = result["choices"][0]["message"]
                usage = result["usage"]
                cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                print(json.dumps({"case": name, "turn": turn + 1, "finish": result["choices"][0]["finish_reason"],
                                  "prompt": usage["prompt_tokens"], "cached": cached,
                                  "content": content(result)}), flush=True)
                sl.check(result["choices"][0]["finish_reason"] == "stop" and not message.get("tool_calls"),
                         f"{name} turn {turn + 1} ends with stop and no tool call")
                sl.soft_check(ANSWERS[turn] in content(result), f"{name} turn {turn + 1} answers {ANSWERS[turn]}",
                              f"{stem}.request.json")
                if turn > 0:
                    sl.check(cached > 3000 and usage["prompt_tokens"] - cached < 128,
                             f"{name} turn {turn + 1} reuses the checkpoint ({cached} cached)")
                assistant = {"role": "assistant", "content": content(result)}
                if echo:
                    assistant["reasoning_content"] = message.get("reasoning_content") or ""
                history += [assistant, {"role": "user",
                                        "content": "Now add 2 to your previous answer. Answer with just the number."}]
        srv.stop()
        text = log.read_text(errors="replace")
        sl.check("KV payload staging failed" not in text, "no payload staging failure")
        sl.check("session has no valid checkpoint to stage" not in text, "every stage had a checkpoint")
        sl.check(text.count("reason=continued") >= 9, "continued checkpoints were stored")
        sl.check("reason=evict" in text, "an evict checkpoint was stored")
        sl.check("kv cache evicted reason=disk-cache-full" in text, "the disk budget evicted old checkpoints")

        srv = sl.Server(log, server_args(cache, False))
        phase2 = refused(srv, replay, "saved with --mtp", out / "phase2")
        srv.stop()

        plain_cache = out / "kv-plain"
        plain_cache.mkdir(exist_ok=True)
        srv = sl.Server(log, server_args(plain_cache, False))
        result = sl.post("/v1/chat/completions", replay, save=out / "phase3-plain")
        sl.check(result["choices"][0]["finish_reason"] == "stop" and content(result).strip() != "",
                 "a plain server answers the replay and stores checkpoints")
        phase3_plain = content(result)
        srv.stop()
        srv = sl.Server(log, server_args(plain_cache, True))
        phase3_mtp = refused(srv, replay, "saved without --mtp", out / "phase3-mtp")
        srv.stop()

        sl.check(phase2 == phase3_plain == phase3_mtp, "the three cold answers are byte-identical")
        sl.soft_check(phase2 == phase1_answer, "the cold answer equals phase 1's answer from the restored checkpoint",
                      f"{out / 'phase2'}.request.json")
    finally:
        srv.stop()
    sl.report("ornith server kv")


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 3: Write `tests/ornith/test_server_live.py`**

```python
#!/usr/bin/env python3
"""Ornith serving checks on a live staging server (M3).

  python3 tests/ornith/test_server_live.py OUT_DIR

Needs DS4_ORNITH_MODEL, a built Metal ./ds4-server and a free GPU.  Servers
run one at a time on port 18296.  Hard checks are structural; what the model
says is a soft check (WARN, see serverlib).

1. ids: /v1/models lists ornith-1.5-35b-a3b, -chat and -reasoner; a request
   without "model" answers as ornith-1.5-35b-a3b; -chat answers without
   reasoning.
2. MTP identity: at temperature 0 a --mtp server and a plain server give
   byte-identical content, reasoning, tool calls and finish reasons for four
   prompts (gate 1 item 3 through the server).  The --mtp server runs with
   --trace, and the script prints how many speculative-boundary rewinds the
   trace records; that count is information only (ds4_test --qwen35-rewind
   pins the rewind itself).
3. OpenAI tool round trip (--mtp): get_weather is called with an arguments
   object; the tool result goes back, and the answer ends with stop and
   reuses the live prefix.
4. Anthropic /v1/messages tool round trip: tool_use, then a tool_result
   block plus a text block in one user message; the answer ends the turn and
   reads the whole first prompt from the cache.
5. Think-cap replay as the gateway sends it: a conversation with tools and
   historical reasoning_content, first with thinking on and a 64-token cap,
   then with enable_thinking=false and preserve_thinking removed; the replay
   returns no reasoning.
6. Think budget: with --think-budget 64 the reasoning ends with the budget
   sentence and an answer follows.
"""
import copy
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import serverlib as sl

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
ANTHROPIC_WEATHER = {"name": "get_weather", "description": "Get the current weather for a city.",
                     "input_schema": {"type": "object", "properties": {"city": {"type": "string"}},
                                      "required": ["city"]}}
WEATHER_RESULT = '{"temp_c": 18, "sky": "cloudy"}'
ASK_WEATHER = "What's the weather in Paris right now? Use the tool."
IDENTITY = [
    ("en", [{"role": "user", "content": "Explain in two sentences why the sky is blue."}], None),
    ("vi", [{"role": "user", "content": "Kể tên ba món ăn nổi tiếng của Hà Nội, mỗi món một câu."}], None),
    ("code", [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number. "
                                          "Code only."}], None),
    ("tool", [{"role": "user", "content": ASK_WEATHER}], [WEATHER]),
]
# the gateway's think-cap replay shape: tools, a tool round trip and
# historical reasoning_content before the new question
REPLAY_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Paris?"},
    {"role": "assistant", "content": "", "reasoning_content": "The user wants weather. Call the tool.",
     "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": WEATHER_RESULT},
    {"role": "assistant", "content": "It is 18 C and cloudy in Paris.",
     "reasoning_content": "Tool returned 18 C cloudy."},
    {"role": "user", "content": "Should I bring an umbrella? Answer yes or no, then one sentence."},
]


def chat(messages, tools=None, save=None, **extra):
    body = {"messages": messages, "temperature": 0, "max_tokens": 600, "stream": False}
    if tools:
        body["tools"] = tools
    body.update(extra)
    return sl.post("/v1/chat/completions", body, save=save)


def reply_key(result):
    message = result["choices"][0]["message"]
    calls = [(c["function"]["name"], c["function"]["arguments"]) for c in message.get("tool_calls") or []]
    return [message.get("content") or "", message.get("reasoning_content") or "", calls,
            result["choices"][0]["finish_reason"]]


def cached_tokens(result):
    return (result["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)


def arguments_object(call):
    """An OpenAI tool call's parsed arguments, or None when they are not a JSON object."""
    try:
        args = json.loads(call["function"]["arguments"])
    except (KeyError, TypeError, ValueError):
        return None
    return args if isinstance(args, dict) else None


def main(argv):
    if len(argv) != 1:
        sys.exit(__doc__)
    out = pathlib.Path(argv[0]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = out / "server.log"

    srv = sl.Server(log, ["-c", "32768"])
    try:
        plain = {name: reply_key(chat(msgs, tools, save=out / f"identity-{name}-plain"))
                 for name, msgs, tools in IDENTITY}
    finally:
        srv.stop()

    trace = out / "trace-mtp.txt"
    srv = sl.Server(log, ["--mtp", "-c", "32768", "--trace", trace])
    try:
        # 1. ids
        ids = [m["id"] for m in srv.models["data"]]
        sl.check(ids == ["ornith-1.5-35b-a3b", "ornith-1.5-35b-a3b-chat", "ornith-1.5-35b-a3b-reasoner"],
                 f"/v1/models lists {ids}")
        r = chat([{"role": "user", "content": "Say ok."}], save=out / "ids-default", max_tokens=64)
        sl.check(r.get("model") == "ornith-1.5-35b-a3b", "a request without model answers as ornith-1.5-35b-a3b")
        r = chat([{"role": "user", "content": "Say ok."}], save=out / "ids-chat",
                 model="ornith-1.5-35b-a3b-chat", max_tokens=64)
        m = r["choices"][0]["message"]
        sl.check(not m.get("reasoning_content") and (m.get("content") or "").strip() != "",
                 "-chat answers without reasoning")

        # 2. MTP identity
        for name, msgs, tools in IDENTITY:
            got = reply_key(chat(msgs, tools, save=out / f"identity-{name}-mtp"))
            sl.check(got == plain[name], f"--mtp reply equals the plain reply ({name})")

        # 3. OpenAI tool round trip
        msgs = [{"role": "user", "content": ASK_WEATHER}]
        first = chat(msgs, [WEATHER], save=out / "openai-tool-1", max_tokens=1024)
        m1 = first["choices"][0]["message"]
        calls = m1.get("tool_calls") or []
        args = arguments_object(calls[0]) if calls else None
        sl.check(first["choices"][0]["finish_reason"] == "tool_calls" and bool(calls) and
                 calls[0]["function"]["name"] == "get_weather" and args is not None,
                 "the model calls get_weather with an arguments object")
        sl.soft_check("paris" in str(args.get("city", "")).lower(), "the call asks for Paris",
                      f"{out / 'openai-tool-1'}.request.json")
        msgs += [{"role": "assistant", "content": m1.get("content") or "",
                  "reasoning_content": m1.get("reasoning_content") or "", "tool_calls": calls},
                 {"role": "tool", "tool_call_id": calls[0]["id"], "content": WEATHER_RESULT + "\n"}]
        second = chat(msgs, [WEATHER], save=out / "openai-tool-2", max_tokens=1024)
        answer = second["choices"][0]["message"].get("content") or ""
        sl.check(second["choices"][0]["finish_reason"] == "stop" and answer.strip() != "",
                 "the tool turn ends with an answer")
        sl.soft_check("18" in answer, "the answer uses the tool result (18)", f"{out / 'openai-tool-2'}.request.json")
        sl.check(cached_tokens(second) >= first["usage"]["prompt_tokens"],
                 f"the tool turn reuses the live prefix ({cached_tokens(second)} cached)")

        # 4. Anthropic tool round trip
        amsgs = [{"role": "user", "content": ASK_WEATHER}]
        a1 = sl.post("/v1/messages", {"max_tokens": 1024, "temperature": 0, "tools": [ANTHROPIC_WEATHER],
                                      "messages": amsgs}, save=out / "anthropic-tool-1")
        uses = [b for b in a1["content"] if b["type"] == "tool_use"]
        sl.check(a1["stop_reason"] == "tool_use" and bool(uses) and uses[0]["name"] == "get_weather" and
                 isinstance(uses[0].get("input"), dict), "anthropic: the model calls get_weather with an input object")
        sl.soft_check("paris" in str(uses[0]["input"].get("city", "")).lower(), "anthropic: the call asks for Paris",
                      f"{out / 'anthropic-tool-1'}.request.json")
        amsgs += [{"role": "assistant", "content": a1["content"]},
                  {"role": "user", "content": [
                      {"type": "tool_result", "tool_use_id": uses[0]["id"], "content": WEATHER_RESULT},
                      {"type": "text", "text": "Answer in one sentence."}]}]
        a2 = sl.post("/v1/messages", {"max_tokens": 1024, "temperature": 0, "tools": [ANTHROPIC_WEATHER],
                                      "messages": amsgs}, save=out / "anthropic-tool-2")
        text = "".join(b.get("text", "") for b in a2["content"] if b["type"] == "text")
        sl.check(a2["stop_reason"] == "end_turn" and text.strip() != "", "anthropic: the tool turn ends with an answer")
        sl.soft_check("18" in text, "anthropic: the answer uses the tool result (18)",
                      f"{out / 'anthropic-tool-2'}.request.json")
        u1 = a1["usage"]
        a1_prompt = (u1.get("input_tokens", 0) + u1.get("cache_read_input_tokens", 0) +
                     u1.get("cache_creation_input_tokens", 0))
        sl.check(a2["usage"].get("cache_read_input_tokens", 0) >= a1_prompt,
                 f"anthropic: the tool turn reads the whole first prompt ({a1_prompt} tokens) from the cache")

        # 5. think-cap replay, as the gateway builds it
        base = {"messages": REPLAY_HISTORY, "tools": [WEATHER], "temperature": 0, "stream": False,
                "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}
        capped = sl.post("/v1/chat/completions", dict(base, max_tokens=64), save=out / "think-cap-1")
        replay = copy.deepcopy(base)
        replay["chat_template_kwargs"]["enable_thinking"] = False
        del replay["chat_template_kwargs"]["preserve_thinking"]
        replay["max_tokens"] = 400
        r = sl.post("/v1/chat/completions", replay, save=out / "think-cap-2")
        m = r["choices"][0]["message"]
        finish = r["choices"][0]["finish_reason"]
        sl.check(not m.get("reasoning_content") and finish in ("stop", "tool_calls") and
                 (finish != "stop" or (m.get("content") or "").strip() != ""),
                 "the think-cap replay returns no reasoning")
        sl.soft_check(finish == "stop", "the replay answers instead of calling the tool again",
                      f"{out / 'think-cap-2'}.request.json")
        print(f"INFO think-cap: capped attempt cached {cached_tokens(capped)}, replay cached {cached_tokens(r)} of "
              f"{r['usage']['prompt_tokens']} (thinking changes the system turn, so a re-prefill is expected)",
              flush=True)
    finally:
        srv.stop()
    trace_text = trace.read_text(errors="replace") if trace.exists() else ""
    print(f"INFO speculative-boundary rewinds in the --mtp trace: {trace_text.count('speculative boundary: kept=')}",
          flush=True)

    # 6. think budget
    srv = sl.Server(log, ["--mtp", "-c", "32768", "--think-budget", "64"])
    try:
        r = chat([{"role": "user", "content": "Prove that there are infinitely many prime numbers, "
                                              "then list the first ten primes."}],
                 save=out / "think-budget", max_tokens=600)
        new = srv.new_log()
        m = r["choices"][0]["message"]
        sl.check("thinking budget reached 64 tokens" in new, "think budget: the cap fired")
        sl.check(sl.BUDGET_MESSAGE in (m.get("reasoning_content") or ""),
                 "think budget: the reasoning ends with the budget sentence")
        sl.check((m.get("content") or "").strip() != "", "think budget: an answer follows")
    finally:
        srv.stop()
    sl.report("ornith server live")


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 4: Syntax check without a model**

```bash
chmod +x tests/ornith/test_server_kv.py tests/ornith/test_server_live.py
python3 -m py_compile tests/ornith/serverlib.py tests/ornith/test_server_kv.py tests/ornith/test_server_live.py && echo compiled
env -u DS4_ORNITH_MODEL python3 tests/ornith/test_server_kv.py /tmp/ornith-m3-nomodel 2>&1 | tail -1
```

Expected: `compiled`; `set DS4_ORNITH_MODEL to the 23G ICE GGUF`.

- [ ] **Step 5: Run the live gate**

Agree the GPU window with the user, pause the live stack (Global Constraints), check that no Metal process is stuck, rebuild the Metal binaries (an earlier `make cpu` leaves CPU-only ones behind), then run both scripts one after the other.

```bash
rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent && make ds4 ds4-server ds4-bench ds4-eval ds4-agent
ps -axo pid,stat,comm | awk '$2 ~ /^(E|U)/'
export DS4_ORNITH_MODEL=$HOME/orca/workspaces/ds4-metal-data/gguf/ornith/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf
mkdir -p speed-bench/ornith/m3/logs
caffeinate -i -s python3 tests/ornith/test_server_kv.py speed-bench/ornith/m3/logs/server-kv \
    | tee speed-bench/ornith/m3/logs/server-kv.txt
caffeinate -i -s python3 tests/ornith/test_server_live.py speed-bench/ornith/m3/logs/server-live \
    | tee speed-bench/ornith/m3/logs/server-live.txt
```

Expected: no `FAIL` line, then `ornith server kv: OK (N WARN)` and `ornith server live: OK (N WARN)`. On a FAIL, stop and use `superpowers:systematic-debugging`: the `*.request.json` / `*.reply.json` files and `server.log` in the output directory hold every request, reply and cache decision. An identity failure (part 2) or unequal cold answers (kv phases 2-3) are MTP or rendering bugs, never tests to relax.

A WARN is not yet a bug. After both scripts have stopped their servers (one model process at a time), send the saved request to llama-server on the same GGUF and compare:

```bash
llama-server -m "$DS4_ORNITH_MODEL" --host 127.0.0.1 --port 18190 -ngl 99 -fa on --jinja -c 32768 \
    > /tmp/ornith-llama-warn.log 2>&1 &
LLAMA=$!
until curl -sf http://127.0.0.1:18190/health > /dev/null; do kill -0 "$LLAMA" || break; sleep 1; done
REQ=speed-bench/ornith/m3/logs/server-live/openai-tool-2.request.json   # the path the WARN line names
curl -s http://127.0.0.1:18190/v1/chat/completions -H 'Content-Type: application/json' -d @"$REQ" \
    | python3 -m json.tool | head -40            # Anthropic requests: POST to /v1/messages
kill -TERM "$LLAMA"; wait "$LLAMA"
```

If llama.cpp gives the same kind of answer, the WARN is model behaviour: record it in the receipt. If llama.cpp answers as expected and ds4 does not, treat it as a ds4 bug (`superpowers:systematic-debugging`). Restore the live stack when the window ends.

- [ ] **Step 6: Write the receipt and commit**

Create `speed-bench/ornith/m3/RESULTS.md`:

```markdown
# Ornith M3 serving receipts

- Branch commit: <git rev-parse --short HEAD>
- Model: 23G ICE (`speed-bench/ornith/oracle/RESULTS.md`).
- Model-free: `./ds4_test --server` ok; `./ds4_agent_test` ok; `make test-ornith-render` 26/26
  (goldens `tests/ornith/chat/golden/`, template sha256 in `TEMPLATE.txt`).
- CLI: `tests/ornith/test_cli_chat.sh` ok.
- Sessions: `ds4_test --session-snapshot --qwen35-payloads --qwen35-rewind` ok with and without
  `DS4_TEST_GLM_MTP=1`; `make test-qwen35-session` ok (incl. the serialized batch case).
- Loader: `tests/ornith/test_loader.sh` ok.

## Live gate (port 18296)

| script | result |
|---|---|
| `tests/ornith/test_server_kv.py` | <OK (N WARN)/FAIL> |
| `tests/ornith/test_server_live.py` | <OK (N WARN)/FAIL> |

Per-turn cached tokens (test_server_kv.py):

<paste the JSON lines of speed-bench/ornith/m3/logs/server-kv.txt>

PASS, WARN and INFO lines (test_server_live.py):

<paste them from speed-bench/ornith/m3/logs/server-live.txt>

WARN follow-up (llama-server on the same request): <none, or one line per WARN with the verdict>
```

```bash
git add tests/ornith/serverlib.py tests/ornith/test_server_kv.py tests/ornith/test_server_live.py speed-bench/ornith/m3/RESULTS.md
git commit -m "tests/ornith: live serving gate and M3 receipts

test_server_kv.py restores disk checkpoints across restarts with --mtp
and refuses a checkpoint of the other MTP presence; test_server_live.py
checks the ids, --mtp against plain replies at temperature 0, OpenAI
and Anthropic tool round trips with live prefix reuse, the think-cap
replay and the think budget.  Both run on staging port 18296.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- tests/ornith/serverlib.py tests/ornith/test_server_kv.py tests/ornith/test_server_live.py speed-bench/ornith/m3/RESULTS.md
```

---

### Task 9: Qwen3.8 full gate and M3 close-out

**Files:**
- Create: `speed-bench/ornith/m3/QWEN_GATE.md`
- Modify: memory `ornith-qwen35moe-port-research.md` (outside the repo)

- [ ] **Step 1: Agree the run window with the user**

The full gate needs the machine free: the live stack paused and no other session running a model. Ask the user and wait for the go-ahead.

- [ ] **Step 2: Run the full Qwen3.8 gate**

```bash
ps -axo pid,stat,comm | awk '$2 ~ /^(E|U)/'
caffeinate -i -s speed-bench/qwen-regression/run.sh full 2>&1 | tee /tmp/qwen-gate-m3-full.log | tail -30
```

Expected: the fast checks (kernel tests, byte-identical vi/code replies, registry command unchanged) and the full checks (paired decode at least 97% of PROD, steady wired within baseline + 0.5 GiB, needle found) pass. A paired-speed failure alone is rerun once (README). Any other failure: stop, bisect over this branch's commits (Tasks 2-7 touch shared files), fix, rerun.

- [ ] **Step 3: Write the receipt and commit**

Create `speed-bench/ornith/m3/QWEN_GATE.md`:

```markdown
# Qwen3.8 regression gate after Ornith M3

- Branch commit: <git rev-parse --short HEAD>
- Command: `speed-bench/qwen-regression/run.sh full`
- Result: <PASS/FAIL>; replies byte-identical <yes/no>; paired decode <branch/PROD %>; wired <GiB>; needle <HIT/MISS>.
- Unit tests that pin the unchanged Qwen3.8 renderer: `./ds4_test --server` and `./ds4_agent_test`, unedited.
- Log tail:

<paste the last lines of /tmp/qwen-gate-m3-full.log>
```

```bash
git add speed-bench/ornith/m3/QWEN_GATE.md
git commit -m "speed-bench/ornith: Qwen3.8 regression gate after M3

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- speed-bench/ornith/m3/QWEN_GATE.md
```

- [ ] **Step 4: Restore services and update memory**

Restore the live stack (Global Constraints) and wait for `/status` 200. Update the memory file `ornith-qwen35moe-port-research.md`:
- M3 done: branch head, `make test-ornith-render` 26/26, session/rewind/payload tests, live gate result, Qwen gate result.
- Rendering facts worth keeping: absent effort = medium for Ornith; terse block and tool-error warning are part of the prompt; thinking on/off changes the system turn, so a think-cap replay re-prefills.
- Next: M4 plan (acceptance).

- [ ] **Step 5: Hand over for review**

M3 is complete when Tasks 1-9 are checked. The next step is `superpowers:requesting-code-review` on the branch diff against `develop`, then the user's OK to merge. Do not merge or push without it.

---

## Coverage check (spec §9 M3, DECISIONS §3)

| Requirement | Task |
|---|---|
| Predicate split leftovers: `encode_chat_prompt`, `ds4_chat_append_message`, `ds4_chat_append_assistant_prefix` | 2 |
| Predicate split: server syntax, agent syntax and system tokens | 3, 7 |
| `chat_push_think_prefix` no-op for Ornith | 2 |
| Server Ornith flavor chosen at startup, settable by tests (§3.2) | 3, 7 |
| Terse block, tools text, `tojson`, effort default and names, leading-only system merge (§3.1) | 3 |
| Tool-response trim, tool-error warning, assistant trim, non-string args, `preserve_thinking`, grouped results (§3.1) | 4 |
| `tool_call_format=json` → 400; other kwargs ignored (§3.1) | 3 |
| Goldens: generator from GGUF, committed fixtures, C-side comparison (§3.3) | 1, 3, 4 |
| Disk KV payload, tag, MTP presence, carry, exact size, guards (§3.5) | 5 |
| `ds4_sessions_eval_batch_metal_supported` false for Ornith (§3.5) | 6 |
| `ds4_engine_routed_quant_bits` unchanged, documented (§3.5) | 5 |
| Rewind snapshot restore, otherwise invalidate (§3.6) | 6 |
| Server ids and aliases, `/v1/models`, refusal removal, loader test (§3.7) | 7 |
| Tests: snapshot on Ornith, cross-family refusal (unit + KVC stub), rewind, golden, live kv/tool/MTP identity/think budget (§3.8) | 5, 6, 4, 8 |
| Gate 1 items 4-5 | 5, 8 / 1, 4 |
| Qwen gate full before merge | 9 |
