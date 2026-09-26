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
