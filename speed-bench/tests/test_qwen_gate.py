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

    def test_full_check_against_fast_baseline(self):
        baseline = {"registry_command": ["a"]}
        current = {"registry_command": ["a"], "tps_median": 9.8, "wired": {"steady_gib": 40.0},
                   "needle_hit": True}
        failures = qwen_gate.evaluate(baseline, current)
        self.assertEqual(len(failures), 1)
        self.assertIn("baseline lacks full-tier data", failures[0])


if __name__ == "__main__":
    unittest.main()
