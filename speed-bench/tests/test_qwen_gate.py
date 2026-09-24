import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "qwen-regression"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import qwen_gate  # noqa: E402
import wired  # noqa: E402

HIGH_VM_STAT = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                "Pages free:                                    10623.\n"
                "Pages wired down:                             600000.\n")   # ~9.16 GiB
FREE_VM_STAT = HIGH_VM_STAT.replace("600000", "1000")


class _Stop(Exception):
    """Raised by a patched registry_command so run() stops before starting a server."""

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


class RunIdleRefusalTest(unittest.TestCase):
    def test_refuses_before_server_starts_when_wired_high(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                qwen_gate.run(None, tmp, False, 650_000, ds4_running=lambda: "",
                              idle_read=lambda: HIGH_VM_STAT, idle_timeout=0)
            # No server.log / kv dir means run() never got past the idle check.
            self.assertFalse(os.path.exists(os.path.join(tmp, "server.log")))

    def test_refuses_when_ds4_already_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                qwen_gate.run(None, tmp, False, 650_000, ds4_running=lambda: "9 ds4-server")


class RunSetupTest(unittest.TestCase):
    def setUp(self):
        self.seen = {}
        self.orig = qwen_gate.registry_command

        def fake(registry, bin_dir, port, kv):
            self.seen["kv"] = kv
            raise _Stop()
        qwen_gate.registry_command = fake
        self.cwd = os.getcwd()

    def tearDown(self):
        qwen_gate.registry_command = self.orig
        os.chdir(self.cwd)

    def test_relative_out_gives_absolute_kv_dir(self):
        # The server runs in PROD's cwd; a relative KV dir would land in the PROD tree.
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            with self.assertRaises(_Stop):
                qwen_gate.run(None, "rel/out", False, 1000, ds4_running=lambda: "",
                              idle_read=lambda: FREE_VM_STAT)
            self.assertTrue(os.path.isabs(self.seen["kv"]))
            self.assertEqual(self.seen["kv"],
                             os.path.join(os.path.realpath(tmp), "rel", "out", "kv"))

    def test_waits_for_previous_server_wiring_to_drain(self):
        with tempfile.TemporaryDirectory() as tmp:
            reads = iter([HIGH_VM_STAT, HIGH_VM_STAT, FREE_VM_STAT])
            with self.assertRaises(_Stop):
                qwen_gate.run(None, tmp, False, 1000, ds4_running=lambda: "",
                              idle_read=lambda: next(reads), idle_timeout=10, idle_interval=0.001)


class PairedSpeedTest(unittest.TestCase):
    def test_prod_runs_registry_binary_and_branch_runs_bin(self):
        calls = []

        def measure(bin_dir, out, tag, port):
            calls.append((bin_dir, tag, port))
            return [40.0, 41.0, 42.0]

        ab = qwen_gate.paired_speed("/branch", "/out", 1234, measure=measure)
        self.assertEqual(calls, [(None, "0-prod", 1234), ("/branch", "1-branch", 1234),
                                 ("/branch", "2-branch", 1234), (None, "3-prod", 1234)])
        self.assertEqual(ab["prod"], [41.0, 41.0])


class CheckPreflightTest(unittest.TestCase):
    def test_refuses_same_out_and_baseline(self):
        self.assertIsNotNone(qwen_gate.check_preflight("/a", "/a"))

    def test_refuses_missing_baseline_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = os.path.join(tmp, "baseline")
            os.makedirs(baseline)
            out = os.path.join(tmp, "out")
            self.assertIsNotNone(qwen_gate.check_preflight(out, baseline))

    def test_passes_when_baseline_has_result_and_differs_from_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = os.path.join(tmp, "baseline")
            os.makedirs(baseline)
            with open(os.path.join(baseline, "result.json"), "w") as fp:
                fp.write("{}")
            out = os.path.join(tmp, "out")
            self.assertIsNone(qwen_gate.check_preflight(out, baseline))


class NeedleSourceTest(unittest.TestCase):
    def test_needle_source_is_pinned_text_file(self):
        self.assertEqual(qwen_gate.NEEDLE_SOURCE, "speed-bench/promessi_sposi.txt")


class SpeedAbTest(unittest.TestCase):
    def test_equal_speeds_pass(self):
        ab = {"prod": [43.92, 42.88], "branch": [43.31, 44.01]}
        self.assertEqual(qwen_gate.speed_ab_failures(ab), [])

    def test_slower_branch_fails(self):
        failures = qwen_gate.speed_ab_failures({"prod": [43.0, 43.0], "branch": [40.0, 41.0]})
        self.assertEqual(len(failures), 1)
        self.assertIn("paired A/B", failures[0])

    def test_interleaved_order_and_medians(self):
        calls = []

        def measure(which):
            calls.append(which)
            return [40.0, 42.0, 41.0] if which == "prod" else [44.0, 43.0, 45.0]

        ab = qwen_gate.speed_ab(measure)
        self.assertEqual(calls, ["prod", "branch", "branch", "prod"])
        self.assertEqual(ab["prod"], [41.0, 41.0])
        self.assertEqual(ab["branch"], [44.0, 44.0])

    def test_empty_side_is_a_failure_not_a_crash(self):
        failures = qwen_gate.speed_ab_failures({"prod": [43.0], "branch": []})
        self.assertEqual(len(failures), 1)
        self.assertIn("no branch", failures[0])

    def test_failure_lists_per_server_medians(self):
        failures = qwen_gate.speed_ab_failures({"prod": [44.1, 41.9], "branch": [40.0, 39.5]})
        self.assertIn("44.10/41.90", failures[0])
        self.assertIn("40.00/39.50", failures[0])

    def test_evaluate_prefers_paired_speed_over_stored_baseline(self):
        base = {"registry_command": ["a"], "tps_median": 42.8, "wired": {"steady_gib": 45.8}}
        cur = {"registry_command": ["a"], "tps_median": 38.7, "wired": {"steady_gib": 45.8},
               "needle_hit": True, "speed_ab": {"prod": [43.9, 42.9], "branch": [43.3, 44.0]}}
        self.assertEqual(qwen_gate.evaluate(base, cur), [])

    def test_evaluate_reports_paired_speed_failure(self):
        base = {"registry_command": ["a"], "tps_median": 42.8, "wired": {"steady_gib": 45.8}}
        cur = {"registry_command": ["a"], "tps_median": 44.0, "wired": {"steady_gib": 45.8},
               "needle_hit": True, "speed_ab": {"prod": [44.0, 44.0], "branch": [40.0, 40.0]}}
        failures = qwen_gate.evaluate(base, cur)
        self.assertEqual(len(failures), 1)
        self.assertIn("paired A/B", failures[0])


if __name__ == "__main__":
    unittest.main()
