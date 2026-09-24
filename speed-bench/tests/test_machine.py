import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import machine  # noqa: E402


def _proc(returncode, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class MachineTest(unittest.TestCase):
    def test_ds4_running_reports_lines(self):
        self.assertEqual(machine.ds4_running(pgrep=lambda: "123 ds4-server --metal\n"),
                         "123 ds4-server --metal")

    def test_ds4_running_empty_when_free(self):
        self.assertEqual(machine.ds4_running(pgrep=lambda: ""), "")

    def test_real_pgrep_rc_0_returns_lines(self):
        out = machine.ds4_running(run=lambda *a, **k: _proc(0, "123 ds4-server\n"))
        self.assertEqual(out, "123 ds4-server")

    def test_real_pgrep_rc_1_means_free(self):
        out = machine.ds4_running(run=lambda *a, **k: _proc(1, ""))
        self.assertEqual(out, "")

    def test_real_pgrep_other_rc_raises(self):
        with self.assertRaises(RuntimeError):
            machine.ds4_running(run=lambda *a, **k: _proc(2, "", "pgrep: bad option"))

    def test_parse_swap_used(self):
        text = "vm.swapusage: total = 2048.00M  used = 938.19M  free = 1109.81M  (encrypted)"
        self.assertEqual(machine.parse_swap_used_mib(text), 938.19)

    def test_parse_swap_rejects_garbage(self):
        with self.assertRaises(ValueError):
            machine.parse_swap_used_mib("nothing")


if __name__ == "__main__":
    unittest.main()
