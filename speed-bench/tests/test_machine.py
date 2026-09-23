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
