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
