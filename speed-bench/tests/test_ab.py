import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v41"))
import ab  # noqa: E402


def row(tps, **kw):
    r = {"gen_steady_tps": tps, "step_ms": 1000.0 / tps, "gpu_busy_ms": 50.0,
         "pread_ms": 10.0, "readahead_ms": 5.0, "host_ms": 20.0,
         "decode_hit_rate": 0.8, "wired_steady_gib": 29.0, "contaminated": False}
    r.update(kw)
    return r


class AbTest(unittest.TestCase):
    def test_order_suffixes_and_ratio(self):
        calls = []

        def run(side, suffix):
            calls.append((side, suffix))
            return row(10.0 if side == "a" else 11.0)

        res = ab.run_ab(run, "q")
        self.assertEqual(calls, [("a", "-q-0a"), ("b", "-q-1b"), ("b", "-q-2b"), ("a", "-q-3a")])
        self.assertAlmostEqual(res["ratio"], 1.1)
        self.assertEqual(res["runs"]["a"], [10.0, 10.0])
        self.assertAlmostEqual(res["terms"]["b"]["gpu_busy_ms"], 50.0)
        self.assertFalse(res["contaminated"])

    def test_contamination_propagates(self):
        res = ab.run_ab(lambda side, suffix: row(10.0, contaminated=side == "b"), "c")
        self.assertTrue(res["contaminated"])

    def test_parse_env(self):
        self.assertEqual(ab.parse_env(["A=1", "B="]), {"A": "1", "B": ""})
        with self.assertRaises(SystemExit):
            ab.parse_env(["NOEQUALS"])


if __name__ == "__main__":
    unittest.main()
