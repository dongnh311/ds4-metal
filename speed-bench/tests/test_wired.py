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

    def test_sampler_collects_timed(self):
        with wired.WiredSampler(interval=0.01, read=lambda: SAMPLE) as ws:
            time.sleep(0.05)
        self.assertGreaterEqual(len(ws.timed), 2)
        for t, b in ws.timed:
            self.assertIsInstance(t, float)
            self.assertEqual(b, 262144 * 16384)


class WindowSummaryTest(unittest.TestCase):
    def test_picks_only_in_window_samples(self):
        g = wired.GIB
        timed = [(0.0, 1 * g), (1.0, 5 * g), (2.0, 6 * g), (3.0, 7 * g), (4.0, 40 * g)]
        s = wired.window_summary(timed, 1.0, 3.0)
        self.assertEqual(s["steady_gib"], 6.0)
        self.assertEqual(s["peak_gib"], 40.0)
        self.assertEqual(s["window"], "decode")
        self.assertEqual(s["n"], 5)

    def test_fallback_when_too_few_in_window(self):
        g = wired.GIB
        timed = [(0.0, 1 * g), (1.0, 2 * g), (2.0, 3 * g), (3.0, 4 * g), (4.0, 5 * g)]
        s = wired.window_summary(timed, 10.0, 11.0)
        self.assertEqual(s["window"], "fallback")
        # median of second half of all samples: [3, 4, 5] * g -> 4 GiB
        self.assertEqual(s["steady_gib"], 4.0)
        self.assertEqual(s["peak_gib"], 5.0)
        self.assertEqual(s["n"], 5)

    def test_fallback_respects_min_samples(self):
        g = wired.GIB
        timed = [(0.0, 1 * g), (1.0, 2 * g), (2.0, 10 * g)]
        s = wired.window_summary(timed, 1.0, 2.0, min_samples=3)
        self.assertEqual(s["window"], "fallback")


class IdleGibTest(unittest.TestCase):
    def test_idle_gib_from_one_sample(self):
        self.assertEqual(wired.idle_gib(read=lambda: SAMPLE), 262144 * 16384 / wired.GIB)

    def test_idle_wired_limit_constant(self):
        self.assertEqual(wired.IDLE_WIRED_LIMIT_GIB, 8.0)

    def _fake_time(self):
        now = {"t": 0.0}
        return (lambda s: now.__setitem__("t", now["t"] + s)), (lambda: now["t"])

    def test_wait_idle_returns_once_wired_settles(self):
        high = SAMPLE.replace("262144", "655360")   # 10 GiB, the previous run still tearing down
        reads = iter([high, high, SAMPLE])
        sleep, clock = self._fake_time()
        got = wired.wait_idle_gib(read=lambda: next(reads), timeout=60, interval=2,
                                  sleep=sleep, clock=clock)
        self.assertEqual(got, 4.0)
        self.assertEqual(clock(), 4)

    def test_wait_idle_gives_up_after_timeout(self):
        high = SAMPLE.replace("262144", "655360")
        sleep, clock = self._fake_time()
        got = wired.wait_idle_gib(read=lambda: high, timeout=10, interval=2,
                                  sleep=sleep, clock=clock)
        self.assertEqual(got, 10.0)
        self.assertEqual(clock(), 10)


if __name__ == "__main__":
    unittest.main()
