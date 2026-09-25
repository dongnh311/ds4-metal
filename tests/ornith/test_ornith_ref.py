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
