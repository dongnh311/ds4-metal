"""Unit tests for the Ornith reference comparison (python3 -m unittest)."""
import os
import sys
import tempfile
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

    def test_tie_step_swap_within_tolerance_passes(self):
        ref = [step(5, (5, -0.10), (7, -3.0)), step(1, (1, -0.69), (2, -0.70))]
        got = [step(5, (5, -0.10), (7, -3.0)), step(2, (2, -0.68), (1, -0.71))]
        res = r.compare(ref, got, tol=0.05, tie=0.05)
        self.assertTrue(res["ok"])
        self.assertEqual(res["stopped_at_tie"], 1)
        self.assertEqual(res["compared"], 1)
        self.assertAlmostEqual(res["max_delta"], 0.02)

    def test_tie_step_probable_delta_above_tolerance_fails(self):
        ref = [step(5, (5, -0.10), (7, -3.0)), step(1, (1, -0.69), (2, -0.70))]
        got = [step(5, (5, -0.10), (7, -3.0)), step(1, (1, -0.40), (2, -1.20))]
        res = r.compare(ref, got, tol=0.05, tie=0.05)
        self.assertFalse(res["ok"])
        self.assertEqual(res["stopped_at_tie"], 1)
        self.assertEqual(res["compared"], 1)
        self.assertEqual(res["first_mismatch"], 1)
        self.assertAlmostEqual(res["max_delta"], 0.5)

    def test_logprob_delta_above_tolerance_fails(self):
        ref = [step(5, (5, -0.1), (7, -3.0))]
        got = [step(5, (5, -0.6), (7, -3.0))]
        res = r.compare(ref, got, tol=0.05, tie=0.05)
        self.assertFalse(res["ok"])
        self.assertAlmostEqual(res["max_delta"], 0.5)

    def test_tail_delta_is_ignored(self):
        ref = [step(5, (5, -0.1), (7, -3.0))]
        got = [step(5, (5, -0.1), (7, -6.0))]
        res = r.compare(ref, got, tol=0.05, tie=0.05)
        self.assertTrue(res["ok"])

    def test_shorter_output_fails_unless_stopped(self):
        ref = [step(5, (5, -0.1), (7, -3.0)), step(9, (9, -0.2), (2, -2.5))]
        got = [step(5, (5, -0.1), (7, -3.0))]
        res = r.compare(ref, got, tol=0.05, tie=0.05)
        self.assertFalse(res["ok"])

    def test_checked_counts_probable_token_pairs(self):
        ref = [step(5, (5, -0.1), (7, -3.0)), step(9, (9, -0.5), (2, -1.2))]
        res = r.compare(ref, ref, tol=0.05, tie=0.05)
        self.assertTrue(res["ok"])
        self.assertEqual(res["checked"], 3)

    def test_checked_is_zero_when_nothing_is_probable(self):
        # A flat reference (every log-prob below PROBABLE) that ties at step 0
        # compares nothing: this is what the gate reports as vacuous.
        ref = [step(5, (5, -2.70), (7, -2.72))]
        res = r.compare(ref, ref, tol=0.05, tie=0.05)
        self.assertTrue(res["ok"])
        self.assertEqual(res["checked"], 0)

    def test_reference_probable_token_missing_from_ds4_fails(self):
        ref = [step(5, (5, -0.3), (7, -1.5))]
        got = [step(5, (5, -0.3), (8, -3.0))]
        res = r.compare(ref, got, tol=0.05, tie=0.05)
        self.assertFalse(res["ok"])
        self.assertEqual(res["first_mismatch"], 0)
        self.assertIn("token 7", res["reason"])
        self.assertIn("missing from ds4", res["reason"])

    def test_ds4_probable_token_missing_from_reference_fails(self):
        ref = [step(5, (5, -0.3), (7, -3.0))]
        got = [step(5, (5, -0.3), (7, -3.0), (8, -1.5))]
        res = r.compare(ref, got, tol=0.05, tie=0.05)
        self.assertFalse(res["ok"])
        self.assertEqual(res["first_mismatch"], 0)
        self.assertIn("token 8", res["reason"])
        self.assertIn("missing from the reference", res["reason"])

    def test_probable_token_missing_at_tie_step_fails(self):
        ref = [step(5, (5, -0.10), (7, -3.0)), step(1, (1, -0.69), (2, -0.70))]
        got = [step(5, (5, -0.10), (7, -3.0)), step(1, (1, -0.69), (3, -0.70))]
        res = r.compare(ref, got, tol=0.05, tie=0.05)
        self.assertFalse(res["ok"])
        self.assertEqual(res["stopped_at_tie"], 1)
        self.assertEqual(res["first_mismatch"], 1)

    def test_deltas_cover_reference_probable_tokens(self):
        # Token 8 is probable only on the ds4 side but is in both lists: it
        # passes the presence check, and the delta stays over the reference's
        # probable tokens, the set the tolerance was calibrated on.
        ref = [step(5, (5, -0.3), (8, -2.1))]
        got = [step(5, (5, -0.3), (8, -1.9))]
        res = r.compare(ref, got, tol=0.05, tie=0.05)
        self.assertTrue(res["ok"])
        self.assertEqual(res["checked"], 1)
        self.assertEqual(res["max_delta"], 0.0)

    def test_verdict_reports_vacuous_comparison(self):
        self.assertEqual(r.verdict({"ok": True, "checked": 4}), "ok")
        self.assertEqual(r.verdict({"ok": False, "checked": 4}), "FAIL")
        self.assertEqual(r.verdict({"ok": True, "checked": 0}), "FAIL vacuous")

    def test_calibrate_uses_floors(self):
        metal = [[step(5, (5, -0.1), (7, -3.0))]]
        cpu = [[step(5, (5, -0.101), (7, -3.002))]]
        cal = r.calibrate(metal, cpu)
        self.assertAlmostEqual(cal["tol"], 0.05)
        self.assertAlmostEqual(cal["tie"], 0.05)
        self.assertAlmostEqual(cal["observed_max_delta"], 0.001, places=6)


class PromptTextTest(unittest.TestCase):
    def test_repeat_chars_appends_the_opening(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "doc.txt"), "w", encoding="utf-8") as f:
                f.write("abcdefghijklmnop")
            p = {"name": "x", "file": "doc.txt", "chars": 10, "repeat_chars": 4, "n_predict": 8}
            self.assertEqual(r.prompt_text(p, root), "abcdefghij\n\nabcd")
            del p["repeat_chars"]
            self.assertEqual(r.prompt_text(p, root), "abcdefghij")


class ServerGroupsTest(unittest.TestCase):
    PROMPTS = [{"name": "a", "text": "x", "n_predict": 1},
               {"name": "long", "file": "f", "chars": 9, "llama_ubatch": 1, "n_predict": 1},
               {"name": "b", "file": "g", "chars": 9, "n_predict": 1}]

    def test_default_batch_prompts_share_the_first_server(self):
        groups = r.llama_server_groups(self.PROMPTS[:1] + self.PROMPTS[2:], cpu=False)
        self.assertEqual([(ub, [p["name"] for p in ps]) for ub, ps in groups], [(None, ["a", "b"])])

    def test_pinned_ubatch_gets_its_own_server(self):
        groups = r.llama_server_groups(self.PROMPTS, cpu=False)
        self.assertEqual([(ub, [p["name"] for p in ps]) for ub, ps in groups],
                         [(None, ["a", "b"]), (1, ["long"])])

    def test_cpu_skips_file_backed_prompts(self):
        groups = r.llama_server_groups(self.PROMPTS, cpu=True)
        self.assertEqual([(ub, [p["name"] for p in ps]) for ub, ps in groups], [(None, ["a"])])


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
