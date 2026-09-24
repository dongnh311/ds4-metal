import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v41"))
import router_locality as rl  # noqa: E402

LOG = """# pos layer e0..e1
10 0 1 2
11 0 1 2
12 0 3 4
13 0 1 2
"""


class ParseTest(unittest.TestCase):
    def test_parse(self):
        tokens = rl.parse_router_log(LOG.splitlines())
        self.assertEqual(tokens, [[(1, 2)], [(1, 2)], [(3, 4)], [(1, 2)]])

    def test_layers_out_of_order(self):
        with self.assertRaises(ValueError):
            rl.parse_router_log(["5 1 1 2", "5 0 3 4"])

    def test_two_layers(self):
        tokens = rl.parse_router_log(["1 0 7 8", "1 1 9 10", "2 0 7 8", "2 1 9 11"])
        self.assertEqual(tokens, [[(7, 8), (9, 10)], [(7, 8), (9, 11)]])


class LruTest(unittest.TestCase):
    def test_hit_rate_by_hand(self):
        tokens = rl.parse_router_log(LOG.splitlines())
        # budget 2 units: t1 hits both, t2 evicts 1 and 2, t3 misses both -> 2/8
        self.assertEqual(rl.lru_hit_rates(tokens, [1], [2]), {2: 0.25})
        # budget 4 units holds everything: only the 4 first-time loads miss -> 4/8
        self.assertEqual(rl.lru_hit_rates(tokens, [1], [4]), {4: 0.5})

    def test_warmup_excludes_early_tokens(self):
        tokens = rl.parse_router_log(LOG.splitlines())
        # count only t2, t3 with budget 4: t2 misses 3,4; t3 hits 1,2 -> 2/4
        self.assertEqual(rl.lru_hit_rates(tokens, [1], [4], warmup=2), {4: 0.5})


class OverlapTest(unittest.TestCase):
    def test_overlap_and_cover(self):
        tokens = rl.parse_router_log(LOG.splitlines())
        s = rl.overlap_stats(tokens, ks=(1, 2))
        self.assertAlmostEqual(s["pair_overlap"], 1 / 3)
        self.assertAlmostEqual(s["union_cover"]["1"], 1 / 3)
        self.assertAlmostEqual(s["union_cover"]["2"], 2 / 3)

    def test_new_per_token(self):
        tokens = rl.parse_router_log(LOG.splitlines())
        # fresh counts 2,0,2,0 -> second half [2,0] -> 1.0
        self.assertEqual(rl.new_experts_per_token(tokens), 1.0)


if __name__ == "__main__":
    unittest.main()
