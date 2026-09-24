import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v41"))
import stages  # noqa: E402

LINES = """\
ds4: gpu stage buffer what=decode part=v41 stage=pre_attn layer=0 pos=10 tokens=1 start_ms=0.000 gpu_ms=1.000
ds4: gpu stage buffer what=decode part=v41 stage=after_moe layer=0 pos=10 tokens=1 start_ms=0.500 gpu_ms=2.000
ds4: gpu stage buffer what=decode part=v41 stage=moe_routed layer=0 pos=10 tokens=1 start_ms=4.000 gpu_ms=1.000
ds4: gpu stage buffer what=decode part=v41 stage=pre_attn layer=0 pos=11 tokens=1 start_ms=0.000 gpu_ms=3.000
ds4: gpu stage buffer what=prefill part=v41 stage=pre_attn layer=0 pos=0 tokens=8 start_ms=0.000 gpu_ms=9.000
"""


class StagesTest(unittest.TestCase):
    def test_exclusive_time_drops_overlap_and_counts_idle(self):
        res = stages.exclusive(LINES.splitlines(), min_pos=10)
        self.assertEqual(res["tokens"], 2)
        # pos 10: pre_attn 1.0; after_moe overlaps 0.5..1.0 -> 1.5 exclusive; idle 2.5..4.0 = 1.5
        self.assertAlmostEqual(res["stages"]["pre_attn"], (1.0 + 3.0) / 2)
        self.assertAlmostEqual(res["stages"]["after_moe"], 1.5 / 2)
        self.assertAlmostEqual(res["stages"]["moe_routed"], 1.0 / 2)
        self.assertAlmostEqual(res["idle_ms"], 1.5 / 2)
        self.assertAlmostEqual(res["busy_ms"], (1.0 + 1.5 + 1.0 + 3.0) / 2)

    def test_min_pos_skips_warmup_tokens(self):
        self.assertEqual(stages.exclusive(LINES.splitlines(), min_pos=11)["tokens"], 1)


if __name__ == "__main__":
    unittest.main()
