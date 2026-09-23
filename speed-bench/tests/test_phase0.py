import json
import os
import stat
import sys
import tempfile
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "v41"))
import phase0  # noqa: E402

PROFILE = ("ds4: V4.1 decode profile: tokens=64 step_ms=100.000 engram_ms=1.000 "
           "gpu_busy_ms=60.000 pread_ms=15.000 pread_mib=3.000 hits=50.00 misses=10.00\n")
PROFILE_LATER = PROFILE.replace("tokens=64", "tokens=128").replace("step_ms=100.000", "step_ms=90.000")
CACHE = ("ds4:   streaming expert cache budget=7000 experts entries=6990 expert=1.42 MiB "
         "target=9.70 GiB live=9.69 GiB, hits=900 misses=100 hit_rate=0.900 wraps=0\n")
CSV = ("ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,"
       "gen_steady_tps,kvcache_bytes\n4096,4096,120.5,512,10.2,150.0,511,10.4,123456\n")
WIRED = {"steady_gib": 30.0, "peak_gib": 32.0, "n": 10}


class PromptTest(unittest.TestCase):
    def test_prompts(self):
        texts = phase0.build_prompts(phase0.ROOT)
        self.assertEqual(sorted(texts), ["code", "docs", "it", "switch"])
        for name, text in texts.items():
            self.assertEqual(len(text), phase0.PROMPT_CHARS, name)
        c = phase0.CHUNK
        self.assertEqual(texts["switch"][:c], texts["code"][:c])
        self.assertEqual(texts["switch"][c:2 * c], texts["docs"][:c])
        self.assertEqual(texts["switch"][2 * c:3 * c], texts["it"][:c])


class PlanTest(unittest.TestCase):
    def test_plans(self):
        self.assertEqual(len(phase0.PLANS["speed"]), 12)
        self.assertEqual(len(phase0.PLANS["locality"]), 4)
        self.assertTrue(all(spec[4] for spec in phase0.PLANS["locality"]))

    def test_bench_cmd(self):
        cmd = phase0.bench_cmd("/b", "/m.gguf", "/p.txt", 8192, 512, 16, "/o.csv")
        self.assertEqual(cmd[0], "/b/ds4-bench")
        for flag in ("--teacher-forced-decode", "--ssd-streaming", "16GB", "8192", "/o.csv"):
            self.assertIn(flag, cmd)


class ParseTest(unittest.TestCase):
    def test_profile_takes_last_line_and_host_gap(self):
        p = phase0.parse_profile(PROFILE + PROFILE_LATER)
        self.assertEqual(p["tokens"], 128)
        self.assertAlmostEqual(p["host_gap_ms"], 90.0 - 60.0 - 15.0 - 1.0)

    def test_profile_missing(self):
        self.assertIsNone(phase0.parse_profile("nothing"))

    def test_cache(self):
        self.assertEqual(phase0.parse_cache(CACHE), {"cache_experts": 7000, "cache_hit_rate": 0.9})

    def test_bench_csv(self):
        b = phase0.parse_bench_csv(CSV)
        self.assertEqual(b["gen_steady_tps"], 10.4)
        self.assertEqual(b["prefill_tps"], 120.5)

    def test_combine(self):
        spec = ("switch", 4096, 8, 512, False)
        row = phase0.combine(spec, phase0.parse_bench_csv(CSV), phase0.parse_profile(PROFILE),
                             phase0.parse_cache(CACHE), WIRED, 12.0, contaminated=False)
        self.assertAlmostEqual(row["decode_hit_rate"], 50 / 60)
        self.assertEqual(row["wired_steady_gib"], 30.0)
        self.assertFalse(row["contaminated"])


def fake_bin(tmp, body):
    path = os.path.join(tmp, "ds4-bench")
    with open(path, "w") as fp:
        fp.write("#!/bin/sh\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return tmp


class RunTest(unittest.TestCase):
    SPEC = ("switch", 4096, 8, 512, False)

    def run_one(self, tmp, body, running=lambda: ""):
        os.makedirs(os.path.join(tmp, "prompts"), exist_ok=True)
        open(os.path.join(tmp, "prompts", "switch.txt"), "w").close()
        return phase0.run_one(fake_bin(tmp, body), "/m.gguf", os.path.join(tmp, "prompts"), tmp,
                              self.SPEC, running=running, swap=lambda: 0.0,
                              sampler=lambda: _FakeSampler())

    def test_failed_run_leaves_no_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                self.run_one(tmp, "exit 3\n")
            self.assertFalse(any(n.endswith(".result.json") for n in os.listdir(tmp)))

    def test_refuses_when_busy(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                self.run_one(tmp, "exit 0\n", running=lambda: "9 ds4-server")

    def test_success_and_contamination(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_line = CSV.replace("\n", "\\n")
            body = (f'for a; do last=$a; done\nprintf "{csv_line}" > "$last"\n'
                    f'printf "%s" "{PROFILE.strip()}" >&2\n')
            calls = iter(["", "7 ds4-server"])   # free before, busy after
            row = self.run_one(tmp, body, running=lambda: next(calls))
            self.assertTrue(row["contaminated"])
            self.assertEqual(row["gen_steady_tps"], 10.4)
            self.assertIsNone(self.run_one(tmp, "exit 9\n"))   # done -> skipped

    def test_dry_run_touches_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = phase0.run_one("/b", "/m.gguf", "/p", tmp, self.SPEC, dry_run=True)
            self.assertIsNone(out)
            self.assertEqual(os.listdir(tmp), [])

    def test_missing_csv_exits_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                self.run_one(tmp, "exit 0\n")
            self.assertFalse(any(n.endswith(".result.json") for n in os.listdir(tmp)))


class _FakeSampler:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def summary(self):
        return WIRED


class ReportTest(unittest.TestCase):
    def test_report_marks_swapped_and_contaminated(self):
        rows = [
            {"workload": "switch", "ctx": 4096, "cache_gb": 8, "gen_steady_tps": 10.0,
             "step_ms": 100.0, "gpu_busy_ms": 60.0, "pread_ms": 15.0, "engram_ms": 1.0,
             "host_gap_ms": 24.0, "decode_hit_rate": 0.9, "wired_steady_gib": 30.0,
             "swap_delta_mib": 900.0, "contaminated": False},
            {"workload": "switch", "ctx": 8192, "cache_gb": 8, "gen_steady_tps": 11.0,
             "step_ms": 91.0, "gpu_busy_ms": 60.0, "pread_ms": 10.0, "engram_ms": 1.0,
             "host_gap_ms": 20.0, "decode_hit_rate": 0.92, "wired_steady_gib": 30.5,
             "swap_delta_mib": 0.0, "contaminated": True},
        ]
        bytes_json = {"per_token": {"resident": 8_000_000_000, "embedding_row": 0,
                                    "routed": 2_000_000_000}, "gbps": 290.0}
        locality = [{"name": "code", "lru_hit": {"8": 0.9}, "pair_overlap": 0.3,
                     "union_cover": {"1": 0.3, "4": 0.6}, "new_per_token": 2.0}]
        text = phase0.report(rows, bytes_json, locality)
        self.assertIn("swapped", text)
        self.assertIn("contaminated", text)
        self.assertIn("34.5 ms/token", text)   # (8e9 + 2e9) / 290e9
        self.assertIn("| code |", text)
        self.assertIn("Best clean run: none", text)


if __name__ == "__main__":
    unittest.main()
