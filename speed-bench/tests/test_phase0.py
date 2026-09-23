import json
import os
import stat
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "v41"))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
import phase0  # noqa: E402
import wired  # noqa: E402

PROFILE = ("ds4: V4.1 decode profile: tokens=64 step_ms=100.000 engram_ms=1.000 "
           "gpu_busy_ms=60.000 pread_ms=15.000 pread_mib=3.000 hits=50.00 misses=10.00\n")
PROFILE_LATER = PROFILE.replace("tokens=64", "tokens=128").replace("step_ms=100.000", "step_ms=90.000")
CACHE = ("ds4:   streaming expert cache budget=7000 experts entries=6990 expert=1.42 MiB "
         "target=9.70 GiB live=9.69 GiB, hits=900 misses=100 hit_rate=0.900 wraps=0\n")
CSV = ("ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,"
       "gen_steady_tps,kvcache_bytes\n4096,4096,120.5,512,10.2,150.0,511,10.4,123456\n")
WIRED = {"steady_gib": 30.0, "peak_gib": 32.0, "n": 10, "window": "decode"}
IDLE_VM_STAT = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                "Pages free:                                    10623.\n"
                "Pages wired down:                             600000.\n")   # ~9.16 GiB
FREE_VM_STAT = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                "Pages free:                                    10623.\n"
                "Pages wired down:                               1000.\n")   # ~0.015 GiB


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

    def test_combine_carries_wired_window(self):
        spec = ("switch", 4096, 8, 512, False)
        row = phase0.combine(spec, phase0.parse_bench_csv(CSV), phase0.parse_profile(PROFILE),
                             phase0.parse_cache(CACHE), WIRED, 12.0, contaminated=False)
        self.assertEqual(row["wired_window"], "decode")


def fake_bin(tmp, body):
    path = os.path.join(tmp, "ds4-bench")
    with open(path, "w") as fp:
        fp.write("#!/bin/sh\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return tmp


class RunTest(unittest.TestCase):
    SPEC = ("switch", 4096, 8, 512, False)
    TAG = "switch-c4096-g8-n512"

    def run_one(self, tmp, body, running=lambda: "", idle_read=None):
        os.makedirs(os.path.join(tmp, "prompts"), exist_ok=True)
        open(os.path.join(tmp, "prompts", "switch.txt"), "w").close()
        return phase0.run_one(fake_bin(tmp, body), "/m.gguf", os.path.join(tmp, "prompts"), tmp,
                              self.SPEC, running=running, swap=lambda: 0.0,
                              sampler=lambda: _FakeSampler(),
                              idle_read=idle_read or (lambda: FREE_VM_STAT))

    def csv_body(self):
        csv_line = CSV.replace("\n", "\\n")
        return (f'for a; do last=$a; done\nprintf "{csv_line}" > "$last"\n'
                f'printf "%s" "{PROFILE.strip()}" >&2\n')

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
            calls = iter(["", "7 ds4-server"])   # free before, busy after
            row = self.run_one(tmp, self.csv_body(), running=lambda: next(calls))
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

    def test_row_carries_wired_window_and_steady(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = self.run_one(tmp, self.csv_body())
            self.assertIn(row["wired_window"], ("decode", "fallback"))
            self.assertIn("wired_steady_gib", row)
            self.assertIn("wired_peak_gib", row)

    def test_refuses_when_idle_wired_high(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                self.run_one(tmp, "exit 0\n", idle_read=lambda: IDLE_VM_STAT)

    def test_row_carries_idle_wired(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = self.run_one(tmp, self.csv_body())
            self.assertIn("wired_idle_gib", row)
            self.assertLess(row["wired_idle_gib"], wired.IDLE_WIRED_LIMIT_GIB)

    def test_mid_run_foreign_process_marks_contaminated(self):
        with tempfile.TemporaryDirectory() as tmp:
            body = "sleep 0.2\n" + self.csv_body()
            calls = {"n": 0}

            def running():
                calls["n"] += 1
                return "" if calls["n"] == 1 else "42 other-process"

            row = self.run_one(tmp, body, running=running)
            self.assertTrue(row["contaminated"])

    def test_mid_run_own_process_not_contaminated(self):
        with tempfile.TemporaryDirectory() as tmp:
            body = "sleep 0.2\n" + self.csv_body()
            csv_path = os.path.join(tmp, self.TAG + ".csv")
            t0 = time.monotonic()

            def running():
                dt = time.monotonic() - t0
                # Pre-check and post-run check see a free machine; only the
                # mid-run poll (during the child's sleep) sees our own line.
                if dt < 0.01 or dt >= 0.18:
                    return ""
                return f"9 ds4-bench --csv {csv_path}"

            row = self.run_one(tmp, body, running=running)
            self.assertFalse(row["contaminated"])

    def test_env_does_not_leak_stray_router_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["DS4_V41_ROUTER_LOG"] = "/should/not/be/used"
            try:
                row = self.run_one(tmp, self.csv_body())
            finally:
                del os.environ["DS4_V41_ROUTER_LOG"]
            self.assertNotIn("DS4_V41_ROUTER_LOG", row["ds4_env"])
            self.assertIn("DS4_V41_DECODE_PROFILE", row["ds4_env"])

    def test_ds4_env_not_in_fields_or_csv(self):
        self.assertNotIn("ds4_env", phase0.FIELDS)

    def test_result_written_atomically_no_leftover_tmp(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_one(tmp, self.csv_body())
            names = os.listdir(tmp)
            self.assertTrue(any(n.endswith(".result.json") for n in names))
            self.assertFalse(any(n.endswith(".result.json.tmp") for n in names))

    def test_router_log_renamed_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "prompts"), exist_ok=True)
            open(os.path.join(tmp, "prompts", "code.txt"), "w").close()
            spec = ("code", 4096, 8, 2000, True)
            tag = "code-c4096-g8-n2000"
            router_log = os.path.join(tmp, tag + ".router.log")
            with open(router_log, "w") as fp:
                fp.write("stale\n")
            with self.assertRaises(SystemExit):
                phase0.run_one(fake_bin(tmp, "exit 0\n"), "/m.gguf", os.path.join(tmp, "prompts"),
                               tmp, spec, running=lambda: "", swap=lambda: 0.0,
                               sampler=lambda: _FakeSampler(), idle_read=lambda: FREE_VM_STAT)
            self.assertFalse(os.path.exists(router_log))
            self.assertTrue(os.path.exists(router_log + ".failed"))


class _FakeSampler:
    def __init__(self):
        self.interval = 0.03
        self.timed = [(0.0, 30 * wired.GIB), (0.1, 30 * wired.GIB), (0.2, 32 * wired.GIB)]

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

    def test_negative_host_gap_flags_overlap_and_is_documented(self):
        rows = [{"workload": "switch", "ctx": 4096, "cache_gb": 8, "gen": 512,
                 "gen_steady_tps": 10.0, "prefill_tps": 100.0, "gen_first_ms": 150.0,
                 "step_ms": 50.0, "gpu_busy_ms": 60.0, "pread_ms": 15.0, "engram_ms": 1.0,
                 "host_gap_ms": -26.0, "decode_hit_rate": 0.9, "wired_steady_gib": 30.0,
                 "swap_delta_mib": 0.0, "contaminated": False, "router_log": None}]
        bytes_json = {"per_token": {"resident": 0, "embedding_row": 0, "routed": 0}, "gbps": 290.0}
        text = phase0.report(rows, bytes_json, [])
        self.assertIn("overlap", text)
        self.assertIn("-26.0", text)
        self.assertIn("host_gap_ms", text)   # residual documented in the report header

    def test_gen_and_log_columns_and_best_clean_excludes_router_log(self):
        rows = [
            {"workload": "switch", "ctx": 4096, "cache_gb": 8, "gen": 512, "gen_steady_tps": 10.0,
             "prefill_tps": 100.0, "gen_first_ms": 150.0, "step_ms": 100.0, "gpu_busy_ms": 60.0,
             "pread_ms": 15.0, "engram_ms": 1.0, "host_gap_ms": 24.0, "decode_hit_rate": 0.9,
             "wired_steady_gib": 30.0, "swap_delta_mib": 0.0, "contaminated": False,
             "router_log": None},
            {"workload": "code", "ctx": 4096, "cache_gb": 8, "gen": 2000, "gen_steady_tps": 50.0,
             "prefill_tps": 200.0, "gen_first_ms": 90.0, "step_ms": 20.0, "gpu_busy_ms": 15.0,
             "pread_ms": 3.0, "engram_ms": 0.5, "host_gap_ms": 1.5, "decode_hit_rate": 0.95,
             "wired_steady_gib": 20.0, "swap_delta_mib": 0.0, "contaminated": False,
             "router_log": "/o/code.router.log"},
        ]
        bytes_json = {"per_token": {"resident": 0, "embedding_row": 0, "routed": 0}, "gbps": 290.0}
        text = phase0.report(rows, bytes_json, [])
        header = next(l for l in text.splitlines() if l.startswith("| workload |"))
        self.assertIn("gen", header)
        self.assertIn("log", header)
        self.assertIn("prefill t/s", header)
        self.assertIn("TTFT ms", header)
        self.assertIn("| yes |", text)         # code row has a router log
        # The code row is faster but has a router log, so it must not win "best clean".
        self.assertIn("10.00 t/s (switch", text)


if __name__ == "__main__":
    unittest.main()
