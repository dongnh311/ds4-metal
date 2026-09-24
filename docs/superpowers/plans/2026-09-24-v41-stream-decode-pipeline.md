# V4.1 SSD-streamed decode pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise DeepSeek-V4.1-Flash Q2 SSD-streamed decode on the M5 Pro 64 GB from 9.04 t/s to ≥ 12 t/s with bit-identical output.

**Architecture:**
- **Tooling first.** An interleaved A/B driver and a streaming exactness check sized for 64 GB.
- **Four V4.1-only runtime changes,** each behind its own `DS4_METAL_DISABLE_V41_*` switch and kept only if exact and faster:
  - keep decode layers queued under streaming;
  - load missed experts on the existing async-load worker;
  - lower the V4.1 split threshold;
  - cap the V4.1 auto cache at the measured best size.
- **Qwen stays unchanged.** Code shared with Qwen PROD keeps its behavior; process-level setters default to today's values.

**Tech Stack:** C99 (`ds4.c`, `tests/test_deepseek41_graph.c`), Objective-C/Metal (`ds4_metal.m`), Python 3.9 stdlib (`speed-bench/`), `make`.

**Spec:** `docs/superpowers/specs/2026-09-24-v41-stream-decode-pipeline-design.md` (parent: `docs/V41_64GB_BUILD.md` DoD 2, §8 steps 3–4; data: `speed-bench/v41/phase0-20260924/RESULTS.md`).

## Global Constraints

- **Bit-identical output:** with each new feature on vs off, logits, history and every KV/state span must match bit for bit.
- **Switches:** every feature has a `DS4_METAL_DISABLE_V41_*` switch that restores the old behavior. It is the exactness control and the rollback.
- **Qwen isolation:** functions shared with Qwen PROD must not change behavior. These are the pread pool and `pread_tasks`, `load_batch`, `prepare_load_buffers`/`take_reusable`, `install_loaded`/`set_addr_slot`, `prune_*`, `note_*`, `end_commands` and the slabs. New behavior is selected at V4.1 call sites or through process-level setters whose defaults equal today's behavior.
- **Other models:** V4-Flash and GLM keep their current behavior in `routed_moe_one_tensor` and `split_worthwhile`.
- **Machine limits:** zero swap; wired ≤ 52.48 GiB (`vm.user_wire_limit`); ≥ 40 GiB disk free.
- **Speed evidence:** speed is judged only by interleaved A/B (A, B, B, A in one run), never against a stored number (6–12 % start-to-start drift).
- **Measurement point:** ctx 8192, `switch` prompt (`~/orca/workspaces/ds4-metal-data/v41-phase0/20260924/prompts/switch.txt`), 512 generated tokens, `--teacher-forced-decode`, Phase-0 diagnostics env.
- **Model:** `M=$HOME/orca/workspaces/ds4-metal-data/gguf/DeepSeek-V4.1-Flash-Q2.gguf` (SHA-256 `1ce6a8f8…6f42`).
- **GPU gate:** a task step marked **[GPU]** runs only after the user confirms the box is free (oMLX and both watchdogs down). Restore them after the last GPU step of a session:
  - `launchctl load ~/Library/LaunchAgents/dev.dongnh.ai-proxy.plist`
  - `launchctl load ~/Library/LaunchAgents/dev.dongnh.gateway-watchdog.plist`
- **GPU run protocol:**
  - Every GPU run uses `caffeinate -i -s`, a swap watch (abort with SIGINT, then SIGTERM, if swap grows > 2 GiB) and a stall watch (log unchanged 15 min → report, never `kill -9`).
  - A process in state `U` inside `pread` means: ask the user for a reboot.
- **Chat and artifacts:** the chat with the user is in Vietnamese; code, comments, docs and commits are in English. Do not push.
- **Commit trailer:**
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01TwriwxNUTW7k4x9zH3WeQT
  ```

## Review Focus

1. **Worker cannot stage a load** (a cache slot is in flight on the queued GPU work). The synchronous retry must give the same output. Task 5 pins it: an exactness run at a 4 GB target (1 cache slot, constant eviction).
2. **Router log with queue + async on.** `DS4_V41_ROUTER_LOG` must still record 1280 lines with logits identical to log-off. Tasks 4 and 5 re-run `--router-log`.
3. **`--quality` streaming (layer-resident path) must keep the old per-layer path.** The conditions exclude it. Task 4 adds a `--quality` exactness run of the queue switch.
4. **Long context (32K) with all features on:** no swap, wired within limits. Task 10's grid re-run covers 32K rows, and the swap/wired flags must be clean.
5. **A/B run already on disk** (rerun after an interruption): the A/B driver must reuse finished runs, not mix in stale ones from another label. Task 1 pins it: the tag carries the label and the order index.

---

### Task 1: A/B driver and readahead accounting

**Files:**
- Modify: `speed-bench/v41/phase0.py` (`bench_cmd`, new `run_tag`, new `parse_readahead`, `run_one`, `FIELDS`)
- Create: `speed-bench/v41/ab.py`
- Test: `speed-bench/tests/test_phase0.py`, `speed-bench/tests/test_ab.py` (create)

**Interfaces:**
- Produces:
  - `phase0.bench_cmd(bin_dir, model, prompt, ctx, gen, cache_gb, csv_path)`. `cache_gb=None` omits `--ssd-streaming-cache-experts`, so the engine auto-sizes the cache.
  - `phase0.run_tag(spec, tag_suffix="") -> str`.
  - `phase0.parse_readahead(text) -> float|None`: the `readahead_total` ms of the last timing-summary line.
  - `phase0.run_one(..., extra_env=None, tag_suffix="")`. Its row adds `readahead_ms` and `host_ms` (per decode token).
  - `ab.run_ab(run, label, order=ORDER) -> dict` with keys `label, a_tps, b_tps, ratio, runs, terms, contaminated`.
  - CLI: `python3 speed-bench/v41/ab.py --label L --model M --prompts P --out D [--a-cache N|auto] [--b-cache N|auto] [--a-env K=V]... [--b-env K=V]... [--ctx 8192] [--gen 512] [--workload switch]`.

- [ ] **Step 1: Write the failing tests**

Append to `speed-bench/tests/test_phase0.py` (inside `class RunTest`, and new module-level test classes):

```python
    def test_extra_env_and_tag_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "prompts"), exist_ok=True)
            open(os.path.join(tmp, "prompts", "switch.txt"), "w").close()
            row = phase0.run_one(fake_bin(tmp, self.csv_body()), "/m.gguf",
                                 os.path.join(tmp, "prompts"), tmp, self.SPEC,
                                 running=lambda: "", swap=lambda: 0.0,
                                 sampler=lambda: _FakeSampler(),
                                 idle_read=lambda: FREE_VM_STAT, idle_timeout=0.05,
                                 idle_interval=0.01, extra_env={"DS4_TEST_AB": "on"},
                                 tag_suffix="-x-1b")
            self.assertEqual(row["ds4_env"]["DS4_TEST_AB"], "on")
            self.assertTrue(os.path.exists(os.path.join(tmp, self.TAG + "-x-1b.result.json")))
```

```python
class AbHelpersTest(unittest.TestCase):
    def test_bench_cmd_auto_cache_omits_flag(self):
        cmd = phase0.bench_cmd("/b", "/m.gguf", "/p.txt", 8192, 512, None, "/o.csv")
        self.assertNotIn("--ssd-streaming-cache-experts", cmd)
        self.assertIn("--ssd-streaming", cmd)

    def test_run_tag(self):
        self.assertEqual(phase0.run_tag(("switch", 8192, None, 512, False), "-q-0a"),
                         "switch-c8192-gauto-n512-q-0a")
        self.assertEqual(phase0.run_tag(("switch", 8192, 24, 512, False)),
                         "switch-c8192-g24-n512")

    def test_parse_readahead_takes_last_summary(self):
        text = ("ds4:   streaming expert timing total selected_calls=20480 read_avg=1.1 "
                "readahead_calls=5 readahead_avg=0.1 readahead_total=100.0 readahead_gib=1.0\n"
                "ds4:   streaming expert timing total selected_calls=20480 read_avg=1.1 "
                "readahead_calls=9 readahead_avg=0.1 readahead_total=8948.464 readahead_gib=232.62\n")
        self.assertAlmostEqual(phase0.parse_readahead(text), 8948.464)
        self.assertIsNone(phase0.parse_readahead("nothing"))
```

Create `speed-bench/tests/test_ab.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest discover -s speed-bench/tests 2>&1 | grep -E "^(ERROR|FAIL):|^Ran|^OK|^FAILED"`
Expected: FAILED. There are errors for `test_extra_env_and_tag_suffix` (unexpected keyword `extra_env`), `AbHelpersTest` (no `run_tag`, no `parse_readahead`; `bench_cmd` renders `NoneGB`) and `test_ab` (no module `ab`).

- [ ] **Step 3: Implement in `speed-bench/v41/phase0.py`**

Replace `bench_cmd` and add `run_tag`, `TIMING_RE`, `parse_readahead`:

```python
def bench_cmd(bin_dir, model, prompt, ctx, gen, cache_gb, csv_path):
    cmd = [os.path.join(bin_dir, "ds4-bench"), "--metal", "-m", model,
           "--prompt-file", prompt, "--ctx-start", str(ctx), "--ctx-max", str(ctx),
           "--gen-tokens", str(gen), "--teacher-forced-decode", "--ssd-streaming"]
    if cache_gb is not None:   # None: let the engine size the cache itself
        cmd += ["--ssd-streaming-cache-experts", f"{cache_gb}GB"]
    return cmd + ["--csv", csv_path]


def run_tag(spec, tag_suffix=""):
    workload, ctx, gb, gen, _ = spec
    return f"{workload}-c{ctx}-g{'auto' if gb is None else gb}-n{gen}{tag_suffix}"


# F_RDADVISE time is outside the timed pread, so the raw host gap includes it.
TIMING_RE = re.compile(r"streaming expert timing total .*?readahead_total=([\d.]+)")


def parse_readahead(text):
    rows = TIMING_RE.findall(text)
    return float(rows[-1]) if rows else None
```

In `run_one`:
- Change the signature to end with `idle_timeout=wired.IDLE_SETTLE_S, idle_interval=2.0, extra_env=None, tag_suffix=""`.
- Replace `tag = f"{workload}-c{ctx}-g{gb}-n{gen}"` with `tag = run_tag(spec, tag_suffix)`.
- Add `env.update(extra_env or {})` right after `env.update(ENV)`.
- After `row["ds4_env"] = ...`, add:

```python
    ra = parse_readahead(stderr)
    if ra is not None and row.get("host_gap_ms") is not None and gen > 0:
        row["readahead_ms"] = ra / gen
        row["host_ms"] = row["host_gap_ms"] - row["readahead_ms"]
```

In `FIELDS`, insert `"readahead_ms", "host_ms",` after `"host_gap_ms",`.

- [ ] **Step 4: Create `speed-bench/v41/ab.py`**

```python
#!/usr/bin/env python3
"""Interleaved A/B for V4.1 streaming decode
(docs/superpowers/specs/2026-09-24-v41-stream-decode-pipeline-design.md).

One workload/context point runs in the order A, B, B, A on fresh ds4-bench
processes; B is reported relative to A. Each side can set its own environment
(--a-env/--b-env NAME=VALUE) and cache target (--a-cache/--b-cache N or auto).
Runs reuse phase0.run_one, so the same refusals and wired/contamination checks
apply, and a finished run is reused on rerun.
"""
import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase0  # noqa: E402

ORDER = ("a", "b", "b", "a")
TERMS = ("step_ms", "gpu_busy_ms", "pread_ms", "readahead_ms", "host_ms",
         "decode_hit_rate", "wired_steady_gib")


def parse_env(items):
    env = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"ab: bad env {item!r}, expected NAME=VALUE")
        env[key] = value
    return env


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.fmean(vals) if vals else None


def run_ab(run, label, order=ORDER):
    """`run(side, tag_suffix)` returns one result row; sides are "a" and "b"."""
    sides = {"a": [], "b": []}
    for i, side in enumerate(order):
        sides[side].append(run(side, f"-{label}-{i}{side}"))
    a_tps, b_tps = _mean(sides["a"], "gen_steady_tps"), _mean(sides["b"], "gen_steady_tps")
    return {"label": label, "a_tps": a_tps, "b_tps": b_tps,
            "ratio": b_tps / a_tps if a_tps else None,
            "runs": {s: [r["gen_steady_tps"] for r in rows] for s, rows in sides.items()},
            "terms": {s: {k: _mean(rows, k) for k in TERMS} for s, rows in sides.items()},
            "contaminated": any(r.get("contaminated") for rows in sides.values() for r in rows)}


def _cache(value):
    return None if value in (None, "auto") else int(value)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin", default=phase0.ROOT)
    ap.add_argument("--workload", default="switch")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--a-cache", default="24")
    ap.add_argument("--b-cache", default=None, help="default: same as --a-cache")
    ap.add_argument("--a-env", action="append", default=[])
    ap.add_argument("--b-env", action="append", default=[])
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    caches = {"a": _cache(args.a_cache)}
    caches["b"] = caches["a"] if args.b_cache is None else _cache(args.b_cache)
    envs = {"a": parse_env(args.a_env), "b": parse_env(args.b_env)}

    def run(side, suffix):
        spec = (args.workload, args.ctx, caches[side], args.gen, False)
        row = phase0.run_one(args.bin, args.model, args.prompts, out, spec,
                             extra_env=envs[side], tag_suffix=suffix)
        if row is None:   # finished earlier: reuse it
            with open(os.path.join(out, phase0.run_tag(spec, suffix) + ".result.json")) as fp:
                row = json.load(fp)
        return row

    result = run_ab(run, args.label)
    result.update({"a_env": envs["a"], "b_env": envs["b"], "a_cache": caches["a"],
                   "b_cache": caches["b"], "ctx": args.ctx, "workload": args.workload})
    path = os.path.join(out, f"ab-{args.label}.json")
    with open(path + ".tmp", "w") as fp:
        json.dump(result, fp, indent=1)
    os.replace(path + ".tmp", path)
    print(f"ab: {args.label} A {result['a_tps']:.2f} B {result['b_tps']:.2f} t/s "
          f"ratio {result['ratio']:.4f}" + (" CONTAMINATED" if result["contaminated"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m unittest discover -s speed-bench/tests 2>&1 | grep -E "^(ERROR|FAIL):|^Ran|^OK|^FAILED"`
Expected: `OK`, with 87 + 6 = 93 tests.

- [ ] **Step 6: Commit**

```bash
git add speed-bench/v41/phase0.py speed-bench/v41/ab.py speed-bench/tests/test_phase0.py speed-bench/tests/test_ab.py
git commit -m "speed-bench/v41: interleaved A/B driver and readahead accounting

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TwriwxNUTW7k4x9zH3WeQT"
```

---

### Task 2: Streaming exactness check that fits 64 GB

**Files:**
- Modify: `tests/test_deepseek41_graph.c` (new `check_stream_control` after `check_decode_control` ~:1270; `main` dispatch ~:2116; usage string ~:2248)

**Interfaces:**
- Produces:
  - `./tests/test_deepseek41_graph MODEL --stream-control PROMPT DISABLE_ENV [CACHE_GIB]` (default 8). `DISABLE_ENV` may list several switches separated by commas. Exit 0 and one `... PASS` line per prefix.
  - `./tests/test_deepseek41_graph MODEL --stream-control-selftest PROMPT`. Exit 0 only if the check detects a planted one-token difference.
  - `./tests/test_deepseek41_graph MODEL --stream-control-quality PROMPT DISABLE_ENV`. The same check with the engine's `quality` option on (8 GB).
  - Later tasks run the first form with their switch as `DISABLE_ENV`.

- [ ] **Step 1: Add the check**

Insert after the end of `check_decode_control` in `tests/test_deepseek41_graph.c`:

```c
/* DISABLE_ENV may name several switches separated by commas (all set to 1
 * for the control session). */
static void stream_control_env(const char *names, bool on) {
    char buf[512];
    snprintf(buf, sizeof(buf), "%s", names);
    char *save = NULL;
    for (char *tok = strtok_r(buf, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        if (on) setenv(tok, "1", 1);
        else unsetenv(tok);
    }
}

/* SSD-streaming exactness at a cache that fits a 64 GB machine (the upstream
 * decode-control fixture asks for 64 GiB). The control session runs with
 * `disable` set and the candidate without it; logits, history and every live
 * state span must match bit for bit for 65 decode steps after two prefixes.
 * perturb_step >= 0 feeds the candidate a different token at that step: the
 * self-test uses it to prove the comparison can fail. quality selects the
 * engine's quality mode (layer-resident streaming). */
static int check_stream_control(const char *path, const char *prompt_path,
                                const char *disable, uint64_t cache_bytes,
                                int perturb_step, bool quality) {
    ds4_engine *engine = NULL;
    ds4_session *control = NULL, *candidate = NULL;
    ds4_tokens tokens = {0};
    char *prompt = NULL, err[256] = "";
    size_t prompt_bytes;
    int rc = 1;
    ds4_engine_options opt = {.model_path = path, .backend = DS4_BACKEND_METAL,
        .context_size = 4096, .power_percent = 100, .ssd_streaming = true,
        .ssd_streaming_cache_bytes = cache_bytes, .quality = quality};
    REQUIRE(imatrix_read_text_file(prompt_path, &prompt, &prompt_bytes));
    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    ds4_tokenize_text(engine, prompt, &tokens);
    REQUIRE(tokens.len > 2112);
    const int prefixes[] = {511, 2047};
    for (unsigned pass = 0; pass < sizeof(prefixes) / sizeof(prefixes[0]); pass++) {
        const int prefix = prefixes[pass];
        REQUIRE(ds4_session_create(&control, engine, 4096) == 0);
        REQUIRE(ds4_session_create(&candidate, engine, 4096) == 0);
        ds4_tokens input = {.v = tokens.v, .len = prefix, .cap = prefix};
        stream_control_env(disable, true);
        REQUIRE(ds4_session_sync(control, &input, err, sizeof(err)) == 0);
        stream_control_env(disable, false);
        REQUIRE(ds4_session_sync(candidate, &input, err, sizeof(err)) == 0);
        for (int step = 0; step <= 64; step++) {
            ds41_gpu_graph *a = &control->ds41_graph, *b = &candidate->ds41_graph;
            ds41_state_span sa[54], sb[54];
            const uint32_t n = ds41_state_spans(a, a->pos, sa);
            REQUIRE(a->pos == b->pos && n == ds41_state_spans(b, b->pos, sb));
            REQUIRE(!memcmp(&a->history, &b->history, sizeof(a->history)));
            REQUIRE(!memcmp(control->logits, candidate->logits, DS4_N_VOCAB * sizeof(float)));
            for (uint32_t j = 0; j < n; j++) {
                REQUIRE(sa[j].bytes == sb[j].bytes);
                if (memcmp(ds4_gpu_tensor_contents(sa[j].tensor),
                           ds4_gpu_tensor_contents(sb[j].tensor), (size_t)sa[j].bytes)) {
                    fprintf(stderr, "stream control state mismatch: pos=%u span=%u\n", a->pos, j);
                    goto done;
                }
            }
            if (step == 64) break;
            const int token = tokens.v[prefix + step];
            stream_control_env(disable, true);
            REQUIRE(ds4_session_eval(control, token, err, sizeof(err)) == 0);
            stream_control_env(disable, false);
            const int cand = step == perturb_step ? (token + 1) % DS4_N_VOCAB : token;
            REQUIRE(ds4_session_eval(candidate, cand, err, sizeof(err)) == 0);
        }
        fprintf(stderr, "V4.1 stream control %s%s cache=%.1f GiB prefix=%d: 65 exact "
                "logits/history/KV states PASS\n", disable, quality ? " (quality)" : "",
                (double)cache_bytes / 1073741824.0, prefix);
        ds4_session_free(candidate); candidate = NULL;
        ds4_session_free(control); control = NULL;
    }
    rc = 0;
done:
    if (err[0]) fprintf(stderr, "%s\n", err);
    stream_control_env(disable, false);
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_session_free(candidate); ds4_session_free(control); ds4_engine_close(engine);
    ds4_tokens_free(&tokens); free(prompt);
    return rc;
}
```

- [ ] **Step 2: Dispatch and usage**

In `main`, inside the `#ifdef __APPLE__` block after the `--router-log` line, add:

```c
    if ((argc == 5 || argc == 6) && !strcmp(argv[2], "--stream-control")) {
        const double gib = argc == 6 ? atof(argv[5]) : 8.0;
        if (!(gib > 0.0)) {
            fprintf(stderr, "--stream-control: cache GiB must be positive\n");
            return 2;
        }
        return check_stream_control(argv[1], argv[3], argv[4],
                                    (uint64_t)(gib * 1073741824.0), -1, false);
    }
    if (argc == 5 && !strcmp(argv[2], "--stream-control-quality"))
        return check_stream_control(argv[1], argv[3], argv[4], UINT64_C(8) << 30, -1, true);
    if (argc == 4 && !strcmp(argv[2], "--stream-control-selftest")) {
        /* A no-op switch plus a planted token change: the check must FAIL. */
        const int rc = check_stream_control(argv[1], argv[3], "DS4_TEST_STREAM_CONTROL_NOOP",
                                            UINT64_C(8) << 30, 3, false);
        fprintf(stderr, "V4.1 stream control self-test: %s\n",
                rc ? "planted difference detected PASS" : "difference NOT detected FAIL");
        return rc ? 0 : 1;
    }
```

In the usage string, add `--router-log PROMPT_FILE | --stream-control PROMPT_FILE DISABLE_ENV [CACHE_GIB] | --stream-control-quality PROMPT_FILE DISABLE_ENV | --stream-control-selftest PROMPT_FILE | ` in front of `--session-accounting`.

- [ ] **Step 3: Build and run the model-free checks**

Run: `make tests/test_deepseek41_graph && ./tests/test_deepseek41_graph --router-log-format && ./tests/test_deepseek41_graph --decode-profile-format && ./tests/test_deepseek41_graph 2>&1 | grep -o "stream-control[a-z-]*" | sort -u | tr '\n' ' '`
Expected: a clean build with no new warnings, both format checks PASS, and the last command prints `stream-control stream-control-quality stream-control-selftest`.

- [ ] **Step 4: [GPU] Self-test, then the Engram check (closes spec §0 open item)**

Run, each under the GPU run protocol:
```bash
./tests/test_deepseek41_graph "$M" --stream-control-selftest speed-bench/promessi_sposi.txt
./tests/test_deepseek41_graph "$M" --stream-control speed-bench/promessi_sposi.txt DS4_METAL_DISABLE_V41_ENGRAM_PARALLEL 8
```
Expected:
- Self-test ends with `planted difference detected PASS` (exit 0).
- The Engram check prints two `V4.1 stream control DS4_METAL_DISABLE_V41_ENGRAM_PARALLEL cache=8.0 GiB prefix=511|2047: ... PASS` lines (exit 0).

If the Engram check fails, STOP. `d3bf293` has a real bug: use superpowers:systematic-debugging before any other task.

- [ ] **Step 5: Commit, and update spec §0 and memory**

In `docs/V41_64GB_BUILD.md` §0, replace the sentence starting **Open:** with: "The Engram resolution in `d3bf293` passes the 64 GB streaming exact check (`--stream-control ... DS4_METAL_DISABLE_V41_ENGRAM_PARALLEL 8`, <date>)."

```bash
git add tests/test_deepseek41_graph.c docs/V41_64GB_BUILD.md
git commit -m "tests: V4.1 streaming exactness check sized for 64 GB

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TwriwxNUTW7k4x9zH3WeQT"
```

---

### Task 3: [GPU] Step a probes — readahead and cache size

**Files:**
- Create: `speed-bench/v41/pipeline-<YYYYMMDD>/` holding the `ab-*.json` summaries (raw runs stay in `~/orca/workspaces/ds4-metal-data/v41-pipeline/<YYYYMMDD>/`)

**Interfaces:**
- Consumes: `ab.py` (Task 1).
- Produces:
  - `readahead_verdict` (keep | drop, used by Task 7);
  - `best_cache_gb` (used by Task 8);
  - both recorded in the ledger and in `pipeline-<date>/probes.md`.

- [ ] **Step 1: Readahead A/B (a1)**

```bash
P=$HOME/orca/workspaces/ds4-metal-data/v41-phase0/20260924/prompts
R=$HOME/orca/workspaces/ds4-metal-data/v41-pipeline/$(date +%Y%m%d)
caffeinate -i -s python3 speed-bench/v41/ab.py --label readahead-off --model "$M" --prompts "$P" --out "$R" \
  --a-cache 24 --b-env DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD=1
```
Expected: `ab: readahead-off A ~9 B … ratio …` with no CONTAMINATED. `readahead_verdict = drop` if the ratio is ≥ 1.01, else `keep`.

- [ ] **Step 2: Cache-size A/B (a2)**

```bash
caffeinate -i -s python3 speed-bench/v41/ab.py --label cache-32 --model "$M" --prompts "$P" --out "$R" --a-cache 24 --b-cache 32
caffeinate -i -s python3 speed-bench/v41/ab.py --label cache-40 --model "$M" --prompts "$P" --out "$R" --a-cache 24 --b-cache 40
```
Expected: two summaries.
- Refuse any row with `swap_delta_mib > 256` or `wired_peak_gib > 52`.
- `best_cache_gb` is the smallest of {24, 32, 40} whose t/s is within 1 % of the fastest. Each size's t/s is taken as 24 GB's A-side mean × that pair's ratio.

- [ ] **Step 3: Record and commit**

Copy `$R/ab-*.json` to `speed-bench/v41/pipeline-<date>/`. Write `speed-bench/v41/pipeline-<date>/probes.md` with:
- a table of label, A t/s, B t/s, ratio, B terms (GPU, pread, readahead, host, hit, wired);
- the two verdicts.

```bash
git add speed-bench/v41/pipeline-*/
git commit -m "speed-bench/v41: readahead and cache-size probes for the streaming pipeline

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TwriwxNUTW7k4x9zH3WeQT"
```

---

### Task 4: Step b — keep V4.1 decode layers queued under streaming

**Files:**
- Modify: `ds4.c` (`ds41_graph_step`, the `queue_layers` block ~:41693–41697 and the drain comment ~:41721)

**Interfaces:**
- Consumes: `--stream-control` (Task 2), `ab.py` (Task 1).
- Produces: switch `DS4_METAL_DISABLE_V41_STREAM_DECODE_QUEUE` (on by default once accepted).

- [ ] **Step 1: [GPU] Establish that the switch does nothing yet**

Run: `./tests/test_deepseek41_graph "$M" --stream-control speed-bench/promessi_sposi.txt DS4_METAL_DISABLE_V41_STREAM_DECODE_QUEUE 8`
Expected: PASS. This is trivially true today because the switch is unused; the run records the baseline behavior of the check.

- [ ] **Step 2: Implement**

In `ds41_graph_step`, replace:

```c
#if defined(__APPLE__)
    queue_layers |= g->tp_world == 1 && !g->streaming && !g->imatrix &&
        !getenv("DS4_METAL_DISABLE_V41_SOLO_DECODE_QUEUE");
#endif
```

with:

```c
#if defined(__APPLE__)
    queue_layers |= g->tp_world == 1 && !g->streaming && !g->imatrix &&
        !getenv("DS4_METAL_DISABLE_V41_SOLO_DECODE_QUEUE");
    /* Streaming keeps its selected-id readback inside each routed MoE; that
     * wait already bounds queued work, so the end-of-layer drain is dropped
     * (antirez #1034 shape). Quality mode maps one layer at a time and keeps
     * the drain. */
    queue_layers |= g->tp_world == 1 && g->streaming && !g->quality && !g->imatrix &&
        !getenv("DS4_METAL_DISABLE_V41_STREAM_DECODE_QUEUE");
#endif
```

Change the comment `Solo streaming and imatrix collection retain their per-layer drain.` to `Quality streaming and imatrix collection retain their per-layer drain.`

- [ ] **Step 3: Build and run the model-free checks**

Run: `make all tests/test_deepseek41_graph && ./tests/test_deepseek41_graph --router-log-format && ./tests/test_deepseek41_graph --decode-profile-format && python3 -m unittest discover -s speed-bench/tests 2>&1 | tail -1`
Expected: a clean build, both PASS, `OK`.

- [ ] **Step 4: [GPU] Exactness**

```bash
./tests/test_deepseek41_graph "$M" --stream-control speed-bench/promessi_sposi.txt DS4_METAL_DISABLE_V41_STREAM_DECODE_QUEUE 8
./tests/test_deepseek41_graph "$M" --stream-control speed-bench/promessi_sposi.txt DS4_METAL_DISABLE_V41_STREAM_DECODE_QUEUE 24
./tests/test_deepseek41_graph "$M" --router-log speed-bench/promessi_sposi.txt
./tests/test_deepseek41_graph "$M" --stream-control-quality speed-bench/promessi_sposi.txt DS4_METAL_DISABLE_V41_STREAM_DECODE_QUEUE
```
Expected:
- Two PASS lines each for the first two runs.
- `V4.1 router log: 1280 lines; logits identical ... PASS`.
- The quality run prints two `(quality) ... PASS` lines. The quality path keeps its per-layer drain by construction; this run pins it (Review Focus 3).

A failure is a bug: use superpowers:systematic-debugging, and do not turn the switch default off to hide it.

- [ ] **Step 5: [GPU] A/B**

```bash
caffeinate -i -s python3 speed-bench/v41/ab.py --label queue --model "$M" --prompts "$P" --out "$R" \
  --a-cache 24 --a-env DS4_METAL_DISABLE_V41_STREAM_DECODE_QUEUE=1
```
Expected: ratio > 1.00 (upstream: +11–13.5 %). If the ratio is ≤ 1.00, revert Step 2 (`git checkout ds4.c`), record the result in the ledger and in `probes.md`, and skip Step 6.

- [ ] **Step 6: Commit**

```bash
cp "$R/ab-queue.json" speed-bench/v41/pipeline-<date>/
git add ds4.c speed-bench/v41/pipeline-*/ab-queue.json
git commit -m "ds4: queue V4.1 decode layers under SSD streaming

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TwriwxNUTW7k4x9zH3WeQT"
```

---

### Task 5: Step c1 — load V4.1 missed experts on the async-load worker

**Files:**
- Modify: `ds4.c`, `ds41_moe_partial` (~:40935–41025). Add the helper `ds41_stream_async_load` right above it.

**Interfaces:**
- Consumes (existing, unchanged):
  - `ds4_gpu_signal_selected_readback_ready(uint64_t *)` (`ds4_gpu.h:184`);
  - `metal_graph_selected_async_load_start_tensor(job, router_selected, model, layer, il, event, gate_expert_bytes, down_expert_bytes)` (`ds4.c:23735`);
  - `metal_graph_selected_async_load_finish(job)` (`ds4.c:23790`);
  - `graph_stream_expert_table_make` (`ds4.c:5234`);
  - `ds4_gpu_stream_expert_cache_begin_selected_load`;
  - `ds4_gpu_routed_moe_set_selected_override`;
  - `ds4_gpu_flush_commands`.
- Produces: switch `DS4_METAL_DISABLE_V41_ASYNC_LOAD`.

- [ ] **Step 1: Implement**

Above `static bool ds41_moe_partial(`, add:

```c
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
/* Solo streaming decode: hand the selected ids to the async-load worker as
 * soon as the router is on the GPU, so misses are read while the shared
 * expert runs (the V4-Flash metal_graph_selected_async_load_* path). */
static bool ds41_stream_async_load(const ds41_gpu_graph *g) {
    return g->streaming && !g->quality && !g->imatrix && g->tp_world == 1 &&
        !getenv("DS4_METAL_DISABLE_V41_ASYNC_LOAD");
}
#endif
```

In `ds41_moe_partial`, right after the router block that ends with `g->route_logits)) return false;`, insert:

```c
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    metal_graph_selected_async_load async_load = {0};
    bool async_started = false;
    if (ds41_stream_async_load(g)) {
        uint64_t selected_event = 0;
        if (!ds4_gpu_signal_selected_readback_ready(&selected_event)) return false;
        if (!metal_graph_selected_async_load_start_tensor(&async_load, g->selected, m, l, il,
                selected_event, gate_row * DS4_N_FF_EXP, down_row * DS4_N_EMBD)) return false;
        async_started = true;
        /* Early commit: the router and everything before it start on the GPU
         * now, so the worker's event wait ends before the shared expert. */
        if (!ds4_gpu_flush_commands()) return false;
    }
#endif
```

Right before `routed_ok = ds4_gpu_routed_moe_one_tensor(routed, g->gate, ...` (the unconditional Apple call), insert:

```c
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    if (async_started) {
        const bool flush_ok = ds4_gpu_flush_commands() != 0;
        bool finish_ok = metal_graph_selected_async_load_finish(&async_load);
        if (!finish_ok && async_load.ids_ok) {
            /* The worker may not wait on in-flight cache entries; this thread
             * may, so stage the same load synchronously. */
            const ds4_gpu_stream_expert_table retry =
                graph_stream_expert_table_make(m, l, il, gate_row * DS4_N_FF_EXP,
                                               down_row * DS4_N_EMBD);
            finish_ok = ds4_gpu_stream_expert_cache_begin_selected_load(
                            &retry, async_load.selected_ids, DS4_N_EXPERT_USED) != 0 &&
                        ds4_gpu_routed_moe_set_selected_override(
                            async_load.selected_ids, DS4_N_EXPERT_USED) != 0;
        }
        if (!flush_ok || !finish_ok) return false;
    }
#endif
```

- [ ] **Step 2: Build and run the model-free checks**

Run: `make all tests/test_deepseek41_graph && ./tests/test_deepseek41_graph --router-log-format && ./tests/test_deepseek41_graph --decode-profile-format`
Expected: a clean build and both PASS. Watch for `unused variable 'async_load'` on non-Apple builds: the `#if` guards cover it.

- [ ] **Step 3: [GPU] Exactness, including the retry path**

```bash
./tests/test_deepseek41_graph "$M" --stream-control speed-bench/promessi_sposi.txt DS4_METAL_DISABLE_V41_ASYNC_LOAD 8
./tests/test_deepseek41_graph "$M" --stream-control speed-bench/promessi_sposi.txt DS4_METAL_DISABLE_V41_ASYNC_LOAD 4
./tests/test_deepseek41_graph "$M" --stream-control speed-bench/promessi_sposi.txt DS4_METAL_DISABLE_V41_ASYNC_LOAD 24
./tests/test_deepseek41_graph "$M" --router-log speed-bench/promessi_sposi.txt
```
Expected: every `--stream-control` run prints two PASS lines, and the router log PASSes. The 4 GB target (1 cache slot) forces constant eviction, which exercises the synchronous retry (Review Focus 1).

- [ ] **Step 4: [GPU] A/B, plus the readahead re-check under async**

```bash
caffeinate -i -s python3 speed-bench/v41/ab.py --label async --model "$M" --prompts "$P" --out "$R" \
  --a-cache 24 --a-env DS4_METAL_DISABLE_V41_ASYNC_LOAD=1
caffeinate -i -s python3 speed-bench/v41/ab.py --label async-readahead-off --model "$M" --prompts "$P" --out "$R" \
  --a-cache 24 --b-env DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD=1
```
Expected:
- `async` ratio > 1.00. Otherwise revert Step 1, ledger it, and skip Step 5 and the second run.
- The second run sets the final `readahead_verdict` for Task 7: `drop` if its ratio is ≥ 1.01.

- [ ] **Step 5: Commit**

```bash
cp "$R"/ab-async*.json speed-bench/v41/pipeline-<date>/
git add ds4.c speed-bench/v41/pipeline-*/ab-async*.json
git commit -m "ds4: load V4.1 missed experts on the async-load worker

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TwriwxNUTW7k4x9zH3WeQT"
```

---

### Task 6: Step c2 — V4.1 split threshold of one miss

**Files:**
- Modify: `ds4_metal.m` (`ds4_gpu_stream_expert_split_worthwhile` ~:13335; new setter right after it)
- Modify: `ds4_gpu.h` (declare the setter next to `ds4_gpu_stream_expert_cache_begin_selected_load`, ~:404)
- Modify: `ds4.c` (`ds41_graph_step`, first line after `bool ok = ds41_embed(...)`)

**Interfaces:**
- Produces:
  - `void ds4_gpu_stream_expert_set_split_min_missing(uint32_t n)`: process-level, default 3, `n == 0` is treated as 1;
  - switch `DS4_METAL_DISABLE_V41_SPLIT_LOW`.

- [ ] **Step 1: Implement the setter (default keeps today's behavior)**

In `ds4_metal.m`, replace the body's last line `return ds4_gpu_stream_expert_popcount(missing_mask) >= 3u;` with `return ds4_gpu_stream_expert_popcount(missing_mask) >= g_stream_expert_split_min_missing;`, and add directly above the function:

```objc
/* Minimum missing experts for the hits-first split. 3 for every model; only
 * the V4.1 graph lowers it (process-level: one model family per process). */
static uint32_t g_stream_expert_split_min_missing = 3u;

void ds4_gpu_stream_expert_set_split_min_missing(uint32_t n) {
    g_stream_expert_split_min_missing = n == 0 ? 1u : n;
}
```

In `ds4_gpu.h`, add:

```c
/* Hits-first split threshold (missing experts per layer); default 3. */
void ds4_gpu_stream_expert_set_split_min_missing(uint32_t n);
```

In `ds4.c` `ds41_graph_step`, after `bool ok = ds41_embed(g, m, w, g->residual, g->x, token, g->pos);`:

```c
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    if (g->streaming)
        ds4_gpu_stream_expert_set_split_min_missing(
            getenv("DS4_METAL_DISABLE_V41_SPLIT_LOW") ? 3u : 1u);
#endif
```

- [ ] **Step 2: Build and run the model-free checks**

Run: `make all tests/test_deepseek41_graph && ./tests/test_deepseek41_graph --router-log-format && ./tests/test_deepseek41_graph --decode-profile-format`
Expected: a clean build and both PASS.

- [ ] **Step 3: [GPU] Exactness, then the Qwen fast tier (`ds4_metal.m` changed)**

```bash
./tests/test_deepseek41_graph "$M" --stream-control speed-bench/promessi_sposi.txt DS4_METAL_DISABLE_V41_SPLIT_LOW 24
./tests/test_deepseek41_graph "$M" --stream-control speed-bench/promessi_sposi.txt DS4_METAL_DISABLE_V41_SPLIT_LOW 8
speed-bench/qwen-regression/run.sh fast
```
Expected: two PASS lines per `--stream-control` run and `qwen_gate: PASS`. The split only starts after 4 tokens and needs a warm cache of at least min(1024, budget/2); the 24 GB run is the one that exercises it.

- [ ] **Step 4: [GPU] A/B**

```bash
caffeinate -i -s python3 speed-bench/v41/ab.py --label split-low --model "$M" --prompts "$P" --out "$R" \
  --a-cache 24 --a-env DS4_METAL_DISABLE_V41_SPLIT_LOW=1
```
Expected: ratio > 1.00. Otherwise revert all three files, ledger it, and skip Step 5. Upstream's own comment predicts a loss at 1–2 misses, so a revert is an acceptable outcome.

- [ ] **Step 5: Commit**

```bash
cp "$R/ab-split-low.json" speed-bench/v41/pipeline-<date>/
git add ds4_metal.m ds4_gpu.h ds4.c speed-bench/v41/pipeline-*/ab-split-low.json
git commit -m "ds4: hits-first split from one miss for V4.1 streaming decode

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TwriwxNUTW7k4x9zH3WeQT"
```

---

### Task 7: V4.1 readahead default (only if `readahead_verdict == drop`)

**Files:**
- Modify: `ds4_metal.m` (`ds4_gpu_stream_expert_readahead_enabled` ~:13451; new setter)
- Modify: `ds4_gpu.h`
- Modify: `ds4.c` (`ds41_graph_step`, next to the Task 6 setter)

**Interfaces:**
- Produces:
  - `void ds4_gpu_stream_expert_set_readahead(bool on)`: process-level, default true;
  - V4.1 turns it off unless `DS4_METAL_ENABLE_V41_READAHEAD` is set. `DS4_METAL_ENABLE_V41_READAHEAD` is the control switch.

If Task 5 Step 4 gave `keep`, record "Task 7 skipped: readahead verdict keep (ratio X)" in the ledger and go to Task 8.

- [ ] **Step 1: Implement**

In `ds4_metal.m`, replace `ds4_gpu_stream_expert_readahead_enabled` with:

```objc
/* V4.1 turns readahead off for its process when the Phase-1 A/B shows the
 * inline F_RDADVISE costs more than it saves; every other model keeps it. */
static bool g_stream_expert_readahead_on = true;

void ds4_gpu_stream_expert_set_readahead(bool on) {
    g_stream_expert_readahead_on = on;
}

static int ds4_gpu_stream_expert_readahead_enabled(void) {
    return g_ssd_streaming_mode && g_stream_expert_readahead_on &&
           getenv("DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD") == NULL;
}
```

In `ds4_gpu.h`: `void ds4_gpu_stream_expert_set_readahead(bool on);`, with the comment `/* Streaming expert F_RDADVISE on misses; default on. */`.

In `ds4.c`, inside the Task 6 `if (g->streaming)` block (make it a braced block):

```c
    if (g->streaming) {
        ds4_gpu_stream_expert_set_split_min_missing(
            getenv("DS4_METAL_DISABLE_V41_SPLIT_LOW") ? 3u : 1u);
        ds4_gpu_stream_expert_set_readahead(getenv("DS4_METAL_ENABLE_V41_READAHEAD") != NULL);
    }
```

If Task 6 was reverted, the block holds only the readahead line.

- [ ] **Step 2: Build, model-free checks, [GPU] exactness and Qwen fast tier**

```bash
make all tests/test_deepseek41_graph && ./tests/test_deepseek41_graph --router-log-format && ./tests/test_deepseek41_graph --decode-profile-format
./tests/test_deepseek41_graph "$M" --stream-control speed-bench/promessi_sposi.txt DS4_METAL_ENABLE_V41_READAHEAD 8
speed-bench/qwen-regression/run.sh fast
```
Expected: all PASS. Here "control" runs with readahead on and "candidate" with it off; readahead is only a hint, so the outputs must be identical.

- [ ] **Step 3: [GPU] A/B and commit**

```bash
caffeinate -i -s python3 speed-bench/v41/ab.py --label readahead-default --model "$M" --prompts "$P" --out "$R" \
  --a-cache 24 --a-env DS4_METAL_ENABLE_V41_READAHEAD=1
```
Expected: ratio ≥ 1.01. Otherwise revert and ledger.

```bash
cp "$R/ab-readahead-default.json" speed-bench/v41/pipeline-<date>/
git add ds4_metal.m ds4_gpu.h ds4.c speed-bench/v41/pipeline-*/ab-readahead-default.json
git commit -m "ds4: skip streaming readahead for V4.1 decode

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TwriwxNUTW7k4x9zH3WeQT"
```

---

### Task 8: Step d1 — cap the V4.1 Metal auto cache at the measured best

**Files:**
- Modify: `ds4.c`, `ds4_engine_configure_streaming_auto_cache` (just before `e->ssd_streaming_cache_experts = cache_experts;`, ~:69715)

**Interfaces:**
- Consumes: `best_cache_gb` (Task 3).
- Produces:
  - `#define DS4_V41_METAL_STREAM_CACHE_CAP_GIB <best_cache_gb>`;
  - switch `DS4_METAL_DISABLE_V41_CACHE_CAP`;
  - the log line `ds4:   V4.1 Metal cache capped to %.2f GiB (measured best)`.

- [ ] **Step 1: Implement**

Near the top of `ds4_engine_configure_streaming_auto_cache`'s file section (just above the function), add the define with the number from Task 3, for example `#define DS4_V41_METAL_STREAM_CACHE_CAP_GIB 32` if `best_cache_gb` was 32:

```c
/* Phase-1 A/B (speed-bench/v41/pipeline-<date>/probes.md): V4.1 decode on a
 * 64 GB Mac is fastest at this cache target; the auto plan goes higher. */
#define DS4_V41_METAL_STREAM_CACHE_CAP_GIB 32
```

Before `e->ssd_streaming_cache_experts = cache_experts;`:

```c
#ifdef __APPLE__
    bool v41_capped = false;
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK41 && e->backend == DS4_BACKEND_METAL &&
        !getenv("DS4_METAL_DISABLE_V41_CACHE_CAP")) {
        const uint64_t cap = (uint64_t)DS4_V41_METAL_STREAM_CACHE_CAP_GIB << 30;
        if (effective_cache_bytes > cap && per_expert_bytes != 0) {
            cache_experts = (uint32_t)(cap / per_expert_bytes);
            effective_cache_bytes = (uint64_t)cache_experts * per_expert_bytes;
            v41_capped = true;
        }
    }
#endif
```

After the existing `fprintf` block that prints `expert budget before prefill reserve`, add:

```c
#ifdef __APPLE__
    if (v41_capped)
        fprintf(stderr, "ds4:   V4.1 Metal cache capped to %.2f GiB (measured best)\n",
                (double)effective_cache_bytes / 1073741824.0);
#endif
```

- [ ] **Step 2: Build and [GPU] verify the cap and the speed**

```bash
make all
caffeinate -i -s python3 speed-bench/v41/ab.py --label cache-cap --model "$M" --prompts "$P" --out "$R" \
  --a-cache auto --a-env DS4_METAL_DISABLE_V41_CACHE_CAP=1 --b-cache auto
grep -h "V4.1 Metal cache capped" "$R"/switch-c8192-gauto-n512-cache-cap-1b.stderr
```
Expected:
- The grep prints the capped line.
- Ratio ≥ 1.00. If the uncapped auto plan is already at or below the cap, the ratio is about 1: record that and keep the cap as a guard.
- No swap on either side.

- [ ] **Step 3: Commit**

```bash
cp "$R/ab-cache-cap.json" speed-bench/v41/pipeline-<date>/
git add ds4.c speed-bench/v41/pipeline-*/ab-cache-cap.json
git commit -m "ds4: cap the V4.1 Metal streaming auto cache at the measured best size

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TwriwxNUTW7k4x9zH3WeQT"
```

---

### Task 9: Step d2 — decide on the eviction scan (measurement only)

**Files:** none unless the threshold trips.

- [ ] **Step 1: Read the scan cost from the newest B-side run**

```bash
F=$(ls -t "$R"/*-1b.stderr | head -1)
python3 - "$F" <<'EOF'
import re, sys
text = open(sys.argv[1]).read()
line = [l for l in text.splitlines() if "streaming expert timing total" in l][-1]
f = dict(re.findall(r"(\w+)=([\d.]+)", line))
tokens = float(f["selected_calls"]) / 40
print(f"reuse_scan {float(f['reuse_scan_total'])/tokens:.2f} ms/token, "
      f"prepare_buffer {float(f['prepare_buffer_total'])/tokens:.2f} ms/token")
EOF
```
Expected: one line with both figures.

- [ ] **Step 2: Decide**

If `reuse_scan` ≤ 2.0 ms/token, record "d2 skipped (X ms/token)" in the ledger. If it is above 2.0, stop and report to the user with the numbers. A victim structure is a new design (it touches the shared cache that Qwen uses) and needs its own spec.

---

### Task 10: [GPU] End of sub-project — grid, exactness, Qwen full tier, results

**Files:**
- Create: `speed-bench/v41/pipeline-<date>/{results.csv,RESULTS.generated.md,RESULTS.md}`
- Modify: `docs/V41_64GB_BUILD.md` (§0 status, §2.1 add a "Phase 1" column), memory `ds41-upstream-work-and-64gb-gap.md`

- [ ] **Step 1: All switches together, exact**

Run each kept switch alone at 8 GB, then all kept switches together (control = every switch set, candidate = shipped defaults) at 8 GB and 24 GB:

```bash
ALL=DS4_METAL_DISABLE_V41_STREAM_DECODE_QUEUE,DS4_METAL_DISABLE_V41_ASYNC_LOAD,DS4_METAL_DISABLE_V41_SPLIT_LOW,DS4_METAL_ENABLE_V41_READAHEAD
for S in ${ALL//,/ }; do
  ./tests/test_deepseek41_graph "$M" --stream-control speed-bench/promessi_sposi.txt "$S" 8 || exit 1
done
./tests/test_deepseek41_graph "$M" --stream-control speed-bench/promessi_sposi.txt "$ALL" 8
./tests/test_deepseek41_graph "$M" --stream-control speed-bench/promessi_sposi.txt "$ALL" 24
```
(Remove from `ALL` the switches whose task was reverted or skipped.)

Expected: all PASS.

- [ ] **Step 2: Re-run the Phase-0 speed grid with the new defaults**

```bash
G=$HOME/orca/workspaces/ds4-metal-data/v41-pipeline/$(date +%Y%m%d)-grid
caffeinate -i -s python3 speed-bench/v41/phase0.py run --plan speed --model "$M" --prompts "$P" --out "$G"
python3 speed-bench/v41/phase0.py table --out "$G"
python3 speed-bench/v41/phase0.py report --out "$G" --bytes speed-bench/v41/phase0-20260924/bytes.json
cp "$G/results.csv" "$G/RESULTS.generated.md" speed-bench/v41/pipeline-<date>/
```
Expected: 12 rows, no swapped or contaminated flags. Compare each row with `speed-bench/v41/phase0-20260924/results.csv`.

- [ ] **Step 3: Target check with the default cache (auto)**

```bash
caffeinate -i -s python3 speed-bench/v41/ab.py --label final --model "$M" --prompts "$P" --out "$R" \
  --a-cache 24 --a-env DS4_METAL_DISABLE_V41_STREAM_DECODE_QUEUE=1 --a-env DS4_METAL_DISABLE_V41_ASYNC_LOAD=1 \
  --a-env DS4_METAL_DISABLE_V41_SPLIT_LOW=1 --a-env DS4_METAL_ENABLE_V41_READAHEAD=1 --b-cache auto
```
Expected: the B side is the shipped configuration; A is the Phase-0 configuration. Report B t/s against the ≥ 12 t/s target and the ratio.

- [ ] **Step 4: Qwen full tier**

Run: `speed-bench/qwen-regression/run.sh full`
Expected: `qwen_gate: PASS`.

- [ ] **Step 5: Write results, update the spec and memory, commit**

`speed-bench/v41/pipeline-<date>/RESULTS.md` holds:
- setup (commit, model SHA, env, cache);
- the per-step A/B table from all `ab-*.json`;
- the grid comparison against Phase 0;
- the new per-token decomposition at 8K from the `final` B side;
- target met or not;
- remaining terms handed to sub-projects 2–3.

In the spec §0, add "Sub-project 1 (streaming pipeline) DONE <date>: <t/s>". In §2.1, add the "Phase 1" measured column. Update memory `ds41-upstream-work-and-64gb-gap.md`.

```bash
git add speed-bench/v41/pipeline-*/ docs/V41_64GB_BUILD.md
git commit -m "docs: V4.1 streaming pipeline results

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TwriwxNUTW7k4x9zH3WeQT"
```

Restore oMLX and the watchdogs (Global Constraints).
