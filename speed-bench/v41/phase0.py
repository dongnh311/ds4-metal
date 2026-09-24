#!/usr/bin/env python3
"""Phase 0(b) driver for DeepSeek-V4.1-Flash Q2 on M5 Pro 64 GB (docs/V41_64GB_BUILD.md §7).

prompts  build the workload prompt files
run      run a plan (speed | locality) with ds4-bench, one result JSON per run
table    merge result JSONs into results.csv
report   render RESULTS.md tables from results, byte accounting and locality

Runs refuse to start while any ds4 process is running: the machine must be free.
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "speed-bench", "lib"))
sys.path.insert(0, HERE)
import machine  # noqa: E402
import wired  # noqa: E402
import gguf_bytes  # noqa: E402

PROMPT_CHARS = 200_000
CHUNK = 2_000
SOURCES = {
    "code": ["ds4_server.c"],
    "docs": ["README.md", "docs/*.md", "QA_BEFORE_RELEASES.md", "MODEL_CARD.md", "EVAL_DATA.md"],
    "it": ["speed-bench/promessi_sposi.txt"],
}
# (workload, ctx, cache_gb, gen_tokens, router_log)
PLANS = {
    "speed": [("switch", ctx, gb, 512, False)
              for ctx in (4096, 8192, 32768) for gb in (4, 8, 16, 24)],
    "locality": [(w, 4096, 8, 2000, True) for w in ("code", "docs", "it", "switch")],
}
ENV = {"DS4_V41_DECODE_PROFILE": "1", "DS4_METAL_GPU_BUSY_PROFILE": "1",
       "DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY": "1"}
SWAP_LIMIT_MIB = 256.0
DECODE_MARGIN_S = 1.0
FIELDS = ["workload", "ctx", "cache_gb", "gen", "prefill_tps", "gen_tps", "gen_steady_tps",
          "gen_first_ms", "step_ms", "gpu_busy_ms", "pread_ms", "engram_ms", "host_gap_ms",
          "pread_mib", "hits", "misses", "decode_hit_rate", "cache_experts", "cache_hit_rate",
          "wired_steady_gib", "wired_peak_gib", "wired_window", "wired_idle_gib",
          "swap_delta_mib", "contaminated", "router_log"]
PROFILE_RE = re.compile(
    r"ds4: V4\.1 decode profile: tokens=(\d+) step_ms=([\d.]+) engram_ms=([\d.]+) "
    r"gpu_busy_ms=([\d.]+) pread_ms=([\d.]+) pread_mib=([\d.]+) hits=([\d.]+) misses=([\d.]+)")
CACHE_RE = re.compile(r"streaming expert cache budget=(\d+) experts .*?hit_rate=([\d.]+)")


def build_prompts(root, chars=PROMPT_CHARS, chunk=CHUNK):
    """name -> text; 'switch' interleaves chunk-sized pieces of code, docs and it."""
    texts = {}
    for name, patterns in SOURCES.items():
        paths = [p for pat in patterns for p in sorted(glob.glob(os.path.join(root, pat)))]
        text = ""
        for path in paths:
            with open(path, encoding="utf-8", errors="replace") as fp:
                text += fp.read() + "\n\n"
        if len(text) < chars:
            raise ValueError(f"{name}: sources have {len(text)} chars, need {chars}")
        texts[name] = text[:chars]
    parts, i = [], 0
    while sum(len(p) for p in parts) < chars:
        for name in ("code", "docs", "it"):
            parts.append(texts[name][i:i + chunk])
        i += chunk
    texts["switch"] = "".join(parts)[:chars]
    return texts


def bench_cmd(bin_dir, model, prompt, ctx, gen, cache_gb, csv_path):
    return [os.path.join(bin_dir, "ds4-bench"), "--metal", "-m", model,
            "--prompt-file", prompt, "--ctx-start", str(ctx), "--ctx-max", str(ctx),
            "--gen-tokens", str(gen), "--teacher-forced-decode",
            "--ssd-streaming", "--ssd-streaming-cache-experts", f"{cache_gb}GB",
            "--csv", csv_path]


def parse_profile(text):
    rows = PROFILE_RE.findall(text)
    if not rows:
        return None
    keys = ("tokens", "step_ms", "engram_ms", "gpu_busy_ms", "pread_ms", "pread_mib",
            "hits", "misses")
    p = dict(zip(keys, (float(x) for x in rows[-1])))
    p["tokens"] = int(p["tokens"])
    p["host_gap_ms"] = p["step_ms"] - p["gpu_busy_ms"] - p["pread_ms"] - p["engram_ms"]
    return p


def parse_cache(text):
    rows = CACHE_RE.findall(text)
    if not rows:
        return None
    return {"cache_experts": int(rows[-1][0]), "cache_hit_rate": float(rows[-1][1])}


def parse_bench_csv(text):
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError("empty ds4-bench CSV")
    return {k: float(rows[-1][k]) for k in ("prefill_tps", "gen_tps", "gen_steady_tps",
                                             "gen_first_ms")}


def combine(spec, bench, profile, cache, wired_summary, swap_delta_mib, contaminated):
    workload, ctx, gb, gen, _ = spec
    row = {"workload": workload, "ctx": ctx, "cache_gb": gb, "gen": gen, **bench,
           "wired_steady_gib": wired_summary["steady_gib"],
           "wired_peak_gib": wired_summary["peak_gib"],
           "wired_window": wired_summary.get("window"),
           "swap_delta_mib": swap_delta_mib, "contaminated": contaminated}
    if profile:
        for key in ("step_ms", "engram_ms", "gpu_busy_ms", "pread_ms", "pread_mib",
                    "hits", "misses", "host_gap_ms"):
            row[key] = profile[key]
        lookups = profile["hits"] + profile["misses"]
        row["decode_hit_rate"] = profile["hits"] / lookups if lookups else None
    if cache:
        row.update(cache)
    return row


def _fail_router_log(router_log):
    """On any failure after the child started, keep a failed run's router log
    around under a distinct name so it is never mistaken for a good one."""
    if router_log and os.path.exists(router_log):
        os.replace(router_log, router_log + ".failed")


def run_one(bin_dir, model, prompts_dir, out_dir, spec, dry_run=False,
            running=machine.ds4_running, swap=machine.swap_used_mib,
            sampler=wired.WiredSampler, idle_read=None,
            idle_timeout=wired.IDLE_SETTLE_S, idle_interval=2.0):
    workload, ctx, gb, gen, log_router = spec
    tag = f"{workload}-c{ctx}-g{gb}-n{gen}"
    result_path = os.path.join(out_dir, tag + ".result.json")
    csv_path = os.path.join(out_dir, tag + ".csv")
    router_log = os.path.join(out_dir, tag + ".router.log") if log_router else None
    cmd = bench_cmd(bin_dir, model, os.path.join(prompts_dir, workload + ".txt"),
                    ctx, gen, gb, csv_path)
    if dry_run:
        print("phase0:", " ".join(cmd))
        return None
    if os.path.exists(result_path):
        print("phase0: skip (done)", tag)
        return None
    idle = wired.wait_idle_gib(read=idle_read, timeout=idle_timeout, interval=idle_interval)
    if idle > wired.IDLE_WIRED_LIMIT_GIB:
        raise SystemExit(f"phase0: {idle:.1f} GiB wired before the run; the machine is not idle")
    busy = running()
    if busy:
        raise SystemExit("phase0: ds4 is running; the machine must be free:\n" + busy)
    env = dict(os.environ)
    env.pop("DS4_V41_ROUTER_LOG", None)
    env.update(ENV)
    if router_log:
        env["DS4_V41_ROUTER_LOG"] = router_log
    stderr_path = os.path.join(out_dir, tag + ".stderr")
    swap0 = swap()
    contam = {"hit": False}
    stop_poll = threading.Event()

    def poll(interval):
        while not stop_poll.wait(interval):
            try:
                line = running()
            except StopIteration:
                # A test's finite fake `running()` iterator was exhausted by
                # this background poll; stop watching rather than crash the
                # thread. Real callers (machine.ds4_running) never raise this.
                return
            if line and csv_path not in line:
                contam["hit"] = True

    with open(stderr_path, "w") as err, sampler() as ws:
        interval = getattr(ws, "interval", 0.5)
        poll_thread = threading.Thread(target=poll, args=(interval,), daemon=True)
        poll_thread.start()
        rc = subprocess.run(cmd, env=env, stdout=err, stderr=err).returncode
        t_exit = time.monotonic()
        stop_poll.set()
        poll_thread.join()
    if rc != 0:
        _fail_router_log(router_log)
        raise SystemExit(f"phase0: {tag} failed (rc={rc}), see {stderr_path}")
    contaminated = contam["hit"] or bool(running())
    with open(stderr_path) as fp:
        stderr = fp.read()
    try:
        with open(csv_path) as fp:
            bench = parse_bench_csv(fp.read())
    except (OSError, ValueError) as exc:
        _fail_router_log(router_log)
        raise SystemExit(f"phase0: {tag} produced no usable CSV ({exc}), see {stderr_path}")
    gen_tps = bench["gen_tps"]
    if gen_tps > 0:
        decode_s = gen / gen_tps
        t_start = t_exit - DECODE_MARGIN_S - decode_s
        t_end = t_exit - DECODE_MARGIN_S
    else:
        t_start, t_end = t_exit, t_exit - 1.0   # empty window -> forces fallback
    wsum = wired.window_summary(ws.timed, t_start, t_end)
    row = combine(spec, bench, parse_profile(stderr), parse_cache(stderr), wsum,
                  swap() - swap0, contaminated)
    row["router_log"] = router_log
    row["wired_idle_gib"] = idle
    row["ds4_env"] = {k: v for k, v in env.items() if k.startswith("DS4_")}
    tmp_path = result_path + ".tmp"
    with open(tmp_path, "w") as fp:
        json.dump(row, fp, indent=1)
    os.replace(tmp_path, result_path)
    print(f"phase0: done {tag} {row['gen_steady_tps']:.2f} t/s"
          + (" CONTAMINATED" if contaminated else ""))
    return row


def table(out_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.result.json"))):
        with open(path) as fp:
            rows.append(json.load(fp))
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return rows


def _flag(row):
    flags = []
    if (row.get("swap_delta_mib") or 0.0) > SWAP_LIMIT_MIB:
        flags.append("swapped")
    if row.get("contaminated"):
        flags.append("contaminated")
    if (row.get("host_gap_ms") or 0.0) < 0:
        flags.append("overlap")
    return ", ".join(flags)


def _fmt(value, spec=".2f"):
    return "—" if value is None else format(value, spec)


def report(rows, bytes_json, locality):
    """Markdown tables for RESULTS.md; `locality` entries carry a 'name' key."""
    roof = gguf_bytes.roofline(bytes_json, bytes_json.get("gbps", 290.0))
    lines = ["## Measured decode (per token)", "",
             "`host_gap_ms` is step_ms minus GPU busy, pread and Engram time: a residual, not a "
             "direct measurement. It can go negative (flagged `overlap`) when pread overlaps GPU "
             "work.", "",
             "| workload | ctx | cache GB | gen | t/s | prefill t/s | TTFT ms | step ms | GPU busy "
             "| pread | Engram | host gaps | hit rate | wired GiB | log | flags |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
             "| ---: | ---: | --- | --- |"]
    for r in sorted(rows, key=lambda r: (r["workload"], r["ctx"], r["cache_gb"])):
        lines.append(
            f"| {r['workload']} | {r['ctx']} | {r['cache_gb']} | {_fmt(r.get('gen'), 'd')} "
            f"| {_fmt(r.get('gen_steady_tps'))} | {_fmt(r.get('prefill_tps'))} "
            f"| {_fmt(r.get('gen_first_ms'), '.1f')} "
            f"| {_fmt(r.get('step_ms'), '.1f')} | {_fmt(r.get('gpu_busy_ms'), '.1f')} "
            f"| {_fmt(r.get('pread_ms'), '.1f')} | {_fmt(r.get('engram_ms'), '.1f')} "
            f"| {_fmt(r.get('host_gap_ms'), '.1f')} | {_fmt(r.get('decode_hit_rate'), '.3f')} "
            f"| {_fmt(r.get('wired_steady_gib'), '.1f')} | {'yes' if r.get('router_log') else '—'} "
            f"| {_flag(r)} |")
    clean = [r for r in rows if not _flag(r) and not r.get("router_log")]
    best = max(clean, key=lambda r: r["gen_steady_tps"]) if clean else None
    lines += ["", "Best clean run: " + (
        f"{best['gen_steady_tps']:.2f} t/s ({best['workload']}, ctx {best['ctx']}, "
        f"cache {best['cache_gb']} GB)" if best else "none"), ""]
    lines += ["## Roofline from byte accounting", "",
              f"Byte floor at {bytes_json.get('gbps', 290.0):g} GB/s: resident "
              f"{roof['resident_ms']:.1f} + routed {roof['routed_ms']:.1f} ms = "
              f"{roof['total_ms']:.1f} ms/token → {roof['tps_ceiling']:.1f} t/s ceiling "
              f"(no I/O, no sync).", ""]
    if locality:
        budgets = sorted({b for loc in locality for b in loc["lru_hit"]}, key=float)
        ks = sorted({k for loc in locality for k in loc["union_cover"]}, key=int)
        lines += ["## Routing locality (decode, LRU)", "",
                  "| workload | " + " | ".join(f"{b} GiB" for b in budgets)
                  + " | pair overlap | " + " | ".join(f"cover K={k}" for k in ks)
                  + " | new/token |",
                  "| --- |" + " ---: |" * (len(budgets) + 2 + len(ks))]
        for loc in locality:
            lines.append(f"| {loc['name']} | "
                         + " | ".join(_fmt(loc["lru_hit"].get(b), ".3f") for b in budgets)
                         + f" | {loc['pair_overlap']:.3f} | "
                         + " | ".join(_fmt(loc["union_cover"].get(k), ".3f") for k in ks)
                         + f" | {loc['new_per_token']:.2f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--plan", choices=sorted(PLANS), required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--prompts", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--bin", default=ROOT)
    r.add_argument("--dry-run", action="store_true")
    t = sub.add_parser("table")
    t.add_argument("--out", required=True)
    rep = sub.add_parser("report")
    rep.add_argument("--out", required=True)
    rep.add_argument("--bytes", required=True)
    rep.add_argument("--locality", nargs="*", default=[])
    args = ap.parse_args()
    if args.mode == "prompts":
        os.makedirs(args.out, exist_ok=True)
        for name, text in build_prompts(ROOT).items():
            with open(os.path.join(args.out, name + ".txt"), "w", encoding="utf-8") as fp:
                fp.write(text)
        print("phase0: prompts in", args.out)
    elif args.mode == "run":
        os.makedirs(args.out, exist_ok=True)
        for spec in PLANS[args.plan]:
            run_one(args.bin, args.model, args.prompts, args.out, spec, dry_run=args.dry_run)
    elif args.mode == "table":
        print(f"phase0: {len(table(args.out))} rows -> results.csv")
    else:
        rows = table(args.out)
        with open(args.bytes) as fp:
            bytes_json = json.load(fp)
        locality = []
        for path in args.locality:
            with open(path) as fp:
                loc = json.load(fp)
            loc["name"] = os.path.basename(path).split("-")[0]
            locality.append(loc)
        with open(os.path.join(args.out, "RESULTS.generated.md"), "w") as fp:
            fp.write(report(rows, bytes_json, locality))
        print("phase0: wrote", os.path.join(args.out, "RESULTS.generated.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
