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
         "decode_hit_rate", "wired_steady_gib", "la_issued_per_tok", "la_used_per_tok")


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


def env_mismatch(row_env, envs, side):
    """Switches whose value in a reused run differs from what `side` asks for
    now (keys of either side; absent means unset)."""
    keys = set(envs["a"]) | set(envs["b"])
    return sorted(k for k in keys if row_env.get(k) != envs[side].get(k))


def run_ab(run, label, order=ORDER):
    """`run(side, tag_suffix)` returns one result row; sides are "a" and "b"."""
    sides = {"a": [], "b": []}
    for i, side in enumerate(order):
        sides[side].append(run(side, f"-{label}-{i}{side}"))
    a_tps, b_tps = _mean(sides["a"], "gen_steady_tps"), _mean(sides["b"], "gen_steady_tps")
    return {"label": label, "a_tps": a_tps, "b_tps": b_tps,
            "ratio": b_tps / a_tps if a_tps and b_tps is not None else None,
            "runs": {s: [r["gen_steady_tps"] for r in rows] for s, rows in sides.items()},
            "terms": {s: {k: _mean(rows, k) for k in TERMS} for s, rows in sides.items()},
            "contaminated": any(r.get("contaminated") for rows in sides.values() for r in rows)}


def summary_line(result):
    def tps(v):
        return "n/a" if v is None else f"{v:.2f}"
    ratio = "ratio n/a" if result["ratio"] is None else f"ratio {result['ratio']:.4f}"
    return (f"ab: {result['label']} A {tps(result['a_tps'])} B {tps(result['b_tps'])} t/s {ratio}"
            + (" CONTAMINATED" if result["contaminated"] else ""))


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
        if row is None:   # finished earlier: reuse it, if it ran with this side's switches
            tag = phase0.run_tag(spec, suffix)
            with open(os.path.join(out, tag + ".result.json")) as fp:
                row = json.load(fp)
            stale = env_mismatch(row.get("ds4_env", {}), envs, side)
            if stale:
                raise SystemExit(f"ab: {tag} was run with different {', '.join(stale)}; "
                                 "use a new --label or --out")
        return row

    result = run_ab(run, args.label)
    result.update({"a_env": envs["a"], "b_env": envs["b"], "a_cache": caches["a"],
                   "b_cache": caches["b"], "ctx": args.ctx, "workload": args.workload})
    path = os.path.join(out, f"ab-{args.label}.json")
    with open(path + ".tmp", "w") as fp:
        json.dump(result, fp, indent=1)
    os.replace(path + ".tmp", path)
    print(summary_line(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
