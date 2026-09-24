#!/usr/bin/env python3
"""Per-stage GPU time of V4.1 decode from DS4_METAL_GPU_STAGE_TIMESTAMPS_DETAIL.

Each stage boundary commits its own command buffer. Buffers can overlap on the
GPU timeline (a buffer starts, then waits for its predecessor), so summing
their GPU spans counts the overlap twice. This walks one token's buffers in
commit order and credits each one only with the time past the latest end
seen so far; the gaps between buffers are GPU idle (host work, expert loads).

    python3 speed-bench/v41/stages.py STDERR_FILE [--min-pos N]
"""
import argparse
import collections
import re
import sys

BUFFER_RE = re.compile(
    r"gpu stage buffer what=decode part=\S+ stage=(\S+) layer=\d+ pos=(\d+) tokens=1 "
    r"start_ms=([-\d.]+) gpu_ms=([\d.]+)")


def exclusive(lines, min_pos=0):
    by_token = collections.OrderedDict()
    for line in lines:
        m = BUFFER_RE.search(line)
        if m and int(m.group(2)) >= min_pos:
            by_token.setdefault(int(m.group(2)), []).append(
                (m.group(1), float(m.group(3)), float(m.group(4))))
    totals = collections.OrderedDict()
    idle = 0.0
    for bufs in by_token.values():
        latest = None
        for stage, start, dur in bufs:
            end = start + dur
            begin = start if latest is None else max(start, latest)
            if latest is not None and start > latest:
                idle += start - latest
            totals[stage] = totals.get(stage, 0.0) + max(0.0, end - begin)
            latest = end if latest is None else max(latest, end)
    n = len(by_token)
    stages = {k: v / n for k, v in totals.items()} if n else {}
    return {"tokens": n, "stages": stages, "busy_ms": sum(stages.values()),
            "idle_ms": idle / n if n else 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stderr")
    ap.add_argument("--min-pos", type=int, default=0,
                    help="skip decode tokens before this position (warm-up)")
    args = ap.parse_args()
    with open(args.stderr) as fp:
        res = exclusive(fp, args.min_pos)
    if not res["tokens"]:
        raise SystemExit("stages: no decode stage buffers (run with DS4_METAL_GPU_STAGE_TIMESTAMPS=1 "
                         "and DS4_METAL_GPU_STAGE_TIMESTAMPS_DETAIL=1)")
    print(f"tokens {res['tokens']}: GPU busy {res['busy_ms']:.2f} ms/token, "
          f"idle between buffers {res['idle_ms']:.2f} ms/token")
    for stage, ms in res["stages"].items():
        print(f"{stage:14s} {ms:7.2f} ms  {100.0 * ms / res['busy_ms']:5.1f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
