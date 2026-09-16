#!/usr/bin/env python3
"""Payload surgery: copy tensors matching `--prefix` from src GGUF into dst
GGUF in place. Used for the Orca garble isolation tests (MTP blk.48 swap,
trunk expert-down swap). Refuses to run unless both files declare the same
tensor names, kinds and dims (layout identity is what makes a positional
payload copy safe).
"""
import argparse
import math
import sys
from pathlib import Path

from qwen4_pack_to_qwen4exp import Reader, tnbytes


def size_of(kind, dims):
    n = math.prod(dims)
    if kind == 10:
        return n // 256 * 84
    if kind == 16:
        return n // 256 * 66
    if kind == 39:
        return n // 32 * 17
    return tnbytes(kind, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--prefix", required=True,
                   help="tensor name prefix, e.g. blk.48. or ffn_down_exps")
    ap.add_argument("--contains", default=None,
                    help="only tensors whose name contains this too")
    args = ap.parse_args()

    src, dst = Reader(args.src), Reader(args.dst)
    dst.f.close()
    writer = open(args.dst, "r+b")
    if src.tensors.keys() != dst.tensors.keys():
        sys.exit(f"tensor sets differ: {len(src.tensors)} vs {len(dst.tensors)}")
    for name, (k1, d1, _o1) in src.tensors.items():
        k2, d2, _o2 = dst.tensors[name]
        if k1 != k2 or list(d1) != list(d2):
            sys.exit(f"layout differs at {name}: {(k1, d1)} vs {(k2, d2)}")

    def match(name):
        return name.startswith(args.prefix) and \
            (args.contains is None or args.contains in name)

    targets = [n for n in src.tensors if match(n)]
    if not targets:
        sys.exit(f"no tensors match prefix={args.prefix!r} "
                 f"contains={args.contains!r}")
    total = sum(size_of(*src.tensors[n][:2]) for n in targets)
    print(f"swapping {len(targets)} tensors ({total / 2**20:.1f} MiB) "
          f"prefix={args.prefix!r} contains={args.contains!r}", flush=True)
    for name in targets:
        sk, sdims, doff = dst.tensors[name]
        ss = size_of(sk, sdims)
        src.f.seek(src.data_start + src.tensors[name][2])
        payload = src.f.read(ss)
        assert len(payload) == ss, f"short read of {name}"
        writer.seek(dst.data_start + doff)
        writer.write(payload)
        print(f"  {name} kind={sk} {ss / 2**20:.1f} MiB", flush=True)
    writer.flush()
    writer.close()
    src.f.close()
    print("done")


if __name__ == "__main__":
    main()
