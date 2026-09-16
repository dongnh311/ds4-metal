#!/usr/bin/env python3
"""List which tensor payloads differ between two built qwen4exp GGUFs.

Ground-truth for the Orca isolation bisection: given two files that must
share the same tensor set/kinds/dims, hash every payload and print the
names whose bytes differ. This removes all hand-tracking: `a` is the
known-coherent reference (usually the base control), `b` is the build
under test, and the output says exactly which families are Orca-unique.
"""
import hashlib
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


def payload_hashes(path, chunk=8 << 20):
    r = Reader(str(path))
    out = {}
    for name, (kind, dims, off) in r.tensors.items():
        sz = size_of(kind, dims)
        r.f.seek(r.data_start + off)
        h = hashlib.sha256()
        left = sz
        while left:
            data = r.f.read(min(chunk, left))
            if not data:
                sys.exit(f"short read of {name} in {path}")
            h.update(data)
            left -= len(data)
        out[name] = (h.hexdigest(), kind, list(dims), sz)
    r.f.close()
    return out


def main():
    a = Path(sys.argv[1])
    b = Path(sys.argv[2])
    ha, hb = payload_hashes(a), payload_hashes(b)
    if set(ha) != set(hb):
        only_a = set(ha) - set(hb)
        only_b = set(hb) - set(ha)
        print(f"tensor sets differ: {len(only_a)} only in A, {len(only_b)} only in B")
        for n in sorted(only_b)[:10]:
            print(f"  B-only: {n}")
        for n in sorted(only_a)[:10]:
            print(f"  A-only: {n}")
    differ = sorted(n for n in ha if n in hb and ha[n][0] != hb[n][0])
    layout_bad = [n for n in ha if n in hb and (ha[n][1:] != hb[n][1:])]
    total = sum(ha[n][3] for n in differ)
    print(f"A: {a.name}  ({len(ha)} tensors)")
    print(f"B: {b.name}  ({len(hb)} tensors)")
    print(f"{len(differ)} tensors differ ({total / 2**30:.3f} GiB), "
          f"{len(layout_bad)} with a kind/dims mismatch")
    # Group by family suffix for a compact readout.
    fam = {}
    for n in differ:
        key = n.split(".weight")[0].split(".")[-1] if ".weight" in n else n
        fam.setdefault(key, []).append(n)
    for key in sorted(fam):
        names = fam[key]
        gi = sum(ha[n][3] for n in names)
        print(f"  {key:20s} {len(names):4d}  {gi / 2**30:.3f} GiB  e.g. {names[0]}")
    if layout_bad:
        print(f"LAYOUT MISMATCH: {layout_bad[:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
