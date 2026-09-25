#!/usr/bin/env python3
"""Write sparse copies of an Ornith GGUF with one field patched.

  make_bad_gguf.py MODEL OUT_DIR

bad_embd.gguf: qwen35moe.embedding_length = 64.
bad_tier.gguf: blk.20 gate/up experts typed IQ4_XS (ggml type 23).
Only the header is copied; the tensor data is a hole of the original size,
so each file costs a few MB on disk.
"""
import os
import struct
import sys

SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_header(path):
    with open(path, "rb") as f:
        buf = f.read(64 << 20)
    pos = 0

    def take(n):
        nonlocal pos
        out = buf[pos:pos + n]
        pos += n
        return out

    def string():
        (n,) = struct.unpack("<Q", take(8))
        return take(n).decode("utf-8", "replace")

    def skip_value(t):
        if t in SCALAR:
            take(SCALAR[t])
        elif t == 8:
            string()
        elif t == 9:
            (et,) = struct.unpack("<I", take(4))
            (n,) = struct.unpack("<Q", take(8))
            for _ in range(n):
                skip_value(et)
        else:
            sys.exit(f"unknown GGUF value type {t}")

    assert take(4) == b"GGUF"
    take(4)
    n_tensors, n_kv = struct.unpack("<QQ", take(16))
    kv_pos = {}
    for _ in range(n_kv):
        key = string()
        (t,) = struct.unpack("<I", take(4))
        kv_pos[key] = (pos, t)
        skip_value(t)
    type_pos = {}
    for _ in range(n_tensors):
        name = string()
        (nd,) = struct.unpack("<I", take(4))
        take(8 * nd)
        type_pos[name] = pos
        take(4 + 8)
    return buf, pos, kv_pos, type_pos


def write(out, header, size):
    with open(out, "wb") as f:
        f.write(header)
        f.truncate(size)


def main(model, out_dir):
    size = os.path.getsize(model)
    buf, end, kv_pos, type_pos = read_header(model)
    header = bytearray(buf[: end + 4096])
    p, t = kv_pos["qwen35moe.embedding_length"]
    assert t == 4
    bad = bytearray(header)
    struct.pack_into("<I", bad, p, 64)
    write(os.path.join(out_dir, "bad_embd.gguf"), bad, size)
    bad = bytearray(header)
    for name in ("blk.20.ffn_gate_exps.weight", "blk.20.ffn_up_exps.weight"):
        struct.pack_into("<I", bad, type_pos[name], 23)
    write(os.path.join(out_dir, "bad_tier.gguf"), bad, size)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
