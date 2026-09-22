#!/usr/bin/env python3
"""Requantize the external PLE n-gram sidecar (ple.weight) to a smaller format.

The Qwen3.8-Flash-Next PLE sidecar is a single tensor `ple.weight`
[row_dimension=160, row_count=320M] in Q4_1 (~30 GiB), demand-paged CPU-side at
inference. On a 64 GiB box it sits alongside the resident model and, for the
larger uncensor tiers, pushes the working set over the paging cliff. This tool
reads that Q4_1 tensor, dequantizes each 160-dim row and requantizes it to a
smaller block-32 format, writing a new sidecar GGUF that preserves every KV
entry (all qwen4exp.ple.* geometry) verbatim -- only ple.weight's type/payload
changes.

Phase A supports --format q4_0 (existing engine dequant, ~-10%). Phase B will add
custom block-32 sub-4-bit formats.

Rows are streamed in numpy chunks; contents never fully materialize. Every
written chunk is optionally round-trip checked (--verify) against a re-dequant.
"""

import argparse
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen4_pack_to_qwen4exp import Reader, w_str, kv_bytes, GGUF_MAGIC, ALIGN

T_Q4_0, T_Q4_1 = 2, 3
BLK = 32
ROW_DIM = 160
NB = ROW_DIM // BLK          # 5 blocks per row
Q41_ROW = NB * 20            # 100 bytes
Q40_ROW = NB * 18            # 90 bytes

FORMATS = {"q4_0": (T_Q4_0, Q40_ROW)}


def _f16_pairs_to_f32(u8):
    # u8: (..., 2) little-endian half -> (...) float32
    u16 = u8[..., 0].astype(np.uint16) | (u8[..., 1].astype(np.uint16) << 8)
    return u16.view(np.float16).astype(np.float32)


def dequant_q4_1(raw):
    """raw: (R, 100) uint8 -> (R, 160) float32."""
    R = raw.shape[0]
    blk = raw.reshape(R, NB, 20)
    d = _f16_pairs_to_f32(blk[:, :, 0:2])          # (R, NB)
    m = _f16_pairs_to_f32(blk[:, :, 2:4])          # (R, NB)
    nib = blk[:, :, 4:20]                          # (R, NB, 16)
    lo = (nib & 0x0F).astype(np.float32)
    hi = (nib >> 4).astype(np.float32)
    q = np.concatenate([lo, hi], axis=2)           # (R, NB, 32): first 16 lo, next 16 hi
    v = d[:, :, None] * q + m[:, :, None]
    return v.reshape(R, ROW_DIM)


def requant_q4_0(v):
    """v: (R, 160) float32 -> (R, 90) uint8 (ggml Q4_0: out = d*(q-8))."""
    R = v.shape[0]
    vb = v.reshape(R, NB, BLK)
    absv = np.abs(vb)
    idx = np.argmax(absv, axis=2, keepdims=True)    # element with max |.|
    maxv = np.take_along_axis(vb, idx, axis=2)[:, :, 0]   # (R, NB), signed
    d = maxv / -8.0
    inv = np.where(d != 0.0, 1.0 / d, 0.0)
    q = np.clip(np.rint(vb * inv[:, :, None]) + 8.0, 0, 15).astype(np.uint8)  # (R,NB,32)
    lo = q[:, :, 0:16]
    hi = q[:, :, 16:32]
    packed = (lo | (hi << 4)).astype(np.uint8)      # (R, NB, 16)
    df16 = d.astype(np.float16).view(np.uint8).reshape(R, NB, 2)
    out = np.concatenate([df16, packed], axis=2)    # (R, NB, 18)
    return out.reshape(R, Q40_ROW)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Q4_1 PLE sidecar gguf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", default="q4_0", choices=list(FORMATS))
    ap.add_argument("--chunk-rows", type=int, default=1 << 20)
    ap.add_argument("--verify", action="store_true", help="round-trip check each chunk")
    ap.add_argument("--limit-rows", type=int, default=0, help="self-test: only first N rows (invalid sidecar)")
    args = ap.parse_args()

    out_type, out_row = FORMATS[args.format]
    r = Reader(args.src)
    if "ple.weight" not in r.tensors:
        raise SystemExit("src has no ple.weight")
    t, dims, off = r.tensors["ple.weight"]
    if t != T_Q4_1 or len(dims) != 2 or dims[0] != ROW_DIM:
        raise SystemExit(f"unexpected ple.weight: type={t} dims={dims}")
    n_rows = dims[1]
    if args.limit_rows:
        n_rows = min(n_rows, args.limit_rows)
    src_data = r.data_start + off

    # KV verbatim; tensor info: ple.weight new type, dims unchanged, offset 0.
    kv_blob = bytearray()
    n_kv = 0
    for k in r.kv:
        kv_blob += kv_bytes(k, r.kv_types[k], r.kv[k])
        n_kv += 1
    tinfo = w_str("ple.weight") + struct.pack("<I", 2) + struct.pack("<qq", ROW_DIM, n_rows) \
        + struct.pack("<I", out_type) + struct.pack("<Q", 0)
    header = 24 + len(kv_blob) + len(tinfo)
    data_start = (header + ALIGN - 1) & ~(ALIGN - 1)

    tmp = args.out + ".incomplete"
    if os.path.exists(args.out) or os.path.exists(tmp):
        raise SystemExit("output exists; pick a new path")
    fout = open(tmp, "wb", buffering=1 << 24)
    fout.write(GGUF_MAGIC)
    fout.write(struct.pack("<I", 3))
    fout.write(struct.pack("<q", 1))       # n_tensors
    fout.write(struct.pack("<q", n_kv))
    fout.write(kv_blob)
    fout.write(tinfo)
    fout.write(b"\0" * (data_start - header))

    t0 = time.time()
    done = 0
    max_rel = 0.0
    while done < n_rows:
        R = min(args.chunk_rows, n_rows - done)
        r.f.seek(src_data + done * Q41_ROW)
        raw = np.frombuffer(r.f.read(R * Q41_ROW), dtype=np.uint8).reshape(R, Q41_ROW)
        v = dequant_q4_1(raw)
        out = requant_q4_0(v) if out_type == T_Q4_0 else None
        fout.write(out.tobytes())
        if args.verify:
            v2 = dequant_q4_0(out)
            num = np.linalg.norm(v - v2, axis=1)
            den = np.linalg.norm(v, axis=1) + 1e-8
            max_rel = max(max_rel, float(np.max(num / den)))
        done += R
        if (done // args.chunk_rows) % 16 == 0 or done == n_rows:
            el = time.time() - t0
            print(f"{done}/{n_rows} rows ({100*done/n_rows:.1f}%) {el:.0f}s "
                  f"{done*out_row/1e9:.1f}GB" + (f" max_relL2={max_rel:.4f}" if args.verify else ""),
                  flush=True)
    fout.close()
    if not args.limit_rows:
        os.replace(tmp, args.out)
        print(f"wrote {args.out} ({os.path.getsize(args.out)/1e9:.2f} GB)")
    else:
        os.remove(tmp)
        print(f"self-test done ({n_rows} rows)" + (f" max_relL2={max_rel:.4f}" if args.verify else ""))


def dequant_q4_0(raw):
    """raw: (R, 90) uint8 -> (R, 160) float32. out = d*(q-8)."""
    R = raw.shape[0]
    blk = raw.reshape(R, NB, 18)
    d = _f16_pairs_to_f32(blk[:, :, 0:2])
    nib = blk[:, :, 2:18]
    lo = (nib & 0x0F).astype(np.float32)
    hi = (nib >> 4).astype(np.float32)
    q = np.concatenate([lo, hi], axis=2)
    v = d[:, :, None] * (q - 8.0)
    return v.reshape(R, ROW_DIM)


if __name__ == "__main__":
    main()
