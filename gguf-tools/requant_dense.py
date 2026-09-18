#!/usr/bin/env python3
"""Re-quantize the per-layer DENSE Q8_0 projections of a qwen4exp GGUF to Q4_0.

Motivation (measured): decode is DRAM-bandwidth-bound; the per-layer dense Q8_0
projections are read in full every token and dominate decode GPU time (~53% at
ctx 32K), while the sparse experts (~10/512 routed) and the KV cache (~4%) do
not. Q8_0 -> Q4_0 roughly halves those dense reads (8.5 -> 4.5 bit), so this is
the lever that can raise decode throughput -- at a quality cost (dense is the
backbone), hence a measured experiment, not a default.

The DS4 runtime accepts only {Q8_0, Q4_0, F16, BF16, F32} for dense projections
(ds4.c:5257); the shipped quantizer can't emit Q4_0, so this packs standard ggml
block_q4_0 in numpy. Pipeline per target: read Q8_0 -> dequant to f32 -> pack
Q4_0 -> write kind=2. Every other tensor is byte-copied. output.weight (final
logits) and any tensor whose row width is not a multiple of 32 stay Q8_0.
Self-verifies the output.
"""
import argparse, hashlib, json, os, shutil, struct, sys
from pathlib import Path
import numpy as np

from qwen4_pack_to_qwen4exp import Reader, kv_bytes, w_str
from qwen4_native_ngrams import size_of, copy_hash

Q8_0 = 8
Q4_0 = 2                       # ggml block_q4_0: fp16 d + 16 bytes qs = 18 B / 32 elems
KEEP_Q8 = {"output.weight"}    # final logit projection: quality-critical, cheap

def dequant_q8_0(raw, n_elem):
    nb = n_elem // 32
    a = np.frombuffer(raw, dtype=np.uint8)[: nb * 34].reshape(nb, 34)
    scales = a[:, :2].copy().view(np.float16).astype(np.float32)
    qs = a[:, 2:].view(np.int8).astype(np.float32)
    return (scales * qs).reshape(-1)[:n_elem]

def q4_0_pack(rows):
    """rows: (nrows, ncols) f32, ncols % 32 == 0 -> ggml block_q4_0 bytes."""
    nrows, ncols = rows.shape
    nb = ncols // 32
    x = np.ascontiguousarray(rows, dtype=np.float32).reshape(nrows, nb, 32)
    ax = np.abs(x)
    idx = np.argmax(ax, axis=2)                                   # index of max-magnitude
    maxv = np.take_along_axis(x, idx[..., None], axis=2)[..., 0]  # signed value there
    d = maxv / -8.0
    inv = np.where(d != 0.0, 1.0 / d, 0.0)[..., None]
    q = np.clip((x * inv + 8.5).astype(np.int32), 0, 15).astype(np.uint8)   # (nrows,nb,32)
    qs = (q[:, :, :16] | (q[:, :, 16:] << 4)).astype(np.uint8)              # (nrows,nb,16)
    dh = d.astype("<f2").view(np.uint8).reshape(nrows, nb, 2)               # fp16 scale
    return np.concatenate([dh, qs], axis=2).reshape(-1).tobytes()          # (nrows,nb,18)

def q4_0_row_bytes(ncols):
    return (ncols // 32) * 18

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    m = Reader(str(args.model))
    if m.kv.get("general.architecture") != "qwen4exp":
        sys.exit("expected qwen4exp GGUF")

    targets = [n for n, (k, sh, o) in m.tensors.items()
               if k == Q8_0 and n not in KEEP_Q8 and sh[0] % 32 == 0]
    nel = lambda sh: int(np.prod(sh))
    old = sum(size_of(Q8_0, m.tensors[n][1]) for n in targets)
    new = sum(q4_0_row_bytes(m.tensors[n][1][0]) * (nel(m.tensors[n][1]) // m.tensors[n][1][0]) for n in targets)
    print(f"targets(Q8_0->Q4_0)={len(targets)}  dense {old/(1<<30):.2f} -> {new/(1<<30):.2f} GiB "
          f"(save {(old-new)/(1<<30):.2f} GiB)", flush=True)
    if args.dry_run:
        for n in targets[:3]:
            sh = m.tensors[n][1]; raw = _read(m, m.tensors[n][2], size_of(Q8_0, sh))
            f = dequant_q8_0(raw, nel(sh)).reshape(-1, sh[0]); b = q4_0_pack(f)
            print(f"  {n}: shape={sh} q8={len(raw)}B q4_0={len(b)}B ratio={len(raw)/len(b):.2f}x", flush=True)
        m.f.close(); return

    pending = Path(str(args.output) + ".incomplete")
    if args.output.exists() or pending.exists(): sys.exit("output exists")
    metadata = {k: (m.kv_types[k], v) for k, v in m.kv.items()}
    metadata["ds4.dense.requant"] = (8, "Q8_0->Q4_0 per-layer dense (decode-bandwidth experiment)")
    alignment = m.kv.get("general.alignment", 32)
    align = lambda n, a=alignment: (n + a - 1) // a * a

    newpayload = {}
    for i, n in enumerate(targets):
        sh = m.tensors[n][1]; raw = _read(m, m.tensors[n][2], size_of(Q8_0, sh))
        f = dequant_q8_0(raw, nel(sh)).reshape(-1, sh[0])
        newpayload[n] = q4_0_pack(f)
        if (i + 1) % 50 == 0: print(f"  requantized {i+1}/{len(targets)}", flush=True)

    plan, off = [], 0
    for n in m.tensors:
        kind = Q4_0 if n in newpayload else m.tensors[n][0]
        sh = m.tensors[n][1]
        size = len(newpayload[n]) if n in newpayload else size_of(m.tensors[n][0], sh)
        plan.append((n, kind, sh, off, size)); off = align(off + size)

    prefix = b"GGUF" + struct.pack("<IQQ", 3, len(plan), len(metadata))
    prefix += b"".join(kv_bytes(k, t, v) for k, (t, v) in metadata.items())
    def header():
        d = bytearray(prefix)
        for n, kind, sh, pos, _ in plan:
            d += w_str(n) + struct.pack("<I", len(sh)) + struct.pack("<" + "Q"*len(sh), *sh) + struct.pack("<IQ", kind, pos)
        return d
    data_start = align(len(header()))
    total = data_start + plan[-1][3] + plan[-1][4]
    if shutil.disk_usage(args.output.parent).free < total + (16 << 30): sys.exit("insufficient disk")
    records = []
    with pending.open("xb") as dst:
        dst.write(header())
        for n, kind, sh, pos, size in plan:
            dst.seek(data_start + pos)
            if n in newpayload:
                dst.write(newpayload[n]); records.append((n, data_start+pos, size, hashlib.sha256(newpayload[n]).hexdigest()))
            else:
                m.f.seek(m.data_start + m.tensors[n][2]); records.append((n, data_start+pos, size, copy_hash(m.f, dst, size)))
        if dst.tell() != total: sys.exit("size mismatch %d!=%d" % (dst.tell(), total))
        dst.flush(); os.fsync(dst.fileno())
    print("verifying...", flush=True)
    with pending.open("rb") as c:
        for n, pos, size, dig in records:
            c.seek(pos)
            if copy_hash(c, None, size) != dig: sys.exit("verify fail " + n)
    rr = Reader(str(pending)); nq4 = sum(1 for _n,(k,_s,_o) in rr.tensors.items() if k == Q4_0); rr.f.close()
    h = hashlib.sha256()
    with pending.open("rb") as fp:
        for ch in iter(lambda: fp.read(8 << 20), b""): h.update(ch)
    Path(str(args.output)+".json").write_text(json.dumps(
        dict(source=str(args.model), bytes=total, sha256=h.hexdigest(),
             requantized=len(targets), q4_0_total=nq4), indent=2) + "\n")
    pending.rename(args.output)
    print(f"DONE -> {args.output} {total} ({total/(1<<30):.2f} GiB) q4_0_tensors={nq4}", flush=True)
    m.f.close()

def _read(m, off, size):
    m.f.seek(m.data_start + off); return m.f.read(size)

if __name__ == "__main__":
    main()
