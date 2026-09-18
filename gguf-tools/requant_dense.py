#!/usr/bin/env python3
"""Re-quantize the per-layer DENSE Q8_0 projections of a qwen4exp GGUF to Q4_K.

Motivation (measured): decode is per-token-weight-bandwidth-bound; the per-layer
dense Q8_0 projections dominate decode GPU time (~53% at ctx 32K). Q8_0->Q4_K
roughly halves those dense reads (8.5 -> 4.5 bit) AND, unlike Q4_0, DS4 ships a
decode-grade dense Q4_K GEMV (kernel_mul_mv_q4_K_dense_f32, ~530-650 GB/s) once
the Qwen dense path is wired to it (ds4.c tensor_type_is_qwen4_dense + qwen4_gemv_
rows). Q4_K also keeps much better quality than Q4_0 (proper K-quant super-blocks).

Pipeline per target: read Q8_0 -> dequant to f32 -> encode Q4_K via libds4quants ->
write kind=12. Every other tensor is byte-copied. output.weight (final logits) and
any tensor whose row width is not a multiple of 256 (Q4_K super-block) stay Q8_0 --
notably ffn_down_shexp (ne0=640). Self-verifies the output.
"""
import argparse, hashlib, json, os, shutil, struct, sys
from pathlib import Path
import numpy as np

from qwen4_pack_to_qwen4exp import Reader, kv_bytes, w_str
from qwen4_native_ngrams import size_of, copy_hash
from qwen4_pack import GGMLQuantizer

Q8_0 = 8
Q4_K = 12                      # ggml block_q4_K: 144 B / 256 elems
KEEP_Q8 = {"output.weight"}
# Two exclusion classes kept at Q8_0:
#  (a) special forward paths the Q4_K dense gates reject / that don't route through
#      qwen4_gemv_rows: nextn, output, ple_, token_embd, hc_.
#  (b) SELECTIVE-mode quality guard: the full-attention q/k/v/output projections
#      (dotted patterns so the fused attn_qkv stays Q4_K). These are the most
#      quality-sensitive dense tensors and a small share of per-token dense bytes,
#      so keeping them Q8_0 recovers quality (cosine 0.915->0.930) at ~no decode
#      cost. Drop the four ".attn_*." entries to reproduce the full variant.
SKIP_SUBSTR = ("nextn", "output", "ple_", "token_embd", "hc_",
                ".attn_q.", ".attn_k.", ".attn_v.", ".attn_output.")
LIBRARY = "/Users/dongnh/orca/workspaces/ds4-metal/scallop/gguf-tools/libds4quants.dylib"

def dequant_q8_0(raw, n_elem):
    nb = n_elem // 32
    a = np.frombuffer(raw, dtype=np.uint8)[: nb * 34].reshape(nb, 34)
    scales = a[:, :2].copy().view(np.float16).astype(np.float32)
    qs = a[:, 2:].view(np.int8).astype(np.float32)
    return (scales * qs).reshape(-1)[:n_elem]

def q4k_row_bytes(ncols):
    return (ncols // 256) * 144

def load_gguf_imatrix(path):
    """Parse a llama.cpp GGUF imatrix (general.type=imatrix). Returns
    {tensor_name -> per-column importance f32[ncols]} for the DENSE tensors,
    computed as in_sum2 / max(count, 1) (llama.cpp mean-activation-square). Only
    <name>.in_sum2 (+ optional <name>.counts) entries are read; expert-packed
    2-D in_sum2 are skipped (dense requant needs 1-D per-column vectors)."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF": sys.exit(f"{path}: not a GGUF imatrix")
        struct.unpack("<I", f.read(4))  # version
        nt, = struct.unpack("<Q", f.read(8)); nk, = struct.unpack("<Q", f.read(8))
        def rstr():
            n, = struct.unpack("<Q", f.read(8)); return f.read(n).decode("utf-8", "replace")
        def skipv(t):
            if t in (0, 1, 7): f.read(1)
            elif t in (2, 3): f.read(2)
            elif t in (4, 5, 6): f.read(4)
            elif t in (10, 11, 12): f.read(8)
            elif t == 8: rstr()
            elif t == 9:
                et, = struct.unpack("<I", f.read(4)); c, = struct.unpack("<Q", f.read(8))
                for _ in range(c): skipv(et)
            else: sys.exit(f"imatrix kv type {t}")
        align = 32
        for _ in range(nk):
            k = rstr(); t, = struct.unpack("<I", f.read(4))
            if k == "general.alignment" and t in (4, 5):
                align, = struct.unpack("<I", f.read(4))
            else:
                skipv(t)
        infos = []
        for _ in range(nt):
            name = rstr(); nd, = struct.unpack("<I", f.read(4))
            shape = struct.unpack("<%dQ" % nd, f.read(8 * nd))
            qt, = struct.unpack("<I", f.read(4)); off, = struct.unpack("<Q", f.read(8))
            infos.append((name, shape, qt, off))
        data_start = (f.tell() + align - 1) // align * align
        raw = {}
        for name, shape, qt, off in infos:
            if qt != 0:  # only F32 imatrix stats
                continue
            f.seek(data_start + off)
            raw[name] = (shape, np.frombuffer(f.read(int(np.prod(shape)) * 4), dtype="<f4"))
    imp = {}
    for name, (shape, vals) in raw.items():
        if not name.endswith(".in_sum2") or len(shape) != 1:
            continue  # skip counts and expert-packed 2-D in_sum2
        base = name[: -len(".in_sum2")]
        cnt = raw.get(base + ".counts")
        c = float(cnt[1][0]) if cnt is not None and cnt[1].size else 1.0
        imp[base] = (vals / (c if c > 0 else 1.0)).astype("<f4")
    return imp

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--imatrix", type=Path, default=None,
                    help="GGUF imatrix; per-column importance weights the Q4_K "
                         "quantization of each matched dense tensor (else plain Q4_K)")
    ap.add_argument("--imatrix-strict", action="store_true",
                    help="fail if any Q4_K target has no matching imatrix entry")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    quant = GGMLQuantizer(LIBRARY)
    quant.lib.ds4q_quantize_init(Q4_K)
    m = Reader(str(args.model))
    if m.kv.get("general.architecture") != "qwen4exp":
        sys.exit("expected qwen4exp GGUF")

    targets = [n for n, (k, sh, o) in m.tensors.items()
               if k == Q8_0 and n not in KEEP_Q8 and sh[0] % 256 == 0
               and not any(sub in n for sub in SKIP_SUBSTR)]

    imp = load_gguf_imatrix(str(args.imatrix)) if args.imatrix else {}
    def imatrix_for(n, ncols):
        v = imp.get(n)
        if v is None or v.size != ncols or not np.all(np.isfinite(v)) or v.max() <= 0:
            return None
        return v
    if args.imatrix:
        cov = sum(1 for n in targets if imatrix_for(n, m.tensors[n][1][0]) is not None)
        miss = [n for n in targets if imatrix_for(n, m.tensors[n][1][0]) is None]
        print(f"imatrix: {cov}/{len(targets)} targets weighted, {len(miss)} plain-Q4_K fallback", flush=True)
        if miss[:6]: print("  fallback e.g.:", ", ".join(miss[:6]), flush=True)
        if args.imatrix_strict and miss:
            sys.exit(f"--imatrix-strict: {len(miss)} targets lack imatrix coverage")
    nel = lambda sh: int(np.prod(sh))
    old = sum(size_of(Q8_0, m.tensors[n][1]) for n in targets)
    new = sum(q4k_row_bytes(m.tensors[n][1][0]) * (nel(m.tensors[n][1]) // m.tensors[n][1][0]) for n in targets)
    kept8 = [n for n, (k, sh, o) in m.tensors.items() if k == Q8_0 and n not in targets]
    print(f"targets(Q8_0->Q4_K)={len(targets)}  kept Q8_0={len(kept8)}  "
          f"dense {old/(1<<30):.2f} -> {new/(1<<30):.2f} GiB (save {(old-new)/(1<<30):.2f} GiB)", flush=True)
    if args.dry_run:
        for n in kept8[:8]:
            print(f"  kept Q8_0: {n} ne0={m.tensors[n][1][0]}")
        for n in targets[:3]:
            sh = m.tensors[n][1]; raw = _read(m, m.tensors[n][2], size_of(Q8_0, sh))
            iv = imatrix_for(n, sh[0])
            f = dequant_q8_0(raw, nel(sh)).reshape(-1, sh[0]); b = quant.encode(f, "Q4_K", imatrix=iv)
            print(f"  {n}: shape={sh} q8={len(raw)}B q4k={len(b)}B ratio={len(raw)/len(b):.2f}x "
                  f"imatrix={'yes' if iv is not None else 'no'}", flush=True)
        m.f.close(); return

    pending = Path(str(args.output) + ".incomplete")
    if args.output.exists() or pending.exists(): sys.exit("output exists")
    metadata = {k: (m.kv_types[k], v) for k, v in m.kv.items()}
    tag = "Q8_0->Q4_K per-layer dense (decode-bandwidth experiment)"
    if args.imatrix:
        tag += f"; imatrix-weighted ({Path(args.imatrix).name})"
    metadata["ds4.dense.requant"] = (8, tag)
    alignment = m.kv.get("general.alignment", 32)
    align = lambda n, a=alignment: (n + a - 1) // a * a

    newpayload = {}
    for i, n in enumerate(targets):
        sh = m.tensors[n][1]; raw = _read(m, m.tensors[n][2], size_of(Q8_0, sh))
        f = dequant_q8_0(raw, nel(sh)).reshape(-1, sh[0])
        newpayload[n] = quant.encode(f, "Q4_K", imatrix=imatrix_for(n, sh[0]))
        if (i + 1) % 50 == 0: print(f"  requantized {i+1}/{len(targets)}", flush=True)

    plan, off = [], 0
    for n in m.tensors:
        kind = Q4_K if n in newpayload else m.tensors[n][0]
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
    rr = Reader(str(pending)); nq4 = sum(1 for _n,(k,_s,_o) in rr.tensors.items() if k == Q4_K); rr.f.close()
    h = hashlib.sha256()
    with pending.open("rb") as fp:
        for ch in iter(lambda: fp.read(8 << 20), b""): h.update(ch)
    Path(str(args.output)+".json").write_text(json.dumps(
        dict(source=str(args.model), bytes=total, sha256=h.hexdigest(),
             requantized=len(targets), q4_k_total=nq4), indent=2) + "\n")
    pending.rename(args.output)
    print(f"DONE -> {args.output} {total} ({total/(1<<30):.2f} GiB) q4_k_tensors={nq4}", flush=True)
    m.f.close()

def _read(m, off, size):
    m.f.seek(m.data_start + off); return m.f.read(size)

if __name__ == "__main__":
    main()
