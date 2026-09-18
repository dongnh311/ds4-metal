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

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
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
            f = dequant_q8_0(raw, nel(sh)).reshape(-1, sh[0]); b = quant.encode(f, "Q4_K")
            print(f"  {n}: shape={sh} q8={len(raw)}B q4k={len(b)}B ratio={len(raw)/len(b):.2f}x", flush=True)
        m.f.close(); return

    pending = Path(str(args.output) + ".incomplete")
    if args.output.exists() or pending.exists(): sys.exit("output exists")
    metadata = {k: (m.kv_types[k], v) for k, v in m.kv.items()}
    metadata["ds4.dense.requant"] = (8, "Q8_0->Q4_K per-layer dense (decode-bandwidth experiment)")
    alignment = m.kv.get("general.alignment", 32)
    align = lambda n, a=alignment: (n + a - 1) // a * a

    newpayload = {}
    for i, n in enumerate(targets):
        sh = m.tensors[n][1]; raw = _read(m, m.tensors[n][2], size_of(Q8_0, sh))
        f = dequant_q8_0(raw, nel(sh)).reshape(-1, sh[0])
        newpayload[n] = quant.encode(f, "Q4_K")
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
