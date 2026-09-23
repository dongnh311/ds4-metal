#!/usr/bin/env python3
"""Bonsai-2-style ternary post-training quantization (offline, numpy only).

Pipeline per weight matrix W (contract/input dim = axis 0):
  1. blockwise signed Hadamard rotation (block 1024) folded into the weights;
  2. group-128 ternary {-1,0,+1} assignment with an imatrix-weighted scale;
  3. pack to PTQ1_0 (dense trits, ~1.6 bpw+scale) or PQ2_0 (2-bit slots).

This is the OFFLINE quantizer + a dequant/"simulate" path (rotate back to the
original basis) so quality can be measured on the existing runtime without any
new Metal kernels. Reference: PrismML Ternary Bonsai 2 27B whitepaper; ds4
issue #1085. No C runtime / Metal / CUDA changes (AGENT.md): lives in gguf-tools.
"""
import numpy as np

HAD_BLOCK = 1024
GROUP = 128

# ---------------------------------------------------------------- Hadamard ---
def _fwht(x):
    """In-place fast Walsh-Hadamard transform along axis 0 (len = power of 2),
    normalized so the transform is orthonormal (H = Hᵀ = H⁻¹)."""
    a = x.astype(np.float32, copy=True)
    n = a.shape[0]
    h = 1
    while h < n:
        a = a.reshape(n // (2 * h), 2, h, -1)
        top, bot = a[:, 0], a[:, 1]
        a = np.concatenate([top + bot, top - bot], axis=1).reshape(n, -1)
        h *= 2
    return a / np.sqrt(n)

def _signs(block, seed=0x5eed):
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1.0, 1.0], np.float32), size=block)

def hadamard_rotate(W, block=HAD_BLOCK, inverse=False, seed=0x5eed):
    """Apply the block-diagonal signed Hadamard H (or Hᵀ) along axis 0.
    Forward stores W' = H W (offline); runtime applies the same H to the
    activation x, since (H W)·? we validate weight-space: W ≈ Hᵀ (H W)."""
    d0 = W.shape[0]
    assert d0 % block == 0, f"axis0 {d0} not divisible by Hadamard block {block}"
    s = _signs(block, seed)
    out = np.empty_like(W, dtype=np.float32)
    for i in range(0, d0, block):
        blk = W[i:i + block].astype(np.float32)
        if not inverse:
            out[i:i + block] = _fwht(blk * s[:, None])       # H = FWHT·diag(s)
        else:
            out[i:i + block] = _fwht(blk) * s[:, None]        # Hᵀ = diag(s)·FWHT
    return out

# ---------------------------------------------------------------- ternary ----
def ternary_group(w, imp, iters=5):
    """imatrix-weighted ternary code+scale for one group, Lloyd-Max refined.
    Minimises Σ imp·(w − s·t)², t∈{−1,0,+1}. Optimal 0/±s boundary is |w|=s/2;
    optimal scale s = Σ imp·w·t / Σ imp·t². Alternate the two to convergence."""
    aw = np.abs(w)
    m = aw.mean()
    if m == 0:
        return np.zeros_like(w, np.int8), np.float32(0.0)
    thr = 0.7 * m                                     # init threshold
    s = 0.0
    for _ in range(iters):
        t = np.where(aw > thr, np.sign(w), 0.0).astype(np.float32)
        denom = float((imp * t * t).sum())
        if denom == 0:
            thr *= 0.6
            continue
        s = float((imp * w * t).sum() / denom)
        thr = abs(s) / 2.0                            # nearest-level boundary
    t = np.where(aw > abs(s) / 2.0, np.sign(w), 0.0).astype(np.float32)
    denom = float((imp * t * t).sum())
    s = float((imp * w * t).sum() / denom) if denom else 0.0
    return t.astype(np.int8), np.float32(s)

def quantize(W, imatrix=None, block=HAD_BLOCK, group=GROUP, rotate=True, seed=0x5eed):
    """Quantize W [d0,d1]. Returns (codes int8 [d0,d1] in {-1,0,1}, scales
    [d0//group, d1] fp16, meta). imatrix: per-input importance [d0] or None."""
    W = W.astype(np.float32)
    d0, d1 = W.shape
    Wr = hadamard_rotate(W, block, seed=seed) if rotate else W
    if imatrix is None:
        imp = np.ones(d0, np.float32)
    else:
        imp = np.asarray(imatrix, np.float32).reshape(d0)
        if rotate:
            # In the Hadamard-rotated basis the per-channel importance is
            # E[(Hx)_i²] ≈ block-mean of E[x²] (|H_ik|²=1/block spreads it).
            imp = imp.copy()
            for i in range(0, d0, block):
                imp[i:i + block] = imp[i:i + block].mean()
    assert d0 % group == 0, f"axis0 {d0} not divisible by group {group}"
    ng = d0 // group
    codes = np.zeros((d0, d1), np.int8)
    scales = np.zeros((ng, d1), np.float16)
    for c in range(d1):
        for g in range(ng):
            sl = slice(g * group, (g + 1) * group)
            t, s = ternary_group(Wr[sl, c], imp[sl])
            codes[sl, c] = t
            scales[g, c] = np.float16(s)
    return codes, scales, {"block": block, "group": group, "rotate": rotate,
                           "seed": seed, "shape": (d0, d1)}

def dequantize(codes, scales, meta):
    """Reconstruct W_hat in the ORIGINAL basis (inverse-rotate), so a simulated
    GGUF holds exactly what a correct ternary+Hadamard runtime would compute."""
    d0, d1 = meta["shape"]; g = meta["group"]
    Wr_hat = (codes.astype(np.float32)
              * np.repeat(scales.astype(np.float32), g, axis=0))
    if meta.get("rotate", True):
        return hadamard_rotate(Wr_hat, meta["block"], inverse=True, seed=meta["seed"])
    return Wr_hat

# ---------------------------------------------------------------- packing ----
def pack_pq2_0(codes):
    """2-bit slot per trit ({-1,0,1}->{2,0,1}); 4 trits/byte."""
    m = {-1: 2, 0: 0, 1: 1}
    flat = np.vectorize(m.get)(codes.reshape(-1)).astype(np.uint8)
    pad = (-len(flat)) % 4
    flat = np.concatenate([flat, np.zeros(pad, np.uint8)])
    q = flat.reshape(-1, 4)
    return (q[:, 0] | (q[:, 1] << 2) | (q[:, 2] << 4) | (q[:, 3] << 6)).astype(np.uint8)

def pack_ptq1_0(codes):
    """Dense trits: 5 trits per byte (3^5=243<256)."""
    t = (codes.reshape(-1).astype(np.int16) + 1)  # {0,1,2}
    pad = (-len(t)) % 5
    t = np.concatenate([t, np.zeros(pad, np.int16)]).reshape(-1, 5)
    b = (t[:, 0] + 3 * t[:, 1] + 9 * t[:, 2] + 27 * t[:, 3] + 81 * t[:, 4])
    return b.astype(np.uint8)

def bits_per_weight(meta, packing="ptq1_0"):
    d0, d1 = meta["shape"]; g = meta["group"]
    n = d0 * d1
    scale_bits = (d0 // g) * d1 * 16
    code_bits = (np.ceil(n / 5) * 8) if packing == "ptq1_0" else (np.ceil(n / 4) * 8)
    return (code_bits + scale_bits) / n

# ------------------------------------------------------------------ utils ----
def rel_l2(a, b):
    return float(np.linalg.norm((a - b).ravel()) / (np.linalg.norm(a.ravel()) + 1e-30))

def weighted_rel_l2(a, b, imp):
    """imatrix-weighted relative error along axis 0 (the model-relevant metric):
    sqrt(Σ imp·(a−b)² / Σ imp·a²)."""
    w = np.asarray(imp, np.float32).reshape(-1, 1)
    num = float((w * (a - b) ** 2).sum())
    den = float((w * a ** 2).sum()) + 1e-30
    return (num / den) ** 0.5

def load_imatrix(path):
    """Parse a llama.cpp GGUF imatrix → {tensor_name: importance[n_in]} where
    importance = in_sum2 / counts (mean squared input activation per channel)."""
    import struct
    f = open(path, "rb"); buf = f.read(96 * 1024 * 1024); p = [0]
    def rd(n): b = buf[p[0]:p[0]+n]; p[0]+=n; return b
    def u32(): return struct.unpack('<I', rd(4))[0]
    def u64(): return struct.unpack('<Q', rd(8))[0]
    def gs(): return rd(u64()).decode('utf-8', 'replace')
    SC = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    def skipval(vt):
        if vt in (0,1,7): rd(1)
        elif vt in (2,3): rd(2)
        elif vt in (4,5,6): rd(4)
        elif vt in (10,11,12): rd(8)
        elif vt == 8: gs()
        elif vt == 9:
            et = u32(); c = u64()
            if et == 8:
                for _ in range(c): rd(u64())
            elif et == 9:
                for _ in range(c): skipval(9)
            else: rd(SC[et]*c)
    assert rd(4) == b'GGUF'; u32(); nt = u64(); nk = u64()
    align = 32
    for _ in range(nk):
        k = gs(); vt = u32()
        if k == 'general.alignment' and vt in (4, 10):
            align = struct.unpack('<Q' if vt==10 else '<I', buf[p[0]:p[0]+(8 if vt==10 else 4)])[0]
        skipval(vt)
    infos = []
    for _ in range(nt):
        name = gs(); nd = u32(); dims = [u64() for _ in range(nd)]; tt = u32(); off = u64()
        infos.append((name, dims, tt, off))
    data_start = (p[0] + align - 1) // align * align
    raw = {}
    for name, dims, tt, off in infos:
        if tt != 0:  # F32 only
            continue
        n = 1
        for d in dims: n *= d
        f.seek(data_start + off)
        raw[name] = (np.frombuffer(f.read(n*4), np.float32).reshape(dims[::-1]) if len(dims) > 1
                     else np.frombuffer(f.read(n*4), np.float32).copy())
    f.close()
    out = {}
    for name, arr in raw.items():
        if not name.endswith('.in_sum2'):
            continue
        base = name[:-len('.in_sum2')]
        cnt = raw.get(base + '.counts')
        c = float(np.mean(cnt)) if cnt is not None else 1.0
        out[base] = np.asarray(arr, np.float32) / max(c, 1.0)
    return out
