#!/usr/bin/env python3
"""Rel-L2 gate: dequant sample rows of the new Q4_K trunk / Q8_0 MTP down
tensors and compare to the Orca BF16 source. Q4_K-class ~0.07, Q8_0 ~0.006."""
import sys, struct, random
import numpy as np
sys.path.insert(0, 'gguf-tools')
from pathlib import Path
from qwen4_pack_to_qwen4exp import Reader
from qwen4_pack import SourceDB, bf16_to_f32


def deq_q4k_block(row144):
    """ggml q4_K layout: [d f16][dmin f16][scales 12B][qs 128B]."""
    d = struct.unpack('<e', row144[0:2])[0]
    dm = struct.unpack('<e', row144[2:4])[0]
    s = row144[4:16]
    qs = row144[16:144]
    out = np.zeros(256, np.float32)
    for g in range(8):
        if g < 4:
            scv = s[g] & 0x3F
            mnv = s[4 + g] & 0x3F
        else:
            scv = ((s[g - 4] & 0xC0) >> 2) | (s[g + 4] & 0x0F)
            mnv = ((s[g] & 0xC0) >> 2) | ((s[g + 4] >> 4) & 0x0F)
        shift = (g & 1) * 4
        for j in range(32):
            out[g * 32 + j] = d * scv * ((qs[(g >> 1) * 32 + j] >> shift) & 0xF) - dm * mnv
    return out


def deq_q8_0_block(row34):
    """DS4 q8_0 layout: [d f16][qs 32B signed] (d first, per the metal reader)."""
    d = struct.unpack('<e', row34[0:2])[0]
    qs = np.frombuffer(row34[2:34], dtype=np.int8)
    return qs.astype(np.float32) * d


def relL2(new, old):
    return float(np.linalg.norm(new - old) / (np.linalg.norm(old) + 1e-9))


random.seed(42)
db = SourceDB(Path('gguf/orca-bf16-selective'))
new = Reader('gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q4KDownPad768-MTP.gguf')

# trunk Q4_K: 3 super-blocks of 144 B per row
q4k_rels = []
for L in (0, 24, 47):
    src = bf16_to_f32(db.read(f'model.language_model.layers.{L}.mlp.experts.down_proj'))
    rels = []
    for e in random.sample(range(512), 4):
        idx = random.sample(range(2560), 6)
        out = np.zeros((len(idx), 640), np.float32)
        base = new.tensors[f'blk.{L}.ffn_down_exps.weight'][2] + e * 2560 * 432
        for i, r in enumerate(idx):
            new.f.seek(new.data_start + base + r * 432)
            row = new.f.read(432)
            out[i] = np.concatenate([deq_q4k_block(row[b * 144:(b + 1) * 144])
                                     for b in range(3)])[:640]
        rels.append(relL2(out, src[e][idx]))
    q4k_rels.extend(rels)
    print(f'blk.{L} trunk Q4_K rel-L2 vs Orca BF16: '
          f'{["%.4f" % x for x in rels]} mean={np.mean(rels):.4f}')
q4k_mean = float(np.mean(q4k_rels))

# MTP Q8_0: 20 blocks of 34 B per row
mtp = bf16_to_f32(db.read('mtp.layers.0.mlp.experts.down_proj'))
rels = []
for e in random.sample(range(512), 6):
    idx = random.sample(range(2560), 6)
    out = np.zeros((len(idx), 640), np.float32)
    base = new.tensors['blk.48.ffn_down_exps.weight'][2] + e * 2560 * 680
    for i, r in enumerate(idx):
        new.f.seek(new.data_start + base + r * 680)
        row = new.f.read(680)
        for b in range(20):
            out[i][b * 32:(b + 1) * 32] = deq_q8_0_block(row[b * 34:(b + 1) * 34])
    rels.append(relL2(out, mtp[e][idx]))
q8_mean = float(np.mean(rels))
print(f'blk.48 MTP Q8_0 rel-L2 vs Orca BF16: '
      f'{["%.5f" % x for x in rels]} mean={q8_mean:.5f}')
new.f.close()

print()
print(f'GATE: trunk Q4_K mean={q4k_mean:.4f} (want ~0.05-0.10); '
      f'MTP Q8_0 mean={q8_mean:.5f} (want ~0.003-0.008)')
ok = 0.04 <= q4k_mean <= 0.12 and 0.002 <= q8_mean <= 0.01
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
