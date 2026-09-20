#!/usr/bin/env python3
"""Build the "Complete" Orca light main: preserved abliteration on the down experts.

WHY: the shipped OrcaUncensored IQ2 light main keeps 48 trunk down experts at
padded Q2_K and the MTP (blk.48) down at MXFP4. Abliteration only edits those
tensors by a small fraction of each weight's own value, and a 2-bit/MXFP4 grid
rounds that delta away (|dW| < dq/2 -> identical bytes to the base). The light
model is therefore only *partially* uncensored. This tool re-encodes:

  * the 48 trunk ffn_down_exps at padded Q4_K (imatrix-weighted, 640->768)
    -- the engine already reads padded Q4_K down rows in the qwen4 kernels
    (logical in_dim 640, physical 768);
  * the MTP blk.48 ffn_down_exps at Q8_0 (640 logical, no padding) --
    a fine grid that survives the small abliteration delta,

and byte-copies every other tensor from the source light main. Two stages:

  quantize  encode the 48 trunk down experts (Q4_K) + 1 MTP down expert
            (Q8_0) from an Orca BF16 source shard
  assemble  rebuild the light main from a template GGUF, byte-copying every
            other tensor, read-back verified, with provenance written to the
            GGUF and a sidecar manifest

All non-replaced tensor payloads are byte-copied and re-hashed after assembly.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

from qwen4_pack import (SourceDB, QwenGGUFImatrix, GGMLQuantizer,
                        Q2_IMATRIX_SHA256, encode_weighted_experts,
                        bf16_to_f32)
from qwen4_pack_to_qwen4exp import Reader, kv_bytes, w_str, tnbytes

Q4_K = 12
Q8_0 = 8
ORCA_REPO = "orcarouter/Qwen3.8-Flash-Next-Uncensored"
ORCA_REV = "8336e613ea508b13c2159bd0f68965d97a606b95"
MTP_DOWN = "blk.48.ffn_down_exps.weight"
MTP_DOWN_SRC = "mtp.layers.0.mlp.experts.down_proj"
PAD_TO = 768          # trunk down physical width (three Q4_K super-blocks of 256)
MTP_DIMS = [640, 2560, 512]
TRUNK_DIMS = [768, 2560, 512]


def tensor_bytes(kind, dims):
    n = math.prod(dims)
    if kind in (Q4_K, 10, 16):
        if dims[0] % 256:
            raise SystemExit(f'K-quant row {dims[0]} not 256-aligned for kind {kind}')
        per256 = 144 if kind == Q4_K else (84 if kind == 10 else 66)
        return n // 256 * per256
    return tnbytes(kind, n)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def stage_quantize(a):
    a.experts_dir.mkdir(parents=True, exist_ok=True)
    db = SourceDB(a.source)
    imatrix = QwenGGUFImatrix(a.imatrix, Q2_IMATRIX_SHA256)
    quant = GGMLQuantizer(a.library)
    manifest = a.experts_dir / 'experts.json'
    expected = {f'blk.{i}.ffn_down_exps.weight' for i in range(48)} | {MTP_DOWN}
    identity = {'source': str(a.source.resolve()), 'imatrix': imatrix.provenance(),
                'library_sha256': digest(a.library),
                'format': {'trunk_down': 'Q4_K', 'mtp_down': 'Q8_0'},
                'padding': {'logical_input': 640, 'physical_input': PAD_TO}}
    report = json.loads(manifest.read_text()) if manifest.exists() else {**identity, 'tensors': {}}
    for key, value in identity.items():
        if report[key] != value:
            raise SystemExit(f'Resume provenance mismatch: {key}')
    for layer in range(48):
        name = f'blk.{layer}.ffn_down_exps.weight'
        source = f'model.language_model.layers.{layer}.mlp.experts.down_proj'
        info = db.tensors[source]
        if info['dtype'] != 'BF16' or info['shape'] != (512, 2560, 640):
            raise SystemExit(f'Unexpected source layout: {source}: {info}')
        if (name in report['tensors'] and (a.experts_dir / name).is_file() and
                digest(a.experts_dir / name) == report['tensors'][name]['sha256']):
            print(f'{name}: verified existing tensor', flush=True)
            continue
        values = db.read(source)
        source_hash = hashlib.sha256(values).hexdigest()
        start = time.monotonic()
        raw = encode_weighted_experts(values, 'Q4_K', imatrix, name, quant, a.threads,
                                      pad_last_to=PAD_TO)
        want = tensor_bytes(Q4_K, TRUNK_DIMS)
        if len(raw) != want:
            raise SystemExit(f'Invalid Q4_K output size for {name}: {len(raw)} != {want}')
        path = a.experts_dir / name
        tmp = path.with_suffix('.incomplete')
        tmp.write_bytes(raw)
        tmp.replace(path)
        report['tensors'][name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                                   'source_tensor': source, 'source_sha256': source_hash,
                                   'seconds': time.monotonic() - start}
        save(manifest, report)
        print(f'{name}: {len(raw)} bytes, {time.monotonic() - start:.1f}s', flush=True)
        del values
    # MTP down: Q8_0 at its unpadded 640 width.
    info = db.tensors[MTP_DOWN_SRC]
    if info['dtype'] != 'BF16' or info['shape'] != (512, 2560, 640):
        raise SystemExit(f'Unexpected MTP down layout: {MTP_DOWN_SRC}: {info}')
    if not (MTP_DOWN in report['tensors'] and (a.experts_dir / MTP_DOWN).is_file() and
            digest(a.experts_dir / MTP_DOWN) == report['tensors'][MTP_DOWN]['sha256']):
        values = db.read(MTP_DOWN_SRC)
        source_hash = hashlib.sha256(values).hexdigest()
        start = time.monotonic()
        raw = quant.encode(bf16_to_f32(values), 'Q8_0')
        want = tensor_bytes(Q8_0, MTP_DIMS)
        if len(raw) != want:
            raise SystemExit(f'Invalid Q8_0 MTP output size: {len(raw)} != {want}')
        path = a.experts_dir / MTP_DOWN
        tmp = path.with_suffix('.incomplete')
        tmp.write_bytes(raw)
        tmp.replace(path)
        report['tensors'][MTP_DOWN] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                                       'source_tensor': MTP_DOWN_SRC, 'source_sha256': source_hash,
                                       'seconds': time.monotonic() - start}
        save(manifest, report)
        print(f'{MTP_DOWN}: {len(raw)} bytes, {time.monotonic() - start:.1f}s', flush=True)
    imatrix.f = None
    print(f'Verified {len(report["tensors"])} payloads in {manifest}', flush=True)


def stage_assemble(a):
    if a.out.exists() or Path(str(a.out) + '.incomplete').exists():
        raise SystemExit('Output already exists; choose a new path')
    manifest = json.loads((a.experts_dir / 'experts.json').read_text())
    expected = {f'blk.{i}.ffn_down_exps.weight' for i in range(48)} | {MTP_DOWN}
    if set(manifest['tensors']) != expected:
        raise SystemExit(f'Expected all {len(expected)} down payloads, found '
                         f'{sorted(set(manifest["tensors"]) ^ expected)[:4]}')
    base = Reader(str(a.template))
    if base.kv.get('general.architecture') != 'qwen4exp' or base.kv.get('qwen4exp.block_count') != 49:
        raise SystemExit('Template must be a combined 49-layer qwen4exp GGUF')
    if 'per_layer_token_embd.weight' in base.tensors:
        raise SystemExit('Template must use an external PLE sidecar')
    for layer in range(48):
        kind, dims, _off = base.tensors[f'blk.{layer}.ffn_down_exps.weight']
        if kind not in (10, 12) or dims != ([768, 2560, 512] if kind == 10 else [768, 2560, 512]):
            raise SystemExit(f'Unexpected trunk down layout: {kind} {dims}')
    kind, dims, _off = base.tensors[MTP_DOWN]
    if dims != MTP_DIMS:
        raise SystemExit(f'Unexpected MTP down layout: {kind} {dims}')

    alignment = base.kv.get('general.alignment', 32)
    align = lambda n: (n + alignment - 1) // alignment * alignment
    metadata = {k: (base.kv_types[k], v) for k, v in base.kv.items()}
    metadata['general.name'] = (8, 'Qwen3.8 Flash Next Orca Uncensored Complete, '
                                   'IQ2_XXS gate/up, padded Q4_K down, MTP')
    metadata['ds4.orca.down_q4k'] = (2, 1)
    metadata['ds4.qwen4.down.logical_input'] = (4, 640)
    metadata['ds4.qwen4.down.physical_input'] = (4, PAD_TO)
    plan = []
    offset = 0
    for name, (kind, dims, old_offset) in base.tensors.items():
        size = tensor_bytes(kind, dims)
        if name in expected:
            size = tensor_bytes(Q4_K, TRUNK_DIMS) if name != MTP_DOWN else tensor_bytes(Q8_0, MTP_DIMS)
            kind = Q4_K if name != MTP_DOWN else Q8_0
            if name != MTP_DOWN:
                dims = TRUNK_DIMS
            payload = (a.experts_dir / name).open('rb')
            if payload.read(0) is None:
                raise SystemExit('missing payload')
            payload.close()
            if not (a.experts_dir / name).is_file() or \
                    digest(a.experts_dir / name) != manifest['tensors'][name]['sha256']:
                raise SystemExit(f'Invalid payload for {name}')
        plan.append((name, kind, dims, offset, size, old_offset))
        offset = align(offset + size)
    prefix = b'GGUF' + struct.pack('<IQQ', 3, len(plan), len(metadata))
    prefix += b''.join(kv_bytes(k, t, v) for k, (t, v) in metadata.items())
    for name, kind, dims, pos, size, _old in plan:
        prefix += w_str(name) + struct.pack('<I', len(dims))
        prefix += struct.pack('<' + 'Q' * len(dims), *dims) + struct.pack('<IQ', kind, pos)
    data_start = align(len(prefix))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(a.out) + '.incomplete')
    records = {}
    with tmp.open('wb') as out:
        out.write(prefix)
        out.write(bytes(data_start - len(prefix)))
        for name, kind, dims, pos, size, old in plan:
            out.seek(data_start + pos)
            if name in expected:
                src = (a.experts_dir / name).open('rb')
            else:
                src = base.f
                src.seek(base.data_start + old)
            h = hashlib.sha256()
            left = size
            while left:
                chunk = src.read(min(left, 8 << 20))
                if not chunk:
                    raise SystemExit(f'Short tensor read: {name}')
                out.write(chunk)
                h.update(chunk)
                left -= len(chunk)
            if name in expected:
                src.close()
            records[name] = {'type': kind, 'bytes': size, 'sha256': h.hexdigest(),
                             'replaced': name in expected}
        out.truncate(data_start + offset)
    check = Reader(str(tmp))
    for name, rec in records.items():
        check.f.seek(check.data_start + check.tensors[name][2])
        h = hashlib.sha256()
        left = rec['bytes']
        while left:
            chunk = check.f.read(min(left, 8 << 20))
            if not chunk:
                raise SystemExit(f'Short verification read: {name}')
            h.update(chunk)
            left -= len(chunk)
        if h.hexdigest() != rec['sha256']:
            raise SystemExit(f'Written tensor mismatch: {name}')
    check.f.close()
    base.f.close()
    tmp.replace(a.out)
    save(Path(str(a.out) + '.json'), {'template': str(a.template.resolve()),
        'quantization': manifest, 'tensors': records, 'bytes': a.out.stat().st_size,
        'sha256': digest(a.out)})
    print(f'Verified {a.out}: {a.out.stat().st_size} bytes, '
          f'{sum(1 for r in records.values() if r["replaced"])} replaced', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='stage', required=True)
    q = sub.add_parser('quantize')
    q.add_argument('--source', type=Path, required=True, help='Orca BF16 shard dir')
    q.add_argument('--imatrix', type=Path, required=True)
    q.add_argument('--library', type=Path, required=True)
    q.add_argument('--threads', type=int, default=8)
    a = sub.add_parser('assemble')
    a.add_argument('--template', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    for p in (q, a):
        p.add_argument('--experts-dir', type=Path, required=True)
    args = ap.parse_args()
    {'quantize': stage_quantize, 'assemble': stage_assemble}[args.stage](args)


if __name__ == '__main__':
    main()
