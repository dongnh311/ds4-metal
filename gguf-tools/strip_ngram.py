#!/usr/bin/env python3
"""Strip the embedded BF16 n-gram tensor from a qwen4exp GGUF -> light main.

Inverse of qwen4_native_ngrams.py:repack(). repack() takes a light main GGUF (no
n-gram) and appends one BF16 tensor `per_layer_token_embd.weight` plus two
`ds4.qwen4.ngram.*` provenance keys, producing the heavy "NNgram" artifact. This
tool removes exactly that tensor and those two keys, reconstructing the light
main so the model loads with an external demand-paged PLE Q4_1 sidecar (Ivan's
packaging) instead of a 95 GiB resident BF16 n-gram.

Every retained tensor payload is byte-copied unchanged from the input; the tool
re-reads and sha256-verifies every output tensor before committing. It never
writes to or reads the source shards, only the one input GGUF (O_RDONLY).
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import struct

from qwen4_pack_to_qwen4exp import Reader, kv_bytes, w_str
from qwen4_native_ngrams import NGRAM, size_of, copy_hash

# The two KV keys repack() appends when embedding the n-gram; meaningless once
# the n-gram is externalised to a sidecar. Everything else (incl. all
# qwen4exp.ple.* geometry and any ds4.orca.* provenance) is preserved verbatim.
DROP_KEYS = ('ds4.qwen4.ngram.source_repository', 'ds4.qwen4.ngram.source_revision')


def strip(model_path, output, dry_run=False):
    pending = Path(str(output) + '.incomplete')
    if not dry_run and (output.exists() or pending.exists()):
        raise ValueError('Output already exists; choose a new path or inspect the incomplete file')
    model = Reader(str(model_path))
    try:
        if model.kv.get('general.architecture') != 'qwen4exp':
            raise ValueError('Expected a qwen4exp GGUF')
        if NGRAM not in model.tensors:
            raise ValueError('Input has no embedded n-gram tensor (%s); nothing to strip' % NGRAM)

        metadata = {k: (model.kv_types[k], v) for k, v in model.kv.items() if k not in DROP_KEYS}
        alignment = model.kv.get('general.alignment', 32)
        if alignment < 1 or alignment & (alignment - 1):
            raise ValueError('Invalid GGUF alignment')
        align = lambda n, a=alignment: (n + a - 1) // a * a

        plan, offset = [], 0
        for name, (kind, shape, old_offset) in model.tensors.items():
            if name == NGRAM:
                continue
            size = size_of(kind, shape)
            if model.data_start + old_offset + size > model_path.stat().st_size:
                raise ValueError('Main tensor exceeds input file: ' + name)
            plan.append((name, kind, shape, offset, size, old_offset))
            offset = align(offset + size)
        if not plan:
            raise ValueError('No tensors left after strip')

        prefix = b'GGUF' + struct.pack('<IQQ', 3, len(plan), len(metadata))
        prefix += b''.join(kv_bytes(k, t, v) for k, (t, v) in metadata.items())

        def header():
            data = bytearray(prefix)
            for name, kind, shape, pos, _, _ in plan:
                data += w_str(name) + struct.pack('<I', len(shape))
                data += struct.pack('<' + 'Q' * len(shape), *shape) + struct.pack('<IQ', kind, pos)
            return data

        data_start = align(len(header()))
        last_name, _, _, last_pos, last_size, _ = plan[-1]
        total = data_start + last_pos + last_size

        if dry_run:
            print('DRY RUN: tensors=%d kv=%d data_start=%d total=%d (%.2f GiB) dropped_keys=%s'
                  % (len(plan), len(metadata), data_start, total, total / (1 << 30), list(DROP_KEYS)),
                  flush=True)
            return dict(tensors=len(plan), kv=len(metadata), bytes=total)

        output.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output.parent).free < total + (16 << 30):
            raise ValueError('Not enough disk space with a 16 GiB reserve')

        records = []
        with pending.open('xb') as dst:
            dst.write(header())
            for name, kind, shape, pos, size, old in plan:
                dst.seek(data_start + pos)
                model.f.seek(model.data_start + old)
                digest = copy_hash(model.f, dst, size)
                records.append(dict(name=name, offset=data_start + pos, bytes=size, sha256=digest))
            if dst.tell() != total:
                raise ValueError('Incorrect assembled size: %d != %d' % (dst.tell(), total))
            dst.flush()
            os.fsync(dst.fileno())

        print('Verifying every output tensor', flush=True)
        with pending.open('rb') as check:
            for rec in records:
                check.seek(rec['offset'])
                if copy_hash(check, None, rec['bytes']) != rec['sha256']:
                    raise ValueError('Output payload verification failed: ' + rec['name'])

        reread = Reader(str(pending))
        try:
            if NGRAM in reread.tensors:
                raise ValueError('n-gram tensor still present after strip')
            if set(reread.tensors) != {p[0] for p in plan}:
                raise ValueError('Output tensor set mismatch')
            for k in DROP_KEYS:
                if k in reread.kv:
                    raise ValueError('Dropped KV still present: ' + k)
        finally:
            reread.f.close()

        with pending.open('rb') as fp:
            checksum = copy_hash(fp, None, total)
        report = dict(source=str(model_path), bytes=total, sha256=checksum,
                      dropped_tensor=NGRAM, dropped_keys=list(DROP_KEYS), tensors=records)
        Path(str(output) + '.json').write_text(json.dumps(report, indent=2) + '\n')
        pending.rename(output)
        print('Stripped ->', output, total, checksum, flush=True)
        return report
    finally:
        model.f.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='heavy NNgram qwen4exp GGUF')
    parser.add_argument('--output', type=Path, required=True, help='light main GGUF to write')
    parser.add_argument('--dry-run', action='store_true', help='build the plan and print counts without writing')
    args = parser.parse_args()
    try:
        strip(args.model, args.output, dry_run=args.dry_run)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
