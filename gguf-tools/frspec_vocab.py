#!/usr/bin/env python3
"""Build an FR-Spec draft vocabulary for DS4_QWEN4_MTP_DRAFT_VOCAB.

The MTP draft only needs its argmax, so scoring it over the most frequent
output rows instead of all 248K saves most of the draft head read. The Qwen3.8
tokenizer puts Vietnamese at high ids, so a plain id prefix
(DS4_QWEN4_MTP_DRAFT_ROWS) collapses Vietnamese acceptance; this ranks tokens by
measured frequency over weighted corpora instead.

  frspec_vocab.py --tokenizer tokenizer.json --out draft_vocab.txt --size 65536 \\
      --corpus vi:0.30:vi.parquet --corpus en:0.30:en.parquet \\
      --corpus code:0.20:~/src --corpus py:0.20:/usr/lib/python3 \\
      [--eval held_out_outputs/*.txt]

A corpus is a .parquet with a `text` column (e.g. wikimedia/wikipedia) or a
directory of source files. Every added/special token is always included. With
--eval, prints the list's coverage of those on-policy texts.
"""
import argparse, glob, json, os, random
import numpy as np
from tokenizers import Tokenizer

V = 248320
EXTS = ('.c', '.h', '.m', '.metal', '.py', '.js', '.ts', '.tsx', '.kt', '.java', '.swift', '.go', '.rs',
        '.sh', '.json', '.yaml', '.yml', '.sql', '.cpp', '.hpp', '.md')
SKIP = {'.git', 'node_modules', 'build', 'dist', '.venv', 'venv', 'Pods', '__pycache__', '.gradle', 'DerivedData'}


def texts(path, cap):
    path = os.path.expanduser(path)
    tot = 0
    if path.endswith('.parquet'):
        import pyarrow.parquet as pq
        rows = pq.read_table(path, columns=['text']).column('text').to_pylist()
        random.shuffle(rows)
        for s in rows:
            if tot >= cap: return
            s = s[:20000]; tot += len(s); yield s
        return
    files = []
    for dp, dn, fn in os.walk(path):
        dn[:] = [d for d in dn if d not in SKIP]
        files += [os.path.join(dp, f) for f in fn if f.endswith(EXTS)]
    random.shuffle(files)
    for p in files:
        if tot >= cap: return
        try:
            if not 200 < os.path.getsize(p) < 400_000: continue
            s = open(p, encoding='utf-8', errors='ignore').read()
        except OSError:
            continue
        tot += len(s); yield s


def count(tok, it):
    c = np.zeros(V, dtype=np.int64); batch = []
    def flush():
        for e in tok.encode_batch(batch):
            ids = np.asarray(e.ids, dtype=np.int64); np.add.at(c, ids[ids < V], 1)
        batch.clear()
    for s in it:
        batch.append(s)
        if len(batch) >= 256: flush()
    if batch: flush()
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', type=int, default=65536)
    ap.add_argument('--corpus', action='append', required=True, help='name:weight:path')
    ap.add_argument('--chars', type=int, default=30_000_000, help='characters read per corpus')
    ap.add_argument('--eval', nargs='*', default=[])
    a = ap.parse_args()
    random.seed(1)
    tok = Tokenizer.from_file(a.tokenizer)
    score = np.zeros(V)
    for spec in a.corpus:
        name, w, path = spec.split(':', 2)
        c = count(tok, texts(path, a.chars)).astype(np.float64)
        print(f'{name}: {int(c.sum())} tokens, {int((c > 0).sum())} distinct')
        score += float(w) * c / max(c.sum(), 1.0)
    added = [t['id'] for t in json.load(open(a.tokenizer))['added_tokens']]
    order = [int(i) for i in np.argsort(-score, kind='stable') if score[i] > 0]
    ids = list(dict.fromkeys(order[:a.size] + added))
    with open(a.out, 'w') as f: f.write('\n'.join(map(str, ids)) + '\n')
    print(f'wrote {len(ids)} ids to {a.out}')
    keep = np.zeros(V, bool); keep[ids] = True
    for pattern in a.eval:
        for p in sorted(glob.glob(pattern)):
            e = np.asarray(tok.encode(open(p, encoding='utf-8', errors='ignore').read()).ids)
            print(f'coverage {100.0 * keep[e].mean():6.2f}%  {p}')


if __name__ == '__main__':
    main()
