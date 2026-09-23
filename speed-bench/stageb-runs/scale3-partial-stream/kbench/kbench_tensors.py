#!/usr/bin/env python3
"""Write the tensors.txt table the qwen4 kbench tools read: one line per
tensor, `name type file_offset bytes`, for the MoE tensors of the given layers
plus the dense tensors named on the command line.

usage: kbench_tensors.py <model.gguf> <layers,comma,separated> [tensor ...] > tensors.txt
"""
import struct
import sys

# bytes per block and elements per block for the types these models use
TYPE_BLOCK = {0: (4, 1), 1: (2, 1), 8: (34, 32), 10: (84, 256), 12: (144, 256),
              16: (66, 256), 30: (2, 1), 39: (17, 32)}


def read_index(path):
    f = open(path, 'rb')
    assert f.read(4) == b'GGUF', path
    struct.unpack('<I', f.read(4))
    n_tensors, n_kv = struct.unpack('<qq', f.read(16))

    def string():
        n = struct.unpack('<q', f.read(8))[0]
        return f.read(n).decode()

    scalars = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def skip_value(t):
        if t in scalars:
            return f.read(scalars[t])
        if t == 8:
            return string()
        if t == 9:
            et = struct.unpack('<I', f.read(4))[0]
            for _ in range(struct.unpack('<q', f.read(8))[0]):
                skip_value(et)
            return None
        raise SystemExit(f'unsupported GGUF value type {t}')

    align = 32
    for _ in range(n_kv):
        key = string()
        t = struct.unpack('<I', f.read(4))[0]
        v = skip_value(t)
        if key == 'general.alignment':
            align = struct.unpack('<I', v)[0]
    tensors = {}
    for _ in range(n_tensors):
        name = string()
        nd = struct.unpack('<I', f.read(4))[0]
        dims = struct.unpack(f'<{nd}q', f.read(8 * nd))
        t, off = struct.unpack('<IQ', f.read(12))
        tensors[name] = (t, dims, off)
    data_start = (f.tell() + align - 1) // align * align
    return tensors, data_start


def main():
    tensors, data_start = read_index(sys.argv[1])
    names = []
    for layer in sys.argv[2].split(','):
        for n in ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps',
                  'ffn_gate_shexp', 'ffn_up_shexp', 'ffn_down_shexp'):
            names.append(f'blk.{layer}.{n}.weight')
    names += sys.argv[3:]
    for name in names:
        t, dims, off = tensors[name]
        n = 1
        for d in dims:
            n *= d
        block_bytes, block_elems = TYPE_BLOCK[t]
        print(f'{name} {t} {data_start + off} {n // block_elems * block_bytes}')


if __name__ == '__main__':
    main()
