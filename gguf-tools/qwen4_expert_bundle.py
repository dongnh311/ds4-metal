#!/usr/bin/env python3
"""Pack qwen4exp expert tensors into a contiguous SSD bundle file + JSON index.

Reads plan-1 GGUF (Plan 1: Orca Uncensored IQ2XXS Q2KDownPad768 MTP NNgram) and
extracts the 48 trunk-layer expert tensors into a single binary bundle:

    [layer][expert]{gate_iq2xxs | up_iq2xxs | down_q2k}

packed in layer-major order (blocks are page-aligned to 4 KiB).  The MTP layer
(blk.48) is EXCLUDED — its experts stay resident because the pager only targets
the 48 trunk layers.

The index records per-bundle offsets, sizes, and sha256 digests so the pager can
byte-verify every read without scanning the GGUF again.

Usage:
    python3 gguf-tools/qwen4_expert_bundle.py \\
        --in gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-MTP-NNgram.gguf \\
        --out qwen38-experts.bin \\
        --index qwen38-experts.index.json

Output:
    qwen38-experts.bin   — dense bundle (~33.5 GiB for 48 layers x 512 experts)
    qwen38-experts.index.json — manifest with per-bundle sha256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# GGUF reader (minimal subset of qwen4_pack_to_qwen4exp.Reader)
# ---------------------------------------------------------------------------

GGUF_MAGIC = b"GGUF"
ALIGN = 32
TYPE_BYTES = {
    0: 4,     # F32
    1: 2,     # F16
    2: 18,    # Q4_0
    3: 20,    # Q4_1
    8: 34,    # Q8_0
    10: 84,   # Q2_K (256 elems per block) -- was wrongly 66 (IQ2_XXS's size);
              # see speed-bench/logit-gate/GATE_RESULT.md for the corruption
              # this caused in every down-expert bundle before this fix.
    12: 144,  # Q4_K (256 elems per block)
    16: 66,   # IQ2_XXS (256 elems per block)
    27: 8,    # I64
    30: 2,    # BF16
    39: 17,   # MXFP4 (32 elems per block)
}
BLOCK_ELEMS = {2: 32, 3: 32, 8: 32, 10: 256, 12: 256, 16: 256, 39: 32}


class Reader:
    """Minimal GGUF reader sufficient for payload extraction."""

    def __init__(self, path: str):
        self.f = open(path, "rb")
        assert self.f.read(4) == GGUF_MAGIC, path
        self.version = struct.unpack("<I", self.f.read(4))[0]
        n_tensors = struct.unpack("<q", self.f.read(8))[0]
        n_kv = struct.unpack("<q", self.f.read(8))[0]
        self.kv = {}
        self.kv_types = {}
        for _ in range(n_kv):
            k = self._str()
            t = struct.unpack("<I", self.f.read(4))[0]
            self.kv[k] = self._value(t)
            self.kv_types[k] = t
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._str()
            nd = struct.unpack("<I", self.f.read(4))[0]
            dims = [struct.unpack("<q", self.f.read(8))[0] for _ in range(nd)]
            tt = struct.unpack("<I", self.f.read(4))[0]
            off = struct.unpack("<Q", self.f.read(8))[0]
            self.tensors[name] = (tt, dims, off)
        align = self.kv.get("general.alignment", ALIGN)
        self.data_start = (self.f.tell() + align - 1) & ~(align - 1)

    def _str(self) -> str:
        n = struct.unpack("<q", self.f.read(8))[0]
        return self.f.read(n).decode()

    def _value(self, t: int):
        scalars = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
                   4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<B", 1),
                   10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
        if t in scalars:
            fmt, sz = scalars[t]
            return struct.unpack(fmt, self.f.read(sz))[0]
        if t == 8:
            return self._str()
        if t == 9:
            # Array — skip for bundle packing (we don't need these values)
            et = struct.unpack("<I", self.f.read(4))[0]
            ct = struct.unpack("<q", self.f.read(8))[0]
            for _ in range(ct):
                self._value(et)
            return None
        raise SystemExit(f"unsupported GGUF value type {t}")

    def seek_tensor(self, name: str):
        t, dims, off = self.tensors[name]
        n = 1
        for d in dims:
            n *= d
        if t in BLOCK_ELEMS:
            nbytes = (n // BLOCK_ELEMS[t]) * TYPE_BYTES[t]
        else:
            nbytes = n * TYPE_BYTES.get(t, 0)
        self.f.seek(self.data_start + off)
        return t, dims, nbytes

    def read_tensor_payload(self, name: str) -> bytes:
        """Read the full payload of a tensor."""
        t, dims, nbytes = self.seek_tensor(name)
        return self.f.read(nbytes)

    def read_tensor_rows(self, name: str, row_start: int, row_end: int) -> bytes:
        """Read a contiguous row-slice of a tensor's raw payload bytes.

        For a tensor with dims [D0, D1, D2], the data is laid out in row-major
        order over the last dimension (D2).  This reads rows [row_start:row_end]
        of that last dimension, returning exactly the raw bytes for those slices.
        """
        t, dims, _ = self.seek_tensor(name)
        n = 1
        for d in dims:
            n *= d
        if t in BLOCK_ELEMS:
            total_bytes = (n // BLOCK_ELEMS[t]) * TYPE_BYTES[t]
        else:
            total_bytes = n * TYPE_BYTES[t]

        elem_size = total_bytes / n  # bytes per element
        # Last dim size in bytes
        last_dim_elems = dims[-1]
        last_dim_bytes = int(elem_size * (n // last_dim_elems))

        start_byte = row_start * last_dim_bytes
        end_byte = row_end * last_dim_bytes
        self.f.seek(self.data_start + self.tensors[name][2] + start_byte)
        return self.f.read(end_byte - start_byte)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str, chunk_bytes: int = 64 * 1024 * 1024) -> str:
    """Stream a whole-file sha256 in bounded-size chunks.

    A single f.read() with no size argument loads the entire file into one
    Python bytes object first. For the ~34 GiB bundle that is wasteful; for
    the 147 GiB source GGUF on a 64 GiB machine it does not fit in RAM at
    all -- the earlier version of this function used exactly that pattern
    and the process was silently OOM-killed here, after already spending
    the time to write the bundle and hash every per-bundle digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Bundle packing
# ---------------------------------------------------------------------------

PAGE_SIZE = 4096


def next_page_aligned(offset: int) -> int:
    """Return the next offset aligned to PAGE_SIZE."""
    return (offset + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack qwen4exp experts into SSD bundle")
    parser.add_argument("--in", required=True, dest="input", help="Input GGUF path")
    parser.add_argument("--out", required=True, dest="output", help="Output bundle path")
    parser.add_argument("--index", required=True, dest="index", help="Output index JSON path")
    args = parser.parse_args()

    print(f"Reading GGUF: {args.input}")
    r = Reader(args.input)

    # Verify plan-1 artifact properties
    orca_sha = r.kv.get("ds4.orca.revision", "?")
    ngrams_sha = r.kv.get("ds4.iq2.imatrix_sha256", "?")[:16]
    print(f"  Orca revision: {orca_sha}")
    print(f"  Imatrix sha256: {ngrams_sha}...")
    print(f"  Model name: {r.kv.get('general.name', '?')[:80]}")
    print(f"  expert_count: {r.kv.get('qwen4exp.expert_count')}")
    print(f"  expert_used_count: {r.kv.get('qwen4exp.expert_used_count')}")

    # Layout per layer (sorted by GGUF offset):
    #   blk.N.ffn_gate_exps.weight  type=16 (IQ2_XXS) dims=[2560, 640, 512]
    #   blk.N.ffn_up_exps.weight    type=16 (IQ2_XXS) dims=[2560, 640, 512]
    #   blk.N.ffn_down_exps.weight  type=10 (Q2_K)    dims=[768, 2560, 512]
    # blk.48.* are MTP (type 12/39) and excluded.

    LAYER_COUNT = 48
    EXPERT_COUNT = 512

    bundles = []
    current_offset = 0

    print("Packing expert bundles...")
    for layer in range(LAYER_COUNT):
        for expert in range(EXPERT_COUNT):
            # Gate tensor — read single expert slice
            gate_name = f"blk.{layer}.ffn_gate_exps.weight"
            gate_data = r.read_tensor_rows(gate_name, expert, expert + 1)
            gate_offset = current_offset
            bundles.append({
                "layer": layer,
                "expert": expert,
                "tensor": "gate",
                "offset": gate_offset,
                "size": len(gate_data),
                "type": 16,
                "dims": [2560, 640],
            })
            current_offset = next_page_aligned(current_offset + len(gate_data))

            # Up tensor — read single expert slice
            up_name = f"blk.{layer}.ffn_up_exps.weight"
            up_data = r.read_tensor_rows(up_name, expert, expert + 1)
            up_offset = current_offset
            bundles.append({
                "layer": layer,
                "expert": expert,
                "tensor": "up",
                "offset": up_offset,
                "size": len(up_data),
                "type": 16,
                "dims": [2560, 640],
            })
            current_offset = next_page_aligned(current_offset + len(up_data))

            # Down tensor — read single expert slice
            down_name = f"blk.{layer}.ffn_down_exps.weight"
            down_data = r.read_tensor_rows(down_name, expert, expert + 1)
            down_offset = current_offset
            bundles.append({
                "layer": layer,
                "expert": expert,
                "tensor": "down",
                "offset": down_offset,
                "size": len(down_data),
                "type": 10,
                "dims": [768, 2560],
            })
            current_offset = next_page_aligned(current_offset + len(down_data))

    total_size = current_offset
    print(f"Total bundle size: {total_size} bytes ({total_size/(1024**3):.2f} GiB)")
    print(f"Bundles: {len(bundles)} ({LAYER_COUNT} layers x {EXPERT_COUNT} experts x 3 tensors)")

    # Write bundle file — stream from GGUF, byte-verify each bundle
    print(f"Writing bundle: {args.output}")
    r.f.seek(0)  # reset to beginning
    written = 0
    with open(args.output, "wb") as out:
        for bundle in bundles:
            name = f"blk.{bundle['layer']}.ffn_{bundle['tensor']}_exps.weight"
            payload = r.read_tensor_rows(name, bundle["expert"], bundle["expert"] + 1)
            assert len(payload) == bundle["size"], (name, len(payload), bundle["size"])
            # Pad to page boundary
            padding = bundle["offset"] - written
            if padding > 0:
                out.write(b"\x00" * padding)
            out.write(payload)
            written = bundle["offset"] + bundle["size"]
            # Progress
            if written % (64 * 1024 * 1024) < 1024 * 1024:
                print(f"  ... {written/(1024**3):.1f} GiB written")
        # Pad final element to page boundary
        remaining = total_size - written
        if remaining > 0:
            out.write(b"\x00" * remaining)
            written = total_size
        assert written == total_size, (written, total_size)
        print(f"Bundle written: {total_size/(1024**3):.2f} GiB")

    # Compute per-bundle sha256 (re-read from bundle for verification)
    print("Computing per-bundle sha256 digests...")
    with open(args.output, "rb") as f:
        for bundle in bundles:
            f.seek(bundle["offset"])
            data = f.read(bundle["size"])
            bundle["sha256"] = sha256_bytes(data)

    # Compute bundle file sha256
    print("Hashing bundle file...")
    file_sha = sha256_file(args.output)

    # Compute input GGUF sha256
    print("Hashing input GGUF (147 GiB, streamed)...")
    gguf_sha = sha256_file(args.input)

    # Build index JSON
    gate_sample = bundles[0]
    up_sample = bundles[1]
    down_sample = bundles[2]

    index = {
        "version": 1,
        "layer_count": LAYER_COUNT,
        "expert_count": EXPERT_COUNT,
        "bundle_bytes": {
            "gate": gate_sample["size"],
            "up": up_sample["size"],
            "down": down_sample["size"],
        },
        "bundles": bundles,
        "gguf_sha256": gguf_sha,
        "sha256": file_sha,
    }

    print(f"Writing index: {args.index}")
    with open(args.index, "w") as f:
        json.dump(index, f, indent=2)

    print(f"Done.")
    print(f"  Bundle: {args.output} ({total_size/(1024**3):.2f} GiB)")
    print(f"  Index:  {args.index}")
    print(f"  Bundle sha256: {file_sha}")
    print(f"  GGUF sha256:   {gguf_sha}")
    print(f"  Per-bundle sha256 count: {len([b for b in bundles if 'sha256' in b])}")


if __name__ == "__main__":
    main()
