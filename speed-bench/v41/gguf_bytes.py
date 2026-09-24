#!/usr/bin/env python3
"""Per-token byte accounting for a DeepSeek-V4.1 GGUF (docs/V41_64GB_BUILD.md §7 step 5).

Reads only the GGUF header. Tensor sizes come from consecutive data offsets
(the last tensor runs to the end of the file), so alignment padding is counted.
"""
import argparse
import json
import os
import re
import struct
import sys

ENGRAM_COLS = 24          # ds4_engram.h DS4_ENGRAM_COLS: rows per table per token
ENGRAM_ROW_BYTES = 264    # ds4_engram.h DS4_ENGRAM_ROW_BYTES
GIB = 1024 ** 3
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
           10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
ROUTED = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
SITE_NORM = re.compile(r"^(output_norm|blk\.\d+\.(attn|ffn)_norm)\.weight$")


class _Reader:
    def __init__(self, fp):
        self.fp = fp

    def take(self, fmt):
        size = struct.calcsize(fmt)
        data = self.fp.read(size)
        if len(data) != size:
            raise ValueError("truncated GGUF header")
        return struct.unpack(fmt, data)[0]

    def string(self):
        n = self.take("<Q")
        data = self.fp.read(n)
        if len(data) != n:
            raise ValueError("truncated GGUF string")
        return data.decode("utf-8", errors="replace")

    def value(self, vtype):
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            etype, n = self.take("<I"), self.take("<Q")
            return [self.value(etype) for _ in range(n)]
        if vtype not in _SCALAR:
            raise ValueError(f"unknown GGUF value type {vtype}")
        return self.take(_SCALAR[vtype])


def read_header(path):
    """(metadata, [(name, dims, type, offset)], data_start)."""
    with open(path, "rb") as fp:
        if fp.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        r = _Reader(fp)
        version = r.take("<I")
        if version != 3:
            raise ValueError(f"{path}: GGUF version {version}, expected 3")
        n_tensors, n_kv = r.take("<Q"), r.take("<Q")
        meta = {}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = r.value(r.take("<I"))
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.take("<Q") for _ in range(r.take("<I"))]
            ttype, offset = r.take("<I"), r.take("<Q")
            tensors.append((name, dims, ttype, offset))
        align = meta.get("general.alignment", 32)
        data_start = (fp.tell() + align - 1) // align * align
    return meta, tensors, data_start


def tensor_sizes(tensors, data_start, file_bytes):
    """name -> bytes from consecutive offsets; the last tensor ends at EOF."""
    ordered = sorted(tensors, key=lambda t: t[3])
    sizes = {}
    for i, (name, _, _, offset) in enumerate(ordered):
        end = ordered[i + 1][3] if i + 1 < len(ordered) else file_bytes - data_start
        sizes[name] = end - offset
    return sizes


def role(name):
    if name == "token_embd.weight":
        return "embedding"
    if SITE_NORM.match(name):
        return "norm"
    if ROUTED.match(name):
        return "routed"
    if name.endswith(".engram_embd.weight"):
        return "engram_disk"
    if ".indexer.attn_k." in name or ".indexer.k_norm." in name:
        return "compressor"
    if name.startswith("output"):
        return "output"
    for key, label in (("_shexp", "shared"), ("ffn_gate_inp", "router"), ("exp_probs", "router"),
                       ("indexer", "indexer"), ("compressor", "compressor"), (".hc_", "mhc"),
                       (".engram_", "engram"), (".attn", "attention"), ("norm", "norm")):
        if key in name:
            return label
    return "other"


def account(meta, tensors, data_start, file_bytes):
    arch = meta.get("general.architecture", "deepseek41")
    n_layer = meta[f"{arch}.num_hidden_layers"]
    n_expert = meta[f"{arch}.n_routed_experts"]
    k = meta[f"{arch}.num_experts_per_tok"]
    sizes = tensor_sizes(tensors, data_start, file_bytes)
    dims = {name: d for name, d, _, _ in tensors}
    roles, units = {}, {}
    for name, size in sizes.items():
        label = role(name)
        roles[label] = roles.get(label, 0) + size
        m = ROUTED.match(name)
        if m:
            layer = int(m.group(1))
            units[layer] = units.get(layer, 0) + size // n_expert
    engram_tables = sum(1 for name in sizes if role(name) == "engram_disk")
    resident = sum(b for label, b in roles.items()
                   if label not in ("routed", "engram_disk", "embedding"))
    per_token = {
        "resident": resident,
        "embedding_row": roles.get("embedding", 0) // dims["token_embd.weight"][1],
        "routed": roles.get("routed", 0) * k // n_expert,
        "engram_rows": engram_tables * ENGRAM_COLS * ENGRAM_ROW_BYTES,
    }
    per_token["ram_total"] = per_token["resident"] + per_token["embedding_row"] + per_token["routed"]
    return {
        "n_layer": n_layer, "n_expert": n_expert, "k": k,
        "file_bytes": file_bytes, "n_tensors": len(tensors),
        "roles": roles,
        "engram_tables": engram_tables,
        "resident_floor": resident + roles.get("embedding", 0),
        "per_token": per_token,
        "expert_unit_bytes": {str(layer): b for layer, b in sorted(units.items())},
    }


def roofline(acct, gbps):
    """Byte-floor milliseconds per decoded token at `gbps` GB/s of memory bandwidth."""
    pt = acct["per_token"]
    out = {f"{key}_ms": pt[key] / (gbps * 1e9) * 1e3
           for key in ("resident", "embedding_row", "routed")}
    out["total_ms"] = sum(out.values())
    out["tps_ceiling"] = 1e3 / out["total_ms"] if out["total_ms"] else 0.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--gbps", type=float, default=290.0,
                    help="memory bandwidth for the roofline (measured Q8_0 gemv: 290)")
    ap.add_argument("--out")
    args = ap.parse_args()
    meta, tensors, data_start = read_header(args.model)
    acct = account(meta, tensors, data_start, os.path.getsize(args.model))
    acct["roofline"] = roofline(acct, args.gbps)
    acct["gbps"] = args.gbps
    text = json.dumps(acct, indent=1)
    if args.out:
        with open(args.out, "w") as fp:
            fp.write(text + "\n")
    pt = acct["per_token"]
    print(f"tensors={acct['n_tensors']} layers={acct['n_layer']} experts={acct['n_expert']} k={acct['k']}")
    for label, b in sorted(acct["roles"].items(), key=lambda kv: -kv[1]):
        print(f"  {label:12s} {b / GIB:9.2f} GiB")
    print(f"resident floor {acct['resident_floor'] / GIB:.2f} GiB; per token: resident "
          f"{pt['resident'] / GIB:.2f} GiB, routed {pt['routed'] / GIB:.2f} GiB, "
          f"engram {pt['engram_rows']} B from {acct['engram_tables']} tables")
    r = acct["roofline"]
    print(f"roofline @{args.gbps:g} GB/s: {r['total_ms']:.1f} ms/token = {r['tps_ceiling']:.1f} t/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
