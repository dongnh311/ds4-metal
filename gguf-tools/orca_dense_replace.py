#!/usr/bin/env python3
"""Assemble the Orca Uncensored qwen4exp main GGUF from the Ivan IQ2
template plus re-encoded abliterated tensors.

The Orca card abliterates 149 residual-writing tensors; everything else
is byte-identical to the base and therefore copied byte-for-byte from
the verified template. Only the changed tensors are re-encoded:

  48 trunk expert downs   Q2_K padded 640->768, imatrix-weighted  (Task 2)
   1 MTP expert down      MXFP4 requant from Orca BF16 (template kind)
  49 shared-expert downs  Q8_0 (template kind)
  36 GDN out_projs        Q8_0
  13 full-attn o_projs    Q8_0 (12 trunk + 1 MTP)
   1 PLE value_proj       Q8_0 (its PLE aux + table ship in the sidecar)
   1 embed_tokens         BF16 copy

The template declares each tensor's GGUF kind; that kind is authoritative,
never hardcoded, so a recipe change cannot silently retarget a tensor.
Payloads are verified by read-back before the file is published, and a
`.json` manifest records per-tensor provenance.
"""
import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

from qwen4_pack import GGMLQuantizer, SourceDB, bf16_to_f32
from qwen4_pack_to_qwen4exp import Reader, kv_bytes, w_str, tnbytes, quant_mxfp4_rows
from qwen4_pack_to_qwen4exp import N_K_HEAD, N_V_PER_K, tiled_v
from orca_diff import is_changed, GATE_UP_SPOT_CHECKS

TEMPLATE_KIND = {
    0: "F32", 1: "F16", 3: "Q4_0", 8: "Q8_0", 10: "Q2_K",
    12: "Q4_K", 16: "IQ2_XXS", 30: "BF16", 39: "MXFP4",
}
T_MXFP4 = 39
# qwen4exp expects the GDN out_proj stored with the 48 head blocks of 128
# output values permuted per the tiled v order (prod_out_colperm in the
# official qwen4exp packer, verified cos=1.0 against the ggml-org file):
# output slot q holds HF head tiled_v(q). A 128-value group is exactly four
# Q8_0 blocks, so the permutation commutes with the encoding.
TILED_ORDER = [tiled_v(j) for j in range(N_K_HEAD * N_V_PER_K)]


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def tensor_bytes(kind: int, dims) -> int:
    n = math.prod(dims)
    if kind in (10, 16):
        if dims[0] % 256:
            fail(f"K-quant weight rows must be block aligned: {dims}")
        return n // 256 * (84 if kind == 10 else 66)
    if kind == T_MXFP4:
        return n // 32 * 17
    return tnbytes(kind, n)


def name_for_experts(rep: dict) -> str:
    layer = rep["source"].split(".layers.")[1].split(".")[0]
    return f"blk.{layer}.ffn_down_exps.weight"


def ssm_colperm(values: np.ndarray) -> np.ndarray:
    """[2560, 6144] GDN out_proj (HF orientation) -> head-block permutation
    of the 6144 input dim: 48 heads of 128 values (four Q8_0 blocks each),
    output slot q holds HF head TILED_ORDER[q] (prod_out_colperm)."""
    out = np.empty_like(values)
    for q, src in enumerate(TILED_ORDER):
        out[:, q * 128:(q + 1) * 128] = values[:, src * 128:(src + 1) * 128]
    return out


class Plan:
    def __init__(self, args):
        base = Reader(str(args.template))
        if base.kv.get("general.architecture") != "qwen4exp" or \
                base.kv.get("qwen4exp.block_count") != 49:
            fail("template must be a combined 49-layer qwen4exp GGUF")
        if "per_layer_token_embd.weight" in base.tensors:
            fail("template must use an external PLE sidecar")
        self.base = base
        self.manifest = json.loads(Path(str(args.template) + ".json").read_text())

        diff = json.loads(args.diff.read_text())
        missing_readers = set(GATE_UP_SPOT_CHECKS) - set(diff["identical_readers"])
        if missing_readers:
            fail(f"reader spot checks not proven identical: {sorted(missing_readers)}; "
                 f"run the orca_diff diff stage first")

        exp_down = {f"blk.{i}.ffn_down_exps.weight" for i in range(48)}
        exp_exp = {f"blk.{i}.ffn_{p}_exps.weight" for i in range(48)
                   for p in ("gate", "up")}
        rec = self.manifest["tensors"]
        replaced = {k for k, v in rec.items() if v.get("replaced")}
        if not exp_down <= replaced:
            fail("template manifest is missing the 48 replaced expert-down payloads")
        if replaced & exp_exp:
            fail("template gate/up expert payloads are marked replaced; "
                 "they must be byte-copied IQ2_XXS")

        db = SourceDB(args.source)
        changed = {n for n in db.tensors if is_changed(n)}
        if len(changed) != 149:
            fail(f"Orca changed set holds {len(changed)} tensors, expected 149")
        self.db = db

        # Replacements: template tensor -> Orca source tensor. The dims below
        # are the template's declared GGUF dims ([ne0, ne1] row-major, ne0 the
        # quantization axis) and are asserted against the template, so the
        # template recipe is authoritative. Orca BF16 payloads come in
        # safetensors order and are already laid out ne0-last, which is the
        # layout the C quantizer and the MXFP4 row quants expect.
        self.replacements: dict[str, dict] = {}
        for layer in range(48):
            self.replacements[f"blk.{layer}.ffn_down_exps.weight"] = {
                "kind": 10, "dims": [768, 2560, 512],
                "source": f"model.language_model.layers.{layer}.mlp.experts.down_proj",
                "encode": "q2k_imatrix"}
        mtp = "mtp.layers.0"
        self.replacements["blk.48.ffn_down_exps.weight"] = {
            "kind": T_MXFP4, "dims": [640, 2560, 512],
            "source": f"{mtp}.mlp.experts.down_proj", "encode": "mxfp4"}
        for layer in range(48):
            self.replacements[f"blk.{layer}.ffn_down_shexp.weight"] = {
                "kind": 8, "dims": [640, 2560],
                "source": f"model.language_model.layers.{layer}.mlp.shared_expert.down_proj.weight",
                "encode": "q8"}
        self.replacements["blk.48.ffn_down_shexp.weight"] = {
            "kind": 8, "dims": [640, 2560],
            "source": f"{mtp}.mlp.shared_expert.down_proj.weight", "encode": "q8"}
        for layer in [i for i in range(48) if i % 4 == 3]:
            self.replacements[f"blk.{layer}.attn_output.weight"] = {
                "kind": 8, "dims": [6144, 2560],
                "source": f"model.language_model.layers.{layer}.self_attn.o_proj.weight",
                "encode": "q8"}
        self.replacements["blk.48.attn_output.weight"] = {
            "kind": 8, "dims": [6144, 2560],
            "source": f"{mtp}.self_attn.o_proj.weight", "encode": "q8"}
        for layer in [i for i in range(48) if i % 4 != 3]:
            self.replacements[f"blk.{layer}.ssm_out.weight"] = {
                "kind": 8, "dims": [6144, 2560],
                "source": f"model.language_model.layers.{layer}.linear_attn.out_proj.weight",
                "encode": "q8_ssm"}
        self.replacements["blk.1.ple_value.weight"] = {
            "kind": 8, "dims": [2560, 2560],
            "source": "model.language_model.layers.1.ple.value_proj.weight",
            "encode": "q8"}
        self.replacements["token_embd.weight"] = {
            "kind": 30, "dims": [2560, 248320],
            "source": "model.language_model.embed_tokens.weight", "encode": "bf16"}

        for tname, rep in self.replacements.items():
            kind, dims, _off = base.tensors[tname]
            if kind != rep["kind"] or list(dims) != rep["dims"]:
                fail(f"template declares {tname} kind={kind} dims={dims}, "
                     f"plan expects kind={rep['kind']} dims={rep['dims']}")

    def encode_payload(self, rep: dict, experts_dir: Path) -> bytes:
        source, kind, encode = rep["source"], rep["kind"], rep["encode"]
        if encode == "bf16":
            return np.ascontiguousarray(self.db.read(source)).tobytes()
        if encode == "q8":
            values = bf16_to_f32(self.db.read(source))
            return self.quant.encode(values, "Q8_0")
        if encode == "q8_ssm":
            values = ssm_colperm(bf16_to_f32(self.db.read(source)))
            return self.quant.encode(values, "Q8_0")
        if encode == "mxfp4":
            values = bf16_to_f32(self.db.read(source))
            blocks = quant_mxfp4_rows(values.reshape(-1, 32))
            return np.ascontiguousarray(blocks).tobytes()
        if encode == "q2k_imatrix":
            payload = (experts_dir / name_for_experts(rep)).read_bytes()
            return payload
        fail(f"unknown encode mode {encode}")

    def build_metadata(self, args) -> dict:
        base = self.base
        metadata = {k: (base.kv_types[k], v) for k, v in base.kv.items()}
        metadata["general.name"] = (
            8, args.name or "Qwen3.8 Flash Next Orca Uncensored, IQ2_XXS imatrix "
               "gate/up, padded Q2_K Orca down, MTP")
        metadata["general.file_type"] = (4, 19)  # LLAMA_FTYPE_MOSTLY_IQ2_XXS
        metadata["ds4.orca.source_repo"] = (8, "orcarouter/Qwen3.8-Flash-Next-Uncensored")
        metadata["ds4.orca.revision"] = (8, "8336e613ea508b13c2159bd0f68965d97a606b95")
        metadata["ds4.orca.dense_replaced"] = (4, len(self.replacements))
        metadata["ds4.orca.abliterated"] = (2, 1)
        return metadata

    def entry_sizes(self) -> list:
        """[(tname, kind, dims, byte_size)] in template order; replaced
        tensors take the template's declared kind/dims, so the output is
        byte-exact with the template for equal-sized payloads."""
        base = self.base
        entries = []
        for tname, (kind, dims, _off) in base.tensors.items():
            rep = self.replacements.get(tname)
            if rep:
                kind, dims = rep["kind"], rep["dims"]
            entries.append((tname, kind, dims, tensor_bytes(kind, dims)))
        return entries

    def assemble(self, args) -> int:
        entries = self.entry_sizes()
        replaced_bytes = sum(s for t, _k, _d, s in entries if t in self.replacements)
        copied_bytes = sum(s for t, _k, _d, s in entries if t not in self.replacements)
        print(f"plan: {len(entries)} tensors; {len(self.replacements)} replaced "
              f"({replaced_bytes / 2**30:.3f} GiB), {copied_bytes / 2**30:.3f} GiB "
              f"byte-copied from template")
        if args.dry_run:
            for tname, kind, dims, size in entries:
                if tname in self.replacements:
                    rep = self.replacements[tname]
                    print(f"  {rep['encode']:>12s} kind={kind} {tname} "
                          f"({size / 1048576:.1f} MiB) <- {rep['source']}")
            print(f"dry run: output would be "
                  f"{(copied_bytes + replaced_bytes) / 2**30:.3f} GiB "
                  f"(template file: {self.manifest.get('bytes', '?')})")
            return 0
        return self.write_out(args, entries)

    def write_out(self, args, entries) -> int:
        exp_manifest = json.loads((args.experts_dir / "experts.json").read_text())
        exp_down = {f"blk.{i}.ffn_down_exps.weight" for i in range(48)}
        if set(exp_manifest["tensors"]) != exp_down:
            fail("experts manifest must hold exactly the 48 trunk expert downs")
        for tname in exp_down:
            payload = (args.experts_dir / tname).read_bytes()
            if hashlib.sha256(payload).hexdigest() != \
                    exp_manifest["tensors"][tname]["sha256"]:
                fail(f"experts payload mismatch: {tname}")

        metadata = self.build_metadata(args)
        alignment = self.base.kv.get("general.alignment", 32)
        align = lambda n: (n + alignment - 1) & ~(alignment - 1)
        offset = 0
        layout = []
        for tname, kind, dims, size in entries:
            layout.append((tname, kind, dims, size, offset))
            offset = align(offset + size)
        prefix = b"GGUF" + struct.pack("<IQQ", 3, len(layout), len(metadata))
        prefix += b"".join(kv_bytes(k, t, v) for k, (t, v) in metadata.items())
        for tname, kind, dims, _size, off in layout:
            prefix += w_str(tname) + struct.pack("<I", len(dims))
            prefix += struct.pack("<" + "Q" * len(dims), *dims) + struct.pack("<IQ", kind, off)
        data_start = align(len(prefix))

        self.quant = GGMLQuantizer(args.library)
        self.quant.lib.ds4q_quantize_init(16)
        base = self.base
        records = {}
        replaced = 0
        tmp = Path(str(args.out) + ".incomplete")
        with tmp.open("wb") as out:
            out.write(prefix)
            out.write(bytes(data_start - len(prefix)))
            for tname, kind, dims, size, off in layout:
                out.seek(data_start + off)
                rep = self.replacements.get(tname)
                if rep is None:
                    old_kind, _old_dims, old = base.tensors[tname]
                    base.f.seek(base.data_start + old)
                    chunk = base.f.read(size)
                    if len(chunk) != size:
                        fail(f"short template read: {tname}")
                    out.write(chunk)
                    records[tname] = {"type": old_kind, "bytes": size,
                                      "sha256": hashlib.sha256(chunk).hexdigest(),
                                      "replaced": False, "source": "template"}
                    continue
                src_values = self.db.read(rep["source"])
                source_sha = hashlib.sha256(np.ascontiguousarray(src_values)).hexdigest()
                payload = self.encode_payload(rep, args.experts_dir)
                if len(payload) != size:
                    fail(f"{tname}: encoded {len(payload)} bytes, expected {size} "
                         f"(kind {kind} dims {dims})")
                out.write(payload)
                records[tname] = {"type": kind, "bytes": size,
                                  "sha256": hashlib.sha256(payload).hexdigest(),
                                  "replaced": True, "source": rep["source"],
                                  "source_sha256": source_sha,
                                  "encode": rep["encode"]}
                replaced += 1
                if replaced % 20 == 0:
                    print(f"re-encoded {replaced}/{len(self.replacements)} ({tname})",
                          flush=True)
            out.truncate(data_start + offset)
        # Read back every payload before publishing the completed artifact.
        check = Reader(str(tmp))
        for tname, rec in records.items():
            check.f.seek(check.data_start + check.tensors[tname][2])
            left, digest = rec["bytes"], hashlib.sha256()
            while left:
                chunk = check.f.read(min(left, 8 << 20))
                if not chunk:
                    fail(f"short verification read: {tname}")
                digest.update(chunk)
                left -= len(chunk)
            if digest.hexdigest() != rec["sha256"]:
                fail(f"written tensor mismatch: {tname}")
        check.f.close()
        base.f.close()
        tmp.replace(args.out)
        out_sha = hashlib.sha256()
        with args.out.open("rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                out_sha.update(chunk)
        sidecar = {
            "template": str(args.template.resolve()),
            "template_sha256": self.manifest.get("sha256"),
            "experts_manifest": {k: exp_manifest[k]
                                 for k in ("imatrix", "library_sha256", "format", "padding")},
            "tensors": records,
            "bytes": args.out.stat().st_size,
            "sha256": out_sha.hexdigest(),
            "ds4.orca.source_repo": "orcarouter/Qwen3.8-Flash-Next-Uncensored",
            "ds4.orca.revision": "8336e613ea508b13c2159bd0f68965d97a606b95",
        }
        Path(str(args.out) + ".json").write_text(json.dumps(sidecar, indent=2) + "\n")
        print(f"Verified {args.out}: {args.out.stat().st_size} bytes "
              f"({replaced} replaced tensors)")
        return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diff", type=Path, required=True,
                    help="orca_diff diff report (readers must be proven identical)")
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True,
                    help="local Orca checkpoint root (config + index + shards)")
    ap.add_argument("--experts-dir", type=Path, required=True,
                    help="Task 2 output: 48 trunk expert-down payloads + manifest")
    ap.add_argument("--library", type=Path, required=True,
                    help="libds4quants.dylib")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default=None, help="general.name override")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.out.exists() or Path(str(args.out) + ".incomplete").exists():
        fail("output already exists; choose a new path")
    sys.exit(Plan(args).assemble(args))


if __name__ == "__main__":
    main()
