#!/usr/bin/env python3
"""Fix the 36 GDN ssm_out payloads in the assembled Orca main GGUF.

Official qwen4exp recipe for blk.N.ssm_out (GDN linear_attn.out_proj):
  * HF out_proj shape is [2560, 6144] = [out, in]; in = 48 v-heads x 128.
  * GGUF ssm_out dims are [6144, 2560]; the Q8_0 payload block axis is the
    in-dim 6144 (the array's last dim in HF orientation, no transpose).
  * prod_out_colperm reorders the 48 in-feature head blocks of 128 values so
    that output head slot q holds HF head tiled_v(q) — matching the GDN
    kernel's tiled v order. A 128-wide group is four Q8_0 blocks, so the
    permutation commutes with the encoding.

The first Orca assembly encoded the Orca out_proj array as plain Q8_0 with no
head-block permutation, so the GDN layers fed out_proj wrong-order v heads
and generated garble. This re-fetches the 36 Orca BF16 payloads from the
pinned hub repo, verifies each against the orca-diff provenance record,
re-encodes with the correct block axis + tiled-v head permutation and writes
them in place, then dequant-checks.
"""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from qwen4_pack import GGMLQuantizer, bf16_to_f32, read_hub_range, read_hub_safetensors_header
from qwen4_pack_to_qwen4exp import Reader, tnbytes, N_K_HEAD, N_V_PER_K, tiled_v

TILED = [tiled_v(j) for j in range(N_K_HEAD * N_V_PER_K)]
ORCA_REPO = "orcarouter/Qwen3.8-Flash-Next-Uncensored"
ORCA_REV = "8336e613ea508b13c2159bd0f68965d97a606b95"
OUT, INN = 2560, 6144          # out_proj [out, in]; in = 48*128
N_HEAD = INN // 128


def outproj_perm(values: np.ndarray) -> np.ndarray:
    """[out, in] HF array -> in-feature head blocks permuted to the tiled
    v order, matching prod_out_colperm (output slot q holds HF head
    tiled_v(q))."""
    assert values.shape == (OUT, INN), values.shape
    out = np.empty_like(values)
    for q in range(N_HEAD):
        out[:, q * 128:(q + 1) * 128] = values[:, TILED[q] * 128:(TILED[q] + 1) * 128]
    return out


def dequant_q8(payload: bytes, rows: int, cols: int) -> np.ndarray:
    """Q8_0 ggml block_q8_0: f16 scale d FIRST, then 32 int8 quants
    (34 bytes = 2 + 32). Block axis = the last dim of the encoded array."""
    a = np.frombuffer(payload, dtype=np.uint8).reshape(rows, cols // 32, 34)
    scale = np.frombuffer(a[:, :, 0:2].copy().tobytes(), dtype="<u2") \
        .reshape(rows, cols // 32).view("<f2").astype(np.float32)
    qs = a[:, :, 2:34].astype(np.int16)
    qs = np.where(qs >= 128, qs - 256, qs)
    return (qs * scale[..., None]).reshape(rows, cols)


def main() -> int:
    gguf_path = Path(sys.argv[1])
    diff_map = json.loads(Path("gguf/orca-diff-orca.json").read_text())
    token = os.environ.get("HF_TOKEN")
    if not token:
        tok = Path.home() / ".cache" / "huggingface" / "token"
        if tok.is_file():
            token = tok.read_text().strip()
        if not token:
            sys.exit("no HF token available for the gated Orca repo")

    reader = Reader(str(gguf_path))
    names = [n for n in reader.tensors
             if n.endswith(".ssm_out.weight") and reader.tensors[n][0] == 8]
    assert len(names) == 36, names

    from huggingface_hub import hf_hub_download
    index = hf_hub_download(ORCA_REPO, revision=ORCA_REV,
                            filename="model.safetensors.index.json", token=token)
    weight_map = json.loads(Path(index).read_text())["weight_map"]
    per_shard: dict[str, list] = {}
    for name in names:
        layer = name.split(".")[1]
        src = f"model.language_model.layers.{layer}.linear_attn.out_proj.weight"
        if src not in diff_map:
            sys.exit(f"no provenance record for {src}")
        per_shard.setdefault(weight_map[src], []).append(src)

    quant = GGMLQuantizer("gguf-tools/libds4quants.dylib")
    quant.lib.ds4q_quantize_init(16)
    writer = open(str(gguf_path), "r+b")
    sidecar = json.loads(Path(str(gguf_path) + ".json").read_text())
    fixed = 0
    for shard, srcs in sorted(per_shard.items()):
        header, data_offset, _ = read_hub_safetensors_header(
            ORCA_REPO, ORCA_REV, shard, token)
        for src in srcs:
            begin, end = header[src]["data_offsets"]
            begin, end = begin + data_offset, end + data_offset
            payload, _ = read_hub_range(ORCA_REPO, ORCA_REV, shard,
                                        begin, end - 1, token)
            if hashlib.sha256(payload).hexdigest() != diff_map[src]:
                sys.exit(f"{src}: hub payload sha256 != orca-diff record")
            vals = bf16_to_f32(np.frombuffer(payload, dtype="<u2"))
            assert vals.size == OUT * INN, vals.size
            vals = vals.reshape(OUT, INN)
            perm = outproj_perm(vals)
            enc = quant.encode(perm, "Q8_0")
            gname = next(n for n in names
                         if src.split(".layers.")[1].split(".")[0]
                         == n.split(".")[1])
            _kind, dims, off = reader.tensors[gname]
            assert list(dims) == [INN, OUT], dims
            assert tnbytes(8, INN * OUT) == len(enc)
            rel = float(np.sqrt(np.mean((dequant_q8(enc, OUT, INN) - perm) ** 2)) /
                        np.sqrt(np.mean(perm ** 2)))
            assert rel < 0.05, (gname, rel)
            writer.seek(reader.data_start + off)
            writer.write(enc)
            writer.flush()
            rec = next(r for r in sidecar["tensors"] if r.get("name") == gname)
            rec["sha256"] = hashlib.sha256(enc).hexdigest()
            rec["fix"] = "Q8_0 over in-dim 6144 + tiled-v head-block permute"
            fixed += 1
            print(f"{gname}: rel_l2={rel:.4f} "
                  f"sha256={hashlib.sha256(enc).hexdigest()[:16]}", flush=True)
    writer.close()
    new_sha = hashlib.sha256()
    with gguf_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            new_sha.update(chunk)
    sidecar["sha256"] = new_sha.hexdigest()
    Path(str(gguf_path) + ".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"rewrote {fixed} ssm_out payloads in {gguf_path}")
    print(f"new file sha256 {new_sha.hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
