#!/usr/bin/env python3
"""Qwen3.8-Flash-Next base <-> Orca Uncensored payload diff.

Orca (orcarouter/Qwen3.8-Flash-Next-Uncensored) is a gated BF16
abliterated finetune of Qwen3.8-Flash-Next. The DS4 Orca build reuses
the Ivan IQ2 template for every tensor the abliteration did not touch,
so we must prove, payload by payload, what the base still carries:
gate/up readers, routers, mixers, indexer, vision and the n-gram table
must stay byte-identical.

Stages:
  qwen      range-hash the gate/up reader spot-checks of the pinned
            base checkpoint straight from the hub (no base download)
  orca      hash the changed-set + reader payloads of a locally
            materialized Orca checkpoint (resumable sidecar)
  diff      readers must be IDENTICAL; changed-set payloads are
            recorded as the Orca source provenance for Tasks 2-3

Neither stage hashes whole shards; the base side never downloads
weights at all (hub range reads), the Orca side touches only the
target payload bytes.
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

ORCA_REPO = "orcarouter/Qwen3.8-Flash-Next-Uncensored"
ORCA_REV = "8336e613ea508b13c2159bd0f68965d97a606b95"
QWEN_REPO = "Qwen/Qwen3.8-Flash-Next"
QWEN_REV = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
EXPECTED_TENSOR_COUNT = 1658
EXPECTED_CHANGED = 149
# gate/up reader tensors whose Orca payload MUST stay byte-identical to the
# base payload; the reuse-from-template plan collapses otherwise. Experts
# are fused 3D tensors: `mlp.experts.gate_up_proj` (512 experts x 1280 x
# 2560) and `mlp.experts.down_proj`, so the readers are the gate_up_proj
# payloads plus representative dense readers from the GDN and
# full-attention blocks.
GATE_UP_SPOT_CHECKS = [
    "model.language_model.layers.0.mlp.experts.gate_up_proj",
    "model.language_model.layers.10.mlp.experts.gate_up_proj",
    "model.language_model.layers.30.mlp.experts.gate_up_proj",
    "model.language_model.layers.40.mlp.experts.gate_up_proj",
    "mtp.layers.0.mlp.experts.gate_up_proj",
    "model.language_model.layers.3.self_attn.q_proj.weight",
    "model.language_model.layers.0.linear_attn.conv1d.weight",
]


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def is_changed(name: str) -> bool:
    """The card's 149-set at the fused index level: 48 trunk + 1 MTP expert
    downs, 49 shared-expert downs, 36 GDN out_projs, 12 trunk + 1 MTP
    full-attention o_projs, the layer-1 PLE value_proj, embed_tokens.
    Gate/up readers, routers, mixers, indexer, vision and the n-gram
    table stay byte-identical to the base and are NOT in this set."""
    return (
        name.endswith(".mlp.experts.down_proj")
        or name.endswith(".mlp.shared_expert.down_proj.weight")
        or name.endswith(".linear_attn.out_proj.weight")
        or name.endswith(".self_attn.o_proj.weight")
        or name == "model.language_model.layers.1.ple.value_proj.weight"
        or name == "model.language_model.embed_tokens.weight"
    )


def read_index(root: Path) -> dict:
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        fail(f"{root}: missing model.safetensors.index.json")
    weight_map = json.loads(index.read_text()).get("weight_map", {})
    if not weight_map:
        fail(f"{index}: empty weight_map")
    return weight_map


def read_safetensors_header(path: Path, wanted: set) -> tuple[dict, int]:
    """(selected header, data_offset): the safetensors data_offset is 8 plus
    the header JSON length, exactly what qwen4_pack's SourceDB records."""
    with path.open("rb") as fp:
        raw_length = fp.read(8)
        length = struct.unpack("<Q", raw_length)[0]
        header = json.loads(fp.read(length))
    data_offset = 8 + length
    return {n: header[n] for n in wanted if n in header}, data_offset


def target_names(weight_map: dict) -> list:
    """Changed-set payload names plus the gate/up readers, de-duplicated."""
    names = {n for n in weight_map if is_changed(n)} | set(GATE_UP_SPOT_CHECKS)
    changed = [n for n in names if is_changed(n)]
    if len(changed) != EXPECTED_CHANGED:
        print(f"changed-set count {len(changed)} != expected {EXPECTED_CHANGED}; "
              f"the card is authoritative — inspect before proceeding", file=sys.stderr)
    return sorted(names)


def qwen_stage(out: Path) -> None:
    """Range-hash the base gate/up readers without downloading base shards:
    the reuse plan only needs these payloads, each a single contiguous
    payload read straight from the hub file."""
    import os
    from huggingface_hub import hf_hub_download
    from qwen4_pack import read_hub_range, read_hub_safetensors_header
    token = os.environ.get("HF_TOKEN")
    index = hf_hub_download(QWEN_REPO, revision=QWEN_REV,
                            filename="model.safetensors.index.json", token=token)
    weight_map = json.loads(Path(index).read_text()).get("weight_map", {})
    if len(weight_map) != EXPECTED_TENSOR_COUNT:
        fail(f"base tensor count {len(weight_map)} != {EXPECTED_TENSOR_COUNT}")
    per_shard: dict[str, list] = {}
    for name in GATE_UP_SPOT_CHECKS:
        if name not in weight_map:
            fail(f"base index has no {name}")
        per_shard.setdefault(weight_map[name], []).append(name)
    done: dict = {}
    for ordinal, (shard, names) in enumerate(sorted(per_shard.items()), 1):
        header, data_offset, _ = read_hub_safetensors_header(
            QWEN_REPO, QWEN_REV, shard, token)
        for name in names:
            begin, end = header[name]["data_offsets"]
            begin, end = begin + data_offset, end + data_offset
            print(f"[{ordinal}/{len(per_shard)}] {shard}: range-hash {name} "
                  f"({(end - begin) // 1048576} MiB)", flush=True)
            payload, _total = read_hub_range(
                QWEN_REPO, QWEN_REV, shard, begin, end - 1, token)
            if len(payload) != end - begin:
                fail(f"{name}: hub range read returned {len(payload)} bytes")
            done[name] = hashlib.sha256(payload).hexdigest()
    out.write_text(json.dumps(done, indent=2) + "\n")
    print(f"wrote {out} ({len(done)} reader payload sha256s)")


def orca_stage(root: Path, out: Path) -> None:
    """Hash the Orca target payloads from a local checkpoint root,
    shard by shard, resumable via `<out>.progress.json`."""
    weight_map = read_index(root)
    if len(weight_map) != EXPECTED_TENSOR_COUNT:
        fail(f"Orca tensor count {len(weight_map)} != {EXPECTED_TENSOR_COUNT}")
    for name in GATE_UP_SPOT_CHECKS:
        if name not in weight_map:
            fail(f"Orca index has no {name}")
    targets = target_names(weight_map)
    progress_path = Path(str(out) + ".progress.json")
    done: dict = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    per_shard: dict[str, list] = {}
    for name in targets:
        per_shard.setdefault(weight_map[name], []).append(name)
    for ordinal, (shard, names) in enumerate(sorted(per_shard.items()), 1):
        path = root / shard
        if not path.is_file():
            fail(f"source shard {shard} is not materialized; download the "
                 f"Orca checkpoint into {root} first (stage `download`)")
        needed = [n for n in names if n not in done]
        if not needed:
            print(f"[{ordinal}/{len(per_shard)}] {shard}: complete")
            continue
        header, data_offset = read_safetensors_header(path, set(needed))
        if missing := [n for n in needed if n not in header]:
            fail(f"{shard} is missing tensors {missing[:3]}")
        with path.open("rb") as fp:
            for name in needed:
                begin, end = header[name]["data_offsets"]
                begin, end = begin + data_offset, end + data_offset
                fp.seek(begin)
                chunk = fp.read(end - begin)
                if len(chunk) != end - begin:
                    fail(f"short read of {name} in {shard}")
                done[name] = hashlib.sha256(chunk).hexdigest()
                progress_path.write_text(json.dumps(done, indent=1) + "\n")
        print(f"[{ordinal}/{len(per_shard)}] {shard}: +{len(needed)} "
              f"({len(done)}/{len(targets)})", flush=True)
    if len(done) != len(targets):
        fail(f"hashed {len(done)}/{len(targets)} payloads")
    out.write_text(json.dumps(done, indent=2) + "\n")
    print(f"wrote {out} ({len(done)} payload sha256s)")


def load_map(path: Path, label: str) -> dict:
    data = json.loads(path.read_text())
    missing = [n for n in GATE_UP_SPOT_CHECKS if n not in data]
    if missing:
        fail(f"{label} map {path} is missing spot-checks {missing}")
    return data


def diff_stage(a: Path, b: Path, out: Path) -> None:
    """a = base reader map, b = Orca target map. Readers must be IDENTICAL
    (template reuse depends on it); the Orca changed-set payloads are
    recorded as provenance for Tasks 2-3."""
    base, orca = load_map(a, "base"), load_map(b, "Orca")
    readers = list(GATE_UP_SPOT_CHECKS)
    mismatches = [n for n in readers if base.get(n) != orca.get(n)]
    report = {
        "identical_readers": [n for n in readers if n not in mismatches],
        "reader_mismatches": mismatches,
        "expected_changed_count": sum(1 for n in orca if is_changed(n)),
        "changed_source_sha256": {n: h for n, h in sorted(orca.items())
                                  if is_changed(n)},
    }
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(report['identical_readers'])} readers identical; "
          f"{report['expected_changed_count']} Orca changed-set payloads recorded")
    if mismatches:
        fail(f"gate/up reader payloads differ base vs Orca: {mismatches} — "
             f"the template reuse plan is invalid")


def download() -> None:
    from huggingface_hub import snapshot_download
    dest = Path("gguf/orca-bf16")
    if not dest.is_dir():
        dest.mkdir(parents=True)
    result = snapshot_download(ORCA_REPO, revision=ORCA_REV, local_dir=str(dest))
    print(f"downloaded {ORCA_REPO}@{ORCA_REV[:8]} -> {result}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("download", "qwen", "orca", "diff"))
    ap.add_argument("--root", type=Path, default=Path("gguf/orca-bf16"),
                    help="local Orca checkpoint root (orca hash stage)")
    ap.add_argument("--qwen", type=Path, default=Path("gguf/orca-diff-qwen.json"))
    ap.add_argument("--orca", type=Path, default=Path("gguf/orca-diff-orca.json"))
    ap.add_argument("--out", type=Path, default=Path("gguf/orca-diff.json"))
    args = ap.parse_args()
    if args.stage == "download":
        download()
    elif args.stage == "qwen":
        qwen_stage(args.qwen)
    elif args.stage == "orca":
        orca_stage(args.root, args.orca)
    else:
        if not args.qwen.is_file() or not args.orca.is_file():
            fail("diff stage needs both --qwen and --orca hash maps")
        diff_stage(args.qwen, args.orca, args.out)


if __name__ == "__main__":
    main()
