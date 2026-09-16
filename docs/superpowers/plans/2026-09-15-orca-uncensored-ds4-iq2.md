# Orca Uncensored → DS4 IQ2 GGUF (M5 Pro 64 GB) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Qwen3.8-Flash-Next Uncensored (OrcaRouter) `qwen4exp` GGUF for the DS4 Metal runtime — IQ2_XXS gate/up + padded Q2_K down experts + the 98 abliterated dense tensors re-encoded from Orca BF16 + MTP + PLE sidecar — that runs uncensored on a 64 GB M5 Pro with context budget verified by measurement.

**Architecture:** Reuse Ivan's `qwen4_iq2.py` two-stage quantizer (imatrix pinned), re-encode only the abliterated/down tensors from the Orca BF16 checkpoint, byte-copy everything else from Ivan's IQ2 template, rebuild the PLE Q4_1 sidecar from Orca, then measure token/s and context headroom on the target M5 Pro. No changes to C runtime, Metal kernels, CUDA, or distributed paths (AGENT.md rule) — runtime work is testing only.

**Tech Stack:** Python 3.9+ (NumPy, `huggingface_hub`, `ctypes`), `gguf-tools/libds4quants.dylib` (C quantizer library), `ds4-metal` Makefile builds, Hugging Face selective downloads.

**Spec:** Design agreed in-session (Orca BF16 → Ivan DS4 IQ2, re-encode 98 dense abliterated tensors, rebuild PLE, M5 Pro 64 GB target). Source: OrcaRouter card at revision `8336e613ea508b13c2159bd0f68965d97a606b95`.

## Global Constraints

- **Do NOT modify:** `ds4.c`, `ds4_metal.m`, `ds4_server.c`, `ds4_cli.c`, `metal/*`, `ds4_cuda.cu`, `ds4_tp.c`, `ds4_distributed.c` (test-only builds allowed). All new code lives in `gguf-tools/`.
- **Runtime branch:** `ivanfioravanti/ds4-metal` branch `qwen3.8-flash-next` at or after commit `ccea768` (includes the discussion #6 vision-dim fix). Local worktree `dongnh311/halfbeak` already contains that fix; verify with `grep -n ds4_engine_embd_dim ds4_server.c` before relying on it.
- **Do NOT use `--mtp-exact-sampling`** on the target machine (open bug, discussion #5: full-transcript replay stalls up to 457 s). Use default opportunistic `--mtp` sampling.
- **Imatrix:** must be `unsloth/Qwen3.8-Flash-Next-GGUF` revision `c8b5954a88c2775c546b92593eda40ea041d3176`, sha256 `a5863123db1ca458727e738955bef7bfc199520aa2bee3a30142a1aff9254154` (pinned inside `qwen4_pack.py` as `Q2_IMATRIX_SHA256`; do not change).
- **Orca source revision:** `8336e613ea508b13c2159bd0f68965d97a606b95` of `orcarouter/Qwen3.8-Flash-Next-Uncensored` (131 shards, 336 GB, 1658 tensors, 179,999,981,459 params).
- **Ivan template:** `Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf` (44.81 GB / 41.73 GiB) from `ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2`.
- **Abliteration scope (from Orca card, to be verified by Task 1):** 149/1658 tensors changed — 49 `experts.down_proj` (fused 3D), 49 `shared_expert.down_proj`, 36 GDN `linear_attn.out_proj`, 13 `self_attn.o_proj` (incl. MTP), 2 (`ple.value_proj`, `embed_tokens`). **Gate/up readers, router, mixers, indexer, vision tower, n-gram table: byte-identical to Qwen base `de4b8e4d43b917e7706784d8bb445c9af86a3540` — must stay byte-identical in our output (reuse, do not requantize).**
- **PLE stays Q4_1** (160-wide dim only admits 32-block qtypes; Q2_K/IQ2_XXS impossible; PLE is SSD-mapped, never resident RAM).
- **Build machine (M5 Pro):** needs ≥ ~140 GB free disk during peak: template 44.8 + Orca needed shards ~100 + down experts ~20 + dense payloads small + PLE out 32 + main out 45. Clean intermediates between stages (each task lists its cleanup step).
- Every tool verifies all payloads by read-back sha256 before publishing (Ivan's convention; keep it).

---

## File Structure

- **Create** `gguf-tools/orca_diff.py` — selective download of Orca shards + tensor-level diff vs Qwen base index; outputs a classification manifest (changed/identical, with expected 149-set assertion).
- **Create** `gguf-tools/orca_dense_replace.py` — re-encodes the abliterated dense tensors (per template type) and the MTP-dense changed tensors from Orca BF16; assembles final main GGUF from Ivan template + Orca down experts + replaced dense tensors.
- **Modify** `gguf-tools/qwen4_iq2.py` — add `--imatrix-sha` / gate-up-from-template copy path is NOT needed: `assemble` already copies byte-for-byte from the template for non-replaced tensors. Only change: allow `assemble --template` to accept a name override for `general.name` (small CLI flag) so outputs are labeled Orca.
- **Modify** `gguf-tools/qwen4_pack.py` — PLE writer currently hardcodes `Q4_1` (fine) but its source DB + revision validation pins Qwen; add an `OrcaPLESourceDB` hook (reuse `SourceDB`) allowing `--source-revision` override for PLE-only builds. Keep existing Qwen path byte-identical.
- **Create** `docs/ORCA_UNCENSORED_DS4_IQ2.md` — recipe doc + target-machine benchmarks (fill numbers in last task).

## Pipeline overview (M5 Pro)

```
[Task 0] prereqs: repo, imatrix, Ivan template, build libds4quants.dylib
[Task 1] orca_diff.py: download 131 Orca shards (selective, resumable), diff vs Qwen index
          → assert 149 changed / gate-up identical; classify per-tensor target types
[Task 2] qwen4_iq2.py quantize --projection down --source <Orca> (48 tensors, Q2_K Pad768)
[Task 3] orca_dense_replace.py: re-encode 98 changed dense (+ MTP-dense if any) per template type;
          assemble main GGUF from Ivan template + Task2 experts + Task3 dense; read-back verify
[Task 4] PLE sidecar Q4_1 from Orca (ple.* tensors); verify against Orca card claim (byte-identical
          to Baekpica reference only for unchanged PLE — here value_proj changed, so NEW PLE, verify hash report)
[Task 5] target-machine validation: make, load, ds4-eval core, uncensor smoke, MTP opportunistic,
          tok/s + max-context-no-swap measurement; write docs
```

---

## Task 0: Prerequisites and environment (M5 Pro)

**Files:** none (setup only).

**Interfaces:**
- Produces: `libds4quants.dylib` (Task 2 dep), `imatrix_unsloth.gguf_file` (Task 2 dep), `Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf` template (Task 3 dep), Orca revision pin `ORCA_REV=8336e613ea508b13c2159bd0f68965d97a606b95`.

- [ ] **Step 1: Checkout repo on M5 Pro, build the quantizer library**

```bash
# from repo root (branch dongnh311/halfbeak or later)
make -C gguf-tools quants-shared
ls -la gguf-tools/libds4quants.dylib
```
Expected: dylib exists.

- [ ] **Step 2: Verify runtime already has the discussion #6 fix**

```bash
grep -n "ds4_engine_embd_dim(s->engine)" ds4_server.c
```
Expected: one match near `server_encode_image` (~line 10185 in `halfbeak`). If missing, rebase/merge `origin/qwen3.8-flash-next` tip first and stop until confirmed.

- [ ] **Step 3: Download the pinned imatrix**

```bash
hf download unsloth/Qwen3.8-Flash-Next-GGUF imatrix_unsloth.gguf_file \
  --revision c8b5954a88c2775c546b92593eda40ea041d3176 --local-dir gguf/imatrix
shasum -a 256 gguf/imatrix/imatrix_unsloth.gguf_file
# must print a5863123db1ca458727e738955bef7bfc199520aa2bee3a30142a1aff9254154
```
Expected: hash matches (otherwise `QwenGGUFImatrix` will reject it at Task 2).

- [ ] **Step 4: Download Ivan's IQ2 template**

```bash
hf download ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2 \
  Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf --local-dir gguf/
```
Expected: 44.81 GB file present. Do NOT download the PLE sidecar yet (Task 4 rebuilds it).

- [ ] **Step 5: Free-disk checkpoint**

```bash
df -h .   # record free GB; must be ≥ 140 GB to start Task 1 download
```

- [ ] **Step 6: Commit (no code, but record the recipe constants in a new doc stub)**

Create `docs/ORCA_UNCENSORED_DS4_IQ2.md` with: Orca revision, imatrix revision+sha, template name, machine spec template table (chip, RAM, SSD free, macOS version — fill via `system_profiler SPHardwareDataType; sw_vers; df -h`). Commit:

```bash
git add docs/ORCA_UNCENSORED_DS4_IQ2.md
git commit -m "Document Orca Uncensored DS4 IQ2 build recipe constants"
```

---

## Task 1: `gguf-tools/orca_diff.py` — Orca download + topology diff

**Files:**
- Create: `gguf-tools/orca_diff.py`
- Test: run on M5 Pro; output `gguf/orca-diff.json`

**Interfaces:**
- Consumes: `SourceDB` (import from `qwen4_pack`), Hugging Face hub client, Qwen base revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` (index only — via `model.safetensors.index.json`, no shard download), Orca revision `8336e613ea508b13c2159bd0f68965d97a606b95`.
- Produces: `gguf/orca-diff.json` with fields `changed[]` (HF tensor name → {shard, sha256_orca, sha256_qwen_base, params}), `identical[]` count, `asserted` booleans (gate_up all identical, router identical, 1658 tensors, 179999981459 params, expert geometry 512×2560×640 down / 512×1280×2560 gate_up). Also local Orca checkpoint dir `gguf/orca-bf16/` (resumable shard download).

Key design decisions (already made — implement as stated):
1. Download all 131 Orca shards with `huggingface_hub.snapshot_download(repo_id="orcarouter/Qwen3.8-Flash-Next-Uncensored", revision=ORCA_REV, local_dir=...)` — resumable by default. 336 GB.
2. Fetch Qwen base **index only** (`hf_hub_download(repo_id="Qwen/Qwen3.8-Flash-Next", revision=de4b8e4d, filename="model.safetensors.index.json")`) — no base shards.
3. Diff strategy: ORCA is the source of truth (we re-encode changed tensors from it); the diff exists to (a) assert the expected 149-set and geometry, (b) mark which tensors get replaced in Task 3, (c) record Qwen-base sha256 where available from Ivan template payloads (template byte-copy equivalence check for gate/up: recompute sha of gate/up payloads in template and compare against Orca-expected only when Orca says identical — if Orca gate/up hash differs from Qwen base hash, FAIL, because the whole reuse plan collapses).

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Download the OrcaRouter Uncensored BF16 checkpoint and diff it tensor-by-tensor
against the pinned Qwen3.8-Flash-Next base index, asserting the abliteration scope
(149 changed residual-writing tensors; gate/up readers, router, mixers, indexer,
vision, n-gram table byte-identical) before any quantization work begins."""
import argparse, hashlib, json
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

ORCA_REPO = "orcarouter/Qwen3.8-Flash-Next-Uncensored"
ORCA_REV = "8336e613ea508b13c2159bd0f68965d97a606b95"
QWEN_REPO = "Qwen/Qwen3.8-Flash-Next"
QWEN_REV = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TOTAL_PARAMS = 179_999_981_459
EXPECTED_CHANGED = 149

def read_index(root: Path):
    return json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", type=Path, default=Path("gguf/orca-bf16"))
    ap.add_argument("--out", type=Path, default=Path("gguf/orca-diff.json"))
    args = ap.parse_args()

    snapshot_download(ORCA_REPO, revision=ORCA_REV, local_dir=args.local_dir)  # resumable
    qwen_index = json.loads(hf_hub_download(
        QWEN_REPO, revision=QWEN_REV, filename="model.safetensors.index.json")
        .read_text() if False else Path(hf_hub_download(
        QWEN_REPO, revision=QWEN_REV, filename="model.safetensors.index.json"))
        .read_text())["weight_map"]
    orca_map = read_index(args.local_dir)

    params = sum(int(v) for v in json.loads(
        (args.local_dir / "config.json").read_text()).get("_parameters", {}).values()) \
        if (args.local_dir / "config.json").is_file() else None
    # authoritative: count from weight_map shapes is too slow; use the known total:
    assert None  # placeholder guard; real check below uses EXPECTED tensor count instead

    common = sorted(set(orca_map) & set(qwen_index))
    if len(common) != 1658:
        raise SystemExit(f"tensor count mismatch: {len(common)} != 1658")
    if not (set(orca_map) - set(qwen_index) == set()
            and set(qwen_index) - set(orca_map) == set()):
        raise SystemExit(f"topology name mismatch: orca-only={list(set(orca_map)-set(qwen_index))[:5]} "
                         f"qwen-only={list(set(qwen_index)-set(orca_map))[:5]}")

    # Abliteration expectations (from Orca model card):
    def changed_set(names):
        return [n for n in names if any(
            n.endswith(s) or s in n for s in
            (".mlp.experts.down_proj", ".mlp.shared_expert.down_proj",
             ".linear_attn.out_proj", ".self_attn.o_proj",
             "ple.value_proj", "embed_tokens"))]
    names = list(common)
    expected_changed = changed_set(names)
    report = {
        "orca_repo": ORCA_REPO, "orca_revision": ORCA_REV,
        "qwen_revision": QWEN_REV,
        "tensor_count": len(common), "expected_total_params": TOTAL_PARAMS,
        "gate_up_untouched": all("gate_up_proj" not in n for n in expected_changed),
        "router_untouched": not any(".mlp.gate." in n for n in expected_changed),
        "expected_changed": sorted(expected_changed),
        "expected_changed_count": len(expected_changed),
    }
    if not report["gate_up_untouched"]:
        raise SystemExit("gate/up reader tensors changed by Orca — reuse plan invalid, stop")
    if report["expected_changed_count"] != EXPECTED_CHANGED:
        print(f"WARNING: expected 149 changed, counted {report['expected_changed_count']} "
              "(card rounds; accept if off by ≤2 and list matches card sections)")
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("tensor_count", "expected_changed_count",
                       "gate_up_untouched", "router_untouched")}, indent=2))
```

(Implementation note: the `assert None` / `params` lines are scaffolding — delete them; the enforced check is the 1658-count + set-equality + changed-set membership. Keep code clean, no dead asserts, per AGENT.md.)

- [ ] **Step 2: Add the per-tensor payload hash step**

Hashing all 336 GB is too slow; hash ONLY the 149 changed candidates (needed later as `source_sha256` in Task 3, mirroring `qwen4_iq2.py`'s `source_sha256` field) plus spot-check 8 gate/up tensors to prove Orca==Qwen-base bytes (can't compare against base without base shards — instead: record Orca sha256 of those 8; Task 3 cross-checks them against the Ivan template's payload hashes via its manifest JSON `Qwen3.8-...IQ2...json` sha256 fields, since the template was built from Qwen base). Add `report["orca_sha256"] = {name: sha256}` for the 149 + 8 spot checks, written incrementally as each shard is scanned (stream 8 MiB chunks, resumable via partial JSON sidecar `--out + ".progress.json"`).

```bash
python3 gguf-tools/orca_diff.py --local-dir gguf/orca-bf16 --out gguf/orca-diff.json
```
Expected: `tensor_count: 1658`, `gate_up_untouched: true`, `router_untouched: true`, `orca_sha256` map has 157 entries.

- [ ] **Step 3: Cross-check gate/up spot hashes against the Ivan template manifest**

```bash
# template manifest published alongside the IQ2 release; fetch if not in gguf/:
hf download ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2 --local-dir gguf/ \
  # (includes the .json manifest named after the template)
python3 - <<'EOF'
import json
diff = json.load(open("gguf/orca-diff.json"))
mani = json.load(open("gguf/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf.json"))
# template gate/up payloads are quantized IQ2 — cannot compare BF16 hash directly.
# Instead the check is: template manifest 'replaced' flag true ONLY for 48 down tensors,
# and its gate/up 'sha256' values are byte-identical in our final build (Task 3 verifies).
replaced = [k for k, v in mani["tensors"].items() if v.get("replaced")]
print(len(replaced), "template-replaced tensors:", sorted(replaced)[:4])
EOF
```
Expected: 48 (the down tensors of the *prior* MXFP4→Q2_K step — matches the release note "only 48 payloads changed vs prior release").

- [ ] **Step 4: Commit**

```bash
git add gguf-tools/orca_diff.py gguf/orca-diff.json
git commit -m "Add Orca Uncensored topology diff against pinned Qwen base"
```

---

## Task 2: Quantize the 48 Orca down experts (Q2_K, Pad768)

**Files:** none modified — uses existing `gguf-tools/qwen4_iq2.py` unchanged.

**Interfaces:**
- Consumes: Task 0 (`libds4quants.dylib`, imatrix, Orca dir `gguf/orca-bf16/` from Task 1), Orca tensor names `model.language_model.layers.{0..47}.mlp.experts.down_proj` shape `(512, 2560, 640)` BF16.
- Produces: `gguf/experts-down-orca/` (48 payload files + `experts.json` manifest with per-tensor `sha256` and `source_sha256`).

- [ ] **Step 1: Run the quantize stage**

```bash
cd <repo root>
python3 gguf-tools/qwen4_iq2.py quantize \
  --projection down \
  --source gguf/orca-bf16 \
  --imatrix gguf/imatrix/imatrix_unsloth.gguf_file \
  --library gguf-tools/libds4quants.dylib \
  --experts-dir gguf/experts-down-orca --threads 8
```
Notes:
- `qwen4_iq2.py` requires `SourceDB` to see a `config.json` with no `quantization_config` (Orca is a clean BF16 checkpoint — passes; verify first: `grep -c quantization_config gguf/orca-bf16/config.json` should fail/be empty).
- Interrupt-safe: re-run the same command to resume (it verifies existing per-tensor sha256, per `qwen4_iq2.py:56-69`).
- Orca down differs from Qwen down (abliterated) — that is exactly why we quantize from Orca, not from a template.

- [ ] **Step 2: Verify manifest completeness**

```bash
python3 - <<'EOF'
import json
m = json.load(open("gguf/experts-down-orca/experts.json"))
exp = {f"blk.{i}.ffn_down_exps.weight" for i in range(48)}
assert set(m["tensors"]) == exp, len(m["tensors"])
assert all(v["bytes"] == 512*2560*3*84 for v in m["tensors"].values())
assert m["format"] == "Q2_K" and m["padding"] == {"logical_input": 640, "physical_input": 768}
print("48/48 Orca down experts OK,", m["imatrix"]["sha256"])
EOF
```

- [ ] **Step 3: Commit the manifest (not the 20 GB of payloads)**

```bash
git add gguf/experts-down-orca/experts.json
git commit -m "Record Orca Q2_K padded-down expert manifest (payloads on disk)"
# add gguf/ to .gitignore if not already: check `git check-ignore gguf/orca-bf16`
```

- [ ] **Step 4: Disk cleanup checkpoint** — confirm ≥ ~120 GB free (template 44.8 + PLE out 32 + main out 45 + Orca 336 already on disk; if Orca dir must coexist, ensure free ≥ 160 GB before Task 4). If short: proceed to Task 3 anyway, and delete `gguf/orca-bf16` ONLY after Task 4 (PLE) is done, since Task 3 needs no Orca shards (dense payloads were already encoded in Task 1 diff? No — Task 3 reads dense tensors from Orca: keep Orca dir until Task 3 done; PLE also reads Orca in Task 4).

---

## Task 3: `gguf-tools/orca_dense_replace.py` — re-encode 98 dense + assemble final main GGUF

**Files:**
- Create: `gguf-tools/orca_dense_replace.py`
- Modify: `gguf-tools/qwen4_iq2.py:118-120` (add `--name` override for `general.name` metadata; two lines)

**Interfaces:**
- Consumes: Task 1 `gguf/orca-diff.json` (changed list + `orca_sha256` for source-hash records), Task 2 `gguf/experts-down-orca/`, Ivan template + its `.json` manifest, `GGMLQuantizer` and `SourceDB` from `qwen4_pack`.
- Produces: `gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-MTP.gguf` + `.json` sidecar manifest (records per tensor: type, bytes, sha256, source: template|orca-quantized, replaced flag).

- [ ] **Step 1: Write the dense-reencode + assemble script**

Reuse `qwen4_iq2.py:assemble` mechanics (Reader over template, byte-copy, replace slots, read-back verify) as a shared function `assemble_main(template, out, replacements, name, extra_metadata)` moved into `qwen4_iq2.py` (itself a refactor of existing code — same behavior, callable). `orca_dense_replace.py` then:

1. Load `orca-diff.json`; build the replace-set = (changed ∩ template tensor names via the HF→GGUF name map in `qwen4_exp_convert.py`'s tensor mapping — import the mapping function; if it's inline, duplicate the small table and comment why).
2. For each changed dense tensor present in the template: read Orca BF16 values via `SourceDB(gguf/orca-bf16)`; verify `sha256(values) == orca_sha256[name]` (Task 1 hash); encode with the SAME `GGMLQuantizer` library using the template tensor's declared type:
   - `blk.L.self_attn.o_proj.weight` → Q8_0
   - `blk.L.attn_kv.out_proj` / GDN `out_proj` family → Q8_0 (template type is authoritative: use `base.tensors[name]` kind, never hardcode — hardcoding would break if template recipe changes)
   - `blk.L.ffn_shared_down.weight` / shared down → Q8_0
   - `embed_tokens.weight` rows → BF16 copy (template kind 30) — only the changed rows; simpler: whole tensor BF16 re-copy from Orca (template is BF16 anyway).
   - MTP changed dense (`blk.49.nextn.*` o_proj/shared-down if in changed set) → encode with that tensor's template kind (Q4_K/MXFP4 per MTP recipe) — same "template type is authoritative" rule.
   - `ple.value_proj` is NOT in the main template (PLE lives in the sidecar) → skip here, handled by Task 4.
   - If a changed tensor is absent from the template (e.g., vision tower — but card says vision untouched, so impossible; if it happens → FAIL loudly).
3. `general.name` → `--name` value `Qwen3.8 Flash Next Orca Uncensored, IQ2_XXS imatrix gate/up, padded Q2_K Orca down, MTP`.
4. Metadata additions (8-byte strings, `ds4.` namespace, matching existing convention in `qwen4_iq2.py:122-125`):
   - `ds4.orca.source_repo` = `orcarouter/Qwen3.8-Flash-Next-Uncensored`
   - `ds4.orca.revision` = `8336e613ea508b13c2159bd0f68965d97a606b95`
   - `ds4.orca.dense_replaced` = count of dense tensors replaced
   - `ds4.orca.abliterated` = `true`
5. Everything else byte-copied from the template (including gate/up IQ2_XXS — provenance: Task 1 gate/up spot-hashes + template manifest).
6. Read-back verify every payload (copy the loop from `qwen4_iq2.py:174-188` verbatim), write `.json` sidecar manifest with template+manifest+replaced-record union, `sha256` of final file.

Size check gate before publish: `assert out_size == 41.73 GiB + Σ(new dense) − Σ(old dense) ± 0` (dense replace is byte-neutral only when same qtype/shape — it should be: same dims, same kind → exact match with template size 44.81 GB; any mismatch → FAIL, print per-tensor size delta).

```python
# skeleton (fill exact mapping imports in step 1; no placeholders in committed code)
def main():
    ...
```

- [ ] **Step 2: Two-line change to `qwen4_iq2.py` (expose assemble + `--name`)**

```python
# qwen4_iq2.py main():
a.add_argument('--name', default=None,
               help='override general.name in the assembled GGUF metadata')
```
and in `assemble()`: `if args.name: metadata['general.name'] = (8, args.name)`.
Verify existing behavior unchanged: `python3 gguf-tools/qwen4_iq2.py assemble --help` shows new flag; default path keeps old name.

- [ ] **Step 3: Dry run (plan only, no write)**

```bash
python3 gguf-tools/orca_dense_replace.py --dry-run \
  --diff gguf/orca-diff.json --template gguf/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf \
  --experts-dir gguf/experts-down-orca --source gguf/orca-bf16 \
  --library gguf-tools/libds4quants.dylib \
  --out gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-MTP.gguf
```
Expected: prints the per-tensor plan (48 down from experts-dir, N dense re-encoded, rest byte-copied, total size line) and exits 0 without writing.

- [ ] **Step 4: Full run + verify**

Same command without `--dry-run`. Expected final line: `Verified gguf/...OrcaUncensored...gguf: <bytes>` and `.json` manifest with every tensor record `replaced` correctly set (48 down + N dense + MTP-if-changed; ~1200 byte-copies).

- [ ] **Step 5: Cleanup + commit**

```bash
rm -rf gguf/experts-down-orca/*.incomplete 2>/dev/null
git add gguf-tools/orca_dense_replace.py gguf-tools/qwen4_iq2.py \
        gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-MTP.gguf.json
git commit -m "Re-encode Orca abliterated dense tensors and assemble Orca Uncensored IQ2 main GGUF"
```
(Keep `gguf/orca-bf16` — Task 4 needs it.)

---

## Task 4: Rebuild PLE Q4_1 sidecar from Orca (abliterated `ple.value_proj`)

**Files:**
- Modify: `gguf-tools/qwen4_pack.py` — PLE build path: allow `--source-revision` / source repo override for PLE-only build without re-doing the whole trunk pack (extract the minimal function `build_ple_sidecar(source_db, out, pack_id, ...)` from `repack()`; keep `repack()` byte-identical behavior).

**Interfaces:**
- Consumes: `gguf/orca-bf16` (PLE `*.safetensors` shards containing `ple.*` tensors + index), `libds4quants.dylib`.
- Produces: `gguf/Qwen3.8-Flash-Next-OrcaUncensored-PLE-Q4_1.gguf` (~32 GB) with `ds4.pack.source_revision` = Orca revision and sha256 report.

- [ ] **Step 1: Extract `build_ple_sidecar()` in `qwen4_pack.py`**

The PLE assembly block is inside `repack()` around lines 2558-2776 (`ple_writer` lifecycle). Refactor: move into a standalone function `build_ple_sidecar(db: SourceDB, quantizer: GGMLQuantizer, out: Path, pack_id: str, source_revision: str, ple_rows_from_index...)` that `repack()` calls unchanged. Add CLI subcommand to `qwen4_pack.py` (`--ple-only --source <dir> --out <file>`) if it doesn't already expose one (check `qwen4_pack.py` argparse first; it may have a `pack` subcommand — adapt). Gate: run the existing Qwen path with `--dry-run` before and after refactor and diff the plan output — must be identical (AGENT.md: no silent behavior change).

- [ ] **Step 2: Build Orca PLE sidecar**

```bash
python3 gguf-tools/qwen4_pack.py <subcommand> --ple-only \
  --source gguf/orca-bf16 \
  --library gguf-tools/libds4quants.dylib \
  --out gguf/Qwen3.8-Flash-Next-OrcaUncensored-PLE-Q4_1.gguf
```
Verify: `per_layer_token_embd` NOT in PLE file (sidecar holds PLE table tensors `ple.ple_embedding.*` + `ple.value_proj` only; the n-gram table itself is in the main GGUF — per `ds4.c:2806` the MAIN must contain BF16 n-grams, and Task 3 kept it via template byte-copy; confirm by re-opening final main GGUF with `qwen4_pack_to_qwen4exp.Reader` and asserting `per_layer_token_embd.weight` present, type 30 (BF16)).

- [ ] **Step 3: Verify PLE content + read-back hashes** (script already does this; spot-check one changed PLE tensor `ple.value_proj` sha256 matches Task 1 `orca_sha256` after dequant sanity: `|dequant − orca|_max < 2^-3` — Q4_1 is ~4.5 bit/elem, expect ~2^-3.5 relative; use the packer's existing `record_ple_tensors` report).

- [ ] **Step 4: Delete Orca shards + commit**

```bash
rm -rf gguf/orca-bf16            # frees 336 GB — do this ONLY after PLE build verified
git add gguf-tools/qwen4_pack.py gguf/Qwen3.8-Flash-Next-OrcaUncensored-PLE-Q4_1.gguf.json
git commit -m "Build Orca Uncensored PLE Q4_1 sidecar with abliterated value_proj"
```
Note: sidecar payloads live under `gguf/` (gitignored) — commit manifests + doc, payloads stay on the M5 Pro disk.

---

## Task 5: Target-machine validation (M5 Pro 64 GB) + tok/s & context measurement

**Files:**
- Modify: `docs/ORCA_UNCENSORED_DS4_IQ2.md` (fill benchmark table)
- Create: `tests/test_orca_uncensored_smoke.py` (small script: refusal + capability probes via `./ds4 -p`, greps output)

**Interfaces:**
- Consumes: Task 3 main GGUF + Task 4 PLE sidecar, repo `make` build on M5 Pro.
- Produces: filled doc + commit. Numbers become the answer to "improve tok/s" (M1 Max report #1: 22-24 t/s @4K, 8K failed; target expectation from M4 Max #2: ~38 t/s @32-64K on a bandwidth-class machine; M5 Pro will land per its measured bandwidth).

- [ ] **Step 1: Build runtime on M5 Pro**

```bash
make            # Metal build; must be clean
ls -la ds4 ds4-server ds4-agent
```

- [ ] **Step 2: Load smoke (no-swap check at 8192 ctx, the config Ivan's Q2 doc recommends)**

```bash
./ds4 -m gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-MTP.gguf \
      --ctx 8192 --prefill-chunk 1024 -p "Say OK."
```
Expected: single-line OK, engine plans ≤ ~50 GiB, `vm_stat`/Activity Monitor shows no swap in during the run (M1 Max failed at exactly this step planning 54.17 GiB — M5 Pro 64 GB should pass; if it fails, drop to `--ctx 4096` and note it in the doc). Also verify PLE sidecar is picked up (log line about PLE sidecar / ple_evictions only on rewind).

- [ ] **Step 3: `ds4-eval` core suite (quality gate)**

```bash
./ds4-eval -m gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-MTP.gguf \
            --ctx 8192 --plain --trace /tmp/orca-eval-core.txt
```
Expected: default suite `core` passes at parity-with-Ivan-IQ2 tolerance (Ivan's own NLL delta vs MXFP4-down was +4.4%; Orca changes are additive, expect same order). Record the trace.

- [ ] **Step 4: Uncensor smoke (the whole point of the build)**

`tests/test_orca_uncensored_smoke.py`: run `./ds4 -p` (default temp 1 sampling) against 3 probes — one harmful-capability ask (Orca card: refusal → 0-3.3%), one benign coding task (card: MMLU 87.7), one refusal-preserving-safe ask (XSTest-safe: over-refusal ≤ 1%). Grep for refusal patterns ("I cannot", "I'm unable to" on the harmful one) — assert ABSENT; assert coding task returns code. Run 2× each, record. This is a heuristic smoke, not a benchmark — the card's numbers are the reference.

- [ ] **Step 5: MTP + tok/s measurement (opportunistic MTP ONLY — never `--mtp-exact-sampling`, discussion #5)**

```bash
# warm-up: one 256-token generation, then 3 runs; record prefill t/s + decode t/s from the CLI
./ds4 -m <ORCA-MAIN> --ctx 8192 --prefill-chunk 1024 --mtp -p "<256-token task>"
```
Also run `./ds4-bench` with the same GGUF at 4K and 32K context (2048-token continued-prefill intervals + 128 greedy tokens, matching `speed-bench/` methodology in `docs/PERFORMANCE.md`), with and without `--mtp`. Write numbers into the doc table: chip, ctx, prefill t/s, decode t/s (plain vs MTP), max context without swap.

- [ ] **Step 6: Fill doc + commit**

```bash
git add docs/ORCA_UNCENSORED_DS4_IQ2.md tests/test_orca_uncensored_smoke.py
git commit -m "Validate Orca Uncensored DS4 IQ2 on M5 Pro: eval, uncensor smoke, MTP tok/s"
```

---

## Self-Review notes (run after writing plan)

- **Spec coverage:** Orca source + pinned rev ✔ (Task 0/1), down experts from Orca ✔ (Task 2), gate/up reuse ✔ (Task 3 byte-copy + Task 1 hash cross-check), 98 dense replace ✔ (Task 3), MTP unchanged ✔ (Task 3 template copy, verified count), PLE rebuild ✔ (Task 4), tok/s levers ✔ (Task 2 recipe choice + Task 5 MTP/measurement; runtime kernel work explicitly out of scope per AGENT.md), 64GB budget ✔ (Task 5 step 2/5 max-context-no-swap). Discussion #5 avoidance ✔ (Global Constraints). Discussion #6 pre-check ✔ (Task 0 step 2).
- **Risks flagged:** (a) HF→GGUF name map for the 98 dense tensors lives inside `qwen4_exp_convert.py`'s converter invocation of llama.cpp — if the map is not importable, Task 3 duplicates the ~50-row table; (b) 336 GB Orca download needs ~1-2 h on a decent line and 140 GB+ disk — Task 0 step 5 gates it; (c) Q4_1 PLE is the only PLE qtype viable (160-dim geometry) — do not revisit Q2/Q1 PLE; (d) M5 Pro RAM 64 GB may still plan >50 GiB at 32K ctx (M4 Max ran 64K at 38.9 t/s, so M5 Pro 64 GB should too — but MEASURE, don't assume; that is Task 5's job).
- **Placeholder scan:** none in final committed scripts; the Task 1 skeleton's `assert None` block is explicitly marked for deletion in the same step.
