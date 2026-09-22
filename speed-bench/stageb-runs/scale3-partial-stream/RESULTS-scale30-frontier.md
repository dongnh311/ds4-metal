# SCALE-3.0: RAM/tok-s frontier vs number of streamed layers (2026-09-22)

`DS4_QWEN4_STREAM_FULL_LAYERS=K` keeps the first K routed layers resident under
`--ssd-streaming` and streams the rest. Every streamed layer pays one host sync
per token whatever the hit rate, so the number of streamed layers, not the cache
size, is the main RAM/speed knob.

Method (same as SCALE-2c): M5 Pro 64 GB, ctx 8192, 600 tokens, greedy, seed 1,
no MTP, prompt `longgen_2c.txt`. RAM = steady system-wide wired pages from
`vm_stat` during decode (includes ~3.3 GiB machine baseline).

## PROD (`...DenseQ4Kselimat-Q2KDownPad768-MTP`, 40.7 GiB, 48 streamable layers)

| streamed | cache | wired GiB | tok/s | vs resident |
|---:|---:|---:|---:|---:|
| 0 | - | 49.17 | 35.82 | 100% |
| 6 | 2GB | 45.45 | 32.10 | 90% |
| 12 | 4GB | 43.19 | 31.39 | 88% |
| 12 | 2GB | 41.19 | 27.32 | 76% (dominated) |
| 24 | 8GB | 38.66 | 28.39 | 79% |
| 24 | 4GB | 34.65 | 24.99 | 70% |
| 48 | 16GB | 29.61 | 21.65 | 60% |
| 48 | 8GB | 21.62 | 18.82 | 53% |

## unc31 (`...SelQ4Down18-48-DenseQ4Kselimat-MTP`, 47.7 GiB)

Layers 0-17 (Q2_K down) are off the slab class and already resident, so only
the 30 Q4_K layers 18-47 can stream: K=0 streams 30, K=24 streams 24, K=36
streams 12, K=42 streams 6.

| streamed | cache | wired GiB | tok/s | vs resident |
|---:|---:|---:|---:|---:|
| 0 | - | 56.16 | 33.86 | 100% |
| 6 | 3GB | 51.71 | 30.53 | 90% |
| 6 | 2GB | 50.82 | 27.05 | 80% (dominated) |
| 12 | 6GB | 49.15 | 29.36 | 87% |
| 12 | 4GB | 47.14 | 28.21 | 83% |
| 24 | 12GB | 44.04 | 26.45 | 78% |
| 24 | 8GB | 40.01 | 25.74 | 76% |
| 30 | 16GB | 42.50 | 25.26 | 75% (dominated by 24/8GB) |
| 30 | 8GB | 34.42 | 21.83 | 64% |

The all-streamed points of both models reproduce SCALE-2c (PROD 21.58/19.4 and
29.59/21.4; unc31 34.42/22.0 and 42.50/24.8), and both resident baselines match.

## Reading

- Cost model, fitted on both models with the cache near half of the streamed
  experts: about 0.3 ms per token per streamed layer plus about 0.6-0.8 ms fixed
  once streaming is on. Tighter caches add SSD misses (0.5-0.7 ms per layer).
- No sharp knee. 90% of resident speed saves 3.7 GiB (PROD) or 4.5 GiB (unc31);
  the middle of the curve loses about 0.65 tok/s per GiB saved.
- At equal RAM, more streamed layers with a larger cache beat fewer layers with a
  small cache (PROD 24/8GB over 12/2GB; unc31 24/8GB over 30/16GB).
- unc31 with 12 streamed layers and a 6GB cache wires 49.15 GiB, the same as PROD
  resident (49.17), at 29.36 tok/s (87% of unc31 resident).
- The drain share printed by `DS4_QWEN4_STREAM_TIMING` is not a waste measure: it
  includes waiting for the GPU work of every resident layer encoded before the
  sync. Compare tok/s against resident instead.

## Output

Every streaming run of a model is byte-identical to every other (PROD
`0fbb2a7a...`, unc31 `8179896f...`), and PROD matches exp-pld c9d494c streaming,
so the knob does not change output. Streaming output differs from resident
output on this prompt for both models, in the parent as well; that divergence
predates this change. A `1GB` cache fails at prefill in the parent too.

## unc31, 12 streamed layers @ 6GB, with MTP: 8K vs 124K real context

`--mtp --mtp-timing`, greedy, `DS4_QWEN4_PLE_PREFETCH_FULL=0`. 124K rows use
`DS4_QWEN4_KV_FP8=1 --prefill-chunk 2048 --ctx 131072` and the `needle.txt`
prompt (124,141 tokens, code at the start, question at the end).

| ctx | mode | peak wired | decode wired | prefill t/s | gen t/s | MTP accept | needle |
|---:|---|---:|---:|---:|---:|---:|---|
| 8K | resident | 56.40 | 56.29 | (short prompt) | 38.20 | 70.9% | - |
| 8K | 12 @ 6GB | 49.40 | 49.18 | (short prompt) | 34.46 (90%) | 73.8% | - |
| 124K | resident | 55.62 | 54.94 | 587.6 | 34.20 | 82.9% | HIT |
| 124K | 12 @ 6GB | 48.41 | 45.56 | 106.2 (18%) | 23.88 (70%) | 85.0% | HIT |

- MTP does not hurt streaming at 8K (90% with MTP vs 87% without).
- At 124K the two runs produce byte-identical output; both answer the needle.
- Decode keeps only 70% at 124K (overhead ~12.7 ms/token vs ~2.8 ms at 8K).
  Not isolated; this run did not set `DS4_QWEN4_STREAM_TIMING`.
- Prefill is the blocker: 124K takes ~1169 s streamed vs ~211 s resident.
  The extra ~958 s over ~61 chunks x 12 layers is ~1.3 s per layer per chunk to
  load roughly the whole layer (~950 MiB, a 2048-token chunk touches nearly all
  512 experts), i.e. ~0.7 GB/s effective, far below sequential SSD bandwidth.
  That points at per-expert reads, not the SSD. Each layer's gate/up/down
  experts are contiguous tensors, so a chunk whose union covers most of a layer
  could read it in bulk.
- Resident unc31 already fits at 124K with FP8 + chunk 2048 (peak 55.62 GiB), so
  streaming is not needed for fit at this context; it buys ~7 GiB of headroom.

## Prefill fix (b704550): GEMMs stay on non-streamed layers

Root cause of the slow streaming prefill and of the streaming/resident output
difference: `qwen4_graph_moe` disabled the prefill expert GEMMs (T > 64) on every
layer whenever `--ssd-streaming` was on, so resident and pinned layers also ran
the per-token row kernels. Streaming with zero streamed layers (K=48) was 116
t/s against 555 resident on a 16K prompt, and chunk 4096 was no faster than
2048, so the cost was per token, not per chunk or per read. The gate is now per
layer, using the span planner's predicate.

unc31, 124K needle (124,141 tokens), FP8 KV, MTP, 12 streamed layers @ 6GB:

| build | chunk | prefill t/s | 124K prefill time | gen t/s | peak GiB | output |
|---|---:|---:|---:|---:|---:|---|
| resident | 2048 | 587.6 | ~211 s | 34.20 | 55.62 | reference |
| streaming, before fix | 2048 | 106-108 | ~1160 s | 23.9-25.7 | 48.4 | == resident |
| streaming, after fix | 2048 | 230.4 | ~540 s | 26.10 | 48.36 | == resident |
| streaming, after fix | 4096 | 229.0 | ~542 s | 24.58 | 49.83 | == resident |

Prefill goes from 18% to 39% of resident. Chunk 4096 buys nothing and costs
1.5 GiB. The remaining gap is the 12 streamed layers, which still use the row
kernels in prefill because no GEMM reads through the expert address table yet.

16K prompt, same model: 0 streamed 522 t/s, 6 streamed 303, 12 streamed 233
(resident 555); every output matched resident.

## Expert staging (SCALE-3.2): GEMM prefill for streamed layers, and a working fallback

`ds4_gpu_qwen4_stream_stage_layer` loads the experts one dispatch needs into a
buffer laid out like the layer's gate/up/down tensors, and `qwen4_bind_weight`
redirects binds of exactly those tensors to it while it is active. The unchanged
GEMM and row kernels then run on streamed layers. Staged prefill is on by
default (`DS4_QWEN4_STREAM_STAGE_PREFILL=0` disables it); the fallback always
stages when a cache load fails for a streamed layer.

unc31, 124K needle, FP8 KV, MTP, 12 streamed layers @ 6GB, chunk 2048:

| build | prefill t/s | 124K prefill time | gen t/s | peak GiB | output |
|---|---:|---:|---:|---:|---|
| resident | 587.6 | ~211 s | 34.20 | 55.62 | reference |
| streaming, original | 106.2 | ~1170 s | 23.88 | 48.41 | == resident |
| per-layer GEMM gate (b704550) | 230.4 | ~540 s | 26.10 | 48.36 | == resident |
| + expert staging | 541.3 | ~230 s | 23.69 | 48.18 | == resident |

16K prompt: 12 streamed 455 t/s (82% of resident 555), 6 streamed 474 t/s;
staging off 214 t/s. All outputs matched resident. The staging buffer is 0.93
GiB (Q4_K layer) or 0.71 GiB (PROD Q2_K layer); measured peak rose only ~0.24
GiB at 16K and not at all at 124K, since prefill no longer fills cache slots.

Fallback: the two configs that used to fail at prefill with "not covered by
mapped model views" now run and match resident output: PROD with a `1GB`
cache (budget reads as 1 slot; gen 31.4 t/s) and unc31 K=47 with 6GB (mlock
fails; peak 57.4 GiB, a mis-sized cache for one streamed layer).

Decode at 124K stays at ~70% of resident (23.7 vs 34.2); it is untouched by
these changes and its cause is still not isolated.

## Correction: long-context decode is ~88% of resident, not ~70%

The 124K "decode ~70%" figures above came from the needle prompt, whose answer
ends at EOS after about 75 tokens (41 verify cycles + 34 accepted drafts), and
they started from a cold expert cache: with staging, prefill no longer fills
the cache. Re-measured with the same 124K context but a closing instruction to
count to 400, so both runs decode ~595 tokens (unc31, FP8, MTP, chunk 2048):

| mode | gen t/s | vs resident | prefill t/s | MTP accept |
|---|---:|---:|---:|---:|
| resident | 35.08 | 100% | 588.3 | 75.5% |
| 12 streamed @ 6GB | 30.73 | 88% | 538.2 | 75.5% |

Outputs are byte-identical. The streaming cost is 4.0 ms/token for 12 layers,
0.34 ms per streamed layer, the same per-layer cost measured at 8K, so long
context adds no decode cost of its own.

Cache warmup after a long prefill: the first ~250 decode calls with misses
average 6.3 missed experts per call, settling to ~1.7. The first 480 calls
spend 531 ms loading against ~90 ms per 480 calls afterwards, and drain adds
~0.45 s, so warmup costs ~0.5-0.9 s once per prompt: 3-4% of a 600-token reply,
20-30% of a 75-token one, which is what the earlier figure measured.

## Decode cache seeding (opt-in, off by default)

With staging, prefill no longer fills the expert cache, so decode starts cold.
`DS4_QWEN4_STREAM_SEED_TOKENS=N` copies, on the prompt's last chunk, the experts
its final N rows chose from the staging buffer into the cache. unc31, 124K
needle (75-token reply), 12 streamed @ 6GB, FP8 + MTP:

| seed | seeded experts | seed cost | decode misses | decode load | gen t/s | decode time |
|---:|---:|---:|---:|---:|---:|---:|
| off | 0 | 0 ms | 2228 | 435 ms | 24.95 | 3.01 s |
| 32 | 1403 | 212 ms | 1104 | 311 ms | 24.57 | 3.05 s |
| 64 | 2159 | 299 ms | 911 | 211 ms | 27.14 | 2.76 s |

Outputs match resident in all three. Seeding 64 tokens cuts early misses by 59%
and lifts short-reply gen t/s by 9%, but its copy runs on the critical path at
the end of prefill: 0.30 s paid against 0.24 s of decode saved, so the
end-to-end latency does not improve. The streamed layers are the last ones, so
little work is left to overlap the copy with. It stays available but off.

## Main config RAM (unc31, streamed layers, 6GB cache, FP8 KV, chunk 2048, MTP)

System-wide wired RAM from `vm_stat` (includes ~3.2 GiB machine baseline). KV
pages wire as the context fills, so the full-context rows use real prompts
(124K / 217K / 248K tokens) with ~440-600 decoded tokens.

| ctx | streamed | ctx nearly empty | ctx full (decode) | expert cache | gen t/s | prefill t/s |
|---:|---:|---:|---:|---|---:|---:|
| 8K | 12 | 45.5 | - | locked | 32.9 | - |
| 32K | 12 | 46.1 | - | locked | 33.1 | - |
| 128K | 12 | 47.8 | 48.2 peak | locked | 30.7 | 538 |
| 224K | 12 | 50.0 | 50.63 | locked | 32.55 | 384 |
| 256K | 12 | 50.7 | 51.2 (peak 52.2) | capped to 120 experts | 26.70 | 386 |
| 256K | 16 | - | 47.63 | locked | 30.60 | 334 |

The machine's `vm.user_wire_limit` is 52.48 GiB. At a full 256K context with
12 streamed layers, KV growth during prefill leaves no room to mlock the expert
cache ("using locked cache cap: 120 experts / 0.22 GiB"), so decode falls to
26.7 t/s. Streaming 16 layers (`DS4_QWEN4_STREAM_FULL_LAYERS=32`) frees ~3.6
GiB, the cache locks fully, and decode returns to 30.6 t/s; prefill drops from
386 to 334 t/s. BF16 KV at 256K fails to lock the cache even with an empty
context, so FP8 KV is required there. All full-context runs found the needle.

## Long-context prefill: resident baseline at 248K

unc31, the same 248,161-token prompt, FP8 KV, chunk 2048, MTP, ctx 262144:

| mode | prefill t/s | gen t/s | peak wired | needle |
|---|---:|---:|---:|---|
| resident | 562.6 | 27.78 | 57.87 GiB | HIT |
| 16 streamed (K=32) @ 6GB, rerun | 446.1 | - | - | - |
| 16 streamed (K=32) @ 6GB, earlier run | 334.4 | 30.60 | 47.63 GiB | HIT |

Resident holds 562 t/s at 248K against 588 at 124K, so attention is not what
slows long-context prefill. The streamed rerun with the staging profile gives
446 t/s (79% of resident) against 334 t/s in the earlier run with the same
config: the earlier number came from a worse machine memory state and does not
reproduce. Staging stayed in the page cache for the whole rerun (27-45 GB/s,
disk reads 50-160 MB/s) and took 44 s of 554 s. Resident decode at a full 256K
(27.78 t/s) is slower than 16 streamed layers (30.60 t/s): at 57.9 GiB peak the
resident run is under memory pressure.

## Where the per-layer decode cost goes

Per-command-buffer timings (`DS4_METAL_CB_TIMES=1`, printed at µs precision),
unc31, 12 streamed layers @ 6GB, 8K, no MTP. Each streamed layer is one command
buffer: its GPU span is ~0.55 ms, like a resident layer, so the address-table
kernels cost nothing extra. Around it, per layer:

| part | ms |
|---|---:|
| commit -> done minus GPU span (submit + completion wake-up) | ~0.18 |
| host gap between buffers (read `selected`, cache lookup, loads) | ~0.15 median, 0.30 mean |
| encode of the segment | ~0.04 |

Resident encodes a whole token in 0.64 ms (~14 µs per layer) while the GPU
runs, so encode is not the cost. Two host-side fixes:

- Submit the expert dispatches right after encoding them
  (`DS4_QWEN4_STREAM_FLUSH`, now on by default; `=0` disables), so the GPU runs
  them while the host encodes up to the next router.
- The eviction scans (`take_reusable`, `take_reusable_batch`, `prune_global`)
  walked all 80 x 512 entries (~6 MB) per miss once the cache is full; they now
  skip layers with no entries. Victim choice is unchanged.

Decode A/B, 8K, 600 tokens, each variant run twice in ABCD/DCBA order; output
is byte-identical across variants within each mode:

| variant | gen t/s, MTP | gen t/s, no MTP |
|---|---:|---:|
| before | 34.15 | 27.90 |
| flush | 35.13 | 28.37 |
| scan skip | 34.73 | 28.13 |
| both | 35.31 (+3.4%) | 28.74 (+3.0%) |

Resident with MTP at 8K is 38.20, so the main config is now at ~92%. What is
left is mostly the submit/wake-up latency of the per-layer drain.

## SCALE-3A: decode gates instead of the per-layer drain (opt-in)

`DS4_QWEN4_STREAM_GATE=1` keeps the whole token encoded ahead. At each streamed
layer a publish kernel copies the selected ids to a mailbox (each word tagged
with the gate's sequence number) and the batch is committed; the next batch
opens with `kernel_dsv4_tp_poll_release` (the TP poll kernel) spinning on a
fresh shared-memory region. A service thread reads the mailbox as soon as the
lines reach memory, loads the cache, writes the layer's address table and
releases the poll. No command buffer is waited on inside the token. Gates
start once the cache has all its slabs; experts the cache cannot hold go to a
gate-owned 120 MiB fallback buffer (never used in these runs).

A/B, unc31, 12 streamed @ 6GB, FP8, 8K, 600 tokens, ABCD/DCBA, output
byte-identical in every run:

| path | gen t/s, MTP | gen t/s, no MTP |
|---|---:|---:|
| drain (after the flush + scan fixes) | 35.21 | 28.40 |
| gate | 36.33 (+3.2%) | 29.86 (+5.1%) |

With MTP the main config goes 34.15 -> 36.33 over today's three changes (95%
of resident 38.20). The gate's service time is now the critical path: 150-200
µs per gate, of which the cache resolve is 100-150 µs and is dominated by
reading missed experts (the service thread waits on the pread pool; `sample`
puts 86% of resolve there), and the release stores ~54 µs (the GPU sees the
release within the first ~10 µs of them). Splitting the miss reads into 128
KB chunks across the pool made it slower (resolve 99 -> 122 µs) and was
dropped.

## MTP knobs on the main config (gates on, 8K, 500 tokens, two runs each)

Draft depth (`DS4_QWEN4_MTP_DEPTH`; 2 = one draft, 3 = two drafts, 0 = the
adaptive default):

| prompt | one draft | two drafts | adaptive |
|---|---:|---:|---:|
| English prose | 34.4 | 31.8 | 35.4 |
| code | 36.2 | 31.0 | 36.6 |

Two drafts lose on the streamed config: a three-row verify reaches more
distinct experts. The adaptive default stays.

Draft vocabulary prefix (`DS4_QWEN4_MTP_DRAFT_ROWS`, the draft scores only the
first N output rows):

| prompt | full | 131072 | 65536 | 32768 |
|---|---:|---:|---:|---:|
| English prose | 35.1 | - | 35.8 | 36.4 |
| code | 36.1 | - | 37.6 | 36.0 |
| Vietnamese | 34.6 (71.0% accept) | 29.1 (38.2%) | 29.0 (36.3%) | - |

Vietnamese tokens sit at high ids, so any prefix collapses its acceptance. Do
not use the prefix knob for multilingual serving; a frequency-built
`DS4_QWEN4_MTP_DRAFT_VOCAB` list covering the served languages is the only
form of this idea that could pay.

## Raw indexer key ring (bit-exact, on by default)

The QSA raw indexer keys (`layer_ik_cache`, float32, 128 per token per full
attention layer) are read only to pool a block's key when its last token
arrives, and a forward of T tokens reads back at most ratio - 1 = 3 keys before
its first position. They now live in a ring of `cap_tokens + 2 * ratio` rows
(rounded to 64) indexed by `pos % ring` instead of `ctx` rows; session payloads
keep their layout (rows older than the ring go out as zeros and are skipped on
load, complete blocks travel as block keys). `DS4_QWEN4_IK_RING=0` keeps every
row. Saves 12 x 512 B per token of context: 0.19 GiB at 32K, 0.75 GiB at 128K,
1.5 GiB at 256K.

| ctx | output vs full cache | KV estimate (raw + compressed) | decode-phase wired |
|---|---|---|---|
| 32K (16K prompt) | identical sha | 1.04 -> 0.85 GiB | - |
| 128K (124K prompt) | identical sha | 4.12 -> 3.37 GiB | 46.96 -> 46.17 GiB |

## FR-Spec draft vocabulary for VI + EN + code

`gguf-tools/frspec_vocab.py` ranks tokens by frequency over weighted corpora
(Vietnamese and English Wikipedia, local code, Python) and writes an id list for
`DS4_QWEN4_MTP_DRAFT_VOCAB`. The 64K list covers 100% / 99.4% / 97.0% of our
model's own Vietnamese / English / code outputs. Gates on, 8K, 500 tokens, two
runs each (gen t/s, first-draft acceptance):

| prompt | full head | 49K list | 64K list |
|---|---:|---:|---:|
| Vietnamese | 35.5 (71.0%) | 36.9 (70.4%) | 36.7 (71.0%) |
| English prose | 35.3 (68.3%) | 36.8 (67.7%) | 36.9 (68.3%) |
| code | 36.1 (72.7%) | 36.5 (67.1%) | 37.5 (70.3%) |

The 64K list is +3.4% to +4.6% on all three without the Vietnamese collapse of
the id prefix, for ~178 MiB of gathered head rows.

## KV low-bit research knobs

`DS4_QWEN4_KV_SIMQ=3|4` (append `r` for a 64-point Hadamard rotation per block)
fake-quantizes K/V to that many bits before the FP8 store, and
`DS4_PPL_PREFIX=N` makes `--perplexity-file` prefill N tokens before scoring, to
measure long-context KV quality before building a packed low-bit cache.

## Gates on by default: long context and server validation

unc31, FP8, 6GB cache. Gate vs drain (flush on):

| ctx | gen t/s drain | gen t/s gate | output | notes |
|---|---:|---:|---|---|
| 128K (124K prompt, K=36) | 27.07 | 32.27 (+19%) | identical sha | needle HIT, +0.19 GiB peak |
| 256K (248K prompt, K=32), decode from one restored checkpoint | - | - | identical sha across drain no-flush, drain flush and gate | correct count 1..82 in all three |

The 256K comparison decodes from a single `ds4-server` disk checkpoint (one
prefill, then a restart per path), so every path starts from the same state;
the restore reproduces the prefilling run byte for byte (247,808 tokens loaded
in 516 ms). Server session test (`ds4-server`, ctx 65536, disk KV cache):
Vietnamese two-turn chat, code, two concurrent requests and a 16K-token prompt
give byte-identical replies with gates off and on; after a restart the 16K
prompt restores 14,336 tokens from its checkpoint (13.1 s vs 51.5 s) with the
same reply. `DS4_QWEN4_STREAM_GATE=0` keeps the per-layer drain.

Harness note: the runtime compiles `./metal` at startup. Several 256K CLI runs
this afternoon used a binary built before the `attn_prep` argument struct grew
(`ik_ring`, `kv_simq`), so their kernels read past the arguments; those runs
(nondeterministic first tokens, a broken count) are void. Runs now use a
snapshot of the binary and `metal/` taken together.

## KV low-bit research: long-context quality

Teacher-forced NLL of 800 tokens scored after a 64,237-token prefix of real
Wikipedia text (`DS4_PPL_PREFIX`), unc31 main config. "fresh" continues with a
new article; "repeat" re-states the document's first article (long-range
retrieval). `simq` fake-quantizes K/V per 64-block (absmax scale) before the FP8
store; `r` adds a 64-point Hadamard rotation.

| KV | EN fresh | repeat | VI fresh |
|---|---:|---:|---:|
| BF16 | 1.0777 | 0.0173 | - |
| FP8 (deployed) | 1.0882 | 0.0163 | 1.9258 |
| 4-bit | 1.0993 (+1.1% ppl vs FP8) | 0.0181 | - |
| 4-bit, rotated | 1.1036 (+1.6%) | 0.0194 | 1.9318 (+0.6%) |
| 3-bit | 1.1278 (+4.0%) | 0.0218 | - |
| 3-bit, rotated | 1.1309 (+4.4%) | 0.0218 | 1.9420 (+1.6%) |

Long-range retrieval survives every format (repeat NLL stays ~0.02). Fresh-text
prediction pays: 4-bit costs about what FP8 already costs over BF16 (+1%), 3-bit
four times that. The Hadamard rotation does not help at this block size. A
packed 4-bit cache would cut the raw K/V from ~3.1 to ~1.6 GiB at 256K; 3-bit
would save only ~0.4 GiB more for four times the loss. Not built yet.

## 256K with gates and the ring: 12 streamed layers fit again

Binary and `metal/` from one snapshot, 248K-token prompt, FP8 KV, 6GB cache,
MTP, `DS4_QWEN4_PLE_EVICT_TOKENS=1024`:

| streamed | path | gen t/s | prefill t/s | peak wired | expert cache | output |
|---:|---|---:|---:|---:|---|---|
| 16 (K=32) | gate | 31.33 | 276 | 46.19 GiB | locked | counts 1..155 |
| 16 (K=32) | drain | 31.58 | 298 | 46.05 GiB | locked | identical sha |
| 12 (K=36) | gate | 32.50 | 332 | 50.01 GiB | locked | identical sha |
| 12 (K=36), morning, no ring | drain | 26.70 | 386 | 52.2 GiB | capped to 120 experts | - |

With the raw indexer keys in a ring, 12 streamed layers keep a fully locked
expert cache at a full 256K context, so the K=36 setting now works at every
context size (the K=32 fallback for 256K is no longer needed). On this counting
output (97% MTP acceptance, few new experts) gate and drain decode at the same
rate. `DS4_QWEN4_PLE_EVICT_TOKENS=1024` costs long-context prefill: the same
248K prefill ran at 435-446 t/s without it.

## 4-bit KV cache (default since the next section; `DS4_QWEN4_KV_Q4=0` opts out)

The 4-bit cache stores K/V as 4-bit signed levels (-7..7) on the FP8
cache's per-64-block fp16 scale (absmax/7), in the FP8 buffers at half their
size: each lane's 8 elements are one 32-bit word, read as one word in the decode
tile and as four words per 32-element segment in the prefill attention. Head
dim 256 only. Its checkpoints carry their own payload tag, so a 4-bit and an
FP8/BF16 session never restore each other's KV (the FP8 server rejects a 4-bit
checkpoint as "a different model family or shape" and prefills).

NLL at 64K depth (same scoring as the fake-quant table):

| KV | EN fresh | repeat | VI fresh |
|---|---:|---:|---:|
| BF16 | 1.0777 | 0.0173 | - |
| FP8 | 1.0882 | 0.0163 | 1.9258 |
| 4-bit packed | 1.0785 | 0.0166 | 1.9472 |

The packed cache is within about +-2% of FP8 (better on English, worse on
Vietnamese, equal on retrieval), much closer than the fake-quant estimate,
which paid the FP8 container's rounding on top of the 4-bit levels. At 8K it
decodes coherently in English and Vietnamese at FP8 speed (35.7 / 35.5 t/s vs
33.4 / 34.2). At a full 256K context with 12 streamed layers: peak wired 48.28
GiB vs 50.01 GiB for FP8 (-1.7 GiB), decode 32.36 t/s, needle HIT, correct
count, expert cache locked. Server checkpoint save and restore reproduce the
reply byte for byte.

## PLE sidecar eviction for the stream config

Full 256K context, 12 streamed layers, gates, ring, FP8 (identical output in
all three, swap unchanged):

| DS4_QWEN4_PLE_EVICT_TOKENS | prefill t/s | gen t/s | peak wired | compressor max |
|---|---:|---:|---:|---:|
| unset (no eviction) | 492 | 32.44 | 50.02 GiB | 3.0 GiB |
| 8192 | 460 | 31.56 | 50.94 GiB | 3.2 GiB |
| 1024 (PROD setting) | 374 | 33.33 | 50.06 GiB | 4.4 GiB |

Dropping the sidecar pages every 1024 tokens makes long prefill re-fault the
table; without eviction the clean pages stay in the page cache (reclaimable)
and nothing swaps. For the stream config, leave the eviction off.

## 4-bit KV on by default; batched sessions with a compressed cache

With the packed cache matching FP8 quality, it is now the default KV storage
for head dim 256: `DS4_QWEN4_KV_Q4` unset means 4-bit whether or not
`DS4_QWEN4_KV_FP8` is set, and `DS4_QWEN4_KV_Q4=0` restores the previous
behaviour (E4M3 with `DS4_QWEN4_KV_FP8=1`, the half cache without it).

8K, 400 tokens, main config, binary and `metal/` from one snapshot, compared
with the runs of the previous build:

| run | KV announced | output |
|---|---|---|
| no KV env | 4-bit | identical to the old `DS4_QWEN4_KV_Q4=1` run (count prompt) |
| `DS4_QWEN4_KV_FP8=1` | 4-bit | identical to the old `DS4_QWEN4_KV_Q4=1` runs (count, Vietnamese) |
| `DS4_QWEN4_KV_FP8=1 DS4_QWEN4_KV_Q4=0` | FP8 | identical to the old FP8 run |

`ds4-server --batched-session N` (two or more concurrent sessions sharing one
arena) could not decode with any compressed cache: the speculative (MTP)
batch and its draft pass built their attention row entries with only the half
cache pointers, so FP8 and 4-bit sessions attended NULL caches with an
uninitialized KV mode. With the previous build, two concurrent requests under
`DS4_QWEN4_KV_Q4=1` fail at once with `metal speculative batch failed`. All
three row builders now share one binder that passes the session's KV storage
(half, E4M3 or 4-bit) and mode; the same two requests now decode coherently at
~18-22 t/s each.

Batched decode is not bit-identical to single-session decode in either KV
format: the batch computes a session's rows alongside other sessions' rows and
rounds differently (the per-position half K/V already differ in the last bits
at the first batched token). With identical prompts in two concurrent sessions,
the logits differ by ~1e-6 at first; the first different token came ~100
tokens in, at a near-tie (top-2 margin 0.004). A compressed cache makes this
visible sooner: a last-bit difference can move a value across a quantization
level. With the half cache, the 200-token replies in these tests happened to
match the single-session replies exactly; with FP8 and 4-bit they diverge
after ~100 tokens and stay coherent. Single-session decode is unchanged and
deterministic (same reply with gates on or off, ring on or off, streamed or
resident); `DS4_QWEN4_SESSION_BATCH=0` makes concurrent requests reproduce
it exactly.

Default config at a full 256K context (4-bit KV default, 12 streamed layers,
6GB cache, gates, ring, MTP with the 64K draft vocabulary, no PLE eviction;
248,161-token prompt, 600 tokens, one run):

| prefill t/s | gen t/s | MTP accept | peak wired | decode wired | swap | output |
|---:|---:|---:|---:|---:|---|---|
| 626.0 | 33.71 | 79.5% | 48.45 GiB | 48.34 GiB | unchanged | needle HIT, count in order; identical to the earlier 4-bit 256K run |

Against FP8 without eviction (492 / 32.44 t/s, 50.02 GiB peak), the 4-bit
cache prefills 27% faster at this depth, where attention reads dominate and
each key now costs half the bytes, and peaks 1.6 GiB lower; that is also above
the resident FP8 baseline (562.6 t/s at 248K). A 248K prompt takes ~6.6 min.

## Resident vs streamed with the 4-bit cache: where 40 t/s is

Same build and defaults (4-bit KV, MTP + 64K draft vocabulary). 8K, 400 tokens:

| config | gen t/s VI | gen t/s EN | MTP accept VI/EN | peak wired |
|---|---:|---:|---:|---:|
| resident (no `--ssd-streaming`) | 40.06 | 40.93 | 70.8 / 76.3% | 53.03 GiB |
| 6 streamed (K=42), 3GB cache | 38.00 | 39.67 | same | 49.67 GiB |
| 12 streamed (K=36), 6GB cache | 35.05 | 35.23 | same | 46.85 GiB |

Full 256K (248,161-token prompt, 600 tokens):

| config | prefill t/s | gen t/s | peak wired | output |
|---|---:|---:|---:|---|
| resident | 658.7 | 35.86 | 55.43 GiB | needle HIT; identical to the streamed run |
| 12 streamed (K=36) | 626.0 | 33.71 | 48.45 GiB | needle HIT |

Resident unc31 now fits a full 256K context (the 4-bit cache and the indexer
key ring together free ~3 GiB against the FP8 resident run, which peaked near
the limit at 28.3 t/s), with swap unchanged. 40 t/s is reached at short
context only when resident; at 256K the attention over ~248K keys caps even
resident decode at ~36 t/s. Wired figures are system-wide (~3.2 GiB machine
baseline); the GPU wired limit is `iogpu.wired_limit_mb` = 57344.

## Split gates: the cached experts run while the misses are read

A gate used to hold the GPU for its whole service: the mailbox arrives, the
service thread resolves every selected expert (reading the misses from the
SSD) and only then releases the poll. Split gates (on by default;
`DS4_QWEN4_STREAM_SPLIT=0` keeps one pass) release as soon as the gate's
experts have been classified against the cache — 0.5 us after the mailbox —
and the GPU runs the cached experts and the shared one (phase 1) while the
misses are read; a second poll in the same command buffer waits for those
(phase 2). The `*_addr` kernels take a phase and the gate's bitmask of missed
experts, and skip the pairs the other pass owns.

Phase 2 reads the missed experts' addresses from a **per-gate** table, not the
layer's: a running command buffer keeps the stale L2 copy of any line it has
already read, and phase 1 reads the layer table. The pending mask and the
pass-2 table of a ring slot are only read after the poll that follows their
write. Down still writes one partial per (row, slot) and the reduce adds the
slots in order, so every pair is computed exactly once with unchanged
arithmetic: 8K Vietnamese and English, a full 256K context, a server session
and `--batched-session 2` all match the unsplit gates and the resident path
byte for byte.

Releases now write the first 2048 lines of a poll region and the tails after
the other release, and the cache bookkeeping (hotness, recency, pruning) runs
after both releases instead of before them.

8K, 1200 tokens, 12 streamed layers @ 6GB, decode t/s and GPU busy per step:

| build | gen t/s | GPU busy/step | poll A release line | poll B |
|---|---:|---:|---:|---|
| resident | 40.8-40.9 | 41.7 ms | - | - |
| gates, one pass | 37.7 | 43.2-45.3 ms | 1170-1340 | - |
| gates, split | 38.8 | 41.9-43.7 ms | 640-660 | line 0 without misses |

## Where the streaming overhead sits now

`DS4_QWEN4_STREAM_TIMING` reports the round trip against the publish batch's
GPU end time, and `DS4_METAL_GPU_IDLE` the GPU busy time per waited group:

| part | per gate | per step (12 gates) |
|---|---:|---:|
| publish batch end -> service thread sees the mailbox | 12.3 us | |
| batch gap (the commit itself) | 0.5 us | |
| poll A start -> release | 12.4 us | ~0.2 ms |
| publish + commit + the extra pass-2 dispatch | ~22 us | ~0.27 ms |
| poll B, gates with misses (55% of gates, 1.94 misses each) | ~60 us avg | ~0.7 ms |

The GPU is never idle *between* command buffers (0.0-0.3%); the cost is the
poll spinning inside them. Dropping both polls (a timing-only experiment with
wrong output) saves 130 us per gate, so the polls, not the command-buffer
boundaries, are what streaming still pays. A miss gate reads 1.94 x 1.86 MiB
in ~355 us, about 10 GB/s, so the reads are at the SSD's ceiling rather than
waiting on latency.

## Expert-cache aging (on by default)

The qwen4 gate path never advanced the cache's aging clock, so victims were
ranked by route hotness accumulated since the process started. The service
thread now ticks it once per decode step: 8K, 1200 tokens, 12 streamed @ 6GB
takes 7570 misses instead of 8634 (hit rate 0.925 -> 0.935) with the same
output. `DS4_METAL_STREAMING_EXPERT_EVICT_LRU=1` (recency only) lands on the
same 7570, `DS4_QWEN4_STREAM_AGING=0` restores the old ranking.

## Router lookahead prefetch (opt-in, off)

`DS4_QWEN4_STREAM_LOOKAHEAD=K` copies a gate's router input to a shared slot;
the service thread runs the *next* streamed layer's router on it (Accelerate
sgemv, off the critical path) and reads the K experts per row it predicts and
the cache lacks. K=6 reads 0.70 experts per gate of which 0.47 are used (67%
accurate), demand misses per miss-gate fall 1.94 -> 1.67 and the second poll
waits on 28.9% of them instead of 36.3%. The extra reads cost what the shorter
waits save: paired 800-token runs put K=6 at 1.007 of K=0 (median 0.99), so it
stays off. antirez/ds4 #849 reports the same on DeepSeek until the packed
expert layout of #848 frees SSD bandwidth.

## Where K=36 stands against resident

The box drifts while these run (about 10% over half an hour, with a disk
cleaner competing for IO), so every comparison below is either paired
(alternating 800-token runs, per-pair ratio) or interpolated between two
resident runs that bracket the series.

| config | share of resident | peak wired at 8K |
|---|---:|---:|
| 12 streamed (K=36) @ 6GB | 0.952-0.963 | 47.3 GiB |
| 12 streamed (K=36) @ 8GB | 0.967-0.977 | 49.4 GiB |
| 8 streamed (K=40) @ 4GB | 0.955 | 49.1 GiB |
| resident | 1.000 | 53.2 GiB |

Warm `ds4-server`, first round of a series (the machine's best state):

| config | Vietnamese | counting (EN) | code | peak wired |
|---|---:|---:|---:|---:|
| resident | 40.2-40.6 | 44.5 | 37.7 | 53.2 GiB |
| K=36 @ 8GB | 38.4 | 41.6 | 34.9 | 49.4 GiB |
| K=36 @ 6GB | 36.9-38.6 | 40.3-41.3 | 33.9 | 47.3 GiB |

K=36 went from ~93% of resident to ~96-98% tonight. The counting workload
(high MTP acceptance) clears 40 t/s streamed; Vietnamese prose does not yet:
it needs the last ~3%, which is the miss waits (1.7%), poll A (0.6%) and the
fixed per-gate cost (0.6%).

## Two more levers, measured and rejected

**Staged miss reads** (`DS4_QWEN4_STREAM_STAGE_MISS=1`): read the missed
experts' gate and up slices, release the misses' mid, then read their down
behind a third poll — the ordering half of antirez/ds4 #1083. It loses here:
the two reads serialise where one batch had them in parallel, so the down
slices land later than the whole expert used to. 8K, 800 tokens, 12 streamed
@ 6GB: first -> second release 390 -> 593 us, 38.57 -> 35.99 t/s. A miss gate
reads 3.6-4.2 MiB at about 10 GB/s, which is the SSD's ceiling rather than a
latency the ordering could hide; the packed expert layout of #848 is the piece
that would change that, and it is why #849's prefetch pays there and not here.

**A larger cache at a full 256K**: 8GB instead of 6GB peaks at 51.60 GiB
(against 48.54) and decodes 31.63 t/s against 32.51, so 6GB stays the setting
at 256K. At 8K the same 8GB is worth +1.4% (paired), which is where its extra
2 GiB belongs.

## Shipping state (scale3, 2026-09-23)

Default for a streamed qwen4 layer: split gates, cache aging on, lookahead and
staged reads off. Validation of the shipping build, all byte-identical to the
resident path and to the previous build: 8K Vietnamese (38.99 t/s) and English
(42.50), a full 256K context (32.39 t/s, peak 49.58 GiB, needle HIT), a server
session, and `--batched-session 2` with two concurrent requests.

| context | config | decode t/s | peak wired |
|---|---|---:|---:|
| 8K | resident | 40.2-41.0 | 53.2 GiB |
| 8K | K=36 @ 8GB | ~39.5 (0.97 of resident) | 49.4 GiB |
| 8K | K=36 @ 6GB | ~38.9 (0.96 of resident) | 47.3 GiB |
| 256K | K=36 @ 6GB | 32.4-32.5 | 48.5-49.6 GiB |
| 256K | resident | 35.9 | 55.4 GiB |
