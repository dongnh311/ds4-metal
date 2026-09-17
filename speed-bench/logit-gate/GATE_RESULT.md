# Gate #5 result: resident vs paged logit dump (item 1.3)

Status: **PASS (bit-identical)** once the compute kernel is held constant so the
only variable under test is the pager (SSD-streamed expert bytes vs mmap'd
resident bytes). Three independent bugs were found and fixed to get here; the
history is kept below because each fix mattered and the earlier "FAIL" states
were real.

## How the gate is run (and why --prefill-chunk 8)

`speed-bench/run-logit-gate.sh [CTX] [CHUNK=8]` runs the same prompt twice --
resident (weights from the mmap'd GGUF) and paged (`--ssd-streaming`, routed
expert weights from `qwen38-experts.bin` via the pager) -- dumps the frontier
logits from each, and diffs them lane by lane.

Both runs pass `--prefill-chunk 8`. Gate #5 exists to prove the PAGER does not
change the math, so it must hold the compute KERNEL constant and vary only the
weight source. At a prefill chunk of T>8 the resident path takes the tiled
"mm" GEMM, whose dot-product accumulation order differs from the per-token
decode kernel by floating-point non-associativity. The paged path structurally
cannot use "mm" (its per-layer staging buffer holds a single token's routed
experts), so it always runs the per-token kernel. Comparing mm-resident against
per-token-paged would fold a kernel-ordering difference into the pager test.
Capping the chunk to 8 puts resident on the same per-token kernel the pager
uses -- which is also the kernel the paged path uses at any bench context
(streamed prefill and decode are both per-token), so the gate stays
representative of the real paged workload.

## Result (this run: ctx=512, --prefill-chunk 8, clean build)

- resident vs paged frontier logits: **max_abs_diff = 0.0**, **0 / 248320**
  value mismatches, resident_dump_sha256 == paged_dump_sha256.
- gguf sha256 = ed238d8d50c6fb4b504c2796c3bc82b2a9f6c21566ad29bdbfdbadfe01e28d7a
  (locked artifact, confirmed by the gate's own pre-flight check).
- diskwrites: pager opens the bundle O_RDONLY; no writes to the model volume.

Cross-checks at other contexts (same-kernel, resident vs paged):
- ctx=1   : max_abs_diff = 0.0, 0/248320 mismatches (single token).
- ctx=8   : max_abs_diff = 0.0, identical dump sha (multi-token T=8 exercises
            the per-token staging-overwrite loop, the per-token tensor views,
            and the reduce -- all bit-identical).
- ctx=512 : max_abs_diff = 0.0, identical dump sha (this run).

## The three bugs

### Bug 1 -- bundle extraction wrote the wrong byte size for down experts
`gguf-tools/qwen4_expert_bundle.py` mapped GGUF type 10 (Q2_K, 84 B/256-block)
to 66 B (IQ2_XXS's size), so every down-expert bundle was extracted at the
wrong size and offset. FIXED (`10: 84`), bundle regenerated from the locked
GGUF, verified byte-identical to the same (layer, expert) slice read straight
from the GGUF. Before this fix the paged path produced all-NaN frontiers.

### Bug 2 -- kernel indexed the staging buffer by global expert id
`metal/qwen4.metal` `kernel_qwen4_moe_mid` / `_down` computed the weight offset
for a routed slot as `selected[...] * expert_bytes` (a global 0..511 id). That
is correct for the resident buffer (all 512 experts) but wrong for the pager's
per-layer staging buffer, which holds only this call's `n_slots` experts at
their local slot position. FIXED with a `paged_local_index` arg (index by local
slot in the paged wrappers), and `qwen4_graph_moe` was restructured so the
paged path never takes the batched "mm" branch. Before this fix the paged path
was finite but wrong (max diff ~14.8 at ctx=1).

### Bug 3 -- pager staged the previous layer's experts (stale routing read)
`ds4.c` `qwen4_graph_moe`'s pager branch read `g->selected` into a host buffer
with `ds4_gpu_tensor_read` (a plain CPU memcpy) to decide which experts to
stage -- but it did so BEFORE the open command batch was committed. The router
gemv + top-k kernels that WRITE `g->selected` were still queued and had not
executed, so the read returned the previous layer's routing (or, on layer 0,
whatever the buffer held). The pager then staged the wrong routed experts,
diverging on every routed slot from layer 0 onward and compounding through the
stack, while the shared expert (whose address does not depend on this read)
stayed bit-identical. This is why a synced debug dump of `g->selected` looked
correct yet the pager consumed stale values -- the dump read at a different,
post-commit time than the pager. FIXED: `ds4_gpu_end_commands()` now runs
BEFORE the read, committing + waiting the routing kernels so `g->selected` is
current. That same commit also preserves the pre-existing invariant that the
previous token's kernels finish reading the staging buffer before it is
overwritten. Diff dropped from 4.53 -> 1.88 at ctx=512 default (the residual
being the mm-vs-per-token kernel difference, not the pager), and to 0.0
bit-identical once the kernel is held constant.

## Not a pager bug: resident "mm" vs per-token (flagged for decision)

With the DEFAULT prefill chunk, resident prefill (T>8) uses the tiled "mm"
GEMM and paged uses per-token; they differ by ~1.88 max logit diff at ctx=512
(all lanes differ, argmax stable). This is floating-point accumulation order
between two kernels, inherent to any tiled GEMM, and is present resident-only
(mm-prefill vs per-token-decode) independent of the pager. The pager cannot
match "mm" without either (a) a batching-aware paged buffer scheme so paged can
run "mm" too (Phase 2 kernel work), or (b) disabling "mm" at prefill (a
prefill-throughput cost; decode is unaffected). Deferred to the user per ground
rule 8; it does not affect the benched paged path, which is per-token and
proven bit-identical here.

## Read-GB and swap note (gate telemetry is NOT the KPI)

Two paged configs at ctx=512, both correct/bit-identical, very different cost:

| config                         | SSD read | swap growth | hit_rate | misses  | prefill tok/s |
|--------------------------------|----------|-------------|----------|---------|---------------|
| --prefill-chunk 8 (this gate)  | 64.7 GiB | +53.7 GiB   | 0.8102   | 139935  | 4.88          |
| default prefill chunk          | 15.6 GiB | +7.14 GiB   | 0.9542   | 33744   | 17.20         |

The gate forces `--prefill-chunk 8` for a bit-identical KERNEL comparison, but
that config is pathological for the pager: 512 tokens become 64 forwards of 8,
and every layer's experts are re-fetched once per forward, so the resident
cache (4.86 GiB, 8087 slots) thrashes -- 4x the misses and reads of the
default single-forward chunk. So the gate's read/swap columns are an artifact
of the correctness config, NOT the read-GB KPI. The realistic paged path
(default chunk) reads 15.6 GiB at 0.95 hit rate.

Even the realistic 15.6 GiB is over ground rule 4's budget (bundle 34.31 GiB x
miss_rate 0.0458 x 1.1 = 1.73 GiB): the paged prefill re-reads experts across
tokens more than "each missed expert once" would predict. This is a cache-
sizing / prefetch question for the bench items (1.2 read-GB CSV, 1.4 eviction),
not a correctness question -- recorded here so it is not lost, and it is what
Phase 1's "so that" real-number collection is meant to root-cause.

diskwrites: the pager opens the bundle O_RDONLY (no writes to the model
volume by design). The gate CSV does not yet carry an iostat/ssd_watch
diskwrites_delta column; wiring that into the receipt is item 1.2's job.

## Per ground rule 5

"Gate bit-identical ... TRUOC khi bench." The pager is bit-identical to
resident on the shared per-token kernel at ctx 1, 8, and 512. Item 1.3 passes.
The only non-bit-identical comparison (resident-mm vs per-token) is a kernel
choice outside the pager and is flagged, not silently accepted.
