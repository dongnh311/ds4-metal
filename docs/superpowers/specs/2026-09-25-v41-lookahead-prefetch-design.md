# V4.1 lookahead expert prefetch into the OS page cache — design

Date: 2026-09-25. Status: approved in conversation (three sections), pending spec review.
Parent: `docs/V41_64GB_BUILD.md` DoD 2 (≥ 20 t/s without quality loss), §8 step 7 (predictive prefetch).
Evidence: the 2026-09-25 spike (memory `v41-lookahead-spike`), throwaway code; results below.

## Problem

DeepSeek-V4.1-Flash Q2 decodes at 11.35 t/s on the M5 Pro 64 GB (ctx 8192,
`switch` prompt, auto cache 35.62 GiB = 3075 experts, develop `6178e90`).

| Term | ms/token |
| --- | ---: |
| GPU busy | 54.8 |
| Miss path: `F_RDADVISE` readahead 15.3 + pread 13.3 (hit rate 0.87) | 28.6 |
| Host and sync | 4.5 |
| Engram | 0.6 |
| **Step** | **88.5** |

The miss path is the largest term that I/O can remove. It runs serially with the GPU:

- A layer's missed experts are only known once its router has run.
- The async worker reads them while the shared expert runs.
- The routed experts wait for that read.

During the ~55 ms of GPU work per token the SSD is otherwise idle.

## Spike evidence

**Cache simulator.** It replays the Metal policy:
- victim = the lowest route hotness, then the oldest use;
- hotness is halved every 16 tokens;
- the current layer's selection is protected.

It reproduced the measured hit rate exactly: `switch`, ctx 4096, 1821 slots, 0.708.

**History-only predictors do not work.**
- The previous token's experts are already cached.
- A learned layer→layer transition table covers about 8 % of misses at a 3-10× read overhead.

**Predictor A = layer L+1's router (sqrt(softplus(logits)) + bias) applied to layer L's FFN-norm input.**
- Recall of the 6 selected experts: 0.62-0.68 in the top 6, 0.69-0.75 in the top 8, 0.81-0.85 in the top 16 (`switch` and `code`, 1100 tokens each).
- Layer L's own router applied to its attention-norm input is worse: 0.54 in the top 6.

**Simulated at 3075 slots, predictor A with k = 1 per layer**, where a guess does not enter the expert cache:
- Demand misses: 39.4 → 24.2 per token (`switch`) and 39.9 → 27.8 (`code`).
- Cost: 39 extra expert reads per token (~370 MiB), about 2.6-3.2 reads per saved miss.
- k = 2 saves more (17.8 / 22.4 demand misses) for 78 extra reads.
- Inserting guesses into the expert cache is worse, because of pollution.

**Upstream #849** (V4-Flash) runs the same predictor on the CPU with staging buffers and reads only while the disk is idle.
- It reported 46 % coverage at the top 4 and lost 5.5 %: that box was disk-bandwidth-bound.
- Here the disk idles through the GPU time.

## Goal

Hide part of the miss path by warming the OS page cache with the predicted experts of the next layer, before that layer's router runs. Target: ≥ 12 t/s at ctx 8192 with the shipped defaults, bit-identical output.

Estimate:
- Page-cache-resident misses cost only the pread copy: about 0.43 of the 0.92 ms per miss.
- k = 1 should save about 5 ms/token, giving about 11.9-12.1 t/s.
- The staging approach (below, out of scope here) could reach about 10 ms.

## Constraints

1. **Bit-identical output.** Prefetch only moves file pages into the OS page cache. The expert cache, the selected experts, the weights and all kernels are unchanged. Switch: `DS4_METAL_DISABLE_V41_LOOKAHEAD`, read per call.
2. **V4.1-only.**
   - New code sits at a V4.1 call site plus one read-only Metal query.
   - The worker job struct shared with V4-Flash is not changed.
   - Nothing in the shared cache code (pread pool, load, install, evict) changes behavior.
3. **No critical-path wait.** The main thread spends only a 20 KB copy and a mailbox post per layer. Everything else runs on a separate thread and can be dropped.
4. **Machine limits unchanged.** No new wired memory: the page cache is managed by the OS. Zero swap.
5. **Speed is judged by interleaved A/B** (A, B, B, A) only.

## Design

### Data flow

1. **Main thread, layer L (`ds41_moe_partial`).**
   - Hook point: after `metal_graph_selected_async_load_finish` returns layer L's ids, and before the routed MoE is encoded.
   - The main thread copies `g->norm` (layer L's FFN input, 5120 floats, 20 KB) into the lookahead mailbox, with the layer and position.
   - The copy is safe here: the router's GPU work is complete (the worker waited on its event), and nothing that writes `g->norm` is encoded yet (layer L+1's pre-attention comes later).
   - The hook runs only when the async load ran (the default); without it there is no lookahead.
2. **Mailbox.** One slot. A post replaces an unconsumed slot, which is counted as dropped. The main thread never waits.
3. **Prediction thread** (V4.1, started lazily):
   - take the slot, with target layer T = L + 1;
   - skip if T = 40, or if the position is an image position (that one uses a different bias);
   - scores = a dot-product loop over layer T's `ffn_gate_inp` (F32, 384 × 5120, resident in the mapped model) with the copy;
   - s = sqrt(softplus(score)) + `ffn_exp_probs_b[T]`;
   - rank the top 16 and take the first k (default 1) that are not cached. Taking the top k first and dropping the cached ones prefetched almost nothing (4.4 experts/token), because the top of the ranking is usually a hot, cached expert; the spike's numbers use the first-uncached rule.
4. **Prefetch.** For each picked expert e (`ds4_gpu_stream_expert_cache_resident_hint(T, e)` reported it not cached), issue `F_RDADVISE` on the thread's own read-only fd for three ranges:
   - gate: `ffn_gate_exps` offset + e × gate_bytes;
   - up: `ffn_up_exps` offset + e × gate_bytes;
   - down: `ffn_down_exps` offset + e × down_bytes.

   These are the offsets `graph_stream_expert_table_make` uses.
5. **Layer T's demand load** is the existing path, unchanged. Its readahead and pread find a guessed expert's pages resident.

### Components

- **`ds4.c`, V4.1 only:**
  - a lookahead state holding the thread, mutex/cond, mailbox slot, fd, k and counters;
  - `ds41_lookahead_post(g, m, w, il)`, called from `ds41_moe_partial`;
  - the thread body;
  - start on the first V4.1 streaming decode step, stop and join in the graph's free path, before the model can be unmapped.
- **`ds4_metal.m`:** `int ds4_gpu_stream_expert_cache_resident_hint(uint32_t layer, uint32_t expert)`. It is a relaxed atomic load of the entry's `valid`, and returns 0 outside streaming or out of range. Declared in `ds4_gpu.h`.
- **Switches:**
  - `DS4_METAL_DISABLE_V41_LOOKAHEAD`: read per call, default on if the A/B keeps it;
  - `DS4_METAL_V41_LOOKAHEAD_K`: 0..6, default 1; 0 turns prediction off.
- **Counters,** printed as `ds4: V4.1 lookahead: posted N dropped N predicted N issued N used N` next to the V4.1 decode profile:
  - predicted = jobs processed;
  - issued = experts advised (first k not cached);
  - used = issued and then selected by layer T. The main thread compares T's selected ids with the prediction the thread recorded for T.

### Errors and safety

- **Failures.** A failure to open the fd, create the thread or allocate turns lookahead off with one stderr line. Decode never fails because of lookahead.
- **Races.**
  - The residency read is a hint: a stale value costs one redundant or one missing prefetch.
  - The model weights are read-only.
  - `F_RDADVISE` runs on a private fd.
  - Joining the thread before the graph is freed keeps the weight pointers valid.
- **Load on the machine.**
  - At most k advisories per layer, issued in the gap between layer L's demand reads and layer T's router.
  - If the thread falls behind, posts are dropped rather than queued.

## Testing and acceptance

- **Model-free tests** (`tests/test_deepseek41_graph.c`):
  - the scorer's top-k against a reference on synthetic data (ties by lower index);
  - the three prefetch ranges against `graph_stream_expert_table_make`;
  - mailbox drop semantics;
  - `DS4_METAL_V41_LOOKAHEAD_K` parsing (clamped to 0..6).
- **Exactness.** `--stream-control ... DS4_METAL_DISABLE_V41_LOOKAHEAD` at 8 and 24 GB. The stream-control selftest must still catch its three planted differences.
- **Speed.**
  - Interleaved A/B at the auto cache and at 24 GB, for k = 1 and k = 2.
  - Counters checked against the spike (used / issued ≈ 0.3).
  - GPU stage profile to confirm GPU busy does not grow.
- **Qwen fast tier,** because `ds4_metal.m` changes.
- **Keep the feature** only if it is exact and the A/B is faster; pick k by A/B.

## Out of scope

- Staging buffers that hand prefetched bytes straight to cache slots (approach 2). This is the follow-up if the pread copy stays large.
- A GPU-side predictor, predicting more than one layer ahead, adaptive k.
- Any change to cache policy, selection or kernels.

## Risks

- **The disk may not be idle enough.** `F_RDADVISE` for ~370 MiB/token could delay layer T's demand reads. Mitigation: k = 1 first; the A/B decides; the counters show whether posts are dropped.
- **Page cache churn** (~4 GB/s at 11 t/s) could evict pages that later misses would have hit. The A/B covers it.
- **CPU memory bandwidth** for the predictor: 7.5 MB per layer, ~300 MB/token, about 1 % of the GPU's bandwidth.
- **The gain may be smaller than estimated** if most misses already come from the page cache. The counters and the A/B show it, and the result sets up approach 2.
