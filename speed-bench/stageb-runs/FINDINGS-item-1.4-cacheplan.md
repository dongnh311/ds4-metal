# Item 1.4 — expert cache sizing: the auto plan over-charges model residency

## Finding
`ds4_qwen4_streaming_auto_configure` (ds4.c ~68145) computes the expert-cache
budget after subtracting `resident_model_bytes` from the working set. It starts
that value at `non_routed_bytes` (6.32 GiB) but then (ds4.c:68170-68182, Metal
streaming path) OVERWRITES it with the **decode-static full span = 41.72 GiB**
(`weights_model_map_decode_static_full_spans`), with a comment asserting the
full span must be charged to avoid swap.

At high ctx this starves the cache. Measured on the locked artefact:
  - **ctx 32768**: auto cache = 1.36 GiB (2263 slots); read 165.9 GiB, hit 99.29%.
  - **ctx 220000**: auto cache collapses to **2 slots / 0.00 GiB** — the warning
    "the floor of 980 experts + 41.72 GiB resident model + 17.27 GiB context
    does not fit the 56 GiB working set" fires and the run would re-read on
    EVERY access (read volume grows with tokens, ~hundreds of TB → impractical).

## The 41.72 charge is ~35 GiB too pessimistic on the PAGED path (measured)
vmmap -summary of the live paged process shows the GGUF mmap ("mapped file")
is 41.9 GiB virtual but only **6.3 GiB resident** throughout a run — the routed
experts are served from the pager BUNDLE (pread), not from the model mmap, so
the 35 GiB expert region of the mmap is never faulted in (except 1/49 "bypass"
layers). Clean mmap pages evict under pressure and re-read from disk on demand.

Empirical test — forced a 10 GiB expert cache at ctx 220000
(`--ssd-streaming-cache-experts 10GB`, an allowed --ssd-streaming-family flag):
  - plan PRINTED "= 68.98 GiB planned" (charging 41.72 model)
  - **actual physical footprint plateaued at 20.5 GiB**, memory_pressure 53%
    free, swap growth ~0.
So the real resident set is ~6.3 GiB model + KV + buffers + 8.58 GiB cache =
~20-27 GiB; the 41.72 charge is phantom on the paged path. There is headroom to
cache the entire 34.31 GiB bundle under 64 GiB (would drop read toward
bundle-size, i.e. each expert read ~once -> within ground rule 4).

## Recommendation (candidate code fix, NOT yet applied — needs the >55K/220K
##  completion numbers first to quantify, and careful validation vs the n-gram
##  mmap under pressure)
On the Metal paged path, charge `resident_model_bytes` at the ACTUAL paged
residency (non_routed static span + the single bypass layer's experts, ~7 GiB),
NOT the full 41.72 GiB decode-static span, so the auto plan grants the expert
cache the ~35 GiB it currently forfeits. Guard with the measured footprint so
the n-gram/embedding mmap pressure the original comment feared is bounded.

Also the pre-existing slot-array note (item 1.7): the slot index array is sized
by max_bundle (645120) so slot COUNT can bind before the byte budget; size by
the smallest bundle (or bundles_count) so the byte budget is the sole cap.

## Eviction E1 (LRU) vs E2 (scored) A/B: pending — re-run at 32K frontier on the
## fixed pager (see eviction_ab.sh). 8K was within noise.

## REVISION (measured during the 220K forced-10GB run — corrects the above)
The "41.72 charge is simply too pessimistic, so force a big cache" conclusion is
INCOMPLETE. vmmap of the live 220K run (forced 10 GiB cache) shows:
  - mapped file (model mmap): 41.9G virt / **6.3G resident** — confirms the model
    itself stays ~6.3 GiB resident (the 41.72 charge IS phantom for the mmap).
  - BUT MALLOC_SMALL (the expert cache): 7.2G alloc / 2.6G resident / **4.2G
    swapped_out**; overall writable regions **49% swapped (19.4 GiB)**.
  - swap grew 3.3 -> 5.0 GiB and climbing (~0.5 GiB / 5 min).
So the forced anon expert cache is itself SWAPPED at 220K. Root reason: at 220K,
KV (6.99) + buffers (10.27) + expert cache (8.58) is ~26 GiB of anon memory, and
it competes with the model mmap (6.3 resident) AND the 95.37 GiB NNgram table's
on-demand file-cache for the 64 GiB machine. The OS swaps the bench's anon pages
(cache/KV) to keep file cache, so a large expert cache does not stay resident —
cache "hits" then read from swap, and the zero-swap DoD is violated.

## Corrected conclusion
At ctx 220K on 64 GiB this is a genuine 3-way squeeze (anon KV+buffers+cache vs
model mmap vs 95 GiB NNgram file cache), NOT a simple plan-arithmetic bug:
  - auto plan (charges 41.72): starves cache to ~0 -> re-read storm (impractical).
  - forced big cache: reduces misses but SWAPS -> zero-swap DoD fails, slow hits.
Neither is clean at 220K. The real levers are the LATER plan phases, exactly as
ordered: Phase 3 FP8 KV (halves the 6.99 GiB KV), Phase 4 memory manager
(4-way budget across model/KV/cache/prefill with a re-fit boundary). The item-1.4
takeaway for the cache PLAN: charging the full 41.72 GiB decode-static span
over-reserves against the MMAP (which is 6.3 resident), but the working-set
pressure it guards against is real once the NNgram file cache is counted — so any
plan change must budget the NNgram/file-cache demand explicitly, not just swap
6.32 for 41.72. Lower ctx (<=32K) is unaffected: the 1.36 GiB auto cache fits with
zero swap (stageB_32k_mtpoff: swap_growth=0).

## Eviction A/B RESULT (32K, gen 32, fixed pager) — measured
| policy | hit%    | misses | read GiB | prefill tps | decode tps | swap Δ |
|--------|---------|--------|----------|-------------|------------|--------|
| E1 LRU | 99.430  | 268181 | 124.0    | 35.01       | 4.62       | 0      |
| E2 scored | 99.430 | 268217 | 124.1  | 34.88       | 4.52       | 0      |
Difference is within noise (misses differ 36 / 268k = 0.013%; identical hit rate
to 3 dp). At 32K the working set cycles nearly uniformly across all experts, so
recency (LRU) ≈ recency+frequency (scored). The eviction POLICY is not the lever;
cache SIZE is (1.36 GiB << working set → 268k re-read misses either way). This
matches the 8K A/B (also noise) and reinforces the cache-plan finding above:
optimise the cache budget, not the eviction score. Both runs diskwrites_delta=0,
swap growth 0.
