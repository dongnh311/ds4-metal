# Phase 3 — Optimization audit (profile-first, execution-first) on the light Orca model

Confirmed-plan Phase 3, run profile-first per plan v2's thesis: "no optimization
without a measured gap; SSD paging = scale-not-speed; KV last." Target = the
Phase-1 light OrcaUncensored model (resident, Ivan-parity) on M5 Pro 64 GiB with
the Phase-2 fixed binary. Result: the resident model is at its hardware ceiling;
the one live lever (expert-matvec specialization) measured a bit-exact no-op.

## SPECIALIZE audit (DS4_QWEN4_MOE_MV_SPECIALIZE) — bit-exact, MEASURED NO-OP
On M5 the default specializes only type 39 (MXFP4 down); types 16 (IQ2_XXS
gate/up) and 10 (Q2_K down) — the Orca experts — use the generic mat-vec
(ds4_metal.m:47660). Forcing DS4_QWEN4_MOE_MV_SPECIALIZE=1 on types 16/10:
- **Bit-exact:** frontier logits byte-identical off vs on (the specialized kernel
  keeps the original per-lane reduction order + padded stride, by design).
- **Speed (3x paired runs, ctx 4096, mean):** prefill 535.4 -> 535.4 t/s
  (**+0.0%**), decode 33.99 -> 34.04 t/s (**+0.1%** = noise). A single earlier
  run showed +16% prefill / +1.9% decode, but that was first-run warmup (cold
  prefill 460 vs steady 535), not a SPECIALIZE effect.
- **Verdict: SKIP.** No measurable win on M5 for types 16/10 (decode is
  DRAM-BW-bound: specialization reads identical bytes -> identical throughput).
  The default-off for M5 is correct. No code change. (The summary's "+9% bit-exact"
  did not reproduce here; it was a different model/config or noisy.)

## Other Phase-3 items — status against the resident light model
- **profile decode / Metal micro-opt:** already characterized (Task 1, b0f6fe6):
  decode is DRAM-bandwidth-bound (GPU ~90% busy, NSG-invariant, bytes/token
  fixed). No bit-identical kernel change raises decode t/s. SKIP.
- **cache / pager / prefetch / memmgr:** these govern the SSD-streaming / heavy
  path. The light model (45.3 GiB resident) needs NO --ssd-streaming, so the
  pager/prefetch/memmgr are inert here. MOOT for this artifact (they remain
  relevant only for the heavy 147 GB build). SKIP.
- **MTP:** already tuned (Task 4: +14% prose / +33% code) and confirmed on the
  Orca model: 40.2 t/s with --mtp vs ~34 without (+18%), MTP block survived the
  Phase-1 strip. No further change.
- **agent / tool-calling throughput:** validated in Phase 2 (server + tools +
  --mtp-exact-sampling): correct tool_calls in 4-5 s each, ~40 t/s, stable.
- **KV quant:** quality-gated, "last" per the plan; out of scope unless the user
  opts into a quality change.

## Bottom line
The light OrcaUncensored model runs at the M5 Pro hardware ceiling: prefill
~535 t/s @4K (up to ~473 @220K), decode ~34 t/s MTP-off / ~40 t/s --mtp,
zero-swap 8K->220K, bit-exact experts. The genuine improvements over the raw
Ivan/Orca starting point were delivered in Phases 1-2 (light uncensored
packaging + the four bug fixes + MTP), not in kernel micro-optimization — which
the profile shows is a dead end for a bandwidth-bound decode on this hardware.
Remaining decode headroom requires a smaller quant (fewer bytes/token, a
quality-gate change) or more memory bandwidth (hardware).

## Receipts
specialize_ab3.log (3x paired), spec_off.csv / spec_on.csv (+ byte-identical
frontier logit dumps in scratchpad spec_off/ spec_on/). Task 1: b0f6fe6.
