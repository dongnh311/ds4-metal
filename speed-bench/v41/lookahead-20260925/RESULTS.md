# V4.1 lookahead expert prefetch: results, 2026-09-25

Spec: `docs/superpowers/specs/2026-09-25-v41-lookahead-prefetch-design.md`.
Plan: `docs/superpowers/plans/2026-09-25-v41-lookahead-prefetch.md`.
Branch: `feature/ds41-lookahead`, from develop `6178e90`.

## Result

**Measurement point:** ctx 8192, `switch` prompt, 512 teacher-forced tokens, interleaved A/B, V4.1 alone.

| A/B | A t/s | B t/s | Ratio |
| --- | ---: | ---: | ---: |
| lookahead off vs on (k = 1), auto cache (`ab-la-k1-auto.json`) | 11.13 | 12.16 | 1.0921 |
| repeat (`ab-la-k1-auto-rep.json`) | 11.34 | 12.20 | 1.0768 |
| lookahead off vs on, 24 GB (`ab-la-k1-24.json`) | 10.53 | 11.21 | 1.0641 |
| k = 1 vs k = 2, auto cache (`ab-la-k2.json`) | 12.17 | 11.50 | 0.9445 |
| Phase-0 configuration vs shipped defaults (`ab-final-la.json`) | 8.89 | 11.64 | 1.3093 |

- **The shipped defaults** are the auto cache, queued layers, async load, #1042 fusions and lookahead at k = 1.
- **Six runs of the shipped defaults across three A/Bs** gave 12.04, 12.28, 11.48, 11.80, 12.29 and 12.12 t/s: mean **12.00**, median 12.08.
- **The ≥ 12 t/s target is met on average but sits at the edge.** The final A/B's B runs (11.48, 11.80) came right after the 24 GB A runs.
- **Lookahead alone is worth ×1.08-1.09 at the auto cache.**
- **Output is bit-exact:**
  - `--stream-control DS4_METAL_DISABLE_V41_LOOKAHEAD` at 8 and 24 GB PASS;
  - the self-test catches its three planted differences;
  - the Qwen fast tier PASS.

## How it works and what it moved

After the async worker returns layer L's ids, the main thread posts layer L's FFN input. A thread then:
1. scores layer L+1's F32 router on that input;
2. ranks the top 16;
3. advises (`F_RDADVISE`) the first expert that is not cached.

The expert cache and the selection are unchanged.

Per token at the auto cache (B side of `la-k1-auto`):

| Term | Off | On |
| --- | ---: | ---: |
| readahead (waiting for the SSD) | 15.24 | 8.79 |
| pread (copy, mostly from the page cache) | 13.45 | 13.52 |
| GPU busy | 57.37 | 57.25 |
| host | 3.60 | 2.54 |

**Counters:** 38.9 experts advised per token, 11.9 of them later selected (30 %). The miss count itself is unchanged: 0.87 hit rate, about 31 misses/token.

**Implementation note.** The first cut took the top k and then dropped cached experts. It advised only 4.4 experts/token, because the top of the ranking is usually already cached. The spike's rule, "first k not cached", is what works.

**k = 2** advises 78 experts/token and loses (×0.9445): GPU busy rises 6 ms, likely from memory-bandwidth contention with the page-cache fills.

**GPU stage profile with lookahead on:** 52.76 ms/token, against 52.65 before (`stages-lookahead.txt`). The predictor runs on the CPU.

## Where the time goes now (ctx 8192, defaults, ms/token)

GPU about 56-60, pread about 13.5, readahead about 9, host about 2.5-3, Engram 0.6. Step about 82-86 ms.

## Next

- **The pread copy of an advised expert is the next lever (about 13.5 ms/token).** It is spec approach 2: advised experts read into staging buffers and handed to cache slots, so a correct guess costs neither the SSD wait nor the copy.
- **GPU side:** routed MoE kernels at about 60 % of bandwidth, and the fused router (tie-order decision pending).
