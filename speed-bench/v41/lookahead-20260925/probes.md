# V4.1 lookahead prefetch A/B, 2026-09-25

Spec: `docs/superpowers/specs/2026-09-25-v41-lookahead-prefetch-design.md`. Plan: `docs/superpowers/plans/2026-09-25-v41-lookahead-prefetch.md`.

Setup:
- Point: ctx 8192, `switch` prompt, 512 teacher-forced tokens, V4.1 alone (oMLX and watchdogs down).
- Binary: `ae8e307` (lookahead on by default, k = 1).
- Each row is one interleaved A/B (A, B, B, A); the terms are per token, means of the two B runs.
- Raw runs: `~/orca/workspaces/ds4-metal-data/v41-lookahead/20260925/`.

| A/B | A t/s | B t/s | Ratio | B readahead | B pread | B GPU | B host | Advised / used per token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| lookahead off vs on (k = 1), auto cache | 11.13 | 12.16 | **1.0921** | 8.79 (A 15.24) | 13.52 (A 13.45) | 57.25 (A 57.37) | 2.54 | 38.9 / 11.9 (30 %) |
| lookahead off vs on (k = 1), 24 GB | 10.53 | 11.21 | 1.0641 | 11.44 (A 17.45) | 17.51 | 57.10 | 3.02 | 39.0 / 16.8 (43 %) |
| k = 1 vs k = 2, auto cache | 12.17 | 11.50 | 0.9445 | 7.24 | 13.98 | **63.16** | 2.46 | 77.5 / 16.1 (21 %) |

- **The advisories take the waiting-for-SSD part (readahead) off the critical path.**
  - At the auto cache, readahead falls 6.5 ms/token.
  - The pread copy from the page cache stays at about 13.5 ms.
  - GPU busy does not move at k = 1. The stage profile gives 52.76 ms/token over 40 tokens, against 52.65 before (`stages-lookahead.txt`).
- **k = 2 advises 78 experts/token.** GPU busy rises 6 ms, likely from memory-bandwidth contention with the page-cache fills, and it loses.
- **No swap.** The decode-window wired peak is at most 48.3 GiB.

## Decision

Lookahead stays on by default with k = 1 (`ds41_la_k` default unchanged). **At ctx 8192 with the shipped defaults, decode is 12.16 t/s: the ≥ 12 t/s target of the streaming work is met.**

The next lever is the ~13.5 ms/token pread copy of the advised experts: approach 2, staging into slots.
