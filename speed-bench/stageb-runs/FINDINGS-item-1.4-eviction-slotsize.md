# Item 1.4 — eviction A/B (LRU vs scored) + expert-cache slot-sizing fix

Re-validated on the FIXED pager (all prior 1.x bugs resolved). Two questions: (a) does
the eviction policy (E1-LRU vs E2-scored) change hit-rate/read-GB; (b) does the 1.7
slot-array finding (slots sized by the largest bundle wastes budget) hold, and does
fixing it help. One commit. Gate bit-identical (cache capacity/policy never change the
math). All runs: ctx 4096, gen 128, prose, --ssd-streaming, default cache 1.86 GiB,
diskwrites=0.

## (a) Eviction A/B — NO discrimination at the tested frontier
| policy | hits/misses | hit_rate | read MB | gen_steady |
|--------|-------------|----------|---------|-----------|
| scored (before fix) | 119702/120868 | 49.76% | 57246 | 5.53 |
| lru    (before fix) | 119687/120883 | 49.75% | 57253 | 5.57 |
| scored (after fix)  | 128794/111776 | 53.54% | 52939 | 6.05 |
| lru    (after fix)  | 128794/111776 | 53.54% | 52939 | 6.13 |

LRU and scored are statistically identical before the fix and **byte-identical after**
(same hits/misses to the digit). At this frontier the working set ≫ cache (~50% hit)
and the access pattern is recency-dominated, so the scored policy reduces to the same
evictions as LRU. **Conclusion: policy is not the lever here — keep the E2-scored
default (no code change to the policy).** A higher frontier (32K+) could in principle
discriminate but the exact equality here makes that unlikely to change the decision;
left as a follow-on rather than a blocker.

## (b) Slot-sizing fix — a clean win on the SAME RAM budget
The cache is capped by MIN(slot_count, byte_budget). Slots were sized by the LARGEST
bundle (`bytes / max_bundle`, down = 645120 B), but 2/3 of bundles are the smaller
gate/up tensors (422400 B), so the slot COUNT bound before the byte budget — leaving
~20-25% of the granted RAM unusable (the code comment at reserve_slot admitted this).
Fix: size the slot array by the SMALLEST bundle (`bytes / min_bundle`) so the byte
budget is the sole cap; `max_bundle` stays as the "budget too small for one bundle"
floor. Extra metadata ≈ 40 B/slot (negligible).

Result (1.86 GiB budget, scored):
```
slots        3094 (by 645120 max)  ->  4726 (by 422400 min)   +53%
hit_rate     49.76%  ->  53.54%     (+3.78 pp)
read MB      57246   ->  52939      (-7.5%)
read-GB/tok  0.01324 ->  0.01224    (-7.5%)
gen_steady   5.53    ->  6.05 tps   (+9.4%)
```
The full byte budget now binds instead of the slot count → ~21% more experts cached →
higher hit rate, less SSD read, faster decode — for the same RAM. Bit-identical:
run-logit-gate.sh 512 512 max_abs_diff=0.0, 0/248320, GATE PASS (cache capacity is not
part of the math). diskwrites=0.

## Scope / honesty
- The gains scale with how far the slot count was binding below the byte budget (~25%
  here); on a larger budget where the byte cap already binds first, the fix is a no-op
  (never harmful). It compounds with the item-(1) adaptive budget: a larger swap-safe
  grant now also gets fully used.
- Eviction policy left at the E2-scored default; no evidence to change it at measured
  frontiers.

## Files / receipts
ds4_expert_pager.c (enable_cache: min_bundle slot sizing + log). Receipts:
evict-1.4/evict_{before,after}_{scored,lru}.csv, gate_after_slotfix.log.
