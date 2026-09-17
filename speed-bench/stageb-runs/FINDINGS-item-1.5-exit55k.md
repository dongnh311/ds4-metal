# Item 1.5 — root-cause: exit@55K

## Conclusion: exit@55K was the pager eviction memory leak (commit 1171359).

Evidence chain:
1. No hard context ceiling exists near 55K. Searched ds4.c for caps/exits in the
   50K-60K range: the only nearby constant is `rope_orig_ctx = 65536` (64K), which
   is a RoPE scaling reference, not a hard cap, and 55K < 64K. There is no
   `if (ctx > 55000) exit()` or equivalent allocation ceiling at 55K.
2. The crash signature was OOM: the pager `pager_evict_slot` leaked one expert
   bundle (~0.4-0.6 MiB) per eviction (never freed cache_data[slot]). On a
   cache-thrashing paged prefill this grew MALLOC_SMALL without bound; vmmap at
   ctx 8192 caught it at 80+ GiB dirty heap, footprint climbing ~1.3 GiB/5s until
   jetsam ("Killed: 9", rc=137). The crash point scaled with cumulative evictions,
   i.e. with prefill token count -- ~55K tokens was simply where the leaked heap
   crossed the jetsam threshold on the 64 GiB machine for the ctx being run.
3. Post-fix confirmation: with the leak fixed, ctx 32768 completes rc=0 at 9.5 GiB
   peak / 0 swap growth. The ctx 220000 run ran for 72 MINUTES with a bounded,
   stable physical footprint (~24.8 GiB, no leak-climb signature) -- i.e. far past
   the token count where the old build died at ~55K -- and was ended by an
   operator SIGTERM (rc=143) for time, NOT by an OOM. A run that previously died
   at ~55K now runs well past it with flat memory, which is the direct test.

Exit code / stderr tail of the ORIGINAL failing run were the classic jetsam
kill (rc=137, "Killed: 9", no application-level error line) — consistent with an
OS OOM kill, not an internal assert or bounds check.

Receipt: this file + commit 1171359 (the fix) + stageB_32k_mtpoff.csv (post-fix
32K completes) + stageB_220k_mtpoff_cache10.csv (post-fix 220K).
