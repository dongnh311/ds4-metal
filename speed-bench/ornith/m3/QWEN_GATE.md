# Qwen3.8 regression gate after Ornith M3

- Branch commit: 01ae3b7 (code head of feature/ornith-m3; later commits are receipts only)
- Command: `speed-bench/qwen-regression/run.sh full` (2026-09-26 22:53-23:07, live stack paused)
- Result: PASS; replies byte-identical yes; paired decode 100.2% of PROD (branch 39.26/38.97 vs
  PROD 40.04/38.04 t/s, run order prod-branch-branch-prod); wired steady 46.12 GiB (peak 47.33);
  needle HIT at 214672 prompt tokens.
- Unit tests that pin the unchanged Qwen3.8 renderer: `./ds4_test --server` and `./ds4_agent_test`, unedited.
- Every M3 commit that touched a shared file also passed `run.sh fast` (after Tasks 2-7).
- Log tail:

```
PASS Qwen MoE specialization type=2 T=8193 mid/down exact, padding intact
./tests/test_qwen4_prefill_pipe
test_qwen4_prefill_pipe: PASS
qwen_gate: PASS
```
