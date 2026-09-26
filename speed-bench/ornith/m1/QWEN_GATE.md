# Qwen3.8 regression gate after Ornith M1

- Branch commit: 81b0f2a
- Command: `speed-bench/qwen-regression/run.sh full`
- Result: PASS; replies byte-identical yes; paired decode 38.10/38.73 = 98.4% of PROD; wired 45.99 GiB (baseline 45.80 + 0.5 = 46.30 limit); needle HIT (214672 prompt tokens).
- Log tail:

```
PASS Qwen MoE specialization type=8 T=8192 mid/down exact, padding intact
PASS Qwen MoE specialization type=39 T=8192 mid/down exact, padding intact
PASS Qwen MoE specialization type=2 T=8192 mid/down exact, padding intact
PASS Qwen MoE specialization type=12 T=8193 mid/down exact, padding intact
PASS Qwen MoE specialization type=16 T=8193 mid/down exact, padding intact
PASS Qwen MoE specialization type=10 T=8193 mid/down exact, padding intact
PASS Qwen MoE specialization type=8 T=8193 mid/down exact, padding intact
PASS Qwen MoE specialization type=39 T=8193 mid/down exact, padding intact
PASS Qwen MoE specialization type=2 T=8193 mid/down exact, padding intact
qwen_gate: PASS
```

Notes: run 1 (branch commit 0acea18) crashed at kernel-test build with
`ModuleNotFoundError: No module named 'machine'` because `speed-bench/lib`
(machine.py, wired.py) had not been imported onto this branch. Fixed by
`git checkout develop -- speed-bench/lib` (byte-identical, commit 81b0f2a),
then this run (run 2) passed cleanly. Full per-check numbers are in
`speed-bench/qwen-regression/last-full/result.json` (untracked, not committed).
