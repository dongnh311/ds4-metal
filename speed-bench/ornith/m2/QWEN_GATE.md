# Qwen3.8 regression gate after Ornith M2

- Branch commit: e9ec342
- Command: `make test-qwen4-kernels test-qwen4-q2 && speed-bench/qwen-regression/run.sh full`
- Result: PASS; replies byte-identical yes; paired decode 39.73/39.70 = 100.1% of PROD; wired 46.07 GiB (baseline 45.80 + 0.5 = 46.30 limit); needle HIT (214672 prompt tokens).
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
./tests/test_qwen4_prefill_pipe
test_qwen4_prefill_pipe: PASS
qwen_gate: PASS
```

Notes: full per-check numbers are in `speed-bench/qwen-regression/last-full/result.json`
(untracked, not committed). Paired decode is the mean of the two `speed_ab.branch`
values over the mean of the two `speed_ab.prod` values (39.7347/39.7019); wired
is `wired.steady_gib` from the same run (46.0657 GiB), against baseline
`speed-bench/qwen-regression/baseline/result.json` `wired.steady_gib` (45.8005 GiB) + 0.5.

## Ornith M2 kernel/graph/session suite

- Command: `make test-qwen35-kernels test-qwen35-graph test-qwen35-mtp test-qwen35-session && tests/ornith/test_loader.sh && tests/ornith/oneshot.sh`
- `qwen35 kernels: ok` — MoE, GDN silu-out, attention prep and the MTP concat kernel all `ok`.
- `qwen35 graph: ok` — MTP graph logits bit-identical to the M1 graph; verify rows at
  31/63/2047/2111/3000 bit-identical to plain decoding; catch-up K-row cosine
  minimum 0.994648 (chunk 64) and 0.990599 (per token), both above the 0.95 gate.
- `qwen35 mtp: ok` — greedy: 150 cycles, 101 accepted, bit-identical to plain
  decoding; forced accepts 20/20; divergent prompt 30 cycles bit-identical;
  context end handled.
- `qwen35 session: ok` — prefill-chunk boundaries 128/129/511/512/513/1024/2049/4096
  replay bit-identical (max|d| 0.00e+00); divergent-prompt and context-full/refuse
  paths behave.
- `ornith loader: ok`.
- `oneshot.sh`: `en_capital`, `code_py`, `vi_hanoi` all `match`.
