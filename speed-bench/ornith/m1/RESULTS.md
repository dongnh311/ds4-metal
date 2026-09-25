# Ornith M1 gate 1 (plain inference against llama.cpp)

- Branch commit: 8f878e2 (ds4 engine code as of a5f819c; the gate uses this commit's tie-step check and 14 prompts)
- Model: 23G ICE (`speed-bench/ornith/oracle/RESULTS.md`), llama.cpp oracle `tests/ornith/ref/ORACLE.txt`.
- Tolerances (`tests/ornith/tolerance.json`): tol 1.4802, tie 0.4478 (from llama.cpp Metal vs CPU).
- Kernel tests: `make test-qwen35-kernels` ok; session test: `make test-qwen35-session` ok.

| run | result | notes |
|---|---|---|
| default prefill chunk | PASS | `compare-default.txt`; 14/14 ok, 73 steps compared, max_delta 0.1795 (math_mul) |
| --prefill-chunk 64 (per-token Q4_K) | PASS | `compare-chunk64.txt`; 14/14 ok, 73 steps, max_delta 0.1795 (math_mul) |
| --prefill-chunk 65 (tile GEMM Q4_K) | PASS | `compare-chunk65.txt`; 14/14 ok, 73 steps, max_delta 0.2003 (code_c_pack, at its tie step) |

Per-prompt compared steps and max delta: see the compare files.
Decode speed is not a gate in M1 (M4); for reference, the one-shot log line on `en_capital`: `ds4: Ornith prefill: 95.47 t/s, generation: 75.30 t/s` (`tests/ornith/oneshot.sh`, 5 prompt tokens, 24 generated).

## Coverage beyond the gate

The gate compares each prompt up to and including the reference's first near tie. At the tie step it checks probable-token log-probs but not the selection.
Compared steps before a tie: 73 in total. vi_hanoi and long_it tie at step 0; en_contributing ties at 14 and code_c_pack at 16.

Tile GEMM coverage. Q4_K routed experts sit in layers 15-39, and the tile GEMM is used when a prefill chunk has more than 64 tokens.

| prompt | tokens | default chunk (2048) | chunk 65 |
|---|---|---|---|
| en_contributing | 324 | 1 tile chunk | 4 tile chunks + a 64-token per-token tail |
| code_c_pack | 405 | 1 tile chunk | 6 tile chunks + a 15-token per-token tail |
| long_it | 9270 | 5 tile chunks | 142 tile chunks + a 40-token per-token tail |

The chunk-64 run uses no tile GEMM at all.

Outside the gate, on the same dumps, taking selected token ids over every reference step:

- 10 of 14 prompts match llama.cpp on every step in the default run (32 or 64). This includes code_c_pack's 32 steps through its tie, with a probable-token max delta of at most 0.37.
- en_story (step 41), code_rust (step 19) and en_contributing (step 14) diverge where llama.cpp's own top-1/top-2 gap is 0.037, 0.013 and 0.072. In each case ds4 swaps near-equal tokens.
- long_it diverges at step 0, where llama.cpp's gap is 0.016.
- In the chunk-65 run, en_contributing and code_c_pack match all 32 steps.
- The 11 short prompts give byte-identical dumps in all three runs, because a chunk of 64 or more holds each whole prompt.
