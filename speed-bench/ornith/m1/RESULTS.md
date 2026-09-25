# Ornith M1 gate 1 (plain inference against llama.cpp)

- Branch commit: a5f819c
- Model: 23G ICE (`speed-bench/ornith/oracle/RESULTS.md`), llama.cpp oracle `tests/ornith/ref/ORACLE.txt`.
- Tolerances (`tests/ornith/tolerance.json`): tol 1.4802, tie 0.4478 (from llama.cpp Metal vs CPU).
- Kernel tests: `make test-qwen35-kernels` ok; session test: `make test-qwen35-session` ok.

| run | result | notes |
|---|---|---|
| default prefill chunk | PASS | `compare-default.txt`; 12/12 ok, 43 steps compared before reference ties, max_delta 0.1795 (math_mul) |
| --prefill-chunk 64 (per-token Q4_K) | PASS | `compare-chunk64.txt`; same lines as default |
| --prefill-chunk 65 (tile GEMM Q4_K) | PASS | `compare-chunk65.txt`; same lines as default |

Per-prompt compared steps and max delta: see the compare files.
Decode speed is not a gate in M1 (M4); for reference, the one-shot log line on `en_capital`: `ds4: Ornith prefill: 95.47 t/s, generation: 75.30 t/s` (`tests/ornith/oneshot.sh`, 5 prompt tokens, 24 generated).

## Coverage beyond the gate

The gate stops each prompt at the reference's first near tie, so it compares few steps (vi_hanoi and long_it tie at step 0).
Outside the gate, on the same dumps, taking selected token ids over every reference step:

- 9 of 12 prompts match llama.cpp on every step (32 or 64). The probable-token max delta over the matched steps is at most 0.18.
- en_story diverges at step 41 and code_rust at step 19. llama.cpp's own top-1/top-2 gap there is 0.037 and 0.013, and ds4 swaps the same two tokens.
- long_it (9270 tokens; the only prompt above 64 tokens, so the only one that reaches the Q4_K tile GEMM in layers 15-39, in the default and chunk-65 runs) diverges at step 0. llama.cpp's gap there is 0.016.
  Default, chunk 64 and chunk 65 all pick the same 16 tokens, within 0.26 of each other on probable tokens. At chunk 65, step 0's top-5 is within 0.21 of llama.cpp.
- The 11 short prompts give byte-identical dumps in all three runs, because a chunk of 64 or more holds each whole prompt.
