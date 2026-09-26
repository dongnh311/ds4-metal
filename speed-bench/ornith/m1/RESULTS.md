# Ornith M1 gate 1 (plain inference against llama.cpp)

- Branch commits: gate tooling fce5091, engine code 56d7628 (final fix wave, ruling 15). The first gate receipt was on 8f878e2/0acea18.
- Model: 23G ICE (`speed-bench/ornith/oracle/RESULTS.md`), llama.cpp oracle `tests/ornith/ref/ORACLE.txt`.
- Tolerances (`tests/ornith/tolerance.json`): tol 1.4802, tie 0.4478 (from llama.cpp Metal vs CPU, 10 prompts). Recalibrated after the two-way probable check: tol and tie unchanged.
- Kernel tests: `make test-qwen35-kernels test-qwen4-kernels test-qwen4-q2` ok. `make test-qwen35-session` ok. `tests/ornith/test_loader.sh` ok. `tests/ornith/oneshot.sh` ok.

| run | result | notes |
|---|---|---|
| default prefill chunk (2048) | PASS | `compare-default.txt`; 13/13 ok, 104 steps compared, 144 probable-token pairs checked, max_delta 0.1795 (math_mul); long_copy 0.0354 |
| --prefill-chunk 64 (per-token Q4_K) | PASS | `compare-chunk64.txt`; 13/13 ok, 104 compared, 144 checked, max_delta 0.1795 (math_mul); long_copy 0.0256 |
| --prefill-chunk 65 (tile GEMM Q4_K) | PASS | `compare-chunk65.txt`; 13/13 ok, 104 compared, 144 checked, max_delta 0.2003 (code_c_pack, at its tie step); long_copy 0.0241 |

Per-prompt compared steps, checked pairs and max delta: see the compare files.

The gate compares each prompt up to and including the reference's first near tie. At every compared step, a token that is probable (log-prob >= -2.0) on either side must be in both top-20 lists. The reference's probable tokens must agree within tol, including at the tie step. A prompt that checks no probable-token pair fails as `FAIL vacuous`.

Decode speed is not a gate in M1 (it is gated in M4). For reference, the one-shot log line on `en_capital` (`tests/ornith/oneshot.sh`, 5 prompt tokens, 24 generated) was `ds4: Ornith prefill: 99.64 t/s, generation: 75.11 t/s`.

## Prompt set changes (ruling 15)

- `long_it` is replaced by `long_copy`. long_it (9,270 tokens) tied at step 0 with a best reference log-prob of -2.715, so it checked nothing.
  - long_copy is the first 31,982 characters of `speed-bench/promessi_sposi.txt`, a blank line, then its first 410 characters again: 9,371 tokens.
  - The continuation copies the opening from about 9K tokens back. Its reference has no near tie in any of the 32 steps (smallest top-1/top-2 gap 2.706, at step 8) and a probable token at every step.
  - In all three runs, ds4 selects llama.cpp's token at all 32 steps: 32 compared, 32 checked.
- `list_tips` is dropped. Its reference top-1 is -2.44 at step 0 and -2.12 at its step-1 tie, so it checks nothing under the vacuous rule. Its Metal-vs-CPU pair did not set either calibrated value.

## long_copy reference: recorded one token per ubatch

long_copy's reference is recorded by llama-server with `-ub 1` (`"llama_ubatch": 1` in `prompts.json`). The other 12 references use the default batch, and re-recording left them byte-identical.

Step 29 is the one contested step. The model has to choose between the copied archaic " de" and the modern-Italian " di". With the default-batch reference, the chunk-64 run failed there (delta 1.5766 > 1.4802). The default and chunk-65 runs passed at 1.0749 and 1.2139.

Log-prob of " di" at step 29 (" de" is top-1 everywhere):

| source | " di" | " de" |
|---|---|---|
| llama.cpp Metal, default ubatch 512 (the first reference) | -1.453 | -0.274 |
| llama.cpp Metal, `-ub 64` | -1.485 | -0.265 |
| llama.cpp Metal, flash attention off | -1.462 | -0.271 |
| llama.cpp CPU (`-ngl 0 --device none`) | -2.455 | -0.092 |
| llama.cpp Metal, `-ub 1` (the reference now) | -3.056 | -0.050 |
| ds4 default chunk (2048) | -2.528 | -0.086 |
| ds4 `--prefill-chunk 65` | -2.667 | -0.075 |
| ds4 `--prefill-chunk 64` | -3.030 | -0.052 |

- At this step, llama.cpp's own batched Metal prefill disagrees with its CPU backend by 1.00. The largest Metal-vs-CPU gap on the calibration prompts is 0.49.
- llama.cpp's `-ub 1` Metal run agrees with its CPU run within 0.041 over the whole comparison.
- Compared with the CPU run, the three ds4 runs are within 0.027-0.043.
- Compared with the `-ub 1` reference, they are within 0.024-0.035 on every step.

The batched llama.cpp prefill is the outlier, not ds4. The per-token reference is the llama.cpp output that agrees with the calibration backend. Tolerances are unchanged.

## Coverage beyond the gate

- Compared steps before a tie: 104 in total.
- vi_hanoi ties at step 0; its tie step still checks 3 probable tokens. en_contributing ties at step 14 and code_c_pack at step 16. long_copy has no tie.

Tile GEMM coverage. Q4_K routed experts sit in layers 15-39. The tile GEMM is used when a prefill chunk has more than 64 tokens. Q5_K layers (0-14) use the per-token row kernels at every prefill size (see the deviation below).

| prompt | tokens | default chunk (2048) | chunk 65 |
|---|---|---|---|
| en_contributing | 324 | 1 tile chunk | 4 tile chunks + a 64-token per-token tail |
| code_c_pack | 405 | 1 tile chunk | 6 tile chunks + a 15-token per-token tail |
| long_copy | 9371 | 5 tile chunks (4 x 2048 + 1179) | 144 tile chunks + an 11-token per-token tail |

The chunk-64 run uses no tile GEMM at all.

Outside the gate, taking selected token ids over every reference step on the same dumps:

- Default run: 10 of 13 prompts match llama.cpp on every step. The exceptions are en_story (step 41), code_rust (step 19) and en_contributing (step 14). There, llama.cpp's own top-1/top-2 gap is 0.037, 0.013 and 0.072, and ds4 swaps near-equal tokens.
- Chunk-65 run: 11 of 13 match on every step. en_contributing matches all 32 steps.
- Chunk-64 run: 9 of 13 match on every step. code_c_pack also swaps at its step-16 tie (gap 0.043).
- The 10 short prompts give byte-identical dumps in all three runs, because a chunk of 64 or more holds each whole prompt.

## Deviation: Q5_K prefill

M1 ships the Q5_K row kernels only. Q5_K layers use the per-token row kernels at every prefill size; Q4_K layers switch to the tile GEMM above 64 tokens. The tiled Q5_K GEMM is deferred to M4, where prefill speed is measured. The spec, §4 (MoE), records the same deviation.
