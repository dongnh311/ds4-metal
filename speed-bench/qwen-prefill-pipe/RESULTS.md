# qwen4 prefill residency + staging pipe: results

Date: 2026-09-25. Branch `feature/qwen4-prefill-pipe` at `90a5c80`. Model unc48L (PROD registry flags:
`Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q4KDownPad768-DenseQ4Kselimat-MTP.gguf` + PLE-Q4_1, `-c 262144
--prefill-chunk 2048 --mtp --ssd-streaming --ssd-streaming-cache-experts 6GB`, `DS4_QWEN4_KV_GROW=1`).

## Knob

`DS4_QWEN4_PREFILL_MODE=off|safe|max` (engine default `off`). `safe` applies the residency scope plus the
double-buffered streaming-expert staging pipe (F_NOCACHE reads) to every prefill chunk except the prompt's
last chunk, which still runs the pre-existing union path with no residency. `max` applies residency + pipe
to every chunk, including the last. See `qwen4_prefill_policy()` in `ds4.c` and the spec's policy table for
the source of truth.

Chosen for PROD: **`off`** (rule: `max` if decode >= 97% of PROD on both long-prompt requests AND prefill >=
+10% on the long request; else `safe` under the same rule; else `off`). Neither `max` nor `safe` passed the
long-prompt paired A/B gate (`qwen_gate.py longab`) run in this task — see below.

## Exactness

- `cli_exact.sh` chat/p5k/p36k/p134k, off vs safe vs max (from Task 4's GPU run, commit `7f4ed87`, unchanged
  since — no engine code has been touched by Task 5 or this task):
  ```
  chat-off prefill 15.19 gen 41.07 md5 61f298f6c6c86e0e3585cd2101bba3b1
  chat-safe prefill 91.29 gen 41.76 md5 61f298f6c6c86e0e3585cd2101bba3b1
  chat-max prefill 55.46 gen 40.62 md5 61f298f6c6c86e0e3585cd2101bba3b1
  PASS chat
  p5k-off prefill 490.42 gen 37.39 md5 9e43f0ee7ad352a45ce43f78b3f1b8f7
  p5k-safe prefill 560.67 gen 36.71 md5 9e43f0ee7ad352a45ce43f78b3f1b8f7
  p5k-max prefill 544.32 gen 37.36 md5 9e43f0ee7ad352a45ce43f78b3f1b8f7
  PASS p5k
  p36k-off prefill 548.18 gen 39.81 md5 77edb7ada254f230335a188a65dc24af
  p36k-safe prefill 555.04 gen 38.35 md5 77edb7ada254f230335a188a65dc24af
  p36k-max prefill 553.47 gen 36.34 md5 77edb7ada254f230335a188a65dc24af
  PASS p36k
  p134k-off prefill 489.59 gen 35.40 md5 e6f7158ebf8e5be43c872c23d238b29a
  p134k-safe prefill 493.60 gen 33.12 md5 e6f7158ebf8e5be43c872c23d238b29a
  p134k-max prefill 509.59 gen 30.94 md5 e6f7158ebf8e5be43c872c23d238b29a
  PASS p134k
  cli_exact exit 0
  ```
  Also confirmed again in this task's Step 4 GPU-idle runs (p36k, ngen=16): off/safe/max md5s all
  `2414077749125b28779737b3c27374bf`.
- `qwen-regression` fast tier, this task's Step 1, commit `90a5c80`, `make -j10 ds4 ds4-server` clean (no
  `error`/`warning:` lines):
  - `DS4_QWEN4_PREFILL_MODE=safe speed-bench/qwen-regression/run.sh fast` -> byte-identical PROD replies
    (vi/code prompts) + all unit kernels (`test-qwen4-kernels`, `test-qwen4-q2`) + `test_qwen4_prefill_pipe:
    PASS` -> **`qwen_gate: PASS`**.
  - `DS4_QWEN4_PREFILL_MODE=max speed-bench/qwen-regression/run.sh fast` -> same coverage -> **`qwen_gate:
    PASS`**.
- `off` ignores the variable by construction (`qwen4_prefill_policy()` only changes behavior for `safe`/`max`);
  the PROD side of every A/B below is byte-identical to `off` by definition (PROD predates this feature).

## Long-prompt paired A/B (qwen_gate longab)

Measured at commit `90a5c80` (this file committed later, at `1e49903`), `--out /tmp/claude-501/qpp-longab-{max,safe}`, interleaved PROD/branch/branch/PROD servers,
long request = ~110K filler chars, medium = ~17K, 300 max tokens each.

| mode | request | PROD prefill (t/s) | branch prefill (t/s) | x | PROD decode (t/s) | branch decode (t/s) | x | verdict |
|---|---|---|---|---|---|---|---|---|
| max | long | 498.9 | 539.4 | 1.081 | 38.25 | 35.88 | 0.938 | FAIL (decode < 0.97; prefill < 1.10) |
| max | medium | 338.2 | 450.3 | 1.331 | 38.57 | 35.73 | 0.927 | FAIL (decode < 0.97) |
| safe | long | 501.3 | 526.2 | 1.050 | 37.18 | 36.64 | 0.986 | FAIL (prefill < 1.10; decode OK) |
| safe | medium | 327.3 | 353.5 | 1.080 | 38.31 | 37.22 | 0.971 | PASS (decode OK; medium has no prefill-gain requirement) |

`qwen_gate.py longab --prefill-mode max`: `qwen_gate: FAIL` (long decode 0.938, medium decode 0.927, long
prefill 1.081 < 1.10).
`qwen_gate.py longab --prefill-mode safe`: `qwen_gate: FAIL` (long prefill 1.050 < 1.10; both decode ratios
pass the 0.97 floor).

Applying the rule: `max` fails on decode and on the +10% prefill-gain requirement. `safe` clears the decode
floor on both requests but misses the +10% prefill-gain requirement on the long request (1.050 vs required
1.10). Neither mode qualifies, so **CHOSEN = off**. Step 3 (full tier with `DS4_QWEN4_PREFILL_MODE=$CHOSEN`)
is skipped per the brief's rule for an `off` outcome — this is a valid, reportable result, not a failure of
this task. `off` is byte-identical to PROD (baseline data below) and needs no further gate.

## GPU idle at 36K (DS4_METAL_GPU_IDLE)

Commit `90a5c80`, `speed-bench/qwen-prefill-pipe/cli_run.sh $O idle-$m $O/prompts/p36k.txt $m 16`,
`DS4_METAL_GPU_IDLE=1`, single run per mode, md5 identical across all three
(`2414077749125b28779737b3c27374bf`, confirming exactness again):

| mode | prefill t/s | gen t/s (ngen=16) | span ms | busy ms | idle % | launch ms (waits) |
|---|---|---|---|---|---|---|
| off | 511.06 | 21.22 | 71926.3 | 50907.1 | 29.2% | 8946.5 (299) |
| safe | 542.44 | 20.28 | 67844.3 | 51205.2 | 24.5% | 572.7 (27) |
| max | 530.61 | 18.54 | 69413.2 | 55904.6 | 19.5% | 6.3 (11) |

The launch-latency totals are summed over different wait counts per mode (off 299, safe 27, max 11) and are
not directly comparable; compare mean latency per wait instead: off 8946.5/299 ≈ 29.9 ms/wait, safe
572.7/27 ≈ 21.2 ms/wait, max 6.3/11 ≈ 0.57 ms/wait. `safe`'s much smaller total mostly reflects that it
waits far less often (the residency scope removes the wait on every chunk but the last, per the Knob
section above), not a large per-wait improvement over `off`. `max` removes launch latency essentially
completely, both in total and per wait, matching the brief's expected under-1-s band; this does not
translate into the required +10%/97% server-side A/B result once decode-side page-cache/read contention
(see Task 4's report) is accounted for.

## Full tier

Skipped: `CHOSEN = off`, and `off` requires no gate beyond the exactness already shown above (it is
byte-identical to PROD by construction). The PROD baseline itself (`speed-bench/qwen-regression/baseline`,
recorded before this feature existed) already carries the full-tier numbers that describe current PROD/`off`
behavior: decode 42.80 t/s median, steady wired 45.80 GiB, needle hit at 214,672 tokens.

## Why `off`: background from Tasks 3-4 (GPU sections, not re-run here)

- Task 3 (commit `c7d1c1b`) originally read `chat-off 15.08 -> chat-safe ~86 t/s` (cli_exact) as a dramatic
  residency win. That reading is wrong: `chat` is a single prompt chunk, which is always the prompt's LAST
  chunk, and `safe` explicitly excludes the last chunk (see the Knob section) — `safe` does no residency or
  pipe work on `chat` at all. The jump is a run-order effect: `cli_exact.sh` always runs `off` first against
  a cold page cache, then `safe` and `max` against a warm one, so `off`'s chat number is depressed and
  `safe`/`max`'s are inflated by the warm cache alone. Task 3's own numbers confirm it: `chat-safe 86.10`
  and `chat-max 85.97` are within noise of each other even though `max` (unlike `safe`) does apply residency
  and the pipe to `chat`'s one chunk — if residency explained the win, `safe` and `max` would not land on
  the same number. `cli_exact.sh`'s speed columns are therefore only an informal sanity signal, biased by
  this fixed run order; the actual pass/fail check there is md5 equality, not the t/s numbers. On the 36K
  prompt (where `safe` does have non-last chunks to act on), residency gave only +3.3% (cli_exact) to +7.0%
  (GPU-idle run) — both below the spec's informal +10% expectation, flagged as a concern at the time.
- Task 4 (commit `7f4ed87`): the staging pipe (`max`) reached 280+ pipe-served layers per run (287/288) and
  cut GPU idle from 28.5% (off) to 14.4% (max) in a profiled run, but prefill gain was only +1% (cli_exact,
  553.5 vs 548.2 t/s) to +11% (profiled pair, 528.7 vs 476.5 t/s) — the whole-layer pipe reads are SSD-bound
  (~11 GiB/s, page-cache miss) versus the `off` union path's selected-expert reads, which mostly hit the page
  cache on these text prompts (~27 GiB/s). CLI decode after `max` was 8.7-12.6% lower than `off` in single
  noisy runs; Task 4 explicitly deferred the mode decision to this task's server-side paired A/B.
- This task's paired A/B (above) confirms that pattern under controlled interleaved server conditions: decode
  drops below the 97% floor for `max` (worst case 0.927), and neither mode reaches the required +10% prefill
  gain on the long request when measured against an interleaved PROD baseline rather than a single `off` run.

## How to reproduce

```bash
# Task 3/4 CLI exactness + launch-latency probes
speed-bench/qwen-prefill-pipe/cli_exact.sh OUT_DIR chat p5k p36k p134k
O=OUT_DIR; python3 speed-bench/qwen-prefill-pipe/make_prompts.py $O/prompts
for m in off safe max; do
  DS4_METAL_GPU_IDLE=1 speed-bench/qwen-prefill-pipe/cli_run.sh $O idle-$m $O/prompts/p36k.txt $m 16
  grep -a 'gpu run:' $O/idle-$m.err
done

# Task 6: fast-tier byte-identical + unit gate for safe/max
make -j10 ds4 ds4-server
DS4_QWEN4_PREFILL_MODE=safe speed-bench/qwen-regression/run.sh fast
DS4_QWEN4_PREFILL_MODE=max speed-bench/qwen-regression/run.sh fast

# Task 6: long-prompt paired A/B (decides the default)
python3 speed-bench/qwen-regression/qwen_gate.py longab --bin . --out OUT/max --prefill-mode max
python3 speed-bench/qwen-regression/qwen_gate.py longab --bin . --out OUT/safe --prefill-mode safe

# Task 6: full tier with the chosen mode (skipped here since CHOSEN=off)
DS4_QWEN4_PREFILL_MODE=$CHOSEN speed-bench/qwen-regression/run.sh full
```

All GPU commands need the machine free (`pgrep -fl 'ds4-server|/ds4 '` empty) and the local AI-gateway
services stopped beforehand, then restored afterward:
```bash
launchctl unload ~/Library/LaunchAgents/dev.dongnh.ai-proxy.plist
launchctl unload ~/Library/LaunchAgents/dev.dongnh.gateway-watchdog.plist
P=$(pgrep -x omlx-server); [ -n "$P" ] && kill -TERM $P
# ... run the gates above ...
launchctl load ~/Library/LaunchAgents/dev.dongnh.ai-proxy.plist
launchctl load ~/Library/LaunchAgents/dev.dongnh.gateway-watchdog.plist
```
