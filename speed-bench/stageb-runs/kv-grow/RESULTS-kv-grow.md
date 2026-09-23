# qwen4 grow-on-demand KV: byte-exact gates + performance A/B

Commit tested: `88f9238566151d4f50dc03a502e7d3112fb20228` (branch `kv-grow`)

Machine: Apple M5 Pro, 64 GiB RAM, Darwin 25.6.0 (mac16, arm64).

Binaries: `./ds4`, `./ds4-server`, `./ds4_test` built from the worktree root
(`make -j10 ds4 ds4-server ds4_test`).

Model: `Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q4KDownPad768-DenseQ4Kselimat-MTP.gguf`
+ `Qwen3.8-Flash-Next-PLE-Q4_1.gguf`, SSD streaming, `--ssd-streaming-cache-experts 6GB`,
`--mtp`, `--prefill-chunk 2048`, `--temp 0`.

Common env for every run:

```
DS4_QWEN4_STREAM_FULL_LAYERS=32 DS4_QWEN4_PLE_PREFETCH_FULL=0
DS4_QWEN4_MTP_DRAFT_VOCAB=/Users/dongnh/.local/share/ai-gateway/ds4-models/Qwen3.8-Flash-Next-draft-vocab-vi-en-code-64k.txt
```

Off runs omit `DS4_QWEN4_KV_GROW`; on runs set `DS4_QWEN4_KV_GROW=1`. Full command
shape (CLI):

```
./ds4 --metal -m <model> --ple <ple> --prefill-chunk 2048 --mtp --ssd-streaming \
  --ssd-streaming-cache-experts 6GB --temp 0 -c 262144 -n N <prompt>
```

Wrapper script used for every CLI run (background `vm_stat` sampler every 0.5s,
peak "Pages wired down" recorded): `.superpowers/sdd/2026-09-23-qwen4-kv-grow-on-demand/task5/run_ds4.sh`.
Server wrapper: `.superpowers/sdd/2026-09-23-qwen4-kv-grow-on-demand/task5/run_server.sh`.
Chat client (urllib): `.superpowers/sdd/2026-09-23-qwen4-kv-grow-on-demand/task5/chat_client.py`.
Prompt generator (ds4.c slice + a needle passphrase comment + closing question):
`.superpowers/sdd/2026-09-23-qwen4-kv-grow-on-demand/task5/make_prompt.py`.

Prompt files used (token counts measured with `--dump-tokens`):

| file | tokens | source |
|---|---|---|
| short | 82 | literal `-p` string |
| 31K | 31,050 | built this task (`prompt_31k.txt`) |
| 40K | 39,903 | built this task (`prompt_40k.txt`) |
| 100K | 98,106 | built this task (`prompt_100k.txt`), server shrink gate only |
| 37K (32k-prose) | 36,798 | reused, `long_prompt_32k_prose.txt` |
| 135K (128k-prose) | 135,151 | reused, `long_prompt_128k_prose.txt` |
| 256K (256k-prose) | 256,145 | reused, `long_prompt_256k_prose.txt` |

## Byte-exact gates (CLI)

All commands run at `-c 262144`. `cmp` compares stdout between the flag-off and
flag-on run of the same case.

| case | prompt | `-n` | extra env | cmp | capacity log lines (flag on) |
|---|---|---|---|---|---|
| short | literal hash-map prompt | 800 | - | **identical** | none |
| reserve | 40K prompt | 400 | - | **identical** | `qwen4 KV capacity 32768 -> 65536 rows (+0.24 GiB)` (one line, at sync) |
| mid-decode — **grows at sync (plan defect)** | 31K prompt | 2048 | `DS4_QWEN4_KV_INIT_CAP=32768` | identical | `qwen4 KV capacity 32768 -> 65536 rows (+0.24 GiB)` — this fired at prefill sync (31,050 + 2048 reserve request already exceeds 32768), not mid-decode as the plan intended. Controller ruling: replace the case (kept below as the corrected run; this row is left for the record). |
| mid-decode (corrected) | 31K prompt | 2048 | `DS4_QWEN4_KV_RESERVE=64` (both off and on) | **identical** | Exactly one line, printed **during decode** (after the prefill-done marker, at generated-token count ≈ 1718 = 32768 − 31050): `qwen4 KV capacity 32768 -> 65536 rows (+0.24 GiB)`. No line at sync — reserve=64 keeps the sync target (31,114 rows) inside the default 32,768 init cap, so growth only happens once decode crosses the boundary, matching the controller's expected behavior exactly. |
| full | 256K prompt | 1000 | - | **identical** | `qwen4 KV capacity 32768 -> 262144 rows (+1.69 GiB)` (one line, at sync — the 256K prompt straight away needs the full context) |

All five gates (four intended cases plus the superseded mid-decode row) produced
byte-identical stdout between flag off and flag on. No case failed.

Raw logs: `task5/gate-{short,reserve,mid,mid2,full}-{off,on}.{out,err}` (mid-decode
sync-defect run renamed to `gate-mid-{off,on}-DEFECT-syncgrow.{out,err}`; the
corrected reserve=64 run is `gate-mid2-{off,on}.{out,err}`).

## Extra coverage: KV layouts other than the default half cache

Same unit-test invocation as Task 3, run once per layout on top of `--qwen-kv-grow`:

```
DS4_TEST_MODEL=<model> DS4_TEST_PLE=<ple> DS4_TEST_GLM_MTP=1 DS4_TEST_SSD_STREAMING=1 \
DS4_TEST_SSD_STREAMING_CACHE_GB=6 DS4_QWEN4_STREAM_FULL_LAYERS=32 DS4_QWEN4_PLE_PREFETCH_FULL=0 \
<MODE> ./ds4_test --qwen-kv-grow
```

| layout | MODE | exit | result | capacity log lines |
|---|---|---|---|---|
| half cache (baseline, `DS4_N_HEAD_DIM==256` default) | `DS4_QWEN4_KV_Q4=0 DS4_QWEN4_KV_FP8=0` | 0 | `qwen-kv-grow: OK` | `4096->8192(+0.10GiB)`, `4096->13312(+0.24GiB)`, `4096->8192(+0.10GiB)`, `4096->10240(+0.16GiB)`, `10240->4096(-0.16GiB)` (shrink), `4096->8192(+0.10GiB)` x2 |
| FP8 in-kernel KV | `DS4_QWEN4_KV_Q4=0 DS4_QWEN4_KV_FP8=1` | 0 | `qwen-kv-grow: OK` | `4096->8192(+0.06GiB)`, `4096->13312(+0.13GiB)`, `4096->8192(+0.06GiB)`, `4096->10240(+0.08GiB)`, `10240->4096(-0.08GiB)`, `4096->8192(+0.06GiB)` x2 |
| raw indexer keys kept per position | `DS4_QWEN4_IK_RING=0` | 0 | `qwen-kv-grow: OK` | identical byte-size pattern to the FP8 run above (indexer-key ring size doesn't change the qwen4 KV row-growth byte accounting) |

All three alternate layouts pass `--qwen-kv-grow` cleanly, growing/shrinking rows
exactly as in the default half-cache layout, only differing in bytes-per-row.
Logs: `task5/unit-half.log`, `task5/unit-fp8.log`, `task5/unit-ikring.log`.

## Server gates

`./ds4-server` bound to `127.0.0.1:18397`, KV disk dir under
`.superpowers/sdd/2026-09-23-qwen4-kv-grow-on-demand/task5/kv-{off,on,shrink,shrink-off}`,
`--kv-disk-space-mb 32768 --kv-cache-cold-max-tokens 262144`. Chat client posts
`POST /v1/chat/completions` with `{"model":"ds4","messages":[...],"max_tokens":N,
"temperature":0,"stream":false,"think":false}` (`think:false` needed — the model
defaults to high-effort thinking mode server-side, which otherwise burns the
small `max_tokens` budgets on reasoning tokens and returns empty `content`).

### 1. Checkpoint restore (byte-identical across flag off/on)

For each of flag-off and flag-on: start server, send the 40K prompt
(`prompt_40k.txt`, 39,896 rendered tokens, `max_tokens=200`) as turn 1, stop the
server, restart it pointed at the same KV dir, send the same conversation
(assistant turn 1 reply appended) plus a new user turn ("Now, in one sentence,
restate the passphrase you found.", `max_tokens=200`) as turn 2.

- Turn 1 reply: **byte-identical** flag off vs flag on (`cmp` rc=0).
- Turn 2 reply (post-restart, post-restore): **byte-identical** flag off vs
  flag on (`cmp` rc=0): `The deployment passphrase is INDIGO-WALNUT-ELEVEN-CASCADE.`
- Restore confirmed by log line on both restarts:
  `ds4-server: kv cache hit text tokens=40056 text=108771 quant=2 key=token-text
  load=6x.x ms file=.../f6d5ccc50bc34c280145851734f6b9229048b26e.kv`
- Flag-on restart additionally shows the KV grow at restore:
  `ds4: qwen4 KV capacity 32768 -> 65536 rows (+0.24 GiB)`.

Logs: `task5/server-off-r1.err`, `server-off-r2.err`, `server-on-r1.err`,
`server-on-r2.err`; replies in `server-{off,on}-turn{1,2}.txt`.

### 2. Shrink (flag on only, then compared against a flag-off baseline)

Flag-on server: sent the 100K prompt (`prompt_100k.txt`, 98,059 rendered tokens,
`max_tokens=100`), then an unrelated short request ("What is the capital of
France?", `max_tokens=50`).

- Capacity grew during the 100K request: `32768 -> 100352` then (at the final
  prefill chunk) `100352 -> 262144` (full context — the server appears to size
  the cache to the full `-c` once the live session is this large, independent
  of the shrink-gate logic under test).
- Shrink fired after the unrelated request completed (session reset):
  `qwen4 KV capacity 262144 -> 43008 rows (-1.62 GiB)`, then a small regrow for
  the short prompt processing (`43008 -> 86016`), then the final settle-to-floor
  shrink `86016 -> 32768 rows (-0.39 GiB)`.
- System wired memory ("Pages wired down" × 16384): **3,154,440 pages
  (48.09 GiB)** right after the 100K request → **3,074,221 pages (46.91 GiB)**
  right after the short request/shrink — a drop of **80,219 pages ≈ 1.22 GiB**,
  consistent with "about 1 GiB" expected.
- Short reply (flag on, post-shrink): `The capital of France is **Paris**.`
  Flag-off baseline (fresh server, same short request, no prior 100K context):
  same reply, **byte-identical** (`cmp` rc=0).

Logs: `task5/server-shrink.err`, `server-shrink-off.err`; vm samples in
`vm-before-100k.txt`, `vm-after-100k.txt`, `vm-after-short.txt`; replies in
`shrink-req2-{on,off}.txt`.

All ds4-server processes started for this task were stopped after use and
confirmed clean with `pgrep -fl ds4-server`.

## Performance A/B

`-c 262144`, flag off vs on. Short/37K/135K alternate off, on, off, on (two
samples each side) because the machine drifts; 256K reuses the byte-exact gate
pair only (one off, one on), as instructed. Peak wired memory sampled every
0.5s with `vm_stat` during each run ("Pages wired down" × 16384).

### Short (800 generated tokens, 82-token prompt)

| run | prefill t/s | decode t/s | peak wired GiB |
|---|---|---|---|
| off r1 | 60.77 | 43.94 | 48.77 |
| on r1 | 72.40 | 45.89 | 46.96 |
| off r2 | 70.63 | 43.69 | 48.67 |
| on r2 | 73.81 | 45.89 | 47.09 |
| **off median** | 65.70 | 43.82 | 48.72 |
| **on median** | 73.11 | 45.89 | 47.03 |

Spec baseline at `-c 65536`: 42.9 t/s decode. Both off and on medians here are
above that (43.8 / 45.9, +2.1% / +7.0%), not within the ~1% band but on the high
side — no regression; likely explained by the larger `-c 262144` allocation
and/or run-to-run drift on this box (prefill on a 82-token prompt is noisy by
construction). Flag on is consistently a little faster than flag off in every
short pairing here, which is the opposite of a regression.

### 37K (1000 generated tokens, `long_prompt_32k_prose.txt`, 36,798 tokens)

| run | prefill t/s | decode t/s | peak wired GiB |
|---|---|---|---|
| off r1 | 459.56 | 40.49 | 49.15 |
| on r1 | 491.14 | 36.58 | 47.27 |
| off r2 | 441.57 | 36.63 | 49.23 |
| on r2 | 473.79 | 36.59 | 47.35 |
| **off median** | 450.57 | 38.56 | 49.19 |
| **on median** | 482.47 | 36.59 | 47.31 |

Spec baseline at `-c 65536`: 39.6-40.0 t/s decode, 404 t/s prefill.
**Decode does not meet the ~1% expectation**: off median 38.56 t/s is ~2.6-3.6%
below the 39.6-40.0 t/s band, and on median 36.59 t/s is ~7.6-8.5% below it. The
effect is present in both flag-off and flag-on runs (off r1's 40.49 t/s is
in-band, off r2's 36.63 t/s is not), so it does not look attributable to the
grow-on-demand patch specifically, but it is a real deviation from the stated
expectation and is reported here as-is rather than tuned away. Prefill is well
above the 404 t/s baseline in every run (441-491 t/s), so the prefill side of
the expectation is met with margin.

### 135K (1000 generated tokens, `long_prompt_128k_prose.txt`, 135,151 tokens)

No spec baseline was given for this case; reported for completeness.

| run | prefill t/s | decode t/s | peak wired GiB |
|---|---|---|---|
| off r1 | 436.79 | 33.48 | 49.14 |
| on r1 | 430.22 | 33.65 | 49.17 |
| off r2 | 425.50 | 33.54 | 49.14 |
| on r2 | 427.80 | 33.76 | 49.14 |
| **off median** | 431.15 | 33.51 | 49.14 |
| **on median** | 429.01 | 33.71 | 49.16 |

Flag on and flag off are within noise of each other (decode +0.6%, prefill
-0.5%). On both flag-on runs a KV capacity growth `32768 -> 262144 rows
(+1.69 GiB)` fires once, at prefill sync (135K prompt needs the full 262144-row
context immediately, same as the 256K case).

### 256K (1000 generated tokens, `long_prompt_256k_prose.txt`, 256,145 tokens) — byte-exact gate pair reused as the perf pair

| run | prefill t/s | decode t/s | peak wired GiB |
|---|---|---|---|
| off | 429.67 | 34.59 | 49.38 |
| on | 426.51 | 34.15 | 49.27 |

Difference: prefill -0.7%, decode -1.3%, peak wired -0.2 GiB — **unchanged
within noise**, matching the expectation ("256K unchanged within noise").

## Summary of deviations from expectations

- The plan's original mid-decode CLI case was defective (grew at sync, not
  mid-decode, because the reserve of `+4096` already exceeded the 32768-row
  init cap for a 31K prompt). Per controller ruling this was replaced with
  `DS4_QWEN4_KV_RESERVE=64` on both flag-off and flag-on; the corrected case
  passes exactly as specified (single capacity line during decode, cmp
  identical).
- All byte-exact gates (short, reserve, corrected mid-decode, full) pass:
  stdout is byte-identical between `DS4_QWEN4_KV_GROW=1` and unset in every
  case, and the capacity log lines match the number/placement expected in each
  case.
- All three alternate KV layouts (half cache, FP8, raw indexer keys) pass
  `--qwen-kv-grow` with exit 0 and sane grow/shrink capacity accounting.
- Server checkpoint-restore and shrink gates both pass: byte-identical replies
  across restarts and across flag off/on, correct shrink log line, and a
  ~1.22 GiB wired-memory drop after the shrink (target: "about 1 GiB").
- Performance: short and 256K meet expectations (on par with or better than
  baseline / unchanged within noise). **37K decode throughput does not meet the
  ~1% expectation** versus the `-c 65536` baseline (39.6-40.0 t/s): measured
  38.56 t/s (flag off) and 36.59 t/s (flag on), 3-9% below the band. This shows
  up with the flag off too, so it is not evidence of a grow-on-demand
  regression, but it is a real miss against the stated target and is reported
  here rather than tuned away, per the "report honestly" instruction. No engine
  changes were made to chase this number, consistent with this task's scope
  (verification only, no engine code changes).

## Work-directory layout (not committed)

All raw logs, scripts, and prompt files for this task live under
`.superpowers/sdd/2026-09-23-qwen4-kv-grow-on-demand/task5/` (git-ignored). Only
this file is committed.
