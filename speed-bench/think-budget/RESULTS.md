# Thinking budget (`--think-budget N`): distribution and quality per N

Date: 2026-09-25. Commit tested: `49c7a97` (branch `think-budget`, code at `0afe5df`).

Machine: Apple M5 Pro, 64 GiB RAM, oMLX backend stopped for the run.

Model/command: PROD registry command for `Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q4KDownPad768-DenseQ4Kselimat-MTP.gguf`
(unc48L), `DS4_QWEN4_STREAM_FULL_LAYERS=32`, 6 GiB expert cache, `DS4_QWEN4_KV_GROW=1`,
`-c 262144`, `--prefill-chunk 2048`, `--mtp`, 64K draft vocab — but with this checkout's
`ds4-server`, a scratch port (18299) and scratch KV dir, plus `--think-budget N` (`N=0` = off).
Requests: temperature 0, thinking on (server default), `max_tokens 16384`.

Command run:

```
caffeinate -i -s python3 speed-bench/think-budget/measure.py --budgets 0,8192,4096,2048,1024
```

Benchmarks: GSM8K-mini (16 problems) and HumanEval-mini (17 problems), code-graded, vendored in
`~/Documents/GitHub/AI-Gateway-MLX/evals/bakeoff/benchmarks`, plus 5 Vietnamese prompts (ungraded,
answers recorded). 38 prompts x 5 budgets = 190 requests.

## Summary

| N | gsm8k pass | humaneval pass | forced | think median | think p90 | think max | total seconds |
|---|---|---|---|---|---|---|---|
| 0 (off) | 16/16 | 17/17 | 0 | 117.5 | 644 | 4053 | 481.6 |
| 8192 | 16/16 | 17/17 | 0 | 117.5 | 644 | 4053 | 479.5 |
| 4096 | 16/16 | 17/17 | 0 | 117.5 | 644 | 4053 | 481.5 |
| 2048 | 16/16 | 17/17 | 1 | 117.5 | 644 | 2048 | 428.5 |
| 1024 | 16/16 | 17/17 | 2 | 117.5 | 644 | 1024 | 436.2 |

Every N keeps every N=0 pass (gsm8k 16/16, humaneval 17/17 throughout). `forced` = number of
prompts (out of 38) whose thinking was cut by the budget (0/0/0/1/2 for N = 0/8192/4096/2048/1024).

## Unlimited-run (N=0) thinking distribution, by kind

| kind | n | median tokens | max tokens |
|---|---|---|---|
| gsm8k | 16 | 80.5 | 217 |
| humaneval | 17 | 309 | 1361 |
| vi | 5 | 172 | 4053 |

Thinking is short and predictable for gsm8k, longer and more variable for humaneval and vi.
Across all 38 prompts, only 2 ever reach 1024 thinking tokens unforced (p90 = 644).

## The forced cases

- **idx 34, VI "Viết một đoạn văn khoảng 120 chữ giới thiệu vịnh Hạ Long cho khách du lịch."**
  Thinks 4053 tokens / 102.8 s at N=0. Cut to 2048 (53.9 s) and 1024 (30.8 s), with a nearly
  identical answer:

  | N | seconds | words | answer |
  |---|---|---|---|
  | 0 | 102.8 | 120 | Vịnh Hạ Long là điểm đến kỳ quan thiên nhiên thế giới, nơi hàng nghìn đảo đá vôi vươn lên giữa làn nước ngọc bích. Đến đây, du khách được ngắm bình minh rực rỡ, khám phá hang động huyền ảo, chèo kayak len lỏi qua các khe đá và dừng chân trên bãi biển hoang sơ. Không gian yên bình, gió biển mát lành cùng cảnh sắc hùng vĩ sẽ để lại cảm giác choáng ngợp. Hạ Long còn hấp dẫn bởi hải sản tươi ngon, làng chài truyền thống và những tour thuyền đêm lãng mạn. Hãy đến đây để tận mắt chiêm ngưỡng vẻ đẹp kỳ diệu của di sản Việt Nam và lưu giữ kỷ niệm. |
  | 2048 (forced) | 53.9 | 121 | Vịnh Hạ Long là điểm đến kỳ quan thiên nhiên thế giới, nơi hàng nghìn đảo đá vôi vươn lên giữa làn nước ngọc bích. Đến đây, du khách được ngắm bình minh rực rỡ, khám phá hang động huyền ảo, chèo kayak len lỏi qua các khe đá và dừng chân trên bãi biển hoang sơ. Không gian yên bình, gió biển mát lành cùng cảnh sắc hùng vĩ sẽ để lại cảm giác choáng ngợp. Hạ Long còn hấp dẫn bởi hải sản tươi ngon, làng chài truyền thống và những tour thuyền đêm lãng mạn. Hãy một lần đến để tận mắt chiêm ngưỡng vẻ đẹp kỳ diệu của di sản Việt Nam và lưu giữ kỷ niệm. |
  | 1024 (forced) | 30.8 | 116 | Vịnh Hạ Long là điểm đến kỳ quan thiên nhiên thế giới, nơi hàng nghìn đảo đá vôi vươn lên giữa làn nước ngọc bích. Đến đây, du khách được ngắm bình minh rực rỡ, khám phá hang động huyền ảo, chèo kayak len lỏi qua các khe đá và dừng chân trên bãi biển hoang sơ. Không gian yên bình, gió biển mát lành cùng cảnh sắc hùng vĩ sẽ để lại cảm giác choáng ngợp. Hạ Long còn hấp dẫn bởi hải sản tươi ngon, làng chài truyền thống và những tour thuyền đêm lãng mạn. Hãy một lần đến để tận mắt chiêm ngưỡng vẻ đẹp kỳ diệu của di sản Việt Nam. |

  This is the waste the feature targets: a simple 120-word paragraph spent 102.8 s thinking
  unlimited; capped at 1024 it finished in 30.8 s with essentially the same answer.

- **idx 29, HumanEval.** Thinks 1361 tokens at N=0, and still passes (graded correct) when cut
  at 1024. Not forced at N=2048 (thinking finished under 2048 on its own).

## Chosen N

**N = 1024** — the lowest budget whose graded pass set (gsm8k 16/16, humaneval 17/17) keeps
every case that passed at N=0.

Caveat, stated plainly: the benchmark set is small and easy (thinking p90 is only 644 tokens;
only 2 of 38 prompts ever reach 1024 unforced) and has no long agentic/tool turns, so the
evidence here for a low cap on real agent traffic is thin. The PROD value should be confirmed
with the user at deploy time; raising N later is a one-line registry change.

Known measurement caveat (from review): a "thinking closed after N tokens (budget)" log line
can also fire when `max_tokens` ends on the fire token, not just on the budget cutting thinking
short. Not reachable in this run (`max_tokens 16384` is far above every budget tested).

## Data

Per-budget rows: `speed-bench/think-budget/runs/budget-{0,8192,4096,2048,1024}.json`
(`kind`, `passed`, `think_tokens`, `forced`, `seconds`, `answer` for `vi` rows; row order gsm8k
idx 0-15, humaneval idx 16-32, vi idx 33-37). Summary: `speed-bench/think-budget/runs/summary.json`.
