# ds4-server: hard thinking budget — design

Date: 2026-09-24. Status: design approved in conversation (approach A, sections 1-4); pending spec review.
Base: branch `think-budget` from develop 2152f86 (includes the DS4.1 merge).

## Problem

PROD ds4 (unc48L K=32, `-c 262144`, MTP) decodes at ~40 tok/s, yet users perceive it as slow because the
model reasons for a long time before the visible answer starts. Evidence (read-only investigation,
2026-09-24):

- ds4-server defaults to `DS4_THINK_HIGH`; the gateway sends agent traffic with thinking on and
  `reasoning_effort=xhigh` (ds4 maps it to HIGH).
- A trivial prompt ("write an iterative Fibonacci function with a docstring and two doctests") spent all
  300 generated tokens (7.3 s) inside `<think>` without emitting any code.
- The gateway's only guard is `MLX_THINK_CAP_CHARS=28000` (~7K tokens, ~3 min): when reasoning crosses it
  without an answer, the gateway discards the whole stream and replays the request without thinking.
- ds4 already has soft effort control (`reasoning_effort` low/medium/high/max, rendered as a system
  instruction in the Qwen3.8 template) but no hard cap, and it ignores Anthropic `thinking.budget_tokens`.

## Goal

A hard, per-request thinking budget in ds4-server: once the model has generated N tokens inside
`<think>`, the server closes the reasoning gracefully and the model answers in the same generation.
With the feature off, output is byte-identical to today.

## Non-goals

- Early exit by confidence (DEER, DART, steering vectors): a separate research spike later.
- Gateway code changes (AI-Gateway-MLX). The gateway keeps its 28K-char guard; lowering it is later.
- Changing default reasoning effort or the chat template.
- `--batched-session` mode: the budget is ignored there with a startup warning (PROD is single-slot).

## Interface

Server flags:

- `--think-budget N` (default 0 = off): server ceiling for thinking tokens per request.
- `--think-budget-message TEXT`: text inserted before `</think>` when the budget is reached. Default is
  Qwen's documented fallback sentence: `Considering the limited time by the user, I have to give the
  solution based on the thinking directly now.`

Request fields (all optional, integer > 0 to take effect):

- Anthropic `/v1/messages`: `thinking.budget_tokens` (sent by Claude Code).
- OpenAI `/v1/chat/completions`: top-level `thinking_budget`, or `chat_template_kwargs.thinking_budget`.
- Responses `/v1/responses`: top-level `thinking_budget`.

Effective budget = the smaller of the server ceiling and the request value, counting only values > 0.
Neither set = unlimited (today's behaviour). The budget applies only when the request's think mode is
enabled and the model is inside `<think>`; only tokens generated inside `<think>` count. `max_tokens`
remains the overall cap, and forced tokens count toward `completion_tokens`.

## Mechanism (approach A: in-loop forced close)

In `generate_job_inner`'s decode loop (ds4_server.c):

1. After `thinking_state_feed` for each kept token, add one to the count if the model is still inside
   `<think>` (the token that closes `</think>` itself is not counted).
2. When the count reaches the effective budget, the model is still inside `<think>`, and no tool call is
   open inside the reasoning: keep that token, stop processing the current MTP block (the existing
   `kept < ntok` path rewinds the session to `block_start + kept` via `server_generation_rewind`), and
   queue the forced suffix.
3. Forced suffix = `"\n\n" + message + "\n</think>\n\n"`, tokenized once per request with
   `ds4_tokenize_rendered_chat` so `</think>` maps to the vocabulary's `think_end_id`.
4. The next loop iterations consume the queue instead of sampling: each forced token is evaluated with
   `server_eval_token` (the same plain-eval path the resample branch already mixes with speculative
   decoding) and then runs through the unchanged per-token body (text append, thinking state, stream
   updates, stop scanning), in chunks of at most the block capacity (17). The `</think>` token flips the
   stream from reasoning to content exactly as a model-generated one would.
5. Sampling resumes from the logits after the last forced token; MTP drafting resumes on the next cycle.
6. If a tool call is open inside the reasoning when the budget is reached, the close is deferred until
   that block ends, so tool syntax is never cut.

Logging: `thinking budget reached N tokens; forced close` when it fires, and a `thinking closed after K
tokens` line whenever reasoning ends (naturally or forced), which the measurement step reads.

## Output and cache

- The forced sentence is part of the reasoning (`reasoning_content` / Anthropic thinking block); the
  answer is `content`. Streaming and non-streaming responses carry the same text. `finish_reason` is
  unchanged.
- The live session state contains exactly the forced tokens the client received, so the next turn
  continues through the existing `thinking-visible` live-prefix path.
- No force when: `max_tokens` ends generation first; the model closes `</think>` itself on the last budget
  token; the request has thinking disabled.
- Other model families (DeepSeek, GLM) use the same `</think>` mechanism; only Qwen3.8 is validated.

## Testing

- Unit (`ds4-server` unit group, no GPU): request parsing for the three APIs, effective-budget rule,
  flag parsing, forced-suffix tokenization contains `think_end_id`.
- Model-backed (needs the GPU; run when the DS4.1 session is idle), small budget (e.g. 64) on a prompt that
  reasons at length:
  1. reasoning tokens <= budget + forced suffix; forced sentence and `</think>` present; non-empty answer;
  2. stream and non-stream produce identical text;
  3. works with MTP on and off;
  4. the next turn after a forced close hits the live cache (`thinking-visible`);
  5. budget off: byte-identical to the pre-change binary on the existing smoke prompts.

## Choosing N (measurement)

When the GPU is free: run a VI / code / agent prompt set (the Qwen regression gate prompts, plus the
gateway's `structured` / `toolcall` suites if they can target the ds4 port) with no budget to record the
thinking-token distribution, then with N in {1024, 2048, 4096, 8192} and unlimited. Pick the lowest N with
no quality loss; write a receipt under `speed-bench/`.

## Rollout

Branch `think-budget` (from develop 2152f86) -> review -> merge to develop (tell the DS4.1 session, which
builds on `ds4_server.c`) -> `prod/think-budget-YYYYMMDD` via `deploy-ai-gateway.sh` -> add
`--think-budget N` to the registry `process_command` (backup first) -> smoke per docs/DEPLOY_AI_GATEWAY.md.

## Risks

- Too small N hurts answer quality on hard tasks; mitigated by measuring before choosing N.
- Forcing inside an unfinished thought can produce a weaker answer than letting the model finish; the
  budget is a ceiling for runaway reasoning, not a way to shorten normal reasoning (that is
  `reasoning_effort`).
- The decode loop is shared by every model family and the DS4.1 work; changes stay behind the budget
  being set, and the byte-identical gate covers the off path.
