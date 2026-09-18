# Phase 3 gate — ignore_eos under qwen4 --mtp (fix #3 validated by A/B)

End-to-end server A/B on the imat model (temp 0, max_tokens 120, prompt that ends
naturally after ~28 tokens). ignore_eos is parsed ONLY by parse_chat_request
(/v1/chat/completions), so all tests use that endpoint.

| path | ignore_eos | binary | finish | tokens |
|---|---|---|---|---|
| plain (MTP off) | false | fixed | stop | 28 |
| plain (MTP off) | **true** | fixed | length | **120** (honored) |
| **MTP on** | **true** | **fix DISABLED (pre-fix rebuild)** | stop | **28 (BUG: early-stop)** |
| **MTP on** | **true** | **fixed** | length | **120 (honored)** |

The pre-fix column was produced by disabling exactly the one fix line
(`if (j->req.ignore_eos) break;`) and rebuilding, isolating the fix on the same
imat-compatible codebase (the 6c1e836 baseline binary can't load the Q4_K imat
model, so a same-source A/B was used instead). Confirms: the bug was real and
MTP-block-specific (plain path always honored ignore_eos); fix #3 corrects it.

## Observation (pre-existing, out of scope): /v1/completions ignores ignore_eos
`ignore_eos` is parsed only in parse_chat_request, NOT in the /v1/completions,
/v1/messages(anthropic), or /v1/responses parsers. An eval client POSTing
ignore_eos to /v1/completions gets it silently dropped (validation passes because
`request_validate_ignore_eos` returns true when the flag is false). This is
independent of the MTP fix; noted for the report as a latent gap, not fixed here
(would touch the completions parser — a behavior change beyond the deep-search
scope). ai-gateway eval clients should use /v1/chat/completions for ignore_eos.
