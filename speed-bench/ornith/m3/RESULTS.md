# Ornith M3 serving receipts

- Branch commit: 01ae3b7 (feature/ornith-m3; code head for every run below, 2026-09-26 22:50-22:53).
- Model: 23G ICE (`speed-bench/ornith/oracle/RESULTS.md`).
- Model-free: `./ds4_test --server` ok; `./ds4_agent_test` ok; `make test-ornith-render` 26/26
  (goldens `tests/ornith/chat/golden/`, template sha256 in `TEMPLATE.txt`).
- CLI: `tests/ornith/test_cli_chat.sh` ok.
- Sessions: `ds4_test --session-snapshot --qwen35-payloads --qwen35-rewind` ok with and without
  `DS4_TEST_GLM_MTP=1`; `make test-qwen35-session` ok (incl. the serialized batch case).
- Loader: `tests/ornith/test_loader.sh` ok.

## Live gate (port 18296)

| script | result |
|---|---|
| `tests/ornith/test_server_kv.py` | OK (0 WARN) |
| `tests/ornith/test_server_live.py` | OK (0 WARN) |

Per-turn cached tokens (test_server_kv.py):

```
{"case": "tool-less-omit", "turn": 1, "finish": "stop", "prompt": 3232, "cached": 0, "content": "51"}
{"case": "tool-less-omit", "turn": 2, "finish": "stop", "prompt": 3284, "cached": 3259, "content": "53"}
{"case": "tool-less-omit", "turn": 3, "finish": "stop", "prompt": 3324, "cached": 3299, "content": "55"}
{"case": "tools-omit", "turn": 1, "finish": "stop", "prompt": 3596, "cached": 0, "content": "51"}
{"case": "tools-omit", "turn": 2, "finish": "stop", "prompt": 3656, "cached": 3631, "content": "53"}
{"case": "tools-omit", "turn": 3, "finish": "stop", "prompt": 3696, "cached": 3671, "content": "55"}
{"case": "tools-echo", "turn": 1, "finish": "stop", "prompt": 3596, "cached": 0, "content": "51"}
{"case": "tools-echo", "turn": 2, "finish": "stop", "prompt": 3645, "cached": 3618, "content": "53"}
{"case": "tools-echo", "turn": 3, "finish": "stop", "prompt": 3687, "cached": 3660, "content": "55"}
```

PASS, WARN and INFO lines (test_server_live.py):

```
PASS /v1/models lists ['ornith-1.5-35b-a3b', 'ornith-1.5-35b-a3b-chat', 'ornith-1.5-35b-a3b-reasoner']
PASS a request without model answers as ornith-1.5-35b-a3b
PASS -chat answers without reasoning
PASS --mtp reply equals the plain reply (en)
PASS --mtp reply equals the plain reply (vi)
PASS --mtp reply equals the plain reply (code)
PASS --mtp reply equals the plain reply (tool)
PASS the model calls get_weather with an arguments object
PASS the call asks for Paris
PASS the tool turn ends with an answer
PASS the answer uses the tool result (18)
PASS the tool turn reuses the live prefix (588 cached)
PASS anthropic: the model calls get_weather with an input object
PASS anthropic: the call asks for Paris
PASS anthropic: the tool turn ends with an answer
PASS anthropic: the answer uses the tool result (18)
PASS anthropic: the tool turn reads the whole first prompt (545 tokens) from the cache
PASS the think-cap replay returns no reasoning
PASS the replay answers instead of calling the tool again
INFO think-cap: capped attempt cached 0, replay cached 0 of 604 (thinking changes the system turn, so a re-prefill is expected)
INFO speculative-boundary rewinds in the --mtp trace: 5
PASS think budget: the cap fired at 64 tokens (64..65)
PASS think budget: the reasoning ends with the budget sentence
PASS think budget: an answer follows
```

WARN follow-up (llama-server on the same request): none (no WARN in either script).
