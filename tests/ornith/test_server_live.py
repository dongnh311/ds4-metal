#!/usr/bin/env python3
"""Ornith serving checks on a live staging server (M3).

  python3 tests/ornith/test_server_live.py OUT_DIR

Needs DS4_ORNITH_MODEL, a built Metal ./ds4-server and a free GPU.  Servers
run one at a time on port 18296.  Hard checks are structural; what the model
says is a soft check (WARN, see serverlib).

1. ids: /v1/models lists ornith-1.5-35b-a3b, -chat and -reasoner; a request
   without "model" answers as ornith-1.5-35b-a3b; -chat answers without
   reasoning.
2. MTP identity: at temperature 0 a --mtp server and a plain server give
   byte-identical content, reasoning, tool calls and finish reasons for four
   prompts (gate 1 item 3 through the server).  The --mtp server runs with
   --trace, and the script prints how many speculative-boundary rewinds the
   trace records; that count is information only (ds4_test --qwen35-rewind
   pins the rewind itself).
3. OpenAI tool round trip (--mtp): get_weather is called with an arguments
   object; the tool result goes back, and the answer ends with stop and
   reuses the live prefix.
4. Anthropic /v1/messages tool round trip: tool_use, then a tool_result
   block plus a text block in one user message; the answer ends the turn and
   reads the whole first prompt from the cache.
5. Think-cap replay as the gateway sends it: a conversation with tools and
   historical reasoning_content, first with thinking on and a 64-token cap,
   then with enable_thinking=false and preserve_thinking removed; the replay
   returns no reasoning.
6. Think budget: with --think-budget 64 the reasoning ends with the budget
   sentence and an answer follows.
"""
import copy
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import serverlib as sl

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
ANTHROPIC_WEATHER = {"name": "get_weather", "description": "Get the current weather for a city.",
                     "input_schema": {"type": "object", "properties": {"city": {"type": "string"}},
                                      "required": ["city"]}}
WEATHER_RESULT = '{"temp_c": 18, "sky": "cloudy"}'
ASK_WEATHER = "What's the weather in Paris right now? Use the tool."
IDENTITY = [
    ("en", [{"role": "user", "content": "Explain in two sentences why the sky is blue."}], None),
    ("vi", [{"role": "user", "content": "Kể tên ba món ăn nổi tiếng của Hà Nội, mỗi món một câu."}], None),
    ("code", [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number. "
                                          "Code only."}], None),
    ("tool", [{"role": "user", "content": ASK_WEATHER}], [WEATHER]),
]
# the gateway's think-cap replay shape: tools, a tool round trip and
# historical reasoning_content before the new question
REPLAY_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Paris?"},
    {"role": "assistant", "content": "", "reasoning_content": "The user wants weather. Call the tool.",
     "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": WEATHER_RESULT},
    {"role": "assistant", "content": "It is 18 C and cloudy in Paris.",
     "reasoning_content": "Tool returned 18 C cloudy."},
    {"role": "user", "content": "Should I bring an umbrella? Answer yes or no, then one sentence."},
]


def chat(messages, tools=None, save=None, **extra):
    body = {"messages": messages, "temperature": 0, "max_tokens": 600, "stream": False}
    if tools:
        body["tools"] = tools
    body.update(extra)
    return sl.post("/v1/chat/completions", body, save=save)


def reply_key(result):
    message = result["choices"][0]["message"]
    calls = [(c["function"]["name"], c["function"]["arguments"]) for c in message.get("tool_calls") or []]
    return [message.get("content") or "", message.get("reasoning_content") or "", calls,
            result["choices"][0]["finish_reason"]]


def cached_tokens(result):
    return (result["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)


def arguments_object(call):
    """An OpenAI tool call's parsed arguments, or None when they are not a JSON object."""
    try:
        args = json.loads(call["function"]["arguments"])
    except (KeyError, TypeError, ValueError):
        return None
    return args if isinstance(args, dict) else None


def main(argv):
    if len(argv) != 1:
        sys.exit(__doc__)
    out = pathlib.Path(argv[0]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = out / "server.log"

    srv = sl.Server(log, ["-c", "32768"])
    try:
        plain = {name: reply_key(chat(msgs, tools, save=out / f"identity-{name}-plain"))
                 for name, msgs, tools in IDENTITY}
    finally:
        srv.stop()

    trace = out / "trace-mtp.txt"
    srv = sl.Server(log, ["--mtp", "-c", "32768", "--trace", trace])
    try:
        # 1. ids
        ids = [m["id"] for m in srv.models["data"]]
        sl.check(ids == ["ornith-1.5-35b-a3b", "ornith-1.5-35b-a3b-chat", "ornith-1.5-35b-a3b-reasoner"],
                 f"/v1/models lists {ids}")
        r = chat([{"role": "user", "content": "Say ok."}], save=out / "ids-default", max_tokens=64)
        sl.check(r.get("model") == "ornith-1.5-35b-a3b", "a request without model answers as ornith-1.5-35b-a3b")
        r = chat([{"role": "user", "content": "Say ok."}], save=out / "ids-chat",
                 model="ornith-1.5-35b-a3b-chat", max_tokens=64)
        m = r["choices"][0]["message"]
        sl.check(not m.get("reasoning_content") and (m.get("content") or "").strip() != "",
                 "-chat answers without reasoning")

        # 2. MTP identity
        for name, msgs, tools in IDENTITY:
            got = reply_key(chat(msgs, tools, save=out / f"identity-{name}-mtp"))
            sl.check(got == plain[name], f"--mtp reply equals the plain reply ({name})")

        # 3. OpenAI tool round trip
        msgs = [{"role": "user", "content": ASK_WEATHER}]
        first = chat(msgs, [WEATHER], save=out / "openai-tool-1", max_tokens=1024)
        m1 = first["choices"][0]["message"]
        calls = m1.get("tool_calls") or []
        args = arguments_object(calls[0]) if calls else None
        sl.check(first["choices"][0]["finish_reason"] == "tool_calls" and bool(calls) and
                 calls[0]["function"]["name"] == "get_weather" and args is not None,
                 "the model calls get_weather with an arguments object")
        sl.soft_check("paris" in str(args.get("city", "")).lower(), "the call asks for Paris",
                      f"{out / 'openai-tool-1'}.request.json")
        msgs += [{"role": "assistant", "content": m1.get("content") or "",
                  "reasoning_content": m1.get("reasoning_content") or "", "tool_calls": calls},
                 {"role": "tool", "tool_call_id": calls[0]["id"], "content": WEATHER_RESULT + "\n"}]
        second = chat(msgs, [WEATHER], save=out / "openai-tool-2", max_tokens=1024)
        answer = second["choices"][0]["message"].get("content") or ""
        sl.check(second["choices"][0]["finish_reason"] == "stop" and answer.strip() != "",
                 "the tool turn ends with an answer")
        sl.soft_check("18" in answer, "the answer uses the tool result (18)", f"{out / 'openai-tool-2'}.request.json")
        sl.check(cached_tokens(second) >= first["usage"]["prompt_tokens"],
                 f"the tool turn reuses the live prefix ({cached_tokens(second)} cached)")

        # 4. Anthropic tool round trip
        amsgs = [{"role": "user", "content": ASK_WEATHER}]
        a1 = sl.post("/v1/messages", {"max_tokens": 1024, "temperature": 0, "tools": [ANTHROPIC_WEATHER],
                                      "messages": amsgs}, save=out / "anthropic-tool-1")
        uses = [b for b in a1["content"] if b["type"] == "tool_use"]
        sl.check(a1["stop_reason"] == "tool_use" and bool(uses) and uses[0]["name"] == "get_weather" and
                 isinstance(uses[0].get("input"), dict), "anthropic: the model calls get_weather with an input object")
        sl.soft_check("paris" in str(uses[0]["input"].get("city", "")).lower(), "anthropic: the call asks for Paris",
                      f"{out / 'anthropic-tool-1'}.request.json")
        amsgs += [{"role": "assistant", "content": a1["content"]},
                  {"role": "user", "content": [
                      {"type": "tool_result", "tool_use_id": uses[0]["id"], "content": WEATHER_RESULT},
                      {"type": "text", "text": "Answer in one sentence."}]}]
        a2 = sl.post("/v1/messages", {"max_tokens": 1024, "temperature": 0, "tools": [ANTHROPIC_WEATHER],
                                      "messages": amsgs}, save=out / "anthropic-tool-2")
        text = "".join(b.get("text", "") for b in a2["content"] if b["type"] == "text")
        sl.check(a2["stop_reason"] == "end_turn" and text.strip() != "", "anthropic: the tool turn ends with an answer")
        sl.soft_check("18" in text, "anthropic: the answer uses the tool result (18)",
                      f"{out / 'anthropic-tool-2'}.request.json")
        u1 = a1["usage"]
        a1_prompt = (u1.get("input_tokens", 0) + u1.get("cache_read_input_tokens", 0) +
                     u1.get("cache_creation_input_tokens", 0))
        sl.check(a2["usage"].get("cache_read_input_tokens", 0) >= a1_prompt,
                 f"anthropic: the tool turn reads the whole first prompt ({a1_prompt} tokens) from the cache")

        # 5. think-cap replay, as the gateway builds it
        base = {"messages": REPLAY_HISTORY, "tools": [WEATHER], "temperature": 0, "stream": False,
                "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}
        capped = sl.post("/v1/chat/completions", dict(base, max_tokens=64), save=out / "think-cap-1")
        replay = copy.deepcopy(base)
        replay["chat_template_kwargs"]["enable_thinking"] = False
        del replay["chat_template_kwargs"]["preserve_thinking"]
        replay["max_tokens"] = 400
        r = sl.post("/v1/chat/completions", replay, save=out / "think-cap-2")
        m = r["choices"][0]["message"]
        finish = r["choices"][0]["finish_reason"]
        sl.check(not m.get("reasoning_content") and finish in ("stop", "tool_calls") and
                 (finish != "stop" or (m.get("content") or "").strip() != ""),
                 "the think-cap replay returns no reasoning")
        sl.soft_check(finish == "stop", "the replay answers instead of calling the tool again",
                      f"{out / 'think-cap-2'}.request.json")
        print(f"INFO think-cap: capped attempt cached {cached_tokens(capped)}, replay cached {cached_tokens(r)} of "
              f"{r['usage']['prompt_tokens']} (thinking changes the system turn, so a re-prefill is expected)",
              flush=True)
    finally:
        srv.stop()
    trace_text = trace.read_text(errors="replace") if trace.exists() else ""
    print(f"INFO speculative-boundary rewinds in the --mtp trace: {trace_text.count('speculative boundary: kept=')}",
          flush=True)

    # 6. think budget
    srv = sl.Server(log, ["--mtp", "-c", "32768", "--think-budget", "64"])
    try:
        r = chat([{"role": "user", "content": "Prove that there are infinitely many prime numbers, "
                                              "then list the first ten primes."}],
                 save=out / "think-budget", max_tokens=600)
        new = srv.new_log()
        m = r["choices"][0]["message"]
        fired = re.search(r"thinking budget reached (\d+) tokens", new)
        n = int(fired.group(1)) if fired else None
        sl.check(fired is not None and 64 <= n <= 65,
                 f"think budget: the cap fired at {n} tokens (64..65)" if fired else
                 "think budget: the cap fired (no matching log line)")
        sl.check(sl.BUDGET_MESSAGE in (m.get("reasoning_content") or ""),
                 "think budget: the reasoning ends with the budget sentence")
        sl.check((m.get("content") or "").strip() != "", "think budget: an answer follows")
    finally:
        srv.stop()
    sl.report("ornith server live")


if __name__ == "__main__":
    main(sys.argv[1:])
