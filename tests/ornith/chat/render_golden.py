#!/usr/bin/env python3
"""Golden renders of Ornith's embedded chat template (gate 1 item 5).

  render_golden.py [--check]

Reads tokenizer.chat_template from $DS4_ORNITH_MODEL (GGUF header only, stdlib
reader; the template is not committed), renders every case in CASES with jinja2
in the environment transformers.apply_chat_template builds, and writes
tests/ornith/chat/golden/<case>.txt plus <case>.input.json:
  api       "chat" (POST /v1/chat/completions) or "anthropic" (POST /v1/messages)
  body      the HTTP body ds4-server parses (OpenAI tool arguments are JSON strings)
  template  what jinja saw: messages, tools, kwargs
OpenAI string tool arguments are parsed into objects before rendering, as
llama.cpp does for templates that accept object arguments.  Anthropic cases
carry a hand-written OpenAI-shaped template input; tools keep ds4's
Anthropic mapping, {"type": "function", "function": <tool as given>}.
--check re-renders and compares with the committed files instead of writing.

jinja2 is not in the system python3; run with
  ~/.local/ai-gateway-env/bin/python3 tests/ornith/chat/render_golden.py
(never pip install into ~/.local/omlx-venv).
"""
import copy
import hashlib
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "golden")
TEMPLATE_VERSION = "qwen3.8-froggeric-v22.4.1"
MODEL_ID = "ornith-1.5-35b-a3b"

GGUF_STRING, GGUF_ARRAY = 8, 9
GGUF_SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _read_exact(f, n):
    data = f.read(n)
    if len(data) != n:
        raise SystemExit("render_golden.py: truncated GGUF header")
    return data


def _read_string(f):
    n, = struct.unpack("<Q", _read_exact(f, 8))
    return _read_exact(f, n).decode("utf-8")


def _skip_value(f, vtype):
    if vtype == GGUF_STRING:
        n, = struct.unpack("<Q", _read_exact(f, 8))
        f.seek(n, 1)
    elif vtype == GGUF_ARRAY:
        etype, count = struct.unpack("<IQ", _read_exact(f, 12))
        if etype in GGUF_SCALAR_BYTES:
            f.seek(GGUF_SCALAR_BYTES[etype] * count, 1)
        else:
            for _ in range(count):
                _skip_value(f, etype)
    elif vtype in GGUF_SCALAR_BYTES:
        f.seek(GGUF_SCALAR_BYTES[vtype], 1)
    else:
        raise SystemExit(f"render_golden.py: unknown GGUF value type {vtype}")


def read_chat_template(path):
    """tokenizer.chat_template from a GGUF file, reading only the metadata."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"render_golden.py: {path} is not a GGUF file")
        _read_exact(f, 4)                                   # version
        _n_tensors, n_kv = struct.unpack("<QQ", _read_exact(f, 16))
        for _ in range(n_kv):
            key = _read_string(f)
            vtype, = struct.unpack("<I", _read_exact(f, 4))
            if key == "tokenizer.chat_template":
                if vtype != GGUF_STRING:
                    raise SystemExit("render_golden.py: tokenizer.chat_template is not a string")
                return _read_string(f)
            _skip_value(f, vtype)
    raise SystemExit(f"render_golden.py: {path} has no tokenizer.chat_template")


def template_view(body):
    """The template inputs llama.cpp derives from an OpenAI chat body."""
    messages = copy.deepcopy(body.get("messages") or [])
    for message in messages:
        for call in message.get("tool_calls") or []:
            fn = call.get("function", call)
            args = fn.get("arguments")
            if not isinstance(args, str):
                continue
            try:
                parsed = json.loads(args)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                fn["arguments"] = parsed
    kwargs = {}
    if "reasoning_effort" in body:
        kwargs["reasoning_effort"] = body["reasoning_effort"]
    kwargs.update(body.get("chat_template_kwargs") or {})
    return {"messages": messages, "tools": copy.deepcopy(body.get("tools")), "kwargs": kwargs}


def case_spec(case):
    api = case.get("api", "chat")
    template = case["template"] if "template" in case else template_view(case["body"])
    return {"api": api, "body": case["body"], "template": template}


def make_renderer(template_text):
    """jinja2 environment of transformers' apply_chat_template."""
    try:
        import jinja2
        import jinja2.ext
        from jinja2.exceptions import TemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError:
        raise SystemExit("render_golden.py: jinja2 missing; run with ~/.local/ai-gateway-env/bin/python3")

    def raise_exception(message):
        raise TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                          sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    compiled = env.from_string(template_text)

    def render(view):
        ctx = {"messages": view["messages"], "add_generation_prompt": True,
               "bos_token": "<|endoftext|>", "eos_token": "<|im_end|>"}
        if view.get("tools"):
            ctx["tools"] = view["tools"]
        ctx.update(view.get("kwargs") or {})
        return compiled.render(**ctx)

    render.jinja2_version = jinja2.__version__
    return render


SYSTEM = "You are a helpful assistant."
USER = "What is the capital of France?"
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string", "description": "City name"},
                                      "unit": {"type": "string", "enum": ["c", "f"]}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from disk.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "max_lines": {"type": "integer"}},
                       "required": ["path"]}}},
]
RUN_TOOL = {"type": "function", "function": {
    "name": "run",
    "description": "Run a shell command.",
    "parameters": {"type": "object",
                   "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
                   "required": ["command"]}}}
UNICODE_TOOL = {"type": "function", "function": {
    "name": "translate",
    "description": "Dịch \"văn bản\" sang tiếng Việt — nhanh.\tTab.",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    "strict": True}}
ANTHROPIC_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
WEATHER_OK = '{"temp_c": 18, "sky": "cloudy"}'


def call(cid, name, args):
    """An OpenAI wire tool call: arguments travel as a JSON string."""
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def chat(messages, **extra):
    body = {"model": MODEL_ID, "messages": messages}
    body.update(extra)
    return {"api": "chat", "body": body}


def sys_user(**extra):
    return chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}], **extra)


def weather_turns(tool_content):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": "", "reasoning_content": "The user wants weather. Call the tool.",
         "tool_calls": [call("call_1", "get_weather", {"city": "Paris", "unit": "c"})]},
        {"role": "tool", "tool_call_id": "call_1", "content": tool_content},
        {"role": "assistant", "content": "It is 18 C and cloudy in Paris.",
         "reasoning_content": "Tool returned 18 C cloudy."},
        {"role": "user", "content": "And should I bring an umbrella?"},
    ]


HISTORY = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": USER},
    {"role": "assistant", "content": "The capital of France is Paris.",
     "reasoning_content": "Simple geography question.\nAnswer: Paris."},
    {"role": "user", "content": "And of Italy?"},
]

CASES = {
    # system turn
    "chat_system_user": sys_user(),
    "chat_think_off": sys_user(chat_template_kwargs={"enable_thinking": False}),
    "chat_no_system": chat([{"role": "user", "content": USER}]),
    "chat_terse_off": sys_user(chat_template_kwargs={"terse": False}),
    "developer_head_merge": chat([
        {"role": "system", "content": "  You are a helpful assistant.  "},
        {"role": "developer", "content": "Prefer metric units.\n"},
        {"role": "user", "content": USER}]),
    "effort_none": sys_user(reasoning_effort="none"),
    "effort_medium": sys_user(reasoning_effort="medium"),
    "effort_high": sys_user(reasoning_effort="high"),
    "effort_low": sys_user(reasoning_effort="low"),
    "effort_max": sys_user(reasoning_effort="max"),
    "effort_low_think_off": sys_user(chat_template_kwargs={"reasoning_effort": "low",
                                                           "enable_thinking": False}),
    "tools": sys_user(tools=TOOLS),
    "tools_think_off": sys_user(tools=TOOLS, chat_template_kwargs={"enable_thinking": False}),
    "tools_unicode": sys_user(tools=[UNICODE_TOOL]),
    # conversation body
    "tool_round_trip": chat(weather_turns(WEATHER_OK), tools=TOOLS),
    "tool_error": chat(weather_turns("Error: city not found"), tools=TOOLS),
    "tool_errors_consecutive": chat([
        {"role": "user", "content": "Run the build."},
        {"role": "assistant", "content": "", "reasoning_content": "Run make.",
         "tool_calls": [call("c1", "run", {"command": "make"})]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"error": "make: not found", "code": 127}'},
        {"role": "assistant", "content": "Retrying with the full path.", "reasoning_content": "Try /usr/bin/make.",
         "tool_calls": [call("c2", "run", {"command": "/usr/bin/make", "timeout": 30})]},
        {"role": "tool", "tool_call_id": "c2",
         "content": "Traceback (most recent call last):\n  File \"build.py\", line 3, in <module>\n"
                    "ImportError: no module named x\n"}],
        tools=[RUN_TOOL]),
    "tool_output_trailing_nl": chat([
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": "", "tool_calls": [call("c1", "read_file", {"path": "/tmp"})]},
        {"role": "tool", "tool_call_id": "c1", "content": "a.txt\nb.txt\n"}], tools=TOOLS),
    "tool_results_grouped": chat([
        {"role": "user", "content": "read it"},
        {"role": "assistant", "content": "Reading the file now.", "tool_calls": [
            call("c1", "read_file", {"path": "/tmp/a.txt", "max_lines": 20,
                                     "opts": {"a": [1, 2], "b": {"deep": True}}}),
            call("c2", "get_weather", {"city": "Paris", "unit": "c"})]},
        {"role": "tool", "tool_call_id": "c1", "content": "line1\nline2"},
        {"role": "tool", "tool_call_id": "c2", "content": '{"temp_c": 18, "error": null}'}], tools=TOOLS),
    "history_reasoning": chat(HISTORY),
    "preserve_thinking_off": chat(HISTORY, chat_template_kwargs={"preserve_thinking": False}),
    "preserve_thinking_off_tool_loop": chat([
        {"role": "user", "content": "Run the build."},
        {"role": "assistant", "content": "", "reasoning_content": "Run make.",
         "tool_calls": [call("c1", "run", {"command": "make"})]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"}],
        tools=[RUN_TOOL], chat_template_kwargs={"preserve_thinking": False}),
    "assistant_trailing_ws": chat([
        {"role": "user", "content": USER},
        {"role": "assistant", "content": "Paris.\n\n", "reasoning_content": "easy"},
        {"role": "user", "content": "And of Italy?"}]),
    "midconv_system": chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
        {"role": "assistant", "content": "Paris."},
        {"role": "system", "content": "Reminder: be brief."},
        {"role": "user", "content": "And of Italy?"}]),
    # the gateway's think-cap replay: enable_thinking false, preserve_thinking removed
    "think_cap_replay": chat(weather_turns(WEATHER_OK), tools=TOOLS,
                             chat_template_kwargs={"enable_thinking": False}),
    "anthropic_tool_result": {
        "api": "anthropic",
        "body": {"model": MODEL_ID, "max_tokens": 256, "system": SYSTEM, "tools": [ANTHROPIC_TOOL],
                 "messages": [
                     {"role": "user", "content": "What's the weather in Paris?"},
                     {"role": "assistant", "content": [
                         {"type": "thinking", "thinking": "The user wants weather. Call the tool.",
                          "signature": "sig"},
                         {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                          "input": {"city": "Paris"}}]},
                     {"role": "user", "content": [
                         {"type": "tool_result", "tool_use_id": "toolu_1",
                          "content": "Error: city not found\n"},
                         {"type": "text", "text": "Try Lyon instead."}]}]},
        "template": {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "What's the weather in Paris?"},
                {"role": "assistant", "content": "", "reasoning_content": "The user wants weather. Call the tool.",
                 "tool_calls": [{"id": "toolu_1", "type": "function",
                                 "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
                {"role": "tool", "tool_call_id": "toolu_1", "content": "Error: city not found\n"},
                {"role": "user", "content": "Try Lyon instead."}],
            "tools": [{"type": "function", "function": ANTHROPIC_TOOL}],
            "kwargs": {}},
    },
}


def main(argv):
    check = argv == ["--check"]
    if argv and not check:
        sys.exit(__doc__)
    model = os.environ.get("DS4_ORNITH_MODEL")
    if not model:
        sys.exit("render_golden.py: set DS4_ORNITH_MODEL to the 23G ICE GGUF")
    template = read_chat_template(model)
    first_line = template.splitlines()[0] if template else ""
    if TEMPLATE_VERSION not in first_line:
        sys.exit(f"render_golden.py: expected template {TEMPLATE_VERSION}, found {first_line!r}")
    render = make_renderer(template)
    info = (f"template_version: {TEMPLATE_VERSION}\n"
            f"sha256: {hashlib.sha256(template.encode('utf-8')).hexdigest()}\n"
            f"source: {os.path.basename(model)}\n"
            f"jinja2: {render.jinja2_version}\n")
    os.makedirs(GOLDEN, exist_ok=True)
    outputs = {"TEMPLATE.txt": info}
    for name, case in CASES.items():
        spec = case_spec(case)
        outputs[name + ".txt"] = render(spec["template"])
        outputs[name + ".input.json"] = json.dumps(spec, ensure_ascii=False, indent=1) + "\n"
    stale = []
    for fname, text in outputs.items():
        path = os.path.join(GOLDEN, fname)
        if check:
            try:
                with open(path, encoding="utf-8", newline="") as f:
                    if f.read() != text:
                        stale.append(fname)
            except FileNotFoundError:
                stale.append(fname)
        else:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(text)
    if check and stale:
        sys.exit("render_golden.py: out of date: " + ", ".join(sorted(stale)))
    print(f"render_golden.py: {'checked' if check else 'wrote'} {len(CASES)} cases in {GOLDEN}")


if __name__ == "__main__":
    main(sys.argv[1:])
