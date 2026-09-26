#!/usr/bin/env python3
"""Ornith disk-KV checkpoints across server restarts (gate 1 item 4, live).

  python3 tests/ornith/test_server_kv.py OUT_DIR

Needs DS4_ORNITH_MODEL, a built Metal ./ds4-server and a free GPU.

Phase 1, --mtp: three conversations x three turns (tool-less, tools with the
reasoning omitted, tools with the reasoning echoed); the server restarts
before turn 3.  Every turn ends with stop and no tool call; turns 2 and 3
reuse more than 3000 cached tokens (turn 3 from disk).
Phase 2, the same cache without --mtp, replays the last turn of tools-echo
(its checkpoints are the newest; the 512 MB budget evicts older ones): the
--mtp checkpoint is refused with a clear log line and the prompt is
prefilled and answered.
Phase 3, a fresh cache: a plain server answers the same replay and stores
checkpoints without MTP, then a --mtp server refuses them the same way.
Hard: the three cold answers of phases 2-3 are byte-identical (greedy, same
prefill chunks, M2's --mtp output equals plain).  Soft: the arithmetic
answers, and the cold answer equal to phase 1's, which came through a
restored checkpoint whose KV was partly written by decode (rounding there
can differ at a near tie).
"""
import copy
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import serverlib as sl

TOOLS = [{"type": "function", "function": {
    "name": "lookup", "description": "Look up information only when explicitly requested.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]
FILLER = "The town archive records weather and routine shipping schedules. " * 300
ANSWERS = ["51", "53", "55"]


def server_args(cache, mtp):
    args = ["-c", "16384", "--prefill-chunk", "1024", "--kv-disk-dir", str(cache),
            "--kv-disk-space-mb", "512", "--kv-cache-min-tokens", "128",
            "--kv-cache-cold-max-tokens", "0", "--kv-cache-continued-interval-tokens", "1024",
            "--kv-cache-boundary-align-tokens", "128"]
    return (["--mtp"] if mtp else []) + args


def content(result):
    return result["choices"][0]["message"].get("content") or ""


def refused(srv, body, saved, save):
    """The best checkpoint for body is refused; the prompt is prefilled and answered."""
    result = sl.post("/v1/chat/completions", body, save=save)
    log = srv.new_log()
    sl.check("kv cache load failed" in log and saved in log, f"a checkpoint {saved} is refused")
    sl.check(result["choices"][0]["finish_reason"] == "stop" and content(result).strip() != "",
             "after the refusal the prompt is prefilled and answered")
    return content(result)


def main(argv):
    if len(argv) != 1:
        sys.exit(__doc__)
    out = pathlib.Path(argv[0]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "kv"
    cache.mkdir(exist_ok=True)
    log = out / "server.log"
    replay = None
    phase1_answer = None
    srv = sl.Server(log, server_args(cache, True))
    try:
        for name, has_tools, echo in [("tool-less-omit", False, False),
                                      ("tools-omit", True, False),
                                      ("tools-echo", True, True)]:
            history = [
                {"role": "system", "content": "You are a helpful assistant. Answer arithmetic yourself; "
                                              "do not call tools. Think briefly."},
                {"role": "user", "content": name + "\n" + FILLER +
                 "\nWhat is 17 multiplied by 3? Answer with just the number."}]
            for turn in range(3):
                if turn == 2:
                    srv.stop()
                    srv = sl.Server(log, server_args(cache, True))
                body = {"messages": history, "temperature": 0, "max_tokens": 512,
                        "reasoning_effort": "low", "stream": False}
                if has_tools:
                    body["tools"] = TOOLS
                stem = out / f"{name}-{turn + 1}"
                result = sl.post("/v1/chat/completions", body, save=stem)
                if name == "tools-echo" and turn == 2:
                    replay = copy.deepcopy(body)
                    phase1_answer = content(result)
                message = result["choices"][0]["message"]
                usage = result["usage"]
                cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                print(json.dumps({"case": name, "turn": turn + 1, "finish": result["choices"][0]["finish_reason"],
                                  "prompt": usage["prompt_tokens"], "cached": cached,
                                  "content": content(result)}), flush=True)
                sl.check(result["choices"][0]["finish_reason"] == "stop" and not message.get("tool_calls"),
                         f"{name} turn {turn + 1} ends with stop and no tool call")
                sl.soft_check(ANSWERS[turn] in content(result), f"{name} turn {turn + 1} answers {ANSWERS[turn]}",
                              f"{stem}.request.json")
                if turn > 0:
                    sl.check(cached > 3000 and usage["prompt_tokens"] - cached < 128,
                             f"{name} turn {turn + 1} reuses the checkpoint ({cached} cached)")
                assistant = {"role": "assistant", "content": content(result)}
                if echo:
                    assistant["reasoning_content"] = message.get("reasoning_content") or ""
                history += [assistant, {"role": "user",
                                        "content": "Now add 2 to your previous answer. Answer with just the number."}]
        srv.stop()
        text = log.read_text(errors="replace")
        sl.check("KV payload staging failed" not in text, "no payload staging failure")
        sl.check("session has no valid checkpoint to stage" not in text, "every stage had a checkpoint")
        sl.check(text.count("reason=continued") >= 9, "continued checkpoints were stored")
        sl.check("reason=evict" in text, "an evict checkpoint was stored")
        sl.check("kv cache evicted reason=disk-cache-full" in text, "the disk budget evicted old checkpoints")

        srv = sl.Server(log, server_args(cache, False))
        phase2 = refused(srv, replay, "saved with --mtp", out / "phase2")
        srv.stop()

        plain_cache = out / "kv-plain"
        plain_cache.mkdir(exist_ok=True)
        srv = sl.Server(log, server_args(plain_cache, False))
        result = sl.post("/v1/chat/completions", replay, save=out / "phase3-plain")
        sl.check(result["choices"][0]["finish_reason"] == "stop" and content(result).strip() != "",
                 "a plain server answers the replay and stores checkpoints")
        phase3_plain = content(result)
        srv.stop()
        srv = sl.Server(log, server_args(plain_cache, True))
        phase3_mtp = refused(srv, replay, "saved without --mtp", out / "phase3-mtp")
        srv.stop()

        sl.check(phase2 == phase3_plain == phase3_mtp, "the three cold answers are byte-identical")
        sl.soft_check(phase2 == phase1_answer, "the cold answer equals phase 1's answer from the restored checkpoint",
                      f"{out / 'phase2'}.request.json")
    finally:
        srv.stop()
    sl.report("ornith server kv")


if __name__ == "__main__":
    main(sys.argv[1:])
