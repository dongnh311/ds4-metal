"""Shared helpers for the Ornith (qwen35moe) correctness gate.

References come from llama.cpp's llama-server on the same GGUF; ds4 output
comes from `ds4 --dump-logprobs`.  Both are reduced to the same step form,
{"selected": id, "top": [[id, logprob], ...]}, and compared greedily: the
selected tokens must match, a probable token (log-prob >= PROBABLE) on
either side must be in both top-k lists, and the reference's probable
tokens must agree within a tolerance, until the reference reaches a near-tie
(top-1/top-2 gap below `tie`).  The tie step's probable tokens are still
checked; from that step on a different but equally valid continuation is
allowed.  The tolerances are calibrated from llama.cpp's own
Metal-versus-CPU spread instead of being picked by hand.
"""
import json
import os

FLOOR = 0.05  # log-prob; the smallest tolerance and tie gap the gate uses

# Tail ranks (roughly log-prob < -2, i.e. under ~13% probability) carry
# activation-quantization noise from the CPU backend that says nothing about
# correctness; only probable tokens are where an implementation error would
# actually show up, so max_delta/tol are only computed over those.
PROBABLE = -2.0


def load_prompts(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def prompt_text(prompt, root):
    """The prompt's raw text.  A file-backed prompt takes the file's first
    `chars` characters; with `repeat_chars` the file's opening follows again
    after a blank line, so the continuation copies from far back."""
    if "text" in prompt:
        return prompt["text"]
    with open(os.path.join(root, prompt["file"]), encoding="utf-8") as f:
        text = f.read()
    if "repeat_chars" in prompt:
        return text[: prompt["chars"]] + "\n\n" + text[: prompt["repeat_chars"]]
    return text[: prompt["chars"]]


def llama_server_groups(prompts, cpu):
    """Prompts grouped by llama-server launch: the default physical batch
    first, then one server per pinned "llama_ubatch" value (a long prompt
    whose batched Metal prefill is not precise enough to be the reference is
    recorded one token per ubatch).  The CPU backend skips file-backed
    prompts."""
    groups = {}
    for p in prompts:
        if cpu and "file" in p:
            continue
        groups.setdefault(p.get("llama_ubatch"), []).append(p)
    return sorted(groups.items(), key=lambda g: (g[0] is not None, g[0] or 0))


def llama_steps(completion):
    steps = []
    for s in completion["completion_probabilities"]:
        tops = s.get("top_logprobs") or []
        steps.append({"selected": s["id"], "top": [[t["id"], t["logprob"]] for t in tops]})
    return steps


def ds4_steps(dump):
    steps = []
    for s in dump["steps"]:
        steps.append({"selected": s["selected"]["id"],
                      "top": [[t["token"]["id"], t["logprob"]] for t in s["top_logprobs"]]})
    return steps


def _gap(step):
    lps = sorted((lp for _, lp in step["top"]), reverse=True)
    return lps[0] - lps[1] if len(lps) >= 2 else float("inf")


def _check_probable(ref, got, res):
    """A token probable on either side must be in both top-k lists; the first
    one that is not is returned as the failure reason.  The log-prob deltas
    are taken over the reference's probable tokens, the set the tolerance is
    calibrated on (adding ds4-only probable tokens would move a CPU-only
    probable token into the Metal-vs-CPU calibration and widen it); each
    pair found adds to res["checked"] and res["max_delta"]."""
    theirs = dict((tid, lp) for tid, lp in ref["top"])
    mine = dict((tid, lp) for tid, lp in got["top"])
    missing = ""
    for tid, lp in ref["top"]:
        if lp < PROBABLE:
            continue
        if tid not in mine:
            missing = missing or f"token {tid} (reference {lp:.4f}) missing from ds4's top list"
        else:
            res["checked"] += 1
            res["max_delta"] = max(res["max_delta"], abs(mine[tid] - lp))
    for tid, lp in got["top"]:
        if lp >= PROBABLE and tid not in theirs:
            missing = missing or f"token {tid} (ds4 {lp:.4f}) missing from the reference's top list"
    return missing


def compare(ref_steps, got_steps, tol, tie):
    res = {"ok": True, "compared": 0, "checked": 0, "stopped_at_tie": None, "max_delta": 0.0,
           "first_mismatch": None, "reason": ""}
    for i, ref in enumerate(ref_steps):
        at_tie = _gap(ref) < tie
        if at_tie:
            res["stopped_at_tie"] = i
        if i >= len(got_steps):
            if at_tie:
                return res
            res.update(ok=False, first_mismatch=i, reason=f"ds4 stopped after {len(got_steps)} steps")
            return res
        got = got_steps[i]
        missing = _check_probable(ref, got, res)
        if missing:
            res.update(ok=False, first_mismatch=i, reason=f"step {i}{' (tie)' if at_tie else ''}: {missing}")
            return res
        if at_tie:
            # The tie excuses a different selection from here on, not the
            # probable-token log-probs of the tie step itself.
            if res["max_delta"] > tol:
                res.update(ok=False, first_mismatch=i,
                           reason=f"step {i} (tie): logprob delta {res['max_delta']:.4f} > {tol:.4f}")
            return res
        if got["selected"] != ref["selected"]:
            res.update(ok=False, first_mismatch=i,
                       reason=f"step {i}: selected {got['selected']} != {ref['selected']}")
            return res
        if res["max_delta"] > tol:
            res.update(ok=False, first_mismatch=i,
                       reason=f"step {i}: logprob delta {res['max_delta']:.4f} > {tol:.4f}")
            return res
        res["compared"] = i + 1
    return res


def verdict(res):
    """Gate status for one prompt: a passing comparison that checked no
    probable-token pair proved nothing, so it fails as vacuous."""
    if not res["ok"]:
        return "FAIL"
    return "ok" if res["checked"] > 0 else "FAIL vacuous"


def calibrate(metal, cpu):
    delta, gap = 0.0, 0.0
    for m_steps, c_steps in zip(metal, cpu):
        res = compare(m_steps, c_steps, tol=float("inf"), tie=0.0)
        delta = max(delta, res["max_delta"])
        if res["first_mismatch"] is not None and res["first_mismatch"] < len(m_steps):
            gap = max(gap, _gap(m_steps[res["first_mismatch"]]))
    return {"tol": max(FLOOR, 3.0 * delta), "tie": max(FLOOR, 2.0 * gap),
            "observed_max_delta": delta, "observed_divergence_gap": gap}
