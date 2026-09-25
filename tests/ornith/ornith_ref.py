"""Shared helpers for the Ornith (qwen35moe) correctness gate.

References come from llama.cpp's llama-server on the same GGUF; ds4 output
comes from `ds4 --dump-logprobs`.  Both are reduced to the same step form,
{"selected": id, "top": [[id, logprob], ...]}, and compared greedily: the
selected tokens must match and every probable token (reference log-prob >=
PROBABLE) in both top-k lists must agree within a tolerance, until the
reference reaches a near-tie (top-1/top-2 gap below `tie`).  The tie step's
probable tokens are still checked against the tolerance; from that step on a
different but equally valid continuation is allowed.  The tolerances are
calibrated from llama.cpp's own Metal-versus-CPU spread instead of being
picked by hand.
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
    if "text" in prompt:
        return prompt["text"]
    with open(os.path.join(root, prompt["file"]), encoding="utf-8") as f:
        return f.read()[: prompt["chars"]]


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


def compare(ref_steps, got_steps, tol, tie):
    res = {"ok": True, "compared": 0, "stopped_at_tie": None, "max_delta": 0.0,
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
        mine = dict((tid, lp) for tid, lp in got["top"])
        for tid, lp in ref["top"]:
            if lp >= PROBABLE and tid in mine:
                res["max_delta"] = max(res["max_delta"], abs(mine[tid] - lp))
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


def calibrate(metal, cpu):
    delta, gap = 0.0, 0.0
    for m_steps, c_steps in zip(metal, cpu):
        res = compare(m_steps, c_steps, tol=float("inf"), tie=0.0)
        delta = max(delta, res["max_delta"])
        if res["first_mismatch"] is not None and res["first_mismatch"] < len(m_steps):
            gap = max(gap, _gap(m_steps[res["first_mismatch"]]))
    return {"tol": max(FLOOR, 3.0 * delta), "tie": max(FLOOR, 2.0 * gap),
            "observed_max_delta": delta, "observed_divergence_gap": gap}
