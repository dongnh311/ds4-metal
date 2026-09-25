#!/usr/bin/env python3
"""Prompts for the prefill-pipe checks, cut from the pinned filler text:
a short chat and about 5K, 36K and 134K tokens (the filler runs ~3.4 chars/token).
usage: make_prompts.py OUT_DIR"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILLER = os.path.join(ROOT, "speed-bench", "promessi_sposi.txt")
SIZES = {"p5k": 17_000, "p36k": 123_000, "p134k": 456_000}
QUESTION = "\n\nSummarize the text above in two sentences."
CHAT = "Viết một đoạn văn khoảng 120 chữ giới thiệu Hà Nội cho khách du lịch."


def build(filler):
    """Prompt name -> text."""
    prompts = {"chat": CHAT}
    for name, chars in SIZES.items():
        if len(filler) < chars:
            raise ValueError(f"filler has {len(filler)} chars, need {chars}")
        prompts[name] = filler[:chars] + QUESTION
    return prompts


def main(out_dir):
    with open(FILLER, encoding="utf-8", errors="replace") as fp:
        prompts = build(fp.read())
    os.makedirs(out_dir, exist_ok=True)
    for name, text in prompts.items():
        with open(os.path.join(out_dir, name + ".txt"), "w", encoding="utf-8") as fp:
            fp.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
