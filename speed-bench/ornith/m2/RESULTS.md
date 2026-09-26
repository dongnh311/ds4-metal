# Ornith M2: MTP (2026-09-26)

Branch `feature/ornith-m2`. Model: 23G ICE GGUF. Oracle: llama.cpp 0.5.0 (build 11146), `--spec-type draft-mtp --spec-draft-n-max 1`.

## Correctness (gate 1 item 3)

- `make test-qwen35-graph`: an MTP graph's logits equal the M1 graph's; each row of a 2-row verify equals plain decoding bit for bit at positions 31, 63, 2047, 2111, 3000; a rejected verify restored by the snapshot swap continues bit-identical; the catch-up K rows agree across prefill chunks 512 / 64 / 1 (per-row cosine >= 0.95; measured minimum 0.9946 for chunk 64 and 0.9906 per token; a one-row seed shift gives 0.30).
- `make test-qwen35-mtp`: 150 greedy cycles (101 accepted) commit the plain argmax sequence with bit-identical logits; forced accepts, a divergent prompt and the context end behave.
- `tests/ornith/test_mtp_cli.py` (`cli-mtp.txt`): plain one-shot, session and `--mtp` print identical text for 5 prompts x prefill chunks 1, 2, 64 and default, and for `long_copy` (9,371 tokens) at default and 64; 362 drafts accepted, 59 rejected.
- gate 1 default compare after the graph change: identical to `speed-bench/ornith/m1/compare-default.txt`.
- Each verify runs its attention, dense projections and GDN layers one row per dispatch (T=1 and T=2 pick different matvec kernels, and the GDN mixer fuses its input projections only for T=1); the experts stay batched.

## Acceptance vs llama.cpp (`accept.md`, n = 128 per prompt)

ds4 0.872 vs llama.cpp 0.869 overall.

## Decode speed (CLI, 256 tokens, M5 Pro)

| prompt | plain t/s | --mtp t/s | acceptance |
|---|---|---|---|
| en_contributing | 71.82 | 70.15 | 93.2% |
| code_c_pack | 71.70 | 70.55 | 94.7% |

The draft head reads the whole output matrix each cycle; the MTP draft vocabulary (M4 lever) is the next step for speed. The row-exact verify also re-reads the dense and GDN weights once per row; a multi-row matvec with T=1-identical arithmetic is the second M4 lever. At 93-95 % acceptance --mtp is break-even with plain decoding today.

## Qwen3.8

Full gate: see `QWEN_GATE.md`.
