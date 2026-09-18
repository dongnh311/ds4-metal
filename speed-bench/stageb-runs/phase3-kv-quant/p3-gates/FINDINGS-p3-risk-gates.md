# Phase 3 — Tier-2 risk gates (imat model, M5 Pro)

## Gate 1 — ignore_eos under --mtp: PASS (fix #3 validated)
See FINDINGS-ignore-eos-gate.md. Same-source A/B (fix on: 120 tokens; fix off,
rebuilt: 28-token early stop) confirms the bug was real and MTP-block-specific and
that fix #3 corrects it. Plain path always honored ignore_eos. Noted gap:
/v1/completions never parses ignore_eos (chat endpoint only) — pre-existing.

## Gate 2 — #1045 repetition-loop stress: PASS (no degeneracy)
Greedy temp0, ctx 32768, n=2000, prompt inviting a long enumeration
("List and explain 50 distinct algorithms..."). Output (rep_stress.log):
- max consecutive identical lines: 1 (none)
- top repeated 8-gram: 1x (no verbatim loop)
- distinct-line ratio: 128/128 = 1.000
- produced 19 distinct, coherent algorithm entries (Bubble Sort..Kruskal) each
  with complexity + use case, cut cleanly at the n=2000 budget.
No verbatim-loop degeneracy on the imat model → the #195 sampler repetition guard
is NOT needed for our shipped operating point.

## Gate 3 — MTP verify-exactness: PASS (byte-identical)
The legacy `test_mtp_verify_depth` (tests/ds4_test.c:6826) targets a separate
MTP-head model at draft depth>2 and self-skips for our embedded-nextn (depth=1)
qwen4 model, so an architecture-appropriate proof was used instead: greedy temp0
output WITH `--mtp` vs WITHOUT was byte-IDENTICAL (exact_mtp.txt == exact_plain.txt,
1168 chars). The greedy MTP accept commits only argmax-matching drafts
(sample_argmax(rows)==d), so verify is autoregressive-exact by construction —
confirmed empirically.

## Gate 4 — M5 tensor-route drift, long-ctx recheck: argmax STABLE to 8192
Prior session covered <=4K (argmax-stable, cosine 0.9885@4096). Extended here to
8192 via ds4-bench frontier dump, default vs --quality (quality metadata on=false/
off=true confirms the mode switched; note --quality also toggles non-tensor
kernels, so this is default-vs-quality, not a pure tensor isolation — same caveat
as prior):
- @8192: cosine 0.99460, rms 0.372, **max_abs 4.11** (drift grows with ctx),
  **argmax MATCH (17=17)**, top5 overlap 3/5 (minor tail reordering only).
The top token is still preserved at 8K; drift magnitude grows but has not flipped
the argmax. Consistent with the prior verdict: real but not argmax-breaking on M5
Pro at practical contexts. a11bf74 (withhold tensor route on M5) remains a
not-urgent, self-maintained mitigation for maximum long-ctx fidelity (PR #947
closed/stale, not inheritable).

## diskwrites: model volume read-only throughout; logs/dumps under the worktree.
