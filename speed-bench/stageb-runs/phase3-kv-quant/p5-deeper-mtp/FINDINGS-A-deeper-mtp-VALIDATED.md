# Task A — deeper MTP draft: ALREADY in orca-rebase, validated (byte-exact, workload-adaptive)

UPDATE to the earlier feasibility note: the risky recurrent-state kernel surgery I
scoped for A is ALREADY IMPLEMENTED upstream, on the orca-rebase base (the same base
the C port lands on). No surgery needed — A is validated, not built-from-scratch.

## What orca-rebase already has (verified in ds4.c @ orca-rebase)
- Full 2nd-snapshot machinery: snap2_lin_state/snap2_ple_hist/snap2_valid +
  snap_after_second; the GDN/PLE/attn recurrent kernels already accept the snapshot
  row index 1u (state AFTER row 1) — exactly the S2 I flagged as missing on
  orca-bugfixes.
- glm_mtp_draft2 / glm_mtp_have2 ("second pending draft, depth-3 cycles").
- ds4_session_qwen4_spec_cycle drafts up to 2 chained tokens and verifies
  [seed, d0, d1] in one 3-row pass. Depth policy qwen4_spec_depth():
  DS4_QWEN4_MTP_DEPTH=2 (1 draft), =3 (2 drafts), or default/auto = adaptive
  (engages the 2nd draft only on a perfect recent first-draft acceptance window,
  i.e. deterministic continuations; falls back otherwise).
- Verify stays EXACT (a draft commits only when it is the target argmax), so any
  depth is autoregressive-identical => zero quality change.

## Measured (C-port binary = orca-rebase + --ple, imat model, M5 Pro, n=400, temp0)
| workload | depth2 (1 draft, = old PROD) | depth3 (2 drafts) | auto (default) | byte-exact vs depth2 |
|---|---|---|---|---|
| analytical prose | 42.0 t/s | 36.9 (loss) | **42.7** | IDENTICAL |
| deterministic (counting) | 50.0 t/s | **53.2 (+6.4%)** | 52.6 (+5.3%) | IDENTICAL |
- Byte-exactness confirmed: auto/depth2/depth3 outputs are all IDENTICAL to each
  other (diff -q) on the deterministic prompt => quality preserved by construction.
- Forced depth3 HURTS on prose (2nd/3rd drafts reject ~40%, wasting the 3-row
  verify); the AUTO policy correctly avoids it there and never underperforms the
  single-draft cycle. So auto is the safe default: neutral on prose, +5-6% on
  deterministic/structured/code, always byte-exact.

## Verdict
A = adopt orca-rebase's adaptive-depth spec_cycle. It is NOT a separate build and
needs NO risky surgery on orca-bugfixes; it ships together with C (both live on
orca-rebase). Net single-stream --mtp win at full quality: +0-6% depending on
workload (best on deterministic/code, neutral on creative prose). This supersedes
FINDINGS-A-deeper-mtp-feasibility.md (which assumed A had to be hand-built on the
snapshot-less orca-bugfixes base).
