# Phase 1 — Tier-1 latent-bug fixes (applied to PROD orca-bugfixes, validated)

Four lineage-agnostic fixes to real latent bugs found in the deep-search
(receipt ../upstream-solutions-sweep/). All in PROD source
`~/.local/share/ai-gateway/ds4-metal`. Metal shaders are runtime-compiled
(exe-relative dir), so the two .metal fixes need no rebuild; the two C fixes
rebuilt `ds4-server`+`ds4-agent` (ds4.c untouched → CORE_OBJS not recompiled).

## Fixes
1. **qwen4_softplus precision** (`metal/qwen4.metal:17`). Was `log(1.0f+exp(x))`
   in [-20,20]; the `+1` loses precision when exp(x) is small (x in ~[-20,-7]),
   diverging from the CPU reference `qwen4_ref_softplus` (log1pf) in the GDN decay
   gate — a RECURRENT state feeding every step to 220K ctx. This Metal target has
   **no `log1p` builtin** (compile error `undeclared identifier 'log1p'`), so the
   fix uses the Kahan correction `log(u)*(e/(u-1))` (u=1+e) which recovers the lost
   precision using only `log`. Kept the x>20 / x<-20 guards.
2. **isfinite guard in `kernel_qwen4_moe_reduce`** (`metal/qwen4.metal`). The
   expert-weighted `acc += weights*part` had no finiteness check; one NaN/Inf
   partial from a corrupted expert GEMV poisons the reduced output AND the
   hyper-connection residual R (cf #1025 silent tool-call corruption). Now each
   partial is added only `if (isfinite(p))`. Finite partials add in the same
   order → **bit-identical in the healthy path**; only diverges when a partial is
   non-finite (where it now drops the bad slot instead of poisoning).
3. **ignore_eos honored under qwen4 `--mtp`** (`ds4_server.c:~13480`). The qwen4
   MTP path (`ds4_session_qwen4_spec_cycle`) returns a target-verified block and
   discards eos_token (dispatcher casts `(void)eos_token`); the server block-
   consumption loop then applied `ds4_token_is_stop_for_think_mode` UNCONDITIONALLY,
   so an MTP block could early-terminate on eos even when the client set
   ignore_eos (only the first token was protected, via argmax_ignoring_eos). Fix:
   when `j->req.ignore_eos`, truncate the block at the stop token WITHOUT marking
   generation stopped; the existing rewind (`block_start+kept`, ds4_server.c:13692)
   rolls back the session and the next outer iteration re-samples that position
   with argmax_ignoring_eos — mirroring the plain-decode path AND the DSpark path's
   ignore_eos handling. toks[0] is always non-stop (chosen by argmax_ignoring_eos),
   so kept>=1 → progress guaranteed, no infinite loop. No signature change to
   spec_cycle needed (cleaner than the planned thread-through; reuses the rewind).
   CLI/agent don't expose ignore_eos → server-only bug.
4. **#1050 clearer tool-call error** (`ds4_agent.c:~2216`). When `<parameter=...>`
   arrives where the tool name (`<function=NAME>`) is expected, the parser now says
   so specifically instead of the generic "expected <function=...>". Diagnostic
   string only.

## Validation (imat model, M5 Pro, ctx 4096/512, temp0/seed1)
- Shaders compile at runtime, coherent output (fibonacci w/ type hints+docstring).
- **Decode t/s unchanged**: post-fix --mtp 41.11/43.00/42.76 (mean 42.29) vs
  baseline 42.84; MTP acceptance 66.9% vs 67.9% (log1p shifts logits slightly, so
  drafts differ marginally — expected, more-accurate). Within run-to-run noise.
- **No quality regression** (frontier logits @512, verbose.txt prompt; dumps via
  ds4-bench, metal swapped to isolate the kernel change):
  | comparison | cosine | argmax |
  |---|---|---|
  | log1p effect, Q8 old-kernel vs new | 0.99508 | MATCH (321) |
  | log1p effect, imat old-kernel vs new | 0.99516 | MATCH (321) |
  | quant gap imat-vs-Q8, OLD kernel | 0.99085 | MATCH |
  | quant gap imat-vs-Q8, NEW kernel | 0.99040 | MATCH |
  The quant relationship is unchanged (Δcosine 0.0005, within the log1p's own
  perturbation); argmax preserved everywhere. log1p perturbs ~0.005 cosine at 512
  ctx (rms 0.24) and grows with context (GDN recurrent) — the intended accuracy
  gain concentrates at long ctx, argmax-stable at short ctx.

## Notes / deferred
- Router softplus (`metal/unary.metal:305`, `metal/dsv4_misc.metal:5188`, float4
  `select(log(1+exp),x,x>20)`) has the same precision shape but feeds MoE routing;
  changing it could shift expert selection (argmax risk) → OUT of scope, noted.
- glm down-family isfinite guard (`metal/moe.metal`) skipped: our model is qwen4exp
  and never routes through the glm kernels; the qwen4 reduce guard already catches
  every corrupted partial for our path.

## diskwrites: model volume read-only; dumps/logs under the worktree only.
