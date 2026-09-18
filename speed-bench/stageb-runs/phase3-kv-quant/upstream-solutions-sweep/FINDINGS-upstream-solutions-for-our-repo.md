# Deep-search of antirez/ds4 PRs+issues: solutions/risks for our qwen4exp light model

Three parallel agents swept antirez/ds4 (PRs + issues) via `gh`, cross-checking
claims against our own source. Filter: Apple M5 Pro 64GB, Metal, Qwen3.8-Flash-Next
(qwen4exp), IQ2/Q2_K experts + Q4_K+imatrix dense (attn q/k/v/o at Q8), MTP, external
Q4_1 PLE sidecar (--ple), resident zero-swap 8K-220K, ds4-server behind an ai-gateway.
Key: our PROD is the orca-bugfixes lineage (has --ple); upstream dropped --ple
(self-contained n-gram) — so lineage-COUPLED wins need a port, but the items below
in Tier 1/2 are lineage-AGNOSTIC (files exist in our PROD; fixes apply directly).

## TIER 1 — cheap fixes to REAL latent bugs in OUR code (code-verified in PROD)
1. **ignore_eos silently broken under qwen4 --mtp.** ds4.c qwen4 speculative branch
   casts `(void)eos_token` (PROD ds4.c:78744/78752/78766/78814); `ds4_session_qwen4_
   spec_cycle` (70178) has ZERO eos logic; the server protects only the first token
   (ds4_server.c:13414), so an MTP-accepted block can contain a real EOS and
   terminate early even with ignore_eos:true. Hits eval/bench clients through the
   ai-gateway. No upstream fix (GLM's #938 also open) — original fix, small: thread
   eos_token/ignore_eos into the qwen4 draft-accept.
2. **qwen4_softplus GPU/host precision gap.** metal/qwen4.metal:17 uses naive
   `log(1.0f+exp(x))` (loses precision for ~-20<x<-3.5); the CPU ref
   qwen4_ref_softplus (ds4.c:67712) already uses accurate log1pf. Feeds the GDN/SSM
   decay gate (qwen4.metal:585,3253) on ~75% of layers every decode step, recurrent
   state to 220K ctx. Fix = #1044's log1p technique (log1p(exp(x))). Small, low risk.
3. **No isfinite guard in kernel_qwen4_moe_reduce.** metal/qwen4.metal ~2530: the
   expert-weighted sum `acc += weights*part` has no finiteness check (same shape as
   #1025's confirmed CUDA NaN bug whose failure mode was SILENT tool-call-name
   corruption). One NaN partial from any IQ2/Q2_K expert poisons the token. 5-line
   guard, measured-free upstream. (Also glm q4_K down family in moe.metal.)
4. **Qwen tool-call error message (#1050, PR c19ab8a).** Model emits <parameter=NAME>
   where <function=NAME> belongs (~5% stanzas); parser rejects correctly but the
   message doesn't say where the tool name belongs -> model burns tokens re-gen
   already-correct output. We run agent_qwen_tool_parse (ds4_agent.c:2202). Trivial
   diagnostic-string fix.

## TIER 2 — risks to TEST (regression gates we lack)
5. **#1045 repetition-loop:** greedy/low-temp verbatim loops on long analytical
   prompts with imatrix-Q4K (reported on DeepSeek, same quant class). We validated
   argmax only to 4K; run greedy temp0 32K long-analytical stress on our shipped
   imatrix-Q4K. Optional insurance: adopt #195's opt-in sampler repetition guard.
6. **M5 tensor-drift recheck:** same bug as our finding (#946/#947), upstream measured
   LARGER (max_abs 7.27 logits M5 Max vs our ~1-1.5 M5 Pro). PR #947 CLOSED unmerged,
   branch a11bf74 ~155 commits stale -> NOT inheritable via pull; self-maintain.
   Recheck "argmax-stable to 4K" at long ctx (220K autoregressive compounding).
7. **Decode stalls in long streaming tool sessions (#931/#783, PR #787):** server
   buffers across </think> before streaming; generic to ds4_server, matches our
   ai-gateway+tools+streaming shape. Stress-test; port #787 to qwen tags if it repros.
8. **Run test_mtp_verify_depth (ds4_test.c:7095) with DS4_TEST_MTP against our GGUF:**
   qwen4 has verify_rows_exact + recent hardening (923b819, 2026-09-14) so #938
   findings 1-2 look mitigated, but the test self-skips unless run — prove it.

## TIER 3 — throughput wins (bigger effort / lineage-coupled)
9. **#1056** — IQ2_XXS/Q2_K activation row-reuse + MTP state-prep: matches our exact
   expert quant, decode win independent of SSD-streaming; extract from the unmerged
   #1047 11-commit bundle. Medium.
10. **DS4_QWEN4_FLUSH_LAYER bench sweep on M5 Pro** — free; the queue-without-waiting
    command-buffer design (#1041/#1042 retrofit onto DeepSeek) is ALREADY our default;
    just tune the flush layer for M5 Pro.
11. **#258 HISA hierarchical indexer** — Qwen4 has a DSA-style indexer; gate ~196K sits
    in our ctx range; could preserve long-ctx decode. Scoping needed.
12. **#1062 batched decode (65x @16 streams)** — multi-session/ai-gateway lever; needs
    --ple ported forward (Path A; base origin/orca-rebase ready).
13. **d0b7434 (n-gram concurrency +6-9%)** — NOT applicable to our light model: our
    --ple PLE read is CPU mmap demand-paged (page faults), not the explicit-pread path
    d0b7434 parallelizes. Only relevant on the self-contained lineage.

## Already OK in our tree (no action)
- Tool-call decode-time protection (#999): we already have dsml_decode_tracker +
  qwen_tool_syntax + greedy forcing (ccea7688). GLM upstream still open.
- Paged-KV lazy-grow (#1010 equiv): our qwen4 pager already grows on demand.

## Recommended batch
Tier 1 (all four) is the clear win: small, lineage-agnostic, fixes real bugs in our
shipped PROD/HF code, no upstream dependency. Then Tier 2 tests (esp. #1045 stress +
ignore_eos regression) as gates. Tier 3 is a separate throughput campaign.
