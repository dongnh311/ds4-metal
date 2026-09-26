# Ornith-1.5-35B-A3B (`qwen35moe`) on ds4 Metal — design

Date: 2026-09-25. Branch: `feature/ornith-qwen35moe`. Status: design approved in
conversation section by section; this document is the written spec for review.

## 1. Goal

Run Ornith-1.5-35B-A3B Abliterated from the llama.cpp `qwen35moe` GGUFs of
[gbuzhf/Ornith-1.5-35B-A3B-Abliterated-CyberTiel-Calibrated-MTPv2-ICE-GGUF](https://huggingface.co/gbuzhf/Ornith-1.5-35B-A3B-Abliterated-CyberTiel-Calibrated-MTPv2-ICE-GGUF)
inside the ds4 engine, as a new model family next to Qwen3.8-Flash-Next
(`qwen4exp`) and DeepSeek V4.1, sharing the same binaries, server and gateway
slot.

The user wants all three outcomes, checked as ordered gates:

1. **Correct.** ds4 reproduces llama.cpp on the same GGUF (section 8).
2. **Quality (B).** The ICE 23G/25G build served by ds4 scores at least the
   current gateway Ornith (Shiftedx MLX abliterated, index 86.6 in
   `AI-Gateway-MLX/reports/ornith-abliterated-vs-current-2026-09-09`), aiming
   at the regular Ornith's 92.3.
3. **Speed (A).** Decode with MTP and prefill are at least as fast as Ornith on
   oMLX on the same machine (about 54.5 t/s decode, 31K-token first turn in
   about 19 s, `reports/ornith15-vs-qwen38-2026-08-21`), at 2K, 32K and 128K.
4. **One runtime (C).** After v2, deploy through `prod/<feature>-YYYYMMDD` and
   move the gateway Ornith slot from oMLX to ds4 with the capabilities the slot
   has today (vision, tools, thinking, 262K context).

Hard constraint: the Qwen3.8 production path stays byte-identical and as fast
as today (section 7).

### Scope

- **v1:** text model, own MTP head, `ds4-server` with tool calls and thinking,
  live prefix reuse and disk KV checkpoints, Q4_K and Q5_K routed experts,
  one session, Metal only.
- **v2:** vision through the Ornith mmproj (try reusing `metal/qwen4_vision.metal`),
  then the gateway switch (goal 4).
- **Out of scope:** SSD expert streaming (the model is about 21 GiB resident),
  CUDA/ROCm/CPU inference, distributed/TP, batched sessions, DFlash, the
  19G/21G tiers (IQ4_XS/IQ3_S experts), audio.

## 2. Verified facts

Source: the 23G GGUF header parsed by HTTP range reads (the parse accounts for
every byte of the 22,836,518,208-byte file), the HF config of the source
checkpoint, and llama.cpp master `src/models/qwen35moe.cpp` (f805c57a).

| field | value |
|---|---|
| `general.architecture` | `qwen35moe` |
| layers | `block_count` 41 = 40 trunk + 1 MTP block (`nextn_predict_layers` 1, `blk.40`) |
| embedding | 2048, vocab 248320, untied `output.weight` |
| layer pattern | full attention where `il % 4 == 3` (10 layers), Gated DeltaNet elsewhere (30 layers); `full_attention_interval` 4 |
| full attention | 16 query heads, 2 KV heads, head dim 256; `attn_q` rows interleave `[q 256 | gate 256]` per head; q/k RMSNorm; sigmoid output gate |
| RoPE | NEOX rotate-half over the first 64 dims (`rope.dimension_count` 64), base 1e7; the MRoPE sections [11,11,10,0] reduce to plain partial RoPE for text |
| GDN | 16 K heads x 128, 32 V heads x 128, conv kernel 4 over 8192 channels, L2-normalised q/k, delta rule with `exp(g)` decay, output `RMSNorm_128(o) * silu(z)` |
| MoE | 256 experts, top 8 of a softmax over all experts, renormalised, expert FF 512; one shared expert (FF 512) scaled by `sigmoid(ffn_gate_inp_shexp . x)` |
| norms | RMSNorm, eps 1e-6 |
| context | 262144 |
| tokenizer | `gpt2` BPE, pre-tokenizer `qwen35`; bos = pad = 248044, eos = 248046 `<|im_end|>`, `add_bos_token` false; `<think>` 248068, `</think>` 248069 |
| chat template | embedded "Qwen-Sharp v22.4.1" (29,674 chars); thinking on by default, the generation prompt opens `<think>\n` |

GGUF conversion facts the engine must follow (llama.cpp converter):

- Every `*norm.weight` except the GDN `ssm_norm` already has +1 folded in, so
  the engine multiplies by the stored weight as is.
- `ssm_a` stores `-exp(A_log)`.
- GDN value heads are stored tiled: value head `j` pairs with key head `j % 16`
  (V rows of `attn_qkv`, conv channels, `attn_gate`, `ssm_alpha`, `ssm_beta`,
  `ssm_a`, `ssm_dt`, input columns of `ssm_out`).

Tensor types in the 23G tier: routed experts Q5_K in layers 0-14 and Q4_K in
15-40 (gate, up and down share a type per layer); `attn_k`/`attn_v` F16;
`token_embd`, `output`, attention, GDN projections and shared experts Q8_0;
norms, router, `ssm_*` scalars and conv F32. The 25G tier uses the same types
(Q5_K in 35 blocks).

Byte budget (23G): routed experts 18.28 GiB, GDN 1.01, `token_embd` 0.50,
`output` 0.50, MTP block 0.46, attention 0.29, shared experts 0.13, routers
0.08; total 21.26 GiB. One decoded token reads about 2.58 GiB of weights, an
upper bound near 90 t/s on the M5 Pro before MTP.

Existing work, checked and not reused as a base:

- antirez/ds4 PR #844 (`ornith15`, audreyt) builds a parallel Qwen3.5 stack on
  a base (84cc882) that predates `qwen4exp`. It conflicts in 17 files, has no
  disk KV for Qwen, no server MTP, and decode falls from about 100 t/s to 22 t/s
  at 10K context on an M5 Max. It has been inactive since 2026-08-25. Used only
  as a reference (nextn tensor map, Q5_K/Q6_K kernels, `tests/test_ornith15_bench.sh`).
- llama.cpp master supports `qwen35moe` with MTP (`--spec-type draft-mtp`,
  PR #22673), which makes it a usable oracle.

## 3. Architecture

### Family and detection

- New `DS4_MODEL_FAMILY_QWEN35_MOE`, variant `DS4_VARIANT_QWEN35_MOE` and shape
  `DS4_SHAPE_QWEN35_MOE` in `ds4.c` with the values of section 2.
- Selected when `general.architecture == "qwen35moe"`. A dedicated validator,
  `config_validate_qwen35moe_model`, reads the `qwen35moe.*` keys and checks
  each against the shape.

### Where the code lives

- The graph, one-shot generation and session helpers for the family go in a
  new `ds4_qwen35moe.inc`; the validator, weight binding and layout
  validation stay beside their qwen4 counterparts in `ds4.c` because
  `weights_bind()` runs before the graph block. The repo already includes
  `.inc` files this way (`ds4_streaming_hotlist.inc`, `ds4_qwen4_unicode.inc`).
- `ds4.c` gets one branch per dispatch point (engine open, session
  create/free, forward, speculative cycle, payload save/load, context cap).
  Graph-level code lives in the `.inc`; session-level code (sync/eval
  branches, the speculative cycle) stays in `ds4.c`, because the `.inc` is
  included before `struct ds4_session`.
- Metal changes are additive: new kernel entry points (or function constants)
  in `metal/qwen4.metal` and `metal/moe.metal`, and their wrappers in
  `ds4_metal.m` / `ds4_gpu.h`.

### Predicate split

Today `ds4_model_is_qwen4()` means two things: "uses the Qwen3.5 tokenizer,
ChatML and XML tool calls" and "runs the qwen4exp graph". It becomes:

- `ds4_model_uses_qwen35_text()`: tokenizer, special tokens, chat rendering,
  tool-call parsing, server syntax. True for both families.
- `ds4_model_is_qwen4()`: the qwen4exp graph only (Qwen3.8).

Every existing call site of the qwen4 predicates is classified in Appendix B.
Decisions that follow from that audit:

- **Session readiness flag.** The Ornith session reuses the
  `ds4_qwen4_gpu_graph` struct but has its own ready flag
  (`qwen35_graph_ready`) and never sets `qwen4_graph_ready`. The 13 places that
  test `qwen4_graph_ready` directly therefore never run qwen4 logic on an Ornith
  session. Of the features behind those places, directional steering and
  batching are refused for Ornith at open (section 6). Internal mechanisms such
  as the qwen4 stale-prefix replay are simply never reached from Ornith code
  paths.
- **Layer helpers.** `ds4_qwen4_layer_is_linear/_is_nextn` stay Qwen3.8-only.
  The family gets its own helpers (`il % 4 == 3` for attention, `il == 40` for
  the MTP block). Reused qwen4 code that calls the qwen4 helpers internally is
  not reused for Ornith.
- **No silent fallthrough.** Where a qwen4 branch is followed by a DeepSeek/GLM
  default (context estimate, one-shot generate, session create/sync/eval,
  speculative cycle, rewind), the default path dies with a clear message if it
  ever sees the Ornith family, so a missed DISPATCH branch cannot silently run
  DeepSeek code.

## 4. Components

### Weight binding

qwen35moe tensor names map onto the existing `ds4_layer_weights` fields:

| GGUF | field |
|---|---|
| `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d`, `ssm_norm`, `ssm_out` | `lin_qkv`, `lin_gate`, `lin_alpha`, `lin_beta`, `lin_a`, `lin_dt_bias`, `lin_conv`, `lin_norm`, `lin_out` |
| `attn_q`, `attn_k`, `attn_v`, `attn_output`, `attn_q_norm`, `attn_k_norm` | same names |
| `ffn_gate_inp`, `ffn_gate_inp_shexp`, `ffn_{gate,up,down}_exps`, `ffn_{gate,up,down}_shexp` | same names |

New fields: per-layer `attn_norm` and `post_attention_norm`, model-level
`output_norm`, and the MTP block's `nextn.eh_proj`, `nextn.enorm`,
`nextn.hnorm` and `nextn.shared_head_norm`. The MTP block reuses `token_embd`
and `output`.

Accepted types: routed experts Q4_K or Q5_K, gate and up of the same type;
dense projections and shared experts Q8_0; `attn_k`/`attn_v` F16 or Q8_0; norms,
router, `ssm_*` scalars and conv F32. Anything else fails the load.

### Graph buffers

The family reuses the `ds4_qwen4_gpu_graph` struct with its own allocation
function. That function allocates only what the family uses: the residual `R`
(`n_embd` wide), `mixed`, `blk`, GDN buffers and per-layer recurrent state and
conv history, attention buffers, the KV cache of the 10 attention layers and
the MTP block, MoE buffers and logits. Hyper-connection, PLE and indexer
buffers stay NULL, and the family's code never passes them anywhere.

### Layer loop

```
R = token_embd rows
for il in 0..39:
    mixed = RMSNorm(R) * attn_norm[il]
    blk   = il % 4 == 3 ? Attention(mixed) : GDN(mixed)
    R    += blk
    mixed = RMSNorm(R) * post_attention_norm[il]
    blk   = MoE(mixed)
    R    += blk
h = RMSNorm(R) * output_norm          (kept for the MTP block)
logits = output . h
```

The norm and add steps use the generic `ds4_gpu_rms_norm_weight_rows_tensor`
and `ds4_gpu_add_tensor`.

- **GDN.** `qwen4_graph_linear` is reused as is: conv, prep, scan and the
  per-layer snapshots used by speculative verify. The output step needs the
  `silu(z)` gate, so a `gdn_out` variant is added as a new kernel entry point;
  the Qwen3.8 `sigmoid` kernel is not touched. qwen4's tiled head mapping
  (`kh = h % Hk`) already matches the stored value-head order.
- **Attention.** A new `qwen35_graph_attention`:
  1. Q8_0 GEMV of `attn_q` into interleaved q/gate, F16 GEMVs of `attn_k` and
     `attn_v`.
  2. A prep kernel without the indexer: q/k RMSNorm, NEOX RoPE on the first
     64 dims, KV append. It is a new entry point beside the qwen4
     `attn_prep`, not a change to it.
  3. Dense attention over all positions through the existing `attn_decode` /
     `attn_mm` kernels (scale 1/16, GQA 8:1), with the qwen4 KV modes.
  4. `o *= sigmoid(gate)`, then the `attn_output` GEMV into `blk`.
- **MoE.** Reuse `qwen4_graph_moe`'s router and top-k kernel (softmax over
  256, top 8, renormalise, sigmoid shared-expert gate) and its mid/down
  kernels. Q5_K routed experts are new in the qwen4 kernels: add Q5_K row-dot
  and tile-GEMM dequant, ported from the GLM `block_q5_K` kernels in
  `metal/moe.metal`. The reduce runs with `n_hc = 0` into `blk`, then the
  residual add.
  *M1 deviation:* M1 ships the Q5_K row kernels only. Q5_K layers use the
  per-token row kernels at every prefill size (Q4_K layers switch to the
  tile GEMM above 64 tokens). The tiled Q5_K GEMM is deferred to M4, where
  prefill speed is measured.
- **MTP block (`blk.40`).**
  1. `x = eh_proj . concat[RMSNorm(embed(tok)) * enorm, RMSNorm(h) * hnorm]`,
     embedding half first; `h` is the trunk hidden after `output_norm` (the
     tensor the LM head reads, llama.cpp PR #24025).
  2. One full-attention layer with its own KV cache, then one MoE layer, using
     the layer-loop functions above with `il = 40`.
  3. `logits = output . (RMSNorm(x) * shared_head_norm)`.
  4. A deeper draft chains `RMSNorm(x) * shared_head_norm` (the step-3 head
     input) as the next step's `h`. M2 drafts one token per cycle; deeper
     drafts are an M4 lever.

### Fusions

Fused variants (norm+add, paired projections, reduce straight into `R`) come
after correctness. Each is kept only if output is unchanged and an interleaved
A/B shows a gain.

## 5. Data flow

- **Prefill** (chunked, `--prefill-chunk`):
  1. Embedding rows go into `R`.
  2. The 40 layers run over T rows.
  3. The last row's logits are computed.
  4. The MTP block runs a catch-up pass over the chunk: MTP KV row `p` comes
     from `(token_p, h_{p-1})` at RoPE position `p`, with `h_{-1} = 0`, as
     llama.cpp's draft-mtp does. Catch-up rows compute only their K/V; the
     trunk keeps its post-norm rows and carries the last one to the next
     forward. Qwen3.8 has no catch-up, so this is Ornith code.
- **Decode with MTP.** Each cycle drafts one token with the MTP block, then
  verifies `[current, draft]` in one T=2 pass, snapshotting GDN state and conv
  history after the first row. An accepted draft keeps both rows. A rejected
  draft restores the snapshot and keeps one. The cycle is
  `ds4_session_qwen35_spec_cycle` in `ds4.c`, shaped like the qwen4 cycle
  (whose snapshot helpers need hyper-connections and PLE); the GDN kernels'
  after-first-row snapshot is reused. The verify runs its attention, dense
  projections and GDN layers one row per dispatch (T=1 and T=2 pick different
  matvec kernels, and the GDN mixer fuses its input projections only for
  T=1), so each row equals plain decoding bit for bit; the experts stay
  batched. Drafts are accepted when they are the target argmax (greedy and
  opportunistic sampling); `--mtp-exact-sampling` is refused. Draft depth
  starts at 1 and follows measured acceptance. At temperature 0 the output
  equals plain decoding.
- **Sessions.** `ds4_session` reuses the qwen4 graph member, dispatched by
  family (Appendix B).
- **Live prefix reuse.** The server's session sync applies unchanged.
- **Disk KV checkpoints.**
  - A new payload tag, `DS4_QWEN35_PAYLOAD_TAG`, covers the 30 GDN recurrent
    states and conv histories plus the KV rows of the 10 attention layers and
    the MTP block.
  - The checkpoint header records `qwen35moe`, so Qwen3.8 and Ornith
    checkpoints can never load into each other. Existing Qwen3.8 checkpoints
    are unaffected.
- **KV cache.** F16 by default. KV is 20 KiB per token, about 5.4 GB at 262K,
  on top of 21 GiB of weights. kv-grow is not used in v1. The qwen4 FP8/Q4 KV
  modes remain available as a speed lever (section 8, gate 3).
- **Chat rendering.**
  - Reuse the Qwen3.8 ChatML and XML tool-call renderer.
  - Ornith differences, gated by family: thinking on by default with the
    generation prompt opened by `<think>\n`, thinking off rendered as
    `<think>\n\n</think>\n\n`, and the tool schema placement of the embedded
    template.
  - The Qwen3.8 "Reasoning effort is set to xhigh" system line is not added.
    It is added in three places (`encode_chat_prompt` through
    `qwen4_chat_system`, the agent's system tokens, and the server's
    `render_qwen_chat_prompt_text`). All three get the text from
    `ds4_qwen4_reasoning_effort_text()`, which gains a family check and returns
    NULL for Ornith. For Qwen3.8 the output does not change.
  - Server model-name aliases for Qwen3.8 (`qwen3.8-flash-next-*`) are not
    predicate sites; Ornith gets its own alias list.
- **Server.** Model id `ornith-1.5-35b-a3b` with the `-chat`, `-reasoner` and
  `-nothink` aliases, following the Qwen3.8 pattern.
  `SERVER_MODEL_SYNTAX_QWEN` for tool calls. Gateway registry changes belong to
  the deploy step after v2.

## 6. Error handling

- **Load.** Every metadata key and every tensor (presence, shape, type) is
  checked. The first mismatch fails the load with the key or tensor name, the
  expected value and the found value. IQ4_XS/IQ3_S experts fail with "expert
  type IQ4_XS not supported; use the 23G or 25G tier". Nothing falls back
  silently.
- **Unsupported modes.** `--ssd-streaming`, `--cuda`, `--rocm`, the CPU
  backend, multi-GPU `--gpu` placement, distributed/TP, `--batched-session`
  above 1, directional steering, `--ple` and `--vision` (v1) are refused at
  open with a message naming the option. The batched-session refusal sits in
  the open gate because the server does not refuse batching per family today.
- **Rewind.** A rewind on an Ornith session restores the GDN snapshot when one
  covers the target position. Otherwise it resets the recurrent state and
  conv history together with the checkpoint, so a later sync can never reuse
  a stale prefix.
- **Runtime.** Graph functions return false up the chain like the qwen4 path.
  If a target or MTP forward fails midway, the session is invalidated and the
  next request prefills again, so GDN state is never half updated. Requests
  beyond the context or with token ids outside the vocabulary are rejected.

## 7. Qwen3.8 isolation

1. **Additive kernels.** The silu GDN gate, the indexer-free attention prep and
   Q5_K support are new kernels, new entry points or function constants. The
   pipelines Qwen3.8 runs compile to the same code as today.
2. **Predicate split.** Every site is classified in Appendix B. QWEN4_ONLY
   sites are left alone.
3. **Separate knobs.** Family-level `DS4_QWEN4_*` knobs (KV modes, prefill
   chunk, PLE, MTP depth, SSD streaming, YaRN, kv-grow) are read only on the
   qwen4exp path; Ornith reads none of them, and its own knobs use a
   `DS4_QWEN35_*` prefix. The `DS4_QWEN4_*` switches inside the shared qwen4
   kernels and helpers (dense GEMV, fused GDN front and decode fusions,
   attention split, tiled and merge) choose between arithmetic-equivalent
   kernel paths and apply to both families by design. Tuning keyed on
   `n_embd == 2560`, `n_rank == 320` or `n_hc == 4` stays as is.
4. **Qwen gate.** Every commit that touches a shared file (`ds4.c` outside the
   `.inc`, `ds4_metal.m`, `ds4_gpu.h`, `metal/*.metal`, `ds4_server.c`) passes
   before merge:
   - `make test-qwen4-kernels test-qwen4-q2`;
   - the full-tier Qwen gate: replies byte-identical to PROD, needle hit,
     interleaved A/B at least 97% of PROD decode speed.

Operations follow the standing rules: one model process at a time, GPU time
coordinated with the V4.1 session, no `kill -9` of a hung Metal process, long
runs under `caffeinate` with a stall watcher.

## 8. Testing and acceptance gates

**Oracle.** llama.cpp installed from Homebrew, as a test tool only, running the
same GGUF at temperature 0. Before it is trusted, `llama-perplexity` on the
repo's `code.test.raw` (64 chunks, `n_ctx` 2048) must reproduce the published
PPL of the tier: 2.194208 x 1.0018 = 2.1982 for 23G.

### Gate 1: correct

1. **Kernel unit tests** against CPU references at small dims: Q5_K row-dot and
   tile GEMM, the silu `gdn_out`, the indexer-free attention prep, the MoE
   reduce with `n_hc = 0`, RMSNorm and add.
2. **Model against llama.cpp** on about 12 prompts (English, Vietnamese, code,
   tool call, one 8K prompt):
   - Greedy tokens must match until the first near-tie, meaning a top-1/top-2
     logit gap below a threshold.
   - The top-20 log-probabilities at each position must agree within a
     tolerance.
   - Both the near-tie threshold and the tolerance come from llama.cpp's own
     Metal-versus-CPU spread on the same prompts, measured once and recorded
     with the test. They are not picked by hand.
   - Comparison method: llama-server `n_probs` against `ds4 --dump-logprobs`.
3. **MTP.** At temperature 0, `--mtp` output is byte-identical to plain
   decoding, including cycles with rejected drafts.
4. **Sessions.** Save to disk KV, restore and continue: the output equals an
   uninterrupted run. A Qwen3.8 checkpoint is refused by Ornith and the other
   way round.
5. **Chat.** ds4 renderings match jinja2 renderings of the embedded template
   for a fixed conversation set: system prompt, tools, multi-turn with tool
   results, thinking on and off.

### Gate 2: quality (B)

- ds4-server on a staging port, never the live stack; 23G and 25G tiers.
- The gateway intelligence harness (`run_ab.py`) scores at least 86.6.
- The 7-case agentic matrix passes with no guard incidents.
- The abliteration smoke probes behave as before.

### Gate 3: speed (A)

- Interleaved A/B against Ornith on oMLX, same machine and prompts, at 2K, 32K
  and 128K context.
- Decode with MTP must be at least oMLX at each context. Prefill must be at
  least oMLX (31K first turn within about 19 s).
- F16 KV reads about 2.7 GB per token at 128K, about as much as the weights. If
  long-context decode falls below oMLX, switch to the existing FP8 KV mode,
  and only if gate 2 still passes with it.

### Gate Qwen3.8

The section 7 gate on every commit touching a shared file.

## 9. Delivery milestones (v1)

v1 is too large for one undivided run. Each milestone ends green on its own
tests and the Qwen gate before the next starts:

- **M1: plain inference correct.** Family, validator, binding, Q5_K and the
  new kernels, layer loop, output head, CLI plain decode. Passes gate 1
  items 1-2.
- **M2: MTP.** MTP block, prefill catch-up, speculative cycle. Passes gate 1
  item 3 and records acceptance.
- **M3: serving.** Predicate split (Appendix B), sessions, live prefix reuse,
  disk KV payload, chat rendering, server ids. Passes gate 1 items 4-5.
- **M4: acceptance.** Gate 2 and gate 3 measurements; FP8 KV only if gate 3
  needs it; report.

v2 (vision) and the gateway switch get their own spec and plan after M4.

## 10. Items the plan verifies first

- That qwen4exp uses the same GDN conventions as this file: `ssm_a = -exp(A_log)`,
  L2-normalised q/k, `1/sqrt(128)` output scale, conv without bias. If any
  differ, a flag is added (new entry point, not a change to the Qwen3.8 path).
- That the indexer-free attention prep can be a separate entry point sharing
  helpers with `attn_prep` without changing the Qwen3.8 kernel's codegen.
- The MTP catch-up during prefill, checked against llama.cpp's
  `graph_mtp` / `draft-mtp` acceptance on the same prompts.
- The branch base. `feature/ornith-qwen35moe` currently sits on
  `origin/develop` 3673192 (PROD). Local `develop` is ahead with unpushed V4.1
  work. The plan picks the base before any code lands.

## 11. Risks

- **Numerical drift against llama.cpp.** Accumulation order differs. Gate 1
  therefore compares greedy tokens up to near-ties and top-k probabilities
  within a calibrated tolerance, not bit equality.
- **Q5_K kernel speed.** 15 of 40 layers in 23G (35 in 25G) use Q5_K. A slow
  kernel shows up directly in gate 3.
- **Long-context attention.** Dense attention over 10 layers at 128K+ is
  bandwidth-heavy. FP8 KV is the prepared lever.
- **Shared-code regressions.** Mitigated by additive kernels, Appendix B and
  the Qwen gate on every shared-file commit.
- **Chat template drift.** The embedded template has many branches
  (reasoning effort, think on/off tokens). Golden renders cover the paths the
  gateway uses; unknown branches are rejected rather than guessed.

## Appendix A: tensor inventory (23G tier, 753 tensors)

| group | layers | ggml shape | type |
|---|---|---|---|
| `token_embd`, `output` | - | [2048, 248320] | Q8_0 |
| `output_norm` | - | [2048] | F32 |
| `attn_norm`, `post_attention_norm`, `ffn_gate_inp_shexp` | 0-40 | [2048] | F32 |
| `ffn_gate_inp` | 0-40 | [2048, 256] | F32 |
| `ffn_{gate,up}_exps` / `ffn_down_exps` | 0-40 | [2048, 512, 256] / [512, 2048, 256] | Q5_K (0-14), Q4_K (15-40) |
| `ffn_{gate,up}_shexp` / `ffn_down_shexp` | 0-40 | [2048, 512] / [512, 2048] | Q8_0 |
| `attn_qkv` / `attn_gate` | 30 GDN layers | [2048, 8192] / [2048, 4096] | Q8_0 |
| `ssm_alpha`, `ssm_beta` | 30 | [2048, 32] | F32 |
| `ssm_a`, `ssm_dt.bias` | 30 | [32] | F32 |
| `ssm_conv1d` | 30 | [4, 8192] | F32 |
| `ssm_norm` | 30 | [128] | F32 |
| `ssm_out` | 30 | [4096, 2048] | Q8_0 |
| `attn_q` | 3, 7, ..., 39, 40 | [2048, 8192] | Q8_0 |
| `attn_k`, `attn_v` | same | [2048, 512] | F16 |
| `attn_output` | same | [4096, 2048] | Q8_0 |
| `attn_q_norm`, `attn_k_norm` | same | [256] | F32 |
| `nextn.eh_proj` | 40 | [4096, 2048] | Q8_0 |
| `nextn.enorm`, `nextn.hnorm`, `nextn.shared_head_norm` | 40 | [2048] | F32 |

## Appendix B: qwen4 predicate call sites

Line numbers refer to commit 3673192. Definitions are at `ds4.c:1012`
(`ds4_model_is_qwen4`), `ds4.c:62242` (`ds4_session_is_qwen4`) and `ds4.c:72991`
(`ds4_engine_is_qwen4`). The 41 `#if DS4_HAS_QWEN4_*` compile guards are not
listed; Ornith code uses the same guards.

Buckets:
- **SHARED** switches to `ds4_model_uses_qwen35_text()` or its engine form.
- **DISPATCH** gets an Ornith branch next to the qwen4 one.
- **REFUSE** rejects Ornith in v1.
- **QWEN4_ONLY** is left unchanged.

| site | predicate | function | branch does | bucket | action |
|---|---|---|---|---|---|
| ds4.c:529, 538, 539 | family / variant enums | enums | enum values | QWEN4_ONLY | add `DS4_MODEL_FAMILY_QWEN35_MOE`, `DS4_VARIANT_QWEN35_MOE` |
| ds4.c:805, 844 | family / variant | `DS4_SHAPE_QWEN4_EXP`, `_MINI` | shape tables | QWEN4_ONLY | add `DS4_SHAPE_QWEN35_MOE` |
| ds4.c:1018, 1028 | model_is_qwen4 | `ds4_qwen4_layer_is_linear/_is_nextn` | layer pattern | QWEN4_ONLY | Ornith gets its own helpers |
| ds4.c:1024 | model_is_qwen4 | `ds4_qwen4_layer_is_ple` | PLE layer | QWEN4_ONLY | keep |
| ds4.c:5352, 5368 | model_is_qwen4 | `weights_have_output_head`, `_partial_` | HC output head check | DISPATCH | Ornith uses the plain `output_norm` + `output` branch |
| ds4.c:5646 | model_is_qwen4 | `weights_layer_has_required` | qwen4 layer check | DISPATCH | Ornith layer check |
| ds4.c:5904 | model_is_qwen4 | `weights_validate_layout` | qwen4 layout validator | DISPATCH | Ornith validator |
| ds4.c:7671 | model_is_qwen4 | `weights_bind_output` | binds `output_hc_*` | DISPATCH | plain `output_norm` branch |
| ds4.c:7848 | model_is_qwen4 | `weights_bind_layer` | qwen4 layer binder | DISPATCH | Ornith layer binder |
| ds4.c:7936, 7971 | model_is_qwen4 | `weights_bind` | trims MTP from trunk, binds it after | DISPATCH | include Ornith |
| ds4.c:7959 | model_is_qwen4 | `weights_bind` | PLE n-gram tensor | QWEN4_ONLY | keep |
| ds4.c:8501, 8529, 8547 | family QWEN4_EXP | SSD expert-cache helpers | streaming eligibility | QWEN4_ONLY | keep (SSD refused for Ornith) |
| ds4.c:39756 | model_is_qwen4 | `ds4_context_memory_estimate_with_prefill_mode` | KV/indexer/HC estimate | DISPATCH | Ornith estimate (no HC/indexer) |
| ds4.c:43316 | model_is_qwen4 | `bpe_tokenize_text` | qwen35 pre-tokenizer | SHARED | switch |
| ds4.c:43443 | model_is_qwen4 | `vocab_load` | ChatML/think/tool special ids | SHARED | switch |
| ds4.c:43629 | model_is_qwen4 | `encode_chat_prompt` | ChatML rendering | SHARED | switch; effort text gated (section 5) |
| ds4.c:43826 | model_is_qwen4 | `ds4_chat_append_message` | ChatML turns, tool_response | SHARED | switch |
| ds4.c:43892 | model_is_qwen4 | `ds4_chat_append_assistant_prefix` | assistant prefix, `<think>` | SHARED | switch |
| ds4.c:60309 | model_is_qwen4 | `generate_metal_graph_raw_swa` | CLI one-shot generate | DISPATCH | Ornith generate path |
| ds4.c:60734 | model_is_qwen4 | `engine_per_tier_graph_overhead_bytes` | multi-GPU placement | REFUSE | `--gpu` refused at open |
| ds4.c:63187 | model_is_qwen4 | `ds4_engine_mtp_draft_tokens` | MTP draft count | DISPATCH | Ornith MTP draft count |
| ds4.c:63456 | session_is_qwen4 | `ds4_session_payload_bytes` | payload size | DISPATCH | Ornith payload size |
| ds4.c:63949 | session_is_qwen4 | `ds4_session_save_payload` | payload save | DISPATCH | Ornith save |
| ds4.c:64336 | session_is_qwen4 | `ds4_session_load_payload` | payload load | DISPATCH | Ornith load |
| ds4.c:69098 | model_is_qwen4 | `ds4_engine_first_token_test` | first-token self test | REFUSE | refused for Ornith (not needed) |
| ds4.c:70304 | model_is_qwen4 | `engine_compute_entry_bytes` | multi-GPU placement | REFUSE | unreachable once `--gpu` is refused |
| ds4.c:71662 | model_is_qwen4 | `ds4_engine_open_internal` | backend/TP/distributed gate | REFUSE | Ornith open gate: Metal only (section 6) |
| ds4.c:71717 | model_is_qwen4 | `ds4_engine_open_internal` | `--vision` allowlist | REFUSE | unchanged; Ornith already rejected |
| ds4.c:71747 | model_is_qwen4 | `ds4_engine_open_internal` | Qwen3.8 vision weights | QWEN4_ONLY | keep |
| ds4.c:71811 | model_is_qwen4 | `ds4_engine_open_internal` | `--ple` sidecar | QWEN4_ONLY | keep; `--ple` refused for Ornith |
| ds4.c:71962 | model_is_qwen4 | `ds4_engine_open_internal` | inspect/first-token return, GPU map | DISPATCH | Ornith early block |
| ds4.c:72114 | model_is_qwen4 | `ds4_engine_open_internal` | `--mtp` needs embedded MTP | DISPATCH | include Ornith |
| ds4.c:72562 | model_is_qwen4 | `ds4_engine_open_internal` | CUDA derived artifacts | REFUSE | CUDA refused |
| ds4.c:72582 | family QWEN4_EXP | `ds4_engine_open_internal` | SSD initial spans | QWEN4_ONLY | keep |
| ds4.c:73274 | model_is_qwen4 | `ds4_chat_append_multimodal_message` | image+text ChatML | QWEN4_ONLY | keep; Ornith text falls to the shared text path |
| ds4.c:73323 | family QWEN4_EXP | `ds4_chat_append_multimodal_message` | image support check | REFUSE | unchanged |
| ds4.c:73955 | model_is_qwen4 | `ds4_session_create` | refuses CPU sessions | REFUSE | include Ornith |
| ds4.c:74029 | model_is_qwen4 | `ds4_session_create` | qwen4 graph/arena allocation | DISPATCH | Ornith session allocation |
| ds4.c:74470 | session_is_qwen4 | `ds4_session_free` | qwen4 graph free | DISPATCH | Ornith free |
| ds4.c:76618 | session_is_qwen4 | `ds4_session_sync_internal` | prefix reuse + prefill | DISPATCH | Ornith sync/prefill |
| ds4.c:78692 | session_is_qwen4 | `ds4_session_eval_internal` | one-token decode | DISPATCH | Ornith decode |
| ds4.c:80392 | session_is_qwen4 | `ds4_sessions_eval_batch_metal_supported` | native batch check | REFUSE | return false for Ornith |
| ds4.c:80812, 80969 | session_is_qwen4 | `ds4_sessions_eval_batch_native` | qwen4 batch flag, log label | QWEN4_ONLY | keep |
| ds4.c:81248, 81257 | session_is_qwen4 | `qwen4_batch_ensure_caps` | batched KV caps, arena | QWEN4_ONLY | keep |
| ds4.c:81325 | session_is_qwen4 | `ds4_sessions_eval_batch_speculative_argmax` | batched MTP verify | QWEN4_ONLY | keep; Ornith is never batched |
| ds4.c:85355, 85485 | session_is_qwen4 | CUDA batch paths | excluded from CUDA batching | REFUSE | CUDA refused |
| ds4.c:85562 | session_is_qwen4 | `ds4_session_eval_speculative_argmax_impl` | qwen4 spec cycle | DISPATCH | Ornith spec cycle |
| ds4.c:86432 | session_is_qwen4 | `ds4_session_eval_speculative` | sampled spec cycle | DISPATCH | Ornith spec cycle |
| ds4.c:86588 | session_is_qwen4 | `ds4_session_rewind` | snapshot restore / state reset | DISPATCH | Ornith rewind (section 6) |
| ds4.c:86652 | session_is_qwen4 | `ds4_test_qwen4_alloc_cap` | test hook | QWEN4_ONLY | keep |
| ds4_server.c:1215 | engine_is_qwen4 | `server_model_syntax_for_engine` | Qwen tool-call syntax | SHARED | switch |
| ds4_server.c:1223 | engine_is_qwen4 | `server_model_id_from_engine` | model id | DISPATCH | `ornith-1.5-35b-a3b` |
| ds4_server.c:14954 | engine_is_qwen4 | `send_models` | model id list | DISPATCH | Ornith id list |
| ds4_server.c:15749 | engine_is_qwen4 | `main` | enables qwen4 batched MTP | QWEN4_ONLY | keep |
| ds4_agent.c:413 | engine_is_qwen4 | `agent_tool_syntax_for_engine` | Qwen tool syntax | SHARED | switch |
| ds4_agent.c:5137 | engine_is_qwen4 | `agent_worker_build_system_tokens` | system effort message | SHARED | switch; effort text gated (section 5) |
| tests/ds4_test.c:165, 273, 371, 454, 7322 | engine_is_qwen4 | qwen4 session tests | skip unless Qwen3.8 | QWEN4_ONLY | keep; Ornith tests are new |
| tests/test_qwen4_ngram_state.c:164, tests/test_qwen4_prefill.c:60 | engine_is_qwen4 | `main` | assert Qwen3.8 | QWEN4_ONLY | keep |

Counts over 76 call sites: SHARED 8, DISPATCH 26, REFUSE 10, QWEN4_ONLY 32
(including 7 enum/shape lines and 7 test lines).

Also relevant, not predicate sites: 13 places test `s->qwen4_graph_ready`
directly (steering 74573/74597, stale replay 75172, batch check 80354, spec
85567/86434, and others). Section 3 explains why Ornith never sets that flag.
