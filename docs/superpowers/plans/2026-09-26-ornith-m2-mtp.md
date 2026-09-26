# Ornith qwen35moe M2: MTP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Ornith-1.5-35B-A3B speculative decoding with its own MTP block (`blk.40`): the prompt catch-up that fills the block's KV cache, one draft per cycle, a 2-row verify, and greedy output byte-identical to plain decoding.

**Architecture:** The Ornith graph (`ds4_qwen35moe.inc`) gains an MTP mode: it keeps the trunk's post-`output_norm` hidden rows in a new seed buffer (`mtp_h`, with a one-row carry across forwards), allocates the MTP block's KV cache, the GDN verify snapshot and the MTP staging buffers, and adds `qwen35_graph_mtp` (catch-up rows and the draft head) plus `qwen35_graph_state_swap`. A verify runs its attention one row per dispatch so every row reproduces plain decoding bit for bit. The session side (`ds4.c`) gets `ds4_session_qwen35_spec_cycle`, shaped like the Qwen3.8 cycle, the catch-up in sync and eval, and the dispatch branches. One new Metal kernel builds the MTP input `[RMSNorm(e)·enorm | RMSNorm(h)·hnorm]`. llama.cpp's `--spec-type draft-mtp` is the acceptance oracle.

**Tech Stack:** C (ds4.c, ds4_qwen35moe.inc), Objective-C (ds4_metal.m), Metal Shading Language, Python 3 stdlib (test and measurement scripts), llama.cpp `llama-server`.

**Spec:** `docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md` (this plan is milestone M2 of §9; M3 serving and M4 acceptance have their own plans: `2026-09-26-ornith-m3-serving.md`, `2026-09-26-ornith-m4-acceptance.md`).

## Global Constraints

- The Qwen3.8 production path stays byte-identical and as fast as today (spec §1, §7). Every commit that touches a shared file (`ds4.c` outside Ornith-only functions, `ds4_metal.m`, `ds4_gpu.h`, `metal/*.metal`, `ds4_server.c`, `ds4_agent.c`, `ds4_kvstore.c`) passes `make test-qwen4-kernels test-qwen4-q2` and `speed-bench/qwen-regression/run.sh fast`; the branch passes `run.sh full` before merge.
- Metal only for Ornith; the M1 refusals stay. This plan lifts only `--mtp`; `--mtp-model` and `--mtp-exact-sampling` stay refused.
- Ornith knobs use the `DS4_QWEN35_*` prefix; Ornith code reads no family-level `DS4_QWEN4_*` knob (spec §7.3; the kernel A/B switches inside shared qwen4 helpers are the documented exception).
- Kernel changes are additive: new kernels/entry points in `metal/qwen35.metal`; `metal/qwen4.metal` is not edited.
- Code, comments, docs and commit messages in English. Model files never go into git. The 23G GGUF: `DS4_ORNITH_MODEL=$HOME/orca/workspaces/ds4-metal-data/gguf/ornith/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf`. Every model test reads `DS4_ORNITH_MODEL` and fails with a message when it is unset.
- One model process at a time on this 64 GB machine. Never `kill -9` a Metal process. Long runs under `caffeinate -i -s`; monitors poll process liveness, not only a log pattern.
- GPU windows: pausing the live stack needs the user's OK once per execution window. Pause = `launchctl unload ~/Library/LaunchAgents/dev.dongnh.gateway-watchdog.plist ~/Library/LaunchAgents/dev.dongnh.gateway-eval.plist ~/Library/LaunchAgents/dev.dongnh.ai-proxy.plist`, then `kill -TERM $(cat ~/.local/share/ai-gateway/omlx.pid)`; restore = `launchctl load` of the three plists, then wait for `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8090/status` = 200.
- Never add a staging entry to the live gateway registry. Staging ports: 18296 (Ornith ds4-server staging), 18190 (llama.cpp oracle), 18191 (loader tests).
- No C++. Follow AGENT.md: small readable code; comments explain why.
- Every commit ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2
  ```
- Branch: `feature/ornith-m2`, cut from `develop` (which carries M1 and these plans).

## Deviations from the spec wording, decided while planning

Research behind these: the llama.cpp sources at the Homebrew commit (`src/models/qwen35moe.cpp` `graph_mtp`, `common/speculative.cpp` `process`/`draft`/`accept`) and the ds4 Qwen3.8 MTP code.

1. **Seed and chaining (spec §4 "MTP block").** The trunk `h` fed to the MTP block is the post-`output_norm` hidden (the tensor the LM head reads; llama.cpp PR #24025). Chaining a second draft uses the MTP output after `shared_head_norm`, not `x`. M2 drafts one token per cycle, so nothing chains; deeper drafts are an M4 lever. Task 4 edits §4.
2. **Catch-up convention (spec §5 "Prefill").** MTP KV row `p` is computed from `(token_p, h_{p-1})` at RoPE position `p`, with `h_{-1} = 0` (llama.cpp's convention). Qwen3.8 has no catch-up to reuse, so this is new code. Catch-up rows need only their K/V: concat, `eh_proj`, `attn_norm`, the q/k/v projections and the KV append; attention, MoE and head run for the draft row alone. Task 4 edits §5.
3. **Own cycle (spec §5 "Decode with MTP").** The qwen4 speculative cycle and its snapshot helpers require hyper-connections and PLE, so Ornith gets `ds4_session_qwen35_spec_cycle`, mirroring the qwen4 structure. The GDN kernels' after-row-0 snapshot (`qwen4_graph_linear` with `snap_after_first`) is reused. Like the M1 session branches, the cycle lives in `ds4.c`: `ds4_qwen35moe.inc` is included before `struct ds4_session`. Task 4 edits §3 and §5.
4. **Row-exact verify.** Dense attention sizes its key splits from the batch's key count, so row 0 of a 2-row verify would sum differently from a plain decode at the same position. The verify runs its attention decode one row per dispatch (`verify_rows_exact`); plain decode and prefill keep the M1 dispatches.
5. **Sampling.** Greedy and opportunistic sampling only (a draft is accepted when it is the target argmax, as Qwen3.8 does without `--mtp-exact-sampling`). `--mtp-exact-sampling` is refused at open for Ornith.

## Review Focus

1. **A verify at a position where plain decode's attention split differs** (P + 1 a multiple of 32, or more than 2048 keys). Each verify row must equal plain decoding bit for bit. Covered by Task 2 (`tests/test_qwen35_graph.c` check 2 at P = 31, 63, 2047, 2111, 3000) and Task 4 (`test_mtp_cli.py` on the 9,371-token `long_copy`).
2. **A rejected draft.** The restored state must continue exactly like plain decoding. Covered by Task 2 (graph check 3: reject, `qwen35_graph_state_swap`, then 700 more tokens compared) and Task 4 (`DS4_QWEN35_SPEC_TRACE` must show rejects while the output stays identical).
3. **A committed draft that is not the argmax** (`DS4_QWEN35_SPEC_FORCE_ACCEPT`, and the opportunistic sampled path). The session must match a plain session fed the same tokens. Covered by Task 3 (`tests/test_qwen35_mtp.c` check 2).
4. **Tiny prefill buffers** (`--prefill-chunk 1` and `2`). The MTP pass splits into sub-batches; a cap below 2 takes the plain path every cycle; output stays identical. Covered by Task 4 (`test_mtp_cli.py` chunks 1 and 2) and Task 2 (graph check 4 with a 1-token graph).
5. **The context filling up during speculative decoding.** The cycle falls back to plain steps near the end, never writes an MTP row past the context, and stops with "context is full". Covered by Task 3 (`test_qwen35_mtp.c` check 4).

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `metal/qwen35.metal` | Modify | `kernel_qwen35_mtp_concat` |
| `ds4_metal.m` | Modify | Kernel enum/name, `ds4_gpu_qwen35_mtp_concat_tensor` |
| `ds4_gpu.h` | Modify | Declaration of the wrapper |
| `tests/test_qwen35_kernels.c` | Modify | Concat kernel test |
| `ds4.c` | Modify | `mtp_h`/`mtp_h_pos0`/`mtp_h_rows` graph fields, free list, caller updates, memory estimate, open gate, `--mtp` embedded check, draft-token count, session create/free/sync/eval, `ds4_session_qwen35_spec_cycle`, speculative dispatch |
| `ds4_qwen35moe.inc` | Rewrite | MTP-aware alloc/reset/forward, seed carry, split attention with the row-exact verify, `qwen35_graph_mtp`, `qwen35_graph_state_swap` |
| `tests/test_qwen35_graph.c` | Create | Real-model graph checks (includes `ds4.c`) |
| `tests/test_qwen35_mtp.c` | Create | Real-model MTP session checks (public API) |
| `tests/ornith/test_loader.sh` | Modify | `--mtp-exact-sampling` and `--mtp-model` refusals |
| `tests/ornith/test_mtp_cli.py` | Create | Plain vs session vs `--mtp` byte-identity across prefill chunks |
| `tests/ornith/mtp_accept.py` | Create | Acceptance of ds4 vs llama.cpp draft-mtp |
| `Makefile` | Modify | `test-qwen35-graph`, `test-qwen35-mtp` |
| `speed-bench/ornith/m2/RESULTS.md` (+ `cli-mtp.txt`, `accept.md`, `QWEN_GATE.md`) | Create | Receipts |
| `docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md` | Modify | §3, §4, §5 wording (deviations 1-5) |

---

### Task 1: MTP input concat kernel

The MTP block's input for token t is `[RMSNorm(embed(t))·enorm | RMSNorm(h)·hnorm]` (2E wide, embedding half first). The catch-up runs it over whole prefill chunks, and `rms_norm_weight_rows` cannot write a strided `[T][2E]` destination, so this is one small additive kernel.

**Files:**
- Modify: `metal/qwen35.metal` (append)
- Modify: `ds4_metal.m` (enum, name, wrapper)
- Modify: `ds4_gpu.h`
- Test: `tests/test_qwen35_kernels.c`

**Interfaces:**
- Consumes: `qwen4_dispatch`, `qwen4_bind_tensor`, `qwen4_bind_weight` (ds4_metal.m); test helpers `arena_f32`, `rand_vec`, `upload`, `download`, `check_close` (tests/test_qwen35_kernels.c).
- Produces:
  ```c
  int ds4_gpu_qwen35_mtp_concat_tensor(
          ds4_gpu_tensor *cat, const ds4_gpu_tensor *e, const ds4_gpu_tensor *h,
          const void *model_map, uint64_t model_size, uint64_t enorm_offset, uint64_t hnorm_offset,
          uint32_t n_embd, uint32_t n_tokens, float eps);
  ```
  `cat` is `[n_tokens][2·n_embd]`, `e` and `h` are `[n_tokens][n_embd]`; returns 1 on success.

- [ ] **Step 1: Write the failing test**

In `tests/test_qwen35_kernels.c`, add this function right before `int main(void) {`:

```c
/* Ornith MTP input: cat[t] = [RMSNorm(e_t) * enorm | RMSNorm(h_t) * hnorm],
 * embedding half first (llama.cpp qwen35moe graph_mtp). */
static void test_mtp_concat(arena_t *a, uint32_t E, uint32_t T) {
    double *we, *wh;
    const uint64_t e_off = arena_f32(a, E, &we, 0.5f, 1.5f);
    const uint64_t h_off = arena_f32(a, E, &wh, 0.5f, 1.5f);
    const uint64_t n = (uint64_t)T * E;
    float *e = rand_vec(n, 1.0f), *h = rand_vec(n, 3.0f);
    double *ref = malloc(2u * n * sizeof(double));
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t half = 0; half < 2u; half++) {
            const float *x = (half ? h : e) + (uint64_t)t * E;
            const double *w = half ? wh : we;
            double ss = 0.0;
            for (uint32_t i = 0; i < E; i++) ss += (double)x[i] * x[i];
            const double r = 1.0 / sqrt(ss / E + 1e-6);
            for (uint32_t i = 0; i < E; i++) ref[(uint64_t)t * 2u * E + (uint64_t)half * E + i] = x[i] * r * w[i];
        }
    }
    ds4_gpu_tensor *ge = upload(e, n), *gh = upload(h, n), *gc = upload(NULL, 2u * n);
    require_ok(ds4_gpu_qwen35_mtp_concat_tensor(gc, ge, gh, a->base, a->size, e_off, h_off, E, T, 1e-6f),
               "qwen35 mtp concat");
    float *got = download(gc, 2u * n);
    check_close("mtp concat", got, ref, 2u * n, 1e-5);
    free(got); free(e); free(h); free(ref); free(we); free(wh);
    ds4_gpu_tensor_free(ge); ds4_gpu_tensor_free(gh); ds4_gpu_tensor_free(gc);
}
```

In `main`, add before `printf("qwen35 kernels: ok\n");`:

```c
    printf("qwen35 mtp concat\n");
    test_mtp_concat(&arena, 2048, 1);
    test_mtp_concat(&arena, 2048, 7);
    test_mtp_concat(&arena, 256, 3);
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `make tests/test_qwen35_kernels`
Expected: a compile error: `ds4_gpu_qwen35_mtp_concat_tensor` is undeclared.

- [ ] **Step 3: Add the kernel**

Append to `metal/qwen35.metal`:

```metal
/* --- MTP input ------------------------------------------------------------ */

struct ds4_metal_args_qwen35_mtp_concat {
    uint32_t n_embd;
    uint32_t n_tokens;
    float    eps;
};

/* The MTP block's input row: [RMSNorm(e) * enorm | RMSNorm(h) * hnorm],
 * embedding half first, as llama.cpp's qwen35moe graph_mtp concatenates
 * them.  One threadgroup of 256 threads per (half, token). */
kernel void kernel_qwen35_mtp_concat(
        constant ds4_metal_args_qwen35_mtp_concat & args,
        device float       *cat,     /* [T][2E] */
        device const float *e,       /* [T][E] token embeddings */
        device const float *h,       /* [T][E] trunk hidden, post output_norm */
        device const float *enorm,   /* [E] */
        device const float *hnorm,   /* [E] */
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint half_ = tgpig.x;
    const uint tok = tgpig.y;
    if (half_ > 1u || tok >= args.n_tokens) return;
    const uint E = args.n_embd;
    device const float *x = (half_ ? h : e) + (uint64_t)tok * E;
    device const float *w = half_ ? hnorm : enorm;
    device float *y = cat + (uint64_t)tok * 2u * E + half_ * E;
    threadgroup float partial[8];
    float ss = 0.0f;
    for (uint i = tid; i < E; i += 256u) ss += x[i] * x[i];
    ss = simd_sum(ss);
    if (tiisg == 0) partial[sgitg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint s = 0; s < 8u; s++) total += partial[s];
    const float r = rsqrt(total / (float)E + args.eps);
    for (uint i = tid; i < E; i += 256u) y[i] = x[i] * r * w[i];
}
```

- [ ] **Step 4: Register the kernel and add the wrapper**

In `ds4_metal.m`, in the qwen4 kernel enum, replace

```objc
    QWEN4_K_QWEN35_GDN_OUT,
    QWEN4_K_COUNT,
```

with

```objc
    QWEN4_K_QWEN35_GDN_OUT,
    QWEN4_K_QWEN35_MTP_CONCAT,
    QWEN4_K_COUNT,
```

In `qwen4_kernel_names`, replace

```objc
    "kernel_qwen35_gdn_out",
};
```

with

```objc
    "kernel_qwen35_gdn_out",
    "kernel_qwen35_mtp_concat",
};
```

Right after the closing brace of `ds4_gpu_qwen35_gdn_out_tensor`, add:

```objc
int ds4_gpu_qwen35_mtp_concat_tensor(
        ds4_gpu_tensor *cat, const ds4_gpu_tensor *e, const ds4_gpu_tensor *h,
        const void *model_map, uint64_t model_size, uint64_t enorm_offset, uint64_t hnorm_offset,
        uint32_t n_embd, uint32_t n_tokens, float eps) {
    struct { uint32_t n_embd, n_tokens; float eps; } args = { n_embd, n_tokens, eps };
    qwen4_bind b[5];
    const uint64_t row = (uint64_t)n_embd * sizeof(float);
    if (n_tokens == 0 || n_embd == 0 ||
        !qwen4_bind_tensor(&b[0], cat, 2u * row * n_tokens, "mtp concat") ||
        !qwen4_bind_tensor(&b[1], e, row * n_tokens, "mtp embedding") ||
        !qwen4_bind_tensor(&b[2], h, row * n_tokens, "mtp hidden") ||
        !qwen4_bind_weight(&b[3], model_map, model_size, enorm_offset, row, "nextn enorm") ||
        !qwen4_bind_weight(&b[4], model_map, model_size, hnorm_offset, row, "nextn hnorm")) {
        return 0;
    }
    return qwen4_dispatch(QWEN4_K_QWEN35_MTP_CONCAT, &args, sizeof(args), b, 5,
                          MTLSizeMake(2, n_tokens, 1), MTLSizeMake(256, 1, 1), 0);
}
```

In `ds4_gpu.h`, right after the declaration of `ds4_gpu_qwen35_gdn_out_tensor(...)`, add:

```c
/* Ornith MTP input rows: cat[t] = [RMSNorm(e_t) * enorm | RMSNorm(h_t) * hnorm]
 * (cat [T][2E], e and h [T][E]; enorm/hnorm F32 in the model map). */
int ds4_gpu_qwen35_mtp_concat_tensor(
        ds4_gpu_tensor *cat, const ds4_gpu_tensor *e, const ds4_gpu_tensor *h,
        const void *model_map, uint64_t model_size, uint64_t enorm_offset, uint64_t hnorm_offset,
        uint32_t n_embd, uint32_t n_tokens, float eps);
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `make test-qwen35-kernels`
Expected: every earlier line still `ok`, plus three `mtp concat ... ok` lines and `qwen35 kernels: ok`.

Run: `make test-qwen4-kernels test-qwen4-q2`
Expected: all PASS (shared files changed: `ds4_metal.m`, `ds4_gpu.h`, `metal/qwen35.metal`).

- [ ] **Step 6: Qwen3.8 fast gate (GPU window)**

Run: `speed-bench/qwen-regression/run.sh fast`
Expected: last line `qwen_gate: PASS`.

- [ ] **Step 7: Commit**

```bash
git add metal/qwen35.metal ds4_metal.m ds4_gpu.h tests/test_qwen35_kernels.c
git commit -m "metal: Ornith MTP input concat kernel

kernel_qwen35_mtp_concat builds the MTP block's input rows,
[RMSNorm(e)*enorm | RMSNorm(h)*hnorm] with the embedding half first,
for whole prefill chunks in one dispatch.  Additive: a new entry point
in qwen35.metal and its wrapper; Qwen3.8 kernels are unchanged.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2"
```

---

### Task 2: MTP state in the Ornith graph

The graph learns everything M2 needs below the session: the post-norm seed rows with their carry, the MTP block's KV cache, the GDN verify snapshot, all-rows logits for a 2-row verify, the row-exact verify attention, the MTP pass (catch-up rows and the draft head) and the snapshot swap. Graphs without MTP (the one-shot generator, sessions without `--mtp`) keep the M1 dispatches exactly.

**Files:**
- Modify: `ds4.c` (graph struct fields, `qwen4_graph_free` list, callers of the changed signatures)
- Rewrite: `ds4_qwen35moe.inc`
- Create: `tests/test_qwen35_graph.c`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `ds4_gpu_qwen35_mtp_concat_tensor` (Task 1); `qwen4_graph_linear` (reads `g->snap_after_first`, `g->snap_lin_state[il]`, `g->snap_lin_hist[il]`), `qwen4_graph_fused`, `qwen4_gemv`, `qwen4_ref_row`, `qwen4_graph_alloc_f32`, `qwen4_graph_free`, `ds4_gpu_qwen4_attn_decode_tensor`, `ds4_gpu_qwen4_argmax_tensor`, `ds4_gpu_tensor_view/copy/read/write`, `glm_graph_begin_commands_if_needed`, `ds4_gpu_end_commands`.
- Produces (in `ds4_qwen35moe.inc`):
  ```c
  static bool qwen35_graph_alloc(ds4_qwen4_gpu_graph *g, uint32_t ctx_cap, uint32_t cap_tokens, bool mtp);
  static void qwen35_graph_reset(ds4_qwen4_gpu_graph *g);
  static bool qwen35_graph_forward_tokens(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                          const int *tokens, uint32_t T, float *logits_out, bool all_rows);
  static bool qwen35_graph_mtp(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                               const int *tokens, uint32_t T, uint32_t pos0, bool want_draft, int *draft_out);
  static bool qwen35_graph_state_swap(ds4_qwen4_gpu_graph *g);
  ```
  and the struct fields `ds4_gpu_tensor *mtp_h; uint32_t mtp_h_pos0, mtp_h_rows;`. Invariants: row k of `mtp_h` holds `h_{mtp_h_pos0 - 1 + k}`; after an MTP pass `g->mtp_pos = pos0 + T`; a verify (caller sets `g->snap_after_first` and `g->verify_rows_exact`) leaves `g->snap_valid` and `g->snap_pos = pos0 + 1`.

- [ ] **Step 1: Write the failing graph test**

Create `tests/test_qwen35_graph.c`:

```c
/* Real-model checks of the Ornith graph's MTP state (ds4_qwen35moe.inc).
 * Usage: test_qwen35_graph MODEL
 *  1. an MTP graph (post-norm seed rows, two logit rows) gives logits
 *     bit-identical to the M1 graph through a chunked prefill and 8 decodes;
 *  2. a 2-row verify (verify_rows_exact, after-row-0 snapshot) gives each
 *     row's logits bit-identical to plain 1-row decodes, at positions where
 *     the attention split geometry of the two differs (31, 63, 2047, 2111,
 *     3000);
 *  3. a rejected verify restored with qwen35_graph_state_swap continues
 *     bit-identical to plain decoding (checked at the next checkpoint and
 *     after a further prefill);
 *  4. the MTP catch-up writes the same block-40 K rows whether the prompt
 *     is prefilled in chunks of 512, of 64 or one token at a time (GEMM vs
 *     GEMV rounding only); the three graphs' drafts are printed. */
#include "../ds4.c"

static int failures;
#define CHECK(cond) do { \
        if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
    } while (0)

enum { TEST_CTX = 4160 };

typedef struct {
    ds4_qwen4_gpu_graph g;
    float *logits;   /* two rows */
} tgraph;

static void tg_init(tgraph *t, uint32_t cap, bool mtp) {
    memset(t, 0, sizeof(*t));
    if (!qwen35_graph_alloc(&t->g, TEST_CTX, cap, mtp)) {
        fprintf(stderr, "graph alloc failed (cap %u, mtp %d)\n", cap, (int)mtp);
        exit(1);
    }
    qwen35_graph_reset(&t->g);
    t->logits = xmalloc(2u * (size_t)DS4_N_VOCAB * sizeof(float));
}

static void tg_free(tgraph *t) {
    qwen4_graph_free(&t->g);
    free(t->logits);
}

/* tokens [a, b) in chunks of the graph's cap, MTP catch-up after each
 * chunk when the graph has MTP; t->logits holds the last row */
static void tg_prefill(tgraph *t, ds4_engine *e, const int *tokens, uint32_t a, uint32_t b) {
    for (uint32_t i = a; i < b;) {
        const uint32_t n = b - i < t->g.cap_tokens ? b - i : t->g.cap_tokens;
        if (!qwen35_graph_forward_tokens(&t->g, &e->model, &e->weights, tokens + i, n, t->logits, false)) {
            fprintf(stderr, "forward failed at %u\n", i);
            exit(1);
        }
        if (t->g.mtp_h && !qwen35_graph_mtp(&t->g, &e->model, &e->weights, tokens + i, n, i, false, NULL)) {
            fprintf(stderr, "catch-up failed at %u\n", i);
            exit(1);
        }
        i += n;
    }
}

static bool same_row(const float *a, const float *b) {
    return memcmp(a, b, (size_t)DS4_N_VOCAB * sizeof(float)) == 0;
}

static void check_mtp_logits_path(ds4_engine *e, const int *tok) {
    tgraph p, m;
    tg_init(&p, 512, false);
    tg_init(&m, 512, true);
    tg_prefill(&p, e, tok, 0, 1100);
    tg_prefill(&m, e, tok, 0, 1100);
    CHECK(same_row(p.logits, m.logits));
    for (int i = 0; i < 8; i++) {
        const int next = sample_argmax(p.logits, DS4_N_VOCAB);
        CHECK(qwen35_graph_forward_tokens(&p.g, &e->model, &e->weights, &next, 1, p.logits, false));
        CHECK(qwen35_graph_forward_tokens(&m.g, &e->model, &e->weights, &next, 1, m.logits, false));
        CHECK(same_row(p.logits, m.logits));
    }
    printf("  MTP graph logits: bit-identical to the M1 graph (prefill 1100 + 8 decodes)\n");
    tg_free(&p);
    tg_free(&m);
}

static void check_verify_rows(ds4_engine *e, const int *tok) {
    const uint32_t checkpoints[] = {31, 63, 2047, 2111, 3000};
    tgraph p, m;
    tg_init(&p, 512, false);
    tg_init(&m, 512, true);
    uint32_t pos = 0;
    for (size_t c = 0; c < sizeof(checkpoints) / sizeof(*checkpoints); c++) {
        const uint32_t P = checkpoints[c];
        tg_prefill(&p, e, tok, pos, P);
        tg_prefill(&m, e, tok, pos, P);
        CHECK(same_row(p.logits, m.logits));
        const bool reject = c % 2u == 1u;
        const int a = tok[P];
        const int b = reject ? (tok[P + 1] + 1) % (int)DS4_N_VOCAB : tok[P + 1];
        const int pair[2] = { a, b };
        m.g.snap_after_first = true;
        m.g.verify_rows_exact = true;
        const bool ok = qwen35_graph_forward_tokens(&m.g, &e->model, &e->weights, pair, 2, m.logits, true);
        m.g.verify_rows_exact = false;
        CHECK(ok && m.g.snap_valid && m.g.snap_pos == P + 1u && m.g.pos == P + 2u);
        CHECK(qwen35_graph_forward_tokens(&p.g, &e->model, &e->weights, &a, 1, p.logits, false));
        CHECK(same_row(p.logits, m.logits));
        if (!reject) {
            CHECK(qwen35_graph_forward_tokens(&p.g, &e->model, &e->weights, &b, 1, p.logits, false));
            CHECK(same_row(p.logits, m.logits + DS4_N_VOCAB));
            pos = P + 2u;
        } else {
            CHECK(qwen35_graph_state_swap(&m.g) && m.g.pos == P + 1u && !m.g.snap_valid);
            pos = P + 1u;
        }
        printf("  verify at %4u: %s bit-identical to plain decoding\n", P,
               reject ? "row 0 and the restored state" : "both rows");
    }
    tg_prefill(&p, e, tok, pos, 3700);
    tg_prefill(&m, e, tok, pos, 3700);
    CHECK(same_row(p.logits, m.logits));
    printf("  prefill to 3700 after the verifies: bit-identical\n");
    tg_free(&p);
    tg_free(&m);
}

/* block-40 K rows [0, rows) as floats */
static float *read_k40(tgraph *t, uint32_t rows) {
    const uint64_t n = (uint64_t)rows * DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    uint16_t *h = xmalloc(n * sizeof(uint16_t));
    float *out = xmalloc(n * sizeof(float));
    (void)ds4_gpu_synchronize();
    if (!ds4_gpu_tensor_read(t->g.layer_k_cache[DS4_N_LAYER - 1u], 0, h, n * sizeof(uint16_t))) {
        fprintf(stderr, "k cache read failed\n");
        exit(1);
    }
    for (uint64_t i = 0; i < n; i++) out[i] = f16_to_f32(h[i]);
    free(h);
    return out;
}

/* the plain-path draft: forward token N, then MTP rows N and N+1 */
static int tg_draft(tgraph *t, ds4_engine *e, int token, uint32_t N) {
    int d = -1;
    if (!qwen35_graph_forward_tokens(&t->g, &e->model, &e->weights, &token, 1, t->logits, false)) return -1;
    const int toks[2] = { token, sample_argmax(t->logits, DS4_N_VOCAB) };
    if (!qwen35_graph_mtp(&t->g, &e->model, &e->weights, toks, 2, N, true, &d)) return -1;
    return d;
}

static void check_catch_up(ds4_engine *e, const int *tok) {
    const uint32_t N = 600;
    tgraph a, b, c;
    tg_init(&a, 512, true);
    tg_init(&b, 64, true);
    tg_init(&c, 1, true);
    tg_prefill(&a, e, tok, 0, N);
    tg_prefill(&b, e, tok, 0, N);
    tg_prefill(&c, e, tok, 0, N);
    CHECK(a.g.mtp_pos == N && b.g.mtp_pos == N && c.g.mtp_pos == N);
    float *ka = read_k40(&a, N), *kb = read_k40(&b, N), *kc = read_k40(&c, N);
    const uint64_t n = (uint64_t)N * DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    double scale = 1e-6, wb = 0.0, wc = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        CHECK(isfinite(ka[i]));
        if (fabs(ka[i]) > scale) scale = fabs(ka[i]);
        if (fabs((double)ka[i] - kb[i]) > wb) wb = fabs((double)ka[i] - kb[i]);
        if (fabs((double)ka[i] - kc[i]) > wc) wc = fabs((double)ka[i] - kc[i]);
    }
    printf("  catch-up K rows: chunk 64 max|d| %.3e, per token max|d| %.3e (scale %.3e)\n", wb, wc, scale);
    CHECK(wb <= 5e-2 * scale && wc <= 5e-2 * scale);
    const int da = tg_draft(&a, e, tok[N], N), db = tg_draft(&b, e, tok[N], N), dc = tg_draft(&c, e, tok[N], N);
    printf("  drafts after %u tokens: %d %d %d (chunks 512, 64, 1)\n", N, da, db, dc);
    /* a 1-token graph runs the 2-row draft pass as two sub-batches */
    CHECK(da >= 0 && db >= 0 && dc >= 0);
    free(ka); free(kb); free(kc);
    tg_free(&a); tg_free(&b); tg_free(&c);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL\n", argv[0]);
        return 1;
    }
    ds4_engine_options opt = {.model_path = argv[1], .context_size = TEST_CTX, .prefill_chunk = 512,
                              .backend = DS4_BACKEND_METAL};
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0 || !ds4_engine_is_qwen35moe(e)) {
        fprintf(stderr, "cannot open the Ornith model\n");
        return 1;
    }
    FILE *fp = fopen("speed-bench/promessi_sposi.txt", "rb");
    if (!fp) { perror("speed-bench/promessi_sposi.txt"); return 1; }
    char *text = calloc(40001, 1);
    if (fread(text, 1, 40000, fp) == 0) { fprintf(stderr, "empty corpus\n"); return 1; }
    fclose(fp);
    ds4_tokens tokens = {0};
    ds4_tokenize_text(e, text, &tokens);
    free(text);
    if (tokens.len < 3702) { fprintf(stderr, "corpus too short: %d tokens\n", tokens.len); return 1; }
    check_mtp_logits_path(e, tokens.v);
    check_verify_rows(e, tokens.v);
    check_catch_up(e, tokens.v);
    ds4_tokens_free(&tokens);
    ds4_engine_close(e);
    if (failures) {
        fprintf(stderr, "qwen35 graph: %d failures\n", failures);
        return 1;
    }
    printf("qwen35 graph: ok\n");
    return 0;
}
```

In `Makefile`, inside the `ifeq ($(UNAME_S),Darwin)` block that holds the `tests/test_qwen35_session` rules, add after the `tests/test_qwen35_session:` rule (before that block's `endif`):

```make
tests/test_qwen35_graph.o: tests/test_qwen35_graph.c ds4.c ds4.h ds4_gpu.h ds4_qwen35moe.inc
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -Wno-unused-function -I. -c -o $@ $<

tests/test_qwen35_graph: tests/test_qwen35_graph.o $(filter-out ds4.o,$(CORE_OBJS))
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -o $@ $^ $(METAL_LDLIBS)

```

After the `test-qwen35-session:` recipe, add:

```make
.PHONY: test-qwen35-graph
test-qwen35-graph: tests/test_qwen35_graph
	./tests/test_qwen35_graph "$(DS4_ORNITH_MODEL)"
```

In the `clean` recipe, replace `rm -f tests/test_qwen35_kernels tests/test_qwen35_session` with `rm -f tests/test_qwen35_kernels tests/test_qwen35_session tests/test_qwen35_graph`.

- [ ] **Step 2: Run the test and confirm it fails**

Run: `make tests/test_qwen35_graph`
Expected: compile errors: `qwen35_graph_alloc` called with 4 arguments, `qwen35_graph_mtp`/`qwen35_graph_state_swap` undeclared, no member `mtp_h`.

- [ ] **Step 3: Add the seed fields to the graph struct**

In `ds4.c`, in `typedef struct ds4_qwen4_gpu_graph`, replace

```c
    /* Ornith (qwen35moe) gates the GDN output with silu(z); Qwen3.8 with
     * sigmoid(z).  qwen4_graph_alloc's memset leaves it false. */
    bool gdn_silu;
} ds4_qwen4_gpu_graph;
```

with

```c
    /* Ornith (qwen35moe) gates the GDN output with silu(z); Qwen3.8 with
     * sigmoid(z).  qwen4_graph_alloc's memset leaves it false. */
    bool gdn_silu;
    /* Ornith MTP seed: post-output_norm trunk hidden rows.  Row k holds
     * h_{mtp_h_pos0 - 1 + k}; row 0 is the carry from the previous forward
     * (zeros at position 0) and rows 1..mtp_h_rows the last forward's rows.
     * NULL on Qwen3.8 and on Ornith graphs without MTP. */
    ds4_gpu_tensor *mtp_h;
    uint32_t mtp_h_pos0;
    uint32_t mtp_h_rows;
} ds4_qwen4_gpu_graph;
```

In `qwen4_graph_free`, replace

```c
        &g->draft_head, &g->steer_dirs,
    };
```

with

```c
        &g->draft_head, &g->steer_dirs, &g->mtp_h,
    };
```

- [ ] **Step 4: Rewrite `ds4_qwen35moe.inc`**

Replace the whole file with:

```c
/* Ornith-1.5-35B-A3B (llama.cpp qwen35moe) on Metal.
 *
 * Included by ds4.c right after the qwen4 Metal graph, so the family reuses
 * the qwen4 graph struct, GEMV and kernels.  GDN layers call
 * qwen4_graph_linear with the silu output gate (g->gdn_silu).  Attention is
 * the qwen4 gated attention without the QSA indexer: every position is
 * attended densely.  The MoE reuses the router and the Q4_K expert kernels,
 * with the qwen35 row kernels for Q5_K layers.  The residual stream is one
 * n_embd row per token, with no hyper-connections and no PLE layer.  The
 * llama.cpp converter already folded +1 into the norm weights, stores
 * ssm_a = -exp(A_log) and tiles the GDN value heads, which is what the qwen4
 * kernels expect.  The prefill chunk rule (qwen35_prefill_chunk_tokens)
 * lives in ds4.c next to the memory estimate that shares it.  The reused
 * qwen4 helpers keep their DS4_QWEN4_* kernel A/B switches (dense GEMV,
 * decode fusions, attention split/tiled/merge), which pick between
 * arithmetic-equivalent paths; Ornith's own knobs use DS4_QWEN35_*.
 *
 * MTP (graphs allocated with mtp): the MTP block blk.40 follows llama.cpp's
 * draft-mtp.  Its K/V row p comes from token p and h_{p-1}, the trunk's
 * post-output_norm hidden (h_{-1} = 0), so every forward also writes its
 * post-norm rows to mtp_h and the next forward carries the last one over.
 * Catch-up rows need only their K/V; the draft row continues through
 * attention, the MoE, shared_head_norm and the output head.  A speculative
 * verify snapshots the GDN state after its first row and runs its
 * attention one row per dispatch, so each row equals plain decoding bit for
 * bit.  The session cycle is ds4_session_qwen35_spec_cycle in ds4.c. */

static bool qwen35_graph_alloc(ds4_qwen4_gpu_graph *g, uint32_t ctx_cap, uint32_t cap_tokens, bool mtp) {
    memset(g, 0, sizeof(*g));
    if ((uint64_t)ctx_cap > DS4_ROPE_ORIG_CTX) {
        fprintf(stderr, "ds4: Ornith supports up to %llu tokens of context (no YaRN)\n",
                (unsigned long long)DS4_ROPE_ORIG_CTX);
        return false;
    }
    const uint64_t E = DS4_N_EMBD, T = cap_tokens;
    const uint64_t conv_dim = DS4_N_LIN_CONV_DIM;
    const uint64_t v_dim = (uint64_t)DS4_N_LIN_V_HEAD * DS4_N_LIN_HEAD_DIM;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t kv_dim = (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    g->ctx_cap = ctx_cap;
    g->alloc_cap = ctx_cap;
    g->cap_tokens = cap_tokens;
    /* a verify reads both of its rows' logits */
    g->n_logit_rows = mtp ? 2u : 1u;
    g->owns_scratch = true;
    g->gdn_silu = true;
    bool ok = true;
#define QWEN35_ALLOC(field_, n_) do { g->field_ = qwen4_graph_alloc_f32(n_); ok = ok && g->field_; } while (0)
    QWEN35_ALLOC(R, T * E);
    QWEN35_ALLOC(mixed, T * E);
    QWEN35_ALLOC(blk, T * E);
    QWEN35_ALLOC(qkv, T * conv_dim);
    QWEN35_ALLOC(z, T * v_dim);
    QWEN35_ALLOC(ga, T * DS4_N_LIN_V_HEAD);
    QWEN35_ALLOC(gb, T * DS4_N_LIN_V_HEAD);
    QWEN35_ALLOC(lin_o, T * v_dim);
    QWEN35_ALLOC(qg, T * 2u * q_dim);
    QWEN35_ALLOC(kp, T * kv_dim);
    QWEN35_ALLOC(vp, T * kv_dim);
    QWEN35_ALLOC(q, T * q_dim);
    QWEN35_ALLOC(gate, T * q_dim);
    QWEN35_ALLOC(attn_o, T * q_dim);
    QWEN35_ALLOC(attn_part, ds4_gpu_qwen4_attn_part_floats(3u, DS4_N_HEAD, DS4_N_HEAD_DIM));
    QWEN35_ALLOC(router, T * DS4_N_EXPERT);
    QWEN35_ALLOC(selected, T * DS4_N_EXPERT_USED);
    QWEN35_ALLOC(weights, T * DS4_N_EXPERT_USED);
    QWEN35_ALLOC(mid, T * (DS4_N_EXPERT_USED + 1u) * DS4_N_FF_EXP);
    QWEN35_ALLOC(part, T * (DS4_N_EXPERT_USED + 1u) * E);
    QWEN35_ALLOC(sh_gate_logit, T);
    QWEN35_ALLOC(moe_lists, (uint64_t)DS4_N_EXPERT * T);
    QWEN35_ALLOC(moe_counts, DS4_N_EXPERT);
    QWEN35_ALLOC(sh_gate, T * DS4_N_FF_EXP);
    QWEN35_ALLOC(sh_up, T * DS4_N_FF_EXP);
    QWEN35_ALLOC(sh_mid, T * DS4_N_FF_EXP);
    QWEN35_ALLOC(sh_out, T * E);
    QWEN35_ALLOC(logits, (uint64_t)g->n_logit_rows * DS4_N_VOCAB);
    QWEN35_ALLOC(pos3, (uint64_t)ctx_cap * 4u);
    if (mtp) {
        QWEN35_ALLOC(mtp_h, (T + 1u) * E);
        QWEN35_ALLOC(mtp_e, T * E);
        QWEN35_ALLOC(mtp_cat, T * 2u * E);
        QWEN35_ALLOC(mtp_R, T * E);
        QWEN35_ALLOC(mtp_argmax, 1u);
        QWEN35_ALLOC(mtp_argmax_tmp, ((DS4_N_VOCAB + 4095u) / 4096u) * 2u);
    }
#undef QWEN35_ALLOC
    /* F16 K/V per full-attention layer (with MTP, the MTP block's too);
     * recurrent state and conv history per GDN layer, and with MTP the
     * copy a verify takes after its first row. */
    const uint64_t kv_bytes = (uint64_t)ctx_cap * kv_dim * 2u;
    const uint64_t state_n = v_dim * DS4_N_LIN_HEAD_DIM;
    const uint64_t hist_n = (uint64_t)(DS4_N_LIN_CONV - 1u) * conv_dim;
    const uint32_t n_layers = mtp ? DS4_N_LAYER : DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
    for (uint32_t il = 0; il < n_layers && ok; il++) {
        if (ds4_qwen35_layer_is_attention(il)) {
            g->layer_k_cache[il] = ds4_gpu_tensor_alloc(kv_bytes);
            g->layer_v_cache[il] = ds4_gpu_tensor_alloc(kv_bytes);
            ok = g->layer_k_cache[il] && g->layer_v_cache[il];
        } else {
            g->layer_lin_state[il] = qwen4_graph_alloc_f32(state_n);
            g->layer_lin_hist[il] = qwen4_graph_alloc_f32(hist_n);
            ok = g->layer_lin_state[il] && g->layer_lin_hist[il];
            if (ok && mtp) {
                g->snap_lin_state[il] = qwen4_graph_alloc_f32(state_n);
                g->snap_lin_hist[il] = qwen4_graph_alloc_f32(hist_n);
                ok = g->snap_lin_state[il] && g->snap_lin_hist[il];
            }
        }
    }
    g->host_row = xmalloc(T * E * sizeof(float));
    g->host_pos3 = xmalloc(T * 4u * sizeof(uint32_t));
    g->host_logits = xmalloc((uint64_t)DS4_N_VOCAB * sizeof(float));
    if (!ok) {
        fprintf(stderr, "ds4: Ornith graph allocation failed (ctx %u)\n", ctx_cap);
        qwen4_graph_free(g);
        return false;
    }
    return true;
}

static void qwen35_graph_reset(ds4_qwen4_gpu_graph *g) {
    (void)ds4_gpu_synchronize();
    const uint64_t conv_dim = DS4_N_LIN_CONV_DIM;
    const uint64_t v_dim = (uint64_t)DS4_N_LIN_V_HEAD * DS4_N_LIN_HEAD_DIM;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (g->layer_lin_state[il]) ds4_gpu_tensor_fill_f32(g->layer_lin_state[il], 0.0f, v_dim * DS4_N_LIN_HEAD_DIM);
        if (g->layer_lin_hist[il])
            ds4_gpu_tensor_fill_f32(g->layer_lin_hist[il], 0.0f, (uint64_t)(DS4_N_LIN_CONV - 1u) * conv_dim);
    }
    /* the MTP row at position 0 reads h_{-1} = 0 */
    if (g->mtp_h) ds4_gpu_tensor_fill_f32(g->mtp_h, 0.0f, DS4_N_EMBD);
    g->pos = 0;
    g->mtp_pos = 0;
    g->mtp_h_pos0 = 0;
    g->mtp_h_rows = 0;
    g->snap_valid = false;
    g->snap_after_first = false;
    g->verify_rows_exact = false;
}

/* Before a forward at pos0: put h_{pos0-1} into row 0 of the seed rows.  It
 * is row pos0 - mtp_h_pos0 of the previous forward (row 0 itself after a
 * reset); a rejected verify rewinds to its pos0 + 1 and still finds it. */
static bool qwen35_graph_carry_h(ds4_qwen4_gpu_graph *g, uint32_t pos0) {
    if (!g->mtp_h) return true;
    if (pos0 < g->mtp_h_pos0 || pos0 - g->mtp_h_pos0 > g->mtp_h_rows) {
        fprintf(stderr, "ds4: Ornith MTP seed for position %u is not resident\n", pos0);
        return false;
    }
    const uint64_t row = (uint64_t)DS4_N_EMBD * sizeof(float);
    const uint32_t k = pos0 - g->mtp_h_pos0;
    return k == 0 || ds4_gpu_tensor_copy(g->mtp_h, 0, g->mtp_h, (uint64_t)k * row, row) != 0;
}

/* Embedding rows (dequantized on the host from Q8_0) and text rope
 * positions (t, h, w) = (p, p, p). */
static bool qwen35_graph_stage_inputs(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                      const int *tokens, uint32_t T) {
    const uint32_t E = DS4_N_EMBD;
    for (uint32_t t = 0; t < T; t++) {
        qwen4_ref_row(m, w->token_embd, (uint64_t)tokens[t], g->host_row + (uint64_t)t * E);
        uint32_t *p4 = g->host_pos3 + (uint64_t)t * 4u;
        p4[0] = p4[1] = p4[2] = g->pos + t;
        p4[3] = 0;
    }
    return ds4_gpu_tensor_write(g->R, 0, g->host_row, (uint64_t)T * E * sizeof(float)) &&
           ds4_gpu_tensor_write(g->pos3, (uint64_t)g->pos * 16u, g->host_pos3, (uint64_t)T * 16u);
}

/* q/gate, k and v projections of the rows in mixed, then the indexer-free
 * prep: q/k norms, RoPE and the F16 K/V append at positions pos0.. */
static bool qwen35_graph_attention_prep(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                                        uint32_t il, uint32_t pos0, uint32_t T) {
    return qwen4_gemv(g->qg, m, l->attn_q, g->mixed, T) &&
           qwen4_gemv(g->kp, m, l->attn_k, g->mixed, T) &&
           qwen4_gemv(g->vp, m, l->attn_v, g->mixed, T) &&
           ds4_gpu_qwen35_attn_prep_tensor(g->q, g->gate, g->layer_k_cache[il], g->layer_v_cache[il],
                                           g->qg, g->kp, g->vp, g->pos3, m->map, m->size,
                                           l->attn_q_norm->abs_offset, l->attn_k_norm->abs_offset,
                                           T, DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT,
                                           pos0, g->alloc_cap, DS4_ROPE_FREQ_BASE, DS4_RMS_EPS) != 0;
}

/* Dense attention (the qwen4 kernels apply sigmoid(gate)) of rows
 * [r0, r0 + n) of q/gate, at positions pos0 + r0.., into the same rows of
 * attn_o.  per_row gives every row its own dispatch: the key split then
 * matches a plain decode at that position, which a speculative verify
 * needs to reproduce plain decoding bit for bit. */
static bool qwen35_graph_attend(ds4_qwen4_gpu_graph *g, uint32_t il, uint32_t pos0,
                                uint32_t r0, uint32_t n, bool per_row) {
    const float scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);
    if (r0 == 0 && !per_row) {
        return ds4_gpu_qwen4_attn_decode_tensor(g->attn_o, g->q, g->gate, g->layer_k_cache[il], g->layer_v_cache[il],
                                                NULL, NULL, n <= 2u ? g->attn_part : NULL, n,
                                                DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, pos0, 0u, 0u, scale,
                                                NULL, NULL, NULL, NULL, 0u) != 0;
    }
    const uint64_t qrow = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM * sizeof(float);
    const uint32_t step = per_row ? 1u : n;
    bool ok = true;
    for (uint32_t t = r0; t < r0 + n && ok; t += step) {
        const uint64_t off = (uint64_t)t * qrow, bytes = (uint64_t)step * qrow;
        ds4_gpu_tensor *q = ds4_gpu_tensor_view(g->q, off, bytes);
        ds4_gpu_tensor *gt = ds4_gpu_tensor_view(g->gate, off, bytes);
        ds4_gpu_tensor *o = ds4_gpu_tensor_view(g->attn_o, off, bytes);
        ok = q && gt && o &&
             ds4_gpu_qwen4_attn_decode_tensor(o, q, gt, g->layer_k_cache[il], g->layer_v_cache[il],
                                              NULL, NULL, step <= 2u ? g->attn_part : NULL, step,
                                              DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, pos0 + t, 0u, 0u, scale,
                                              NULL, NULL, NULL, NULL, 0u) != 0;
        ds4_gpu_tensor_free(q);
        ds4_gpu_tensor_free(gt);
        ds4_gpu_tensor_free(o);
    }
    return ok;
}

/* Gated full attention over every cached position: projections and prep,
 * dense attention, output projection into blk. */
static bool qwen35_graph_attention(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                                   uint32_t il, uint32_t pos0, uint32_t T) {
    return qwen35_graph_attention_prep(g, m, l, il, pos0, T) &&
           qwen35_graph_attend(g, il, pos0, 0u, T, g->verify_rows_exact && T > 1u) &&
           qwen4_gemv(g->blk, m, l->attn_output, g->attn_o, T);
}

/* Router (softmax over 256, top 8, renormalized, sigmoid shared gate), the
 * routed experts and the shared expert, reduced into blk without a residual
 * (n_hc = 0); the caller adds blk to R.  Q4_K layers take the tiled expert
 * GEMMs above one tile of tokens, as qwen4 prefill does; Q5_K layers keep
 * the per-token row kernels.  Batches above 8 rows run the shared expert as
 * dense projections, single rows and verifies keep it as an extra slot. */
static bool qwen35_graph_moe(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                             uint32_t T) {
    const uint32_t E = DS4_N_EMBD, F = DS4_N_FF_EXP, NE = DS4_N_EXPERT, K = DS4_N_EXPERT_USED;
    const uint32_t xt = l->ffn_gate_exps->type, dt = l->ffn_down_exps->type;
    bool ok = qwen4_gemv(g->router, m, l->ffn_gate_inp, g->mixed, T) &&
        ds4_gpu_qwen4_router_topk_tensor(g->selected, g->weights, g->router, g->mixed, m->map, m->size,
                                         l->ffn_gate_inp_shexp->abs_offset, l->ffn_gate_inp_shexp->type,
                                         E, g->sh_gate_logit, T, NE, K) != 0;
    const bool mm = T > 64u && xt == DS4_TENSOR_Q4_K && dt == DS4_TENSOR_Q4_K;
    const bool shared_dense = mm || T > 8u;
    if (ok && shared_dense) {
        ok = qwen4_gemv(g->sh_gate, m, l->ffn_gate_shexp, g->mixed, T) &&
             qwen4_gemv(g->sh_up, m, l->ffn_up_shexp, g->mixed, T) &&
             ds4_gpu_swiglu_tensor(g->sh_mid, g->sh_gate, g->sh_up, T * F, 0.0f, 1.0f) &&
             qwen4_gemv(g->sh_out, m, l->ffn_down_shexp, g->sh_mid, T);
    }
    if (ok && mm) {
        ok = ds4_gpu_qwen4_moe_build_lists_tensor(g->moe_lists, g->moe_counts, g->selected, T, K, NE,
                                                  g->cap_tokens) &&
             ds4_gpu_qwen4_moe_mm_mid_tensor(g->mid, g->mixed, g->moe_lists, g->moe_counts, m->map, m->size,
                                             l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset, xt,
                                             NE, T, K, K, E, F, g->cap_tokens) &&
             ds4_gpu_qwen4_moe_mm_down_tensor(g->part, g->mid, g->moe_lists, g->moe_counts, m->map, m->size,
                                              l->ffn_down_exps->abs_offset, dt, NE, T, K, K, F, E, g->cap_tokens);
    } else if (ok) {
        const uint64_t sg = shared_dense ? 0u : l->ffn_gate_shexp->abs_offset;
        const uint64_t su = shared_dense ? 0u : l->ffn_up_shexp->abs_offset;
        const uint64_t sd = shared_dense ? 0u : l->ffn_down_shexp->abs_offset;
        const uint32_t st = shared_dense ? UINT32_MAX : l->ffn_gate_shexp->type;
        const uint32_t sdt = shared_dense ? UINT32_MAX : l->ffn_down_shexp->type;
        if (xt == DS4_TENSOR_Q5_K) {
            ok = ds4_gpu_qwen35_moe_mid_tensor(g->mid, g->mixed, g->selected, m->map, m->size,
                                               l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset, xt,
                                               NE, T, K, E, F, sg, su, st) != 0;
        } else {
            ok = ds4_gpu_qwen4_moe_mid_tensor(g->mid, g->mixed, g->selected, m->map, m->size,
                                              l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset, xt,
                                              NE, T, K, E, F, sg, su, st) != 0;
        }
        if (ok && dt == DS4_TENSOR_Q5_K) {
            ok = ds4_gpu_qwen35_moe_down_tensor(g->part, g->mid, g->selected, m->map, m->size,
                                                l->ffn_down_exps->abs_offset, dt, NE, T, K, F, E, sd, sdt) != 0;
        } else if (ok) {
            ok = ds4_gpu_qwen4_moe_down_tensor(g->part, g->mid, g->selected, m->map, m->size,
                                               l->ffn_down_exps->abs_offset, dt, NE, T, K, F, E, sd, sdt) != 0;
        }
    }
    if (ok) {
        ok = ds4_gpu_qwen4_moe_reduce_tensor(g->blk, g->part, g->weights, g->sh_gate_logit,
                                             shared_dense ? g->sh_out : NULL, NULL, NULL, T, K,
                                             K + (shared_dense ? 0u : 1u), E, 0u) != 0;
    }
    return ok;
}

/* One trunk layer: pre-norm mixer and pre-norm MoE, each added to R. */
static bool qwen35_graph_layer(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                               uint32_t il, uint32_t pos0, uint32_t T) {
    const ds4_layer_weights *l = &w->layer[il];
    const uint32_t n = T * DS4_N_EMBD;
    bool ok = ds4_gpu_rms_norm_weight_rows_tensor(g->mixed, g->R, m->map, m->size, l->attn_norm->abs_offset,
                                                  DS4_N_EMBD, T, DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_qwen35_layer_is_attention(il) ? qwen35_graph_attention(g, m, l, il, pos0, T)
                                                    : qwen4_graph_linear(g, m, l, il, T);
    if (ok) ok = ds4_gpu_add_tensor(g->R, g->R, g->blk, n) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->mixed, g->R, m->map, m->size, l->ffn_norm->abs_offset,
                                                     DS4_N_EMBD, T, DS4_RMS_EPS) != 0;
    if (ok) ok = qwen35_graph_moe(g, m, l, T);
    if (ok) ok = ds4_gpu_add_tensor(g->R, g->R, g->blk, n) != 0;
    return ok;
}

/* Forward T tokens at g->pos..; logits_out (optional) receives the last
 * row, or all T rows with all_rows (an MTP verify).  Causal by
 * construction: the recurrent kernels walk tokens in order and attention
 * reads the caches written for the same chunk.  A caller that sets
 * g->snap_after_first gets the GDN state after row 0 in the snapshot set
 * (g->snap_valid, g->snap_pos). */
static bool qwen35_graph_forward_tokens(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                        const int *tokens, uint32_t T, float *logits_out, bool all_rows) {
    if (!g) return false;
    const bool snap = g->snap_after_first;
    g->snap_after_first = false;
    if (T == 0 || T > g->cap_tokens || g->pos + T > g->ctx_cap) return false;
    if (all_rows && (!logits_out || !g->mtp_h || T > g->n_logit_rows)) return false;
    for (uint32_t t = 0; t < T; t++) {
        if (tokens[t] < 0 || tokens[t] >= (int)DS4_N_VOCAB) {
            fprintf(stderr, "ds4: Ornith token id %d is outside the vocabulary\n", tokens[t]);
            return false;
        }
    }
    const uint32_t pos0 = g->pos;
    const uint64_t row = (uint64_t)DS4_N_EMBD * sizeof(float);
    if (!qwen35_graph_stage_inputs(g, m, w, tokens, T)) return false;
    if (!glm_graph_begin_commands_if_needed()) return false;
    g->snap_after_first = snap;
    if (snap) g->snap_pos = pos0 + 1u;
    bool ok = qwen35_graph_carry_h(g, pos0);
    for (uint32_t il = 0; il + DS4_N_NEXTN_PREDICT < DS4_N_LAYER && ok; il++) {
        ok = qwen35_graph_layer(g, m, w, il, pos0, T);
    }
    if (ok && g->mtp_h) {
        /* post-output_norm rows: the MTP seed rows 1..T and the LM head input */
        ds4_gpu_tensor *h = ds4_gpu_tensor_view(g->mtp_h, row, (uint64_t)T * row);
        ok = h && ds4_gpu_rms_norm_weight_rows_tensor(h, g->R, m->map, m->size, w->output_norm->abs_offset,
                                                      DS4_N_EMBD, T, DS4_RMS_EPS) != 0;
        if (ok && logits_out) {
            ds4_gpu_tensor *last = all_rows ? NULL : ds4_gpu_tensor_view(g->mtp_h, (uint64_t)T * row, row);
            ok = (all_rows || last) && qwen4_gemv(g->logits, m, w->output, all_rows ? h : last, all_rows ? T : 1u);
            ds4_gpu_tensor_free(last);
        }
        ds4_gpu_tensor_free(h);
    } else if (ok && logits_out) {
        ds4_gpu_tensor *last = ds4_gpu_tensor_view(g->R, (uint64_t)(T - 1u) * row, row);
        ok = last && ds4_gpu_rms_norm_weight_rows_tensor(g->mixed, last, m->map, m->size, w->output_norm->abs_offset,
                                                         DS4_N_EMBD, 1u, DS4_RMS_EPS) != 0 &&
             qwen4_gemv(g->logits, m, w->output, g->mixed, 1u);
        ds4_gpu_tensor_free(last);
    }
    if (!ds4_gpu_end_commands()) ok = false;
    const uint32_t n_rows = all_rows ? T : 1u;
    if (ok && logits_out) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits_out, (uint64_t)n_rows * DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (ok) {
        g->pos += T;
        if (g->mtp_h) {
            g->mtp_h_pos0 = pos0;
            g->mtp_h_rows = T;
        }
    }
    if (snap) g->snap_valid = ok;
    g->snap_after_first = false;
    return ok;
}

/* The MTP block (blk.40) over tokens at positions pos0..pos0+T-1.  Row t
 * reads x0 = eh_proj . [RMSNorm(embed(token_t)) * enorm | RMSNorm(h) * hnorm]
 * with h = h_{pos0+t-1} from mtp_h, and appends its K/V to the block's own
 * cache: the catch-up that keeps that cache complete for every position.
 * Rows run in sub-batches of cap_tokens.  With want_draft the last row
 * continues through attention, the MoE, shared_head_norm and the output
 * head, and *draft_out gets the argmax: the token after tokens[T-1]. */
static bool qwen35_graph_mtp(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                             const int *tokens, uint32_t T, uint32_t pos0, bool want_draft, int *draft_out) {
    const uint32_t il = DS4_N_LAYER - 1u, E = DS4_N_EMBD;
    const ds4_layer_weights *l = &w->layer[il];
    const uint64_t row = (uint64_t)E * sizeof(float);
    const uint64_t qrow = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM * sizeof(float);
    if (!g->mtp_h || T == 0 || pos0 + T > g->ctx_cap || (want_draft && !draft_out) ||
        pos0 < g->mtp_h_pos0 || pos0 - g->mtp_h_pos0 + T - 1u > g->mtp_h_rows) {
        return false;
    }
    for (uint32_t t = 0; t < T; t++) {
        if (tokens[t] < 0 || tokens[t] >= (int)DS4_N_VOCAB) return false;
    }
    bool ok = true;
    uint32_t sub0 = pos0, last = 0;
    for (uint32_t done = 0; done < T && ok;) {
        const uint32_t n = T - done < g->cap_tokens ? T - done : g->cap_tokens;
        const uint32_t p = pos0 + done;
        for (uint32_t t = 0; t < n; t++) {
            qwen4_ref_row(m, w->token_embd, (uint64_t)tokens[done + t], g->host_row + (uint64_t)t * E);
            uint32_t *p4 = g->host_pos3 + (uint64_t)t * 4u;
            p4[0] = p4[1] = p4[2] = p + t;
            p4[3] = 0;
        }
        ok = ds4_gpu_tensor_write(g->mtp_e, 0, g->host_row, (uint64_t)n * row) &&
             ds4_gpu_tensor_write(g->pos3, (uint64_t)p * 16u, g->host_pos3, (uint64_t)n * 16u) &&
             glm_graph_begin_commands_if_needed();
        ds4_gpu_tensor *h = ok ? ds4_gpu_tensor_view(g->mtp_h, (uint64_t)(p - g->mtp_h_pos0) * row,
                                                     (uint64_t)n * row) : NULL;
        ok = ok && h &&
             ds4_gpu_qwen35_mtp_concat_tensor(g->mtp_cat, g->mtp_e, h, m->map, m->size,
                                              l->nextn_enorm->abs_offset, l->nextn_hnorm->abs_offset,
                                              E, n, DS4_RMS_EPS) != 0 &&
             qwen4_gemv(g->mtp_R, m, l->nextn_eh_proj, g->mtp_cat, n) &&
             ds4_gpu_rms_norm_weight_rows_tensor(g->mixed, g->mtp_R, m->map, m->size, l->attn_norm->abs_offset,
                                                 E, n, DS4_RMS_EPS) != 0 &&
             qwen35_graph_attention_prep(g, m, l, il, p, n);
        ds4_gpu_tensor_free(h);
        sub0 = p;
        last = n - 1u;
        done += n;
        /* the next sub-batch rewrites the staged embeddings */
        if (done < T && !ds4_gpu_end_commands()) ok = false;
    }
    if (ok && want_draft) {
        ds4_gpu_tensor *x = ds4_gpu_tensor_view(g->mtp_R, (uint64_t)last * row, row);
        ds4_gpu_tensor *o = ds4_gpu_tensor_view(g->attn_o, (uint64_t)last * qrow, qrow);
        ok = x && o &&
             qwen35_graph_attend(g, il, sub0, last, 1u, false) &&
             qwen4_gemv(g->blk, m, l->attn_output, o, 1u) &&
             ds4_gpu_add_tensor(x, x, g->blk, E) != 0 &&
             ds4_gpu_rms_norm_weight_rows_tensor(g->mixed, x, m->map, m->size, l->ffn_norm->abs_offset,
                                                 E, 1u, DS4_RMS_EPS) != 0 &&
             qwen35_graph_moe(g, m, l, 1u) &&
             ds4_gpu_add_tensor(x, x, g->blk, E) != 0 &&
             ds4_gpu_rms_norm_weight_rows_tensor(g->mixed, x, m->map, m->size,
                                                 l->nextn_shared_head_norm->abs_offset, E, 1u, DS4_RMS_EPS) != 0 &&
             qwen4_gemv(g->logits, m, w->output, g->mixed, 1u) &&
             ds4_gpu_qwen4_argmax_tensor(g->mtp_argmax, g->mtp_argmax_tmp, g->logits, DS4_N_VOCAB) != 0;
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(o);
    }
    if (!ds4_gpu_end_commands()) ok = false;
    if (ok && want_draft) {
        int32_t token = -1;
        ok = ds4_gpu_tensor_read(g->mtp_argmax, 0, &token, sizeof(token)) != 0 &&
             token >= 0 && token < (int32_t)DS4_N_VOCAB;
        if (ok) *draft_out = token;
    }
    if (ok) g->mtp_pos = pos0 + T;
    return ok;
}

/* A rejected draft returns to the state the verify snapshotted after its
 * first row: swap the live and snapshot GDN buffers, as the qwen4 cycle
 * does, without the PLE state Ornith does not have.  K/V rows past the
 * restored position are overwritten by the next forward, and the seed row
 * for it is still resident (the verify's row 1). */
static bool qwen35_graph_state_swap(ds4_qwen4_gpu_graph *g) {
    if (!g->snap_valid) return false;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!g->snap_lin_state[il]) continue;
        ds4_gpu_tensor *t = g->layer_lin_state[il];
        g->layer_lin_state[il] = g->snap_lin_state[il];
        g->snap_lin_state[il] = t;
        t = g->layer_lin_hist[il];
        g->layer_lin_hist[il] = g->snap_lin_hist[il];
        g->snap_lin_hist[il] = t;
    }
    g->pos = g->snap_pos;
    g->snap_valid = false;
    return true;
}

static int generate_qwen35_metal_argmax(
        const ds4_model *model, const ds4_vocab *vocab, const ds4_weights *weights,
        const token_vec *prompt, int n_predict, int ctx_size, uint32_t prefill_chunk,
        ds4_token_emit_fn emit, ds4_generation_done_fn done, void *emit_ud,
        ds4_session_progress_fn progress, void *progress_ud) {
    if (!prompt || prompt->len <= 0 || prompt->len >= ctx_size) {
        fprintf(stderr, "ds4: prompt is empty or exceeds context size\n");
        return 1;
    }
    const uint32_t cap = prefill_chunk && prefill_chunk < (uint32_t)ctx_size ?
        prefill_chunk : qwen35_prefill_chunk_tokens((uint32_t)ctx_size);
    ds4_qwen4_gpu_graph *g = xcalloc(1, sizeof(*g));
    if (!qwen35_graph_alloc(g, (uint32_t)ctx_size, cap, false)) {
        free(g);
        return 1;
    }
    qwen35_graph_reset(g);
    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    const double t_prefill0 = now_sec();
    bool ok = true;
    for (int i = 0; i < prompt->len && ok;) {
        uint32_t chunk = (uint32_t)(prompt->len - i);
        if (chunk > g->cap_tokens) chunk = g->cap_tokens;
        ok = qwen35_graph_forward_tokens(g, model, weights, prompt->v + i, chunk,
                                         i + (int)chunk == prompt->len ? logits : NULL, false);
        i += (int)chunk;
        if (progress) progress(progress_ud, "prefill_chunk", i, prompt->len);
    }
    const double t_prefill1 = now_sec();
    int n_generated = 0;
    const double t_decode0 = now_sec();
    for (int i = 0; ok && i < n_predict && g->pos < g->ctx_cap; i++) {
        const int token = sample_argmax(logits, DS4_N_VOCAB);
        if (vocab_token_is_generation_stop(vocab, token)) break;
        if (emit) emit(emit_ud, token);
        n_generated++;
        if (i == n_predict - 1 || g->pos + 1u >= g->ctx_cap) break;
        ok = qwen35_graph_forward_tokens(g, model, weights, &token, 1, logits, false);
    }
    const double t_decode1 = now_sec();
    if (!ok) fprintf(stderr, "ds4: Ornith forward failed at position %u\n", g->pos);
    if (ok && done) done(emit_ud);
    ds4_log(stderr, DS4_LOG_TIMING, "ds4: Ornith prefill: %.2f t/s, generation: %.2f t/s\n",
            t_prefill1 > t_prefill0 ? (double)prompt->len / (t_prefill1 - t_prefill0) : 0.0,
            t_decode1 > t_decode0 ? (double)n_generated / (t_decode1 - t_decode0) : 0.0);
    free(logits);
    qwen4_graph_free(g);
    free(g);
    return ok ? 0 : 1;
}
```

- [ ] **Step 5: Update the callers in `ds4.c` to the new signatures (behaviour unchanged)**

In `ds4_session_create`'s Ornith branch, replace

```c
        if (!qwen35_graph_alloc(&s->qwen4_graph, (uint32_t)ctx_size, cap_tokens)) {
```

with

```c
        if (!qwen35_graph_alloc(&s->qwen4_graph, (uint32_t)ctx_size, cap_tokens, false)) {
```

In `ds4_session_sync_internal`'s Ornith branch, replace

```c
            if (!qwen35_graph_forward_tokens(g, &e->model, &e->weights, prompt->v + i, chunk, s->logits)) {
```

with

```c
            if (!qwen35_graph_forward_tokens(g, &e->model, &e->weights, prompt->v + i, chunk, s->logits, false)) {
```

In `ds4_session_eval_internal`'s Ornith branch, replace

```c
        if (!qwen35_graph_forward_tokens(g, &e->model, &e->weights, &token, 1, s->logits)) {
```

with

```c
        if (!qwen35_graph_forward_tokens(g, &e->model, &e->weights, &token, 1, s->logits, false)) {
```

(Task 3 turns MTP on in the session; nothing in this task changes a session's behaviour.)

- [ ] **Step 6: Build and run the graph test (GPU window)**

Run: `make ds4 ds4-server ds4-agent tests/test_qwen35_graph && DS4_ORNITH_MODEL=$HOME/orca/workspaces/ds4-metal-data/gguf/ornith/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf make test-qwen35-graph`
Expected: no build warnings; the lines `MTP graph logits: bit-identical ...`, five `verify at ...` lines, `prefill to 3700 after the verifies: bit-identical`, `catch-up K rows: ...` and `drafts after 600 tokens: ...`, then `qwen35 graph: ok`.

If a verify row differs, the 2-row pass and the 1-row decode picked different arithmetic somewhere other than attention. Use superpowers:systematic-debugging: dump the row after each stage (`DS4_QWEN35_*` knobs are not needed; add a temporary `ds4_gpu_tensor_read` of `g->R` after each layer in the test build) and compare row 0 of the verify with the plain decode to find the first diverging layer and stage. Fix by giving that stage per-row dispatches under `verify_rows_exact`, as `qwen35_graph_attend` does, then rerun. Record the finding in the ledger.

- [ ] **Step 7: M1 regressions stay green (GPU window)**

Run:
```bash
export DS4_ORNITH_MODEL=$HOME/orca/workspaces/ds4-metal-data/gguf/ornith/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf
make test-qwen35-kernels test-qwen35-session
tests/ornith/test_loader.sh
tests/ornith/oneshot.sh
S=$(mktemp -d); python3 tests/ornith/gate1.py "$S/g" > "$S/compare-default.txt"
cmp "$S/compare-default.txt" speed-bench/ornith/m1/compare-default.txt && echo "gate1 default: identical to M1"
```
Expected: all ok, and `gate1 default: identical to M1` (the non-MTP graph runs the M1 dispatches, so even `max_delta` is unchanged).

- [ ] **Step 8: Qwen3.8 checks (GPU window)**

Run: `make test-qwen4-kernels test-qwen4-q2 && speed-bench/qwen-regression/run.sh fast`
Expected: all PASS, `qwen_gate: PASS` (the shared struct gained fields and `qwen4_graph_free` frees one more pointer, NULL on Qwen3.8).

- [ ] **Step 9: Commit**

```bash
git add ds4.c ds4_qwen35moe.inc tests/test_qwen35_graph.c Makefile
git commit -m "ds4: MTP state in the Ornith graph

Graphs allocated with MTP keep the trunk's post-output_norm rows as the
MTP seed (mtp_h, with a one-row carry across forwards), the MTP block's
K/V cache, the GDN verify snapshot and two logit rows.  qwen35_graph_mtp
runs blk.40: catch-up rows append their K/V (row p from token p and
h_{p-1}, llama.cpp's convention) and the draft row continues through
attention, the MoE, shared_head_norm and the head.  A verify runs its
attention one row per dispatch so each row equals plain decoding bit
for bit; qwen35_graph_state_swap restores the after-row-0 snapshot.
Graphs without MTP run the M1 dispatches unchanged.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2"
```

---

### Task 3: Ornith speculative sessions

Turn `--mtp` on for Ornith: sessions allocate the MTP graph, sync and plain eval keep the MTP cache complete, the speculative entry points run `ds4_session_qwen35_spec_cycle`, and the engine reports two draft tokens.

**Files:**
- Modify: `ds4.c`
- Create: `tests/test_qwen35_mtp.c`
- Modify: `tests/ornith/test_loader.sh`
- Modify: `Makefile`

**Interfaces:**
- Consumes (Task 2): `qwen35_graph_alloc(..., bool mtp)`, `qwen35_graph_forward_tokens(..., bool all_rows)`, `qwen35_graph_mtp`, `qwen35_graph_state_swap`, `g->mtp_h`, `g->snap_after_first`, `g->verify_rows_exact`. Session fields `glm_mtp_have`, `glm_mtp_draft`, `glm_mtp_parent`, `glm_spec_inside`, `qwen4_verify_logits`, `qwen4_spec_cycles`, `qwen4_spec_accepted`, `mtp_draft_valid`.
- Produces:
  ```c
  static int ds4_session_qwen35_spec_cycle(ds4_session *s, int first_token, float temperature, int top_k,
                                           float top_p, float min_p, uint64_t *rng, bool exact_sampling,
                                           int *accepted, int accepted_cap, char *err, size_t errlen);
  ```
  `ds4_engine_mtp_draft_tokens()` = 2 for Ornith opened with `--mtp`; the knobs `DS4_QWEN35_SPEC_TRACE` and `DS4_QWEN35_SPEC_FORCE_ACCEPT`; `--mtp-timing` prints `ds4: Ornith mtp: N verify cycles, M drafts accepted (P%)` when the session is freed.

- [ ] **Step 1: Write the failing session test**

Create `tests/test_qwen35_mtp.c`:

```c
/* Real-model Ornith MTP session checks (--mtp, prefill chunk 512).
 * Usage: test_qwen35_mtp MODEL
 *  1. greedy speculative decoding commits exactly the plain argmax
 *     sequence, with logits bit-identical to a plain session after every
 *     cycle, and drafts are accepted (at least 10 of 150 cycles) as well as
 *     not accepted (at least 1);
 *  2. DS4_QWEN35_SPEC_FORCE_ACCEPT commits drafts that need not be the
 *     argmax; a plain session fed the same tokens stays bit-identical;
 *  3. a divergent prompt after speculative cycles resets the MTP state and
 *     the next cycles still match plain decoding;
 *  4. near the end of the context the cycle falls back to plain steps, never
 *     passes the context, and stops with "context is full". */
#define _POSIX_C_SOURCE 200809L
#include "../ds4.h"
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CTX = 4160, CHUNK = 512 };

static int g_vocab;
static float *g_a, *g_b;

static void sync_len(ds4_session *s, const ds4_tokens *tokens, int n) {
    ds4_tokens prefix = *tokens;
    prefix.len = n;
    char err[256] = {0};
    const int rc = ds4_session_sync(s, &prefix, err, sizeof(err));
    if (rc) fprintf(stderr, "sync %d: %s\n", n, err);
    assert(rc == 0 && ds4_session_pos(s) == n);
}

static void eval_plain(ds4_session *s, int token) {
    char err[256] = {0};
    if (ds4_session_eval(s, token, err, sizeof(err)) != 0) {
        fprintf(stderr, "plain eval: %s\n", err);
        exit(1);
    }
}

static void same_logits(ds4_session *spec, ds4_session *plain) {
    assert(ds4_session_copy_logits(spec, g_a, g_vocab) == g_vocab);
    assert(ds4_session_copy_logits(plain, g_b, g_vocab) == g_vocab);
    assert(memcmp(g_a, g_b, (size_t)g_vocab * sizeof(float)) == 0);
    assert(ds4_session_pos(spec) == ds4_session_pos(plain));
}

/* One speculative cycle from spec's argmax; plain follows the committed
 * tokens.  Returns the committed count, or -1 with err set. */
static int cycle(ds4_session *spec, ds4_session *plain, int eos, bool expect_argmax, char *err, size_t errlen) {
    const int first = ds4_session_argmax(spec);
    int acc[17];
    const int n = ds4_session_eval_speculative_argmax(spec, first, 16, eos, acc, 17, err, errlen);
    if (n < 0) return -1;
    assert(n >= 1 && n <= 2 && acc[0] == first);
    for (int i = 0; i < n; i++) {
        if (expect_argmax) assert(ds4_session_argmax(plain) == acc[i]);
        eval_plain(plain, acc[i]);
    }
    same_logits(spec, plain);
    return n;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL\n", argv[0]);
        return 1;
    }
    unsetenv("DS4_QWEN35_SPEC_FORCE_ACCEPT");
    ds4_engine_options opt = {.model_path = argv[1], .context_size = CTX, .prefill_chunk = CHUNK,
                              .backend = DS4_BACKEND_METAL, .glm_mtp = true};
    ds4_engine *engine = NULL;
    assert(ds4_engine_open(&engine, &opt) == 0 && ds4_engine_is_qwen35moe(engine));
    assert(ds4_engine_mtp_draft_tokens(engine) == 2);
    const int eos = ds4_token_eos(engine);
    FILE *fp = fopen("speed-bench/promessi_sposi.txt", "rb");
    assert(fp);
    char *text = calloc(40001, 1);
    assert(fread(text, 1, 40000, fp) > 0);
    fclose(fp);
    ds4_tokens tokens = {0};
    ds4_tokenize_text(engine, text, &tokens);
    free(text);
    assert(tokens.len >= CTX);
    g_vocab = ds4_engine_vocab_size(engine);
    g_a = malloc((size_t)g_vocab * sizeof(float));
    g_b = malloc((size_t)g_vocab * sizeof(float));
    ds4_session *spec = NULL, *plain = NULL;
    assert(ds4_session_create(&spec, engine, CTX) == 0);
    assert(ds4_session_create(&plain, engine, CTX) == 0);
    char err[256] = {0};

    /* 1. greedy: the plain argmax sequence, bit-identical logits */
    sync_len(spec, &tokens, 1500);
    sync_len(plain, &tokens, 1500);
    same_logits(spec, plain);
    int cycles = 0, accepts = 0;
    for (; cycles < 150; cycles++) {
        const int n = cycle(spec, plain, eos, true, err, sizeof(err));
        assert(n > 0);
        accepts += n == 2;
    }
    printf("  greedy: %d cycles, %d accepted, bit-identical to plain decoding\n", cycles, accepts);
    assert(accepts >= 10 && cycles - 1 - accepts >= 1);

    /* 2. forced accepts: committed drafts need not be the argmax */
    setenv("DS4_QWEN35_SPEC_FORCE_ACCEPT", "1", 1);
    int forced = 0;
    for (int i = 0; i < 20; i++) forced += cycle(spec, plain, eos, false, err, sizeof(err)) == 2;
    unsetenv("DS4_QWEN35_SPEC_FORCE_ACCEPT");
    printf("  forced accepts: %d of 20 cycles committed two tokens, bit-identical\n", forced);
    assert(forced >= 15);

    /* 3. a divergent prompt resets the MTP state */
    ds4_tokens other = {0};
    ds4_tokenize_text(engine, "Xin chào! Hôm nay tôi muốn kể cho bạn nghe về Hà Nội và Hồ Gươm.", &other);
    sync_len(spec, &other, other.len);
    sync_len(plain, &other, other.len);
    same_logits(spec, plain);
    for (int i = 0; i < 30; i++) assert(cycle(spec, plain, eos, true, err, sizeof(err)) > 0);
    printf("  divergent prompt: 30 cycles bit-identical\n");

    /* 4. the end of the context */
    sync_len(spec, &tokens, CTX - 6);
    sync_len(plain, &tokens, CTX - 6);
    int rc = 0;
    for (int i = 0; i < 12; i++) {
        rc = cycle(spec, plain, eos, true, err, sizeof(err));
        assert(ds4_session_pos(spec) <= CTX);
        if (rc < 0) break;
    }
    printf("  context end: '%s' at position %d\n", err, ds4_session_pos(spec));
    assert(rc < 0 && strcmp(err, "context is full") == 0);

    ds4_session_free(plain);
    ds4_session_free(spec);
    ds4_tokens_free(&other);
    ds4_tokens_free(&tokens);
    free(g_a);
    free(g_b);
    ds4_engine_close(engine);
    printf("qwen35 mtp: ok\n");
    return 0;
}
```

In `Makefile`, in the same Darwin block as `tests/test_qwen35_graph`, add:

```make
tests/test_qwen35_mtp.o: tests/test_qwen35_mtp.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_qwen35_mtp.c

tests/test_qwen35_mtp: tests/test_qwen35_mtp.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

```

After the `test-qwen35-graph` recipe, add:

```make
.PHONY: test-qwen35-mtp
test-qwen35-mtp: tests/test_qwen35_mtp
	./tests/test_qwen35_mtp "$(DS4_ORNITH_MODEL)"
```

In the `clean` recipe, extend the Ornith line to `rm -f tests/test_qwen35_kernels tests/test_qwen35_session tests/test_qwen35_graph tests/test_qwen35_mtp`.

Append to `tests/ornith/test_loader.sh`, right before the final `echo "ornith loader: ok"`:

```sh
# M2: --mtp runs the embedded blk.40 head; an external MTP model and exact
# speculative sampling stay refused.
if ./ds4 -m "$model" --raw --mtp --mtp-exact-sampling -p hi -n 1 > "$tmp/mtp_exact.txt" 2>&1; then
    echo "--mtp-exact-sampling accepted for Ornith"; exit 1
fi
grep -q -- '--mtp-exact-sampling is not supported' "$tmp/mtp_exact.txt" || { cat "$tmp/mtp_exact.txt"; exit 1; }
if ./ds4 -m "$model" --raw --mtp-model "$model" -p hi -n 1 > "$tmp/mtp_model.txt" 2>&1; then
    echo "--mtp-model accepted for Ornith"; exit 1
fi
grep -q -- '--mtp-model is not supported' "$tmp/mtp_model.txt" || { cat "$tmp/mtp_model.txt"; exit 1; }
```

and in the header comment of `tests/ornith/test_loader.sh` replace

```sh
# refuses a non-Metal backend, --batched-session and a context above the
# native 262144, and ds4-server / ds4-agent refuse Ornith until milestone M3.
```

with

```sh
# refuses a non-Metal backend, --batched-session, a context above the
# native 262144, --mtp-model and --mtp-exact-sampling, and ds4-server /
# ds4-agent refuse Ornith until milestone M3.
```

- [ ] **Step 2: Run the tests and confirm they fail (GPU window)**

Run: `export DS4_ORNITH_MODEL=...23G-ICE.gguf; make test-qwen35-mtp`
Expected: FAIL: the open with `.glm_mtp = true` returns non-zero (`--mtp is not supported`), so `assert(ds4_engine_open(...) == 0 ...)` fails.

Run: `tests/ornith/test_loader.sh`
Expected: FAIL at the new block: the refusal message names `--mtp`, not `--mtp-exact-sampling`.

- [ ] **Step 3: Open gate, embedded-MTP check and draft count**

In `ds4.c`, in the Ornith open gate inside `ds4_engine_open_internal`, replace

```c
            opt->glm_mtp ? "--mtp" :
```

with

```c
            opt->dspark_exact_sampling ? "--mtp-exact-sampling" :
```

In the embedded-MTP check a little further down, replace

```c
    if (opt->glm_mtp &&
        ((DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_GLM_DSA && !ds4_model_is_qwen4()) ||
         DS4_N_NEXTN_PREDICT == 0)) {
```

with

```c
    if (opt->glm_mtp &&
        ((DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_GLM_DSA && !ds4_model_is_qwen4() &&
          !ds4_model_is_qwen35moe()) ||
         DS4_N_NEXTN_PREDICT == 0)) {
```

In `ds4_engine_mtp_draft_tokens`, replace

```c
    if (e && DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        return e->glm_mtp && DS4_N_NEXTN_PREDICT != 0 ? 2 : 0;
    }
```

with

```c
    if (e && DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        return e->glm_mtp && DS4_N_NEXTN_PREDICT != 0 ? 2 : 0;
    }
    if (e && ds4_model_is_qwen35moe()) {
        return e->glm_mtp && DS4_N_NEXTN_PREDICT != 0 && e->backend == DS4_BACKEND_METAL ? 2 : 0;
    }
```

- [ ] **Step 4: Memory estimate reserves the MTP share**

In `ds4_context_memory_estimate_with_prefill_mode`, replace the Ornith block

```c
    if (ds4_backend_uses_graph(backend) && ds4_model_is_qwen35moe()) {
        /* F16 K/V per full-attention trunk layer, rope positions, fixed GDN
         * state and conv history; transients scale with the prefill chunk. */
        const uint64_t T = prefill_chunk ? prefill_chunk : qwen35_prefill_chunk_tokens(ctx);
        const uint64_t E = DS4_N_EMBD, kv_row = 2ull * DS4_N_HEAD_KV * DS4_N_HEAD_DIM * 2u;
        uint32_t n_attn = 0;
        for (uint32_t il = 0; il + DS4_N_NEXTN_PREDICT < DS4_N_LAYER; il++) n_attn += ds4_qwen35_layer_is_attention(il);
        const uint32_t n_lin = DS4_N_LAYER - DS4_N_NEXTN_PREDICT - n_attn;
        m.prefill_cap = (uint32_t)T;
        m.raw_cap = ctx;
        m.raw_bytes = (uint64_t)n_attn * ctx * kv_row + (uint64_t)ctx * 16u;
        m.scratch_bytes = T * (8u * E + 4u * DS4_N_LIN_CONV_DIM + 6u * (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM +
                               DS4_N_EXPERT + (uint64_t)(DS4_N_EXPERT_USED + 1u) * (E + DS4_N_FF_EXP)) * 4u +
                          (uint64_t)n_lin * ((uint64_t)DS4_N_LIN_V_HEAD * DS4_N_LIN_HEAD_DIM * DS4_N_LIN_HEAD_DIM +
                                             (uint64_t)(DS4_N_LIN_CONV - 1u) * DS4_N_LIN_CONV_DIM) * sizeof(float);
        m.total_bytes = m.raw_bytes + m.scratch_bytes;
        return m;
    }
```

with

```c
    if (ds4_backend_uses_graph(backend) && ds4_model_is_qwen35moe()) {
        /* F16 K/V per full-attention layer, the MTP block's included, rope
         * positions, fixed GDN state and conv history with their verify
         * snapshot, two logit rows; transients scale with the prefill chunk,
         * MTP staging included.  The estimate cannot see --mtp, so it
         * reserves the MTP share (2 KiB per context token plus ~66 MB),
         * as the Qwen3.8 estimate reserves its snapshots. */
        const uint64_t T = prefill_chunk ? prefill_chunk : qwen35_prefill_chunk_tokens(ctx);
        const uint64_t E = DS4_N_EMBD, kv_row = 2ull * DS4_N_HEAD_KV * DS4_N_HEAD_DIM * 2u;
        uint32_t n_attn = 0;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) n_attn += ds4_qwen35_layer_is_attention(il);
        const uint32_t n_lin = DS4_N_LAYER - n_attn;
        m.prefill_cap = (uint32_t)T;
        m.raw_cap = ctx;
        m.raw_bytes = (uint64_t)n_attn * ctx * kv_row + (uint64_t)ctx * 16u;
        m.scratch_bytes = T * (13u * E + 4u * DS4_N_LIN_CONV_DIM + 6u * (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM +
                               DS4_N_EXPERT + (uint64_t)(DS4_N_EXPERT_USED + 1u) * (E + DS4_N_FF_EXP)) * 4u +
                          2ull * DS4_N_VOCAB * sizeof(float) +
                          2ull * n_lin * ((uint64_t)DS4_N_LIN_V_HEAD * DS4_N_LIN_HEAD_DIM * DS4_N_LIN_HEAD_DIM +
                                          (uint64_t)(DS4_N_LIN_CONV - 1u) * DS4_N_LIN_CONV_DIM) * sizeof(float);
        m.total_bytes = m.raw_bytes + m.scratch_bytes;
        return m;
    }
```

- [ ] **Step 5: Session create, free, sync and eval**

In `ds4_session_create`'s Ornith branch, replace

```c
        if (!qwen35_graph_alloc(&s->qwen4_graph, (uint32_t)ctx_size, cap_tokens, false)) {
```

with

```c
        if (!qwen35_graph_alloc(&s->qwen4_graph, (uint32_t)ctx_size, cap_tokens, e->glm_mtp)) {
```

In `ds4_session_free`, replace

```c
        if (s->qwen35_graph_ready) {
            qwen4_graph_free(&s->qwen4_graph);
        } else
```

with

```c
        if (s->qwen35_graph_ready) {
            if (s->engine && s->engine->glm_mtp_timing && s->qwen4_spec_cycles) {
                fprintf(stderr, "ds4: Ornith mtp: %" PRIu64 " verify cycles, %" PRIu64 " drafts accepted (%.1f%%)\n",
                        s->qwen4_spec_cycles, s->qwen4_spec_accepted,
                        100.0 * (double)s->qwen4_spec_accepted / (double)s->qwen4_spec_cycles);
            }
            free(s->qwen4_verify_logits);
            qwen4_graph_free(&s->qwen4_graph);
        } else
```

In `ds4_session_sync_internal`'s Ornith branch, replace

```c
    if (ds4_session_is_qwen35(s)) {
        ds4_engine *e = s->engine;
        ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
        int start = 0;
```

with

```c
    if (ds4_session_is_qwen35(s)) {
        ds4_engine *e = s->engine;
        ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
        int start = 0;
        s->glm_mtp_have = 0;
```

and in the same branch replace

```c
            if (!qwen35_graph_forward_tokens(g, &e->model, &e->weights, prompt->v + i, chunk, s->logits, false)) {
                snprintf(err, errlen, "Ornith prefill failed at token %d", i);
                return 1;
            }
```

with

```c
            if (!qwen35_graph_forward_tokens(g, &e->model, &e->weights, prompt->v + i, chunk, s->logits, false)) {
                snprintf(err, errlen, "Ornith prefill failed at token %d", i);
                return 1;
            }
            /* MTP catch-up: the block's K/V rows for this chunk's positions */
            if (g->mtp_h && !qwen35_graph_mtp(g, &e->model, &e->weights, prompt->v + i, chunk, (uint32_t)i,
                                              false, NULL)) {
                snprintf(err, errlen, "Ornith MTP catch-up failed at token %d", i);
                return 1;
            }
```

In `ds4_session_eval_internal`'s Ornith branch, replace

```c
        if (!qwen35_graph_forward_tokens(g, &e->model, &e->weights, &token, 1, s->logits, false)) {
            if (errlen) snprintf(err, errlen, "Ornith decode failed at position %u", g->pos);
            s->checkpoint_valid = false;
            return 1;
        }
        token_vec_push(&s->checkpoint, token);
```

with

```c
        if (!s->glm_spec_inside) s->glm_mtp_have = 0;
        if (!qwen35_graph_forward_tokens(g, &e->model, &e->weights, &token, 1, s->logits, false)) {
            if (errlen) snprintf(err, errlen, "Ornith decode failed at position %u", g->pos);
            s->checkpoint_valid = false;
            return 1;
        }
        /* A token that did not come through the speculative cycle (plain
         * decode, server-forced tokens) still gets its MTP row, so later
         * drafts attend a complete cache; the cycle writes its own rows. */
        if (!s->glm_spec_inside && g->mtp_h &&
            !qwen35_graph_mtp(g, &e->model, &e->weights, &token, 1u, g->pos - 1u, false, NULL)) {
            if (errlen) snprintf(err, errlen, "Ornith MTP catch-up failed at position %u", g->pos - 1u);
            s->checkpoint_valid = false;
            return 1;
        }
        token_vec_push(&s->checkpoint, token);
```

- [ ] **Step 6: The Ornith speculative cycle**

In `ds4.c`, insert right before the line `int ds4_session_glm_tp_spec_cycle(ds4_session *s, int token, int limit,`:

```c
#ifdef DS4_HAS_QWEN4_METAL
/* test knobs: DS4_QWEN35_SPEC_FORCE_ACCEPT commits every draft,
 * DS4_QWEN35_SPEC_TRACE logs each cycle */
static bool qwen35_spec_trace(void) {
    static int v = -1;
    if (v < 0) v = getenv("DS4_QWEN35_SPEC_TRACE") != NULL;
    return v != 0;
}

static bool qwen35_spec_force_accept(void) {
    return getenv("DS4_QWEN35_SPEC_FORCE_ACCEPT") != NULL;
}

/* Run the MTP block over tokens at pos0.. and keep the draft that follows
 * the last of them; without a draft the next cycle takes the plain path. */
static void qwen35_session_draft(ds4_session *s, const int *tokens, uint32_t T, uint32_t pos0) {
    int d = -1;
    s->glm_mtp_have = 0;
    if (qwen35_graph_mtp(&s->qwen4_graph, &s->engine->model, &s->engine->weights, tokens, T, pos0, true, &d)) {
        s->glm_mtp_draft = d;
        s->glm_mtp_parent = tokens[T - 1u];
        s->glm_mtp_have = 1;
    }
}

/* One Ornith MTP cycle at depth 1, shaped like ds4_session_qwen4_spec_cycle:
 * evaluate first_token, or verify [first_token, draft] in one 2-row pass
 * whose attention runs per row, so every row equals plain decoding bit for
 * bit.  A draft is accepted when it is the target argmax (greedy and
 * opportunistic sampling: the caller samples first_token and the token
 * after the block from s->logits, so the sampling parameters are unused);
 * a rejected one restores the GDN snapshot the verify took after its first
 * row.  The MTP block then catches up over the committed tokens and drafts
 * from the argmax parent the caller will most likely pass back.  Exact
 * sampling is refused at open.  Returns the committed count (1 or 2) with
 * s->logits at the last one, or -1. */
static int ds4_session_qwen35_spec_cycle(ds4_session *s, int first_token, float temperature, int top_k,
                                         float top_p, float min_p, uint64_t *rng, bool exact_sampling,
                                         int *accepted, int accepted_cap, char *err, size_t errlen) {
    (void)temperature; (void)top_k; (void)top_p; (void)min_p; (void)rng; (void)exact_sampling;
    ds4_engine *e = s->engine;
    ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
    const ds4_model *m = &e->model;
    const ds4_weights *w = &e->weights;
    const uint32_t pos = g->pos, V = DS4_N_VOCAB;
    if (s->glm_mtp_have && first_token != s->glm_mtp_parent) s->glm_mtp_have = 0;
    if (!s->glm_mtp_have || accepted_cap < 2 || pos + 2u > g->ctx_cap || g->cap_tokens < 2u ||
        !qwen4_graph_fused(g, 2u)) {
        s->glm_spec_inside = 1;
        const int rc = ds4_session_eval_internal(s, first_token, false, err, errlen);
        s->glm_spec_inside = 0;
        if (rc != 0) return -1;
        /* rewrite MTP row pos for first_token, then draft after the parent */
        const int toks[2] = { first_token, sample_argmax(s->logits, V) };
        qwen35_session_draft(s, toks, 2u, pos);
        if (qwen35_spec_trace()) fprintf(stderr, "ds4: Ornith spec pos %u token %d plain\n", pos, first_token);
        accepted[0] = first_token;
        return 1;
    }
    s->glm_mtp_have = 0;
    const int d = s->glm_mtp_draft;
    const int toks[2] = { first_token, d };
    /* rows 0 and 1 hold the verify rows (4 rows, as Qwen3.8 sizes it) */
    if (!s->qwen4_verify_logits) s->qwen4_verify_logits = xmalloc(4u * (size_t)V * sizeof(float));
    float *rows = s->qwen4_verify_logits;
    g->snap_after_first = true;
    g->verify_rows_exact = true;
    const bool ok = qwen35_graph_forward_tokens(g, m, w, toks, 2u, rows, true);
    g->verify_rows_exact = false;
    if (!ok) {
        if (errlen) snprintf(err, errlen, "Ornith mtp: verify failed");
        s->checkpoint_valid = false;
        return -1;
    }
    s->qwen4_spec_cycles++;
    token_vec_push(&s->checkpoint, first_token);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    const bool accept = sample_argmax(rows, V) == d || qwen35_spec_force_accept();
    if (qwen35_spec_trace()) {
        fprintf(stderr, "ds4: Ornith spec pos %u token %d draft %d %s\n", pos, first_token, d,
                accept ? "accept" : "reject");
    }
    if (accept) {
        token_vec_push(&s->checkpoint, d);
        memcpy(s->logits, rows + V, (size_t)V * sizeof(float));
        /* rows pos+1 (d with h_pos) and pos+2 (the parent with h_{pos+1}) */
        const int next[2] = { d, sample_argmax(s->logits, V) };
        qwen35_session_draft(s, next, 2u, pos + 1u);
        s->qwen4_spec_accepted++;
        accepted[0] = first_token;
        accepted[1] = d;
        return 2;
    }
    if (!qwen35_graph_state_swap(g)) {
        if (errlen) snprintf(err, errlen, "Ornith mtp: rejection restore failed");
        s->checkpoint_valid = false;
        return -1;
    }
    memcpy(s->logits, rows, (size_t)V * sizeof(float));
    const int parent = sample_argmax(s->logits, V);
    qwen35_session_draft(s, &parent, 1u, pos + 1u);
    accepted[0] = first_token;
    return 1;
}
#endif

```

The accept branch rewrites MTP row `pos + 1`, which the previous draft pass already wrote with the same inputs; one extra row keeps the MTP pass the same for every entry.

- [ ] **Step 7: Dispatch from the speculative entry points**

In `ds4_session_eval_speculative_argmax_impl`, replace

```c
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
    if (ds4_session_is_glm(s)) {
```

with

```c
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
#ifdef DS4_HAS_QWEN4_METAL
    if (ds4_session_is_qwen35(s)) {
        if (!accepted || accepted_cap <= 0) return 0;
        if (s->engine->glm_mtp && s->qwen35_graph_ready && s->qwen4_graph.mtp_h) {
            return ds4_session_qwen35_spec_cycle(s, first_token, 0.0f, 0, 0.0f, 0.0f, NULL, false,
                                                 accepted, accepted_cap, err, errlen);
        }
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
#endif
    ds4_qwen35_not_reached("speculative decode");
    if (ds4_session_is_glm(s)) {
```

(The first four replaced lines are the end of the qwen4 branch; the 5-line block occurs once in `ds4.c`.)

In `ds4_session_eval_speculative`, replace

```c
        return ds4_session_eval_speculative_argmax(
            s, first_token, max_tokens, eos_token,
            accepted, accepted_cap, err, errlen);
    }
#ifdef DS4_NO_GPU
```

with

```c
        return ds4_session_eval_speculative_argmax(
            s, first_token, max_tokens, eos_token,
            accepted, accepted_cap, err, errlen);
    }
#ifdef DS4_HAS_QWEN4_METAL
    if (ds4_session_is_qwen35(s)) {
        if (s->engine->glm_mtp && s->qwen35_graph_ready && s->qwen4_graph.mtp_h) {
            return ds4_session_qwen35_spec_cycle(s, first_token, temperature, top_k, top_p, min_p, rng, false,
                                                 accepted, accepted_cap, err, errlen);
        }
        return ds4_session_eval_speculative_argmax(
            s, first_token, max_tokens, eos_token,
            accepted, accepted_cap, err, errlen);
    }
#endif
    ds4_qwen35_not_reached("sampled speculative decode");
#ifdef DS4_NO_GPU
```

`ds4_session_is_qwen35` is `DS4_MAYBE_UNUSED` and Metal-only builds define both guards; a CUDA build compiles neither branch and keeps its behaviour.

- [ ] **Step 8: Run the tests and confirm they pass (GPU window)**

Run:
```bash
make ds4 ds4-server ds4-agent tests/test_qwen35_mtp tests/test_qwen35_graph
make test-qwen35-mtp test-qwen35-graph test-qwen35-session
tests/ornith/test_loader.sh
```
Expected: no build warnings; `qwen35 mtp: ok` with the four summary lines (greedy cycles and accepts printed), `qwen35 graph: ok`, `qwen35 session: ok`, `ornith loader: ok`.

- [ ] **Step 9: Qwen3.8 checks (GPU window)**

Run: `make test-qwen4-kernels test-qwen4-q2 && speed-bench/qwen-regression/run.sh fast`
Expected: all PASS, `qwen_gate: PASS` (shared functions changed: the draft-token count, the open checks, the memory estimate's Ornith block and the two speculative entry points; their Qwen3.8 branches come first and are untouched).

- [ ] **Step 10: Commit**

```bash
git add ds4.c tests/test_qwen35_mtp.c tests/ornith/test_loader.sh Makefile
git commit -m "ds4: Ornith speculative decoding with the embedded MTP head

--mtp now opens Ornith with its blk.40 head (external MTP models and
exact speculative sampling stay refused).  Sessions allocate the MTP
graph, sync and plain eval write the MTP block's K/V rows, and
ds4_session_qwen35_spec_cycle drafts one token, verifies [token, draft]
in a 2-row pass and restores the GDN snapshot on a reject.  Greedy
output equals plain decoding bit for bit.  Knobs:
DS4_QWEN35_SPEC_TRACE, DS4_QWEN35_SPEC_FORCE_ACCEPT; --mtp-timing
prints the acceptance.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2"
```

---

### Task 4: CLI byte-identity, acceptance against llama.cpp, receipts and spec

Gate 1 item 3 through the real CLI, the acceptance record against llama.cpp's draft-mtp (the check that the catch-up and the draft head are right, since greedy output cannot show a wrong draft), decode speed receipts, the spec wording, and the full Qwen gate before merge.

**Files:**
- Create: `tests/ornith/test_mtp_cli.py`
- Create: `tests/ornith/mtp_accept.py`
- Create: `speed-bench/ornith/m2/RESULTS.md`, `speed-bench/ornith/m2/cli-mtp.txt`, `speed-bench/ornith/m2/accept.md`, `speed-bench/ornith/m2/QWEN_GATE.md`
- Modify: `docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md`

**Interfaces:**
- Consumes: `./ds4` with `--mtp`, `--mtp-timing`, `DS4_QWEN35_SPEC_TRACE`, `DS4_CLI_FORCE_SESSION`; `tests/ornith/ornith_ref.py` (`load_prompts`, `prompt_text`); `tests/ornith/ref/<name>.json` (`prompt_ids`); `llama-server` 0.5.0 on `PATH`.
- Produces: receipts and the spec text; nothing later tasks call.

- [ ] **Step 1: Write the CLI byte-identity test**

Create `tests/ornith/test_mtp_cli.py`:

```python
"""Ornith --mtp must not change greedy output (spec section 8, gate 1 item 3).

For each (prompt, prefill chunk) the one-shot plain generator, the session
path without MTP (DS4_CLI_FORCE_SESSION=1) and the session path with --mtp
must print byte-identical text.  DS4_QWEN35_SPEC_TRACE counts the verify
outcomes: the whole run must see accepted and rejected drafts, and every
long_copy run (9,371 prompt tokens, past the 2048-key attention split
change) must verify at least once.

Usage: python3 tests/ornith/test_mtp_cli.py OUT_DIR
Needs DS4_ORNITH_MODEL and a built ./ds4.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ornith_ref as r  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHORT = ["en_capital", "code_py", "vi_hanoi", "en_contributing", "code_c_pack"]
RUNS = [(p, c) for p in SHORT for c in (1, 2, 64, 0)] + [("long_copy", 0), ("long_copy", 64)]
N_PREDICT = 48
MODES = (
    ("plain", [], {}),
    ("session", [], {"DS4_CLI_FORCE_SESSION": "1"}),
    ("mtp", ["--mtp"], {"DS4_QWEN35_SPEC_TRACE": "1"}),
)


def main(argv):
    if len(argv) != 1:
        sys.exit(__doc__)
    out = argv[0]
    os.makedirs(out, exist_ok=True)
    model = os.environ.get("DS4_ORNITH_MODEL")
    if not model:
        sys.exit("set DS4_ORNITH_MODEL to the 23G ICE GGUF")
    prompts = {p["name"]: p for p in r.load_prompts(os.path.join(ROOT, "tests/ornith/prompts.json"))}
    accepts = rejects = failed = 0
    for name, chunk in RUNS:
        txt = os.path.join(out, name + ".txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write(r.prompt_text(prompts[name], ROOT))
        base = [os.path.join(ROOT, "ds4"), "-m", model, "--metal", "--raw", "--prompt-file", txt,
                "-c", "16384", "--temp", "0", "-n", str(N_PREDICT)]
        if chunk:
            base += ["--prefill-chunk", str(chunk)]
        outs = {}
        for mode, extra, env_extra in MODES:
            env = dict(os.environ)
            env.pop("DS4_QWEN35_SPEC_FORCE_ACCEPT", None)
            env.pop("DS4_CLI_FORCE_SESSION", None)
            env.update(env_extra)
            res = subprocess.run(base + extra, cwd=ROOT, env=env, capture_output=True, timeout=1800)
            tag = f"{name}-c{chunk}-{mode}"
            with open(os.path.join(out, tag + ".stdout"), "wb") as f:
                f.write(res.stdout)
            with open(os.path.join(out, tag + ".stderr"), "wb") as f:
                f.write(res.stderr)
            if res.returncode != 0 or b"failed" in res.stderr.lower():
                print(f"{tag}: FAIL exit {res.returncode} (see {tag}.stderr)")
                failed += 1
                continue
            outs[mode] = res.stdout
            if mode == "mtp":
                a = res.stderr.count(b" accept\n")
                j = res.stderr.count(b" reject\n")
                accepts += a
                rejects += j
                if name == "long_copy" and a + j == 0:
                    print(f"{tag}: FAIL no verify cycle")
                    failed += 1
        same = len(outs) == 3 and outs["plain"] == outs["session"] == outs["mtp"]
        print(f"{name} chunk {chunk or 'default'}: {'ok' if same else 'FAIL output differs'}")
        failed += 0 if same else 1
    print(f"mtp cli: {accepts} accepted, {rejects} rejected drafts")
    if accepts == 0 or rejects == 0:
        print("FAIL: both verify outcomes must occur")
        failed += 1
    print(f"test_mtp_cli: {'PASS' if failed == 0 else f'FAIL ({failed})'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 2: Run it (GPU window, about 30 minutes)**

Run:
```bash
export DS4_ORNITH_MODEL=$HOME/orca/workspaces/ds4-metal-data/gguf/ornith/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf
S=$(mktemp -d); caffeinate -i -s python3 tests/ornith/test_mtp_cli.py "$S/cli" | tee speed-bench/ornith/m2/cli-mtp.txt
```
Expected: 22 `ok` lines, `mtp cli: N accepted, M rejected drafts` with N, M > 0, and `test_mtp_cli: PASS`. A difference means a verify row is not exact: rerun `make test-qwen35-graph` and debug with superpowers:systematic-debugging before continuing.

- [ ] **Step 3: Write the acceptance script**

Create `tests/ornith/mtp_accept.py`:

```python
"""MTP acceptance of ds4 against llama.cpp's draft-mtp on the M1 prompts.

Greedy output cannot show a wrong draft, only a lower acceptance, so this
is the check on the MTP catch-up and draft head.  Both sides draft one
token per cycle at temperature 0 from the same prompt token ids.  One model
process at a time: every ds4 run first, then one fresh llama-server per
prompt (llama.cpp carries its MTP seed across requests in a slot).

Usage: python3 tests/ornith/mtp_accept.py OUT_DIR [--n 128]
Needs DS4_ORNITH_MODEL, ./ds4 and llama-server (0.5.0) on PATH.
Writes OUT_DIR/accept.json and OUT_DIR/accept.md; exits 1 when ds4's
overall acceptance is more than 5 points below llama.cpp's.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ornith_ref as r  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PORT = 18190
DS4_RE = re.compile(r"Ornith mtp: (\d+) verify cycles, (\d+) drafts accepted")


def run_ds4(model, prompt, n, out):
    txt = os.path.join(out, prompt["name"] + ".txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(r.prompt_text(prompt, ROOT))
    cmd = [os.path.join(ROOT, "ds4"), "-m", model, "--metal", "--raw", "--prompt-file", txt,
           "-c", "16384", "--temp", "0", "-n", str(n), "--mtp", "--mtp-timing"]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    with open(os.path.join(out, prompt["name"] + ".ds4.stderr"), "w") as f:
        f.write(res.stderr)
    m = DS4_RE.search(res.stderr)
    if res.returncode != 0 or not m:
        sys.exit(f"{prompt['name']}: ds4 failed or printed no acceptance (see {out})")
    return int(m.group(1)), int(m.group(2))


def http_json(url, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def run_llama(model, name, ids, n, out):
    log = open(os.path.join(out, name + ".llama.log"), "w")
    srv = subprocess.Popen(
        ["llama-server", "-m", model, "--spec-type", "draft-mtp", "--spec-draft-n-max", "1",
         "--spec-draft-p-min", "0", "-np", "1", "-c", "16384", "-b", "2048", "-ub", "512",
         "-ngl", "all", "-fa", "on", "--temp", "0", "--top-k", "1",
         "--host", "127.0.0.1", "--port", str(PORT)],
        stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 600
        while True:
            if srv.poll() is not None:
                sys.exit(f"{name}: llama-server exited (see {log.name})")
            try:
                if http_json(f"http://127.0.0.1:{PORT}/health", timeout=5).get("status") == "ok":
                    break
            except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
                pass
            if time.time() > deadline:
                sys.exit(f"{name}: llama-server did not become ready")
            time.sleep(2)
        body = {"prompt": ids, "n_predict": n, "temperature": 0, "top_k": 1, "cache_prompt": False}
        t = http_json(f"http://127.0.0.1:{PORT}/completion", body)["timings"]
        return int(t.get("draft_n", 0)), int(t.get("draft_n_accepted", 0))
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=120)
        except subprocess.TimeoutExpired:
            sys.exit(f"{name}: llama-server did not stop after SIGTERM; do not kill -9, check it by hand")
        log.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = os.environ.get("DS4_ORNITH_MODEL") or sys.exit("set DS4_ORNITH_MODEL")
    prompts = r.load_prompts(os.path.join(ROOT, "tests/ornith/prompts.json"))
    rows = []
    for p in prompts:
        cycles, acc = run_ds4(model, p, args.n, args.out)
        rows.append({"name": p["name"], "ds4_drafts": cycles, "ds4_accepted": acc})
    for row in rows:
        ids = json.load(open(os.path.join(ROOT, "tests/ornith/ref", row["name"] + ".json")))["prompt_ids"]
        drafted, acc = run_llama(model, row["name"], ids, args.n, args.out)
        row["llama_drafts"], row["llama_accepted"] = drafted, acc
    tot = {k: sum(x[k] for x in rows) for k in ("ds4_drafts", "ds4_accepted", "llama_drafts", "llama_accepted")}
    ds4_rate = tot["ds4_accepted"] / max(tot["ds4_drafts"], 1)
    llama_rate = tot["llama_accepted"] / max(tot["llama_drafts"], 1)
    json.dump({"rows": rows, "total": tot, "ds4_rate": ds4_rate, "llama_rate": llama_rate},
              open(os.path.join(args.out, "accept.json"), "w"), indent=1)
    lines = ["| prompt | ds4 accepted/drafts | ds4 rate | llama.cpp accepted/drafts | llama.cpp rate |",
             "|---|---|---|---|---|"]
    for x in rows:
        dr = x["ds4_accepted"] / max(x["ds4_drafts"], 1)
        lr = x["llama_accepted"] / max(x["llama_drafts"], 1)
        lines.append(f"| {x['name']} | {x['ds4_accepted']}/{x['ds4_drafts']} | {dr:.3f} | "
                     f"{x['llama_accepted']}/{x['llama_drafts']} | {lr:.3f} |")
    lines.append(f"| **total** | {tot['ds4_accepted']}/{tot['ds4_drafts']} | **{ds4_rate:.3f}** | "
                 f"{tot['llama_accepted']}/{tot['llama_drafts']} | **{llama_rate:.3f}** |")
    open(os.path.join(args.out, "accept.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    ok = ds4_rate >= llama_rate - 0.05
    print(f"mtp_accept: {'PASS' if ok else 'FAIL'} ds4 {ds4_rate:.3f} vs llama.cpp {llama_rate:.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Measure acceptance (GPU window, about 20 minutes)**

Run:
```bash
S=$(mktemp -d); caffeinate -i -s python3 tests/ornith/mtp_accept.py "$S/acc" --n 128
cp "$S/acc/accept.md" speed-bench/ornith/m2/accept.md
```
Expected: a 14-row table (13 prompts and the total) and `mtp_accept: PASS` (ds4 within 5 points of llama.cpp or above). ds4 counts verify cycles, llama.cpp drafted tokens; at depth 1 both are one draft per cycle. A FAIL points at the catch-up or the draft head (seed row, RoPE position, concat order, norms): debug with superpowers:systematic-debugging, starting from `make test-qwen35-graph` check 4.

- [ ] **Step 5: Speed receipts (GPU window)**

Run:
```bash
for p in en_contributing code_c_pack; do
  f="$S/cli/$p.txt"
  ./ds4 -m "$DS4_ORNITH_MODEL" --metal --raw --prompt-file "$f" -c 16384 --temp 0 -n 256 \
      > /dev/null 2> "$S/$p.plain.err"
  ./ds4 -m "$DS4_ORNITH_MODEL" --metal --raw --prompt-file "$f" -c 16384 --temp 0 -n 256 --mtp --mtp-timing \
      > /dev/null 2> "$S/$p.mtp.err"
  echo "== $p"; grep -h 'generation' "$S/$p.plain.err" "$S/$p.mtp.err"; grep -h 'Ornith mtp:' "$S/$p.mtp.err"
done
```
Expected: plain lines `ds4: Ornith prefill: ... t/s, generation: G t/s` and MTP lines `ds4: prefill: ... t/s, generation: G' t/s` plus the acceptance line. Record the numbers; there is no pass bar here (M4 owns gate 3). If G' < G, write down the acceptance and the per-cycle cost: the draft head reads the full 248,320-row output matrix (0.5 GB) each cycle, which M4's draft-vocabulary lever addresses.

- [ ] **Step 6: Write the receipt**

Create `speed-bench/ornith/m2/RESULTS.md` with this content, filling the bracketed numbers from Steps 2, 4 and 5:

```markdown
# Ornith M2: MTP (2026-09-26)

Branch `feature/ornith-m2`. Model: 23G ICE GGUF. Oracle: llama.cpp 0.5.0 (build 11146), `--spec-type draft-mtp --spec-draft-n-max 1`.

## Correctness (gate 1 item 3)

- `make test-qwen35-graph`: an MTP graph's logits equal the M1 graph's; each row of a 2-row verify equals plain decoding bit for bit at positions 31, 63, 2047, 2111, 3000; a rejected verify restored by the snapshot swap continues bit-identical; the catch-up K rows agree across prefill chunks 512 / 64 / 1.
- `make test-qwen35-mtp`: 150 greedy cycles ([accepts] accepted) commit the plain argmax sequence with bit-identical logits; forced accepts, a divergent prompt and the context end behave.
- `tests/ornith/test_mtp_cli.py` (`cli-mtp.txt`): plain one-shot, session and `--mtp` print identical text for 5 prompts x prefill chunks 1, 2, 64 and default, and for `long_copy` (9,371 tokens) at default and 64; [N] drafts accepted, [M] rejected.
- gate 1 default compare after the graph change: identical to `speed-bench/ornith/m1/compare-default.txt`.

## Acceptance vs llama.cpp (`accept.md`, n = 128 per prompt)

ds4 [rate] vs llama.cpp [rate] overall.

## Decode speed (CLI, 256 tokens, M5 Pro)

| prompt | plain t/s | --mtp t/s | acceptance |
|---|---|---|---|
| en_contributing | [G] | [G'] | [rate] |
| code_c_pack | [G] | [G'] | [rate] |

The draft head reads the whole output matrix each cycle; the MTP draft vocabulary (M4 lever) is the next step for speed.

## Qwen3.8

Full gate: see `QWEN_GATE.md`.
```

- [ ] **Step 7: Update the spec**

Run this script from the repo root (it fails if any old text is missing):

```bash
python3 - <<'EOF'
p = "docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md"
s = open(p, encoding="utf-8").read()
edits = [
("""  1. `x = eh_proj . concat[RMSNorm(embed(tok)) * enorm, RMSNorm(h) * hnorm]`,
     embedding half first.""",
"""  1. `x = eh_proj . concat[RMSNorm(embed(tok)) * enorm, RMSNorm(h) * hnorm]`,
     embedding half first; `h` is the trunk hidden after `output_norm` (the
     tensor the LM head reads, llama.cpp PR #24025)."""),
("""  4. Drafting chains `x` as the next step's `h`.""",
"""  4. A deeper draft chains `RMSNorm(x) * shared_head_norm` (the step-3 head
     input) as the next step's `h`. M2 drafts one token per cycle; deeper
     drafts are an M4 lever."""),
("""  4. The MTP block runs a catch-up pass over the chunk: `h` shifted right by one
     fills the MTP KV cache for every prompt position, as llama.cpp does.""",
"""  4. The MTP block runs a catch-up pass over the chunk: MTP KV row `p` comes
     from `(token_p, h_{p-1})` at RoPE position `p`, with `h_{-1} = 0`, as
     llama.cpp's draft-mtp does. Catch-up rows compute only their K/V; the
     trunk keeps its post-norm rows and carries the last one to the next
     forward. Qwen3.8 has no catch-up, so this is Ornith code."""),
("""- **Decode with MTP.** Each cycle drafts one token with the MTP block, then
  verifies `[current, draft]` in one T=2 pass, snapshotting GDN state and conv
  history after row 1. An accepted draft keeps both rows. A rejected draft
  restores the snapshot and keeps one. This reuses the qwen4 speculative cycle
  and its snapshot/swap machinery. Draft depth starts at 1 and follows measured
  acceptance. At temperature 0 the output equals plain decoding.""",
"""- **Decode with MTP.** Each cycle drafts one token with the MTP block, then
  verifies `[current, draft]` in one T=2 pass, snapshotting GDN state and conv
  history after the first row. An accepted draft keeps both rows. A rejected
  draft restores the snapshot and keeps one. The cycle is
  `ds4_session_qwen35_spec_cycle` in `ds4.c`, shaped like the qwen4 cycle
  (whose snapshot helpers need hyper-connections and PLE); the GDN kernels'
  after-first-row snapshot is reused. The verify runs its attention one row
  per dispatch so each row equals plain decoding bit for bit. Drafts are
  accepted when they are the target argmax (greedy and opportunistic
  sampling); `--mtp-exact-sampling` is refused. Draft depth starts at 1 and
  follows measured acceptance. At temperature 0 the output equals plain
  decoding."""),
("""- `ds4.c` gets one branch per dispatch point (engine open, session
  create/free, forward, speculative cycle, payload save/load, context cap),
  each calling into the `.inc`.""",
"""- `ds4.c` gets one branch per dispatch point (engine open, session
  create/free, forward, speculative cycle, payload save/load, context cap).
  Graph-level code lives in the `.inc`; session-level code (sync/eval
  branches, the speculative cycle) stays in `ds4.c`, because the `.inc` is
  included before `struct ds4_session`."""),
]
for old, new in edits:
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
print("spec updated")
EOF
```

Expected: `spec updated`.

- [ ] **Step 8: Full Qwen3.8 gate and the final Ornith suite (GPU window, about 1 hour)**

Run:
```bash
make test-qwen4-kernels test-qwen4-q2
caffeinate -i -s speed-bench/qwen-regression/run.sh full | tee "$S/qwen-full.txt"
make test-qwen35-kernels test-qwen35-graph test-qwen35-mtp test-qwen35-session
tests/ornith/test_loader.sh && tests/ornith/oneshot.sh
```
Expected: `qwen_gate: PASS` (replies byte-identical, paired decode ≥ 97 % of PROD, wired within baseline + 0.5 GiB, needle HIT) and every Ornith test ok. Write `speed-bench/ornith/m2/QWEN_GATE.md` in the format of `speed-bench/ornith/m1/QWEN_GATE.md`: the commit, the result line (byte-identical, paired decode ratio, wired GiB, needle), and the kernel test PASS lines.

- [ ] **Step 9: Commit**

```bash
git add tests/ornith/test_mtp_cli.py tests/ornith/mtp_accept.py speed-bench/ornith/m2 \
    docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md
git commit -m "tests/ornith: MTP byte-identity and acceptance; M2 receipts

test_mtp_cli.py checks that plain one-shot, session and --mtp decoding
print identical text across prefill chunks 1/2/64/default and a
9,371-token prompt, with both verify outcomes seen.  mtp_accept.py
compares ds4's draft acceptance with llama.cpp's draft-mtp on the M1
prompts (same token ids, one draft per cycle).  The spec now states the
MTP seed (post-output_norm h), the catch-up convention, the own cycle
and the row-exact verify.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2"
```

---

## Self-review notes (planning)

- Spec coverage for M2 (§9: MTP block, prefill catch-up, speculative cycle, gate 1 item 3, acceptance recorded): Task 2 (block, catch-up, verify state), Task 3 (cycle, sessions, CLI/server entry points), Task 4 (gate 1 item 3 via the CLI, acceptance vs llama.cpp as spec §10 asks). Server serving of Ornith is M3; the server's speculative loop already calls the entry points dispatched here.
- Types and names: `qwen35_graph_alloc(g, ctx, cap, mtp)`, `qwen35_graph_forward_tokens(..., all_rows)`, `qwen35_graph_mtp(g, m, w, tokens, T, pos0, want_draft, draft_out)`, `qwen35_graph_state_swap(g)`, `ds4_session_qwen35_spec_cycle(...)` are used with the same signatures in Tasks 2-4 and in the tests.
- Deliberately not in M2: rewind with snapshot restore and the disk payload (M3), deeper drafts, the draft vocabulary and every speed lever (M4).
