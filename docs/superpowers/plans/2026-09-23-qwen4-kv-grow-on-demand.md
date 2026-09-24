# qwen4 Grow-on-Demand KV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allocate the qwen4 context-sized GPU caches for the context in use, grow them on demand and shrink them at a session boundary, behind `DS4_QWEN4_KV_GROW=1`, with greedy output byte-identical to the full-capacity allocation.

**Architecture:** A new `alloc_cap` (rows actually allocated) sits beside the logical `ctx_cap` in `ds4_qwen4_gpu_graph`. The context-sized buffers are built by one helper so a resize can allocate a new set, blit-copy the used prefix, and swap. Session-level entry points (sync, eval, spec cycle, batched eval, payload load) call `qwen4_session_ensure_cap` before any forward; low-level buffer guards and Metal bindings compare against `alloc_cap`. Metal kernels do not change.

**Tech Stack:** C (ds4.c, ~80K lines), Objective-C Metal backend (ds4_metal.m, unchanged), the repo's `ds4_test` runner (tests/ds4_test.c, model-backed via `DS4_TEST_MODEL`), GNU make.

**Spec:** `docs/superpowers/specs/2026-09-23-qwen4-kv-grow-on-demand-design.md`

## Global Constraints

- `DS4_QWEN4_KV_GROW=1` enables the feature for session graphs on Metal. Unset or `0`: `alloc_cap = ctx_cap`, identical to today.
- Initial capacity 32768 rows (`DS4_QWEN4_KV_INIT_CAP` overrides), rounded up to a multiple of 256 rows, never above `ctx_cap`.
- Reserve on sync: `need = prompt_len + reserve`, reserve 4096 (`DS4_QWEN4_KV_RESERVE` overrides, for tests).
- Grow before a forward when `pos + T + 64 > alloc_cap`; target `max(need, 2*alloc_cap)` rounded up to 256, clamped to `ctx_cap`; above `ctx_cap/2` jump to `ctx_cap`.
- Shrink only when `ds4_session_sync` resets the graph and `need <= alloc_cap/4`, to `max(init_cap, need)`.
- Resize only at safe points (before `begin_commands` of an entry point). On failure keep the old buffers and return an error naming the KV capacity. `DS4_QWEN4_KV_GROW_TEST_FAIL=1` makes every resize fail.
- `ctx_cap` keeps its meaning: maximum position, ik-ring choice, checkpoint header fields, startup memory plan.
- CUDA/ROCm, one-shot `generate_qwen4_metal_argmax`, and test graphs keep full allocation. Metal kernels do not change.
- Greedy output with the flag on must be byte-identical to the flag off in the same configuration.
- Code, comments and commit messages in English. Every commit ends with:
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_017f1tsCGFkVQ3wrmPsJ7FBn`.
- Model artifacts never go into git.

## Test Environment

Model-backed tests need the unc48L model and its PLE sidecar. Task 0 creates an untracked file
`kv-grow-test.env` in the worktree root (added to `.git/info/exclude` so it is never committed):

```bash
export DS4_TEST_MODEL=/Users/dongnh/orca/workspaces/ds4-metal-data/gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q4KDownPad768-DenseQ4Kselimat-MTP.gguf
export DS4_TEST_PLE=/Users/dongnh/orca/workspaces/ds4-metal-data/gguf/Qwen3.8-Flash-Next-PLE-Q4_1.gguf
export DS4_TEST_GLM_MTP=1 DS4_TEST_SSD_STREAMING=1 DS4_TEST_SSD_STREAMING_CACHE_GB=6
export DS4_QWEN4_STREAM_FULL_LAYERS=32 DS4_QWEN4_PLE_PREFETCH_FULL=0
```

Every model-backed command below starts with `. ./kv-grow-test.env &&`. Stop any running `ds4`,
`ds4-server` or other large-memory process before a model-backed run.

## Review Focus

1. A session that borrows the engine arena while another session grows (or shrinks) the arena's `score`/`tile_max` must never forward with a stale pointer — Task 4, case "arena refresh".
2. Restoring a KV payload longer than a fresh session's `alloc_cap` must grow first and then continue identically — Task 3, case E.
3. A decode that crosses `alloc_cap - 64` one token at a time (off-by-one at the growth threshold) must stay identical — Task 3, case A (plain) and case C (MTP).
4. A failed resize must leave the session usable: the same session syncs successfully once resizing works again — Task 3, case F.
5. A session shrinking at a reset while another live session is larger must not shrink the shared arena below the larger session — Task 4, case "shrink with a larger neighbour".

## File Structure

- `ds4.c` — all engine changes: policy helpers (plain C, before the `#ifdef DS4_HAS_QWEN4_GPU` graph section), graph fields, context-buffer helper, resize, session ensure/shrink, call sites, test hook.
- `tests/ds4_test.c` — policy unit test (`--qwen-kv-grow-policy`), model-backed test (`--qwen-kv-grow`), two new optional engine env knobs (`DS4_TEST_PLE`, `DS4_TEST_SHARE_WORKSPACE`).
- `speed-bench/stageb-runs/kv-grow/RESULTS-kv-grow.md` — end-to-end verification and performance receipt (Task 5).

Line numbers below refer to `develop` at `5f0bfef`; they drift as you edit, so locate each edit by the quoted anchor text (every anchor is unique in its file — confirm with `grep -c` before editing).

---

### Task 0: Isolated worktree

- [ ] **Step 1:** Use superpowers:using-git-worktrees to create branch `kv-grow` from `develop` (`5f0bfef` or later) in a new worktree, e.g. `~/orca/workspaces/ds4-metal/kv-grow`. All following commands run in that worktree.
- [ ] **Step 2:** Create `kv-grow-test.env` with the contents from "Test Environment" and run
`echo kv-grow-test.env >> "$(git rev-parse --git-common-dir)/info/exclude"`.
- [ ] **Step 3:** Build once to confirm a clean baseline:

Run: `make -j10 ds4 ds4-server ds4_test`
Expected: builds without errors.

---

### Task 1: Capacity policy helpers

**Files:**
- Modify: `ds4.c` (insert immediately before the line `#ifdef DS4_HAS_QWEN4_GPU` that is followed by `/* ------------------------------------------------------------------------` and ` * Qwen3.8-Flash-Next GPU graph.`, ~line 57782)
- Test: `tests/ds4_test.c`

**Interfaces:**
- Produces:
  - `uint32_t ds4_qwen4_kv_initial_cap(uint32_t ctx_cap, const char *env_value);`
  - `uint32_t ds4_qwen4_kv_grow_target(uint32_t alloc, uint32_t need, uint32_t ctx_cap);`
  - `uint32_t ds4_qwen4_kv_shrink_target(uint32_t alloc, uint32_t need, uint32_t init, uint32_t ctx_cap);`
  - `uint32_t ds4_qwen4_kv_reserve(const char *env_value);`
  - macros `DS4_QWEN4_KV_MARGIN` (64u) used by Task 3/4.

- [ ] **Step 1: Write the failing test**

In `tests/ds4_test.c`, add next to the other `ds4_test_*` prototypes at the top of the file (after `bool ds4_test_dspark_prefix_capture(ds4_engine *engine, const ds4_tokens *prompt);`):

```c
uint32_t ds4_qwen4_kv_initial_cap(uint32_t ctx_cap, const char *env_value);
uint32_t ds4_qwen4_kv_grow_target(uint32_t alloc, uint32_t need, uint32_t ctx_cap);
uint32_t ds4_qwen4_kv_shrink_target(uint32_t alloc, uint32_t need, uint32_t init, uint32_t ctx_cap);
uint32_t ds4_qwen4_kv_reserve(const char *env_value);
```

Add this function before `static void test_server_unit_group(void) {`:

```c
/* Grow-on-demand KV capacity policy: plain arithmetic, no model needed. */
static void test_qwen_kv_grow_policy(void) {
    TEST_ASSERT(ds4_qwen4_kv_initial_cap(262144, NULL) == 32768);
    TEST_ASSERT(ds4_qwen4_kv_initial_cap(262144, "") == 32768);
    TEST_ASSERT(ds4_qwen4_kv_initial_cap(262144, "junk") == 32768);
    TEST_ASSERT(ds4_qwen4_kv_initial_cap(262144, "1000") == 1024);
    TEST_ASSERT(ds4_qwen4_kv_initial_cap(8192, NULL) == 8192);
    TEST_ASSERT(ds4_qwen4_kv_initial_cap(512, "256") == 256);

    TEST_ASSERT(ds4_qwen4_kv_grow_target(32768, 32768, 262144) == 32768);
    TEST_ASSERT(ds4_qwen4_kv_grow_target(32768, 32769, 262144) == 65536);
    TEST_ASSERT(ds4_qwen4_kv_grow_target(32768, 100000, 262144) == 100096);
    TEST_ASSERT(ds4_qwen4_kv_grow_target(65536, 70000, 262144) == 131072);
    TEST_ASSERT(ds4_qwen4_kv_grow_target(131072, 131073, 262144) == 262144);
    TEST_ASSERT(ds4_qwen4_kv_grow_target(32768, 140000, 262144) == 262144);
    TEST_ASSERT(ds4_qwen4_kv_grow_target(256, 300, 512) == 512);
    TEST_ASSERT(ds4_qwen4_kv_grow_target(262144, 400000, 262144) == 262144);
    TEST_ASSERT(ds4_qwen4_kv_grow_target(4096, 4097, 65536) == 8192);
    TEST_ASSERT(ds4_qwen4_kv_grow_target(4096, 10064, 65536) == 10240);

    TEST_ASSERT(ds4_qwen4_kv_shrink_target(262144, 5000, 32768, 262144) == 32768);
    TEST_ASSERT(ds4_qwen4_kv_shrink_target(262144, 40000, 32768, 262144) == 40192);
    TEST_ASSERT(ds4_qwen4_kv_shrink_target(262144, 70000, 32768, 262144) == 262144);
    TEST_ASSERT(ds4_qwen4_kv_shrink_target(32768, 100, 32768, 262144) == 32768);
    TEST_ASSERT(ds4_qwen4_kv_shrink_target(10240, 264, 4096, 65536) == 4096);

    TEST_ASSERT(ds4_qwen4_kv_reserve(NULL) == 4096);
    TEST_ASSERT(ds4_qwen4_kv_reserve("64") == 64);
    TEST_ASSERT(ds4_qwen4_kv_reserve("0") == 4096);
}
```

Register it in `test_entries[]` directly under `static const ds4_test_entry test_entries[] = {` (before `#ifndef DS4_NO_GPU`, because it needs no GPU):

```c
    {"--qwen-kv-grow-policy", "qwen-kv-grow-policy", "Qwen3.8 grow-on-demand KV capacity policy (no model)", test_qwen_kv_grow_policy},
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make ds4_test 2>&1 | tail -5`
Expected: link error `Undefined symbols ... _ds4_qwen4_kv_initial_cap` (and the other three).

- [ ] **Step 3: Write minimal implementation**

In `ds4.c`, immediately before `#ifdef DS4_HAS_QWEN4_GPU` (the one introducing the Qwen3.8-Flash-Next GPU graph section), insert:

```c
/* Grow-on-demand KV (DS4_QWEN4_KV_GROW=1).  The qwen4 caches sized by the
 * context (KV, scales, block keys, positions, indexer scores) are allocated
 * for alloc_cap rows instead of the full -c capacity, grow before a forward
 * that needs more and shrink when a session starts over.  These helpers are
 * the policy only: plain arithmetic, so the tests call them without a model. */
#define DS4_QWEN4_KV_GRAIN 256u
#define DS4_QWEN4_KV_DEFAULT_INIT 32768u
#define DS4_QWEN4_KV_DEFAULT_RESERVE 4096u
#define DS4_QWEN4_KV_MARGIN 64u  /* widest verify is 17 rows; 64 leaves room */

static uint64_t qwen4_kv_round(uint64_t rows) {
    return (rows + DS4_QWEN4_KV_GRAIN - 1u) / DS4_QWEN4_KV_GRAIN * DS4_QWEN4_KV_GRAIN;
}

static uint32_t qwen4_kv_env_u32(const char *env_value, uint32_t fallback) {
    if (!env_value || !env_value[0]) return fallback;
    const long v = strtol(env_value, NULL, 10);
    if (v <= 0) return fallback;
    return v > (long)UINT32_MAX ? UINT32_MAX : (uint32_t)v;
}

uint32_t ds4_qwen4_kv_initial_cap(uint32_t ctx_cap, const char *env_value) {
    const uint64_t init = qwen4_kv_round(qwen4_kv_env_u32(env_value, DS4_QWEN4_KV_DEFAULT_INIT));
    return init < ctx_cap ? (uint32_t)init : ctx_cap;
}

uint32_t ds4_qwen4_kv_reserve(const char *env_value) {
    return qwen4_kv_env_u32(env_value, DS4_QWEN4_KV_DEFAULT_RESERVE);
}

/* Rows to allocate for a graph holding alloc rows that needs need rows: double
 * (or jump to need), and past half of -c go straight to -c so the largest
 * copy happens at most once. */
uint32_t ds4_qwen4_kv_grow_target(uint32_t alloc, uint32_t need, uint32_t ctx_cap) {
    if (need > ctx_cap) need = ctx_cap;
    if (need <= alloc) return alloc;
    uint64_t target = (uint64_t)alloc * 2u;
    if (target < need) target = need;
    target = qwen4_kv_round(target);
    if (target > ctx_cap / 2u) target = ctx_cap;
    return (uint32_t)target;
}

/* Rows to keep when a session starts over and needs need rows: shrink only
 * when need fits in a quarter of what is allocated, never below init. */
uint32_t ds4_qwen4_kv_shrink_target(uint32_t alloc, uint32_t need, uint32_t init, uint32_t ctx_cap) {
    if (need > alloc / 4u) return alloc;
    uint64_t target = qwen4_kv_round(need);
    if (target < init) target = init;
    if (target > ctx_cap) target = ctx_cap;
    return target < alloc ? (uint32_t)target : alloc;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `make ds4_test && ./ds4_test --qwen-kv-grow-policy`
Expected: exit code 0, no `FAIL` lines.

- [ ] **Step 5: Commit**

```bash
git add ds4.c tests/ds4_test.c
git commit -m "qwen4: grow-on-demand KV capacity policy helpers

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017f1tsCGFkVQ3wrmPsJ7FBn"
```

---

### Task 2: Allocate the context caches at `alloc_cap`

The graph gains `alloc_cap`; the context-sized buffers are built by one helper; every buffer guard and Metal binding uses `alloc_cap`. Nothing grows yet, so with the flag on a prompt that does not fit the initial capacity must fail cleanly.

**Files:**
- Modify: `ds4.c` — `ds4_qwen4_gpu_graph` struct (~57814), new helpers before `qwen4_graph_alloc` (~58141), `qwen4_graph_alloc` (~58149-58346), its four callers (~59954, ~68533, ~68671, ~73794), session create (~73761-73798), guards (~59064, ~59424, ~59768, ~59862, ~79548, ~79580, ~79878, ~79921), new test hook after `int ds4_session_prefill_cap(ds4_session *s) {...}` (~86272)
- Modify: `tests/ds4_test.c` — `test_open_engine`, new `--qwen-kv-grow` test

**Interfaces:**
- Consumes: `ds4_qwen4_kv_initial_cap`, `ds4_qwen4_kv_reserve` (Task 1).
- Produces:
  - struct fields `uint32_t alloc_cap; bool kv_grow; uint32_t kv_init_cap; uint32_t kv_reserve;`
  - `typedef struct qwen4_ctx_bufs qwen4_ctx_bufs;` with `qwen4_ctx_bufs_alloc(const ds4_qwen4_gpu_graph *g, uint32_t rows, qwen4_ctx_bufs *b)`, `qwen4_ctx_bufs_free(qwen4_ctx_bufs *b)`, `qwen4_ctx_bufs_swap(ds4_qwen4_gpu_graph *g, qwen4_ctx_bufs *b)`, `qwen4_ctx_bufs_bytes(const qwen4_ctx_bufs *b)`
  - `qwen4_graph_alloc(..., uint32_t slot, uint32_t alloc_rows)` — `alloc_rows == 0` means full `ctx_cap`
  - test hook `int ds4_test_qwen4_alloc_cap(ds4_session *s);` (returns `alloc_cap`, or -1)

- [ ] **Step 1: Write the failing test**

In `tests/ds4_test.c`:

(a) In `test_open_engine`, add two fields to the `ds4_engine_options opt = {` initializer, after `.dspark_exact_sampling = test_env_bool("DS4_TEST_MTP_EXACT"),`:

```c
        .ple_path = getenv("DS4_TEST_PLE"),
        .share_session_prefill_workspace = test_env_bool("DS4_TEST_SHARE_WORKSPACE"),
```

(b) Add the prototype next to the Task 1 prototypes:

```c
int ds4_test_qwen4_alloc_cap(ds4_session *s);
```

(c) Add these helpers and the test before `static void test_server_unit_group(void) {`:

```c
/* Grow-on-demand KV, model-backed.  Needs a Qwen3.8 model and its PLE:
 *   DS4_TEST_MODEL=...gguf DS4_TEST_PLE=...PLE-Q4_1.gguf DS4_TEST_GLM_MTP=1 \
 *   DS4_TEST_SSD_STREAMING=1 DS4_TEST_SSD_STREAMING_CACHE_GB=6 \
 *   DS4_QWEN4_STREAM_FULL_LAYERS=32 DS4_QWEN4_PLE_PREFETCH_FULL=0 \
 *   ./ds4_test --qwen-kv-grow
 * Every case compares greedy tokens with the flag on against the flag off. */
#define TEST_KV_CTX 65536

static void test_kv_grow_env(bool grow, uint32_t init, uint32_t reserve) {
    char v[32];
    if (grow) setenv("DS4_QWEN4_KV_GROW", "1", 1); else unsetenv("DS4_QWEN4_KV_GROW");
    snprintf(v, sizeof(v), "%u", init);
    setenv("DS4_QWEN4_KV_INIT_CAP", v, 1);
    snprintf(v, sizeof(v), "%u", reserve);
    setenv("DS4_QWEN4_KV_RESERVE", v, 1);
}

/* n raw (untemplated) tokens of varied text: numbers change on every line so
 * greedy decode keeps reading the KV instead of settling into a loop. */
static void test_kv_grow_prompt(ds4_engine *engine, int n, int seed, ds4_tokens *out) {
    buf text = {0};
    ds4_tokens all = {0};
    for (int i = 0; all.len < n; ) {
        for (int j = 0; j < 256; j++, i++) {
            char line[192];
            snprintf(line, sizeof(line),
                     "Log %d.%d: the harbor recorded %d ships, wind %d knots, tide %d cm.\n",
                     seed, i, (i * 37 + seed) % 91, (i * 13 + seed) % 43, (i * 29 + seed) % 311);
            buf_puts(&text, line);
        }
        ds4_tokens_free(&all);
        ds4_tokenize_text(engine, text.ptr, &all);
    }
    ds4_tokens_free(out);
    for (int i = 0; i < n; i++) ds4_tokens_push(out, all.v[i]);
    ds4_tokens_free(&all);
    buf_free(&text);
}

/* Greedy-decode n tokens after prompt into out (plain eval, or MTP speculation
 * when spec).  Unused slots are -1.  Returns the session's allocated KV rows
 * at the end, or -1 when a step failed. */
static int test_kv_grow_decode(ds4_engine *engine, ds4_session *s, int n, bool spec, int *out) {
    char err[192] = {0};
    const int eos = ds4_token_eos(engine);
    int got = 0;
    for (int j = 0; j < n; j++) out[j] = -1;
    while (got < n) {
        const int token = ds4_session_argmax(s);
        if (!spec) {
            out[got++] = token;
            if (ds4_session_eval(s, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-test: kv-grow eval failed: %s\n", err);
                return -1;
            }
            continue;
        }
        int toks[17];
        const int ntok = ds4_session_eval_speculative_argmax(s, token, n - got, eos, toks,
                                                             (int)(sizeof(toks) / sizeof(toks[0])),
                                                             err, sizeof(err));
        if (ntok <= 0) {
            fprintf(stderr, "ds4-test: kv-grow speculative eval failed: %s\n", err);
            return -1;
        }
        for (int j = 0; j < ntok && got < n; j++) out[got++] = toks[j];
        if (toks[ntok - 1] == eos) break;
    }
    return ds4_test_qwen4_alloc_cap(s);
}

/* Fresh session, sync prompt, decode n tokens.  Returns alloc rows or -1. */
static int test_kv_grow_run(ds4_engine *engine, const ds4_tokens *prompt, int n, bool spec, int *out) {
    ds4_session *s = NULL;
    char err[192] = {0};
    int alloc = -1;
    if (ds4_session_create(&s, engine, TEST_KV_CTX) != 0 || !s) return -1;
    if (ds4_session_sync(s, prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-test: kv-grow sync failed: %s\n", err);
    } else {
        alloc = test_kv_grow_decode(engine, s, n, spec, out);
    }
    ds4_session_free(s);
    return alloc;
}

static void test_qwen_kv_grow(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine || !ds4_engine_is_qwen4(engine)) {
        puts("qwen-kv-grow: Qwen3.8 model required, skipped");
        return;
    }
    char *saved[4] = {
        test_save_env("DS4_QWEN4_KV_GROW"), test_save_env("DS4_QWEN4_KV_INIT_CAP"),
        test_save_env("DS4_QWEN4_KV_RESERVE"), test_save_env("DS4_QWEN4_KV_GROW_TEST_FAIL"),
    };
    ds4_tokens prompt = {0};
    static int ref[640], got[640];

    /* The flag off keeps the full capacity; on starts at the initial one and
     * decodes the same tokens while everything fits. */
    test_kv_grow_prompt(engine, 1000, 1, &prompt);
    test_kv_grow_env(false, 4096, 64);
    TEST_ASSERT(test_kv_grow_run(engine, &prompt, 32, false, ref) == TEST_KV_CTX);
    test_kv_grow_env(true, 4096, 64);
    TEST_ASSERT(test_kv_grow_run(engine, &prompt, 32, false, got) == 4096);
    TEST_ASSERT(memcmp(ref, got, 32 * sizeof(int)) == 0);

    /* Without growth a prompt past the initial capacity fails cleanly. */
    test_kv_grow_prompt(engine, 5000, 2, &prompt);
    TEST_ASSERT(test_kv_grow_run(engine, &prompt, 1, false, got) == -1);

    ds4_tokens_free(&prompt);
    test_restore_env("DS4_QWEN4_KV_GROW", saved[0]);
    test_restore_env("DS4_QWEN4_KV_INIT_CAP", saved[1]);
    test_restore_env("DS4_QWEN4_KV_RESERVE", saved[2]);
    test_restore_env("DS4_QWEN4_KV_GROW_TEST_FAIL", saved[3]);
}
```

(d) Register it inside the `#ifndef DS4_NO_GPU` block of `test_entries[]`, next to `--qwen4-restore-reuse`:

```c
    {"--qwen-kv-grow", "qwen-kv-grow", "Qwen3.8 grow-on-demand KV decodes byte-identically to full capacity", test_qwen_kv_grow},
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make ds4_test 2>&1 | tail -3`
Expected: link error `Undefined symbols ... _ds4_test_qwen4_alloc_cap`.

- [ ] **Step 3: Add the graph fields**

In `ds4.c`, in `typedef struct ds4_qwen4_gpu_graph {`, replace

```c
typedef struct ds4_qwen4_gpu_graph {
    uint32_t ctx_cap;
```

with

```c
typedef struct ds4_qwen4_gpu_graph {
    uint32_t ctx_cap;
    /* Rows allocated for the context-sized caches (KV, scales, block keys,
     * positions, indexer scores).  Equals ctx_cap unless kv_grow; the Metal
     * bindings and every buffer guard use it, while ctx_cap stays the logical
     * limit (maximum position, ik ring, checkpoint header). */
    uint32_t alloc_cap;
    bool kv_grow;          /* DS4_QWEN4_KV_GROW: alloc_cap follows the context in use */
    uint32_t kv_init_cap;  /* alloc_cap floor when kv_grow */
    uint32_t kv_reserve;   /* rows reserved past the prompt on sync */
```

- [ ] **Step 4: Add the context-buffer helpers**

In `ds4.c`, immediately before the comment `/* shared borrows the engine arena for the transients; NULL allocates private` (just above `qwen4_graph_alloc`), insert:

```c
/* The context-sized caches of one graph, allocated together so a resize can
 * build the new set before it lets go of the old one.  The ring-sized raw
 * indexer cache (ik_ring != 0) does not depend on the context and is not
 * part of the set. */
typedef struct qwen4_ctx_bufs {
    ds4_gpu_tensor *k[DS4_MAX_LAYER], *v[DS4_MAX_LAYER];    /* half cache */
    ds4_gpu_tensor *kf[DS4_MAX_LAYER], *vf[DS4_MAX_LAYER];  /* FP8 / 4-bit cache */
    ds4_gpu_tensor *ks[DS4_MAX_LAYER], *vs[DS4_MAX_LAYER];  /* per-64 scales */
    ds4_gpu_tensor *ik[DS4_MAX_LAYER];                      /* only when the ring is off */
    ds4_gpu_tensor *bk[DS4_MAX_LAYER];                      /* pooled block keys */
    ds4_gpu_tensor *pos3;
} qwen4_ctx_bufs;

static void qwen4_ctx_bufs_free(qwen4_ctx_bufs *b) {
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor_free(b->k[il]);
        ds4_gpu_tensor_free(b->v[il]);
        ds4_gpu_tensor_free(b->kf[il]);
        ds4_gpu_tensor_free(b->vf[il]);
        ds4_gpu_tensor_free(b->ks[il]);
        ds4_gpu_tensor_free(b->vs[il]);
        ds4_gpu_tensor_free(b->ik[il]);
        ds4_gpu_tensor_free(b->bk[il]);
    }
    ds4_gpu_tensor_free(b->pos3);
    memset(b, 0, sizeof(*b));
}

static uint64_t qwen4_ctx_bufs_bytes(const qwen4_ctx_bufs *b) {
    uint64_t total = ds4_gpu_tensor_bytes(b->pos3);
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        total += ds4_gpu_tensor_bytes(b->k[il]) + ds4_gpu_tensor_bytes(b->v[il]) +
                 ds4_gpu_tensor_bytes(b->kf[il]) + ds4_gpu_tensor_bytes(b->vf[il]) +
                 ds4_gpu_tensor_bytes(b->ks[il]) + ds4_gpu_tensor_bytes(b->vs[il]) +
                 ds4_gpu_tensor_bytes(b->ik[il]) + ds4_gpu_tensor_bytes(b->bk[il]);
    }
    return total;
}

/* Allocate the context-sized caches for rows positions in g's KV format.
 * Sizes match what qwen4_graph_alloc always allocated for ctx_cap rows. */
static bool qwen4_ctx_bufs_alloc(const ds4_qwen4_gpu_graph *g, uint32_t rows, qwen4_ctx_bufs *b) {
    memset(b, 0, sizeof(*b));
    const uint64_t kv_dim = (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    const uint64_t blocks = (uint64_t)rows / 4u + 1u;
    bool ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        if (ds4_qwen4_layer_is_linear(il)) continue;
        if (g->kv_fp8) {
            const uint64_t kv_bytes = g->kv_q4 ? (uint64_t)rows * kv_dim / 2u : (uint64_t)rows * kv_dim;
            b->kf[il] = ds4_gpu_tensor_alloc(kv_bytes);
            b->vf[il] = ds4_gpu_tensor_alloc(kv_bytes);
            b->ks[il] = ds4_gpu_tensor_alloc((uint64_t)rows * (kv_dim / 64u) * 2u);
            b->vs[il] = ds4_gpu_tensor_alloc((uint64_t)rows * (kv_dim / 64u) * 2u);
            ok = b->kf[il] && b->vf[il] && b->ks[il] && b->vs[il];
        } else {
            b->k[il] = ds4_gpu_tensor_alloc((uint64_t)rows * kv_dim * 2u);
            b->v[il] = ds4_gpu_tensor_alloc((uint64_t)rows * kv_dim * 2u);
            ok = b->k[il] && b->v[il];
        }
        if (ok && !g->ik_ring) {
            b->ik[il] = qwen4_graph_alloc_f32((uint64_t)rows * DS4_N_INDEXER_HEAD_DIM);
            ok = b->ik[il] != NULL;
        }
        if (ok) {
            b->bk[il] = ds4_gpu_tensor_alloc(blocks * DS4_N_INDEXER_HEAD_DIM * 2u);
            ok = b->bk[il] != NULL;
        }
    }
    if (ok) {
        b->pos3 = qwen4_graph_alloc_f32((uint64_t)rows * 4u);
        ok = b->pos3 != NULL;
    }
    if (!ok) qwen4_ctx_bufs_free(b);
    return ok;
}

/* Exchange the graph's context-sized caches with the set in b. */
static void qwen4_ctx_bufs_swap(ds4_qwen4_gpu_graph *g, qwen4_ctx_bufs *b) {
#define QWEN4_SWAP(a_, b_) do { ds4_gpu_tensor *t_ = (a_); (a_) = (b_); (b_) = t_; } while (0)
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        QWEN4_SWAP(g->layer_k_cache[il], b->k[il]);
        QWEN4_SWAP(g->layer_v_cache[il], b->v[il]);
        QWEN4_SWAP(g->layer_k_cache_fp8[il], b->kf[il]);
        QWEN4_SWAP(g->layer_v_cache_fp8[il], b->vf[il]);
        QWEN4_SWAP(g->layer_k_scale[il], b->ks[il]);
        QWEN4_SWAP(g->layer_v_scale[il], b->vs[il]);
        if (!g->ik_ring) QWEN4_SWAP(g->layer_ik_cache[il], b->ik[il]);
        QWEN4_SWAP(g->layer_block_key[il], b->bk[il]);
    }
    QWEN4_SWAP(g->pos3, b->pos3);
#undef QWEN4_SWAP
}
```

- [ ] **Step 5: Use the helper in `qwen4_graph_alloc`**

(a) Change the signature

```c
                              ds4_gpu_tensor *const *hist_pool, uint32_t slot) {
    memset(g, 0, sizeof(*g));
```

to

```c
                              ds4_gpu_tensor *const *hist_pool, uint32_t slot,
                              uint32_t alloc_rows) {
    memset(g, 0, sizeof(*g));
```

and update the comment above the function by appending: ` alloc_rows, when non-zero, allocates the context-sized caches for that many rows (grow-on-demand); 0 allocates ctx_cap.`

(b) Replace

```c
    g->ctx_cap = ctx_cap;
    g->cap_tokens = cap_tokens;
```

with

```c
    g->ctx_cap = ctx_cap;
    g->alloc_cap = alloc_rows && alloc_rows < ctx_cap ? alloc_rows : ctx_cap;
    g->kv_grow = alloc_rows != 0;
    g->kv_init_cap = g->alloc_cap;
    g->cap_tokens = cap_tokens;
```

(c) Replace `    g->n_block_cap = ctx_cap / 4u + 1u;` with `    g->n_block_cap = g->alloc_cap / 4u + 1u;`

(d) Replace the whole attention-layer allocation (from `            if (g->kv_fp8) {` + `                /* In-kernel FP8: E4M3 byte cache (1 B/elem) + per-64-block fp16` down to and including the closing `            }` after `g->layer_ik_cache[il] && g->layer_block_key[il];`) with:

```c
            /* The context-sized caches come from qwen4_ctx_bufs_alloc below;
             * only the ring-sized raw indexer cache is allocated here. */
            if (g->ik_ring) {
                g->layer_ik_cache[il] = qwen4_graph_alloc_f32((uint64_t)g->ik_ring * DS4_N_INDEXER_HEAD_DIM);
                ok = ok && g->layer_ik_cache[il];
            }
```

(e) Replace

```c
    g->pos3 = qwen4_graph_alloc_f32((uint64_t)ctx_cap * 4u);
    ok = ok && g->pos3;
```

with

```c
    if (ok) {
        qwen4_ctx_bufs ctx_bufs;
        ok = qwen4_ctx_bufs_alloc(g, g->alloc_cap, &ctx_bufs);
        if (ok) qwen4_ctx_bufs_swap(g, &ctx_bufs);  /* ctx_bufs now holds the graph's NULLs */
    }
```

- [ ] **Step 6: Pass `alloc_rows` at the four call sites**

- `qwen4_graph_alloc(g, weights, (uint32_t)ctx_size, qwen4_prefill_chunk_tokens((uint32_t)ctx_size), false, NULL, NULL, NULL, 0)` → append `, 0u` before the final `)` (one-shot generate).
- `qwen4_graph_alloc(g, weights, 8192u, 128u, false, NULL, NULL, NULL, 0)` → append `, 0u`.
- `qwen4_graph_alloc(g, weights, n_seq + 4u, chunk, draft != NULL, NULL, NULL, NULL, 0)` → append `, 0u`.
- Session create: replace

```c
        const uint32_t block_cap = (uint32_t)ctx_size / 4u + 1u;
```

with

```c
        /* DS4_QWEN4_KV_GROW=1 (Metal): allocate the context-sized caches for
         * the initial capacity and grow them as the conversation does. */
        uint32_t kv_alloc = 0;
        {
            const char *grow = getenv("DS4_QWEN4_KV_GROW");
            if (grow && grow[0] && grow[0] != '0' && e->backend == DS4_BACKEND_METAL)
                kv_alloc = ds4_qwen4_kv_initial_cap((uint32_t)ctx_size, getenv("DS4_QWEN4_KV_INIT_CAP"));
        }
        const uint32_t block_cap = (kv_alloc ? kv_alloc : (uint32_t)ctx_size) / 4u + 1u;
```

and in the session `qwen4_graph_alloc(&s->qwen4_graph, ...)` call replace `s->qwen4_slot >= 0 ? (uint32_t)s->qwen4_slot : 0u)) {` with `s->qwen4_slot >= 0 ? (uint32_t)s->qwen4_slot : 0u, kv_alloc)) {`.
Right after that `if (!qwen4_graph_alloc(...)) { ... }` block, add:

```c
        s->qwen4_graph.kv_reserve = ds4_qwen4_kv_reserve(getenv("DS4_QWEN4_KV_RESERVE"));
        if (kv_alloc) {
            static bool announced_grow = false;
            if (!announced_grow) {
                announced_grow = true;
                fprintf(stderr, "ds4: qwen4 grow-on-demand KV: %u of %d rows allocated up front\n",
                        s->qwen4_graph.alloc_cap, ctx_size);
            }
        }
```

- [ ] **Step 7: Use `alloc_cap` for every buffer guard and binding**

In `ds4.c` make exactly these replacements (each anchor is unique):
- In `qwen4_graph_attention_tail`: `pos0, g->ctx_cap, DS4_ROPE_FREQ_BASE, DS4_RMS_EPS,` → `pos0, g->alloc_cap, DS4_ROPE_FREQ_BASE, DS4_RMS_EPS,`
- In the batched rows prep: `pos0, r->ctx_cap, DS4_ROPE_FREQ_BASE, DS4_RMS_EPS,` → `pos0, r->alloc_cap, DS4_ROPE_FREQ_BASE, DS4_RMS_EPS,`
- `if (!g || T == 0 || T > g->cap_tokens || g->pos + T > g->ctx_cap) return false;` → `... g->pos + T > g->alloc_cap) return false;`
- `if (!g->mtp_R || T == 0u || T > 3u || idx > g->ctx_cap || T > g->ctx_cap - idx ||` → `if (!g->mtp_R || T == 0u || T > 3u || idx > g->alloc_cap || T > g->alloc_cap - idx ||`
- `idx >= g->ctx_cap || next_token < 0 || next_token >= (int)DS4_N_VOCAB) return false;` → `idx >= g->alloc_cap || ...`
- `if (r->pos >= r->ctx_cap) return false;` → `if (r->pos >= r->alloc_cap) return false;`
- `if (e->pos >= r->ctx_cap) ok = false;` → `if (e->pos >= r->alloc_cap) ok = false;`
- `if (e->pos >= r->ctx_cap) return false;` → `if (e->pos >= r->alloc_cap) return false;`

Leave every other `ctx_cap` use unchanged (logical checks in the spec cycle, decode, batch room, checkpoint header).

- [ ] **Step 8: Add the test hook**

In `ds4.c`, directly after

```c
int ds4_session_prefill_cap(ds4_session *s) {
    return s ? (int)s->prefill_cap : 0;
}
```

add

```c
#ifndef DS4_NO_GPU
/* Test hook: rows allocated for the qwen4 context-sized caches, or -1. */
int ds4_test_qwen4_alloc_cap(ds4_session *s) {
#ifdef DS4_HAS_QWEN4_GPU
    if (s && ds4_session_is_qwen4(s) && s->qwen4_graph_ready) return (int)s->qwen4_graph.alloc_cap;
#else
    (void)s;
#endif
    return -1;
}
#endif
```

- [ ] **Step 9: Run tests to verify they pass**

Run:
```bash
make -j10 ds4 ds4-server ds4_test && ./ds4_test --qwen-kv-grow-policy && \
. ./kv-grow-test.env && ./ds4_test --qwen-kv-grow
```
Expected: both exit 0. The second prints the `grow-on-demand KV: 4096 of 65536 rows` line and a `sync failed` message for the 5000-token prompt (that failure is asserted).
Also run the existing Qwen regressions with the flag off (`DS4_QWEN4_KV_GROW` unset) to prove nothing moved: `. ./kv-grow-test.env && ./ds4_test --qwen4-prefill-checkpoints --qwen4-restore-reuse --session-rewind` — expected exit 0.

- [ ] **Step 10: Commit**

```bash
git add ds4.c tests/ds4_test.c
git commit -m "qwen4: allocate context caches at alloc_cap behind DS4_QWEN4_KV_GROW

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017f1tsCGFkVQ3wrmPsJ7FBn"
```

---

### Task 3: Grow, shrink, and the single-session entry points

**Files:**
- Modify: `ds4.c` — new `qwen4_graph_resize_ctx` after `qwen4_ctx_bufs_swap`; new `qwen4_session_ensure_cap` / `qwen4_session_shrink_cap` just before `static int ds4_session_qwen4_spec_cycle(`; call sites in `ds4_session_sync_internal`, the qwen4 branch of the single-token eval, `ds4_session_qwen4_spec_cycle`, `qwen4_session_load_payload`
- Modify: `tests/ds4_test.c` — extend `test_qwen_kv_grow`

**Interfaces:**
- Consumes: Task 1 policy, Task 2 fields/helpers/hook.
- Produces:
  - `static bool qwen4_graph_resize_ctx(ds4_qwen4_gpu_graph *g, uint32_t rows, ds4_qwen4_gpu_graph *arena, bool arena_sole_user);`
  - `static int qwen4_session_ensure_cap(ds4_session *s, uint32_t need, char *err, size_t errlen);` (0 = ok)
  - `static void qwen4_session_shrink_cap(ds4_session *s, uint32_t need);`

- [ ] **Step 1: Write the failing tests**

In `test_qwen_kv_grow`, replace the block

```c
    /* Without growth a prompt past the initial capacity fails cleanly. */
    test_kv_grow_prompt(engine, 5000, 2, &prompt);
    TEST_ASSERT(test_kv_grow_run(engine, &prompt, 1, false, got) == -1);
```

with

```c
    /* A. growth mid-decode (plain eval): 3800 prompt tokens fit 4096, the
     *    decode crosses 4096 - 64 one token at a time. */
    test_kv_grow_prompt(engine, 3800, 3, &prompt);
    test_kv_grow_env(false, 4096, 64);
    TEST_ASSERT(test_kv_grow_run(engine, &prompt, 600, false, ref) == TEST_KV_CTX);
    test_kv_grow_env(true, 4096, 64);
    TEST_ASSERT(test_kv_grow_run(engine, &prompt, 600, false, got) == 8192);
    TEST_ASSERT(memcmp(ref, got, 600 * sizeof(int)) == 0);

    /* B. growth at the sync reservation: 9000 + 4096 -> 13312 rows. */
    test_kv_grow_prompt(engine, 9000, 4, &prompt);
    test_kv_grow_env(false, 4096, 4096);
    TEST_ASSERT(test_kv_grow_run(engine, &prompt, 32, false, ref) == TEST_KV_CTX);
    test_kv_grow_env(true, 4096, 4096);
    TEST_ASSERT(test_kv_grow_run(engine, &prompt, 32, false, got) == 13312);
    TEST_ASSERT(memcmp(ref, got, 32 * sizeof(int)) == 0);

    /* C. growth mid-decode under MTP speculation. */
    test_kv_grow_prompt(engine, 3800, 5, &prompt);
    test_kv_grow_env(false, 4096, 64);
    TEST_ASSERT(test_kv_grow_run(engine, &prompt, 600, true, ref) == TEST_KV_CTX);
    test_kv_grow_env(true, 4096, 64);
    TEST_ASSERT(test_kv_grow_run(engine, &prompt, 600, true, got) == 8192);
    TEST_ASSERT(memcmp(ref, got, 600 * sizeof(int)) == 0);

    /* D. shrink at a session boundary: 10000 tokens grow to 10240 rows, an
     *    unrelated 200-token prompt resets the graph and shrinks it to 4096. */
    {
        ds4_tokens small = {0};
        test_kv_grow_prompt(engine, 200, 7, &small);
        test_kv_grow_env(false, 4096, 64);
        TEST_ASSERT(test_kv_grow_run(engine, &small, 32, false, ref) == TEST_KV_CTX);
        test_kv_grow_env(true, 4096, 64);
        test_kv_grow_prompt(engine, 10000, 6, &prompt);
        ds4_session *s = NULL;
        char err[192] = {0};
        TEST_ASSERT(ds4_session_create(&s, engine, TEST_KV_CTX) == 0);
        TEST_ASSERT(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
        TEST_ASSERT(ds4_test_qwen4_alloc_cap(s) == 10240);
        TEST_ASSERT(ds4_session_sync(s, &small, err, sizeof(err)) == 0);
        TEST_ASSERT(ds4_test_qwen4_alloc_cap(s) == 4096);
        TEST_ASSERT(test_kv_grow_decode(engine, s, 32, false, got) == 4096);
        TEST_ASSERT(memcmp(ref, got, 32 * sizeof(int)) == 0);
        ds4_session_free(s);
        ds4_tokens_free(&small);
    }

    /* E. a KV payload longer than a fresh session's capacity grows it first. */
    {
        test_kv_grow_env(true, 4096, 64);
        test_kv_grow_prompt(engine, 6000, 8, &prompt);
        ds4_session *live = NULL, *restored = NULL;
        char err[192] = {0};
        int head[16];
        TEST_ASSERT(ds4_session_create(&live, engine, TEST_KV_CTX) == 0);
        TEST_ASSERT(ds4_session_create(&restored, engine, TEST_KV_CTX) == 0);
        TEST_ASSERT(ds4_session_sync(live, &prompt, err, sizeof(err)) == 0);
        TEST_ASSERT(test_kv_grow_decode(engine, live, 16, false, head) == 8192);
        FILE *fp = tmpfile();
        TEST_ASSERT(fp != NULL);
        TEST_ASSERT(ds4_session_save_payload(live, fp, err, sizeof(err)) == 0);
        const uint64_t bytes = (uint64_t)ftell(fp);
        rewind(fp);
        TEST_ASSERT(ds4_test_qwen4_alloc_cap(restored) == 4096);
        TEST_ASSERT(ds4_session_load_payload(restored, fp, bytes, err, sizeof(err)) == 0);
        fclose(fp);
        TEST_ASSERT(ds4_test_qwen4_alloc_cap(restored) == 8192);
        TEST_ASSERT(test_kv_grow_decode(engine, live, 32, false, ref) == 8192);
        TEST_ASSERT(test_kv_grow_decode(engine, restored, 32, false, got) == 8192);
        TEST_ASSERT(memcmp(ref, got, 32 * sizeof(int)) == 0);
        ds4_session_free(restored);
        ds4_session_free(live);
    }

    /* F. a failed resize leaves the session usable. */
    {
        test_kv_grow_prompt(engine, 5000, 9, &prompt);
        test_kv_grow_env(false, 4096, 64);
        TEST_ASSERT(test_kv_grow_run(engine, &prompt, 16, false, ref) == TEST_KV_CTX);
        test_kv_grow_env(true, 4096, 64);
        ds4_session *s = NULL;
        char err[192] = {0};
        TEST_ASSERT(ds4_session_create(&s, engine, TEST_KV_CTX) == 0);
        setenv("DS4_QWEN4_KV_GROW_TEST_FAIL", "1", 1);
        TEST_ASSERT(ds4_session_sync(s, &prompt, err, sizeof(err)) != 0);
        TEST_ASSERT(strstr(err, "KV capacity") != NULL);
        TEST_ASSERT(ds4_test_qwen4_alloc_cap(s) == 4096);
        unsetenv("DS4_QWEN4_KV_GROW_TEST_FAIL");
        TEST_ASSERT(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
        TEST_ASSERT(test_kv_grow_decode(engine, s, 16, false, got) == 8192);
        TEST_ASSERT(memcmp(ref, got, 16 * sizeof(int)) == 0);
        ds4_session_free(s);
    }
```

Check the payload save API returns 0 on success: `grep -n 'int ds4_session_save_payload' ds4.c` and read its first lines; if it returns non-zero on success, adapt the two asserts to its convention.

- [ ] **Step 2: Run test to verify it fails**

Run: `make ds4_test && . ./kv-grow-test.env && ./ds4_test --qwen-kv-grow`
Expected: FAIL at case A (`sync failed` or `eval failed` once the decode reaches the initial capacity).

- [ ] **Step 3: Implement resize**

In `ds4.c`, directly after `qwen4_ctx_bufs_swap`, add:

```c
/* Reallocate g's context-sized caches for rows positions, keeping the first
 * g->pos positions and their blocks.  Runs only between forwards: every qwen4
 * entry point ends with ds4_gpu_end_commands, so no command buffer holds the
 * old buffers.  The indexer scores (score/tile_max) live in the graph's own
 * scratch or in the engine arena it borrows; they are transients, so they are
 * reallocated without a copy, and the arena only shrinks when this session is
 * its sole user.  On failure nothing changes. */
static bool qwen4_graph_resize_ctx(ds4_qwen4_gpu_graph *g, uint32_t rows,
                                   ds4_qwen4_gpu_graph *arena, bool arena_sole_user) {
    if (rows == g->alloc_cap) return true;
    if (rows < g->pos || rows > g->ctx_cap) return false;
    if (getenv("DS4_QWEN4_KV_GROW_TEST_FAIL")) return false;

    qwen4_ctx_bufs b;
    if (!qwen4_ctx_bufs_alloc(g, rows, &b)) return false;

    const uint32_t blocks = rows / 4u + 1u;
    ds4_qwen4_gpu_graph *owner = g->owns_scratch ? g : arena;
    const bool resize_scratch = owner &&
        (blocks > owner->n_block_cap ||
         (blocks < owner->n_block_cap && (g->owns_scratch || arena_sole_user)));
    ds4_gpu_tensor *score = NULL, *tile_max = NULL;
    if (resize_scratch) {
        const uint64_t T = owner->cap_tokens;
        score = qwen4_graph_alloc_f32(T * blocks);
        tile_max = qwen4_graph_alloc_f32(T * (blocks / 8u + 1u));
        if (!score || !tile_max) {
            ds4_gpu_tensor_free(score);
            ds4_gpu_tensor_free(tile_max);
            qwen4_ctx_bufs_free(&b);
            return false;
        }
    }

    const uint64_t used = g->pos;
    bool ok = true;
    if (used) {
        const uint64_t kv_dim = (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
        const uint64_t kv_row = g->kv_fp8 ? (g->kv_q4 ? kv_dim / 2u : kv_dim) : kv_dim * 2u;
        const uint64_t sc_row = (kv_dim / 64u) * 2u;
        const uint64_t bk_bytes = (used / 4u + 1u) * DS4_N_INDEXER_HEAD_DIM * 2u;
        ok = ds4_gpu_begin_commands() != 0;
        for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
            if (ds4_qwen4_layer_is_linear(il)) continue;
            if (g->kv_fp8) {
                ok = ds4_gpu_tensor_copy(b.kf[il], 0, g->layer_k_cache_fp8[il], 0, used * kv_row) &&
                     ds4_gpu_tensor_copy(b.vf[il], 0, g->layer_v_cache_fp8[il], 0, used * kv_row) &&
                     ds4_gpu_tensor_copy(b.ks[il], 0, g->layer_k_scale[il], 0, used * sc_row) &&
                     ds4_gpu_tensor_copy(b.vs[il], 0, g->layer_v_scale[il], 0, used * sc_row);
            } else {
                ok = ds4_gpu_tensor_copy(b.k[il], 0, g->layer_k_cache[il], 0, used * kv_row) &&
                     ds4_gpu_tensor_copy(b.v[il], 0, g->layer_v_cache[il], 0, used * kv_row);
            }
            if (ok && !g->ik_ring)
                ok = ds4_gpu_tensor_copy(b.ik[il], 0, g->layer_ik_cache[il], 0,
                                         used * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
            if (ok) ok = ds4_gpu_tensor_copy(b.bk[il], 0, g->layer_block_key[il], 0, bk_bytes);
        }
        if (ok) ok = ds4_gpu_tensor_copy(b.pos3, 0, g->pos3, 0, used * 4u * sizeof(uint32_t));
        ok = ds4_gpu_end_commands() != 0 && ok;
    }
    if (!ok) {
        ds4_gpu_tensor_free(score);
        ds4_gpu_tensor_free(tile_max);
        qwen4_ctx_bufs_free(&b);
        return false;
    }

    const uint64_t new_bytes = qwen4_ctx_bufs_bytes(&b);
    qwen4_ctx_bufs_swap(g, &b);                   /* b now holds the old set */
    const uint64_t old_bytes = qwen4_ctx_bufs_bytes(&b);
    qwen4_ctx_bufs_free(&b);
    if (resize_scratch) {
        ds4_gpu_tensor_free(owner->score);
        ds4_gpu_tensor_free(owner->tile_max);
        owner->score = score;
        owner->tile_max = tile_max;
        owner->n_block_cap = blocks;
    }
    if (!g->owns_scratch && arena) {
        g->score = arena->score;
        g->tile_max = arena->tile_max;
    }
    fprintf(stderr, "ds4: qwen4 KV capacity %u -> %u rows (%+.2f GiB)\n", g->alloc_cap, rows,
            ((double)new_bytes - (double)old_bytes) / (1024.0 * 1024.0 * 1024.0));
    g->alloc_cap = rows;
    g->n_block_cap = blocks;
    return true;
}
```

If `bk_bytes` could exceed the old buffer (it cannot: `used <= alloc_cap` so `used/4+1 <= alloc_cap/4+1`), `ds4_gpu_tensor_copy` returns 0 and the resize fails safely.

- [ ] **Step 4: Implement the session helpers**

In `ds4.c`, directly before `static int ds4_session_qwen4_spec_cycle(ds4_session *s, int first_token, float temperature, int top_k,`, add:

```c
/* Make s->qwen4_graph hold at least need positions (clamped to -c), and
 * re-read the scratch it borrows from the engine arena, which another
 * session may have resized.  Call only between forwards (see
 * qwen4_graph_resize_ctx).  0 on success. */
static int qwen4_session_ensure_cap(ds4_session *s, uint32_t need, char *err, size_t errlen) {
    ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
    ds4_qwen4_gpu_graph *arena = g->owns_scratch ? NULL : s->engine->qwen4_shared_workspace;
    if (g->kv_grow) {
        const uint32_t target = ds4_qwen4_kv_grow_target(g->alloc_cap, need, g->ctx_cap);
        if (target != g->alloc_cap &&
            !qwen4_graph_resize_ctx(g, target, arena, s->engine->qwen4_arena_users <= 1u)) {
            if (err && errlen)
                snprintf(err, errlen, "Qwen3.8 KV capacity %u -> %u rows failed", g->alloc_cap, target);
            return 1;
        }
    }
    if (arena) {
        g->score = arena->score;
        g->tile_max = arena->tile_max;
    }
    return 0;
}

/* The graph was just reset for a conversation that needs need positions:
 * give back capacity it will not use.  Failure only keeps the larger set. */
static void qwen4_session_shrink_cap(ds4_session *s, uint32_t need) {
    ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
    if (!g->kv_grow || g->pos != 0) return;
    const uint32_t target = ds4_qwen4_kv_shrink_target(g->alloc_cap, need, g->kv_init_cap, g->ctx_cap);
    if (target == g->alloc_cap) return;
    ds4_qwen4_gpu_graph *arena = g->owns_scratch ? NULL : s->engine->qwen4_shared_workspace;
    (void)qwen4_graph_resize_ctx(g, target, arena, s->engine->qwen4_arena_users <= 1u);
}
```

`qwen4_session_load_payload` (~63539) comes earlier in the file than these definitions, so also add these prototypes directly after the `qwen4_graph_resize_ctx` definition:

```c
static int qwen4_session_ensure_cap(ds4_session *s, uint32_t need, char *err, size_t errlen);
static void qwen4_session_shrink_cap(ds4_session *s, uint32_t need);
```

(`ds4_session` is only a typedef'd incomplete type at that point if its struct is defined later; the prototypes only need the typedef, which exists near the top of ds4.c — confirm with `grep -n 'typedef struct ds4_session ds4_session' ds4.h ds4.c`.)

- [ ] **Step 5: Call sites**

(a) `ds4_session_sync_internal`, qwen4 branch — replace

```c
        } else {
            qwen4_graph_reset(&s->qwen4_graph);
            s->checkpoint.len = 0;
            s->checkpoint_valid = false;
        }
        for (int i = start; i < prompt->len; i++) {
            if (prompt->v[i] < 0 || prompt->v[i] >= (int)DS4_N_VOCAB) {
```

with

```c
        } else {
            qwen4_graph_reset(&s->qwen4_graph);
            s->checkpoint.len = 0;
            s->checkpoint_valid = false;
            qwen4_session_shrink_cap(s, (uint32_t)prompt->len + s->qwen4_graph.kv_reserve);
        }
        if (qwen4_session_ensure_cap(s, (uint32_t)prompt->len + s->qwen4_graph.kv_reserve,
                                     err, errlen) != 0) {
            return 1;
        }
        for (int i = start; i < prompt->len; i++) {
            if (prompt->v[i] < 0 || prompt->v[i] >= (int)DS4_N_VOCAB) {
```

(b) Single-token eval, qwen4 branch — replace

```c
        if (qwen4_session_replay_if_stale(s, err, errlen) != 0) return 1;
        if (s->qwen4_graph.pos >= s->qwen4_graph.ctx_cap) {
```

with

```c
        if (qwen4_session_replay_if_stale(s, err, errlen) != 0) return 1;
        if (qwen4_session_ensure_cap(s, s->qwen4_graph.pos + DS4_QWEN4_KV_MARGIN, err, errlen) != 0) return 1;
        if (s->qwen4_graph.pos >= s->qwen4_graph.ctx_cap) {
```

(c) `ds4_session_qwen4_spec_cycle` — replace

```c
    if (qwen4_session_replay_if_stale(s, err, errlen) != 0) return -1;
    const uint32_t pos = g->pos;
```

with

```c
    if (qwen4_session_replay_if_stale(s, err, errlen) != 0) return -1;
    if (qwen4_session_ensure_cap(s, g->pos + DS4_QWEN4_KV_MARGIN, err, errlen) != 0) return -1;
    const uint32_t pos = g->pos;
```

(d) `qwen4_session_load_payload` — replace

```c
    if (rows > g->ctx_cap || rows > (uint32_t)s->ctx_size) {
        payload_set_err(err, errlen, "KV checkpoint is longer than this session's context");
        return 1;
    }
```

with

```c
    if (rows > g->ctx_cap || rows > (uint32_t)s->ctx_size) {
        payload_set_err(err, errlen, "KV checkpoint is longer than this session's context");
        return 1;
    }
    if (qwen4_session_ensure_cap(s, rows + DS4_QWEN4_KV_MARGIN, err, errlen) != 0) return 1;
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `make -j10 ds4 ds4-server ds4_test && . ./kv-grow-test.env && ./ds4_test --qwen-kv-grow && ./ds4_test --qwen-kv-grow-policy`
Expected: exit 0; the log shows `qwen4 KV capacity 4096 -> 8192 rows`, `4096 -> 13312`, `4096 -> 10240`, `10240 -> 4096`.
Also re-run the flag-off regressions: `. ./kv-grow-test.env && ./ds4_test --qwen4-prefill-checkpoints --qwen4-restore-reuse --session-rewind --mtp-verify-depth` — expected exit 0 (`--mtp-verify-depth` may be skipped by the runner if its prerequisites are missing; that is fine).

- [ ] **Step 7: Commit**

```bash
git add ds4.c tests/ds4_test.c
git commit -m "qwen4: grow KV on demand and shrink it at a session reset

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017f1tsCGFkVQ3wrmPsJ7FBn"
```

---

### Task 4: Batched decode and the shared arena

**Files:**
- Modify: `ds4.c` — `ds4_sessions_eval_batch_speculative_argmax` and `ds4_sessions_eval_batch`
- Modify: `tests/ds4_test.c` — extend `test_qwen_kv_grow` (runs only when the engine has the shared arena)

**Interfaces:**
- Consumes: `qwen4_session_ensure_cap` (Task 3), `DS4_TEST_SHARE_WORKSPACE` (Task 2).

- [ ] **Step 1: Write the failing test**

At the START of `test_qwen_kv_grow`, directly after the line `static int ref[640], got[640];`, add the block below. It must run first and with the flag on first: the engine's shared arena is created by the first session ever opened on it, so only then does it start small (4096 rows) and have to grow when P crosses its capacity. Run `--qwen-kv-grow` on its own (not together with other model tests) so no earlier session sizes the arena.

```c
    /* Batched decode over the shared arena (DS4_TEST_SHARE_WORKSPACE=1):
     * P crosses its capacity inside the batch and grows the arena's indexer
     * scores; Q keeps borrowing them. The flag on must reproduce the same
     * batched tokens as the flag off.  Then P restarts on a short prompt and
     * shrinks while Q, still larger, keeps decoding correctly. */
    if (test_env_bool("DS4_TEST_SHARE_WORKSPACE")) {
        ds4_tokens pp = {0}, qp = {0}, small = {0};
        test_kv_grow_prompt(engine, 3900, 10, &pp);
        test_kv_grow_prompt(engine, 4000, 11, &qp);
        test_kv_grow_prompt(engine, 200, 12, &small);
        static int toks[2][2][300];   /* [flag][session][step] */
        static int tail[2][32];       /* Q after P shrank, [flag][step] */
        for (int flag = 1; flag >= 0; flag--) {
            test_kv_grow_env(flag == 1, 4096, 64);
            ds4_session *p = NULL, *q = NULL;
            char err[192] = {0};
            TEST_ASSERT(ds4_session_create(&p, engine, TEST_KV_CTX) == 0);
            TEST_ASSERT(ds4_session_create(&q, engine, TEST_KV_CTX) == 0);
            TEST_ASSERT(ds4_session_sync(p, &pp, err, sizeof(err)) == 0);
            TEST_ASSERT(ds4_session_sync(q, &qp, err, sizeof(err)) == 0);
            for (int step = 0; step < 300; step++) {
                ds4_decode_item items[2] = {
                    {.session = p, .token = ds4_session_argmax(p)},
                    {.session = q, .token = ds4_session_argmax(q)},
                };
                toks[flag][0][step] = items[0].token;
                toks[flag][1][step] = items[1].token;
                TEST_ASSERT(ds4_sessions_eval_batch(items, 2, err, sizeof(err)) == 0);
            }
            if (flag == 1) {
                TEST_ASSERT(ds4_test_qwen4_alloc_cap(p) == 8192);
                TEST_ASSERT(ds4_test_qwen4_alloc_cap(q) == 8192);
            }
            TEST_ASSERT(ds4_session_sync(p, &small, err, sizeof(err)) == 0);
            if (flag == 1) TEST_ASSERT(ds4_test_qwen4_alloc_cap(p) == 4096);
            TEST_ASSERT(test_kv_grow_decode(engine, q, 32, false, tail[flag]) >= 0);
            ds4_session_free(q);
            ds4_session_free(p);
        }
        TEST_ASSERT(memcmp(toks[0], toks[1], sizeof(toks[0])) == 0);
        TEST_ASSERT(memcmp(tail[0], tail[1], sizeof(tail[0])) == 0);
        ds4_tokens_free(&pp);
        ds4_tokens_free(&qp);
        ds4_tokens_free(&small);
    }
```

Check the `ds4_decode_item` field names with `grep -n -A4 'typedef struct' ds4.h | grep -B2 -A4 ds4_decode_item` and adapt `.session`/`.token` if they differ.

- [ ] **Step 2: Run test to verify it fails**

Run: `make ds4_test && . ./kv-grow-test.env && DS4_TEST_SHARE_WORKSPACE=1 ./ds4_test --qwen-kv-grow`
Expected: FAIL in the batched block (a batch step fails with a guard error or the token streams differ) because the batch entry points never grow.

- [ ] **Step 3: Grow at the batch entry points**

(a) In `ds4_sessions_eval_batch_speculative_argmax`, after its validation `for` loop (the one ending with the `"decode batch item %d reached its context limit"` error and its closing braces) and before `#ifdef DS4_HAS_QWEN4_METAL`, insert:

```c
#ifdef DS4_HAS_QWEN4_GPU
    for (int i = 0; i < count; i++) {
        ds4_session *s = items[i].session;
        if (ds4_session_is_qwen4(s) && s->qwen4_graph_ready &&
            qwen4_session_ensure_cap(s, s->qwen4_graph.pos + DS4_QWEN4_KV_MARGIN, err, errlen) != 0) {
            return 1;
        }
    }
#endif
```

(b) In `ds4_sessions_eval_batch`, insert the same block after its validation `for` loop and before the `#ifndef DS4_NO_GPU` that starts with `if (e->backend == DS4_BACKEND_CUDA) {` / `return ds4_sessions_eval_batch_cuda(items, count, err, errlen);`.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
make -j10 ds4 ds4-server ds4_test && . ./kv-grow-test.env && DS4_TEST_SHARE_WORKSPACE=1 ./ds4_test --qwen-kv-grow && \
./ds4_test --qwen-kv-grow --qwen-kv-grow-policy --qwen4-prefill-checkpoints --qwen4-restore-reuse
```
Expected: exit 0 for both. Then run the existing batch oracle with the flag on and off to be sure the rows path is unaffected:
`DS4_TEST_MODEL=<model> make test-metal-session-batch` — note it does not load a PLE; if it fails to open the model for that reason, add `.ple_path = getenv("DS4_TEST_PLE"),` to its `ds4_engine_options opt = {` in `tests/test_metal_session_batch.c` and run it with `DS4_TEST_PLE=<ple>` twice: once plain, once with `DS4_QWEN4_KV_GROW=1 DS4_QWEN4_KV_INIT_CAP=256 DS4_QWEN4_KV_RESERVE=64`. Expected: PASS both times.

- [ ] **Step 5: Commit**

```bash
git add ds4.c tests/ds4_test.c tests/test_metal_session_batch.c
git commit -m "qwen4: grow KV at the batched decode entry points

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017f1tsCGFkVQ3wrmPsJ7FBn"
```

---

### Task 5: End-to-end verification at real sizes and performance receipt

No engine code changes. Everything uses the built `ds4`/`ds4-server` from the worktree root (the runtime loads `./metal` from the executable's directory).

**Files:**
- Create: `speed-bench/stageb-runs/kv-grow/RESULTS-kv-grow.md`
- Create (not committed, scratch): prompt files under the session scratchpad.

Common env for every run below (PROD scale config):

```bash
M=/Users/dongnh/orca/workspaces/ds4-metal-data/gguf
export DS4_QWEN4_STREAM_FULL_LAYERS=32 DS4_QWEN4_PLE_PREFETCH_FULL=0 \
  DS4_QWEN4_MTP_DRAFT_VOCAB=/Users/dongnh/.local/share/ai-gateway/ds4-models/Qwen3.8-Flash-Next-draft-vocab-vi-en-code-64k.txt
ARGS="--metal -m $M/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q4KDownPad768-DenseQ4Kselimat-MTP.gguf \
  --ple $M/Qwen3.8-Flash-Next-PLE-Q4_1.gguf --prefill-chunk 2048 --mtp --ssd-streaming \
  --ssd-streaming-cache-experts 6GB --temp 0"
```

- [ ] **Step 1: Build long prompts**

Generate prompt files from `ds4.c` text (code filler + a needle in the middle): about 37K, 40K, 135K and 256K tokens, plus a 31K prompt for the mid-decode case. Count tokens with `./ds4 -m $M/<model> --prompt-file F --dump-tokens 2>/dev/null | grep -cE '^\s*[0-9]+\s'` and trim the character count until each file is within 1% of its target (256K target: at most 256,500 tokens so `-c 262144` leaves room for 1000 generated tokens).

- [ ] **Step 2: Byte-exact gate (CLI)**

For each case run the flag off and on with the same command and compare stdout with `cmp`:

| case | prompt | `-n` | `-c` |
|---|---|---|---|
| short | `-p "Explain how a hash map handles collisions, compare chaining and open addressing, and say when each is the better choice."` | 800 | 262144 |
| reserve | 40K prompt file | 400 | 262144 |
| mid-decode | 31K prompt file, `DS4_QWEN4_KV_INIT_CAP=32768` | 2048 | 262144 |
| full | 256K prompt file | 1000 | 262144 |

Command shape: `./ds4 $ARGS -c 262144 -n N <prompt> > off.out 2> off.err` then `DS4_QWEN4_KV_GROW=1 ./ds4 $ARGS -c 262144 -n N <prompt> > on.out 2> on.err`; `cmp off.out on.out` must print nothing. `on.err` must show the expected `qwen4 KV capacity A -> B rows` lines (none for short; one at reserve for 40K; one mid-decode for 31K+2K; a jump to 262144 for 256K).

- [ ] **Step 3: Server gates**

Start `./ds4-server $ARGS -c 262144 --kv-disk-dir <scratch>/kv --kv-disk-space-mb 32768 --kv-cache-cold-max-tokens 262144 --host 127.0.0.1 --port 18397` with and without `DS4_QWEN4_KV_GROW=1`.
1. Checkpoint restore: send a 40K-token chat request (`max_tokens` 200), stop the server, restart it with the same KV dir, send the same conversation plus one more user turn; the reply must be byte-identical between flag on and flag off.
2. Shrink: with the flag on, send a 100K-token request, then an unrelated short request. The server log must show the shrink line, system wired memory (`vm_stat` "Pages wired down" × 16384) must drop by about 1 GiB between the two requests, and the short reply must equal the flag-off reply.

- [ ] **Step 4: Performance A/B**

At `-c 262144`, flag off vs on, alternating runs (off, on, off, on) because the machine drifts: short prompt (800 tokens), 37K summary prompt (1000 tokens), 135K (1000), 256K (1000). Record decode t/s and prefill t/s from `ds4: prefill: X t/s, generation: Y t/s`, and peak system wired memory sampled every 0.5 s with `vm_stat` during each run.
Expected: short and 37K within ~1% of the `-c 65536` numbers measured in the spec (42.9 t/s short, 39.6-40.0 t/s 37K decode; 404 t/s 37K prefill); 256K unchanged within noise.

- [ ] **Step 5: Write the receipt**

Create `speed-bench/stageb-runs/kv-grow/RESULTS-kv-grow.md` with: the commit tested, machine, command lines, the byte-exact table (case → cmp result → capacity log lines), the server gates, and the performance table (median of the paired runs, peak wired). State any case that did not meet expectations plainly.

- [ ] **Step 6: Commit**

```bash
git add speed-bench/stageb-runs/kv-grow/RESULTS-kv-grow.md
git commit -m "RESULTS: qwen4 grow-on-demand KV byte-exact gates and A/B

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017f1tsCGFkVQ3wrmPsJ7FBn"
```

---

### Task 6: Review and merge

- [ ] **Step 1:** Use superpowers:requesting-code-review on the whole branch (`git diff develop...kv-grow`), with the spec and this plan as context. Address findings with superpowers:receiving-code-review.
- [ ] **Step 2:** Re-run `./ds4_test --qwen-kv-grow-policy`, the model-backed `--qwen-kv-grow` (with and without `DS4_TEST_SHARE_WORKSPACE=1`), the Task 2 flag-off regressions, and the kernel suites `make test-qwen4-kernels test-qwen4-q2` (expected: all pass; kernels are unchanged, this guards the build).
- [ ] **Step 3:** Use superpowers:finishing-a-development-branch to merge `kv-grow` into `develop` with the flag off by default. Pushing and any PROD deploy (`prod/<feature>-YYYYMMDD` via `deploy-ai-gateway.sh`, adding `DS4_QWEN4_KV_GROW=1` to the registry) are separate steps that need the user's explicit approval.
