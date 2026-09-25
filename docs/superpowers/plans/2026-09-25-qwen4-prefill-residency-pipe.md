# qwen4 Prefill Residency and Staging Pipe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut qwen4 prefill time under `--ssd-streaming` behind `DS4_QWEN4_PREFILL_MODE=off|safe|max`:
- a residency set on prefill command buffers removes the launch latency;
- a double-buffered staging pipe removes the per-streamed-layer drain;
- output stays byte-identical in every mode.

**Architecture:**
- `ds4.c` owns the policy. These are pure functions: mode parse, flags per chunk role, next streamed layer,
  prompt-end rule. The graph gets a `prefill_role` that only the CLI and server prompt loops set.
- `ds4_metal.m` owns two independent mechanisms:
  1. a residency set attached to command buffers created inside a prefill scope;
  2. a staging pipe: two whole-layer buffers, one reader thread, and per-buffer last-reader
     tracking hooked into `ds4_gpu_flush_commands` and `ds4_gpu_end_commands`.
- The pipe's job bookkeeping is plain C in `ds4_qwen4_stage_pipe.h`, so it is testable without Metal.

**Tech Stack:**
- C (`ds4.c`, ~82K lines);
- Objective-C Metal backend (`ds4_metal.m`);
- plain-C model-free test `tests/test_qwen4_prefill_pipe.c` (includes `ds4.c`, like `tests/test_qwen4_ngram_state.c`);
- Python 3 `unittest` for `speed-bench/qwen-regression/qwen_gate.py`;
- GNU make.

**Spec:** `docs/superpowers/specs/2026-09-25-qwen4-prefill-residency-pipe-design.md`

## Global Constraints

- `DS4_QWEN4_PREFILL_MODE=off|safe|max`. Unset, empty or any other value is `off`. The engine default is `off`.
- Policy (`qwen4_prefill_policy(mode, role, T, streaming)` returns `{residency, pipe, pipe_nocache}`):
  - `off`, role `NONE`, `T <= qwen4_moe_mm_min()` (64 on Metal), or not streaming: all false;
  - `max` with role `MIDDLE` or `LAST`: residency and pipe, no nocache;
  - `safe` with role `MIDDLE`: all three true;
  - `safe` with role `LAST`: all false.
- Only the CLI prompt loop and the server session loop set `prefill_role`. Every other caller keeps `NONE`,
  which behaves exactly like `off`.
- Output must be byte-identical across `off`, `safe` and `max`.
- The decode fallback staging call site keeps the old function and never uses the pipe.
- Pipe buffers: buffer 0 is `g_qwen4_stage.buf`, also used by the union path. Buffer 1 (the spare) is allocated
  on demand and released at prompt end. Each holds one whole layer: `need = 2*gate_bytes + down_bytes`.
- The reader reads 32 MiB pieces with 8 reader threads of its own, never the shared pread pool. With `nocache` it
  reads through a second fd with `F_NOCACHE`.
- `DS4_QWEN4_STREAM_SEED_TOKENS > 0` forces the union path for the chunk that seeds.
- Default selection:
  1. `max` if its decode is at least 97 % of PROD on both long-prompt requests (paired server A/B);
  2. otherwise `safe` under the same rule;
  3. otherwise `off`.
  The chosen mode must also show at least +10 % prefill on the ~32K request.
- Other model families and non-streaming runs stay byte-for-byte as today.
- PROD deployment is out of scope (separate approval).
- Code, comments and commit messages in English. Every commit ends with:
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_015npc1LHTBTYaTXLaJt8Uqc`.
- Model artifacts never go into git.

## GPU Steps

Steps marked **[GPU]** need the machine free: no other `ds4`/`ds4-server`, and oMLX plus the watchdogs stopped.
The user authorized stopping them for this project; restore them when the GPU work of the task ends.

Stop:
```bash
launchctl unload ~/Library/LaunchAgents/dev.dongnh.ai-proxy.plist
launchctl unload ~/Library/LaunchAgents/dev.dongnh.gateway-watchdog.plist
P=$(pgrep -x omlx-server); [ -n "$P" ] && kill -TERM $P
```

Restore (the watchdog restarts oMLX within about a minute):
```bash
launchctl load ~/Library/LaunchAgents/dev.dongnh.ai-proxy.plist
launchctl load ~/Library/LaunchAgents/dev.dongnh.gateway-watchdog.plist
```

Never `kill -9` a Metal process stuck in state `U`/`E`; ask the user for a reboot instead.

## Review Focus

1. **A server request cancelled between prefill chunks** must leave no queued read and no spare buffer, and the
   next prompt must stay byte-identical. Task 2 pins the bookkeeping with `check_pipe_prompt_end`; Task 4 wires
   `qwen4_prefill_abandon` into the cancellation branch.
2. **The decode fallback staging (a failed cache load) while a pipe read is queued** must not overwrite or read a
   buffer that a job is filling. Task 4's wrapper waits for queued jobs and invalidates slot 0 before the union
   path writes. Task 2's `check_pipe_match` pins invalidation.
3. **A job from another model map (engine reopened)** must never be accepted. Task 2's `check_pipe_match` covers
   the map mismatch.
4. **Prompt tails at the staging threshold** (`T = mm_min` and `mm_min + 1`) and a `LAST` chunk too small to stage
   must still end the prompt. Task 1's `check_policy` and `check_prompt_end` cover them.
5. **`DS4_QWEN4_STREAM_SEED_TOKENS` set together with `max`** must keep seeding, via the union path on the seeding
   chunk. Task 1's `check_use_pipe` covers it.

## File Structure

- `ds4.c`: policy block (plain C, right after `qwen4_stream_expert_cache_addr_layout_supported`), graph
  fields, `qwen4_graph_moe` pipe call, `qwen4_graph_forward_tokens` residency scope and prompt end, and the prompt loops.
- `ds4_qwen4_stage_pipe.h` (new): plain-C pipe bookkeeping (`qsp_*`).
- `ds4_gpu.h`: residency scope, pipe entry, prompt end.
- `ds4_metal.m`: prefill residency (before `ds4_gpu_new_command_buffer`), whole-run GPU diagnostics
  (`DS4_METAL_GPU_IDLE`), the pipe module (next to the qwen4 staging code), hooks in flush/end/cleanup.
- `tests/test_qwen4_prefill_pipe.c` (new) and a `Makefile` target `test-qwen4-prefill-pipe`.
- `speed-bench/qwen-regression/run.sh`: also builds and runs the new test.
- `speed-bench/qwen-regression/qwen_gate.py` and `speed-bench/tests/test_qwen_gate.py`: `longab` subcommand.
- `speed-bench/qwen-prefill-pipe/` (new): `make_prompts.py`, `cli_run.sh`, `cli_exact.sh`, `RESULTS.md`.

Line numbers refer to `develop` at `500a306` and drift as you edit. Locate each edit by the quoted anchor text,
and confirm it is unique with `grep -c` before editing.

---

### Task 0: Workspace check

**Files:** none

- [ ] **Step 1: Confirm the worktree, branch and clean tree**

Run:
```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/dugong && git branch --show-current && git status --short && git log --oneline -3
```
Expected: branch `feature/qwen4-prefill-pipe`, no output from `status`, top commit is this plan or the spec on top of `500a306`.

- [ ] **Step 2: Baseline build**

Run: `make -j10 ds4 ds4-server 2>&1 | grep -E 'error|warning:'; echo "exit ${pipestatus[1]}"`
Expected: no lines, `exit 0`.

---

### Task 1: Policy helpers and the model-free test harness

**Files:**
- Modify: `ds4.c`, add a block after `qwen4_stream_expert_cache_addr_layout_supported`, before
  `static void model_map_span_vec_include_layer_decode(`; replace the `mm_min` block in `qwen4_graph_moe`.
- Create: `tests/test_qwen4_prefill_pipe.c`
- Modify: `Makefile` (new target next to `test-qwen4-ngrams`; `clean`), `speed-bench/qwen-regression/run.sh`

**Interfaces:**
- Produces (all `static` in `ds4.c`):
  - `typedef enum { QWEN4_PREFILL_OFF = 0, QWEN4_PREFILL_SAFE, QWEN4_PREFILL_MAX } qwen4_prefill_mode;`
  - `typedef enum { QWEN4_PREFILL_ROLE_NONE = 0, QWEN4_PREFILL_ROLE_MIDDLE, QWEN4_PREFILL_ROLE_LAST } qwen4_prefill_role;`
  - `typedef struct { bool residency, pipe, pipe_nocache; } qwen4_prefill_flags;`
  - `uint32_t qwen4_moe_mm_min(void)`
  - `qwen4_prefill_mode qwen4_prefill_mode_parse(const char *s)`
  - `qwen4_prefill_mode qwen4_prefill_mode_env(void)`
  - `qwen4_prefill_flags qwen4_prefill_policy(qwen4_prefill_mode mode, qwen4_prefill_role role, uint32_t T, bool streaming)`
  - `bool qwen4_prefill_prompt_ends(qwen4_prefill_mode mode, qwen4_prefill_role role, bool ok, bool streaming)`
  - `bool qwen4_prefill_use_pipe(qwen4_prefill_flags flags, uint32_t seed_tokens)`
  - `uint32_t qwen4_stream_next_staged(const uint32_t *list, uint32_t n, uint32_t layer, qwen4_prefill_role role)` (returns `UINT32_MAX` for none)
  - `uint32_t qwen4_stream_layer_list(const ds4_weights *w, uint32_t *out, uint32_t cap)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_qwen4_prefill_pipe.c`:
```c
/* Model-free checks for the qwen4 prefill residency / staging pipe policy
 * (docs/superpowers/specs/2026-09-25-qwen4-prefill-residency-pipe-design.md). */
#include "../ds4.c"

static int failures;
#define CHECK(cond) do { \
        if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
    } while (0)

static void check_mode_parse(void) {
    CHECK(qwen4_prefill_mode_parse(NULL) == QWEN4_PREFILL_OFF);
    CHECK(qwen4_prefill_mode_parse("") == QWEN4_PREFILL_OFF);
    CHECK(qwen4_prefill_mode_parse("off") == QWEN4_PREFILL_OFF);
    CHECK(qwen4_prefill_mode_parse("safe") == QWEN4_PREFILL_SAFE);
    CHECK(qwen4_prefill_mode_parse("max") == QWEN4_PREFILL_MAX);
    CHECK(qwen4_prefill_mode_parse("MAX") == QWEN4_PREFILL_OFF);
    CHECK(qwen4_prefill_mode_parse("1") == QWEN4_PREFILL_OFF);
    CHECK(qwen4_prefill_mode_parse("safe ") == QWEN4_PREFILL_OFF);
}

static bool flags_eq(qwen4_prefill_flags f, bool r, bool p, bool n) {
    return f.residency == r && f.pipe == p && f.pipe_nocache == n;
}

static void check_policy(void) {
    const uint32_t lo = qwen4_moe_mm_min(), hi = qwen4_moe_mm_min() + 1u;
    CHECK(lo == 64u);
    /* off, NONE, too few rows, or not streaming: nothing */
    CHECK(flags_eq(qwen4_prefill_policy(QWEN4_PREFILL_OFF, QWEN4_PREFILL_ROLE_MIDDLE, 2048, true), 0, 0, 0));
    CHECK(flags_eq(qwen4_prefill_policy(QWEN4_PREFILL_MAX, QWEN4_PREFILL_ROLE_NONE, 2048, true), 0, 0, 0));
    CHECK(flags_eq(qwen4_prefill_policy(QWEN4_PREFILL_SAFE, QWEN4_PREFILL_ROLE_NONE, 2048, true), 0, 0, 0));
    CHECK(flags_eq(qwen4_prefill_policy(QWEN4_PREFILL_MAX, QWEN4_PREFILL_ROLE_MIDDLE, lo, true), 0, 0, 0));
    CHECK(flags_eq(qwen4_prefill_policy(QWEN4_PREFILL_MAX, QWEN4_PREFILL_ROLE_MIDDLE, 2048, false), 0, 0, 0));
    /* max: residency + pipe on every chunk of a prompt */
    CHECK(flags_eq(qwen4_prefill_policy(QWEN4_PREFILL_MAX, QWEN4_PREFILL_ROLE_MIDDLE, hi, true), 1, 1, 0));
    CHECK(flags_eq(qwen4_prefill_policy(QWEN4_PREFILL_MAX, QWEN4_PREFILL_ROLE_LAST, 2048, true), 1, 1, 0));
    /* safe: middle chunks only, pipe reads bypass the page cache */
    CHECK(flags_eq(qwen4_prefill_policy(QWEN4_PREFILL_SAFE, QWEN4_PREFILL_ROLE_MIDDLE, hi, true), 1, 1, 1));
    CHECK(flags_eq(qwen4_prefill_policy(QWEN4_PREFILL_SAFE, QWEN4_PREFILL_ROLE_LAST, 2048, true), 0, 0, 0));
}

static void check_prompt_end(void) {
    CHECK(!qwen4_prefill_prompt_ends(QWEN4_PREFILL_OFF, QWEN4_PREFILL_ROLE_LAST, true, true));
    CHECK(!qwen4_prefill_prompt_ends(QWEN4_PREFILL_MAX, QWEN4_PREFILL_ROLE_NONE, false, true));
    CHECK(!qwen4_prefill_prompt_ends(QWEN4_PREFILL_MAX, QWEN4_PREFILL_ROLE_LAST, true, false));
    CHECK(!qwen4_prefill_prompt_ends(QWEN4_PREFILL_MAX, QWEN4_PREFILL_ROLE_MIDDLE, true, true));
    /* the last chunk ends the prompt even when it was too small to stage */
    CHECK(qwen4_prefill_prompt_ends(QWEN4_PREFILL_SAFE, QWEN4_PREFILL_ROLE_LAST, true, true));
    CHECK(qwen4_prefill_prompt_ends(QWEN4_PREFILL_MAX, QWEN4_PREFILL_ROLE_LAST, true, true));
    /* a failed middle chunk abandons the prompt */
    CHECK(qwen4_prefill_prompt_ends(QWEN4_PREFILL_SAFE, QWEN4_PREFILL_ROLE_MIDDLE, false, true));
}

static void check_use_pipe(void) {
    const qwen4_prefill_flags on = {true, true, false}, off = {true, false, false};
    CHECK(qwen4_prefill_use_pipe(on, 0));
    CHECK(!qwen4_prefill_use_pipe(on, 64));   /* seeding needs the union path */
    CHECK(!qwen4_prefill_use_pipe(off, 0));
}

static void check_next_staged(void) {
    uint32_t list[16];
    for (uint32_t i = 0; i < 16; i++) list[i] = 32u + i;
    CHECK(qwen4_stream_next_staged(list, 16, 33, QWEN4_PREFILL_ROLE_MIDDLE) == 34);
    CHECK(qwen4_stream_next_staged(list, 16, 40, QWEN4_PREFILL_ROLE_LAST) == 41);
    CHECK(qwen4_stream_next_staged(list, 16, 47, QWEN4_PREFILL_ROLE_MIDDLE) == 32);   /* next chunk */
    CHECK(qwen4_stream_next_staged(list, 16, 47, QWEN4_PREFILL_ROLE_LAST) == UINT32_MAX);
    CHECK(qwen4_stream_next_staged(list, 16, 5, QWEN4_PREFILL_ROLE_MIDDLE) == UINT32_MAX);
    CHECK(qwen4_stream_next_staged(list, 0, 32, QWEN4_PREFILL_ROLE_MIDDLE) == UINT32_MAX);
    const uint32_t one[1] = {40};
    CHECK(qwen4_stream_next_staged(one, 1, 40, QWEN4_PREFILL_ROLE_MIDDLE) == 40);
    CHECK(qwen4_stream_next_staged(one, 1, 40, QWEN4_PREFILL_ROLE_LAST) == UINT32_MAX);
}

int main(void) {
    check_mode_parse();
    check_policy();
    check_prompt_end();
    check_use_pipe();
    check_next_staged();
    if (failures) {
        fprintf(stderr, "test_qwen4_prefill_pipe: %d failure(s)\n", failures);
        return 1;
    }
    puts("test_qwen4_prefill_pipe: PASS");
    return 0;
}
```

In `Makefile`, after the `test-qwen4-ngrams:` recipe (anchor `test-qwen4-ngrams: tests/test_qwen4_ngrams`), add:
```make
tests/test_qwen4_prefill_pipe.o: tests/test_qwen4_prefill_pipe.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -Wno-unused-function -I. -c -o $@ $<

tests/test_qwen4_prefill_pipe: tests/test_qwen4_prefill_pipe.o $(filter-out ds4.o,$(CORE_OBJS))
ifeq ($(UNAME_S),Darwin)
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -o $@ $^ $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
endif

.PHONY: test-qwen4-prefill-pipe
test-qwen4-prefill-pipe: tests/test_qwen4_prefill_pipe
	./tests/test_qwen4_prefill_pipe
```
In the `clean` recipe, next to `rm -f tests/test_qwen4_ngram_state`, add `rm -f tests/test_qwen4_prefill_pipe`.

In `speed-bench/qwen-regression/run.sh`, change
`make ds4-server test-qwen4-kernels test-qwen4-q2` to
`make ds4-server test-qwen4-kernels test-qwen4-q2 test-qwen4-prefill-pipe`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `make tests/test_qwen4_prefill_pipe 2>&1 | grep -E 'error' | head -5`
Expected: compile errors such as `use of undeclared identifier 'QWEN4_PREFILL_OFF'`.

- [ ] **Step 3: Implement the policy block**

In `ds4.c`, right after the closing `}` of `qwen4_stream_expert_cache_addr_layout_supported` (anchor:
`    return weights_streaming_layer_experts_uniform(w, il);\n}`), insert:
```c

/* Prefill residency and staging pipe (DS4_QWEN4_PREFILL_MODE=off|safe|max),
 * docs/superpowers/specs/2026-09-25-qwen4-prefill-residency-pipe-design.md.
 * off: as before. max: residency set on prefill command buffers and the
 * double-buffered staging pipe on every chunk of a prompt. safe: both on all
 * but the prompt's last chunk (pipe reads bypass the page cache), so decode
 * starts with the page cache the union path leaves behind. */
typedef enum {
    QWEN4_PREFILL_OFF = 0,
    QWEN4_PREFILL_SAFE,
    QWEN4_PREFILL_MAX,
} qwen4_prefill_mode;

/* Set only by the CLI and server prompt loops; every other forward is NONE. */
typedef enum {
    QWEN4_PREFILL_ROLE_NONE = 0,
    QWEN4_PREFILL_ROLE_MIDDLE,
    QWEN4_PREFILL_ROLE_LAST,
} qwen4_prefill_role;

typedef struct {
    bool residency;
    bool pipe;
    bool pipe_nocache;
} qwen4_prefill_flags;

/* Rows above which prefill MoE takes the tiled GEMMs (and a streamed layer is
 * staged); below it the per-token expert path wins. */
static uint32_t qwen4_moe_mm_min(void) {
#ifdef DS4_HAS_QWEN4_METAL
    return 64u;
#else
    return 8u;
#endif
}

static qwen4_prefill_mode qwen4_prefill_mode_parse(const char *s) {
    if (s && !strcmp(s, "safe")) return QWEN4_PREFILL_SAFE;
    if (s && !strcmp(s, "max")) return QWEN4_PREFILL_MAX;
    return QWEN4_PREFILL_OFF;
}

/* Read once per process, like the other qwen4 streaming knobs. The helpers
 * below are unused on builds without the Metal qwen4 graph. */
static DS4_MAYBE_UNUSED qwen4_prefill_mode qwen4_prefill_mode_env(void) {
    static int mode = -1;
    if (mode < 0) mode = (int)qwen4_prefill_mode_parse(getenv("DS4_QWEN4_PREFILL_MODE"));
    return (qwen4_prefill_mode)mode;
}

static DS4_MAYBE_UNUSED qwen4_prefill_flags qwen4_prefill_policy(qwen4_prefill_mode mode, qwen4_prefill_role role,
                                                                 uint32_t T, bool streaming) {
    qwen4_prefill_flags f = {false, false, false};
    if (mode == QWEN4_PREFILL_OFF || role == QWEN4_PREFILL_ROLE_NONE ||
        T <= qwen4_moe_mm_min() || !streaming) {
        return f;
    }
    if (mode == QWEN4_PREFILL_MAX) {
        f.residency = true;
        f.pipe = true;
    } else if (role == QWEN4_PREFILL_ROLE_MIDDLE) {
        f.residency = true;
        f.pipe = true;
        f.pipe_nocache = true;
    }
    return f;
}

/* True when this chunk ends the prompt for the pipe: the last chunk (staged
 * or not) or a failed one. The pipe then drops queued reads and its spare. */
static DS4_MAYBE_UNUSED bool qwen4_prefill_prompt_ends(qwen4_prefill_mode mode, qwen4_prefill_role role,
                                                       bool ok, bool streaming) {
    if (mode == QWEN4_PREFILL_OFF || role == QWEN4_PREFILL_ROLE_NONE || !streaming) return false;
    return role == QWEN4_PREFILL_ROLE_LAST || !ok;
}

/* Seeding the decode cache reads `selected` on the host, which only the
 * union path does. */
static DS4_MAYBE_UNUSED bool qwen4_prefill_use_pipe(qwen4_prefill_flags flags, uint32_t seed_tokens) {
    return flags.pipe && seed_tokens == 0;
}

/* The streamed layer staged after `layer` (UINT32_MAX: none): the next one in
 * `list`; after the last one, the first one again for the next chunk, unless
 * this chunk ends the prompt. */
static DS4_MAYBE_UNUSED uint32_t qwen4_stream_next_staged(const uint32_t *list, uint32_t n, uint32_t layer,
                                                          qwen4_prefill_role role) {
    for (uint32_t i = 0; i < n; i++) {
        if (list[i] != layer) continue;
        if (i + 1u < n) return list[i + 1u];
        return role == QWEN4_PREFILL_ROLE_LAST ? UINT32_MAX : list[0];
    }
    return UINT32_MAX;
}

/* Streamed layers in ascending order; returns how many (at most cap). */
static DS4_MAYBE_UNUSED uint32_t qwen4_stream_layer_list(const ds4_weights *w, uint32_t *out, uint32_t cap) {
    uint32_t n = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER && n < cap; il++) {
        if (qwen4_stream_expert_cache_addr_layout_supported(w, &w->layer[il], il)) out[n++] = il;
    }
    return n;
}
```

In `qwen4_graph_moe`, replace:
```c
#ifdef DS4_HAS_QWEN4_METAL
    const uint32_t mm_min = 64u;
#else
    const uint32_t mm_min = 8u;
#endif
```
with:
```c
    const uint32_t mm_min = qwen4_moe_mm_min();
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `make test-qwen4-prefill-pipe 2>&1 | tail -2 && make -j10 ds4 ds4-server 2>&1 | grep -E 'error|warning:'; echo "build exit ${pipestatus[1]}"`
Expected: `test_qwen4_prefill_pipe: PASS`, no warnings, `build exit 0`.

- [ ] **Step 5: Commit**

```bash
git add ds4.c tests/test_qwen4_prefill_pipe.c Makefile speed-bench/qwen-regression/run.sh
git commit -F - <<'EOF'
ds4: qwen4 prefill mode policy helpers and model-free test

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015npc1LHTBTYaTXLaJt8Uqc
EOF
```

---

### Task 2: Pipe bookkeeping header

**Files:**
- Create: `ds4_qwen4_stage_pipe.h`
- Modify: `tests/test_qwen4_prefill_pipe.c`, `Makefile` (object dependency)

**Interfaces:**
- Produces (all `static inline`):
  - enum `QSP_IDLE, QSP_QUEUED, QSP_DONE, QSP_FAILED`
  - `qsp_slot {int state; uint64_t seq; uint32_t layer; uint64_t off[3]; const void *map; bool nocache; bool reader_pending; bool has_reader;}`
  - `qsp_state {qsp_slot slot[2]; uint64_t seq;}`
  - `int qsp_find(const qsp_state *p, uint32_t layer, const uint64_t off[3], const void *map)`
  - `void qsp_queue(qsp_state *p, int i, uint32_t layer, const uint64_t off[3], const void *map, bool nocache)`
  - `int qsp_next_queued(const qsp_state *p)`
  - `bool qsp_any_queued(const qsp_state *p)`
  - `void qsp_finish(qsp_state *p, int i, bool ok)`
  - `void qsp_activate(qsp_state *p, int i)`
  - `unsigned qsp_note_commit(qsp_state *p)` (bitmask of the slots that took this commit as their last reader)
  - `void qsp_all_complete(qsp_state *p)`
  - `void qsp_invalidate(qsp_state *p, int i)`
  - `void qsp_prompt_end(qsp_state *p)`

- [ ] **Step 1: Write the failing test**

In `tests/test_qwen4_prefill_pipe.c`, after `#include "../ds4.c"`, add `#include "../ds4_qwen4_stage_pipe.h"`.
Before `int main(void) {`, add:
```c
static void check_pipe_order(void) {
    qsp_state p = {0};
    const uint64_t a[3] = {1, 2, 3}, b[3] = {4, 5, 6};
    int map;
    CHECK(qsp_next_queued(&p) == -1);
    qsp_queue(&p, 1, 33, a, &map, false);
    qsp_queue(&p, 0, 34, b, &map, true);
    CHECK(qsp_next_queued(&p) == 1);   /* queued first, served first */
    qsp_finish(&p, 1, true);
    CHECK(qsp_next_queued(&p) == 0);
    CHECK(p.slot[0].nocache && !p.slot[1].nocache);
    qsp_finish(&p, 0, false);
    CHECK(!qsp_any_queued(&p));
    CHECK(p.slot[0].state == QSP_FAILED && p.slot[1].state == QSP_DONE);
}

static void check_pipe_match(void) {
    qsp_state p = {0};
    const uint64_t a[3] = {1, 2, 3};
    uint64_t other[3] = {1, 2, 3};
    int map1, map2;
    CHECK(qsp_find(&p, 33, a, &map1) == -1);   /* idle slots never match */
    qsp_queue(&p, 0, 33, a, &map1, false);
    CHECK(qsp_find(&p, 33, a, &map1) == 0);    /* queued matches: the caller waits */
    CHECK(qsp_find(&p, 34, a, &map1) == -1);   /* other layer */
    CHECK(qsp_find(&p, 33, a, &map2) == -1);   /* other model map (engine reopened) */
    for (int t = 0; t < 3; t++) {
        other[t]++;
        CHECK(qsp_find(&p, 33, other, &map1) == -1);
        other[t]--;
    }
    qsp_finish(&p, 0, false);
    CHECK(qsp_find(&p, 33, a, &map1) == 0);    /* a failed read is found, so it is reread */
    qsp_invalidate(&p, 0);                     /* the union path overwrites slot 0 */
    CHECK(qsp_find(&p, 33, a, &map1) == -1);
}

static void check_pipe_readers(void) {
    qsp_state p = {0};
    const uint64_t a[3] = {1, 2, 3};
    int map;
    qsp_queue(&p, 0, 33, a, &map, false);
    qsp_finish(&p, 0, true);
    qsp_activate(&p, 0);
    CHECK(p.slot[0].state == QSP_IDLE);   /* consumed: no second match */
    CHECK(qsp_find(&p, 33, a, &map) == -1);
    CHECK(p.slot[0].reader_pending && !p.slot[0].has_reader);
    CHECK(qsp_note_commit(&p) == 1u);     /* the batch holding the GEMMs */
    CHECK(!p.slot[0].reader_pending && p.slot[0].has_reader);
    CHECK(qsp_note_commit(&p) == 0u);     /* a later commit does not move it */
    qsp_activate(&p, 1);
    CHECK(qsp_note_commit(&p) == 2u);
    qsp_all_complete(&p);
    CHECK(!p.slot[0].has_reader && !p.slot[1].has_reader);
    CHECK(!p.slot[0].reader_pending && !p.slot[1].reader_pending);
}

static void check_pipe_prompt_end(void) {
    qsp_state p = {0};
    const uint64_t a[3] = {1, 2, 3}, b[3] = {4, 5, 6};
    int map;
    /* an abandoned prompt: a finished wrap read and a consumed slot with a reader */
    qsp_queue(&p, 0, 32, a, &map, false);
    qsp_queue(&p, 1, 47, b, &map, false);
    qsp_finish(&p, 1, true);
    qsp_activate(&p, 1);
    (void)qsp_note_commit(&p);
    qsp_finish(&p, 0, true);
    qsp_prompt_end(&p);
    CHECK(!qsp_any_queued(&p));
    CHECK(qsp_find(&p, 32, a, &map) == -1 && qsp_find(&p, 47, b, &map) == -1);
    CHECK(!p.slot[0].has_reader && !p.slot[1].has_reader);
    CHECK(!p.slot[0].reader_pending && !p.slot[1].reader_pending);
}
```
In `main`, after `check_next_staged();`, add:
```c
    check_pipe_order();
    check_pipe_match();
    check_pipe_readers();
    check_pipe_prompt_end();
```
In `Makefile`, extend the object dependency line to
`tests/test_qwen4_prefill_pipe.o: tests/test_qwen4_prefill_pipe.c ds4.c ds4.h ds4_gpu.h ds4_qwen4_stage_pipe.h`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `make tests/test_qwen4_prefill_pipe 2>&1 | grep -E 'error' | head -3`
Expected: `fatal error: '../ds4_qwen4_stage_pipe.h' file not found`.

- [ ] **Step 3: Write the header**

Create `ds4_qwen4_stage_pipe.h`:
```c
/* Bookkeeping for the qwen4 prefill staging pipe: two whole-layer staging
 * buffers ("slots") and one background reader. Plain C so it can be tested
 * without Metal; ds4_metal.m owns the buffers, the reader thread, the command
 * buffers, and the lock held around every call here.
 *
 * A slot's job reads one streamed layer's gate/up/down tensors. A stage call
 * consumes a finished job (activate); the batch then encoded reads the slot,
 * and the first commit after that is the slot's last reader, which a later
 * read into the slot must wait for. */
#ifndef DS4_QWEN4_STAGE_PIPE_H
#define DS4_QWEN4_STAGE_PIPE_H

#include <stdbool.h>
#include <stdint.h>

enum { QSP_IDLE = 0, QSP_QUEUED, QSP_DONE, QSP_FAILED };

typedef struct {
    int state;
    uint64_t seq;           /* queue order */
    uint32_t layer;
    uint64_t off[3];        /* gate, up, down tensor offsets in the model */
    const void *map;        /* model map the offsets belong to */
    bool nocache;           /* read through the F_NOCACHE fd */
    bool reader_pending;    /* the open batch holds GEMMs reading this slot */
    bool has_reader;        /* a committed batch may still read this slot */
} qsp_slot;

typedef struct {
    qsp_slot slot[2];
    uint64_t seq;
} qsp_state;

/* The slot whose queued, finished or failed job holds this layer, or -1. */
static inline int qsp_find(const qsp_state *p, uint32_t layer, const uint64_t off[3], const void *map) {
    for (int i = 0; i < 2; i++) {
        const qsp_slot *s = &p->slot[i];
        if (s->state != QSP_IDLE && s->layer == layer && s->map == map &&
            s->off[0] == off[0] && s->off[1] == off[1] && s->off[2] == off[2]) {
            return i;
        }
    }
    return -1;
}

/* Queue a read of `layer` into slot i. */
static inline void qsp_queue(qsp_state *p, int i, uint32_t layer, const uint64_t off[3],
                             const void *map, bool nocache) {
    qsp_slot *s = &p->slot[i];
    s->state = QSP_QUEUED;
    s->seq = ++p->seq;
    s->layer = layer;
    s->off[0] = off[0];
    s->off[1] = off[1];
    s->off[2] = off[2];
    s->map = map;
    s->nocache = nocache;
}

/* The oldest queued slot, or -1: the reader serves jobs in queue order. */
static inline int qsp_next_queued(const qsp_state *p) {
    int best = -1;
    for (int i = 0; i < 2; i++) {
        if (p->slot[i].state == QSP_QUEUED &&
            (best < 0 || p->slot[i].seq < p->slot[best].seq)) {
            best = i;
        }
    }
    return best;
}

static inline bool qsp_any_queued(const qsp_state *p) {
    return qsp_next_queued(p) >= 0;
}

static inline void qsp_finish(qsp_state *p, int i, bool ok) {
    p->slot[i].state = ok ? QSP_DONE : QSP_FAILED;
}

/* Slot i is staged for the batch being encoded: its job is consumed and the
 * batch's next commit becomes its last reader. */
static inline void qsp_activate(qsp_state *p, int i) {
    p->slot[i].state = QSP_IDLE;
    p->slot[i].reader_pending = true;
}

/* A batch was committed. Returns the slots (bit i) that took it as their last
 * reader; a slot keeps the first commit after its activation. */
static inline unsigned qsp_note_commit(qsp_state *p) {
    unsigned mask = 0;
    for (int i = 0; i < 2; i++) {
        if (p->slot[i].reader_pending) {
            p->slot[i].reader_pending = false;
            p->slot[i].has_reader = true;
            mask |= 1u << i;
        }
    }
    return mask;
}

/* Everything committed so far has completed (the host waited): no readers. */
static inline void qsp_all_complete(qsp_state *p) {
    for (int i = 0; i < 2; i++) {
        p->slot[i].reader_pending = false;
        p->slot[i].has_reader = false;
    }
}

/* Drop slot i's job: its buffer is about to be overwritten or released. */
static inline void qsp_invalidate(qsp_state *p, int i) {
    p->slot[i].state = QSP_IDLE;
}

/* The prompt ended or was abandoned, and the caller waited for queued reads
 * and for the GPU: forget every job and reader. */
static inline void qsp_prompt_end(qsp_state *p) {
    qsp_all_complete(p);
    qsp_invalidate(p, 0);
    qsp_invalidate(p, 1);
}

#endif
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `make test-qwen4-prefill-pipe 2>&1 | tail -1`
Expected: `test_qwen4_prefill_pipe: PASS`.

- [ ] **Step 5: Commit**

```bash
git add ds4_qwen4_stage_pipe.h tests/test_qwen4_prefill_pipe.c Makefile
git commit -F - <<'EOF'
ds4: qwen4 staging pipe bookkeeping header and tests

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015npc1LHTBTYaTXLaJt8Uqc
EOF
```

---

### Task 3: Prefill residency scope, chunk roles, and GPU run diagnostics

**Files:**
- Modify: `ds4_metal.m`: residency block before `static id<MTLCommandBuffer> ds4_gpu_new_command_buffer(void) {`;
  view-generation bumps; `ds4_gpu_idle_note_cb`; `ds4_gpu_finish_command_buffer`; `ds4_gpu_cleanup`.
- Modify: `ds4_gpu.h` (next to `void ds4_gpu_qwen4_stream_stage_clear(void);`)
- Modify: `ds4.c`: graph struct, `qwen4_graph_forward_tokens`, CLI loop, server loop.
- Create: `speed-bench/qwen-prefill-pipe/make_prompts.py`, `speed-bench/qwen-prefill-pipe/cli_run.sh`,
  `speed-bench/qwen-prefill-pipe/cli_exact.sh`

**Interfaces:**
- Consumes: `qwen4_prefill_mode_env`, `qwen4_prefill_policy`, `qwen4_prefill_role` (Task 1).
- Produces:
  - `void ds4_gpu_prefill_residency_begin(void)`, `void ds4_gpu_prefill_residency_end(void)` (ds4_gpu.h);
  - graph field `qwen4_prefill_role prefill_role`;
  - `cli_run.sh OUT TAG PROMPT MODE [NGEN]` prints `TAG prefill P gen G md5 M`;
  - `cli_exact.sh OUT [PROMPTS...]` prints `PASS <prompt>`/`FAIL <prompt>` and exits non-zero on any difference;
  - with `DS4_METAL_GPU_IDLE=1`, at exit: `ds4: gpu run: N cbs, span X ms, busy Y ms, idle Z ms (P%), launch L ms over K waits`.

- [ ] **Step 1: Add the model-free check for the role default**

The residency scope itself needs a GPU. The model-free part is that a fresh graph starts with role `NONE`
(so eval, verify and decode paths stay `off`). In `tests/test_qwen4_prefill_pipe.c`, before `int main`, add:
```c
static void check_role_default(void) {
    ds4_qwen4_gpu_graph *g = xcalloc(1, sizeof(*g));
    CHECK(g->prefill_role == QWEN4_PREFILL_ROLE_NONE);
    free(g);
}
```
and call `check_role_default();` in `main`.

Run: `make tests/test_qwen4_prefill_pipe 2>&1 | grep error | head -2`
Expected: `no member named 'prefill_role'`.

- [ ] **Step 2: Add the graph field and set it in the prompt loops**

In `ds4.c`, in `ds4_qwen4_gpu_graph`, replace:
```c
    /* Set by the prefill loops for the prompt's last chunk: staged streamed
     * layers then seed the decode expert cache from its final tokens. */
    bool stream_seed_last;
} ds4_qwen4_gpu_graph;
```
with:
```c
    /* Set by the prefill loops for the prompt's last chunk: staged streamed
     * layers then seed the decode expert cache from its final tokens. */
    bool stream_seed_last;
    /* DS4_QWEN4_PREFILL_MODE: role of the chunk being run, set only by the CLI
     * and server prompt loops (NONE everywhere else), and the streamed layers
     * in ascending order, filled on first use by the staging pipe. */
    qwen4_prefill_role prefill_role;
    uint32_t stream_layers[DS4_MAX_LAYER];
    uint32_t n_stream_layers;
    bool stream_layers_ready;
} ds4_qwen4_gpu_graph;
```

CLI loop. Replace:
```c
        g->stream_seed_last = i + (int)chunk == prompt->len;
        ok = qwen4_graph_forward_tokens(g, model, weights, prompt->v + i, chunk,
                                        i + (int)chunk == prompt->len ? logits : NULL, false);
        g->stream_seed_last = false;
```
with:
```c
        g->stream_seed_last = i + (int)chunk == prompt->len;
        g->prefill_role = g->stream_seed_last ? QWEN4_PREFILL_ROLE_LAST : QWEN4_PREFILL_ROLE_MIDDLE;
        ok = qwen4_graph_forward_tokens(g, model, weights, prompt->v + i, chunk,
                                        i + (int)chunk == prompt->len ? logits : NULL, false);
        g->stream_seed_last = false;
        g->prefill_role = QWEN4_PREFILL_ROLE_NONE;
```

Server loop. Replace:
```c
            s->qwen4_graph.stream_seed_last = i + (int)chunk == prompt->len;
            const bool chunk_ok = qwen4_graph_forward_tokens(&s->qwen4_graph, &e->model, &e->weights,
                                                             prompt->v + i, chunk, s->logits, false);
            s->qwen4_graph.stream_seed_last = false;
```
with:
```c
            s->qwen4_graph.stream_seed_last = i + (int)chunk == prompt->len;
            s->qwen4_graph.prefill_role = s->qwen4_graph.stream_seed_last ? QWEN4_PREFILL_ROLE_LAST
                                                                          : QWEN4_PREFILL_ROLE_MIDDLE;
            const bool chunk_ok = qwen4_graph_forward_tokens(&s->qwen4_graph, &e->model, &e->weights,
                                                             prompt->v + i, chunk, s->logits, false);
            s->qwen4_graph.stream_seed_last = false;
            s->qwen4_graph.prefill_role = QWEN4_PREFILL_ROLE_NONE;
```

Run: `make test-qwen4-prefill-pipe 2>&1 | tail -1`
Expected: `test_qwen4_prefill_pipe: PASS`.

- [ ] **Step 3: Residency API in `ds4_gpu.h` and `ds4_metal.m`**

In `ds4_gpu.h`, after `void ds4_gpu_qwen4_stream_stage_clear(void);`, add:
```c
/* qwen4 prefill residency (DS4_QWEN4_PREFILL_MODE): while the scope is open,
 * new command buffers use a residency set of the mapped model views. A no-op
 * before macOS 15 or when the set cannot be built. */
void ds4_gpu_prefill_residency_begin(void);
void ds4_gpu_prefill_residency_end(void);
```

In `ds4_metal.m`, after `static uint32_t g_model_view_count;`, add:
```objc
static uint64_t g_model_view_gen;   /* bumped whenever the model views change */
```
In `ds4_gpu_model_views_clear`, after `g_model_view_count = 0;`, add `g_model_view_gen++;`.
In `ds4_gpu_model_views_remove_map`, after `g_model_view_count = kept;`, add `g_model_view_gen++;`.
In `ds4_gpu_add_model_view_range`, after `g_model_view_count++;` (unique), add `g_model_view_gen++;`.

Immediately before `static id<MTLCommandBuffer> ds4_gpu_new_command_buffer(void) {`, add:
```objc
/* qwen4 prefill residency (DS4_QWEN4_PREFILL_MODE). Under --ssd-streaming the
 * model views are in no residency set, so the driver revalidates them lazily
 * and a command buffer can start hundreds of ms after its commit (17 s of a
 * 36K-token prefill). While a prefill scope is open, new command buffers use a
 * set of the mapped views. The set is only committed, never requested and never
 * added to the queue, so decode and every other command buffer run as before. */
static id g_prefill_residency_set;
static uint64_t g_prefill_residency_gen = UINT64_MAX;
static int g_prefill_residency_scope;
static int g_prefill_residency_warned;

static id ds4_gpu_prefill_residency_set(void) {
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        if (g_prefill_residency_set && g_prefill_residency_gen == g_model_view_gen) {
            return g_prefill_residency_set;
        }
        g_prefill_residency_set = nil;
        if (!g_device || g_model_view_count == 0) return nil;
        MTLResidencySetDescriptor *desc = [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"ds4_qwen4_prefill";
        desc.initialCapacity = g_model_view_count;
        NSError *error = nil;
        id<MTLResidencySet> set = [g_device newResidencySetWithDescriptor:desc error:&error];
        if (!set) {
            if (!g_prefill_residency_warned) {
                g_prefill_residency_warned = 1;
                fprintf(stderr, "ds4: qwen4 prefill residency set creation failed: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
            return nil;
        }
        for (uint32_t i = 0; i < g_model_view_count; i++) {
            [set addAllocation:g_model_views[i].buffer];
        }
        [set commit];
        g_prefill_residency_set = set;
        g_prefill_residency_gen = g_model_view_gen;
        return set;
    }
#endif
    return nil;
}

void ds4_gpu_prefill_residency_begin(void) {
    g_prefill_residency_scope = g_initialized && ds4_gpu_prefill_residency_set() != nil;
}

void ds4_gpu_prefill_residency_end(void) {
    g_prefill_residency_scope = 0;
}

static void ds4_gpu_prefill_residency_attach(id<MTLCommandBuffer> cb) {
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        if (cb && g_prefill_residency_scope && g_prefill_residency_set &&
            [cb respondsToSelector:@selector(useResidencySet:)]) {
            [cb useResidencySet:g_prefill_residency_set];
        }
    }
#endif
}
```
Replace the body tail of `ds4_gpu_new_command_buffer`:
```objc
    if (use_unretained) {
        return [g_queue commandBufferWithUnretainedReferences];
    }
    return [g_queue commandBuffer];
}
```
with:
```objc
    id<MTLCommandBuffer> cb = use_unretained ? [g_queue commandBufferWithUnretainedReferences]
                                             : [g_queue commandBuffer];
    ds4_gpu_prefill_residency_attach(cb);
    return cb;
}
```
In `ds4_gpu_cleanup`, after `(void)ds4_gpu_wait_pending_command_buffers("cleanup");`, add:
```objc
        g_prefill_residency_scope = 0;
        g_prefill_residency_set = nil;
        g_prefill_residency_gen = UINT64_MAX;
```

- [ ] **Step 4: Open and close the scope in `qwen4_graph_forward_tokens`**

In `ds4.c`, replace:
```c
    if (!qwen4_graph_stage_inputs(g, m, w, tokens, T)) return false;
    const double t1 = timing ? now_sec() : 0.0;
    if (!glm_graph_begin_commands_if_needed()) return false;
```
with:
```c
    if (!qwen4_graph_stage_inputs(g, m, w, tokens, T)) return false;
    const double t1 = timing ? now_sec() : 0.0;
#ifdef DS4_HAS_QWEN4_METAL
    const qwen4_prefill_mode pmode = qwen4_prefill_mode_env();
    const bool pstreaming = ds4_gpu_ssd_streaming_enabled();
    const qwen4_prefill_flags pflags = qwen4_prefill_policy(pmode, g->prefill_role, T, pstreaming);
    if (pflags.residency) ds4_gpu_prefill_residency_begin();
#endif
    if (!glm_graph_begin_commands_if_needed()) {
#ifdef DS4_HAS_QWEN4_METAL
        if (pflags.residency) ds4_gpu_prefill_residency_end();
#endif
        return false;
    }
```
and replace:
```c
    if (!ds4_gpu_end_commands()) ok = false;
    const double t3 = timing ? now_sec() : 0.0;
    if (T == 1u) qwen4_router_flush(pos0);
```
with:
```c
    if (!ds4_gpu_end_commands()) ok = false;
#ifdef DS4_HAS_QWEN4_METAL
    if (pflags.residency) ds4_gpu_prefill_residency_end();
#endif
    const double t3 = timing ? now_sec() : 0.0;
    if (T == 1u) qwen4_router_flush(pos0);
```

- [ ] **Step 5: Whole-run GPU diagnostics under `DS4_METAL_GPU_IDLE`**

In `ds4_metal.m`, immediately before `static void ds4_gpu_idle_note_cb(id<MTLCommandBuffer> cb) {`, add:
```objc
/* Whole-run accounting (DS4_METAL_GPU_IDLE), printed once at exit: the union
 * of GPU busy intervals over the first-start..last-end span, and the launch
 * latency (commit to GPUStartTime) of waited command buffers that had nothing
 * queued ahead of them. This is how the prefill launch latency was found. */
static double g_gpu_run_first, g_gpu_run_last_end, g_gpu_run_busy, g_gpu_run_launch;
static uint64_t g_gpu_run_cbs, g_gpu_run_launch_n;

static void ds4_gpu_run_report(void) {
    const double span = g_gpu_run_last_end - g_gpu_run_first;
    fprintf(stderr, "ds4: gpu run: %llu cbs, span %.1f ms, busy %.1f ms, idle %.1f ms (%.1f%%), "
            "launch %.1f ms over %llu waits\n",
            (unsigned long long)g_gpu_run_cbs, span * 1e3, g_gpu_run_busy * 1e3,
            (span - g_gpu_run_busy) * 1e3, span > 0.0 ? 100.0 * (span - g_gpu_run_busy) / span : 0.0,
            g_gpu_run_launch * 1e3, (unsigned long long)g_gpu_run_launch_n);
}
```
At the top of `ds4_gpu_idle_note_cb`, after `if (en <= st) return;`, add:
```objc
    if (g_gpu_run_cbs++ == 0) {
        g_gpu_run_first = st;
        g_gpu_run_last_end = st;
        atexit(ds4_gpu_run_report);
    }
    if (en > g_gpu_run_last_end) {
        g_gpu_run_busy += en - (st > g_gpu_run_last_end ? st : g_gpu_run_last_end);
        g_gpu_run_last_end = en;
    }
```
In `ds4_gpu_finish_command_buffer`, replace:
```objc
    const double t_commit = ds4_gpu_now_ms();
    [cb commit];
```
with:
```objc
    const double t_commit = ds4_gpu_now_ms();
    const double h_commit = ds4_gpu_host_seconds();
    const BOOL had_pending = [g_pending_cbs count] != 0;
    [cb commit];
```
and replace:
```objc
    if (g_gpu_idle_prof > 0) {
        ds4_gpu_idle_note_cb(cb);
        ds4_gpu_idle_end_group();
    }
```
with:
```objc
    if (g_gpu_idle_prof > 0) {
        ds4_gpu_idle_note_cb(cb);
        ds4_gpu_idle_end_group();
        if (!had_pending && cb.GPUStartTime > 0.0) {
            g_gpu_run_launch += cb.GPUStartTime - h_commit;
            g_gpu_run_launch_n++;
        }
    }
```

- [ ] **Step 6: Build and run the model-free test**

Run: `make -j10 ds4 ds4-server 2>&1 | grep -E 'error|warning:'; echo "exit ${pipestatus[1]}"; make test-qwen4-prefill-pipe 2>&1 | tail -1`
Expected: no lines, `exit 0`, `test_qwen4_prefill_pipe: PASS`.

- [ ] **Step 7: Add the CLI check tools**

Create `speed-bench/qwen-prefill-pipe/make_prompts.py`:
```python
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
```
Create `speed-bench/qwen-prefill-pipe/cli_run.sh` (then `chmod +x`):
```bash
#!/bin/bash
# One CLI prefill + decode run with the PROD ds4 flags (gateway registry, 2026-09-24).
# usage: cli_run.sh OUT_DIR TAG PROMPT_FILE MODE [NGEN]
# MODE is DS4_QWEN4_PREFILL_MODE (off|safe|max); other env (DS4_METAL_GPU_IDLE, ...)
# passes through. Prints "TAG prefill P gen G md5 M". Needs the machine free.
set -euo pipefail
out=$1 tag=$2 prompt=$3 mode=$4 ngen=${5:-300}
root=$(cd "$(dirname "$0")/../.." && pwd)
models=${DS4_MODELS_DIR:-$HOME/.local/share/ai-gateway/ds4-models}
mkdir -p "$out"
cd "$root"
env DS4_QWEN4_STREAM_FULL_LAYERS=32 DS4_QWEN4_PLE_PREFETCH_FULL=0 \
    DS4_QWEN4_MTP_DRAFT_VOCAB="$models/Qwen3.8-Flash-Next-draft-vocab-vi-en-code-64k.txt" \
    DS4_QWEN4_KV_GROW=1 DS4_QWEN4_PREFILL_MODE="$mode" \
    ./ds4 --metal \
    -m "$models/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q4KDownPad768-DenseQ4Kselimat-MTP.gguf" \
    --ple "$models/Qwen3.8-Flash-Next-PLE-Q4_1.gguf" \
    -c 262144 --prefill-chunk 2048 --mtp --ssd-streaming --ssd-streaming-cache-experts 6GB \
    --temp 0 -n "$ngen" --prompt-file "$prompt" \
    > "$out/$tag.out" 2> "$out/$tag.err"
speeds=$(grep -a 'prefill:' "$out/$tag.err" | tail -1 |
         sed -E 's/.*prefill: ([0-9.]+) t\/s, generation: ([0-9.]+) t\/s.*/prefill \1 gen \2/')
echo "$tag $speeds md5 $(md5 -q "$out/$tag.out")"
```
Create `speed-bench/qwen-prefill-pipe/cli_exact.sh` (then `chmod +x`):
```bash
#!/bin/bash
# Byte-identical check across DS4_QWEN4_PREFILL_MODE=off|safe|max.
# usage: cli_exact.sh OUT_DIR [PROMPT...]   (default: chat p5k p36k p134k)
# Exits non-zero if any prompt's replies differ between modes. Needs the machine free.
set -euo pipefail
out=$1
shift
here=$(cd "$(dirname "$0")" && pwd)
python3 "$here/make_prompts.py" "$out/prompts"
prompts=("$@")
[ ${#prompts[@]} -eq 0 ] && prompts=(chat p5k p36k p134k)
fail=0
for p in "${prompts[@]}"; do
    sums=()
    for mode in off safe max; do
        line=$("$here/cli_run.sh" "$out" "$p-$mode" "$out/prompts/$p.txt" "$mode" 300)
        echo "$line"
        sums+=("${line##* }")
    done
    if [ "$(printf '%s\n' "${sums[@]}" | sort -u | wc -l)" -ne 1 ]; then
        echo "FAIL $p: replies differ between modes"
        fail=1
    else
        echo "PASS $p"
    fi
done
exit $fail
```

- [ ] **Step 8: [GPU] Residency exactness and launch latency**

Stop oMLX and the watchdogs (see GPU Steps). Run:
```bash
O=/tmp/claude-501/qpp-task3
speed-bench/qwen-prefill-pipe/cli_exact.sh $O chat p36k
for m in off max; do DS4_METAL_GPU_IDLE=1 speed-bench/qwen-prefill-pipe/cli_run.sh $O idle-$m $O/prompts/p36k.txt $m 16; grep -a 'gpu run:' $O/idle-$m.err; done
```
Expected:
- `PASS chat` and `PASS p36k`. `max` has no pipe yet, only residency; `safe` uses residency on middle chunks.
- The `gpu run:` launch total is roughly 10-20 s for `off` and under 1 s for `max`.
- Prefill for `max` is at least 10 % above `off`.

Restore the services. Record the numbers for Task 6's RESULTS.

- [ ] **Step 9: Commit**

```bash
chmod +x speed-bench/qwen-prefill-pipe/cli_run.sh speed-bench/qwen-prefill-pipe/cli_exact.sh
git add ds4.c ds4_gpu.h ds4_metal.m tests/test_qwen4_prefill_pipe.c speed-bench/qwen-prefill-pipe
git commit -F - <<'EOF'
ds4: qwen4 prefill residency scope and GPU run diagnostics

DS4_QWEN4_PREFILL_MODE=safe|max attach a residency set of the mapped model
views to prefill command buffers under --ssd-streaming, removing the lazy
revalidation that delayed command-buffer starts by up to ~0.5 s. The set is
never requested nor added to the queue. DS4_METAL_GPU_IDLE now also prints a
whole-run busy/idle/launch line at exit. CLI check tools under
speed-bench/qwen-prefill-pipe/.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015npc1LHTBTYaTXLaJt8Uqc
EOF
```

---

### Task 4: Drain-free double-buffered staging pipe

**Files:**
- Modify: `ds4_metal.m`:
  - `#include "ds4_qwen4_stage_pipe.h"`;
  - forward declarations near `static id<MTLCommandBuffer> ds4_gpu_new_command_buffer(void);` (anchor at ~1287);
  - `ds4_gpu_flush_commands` and `ds4_gpu_end_commands` hooks;
  - `ds4_gpu_stream_expert_pread_into` split;
  - `g_qwen4_stage` gets `bind_buf`, used by `qwen4_bind_weight`;
  - `ds4_gpu_qwen4_stream_stage_layer` split into `qwen4_stage_union` plus a wrapper;
  - the pipe module after `ds4_gpu_qwen4_stream_stage_clear`;
  - `ds4_gpu_cleanup`.
- Modify: `ds4_gpu.h`, `ds4.c` (`qwen4_graph_moe`, `qwen4_graph_forward_tokens`, server cancellation branch)

**Interfaces:**
- Consumes: `qsp_*` (Task 2); `qwen4_prefill_policy`, `qwen4_prefill_use_pipe`, `qwen4_prefill_prompt_ends`,
  `qwen4_stream_next_staged`, `qwen4_stream_layer_list` (Task 1); graph fields `prefill_role`, `stream_layers`,
  `n_stream_layers`, `stream_layers_ready` (Task 3).
- Produces (ds4_gpu.h):
  - `int ds4_gpu_qwen4_stream_stage_layer_pipe(const void *model_map, uint64_t model_size, uint32_t layer, const ds4_gpu_tensor *selected, uint32_t n_tokens, uint32_t n_slots, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint32_t n_expert, uint32_t in_dim, uint32_t ff_dim, uint32_t out_dim, uint32_t next_layer, uint64_t next_gate_offset, uint64_t next_up_offset, uint64_t next_down_offset, uint32_t next_gate_type, uint32_t next_down_type, int nocache)`
  - `void ds4_gpu_qwen4_stream_stage_prompt_end(void)`

- [ ] **Step 1: Declarations in `ds4_gpu.h`**

After the residency declarations from Task 3, add:
```c
/* qwen4 prefill staging pipe (DS4_QWEN4_PREFILL_MODE): stage `layer` from a
 * whole-layer read queued by the previous call, without draining the GPU, and
 * queue the read of next_layer (UINT32_MAX: none) into the other buffer. Falls
 * back to ds4_gpu_qwen4_stream_stage_layer when nothing was queued for `layer`
 * or the read failed. The kernels see the same bytes at the same offsets. */
int ds4_gpu_qwen4_stream_stage_layer_pipe(
        const void *model_map, uint64_t model_size, uint32_t layer,
        const ds4_gpu_tensor *selected, uint32_t n_tokens, uint32_t n_slots,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint32_t gate_type, uint32_t down_type,
        uint32_t n_expert, uint32_t in_dim, uint32_t ff_dim, uint32_t out_dim,
        uint32_t next_layer, uint64_t next_gate_offset, uint64_t next_up_offset,
        uint64_t next_down_offset, uint32_t next_gate_type, uint32_t next_down_type,
        int nocache);
/* The prompt ended or was abandoned: wait for queued reads, forget them, and
 * release the spare staging buffer. No-op until the pipe has run. */
void ds4_gpu_qwen4_stream_stage_prompt_end(void);
```

- [ ] **Step 2: Split the pread helper so a caller can pick the fd**

In `ds4_metal.m`, rename `static int ds4_gpu_stream_expert_pread_into(` to
`static int ds4_gpu_stream_expert_pread_fd(` and add `int fd,` as its first parameter. Inside it:
- replace `if (g_model_fd < 0 ||` with `if (fd < 0 ||`;
- replace `nread = pread(g_model_fd, dst + pos, want, (off_t)(offset + pos));` with
  `nread = pread(fd, dst + pos, want, (off_t)(offset + pos));`.

Right after that function, add the old name back as a wrapper:
```objc
static int ds4_gpu_stream_expert_pread_into(
        uint64_t  offset,
        uint64_t  len,
        uint8_t  *dst,
        uint64_t *read_bytes,
        double   *ms_out) {
    return ds4_gpu_stream_expert_pread_fd(g_model_fd, offset, len, dst, read_bytes, ms_out);
}
```

- [ ] **Step 3: Separate the bind buffer from the union buffer**

In the `g_qwen4_stage` struct, after `id<MTLBuffer> buf;`, add:
```objc
    id<MTLBuffer> bind_buf;   /* buffer the staged tensors bind to (buf or the pipe spare) */
```
In `qwen4_bind_weight`, replace `b->buf = g_qwen4_stage.buf;` with `b->buf = g_qwen4_stage.bind_buf;`.

- [ ] **Step 4: Turn the union path into `qwen4_stage_union` plus a wrapper**

Rename the existing definition `int ds4_gpu_qwen4_stream_stage_layer(` to `static int qwen4_stage_union(`, keeping
its parameter list. In its activation block, replace:
```objc
    if (ok) {
        g_qwen4_stage.map = model_map;
```
with:
```objc
    if (ok) {
        g_qwen4_stage.bind_buf = g_qwen4_stage.buf;
        g_qwen4_stage.map = model_map;
```
Make the pipe module (Step 5) and the new public wrapper (Step 6) follow `qwen4_stage_union` in the file.

- [ ] **Step 5: The pipe module**

Add `#include "ds4_qwen4_stage_pipe.h"` next to the other local includes at the top of `ds4_metal.m`. Next to the
forward declaration `static id<MTLCommandBuffer> ds4_gpu_new_command_buffer(void);`, add:
```objc
static void ds4_gpu_qpipe_note_commit(id<MTLCommandBuffer> cb);
static void ds4_gpu_qpipe_note_all_complete(void);
static void ds4_gpu_qpipe_shutdown(void);
```
Right after the closing `}` of `qwen4_stage_union`, add the pipe module:
```objc
/* qwen4 prefill staging pipe (DS4_QWEN4_PREFILL_MODE). Slot 0 is the union
 * path's buffer (g_qwen4_stage.buf); slot 1 is a spare allocated on demand and
 * released at prompt end. One reader thread serves queued whole-layer reads in
 * order; before reading into a slot it waits for the command buffer that last
 * read it (per-slot last reader, recorded by the flush hook). Bookkeeping in
 * ds4_qwen4_stage_pipe.h; every qsp_* call and slot pointer is under g_qpipe_mu. */
#define QPIPE_READERS 8u
#define QPIPE_PIECE (32ull << 20)

static pthread_mutex_t g_qpipe_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_qpipe_cv = PTHREAD_COND_INITIALIZER;
static qsp_state g_qpipe;
static id<MTLBuffer> g_qpipe_spare;
static id<MTLCommandBuffer> g_qpipe_wait_cb[2];
static id<MTLCommandBuffer> g_qpipe_reader_cb[2];
static uint64_t g_qpipe_bytes[3];
static NSUInteger g_qpipe_region[3];
static uint64_t g_qpipe_need;
static double g_qpipe_read_ms[2], g_qpipe_cbwait_ms[2];
static pthread_t g_qpipe_thread;
static int g_qpipe_started;     /* the pipe has run: hooks are live */
static int g_qpipe_stop;
static int g_qpipe_disabled;    /* an allocation failed: union path only */
static int g_qpipe_nocache_fd = -1;

static id<MTLBuffer> qpipe_buffer(int s) {
    return s == 0 ? g_qwen4_stage.buf : g_qpipe_spare;
}

/* Reader thread only: the model file through a second fd that bypasses the
 * page cache, or the shared fd when that cannot be opened. */
static int qpipe_nocache_fd(void) {
    if (g_qpipe_nocache_fd < 0 && g_model_fd >= 0) {
        char path[MAXPATHLEN];
        if (fcntl(g_model_fd, F_GETPATH, path) != -1) {
            const int fd = open(path, O_RDONLY | O_CLOEXEC);
            if (fd >= 0 && fcntl(fd, F_NOCACHE, 1) != -1) g_qpipe_nocache_fd = fd;
            else if (fd >= 0) close(fd);
        }
    }
    return g_qpipe_nocache_fd >= 0 ? g_qpipe_nocache_fd : g_model_fd;
}

typedef struct {
    ds4_gpu_stream_expert_pread_task *tasks;
    uint32_t n, first, stride;
    int fd;
} qpipe_reader_args;

static void *qpipe_reader_main(void *arg) {
    qpipe_reader_args *a = (qpipe_reader_args *)arg;
    ds4_gpu_stream_expert_pread_thread_qos();
    for (uint32_t i = a->first; i < a->n; i += a->stride) {
        ds4_gpu_stream_expert_pread_task *t = &a->tasks[i];
        t->ok = ds4_gpu_stream_expert_pread_fd(a->fd, t->offset, t->len, t->dst, &t->read_bytes, &t->ms);
    }
    return NULL;
}

/* Read one streamed layer's three whole tensors into base at the union path's
 * regions, in QPIPE_PIECE pieces over QPIPE_READERS threads of our own (the
 * shared pread pool stays free for the main thread). */
static int qpipe_read_layer(uint8_t *base, const uint64_t off[3], int fd) {
    uint32_t n = 0;
    for (int t = 0; t < 3; t++) n += (uint32_t)((g_qpipe_bytes[t] + QPIPE_PIECE - 1u) / QPIPE_PIECE);
    ds4_gpu_stream_expert_pread_task *tasks = (ds4_gpu_stream_expert_pread_task *)calloc(n, sizeof(*tasks));
    if (!tasks) return 0;
    n = 0;
    for (int t = 0; t < 3; t++) {
        for (uint64_t p = 0; p < g_qpipe_bytes[t]; p += QPIPE_PIECE) {
            const uint64_t len = g_qpipe_bytes[t] - p < QPIPE_PIECE ? g_qpipe_bytes[t] - p : QPIPE_PIECE;
            tasks[n++] = (ds4_gpu_stream_expert_pread_task){
                .offset = off[t] + p, .len = len, .dst = base + g_qpipe_region[t] + p };
        }
    }
    pthread_t th[QPIPE_READERS];
    qpipe_reader_args args[QPIPE_READERS];
    int live[QPIPE_READERS] = {0};
    for (uint32_t i = 0; i < QPIPE_READERS; i++) args[i] = (qpipe_reader_args){ tasks, n, i, QPIPE_READERS, fd };
    for (uint32_t i = 1; i < QPIPE_READERS; i++) {
        live[i] = pthread_create(&th[i], NULL, qpipe_reader_main, &args[i]) == 0;
    }
    qpipe_reader_main(&args[0]);
    for (uint32_t i = 1; i < QPIPE_READERS; i++) {
        if (live[i]) pthread_join(th[i], NULL);
        else qpipe_reader_main(&args[i]);   /* a thread failed to start: run its share here */
    }
    int ok = 1;
    for (uint32_t i = 0; i < n; i++) if (!tasks[i].ok) ok = 0;
    free(tasks);
    return ok;
}

static void *qpipe_thread_main(void *arg) {
    (void)arg;
    ds4_gpu_stream_expert_pread_thread_qos();
    pthread_mutex_lock(&g_qpipe_mu);
    for (;;) {
        int s;
        while (!g_qpipe_stop && (s = qsp_next_queued(&g_qpipe)) < 0) {
            pthread_cond_wait(&g_qpipe_cv, &g_qpipe_mu);
        }
        if (g_qpipe_stop) break;
        @autoreleasepool {
            id<MTLCommandBuffer> wait_cb = g_qpipe_wait_cb[s];
            g_qpipe_wait_cb[s] = nil;
            const uint64_t off[3] = { g_qpipe.slot[s].off[0], g_qpipe.slot[s].off[1], g_qpipe.slot[s].off[2] };
            const bool nocache = g_qpipe.slot[s].nocache;
            uint8_t *base = (uint8_t *)[qpipe_buffer(s) contents];
            pthread_mutex_unlock(&g_qpipe_mu);
            const double t0 = ds4_gpu_now_ms();
            if (wait_cb) [wait_cb waitUntilCompleted];
            wait_cb = nil;
            const double t1 = ds4_gpu_now_ms();
            const int ok = base && qpipe_read_layer(base, off, nocache ? qpipe_nocache_fd() : g_model_fd);
            const double t2 = ds4_gpu_now_ms();
            pthread_mutex_lock(&g_qpipe_mu);
            qsp_finish(&g_qpipe, s, ok != 0);
            g_qpipe_cbwait_ms[s] = t1 - t0;
            g_qpipe_read_ms[s] = t2 - t1;
            pthread_cond_broadcast(&g_qpipe_cv);
        }
    }
    pthread_mutex_unlock(&g_qpipe_mu);
    return NULL;
}

/* Main thread, lock not held. */
static int qpipe_start(void) {
    if (g_qpipe_started) return 1;
    if (pthread_create(&g_qpipe_thread, NULL, qpipe_thread_main, NULL) != 0) return 0;
    g_qpipe_started = 1;
    return 1;
}

/* Wait until no read is queued (lock held). */
static void qpipe_wait_idle_locked(void) {
    while (qsp_any_queued(&g_qpipe)) pthread_cond_wait(&g_qpipe_cv, &g_qpipe_mu);
}

/* Flush hook: the committed batch becomes the last reader of every slot whose
 * GEMMs it carries. */
static void ds4_gpu_qpipe_note_commit(id<MTLCommandBuffer> cb) {
    if (!g_qpipe_started) return;
    pthread_mutex_lock(&g_qpipe_mu);
    const unsigned mask = qsp_note_commit(&g_qpipe);
    for (int i = 0; i < 2; i++) if (mask & (1u << i)) g_qpipe_reader_cb[i] = cb;
    pthread_mutex_unlock(&g_qpipe_mu);
}

/* end_commands hook: the host waited for everything committed. */
static void ds4_gpu_qpipe_note_all_complete(void) {
    if (!g_qpipe_started) return;
    pthread_mutex_lock(&g_qpipe_mu);
    qsp_all_complete(&g_qpipe);
    g_qpipe_reader_cb[0] = nil;
    g_qpipe_reader_cb[1] = nil;
    pthread_mutex_unlock(&g_qpipe_mu);
}

/* Queue the read of next_layer into slot s behind s's last reader. Main
 * thread, lock not held; the spare is allocated here when s is 1. */
static void qpipe_queue(int s, uint32_t next_layer, const uint64_t next_off[3], const void *map, int nocache) {
    if (s == 1 && !g_qpipe_spare) {
        id<MTLBuffer> spare = [g_device newBufferWithLength:(NSUInteger)g_qpipe_need
                                                    options:MTLResourceStorageModeShared];
        if (!spare) {
            g_qpipe_disabled = 1;
            fprintf(stderr, "ds4: qwen4 stage pipe spare buffer allocation failed (%.2f GiB); "
                    "staging through the union path\n", ds4_gpu_gib(g_qpipe_need));
            return;
        }
        pthread_mutex_lock(&g_qpipe_mu);
        g_qpipe_spare = spare;
        pthread_mutex_unlock(&g_qpipe_mu);
    }
    pthread_mutex_lock(&g_qpipe_mu);
    if (g_qpipe.slot[s].state != QSP_QUEUED) {
        g_qpipe_wait_cb[s] = g_qpipe.slot[s].has_reader ? g_qpipe_reader_cb[s] : nil;
        qsp_queue(&g_qpipe, s, next_layer, next_off, map, nocache != 0);
        pthread_cond_broadcast(&g_qpipe_cv);
    }
    pthread_mutex_unlock(&g_qpipe_mu);
}

void ds4_gpu_qwen4_stream_stage_prompt_end(void) {
    if (!g_qpipe_started) return;
    pthread_mutex_lock(&g_qpipe_mu);
    qpipe_wait_idle_locked();
    qsp_prompt_end(&g_qpipe);
    g_qpipe_wait_cb[0] = g_qpipe_wait_cb[1] = nil;
    g_qpipe_reader_cb[0] = g_qpipe_reader_cb[1] = nil;
    g_qpipe_spare = nil;
    pthread_mutex_unlock(&g_qpipe_mu);
}

static void ds4_gpu_qpipe_shutdown(void) {
    if (g_qpipe_started) {
        pthread_mutex_lock(&g_qpipe_mu);
        g_qpipe_stop = 1;
        pthread_cond_broadcast(&g_qpipe_cv);
        pthread_mutex_unlock(&g_qpipe_mu);
        pthread_join(g_qpipe_thread, NULL);
    }
    memset(&g_qpipe, 0, sizeof(g_qpipe));
    g_qpipe_spare = nil;
    g_qpipe_wait_cb[0] = g_qpipe_wait_cb[1] = nil;
    g_qpipe_reader_cb[0] = g_qpipe_reader_cb[1] = nil;
    if (g_qpipe_nocache_fd >= 0) close(g_qpipe_nocache_fd);
    g_qpipe_nocache_fd = -1;
    g_qpipe_started = 0;
    g_qpipe_stop = 0;
    g_qpipe_disabled = 0;
    g_qpipe_need = 0;
}
```

- [ ] **Step 6: Public wrapper and pipe entry**

After the pipe module, add:
```objc
int ds4_gpu_qwen4_stream_stage_layer(
        const void *model_map, uint64_t model_size, uint32_t layer,
        const ds4_gpu_tensor *selected, uint32_t n_tokens, uint32_t n_slots,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint32_t gate_type, uint32_t down_type,
        uint32_t n_expert, uint32_t in_dim, uint32_t ff_dim, uint32_t out_dim,
        uint32_t seed_tokens) {
    /* The union path writes slot 0: let queued reads finish and drop slot 0's job. */
    if (g_qpipe_started) {
        pthread_mutex_lock(&g_qpipe_mu);
        qpipe_wait_idle_locked();
        qsp_invalidate(&g_qpipe, 0);
        pthread_mutex_unlock(&g_qpipe_mu);
    }
    const int ok = qwen4_stage_union(model_map, model_size, layer, selected, n_tokens, n_slots,
                                     gate_offset, up_offset, down_offset, gate_type, down_type,
                                     n_expert, in_dim, ff_dim, out_dim, seed_tokens);
    if (ok && g_qpipe_started) {
        pthread_mutex_lock(&g_qpipe_mu);
        qsp_activate(&g_qpipe, 0);   /* the batch now encoded reads slot 0 */
        pthread_mutex_unlock(&g_qpipe_mu);
    }
    return ok;
}

int ds4_gpu_qwen4_stream_stage_layer_pipe(
        const void *model_map, uint64_t model_size, uint32_t layer,
        const ds4_gpu_tensor *selected, uint32_t n_tokens, uint32_t n_slots,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint32_t gate_type, uint32_t down_type,
        uint32_t n_expert, uint32_t in_dim, uint32_t ff_dim, uint32_t out_dim,
        uint32_t next_layer, uint64_t next_gate_offset, uint64_t next_up_offset,
        uint64_t next_down_offset, uint32_t next_gate_type, uint32_t next_down_type,
        int nocache) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    const int profile = getenv("DS4_METAL_STREAMING_EXPERT_PREAD_PROFILE") != NULL;
    /* Same sizes as the union path. */
    const uint32_t gate_row_bytes = qwen4_expert_row_bytes(gate_type, in_dim);
    const uint32_t down_dim = (down_type == 10u || down_type == 12u) ? (ff_dim + 255u) / 256u * 256u : ff_dim;
    const uint32_t down_row_bytes = qwen4_expert_row_bytes(down_type, down_dim);
    const uint64_t gate_bytes = (uint64_t)gate_row_bytes * ff_dim * n_expert;
    const uint64_t down_bytes = (uint64_t)down_row_bytes * out_dim * n_expert;
    const uint64_t need = 2u * gate_bytes + down_bytes;
    const uint64_t off[3] = { gate_offset, up_offset, down_offset };
    const uint64_t next_off[3] = { next_gate_offset, next_up_offset, next_down_offset };
    const bool can_queue = next_layer != UINT32_MAX && next_gate_type == gate_type && next_down_type == down_type;
    if (g_qpipe_disabled || gate_row_bytes == 0 || down_row_bytes == 0 ||
        (g_qpipe_need != 0 && g_qpipe_need != need) || !qpipe_start()) {
        return ds4_gpu_qwen4_stream_stage_layer(model_map, model_size, layer, selected, n_tokens, n_slots,
                                                gate_offset, up_offset, down_offset, gate_type, down_type,
                                                n_expert, in_dim, ff_dim, out_dim, 0u);
    }
    g_qpipe_need = need;
    g_qpipe_bytes[0] = gate_bytes;
    g_qpipe_bytes[1] = gate_bytes;
    g_qpipe_bytes[2] = down_bytes;
    g_qpipe_region[0] = 0;
    g_qpipe_region[1] = (NSUInteger)gate_bytes;
    g_qpipe_region[2] = (NSUInteger)(2u * gate_bytes);

    pthread_mutex_lock(&g_qpipe_mu);
    const int s = qsp_find(&g_qpipe, layer, off, model_map);
    pthread_mutex_unlock(&g_qpipe_mu);
    if (s < 0) {
        /* Nothing read ahead (first streamed layer of a prompt, or after a
         * failure): union path into slot 0, then read the next layer into
         * slot 1. Everything was drained, so slot 1 has no reader. */
        const int ok = ds4_gpu_qwen4_stream_stage_layer(model_map, model_size, layer, selected, n_tokens,
                                                        n_slots, gate_offset, up_offset, down_offset,
                                                        gate_type, down_type, n_expert, in_dim, ff_dim,
                                                        out_dim, 0u);
        if (ok && can_queue && !g_qpipe_disabled) qpipe_queue(1, next_layer, next_off, model_map, nocache);
        return ok;
    }
    /* Keep the GPU busy: commit what is encoded (the flush hook records the
     * other slot's last reader), queue the next read behind it, then wait for
     * this layer's read, normally long done. */
    g_qwen4_stage.active = 0;
    if (g_batch_cb && !ds4_gpu_flush_commands()) return 0;
    if (can_queue) qpipe_queue(1 - s, next_layer, next_off, model_map, nocache);
    const double w0 = ds4_gpu_now_ms();
    pthread_mutex_lock(&g_qpipe_mu);
    while (g_qpipe.slot[s].state == QSP_QUEUED) pthread_cond_wait(&g_qpipe_cv, &g_qpipe_mu);
    const int done = g_qpipe.slot[s].state == QSP_DONE;
    if (done) qsp_activate(&g_qpipe, s);
    else qsp_invalidate(&g_qpipe, s);
    id<MTLBuffer> buf = qpipe_buffer(s);
    const double read_ms = g_qpipe_read_ms[s], cbwait_ms = g_qpipe_cbwait_ms[s];
    pthread_mutex_unlock(&g_qpipe_mu);
    const double wait_ms = ds4_gpu_now_ms() - w0;
    if (!done) {
        return ds4_gpu_qwen4_stream_stage_layer(model_map, model_size, layer, selected, n_tokens, n_slots,
                                                gate_offset, up_offset, down_offset, gate_type, down_type,
                                                n_expert, in_dim, ff_dim, out_dim, 0u);
    }
    g_qwen4_stage.bind_buf = buf;
    g_qwen4_stage.map = model_map;
    for (int t = 0; t < 3; t++) {
        g_qwen4_stage.off[t] = off[t];
        g_qwen4_stage.bytes[t] = g_qpipe_bytes[t];
        g_qwen4_stage.region[t] = g_qpipe_region[t];
    }
    g_qwen4_stage.active = 1;
    if (profile) {
        fprintf(stderr, "ds4: qwen4 pipe layer=%u tokens=%u slot=%d wait=%.3f ms read=%.3f ms cbwait=%.3f ms\n",
                layer, n_tokens, s, wait_ms, read_ms, cbwait_ms);
    }
    (void)selected;
    return 1;
}
```
Note: `qwen4_stage_union` allocates `g_qwen4_stage.buf` itself. The pipe only queues into slot 0 from a
stage call that found its job in slot 1, and by then the union path has allocated slot 0 for the first streamed
layer of the prompt.

- [ ] **Step 7: Hooks and cleanup**

In `ds4_gpu_flush_commands`, after `[g_pending_cbs addObject:cb];`, add `ds4_gpu_qpipe_note_commit(cb);`.
In `ds4_gpu_end_commands`, replace:
```objc
    const int ok = ds4_gpu_finish_command_buffer(cb, 1, "command batch");
    return qgate_check_after_wait() && ok;
```
with:
```objc
    const int ok = ds4_gpu_finish_command_buffer(cb, 1, "command batch");
    ds4_gpu_qpipe_note_all_complete();
    return qgate_check_after_wait() && ok;
```
(Check with `grep -c` that this `finish_command_buffer(cb, 1, "command batch")` line inside `ds4_gpu_end_commands`
is the one you edit.)
In `ds4_gpu_cleanup`, after the residency lines added in Task 3, add `ds4_gpu_qpipe_shutdown();`.

- [ ] **Step 8: Call the pipe from `ds4.c`**

In `qwen4_graph_moe`, replace:
```c
    if (ok && mm_shape && experts_streamed && qwen4_stream_stage_prefill_enabled()) {
        staged = ds4_gpu_qwen4_stream_stage_layer(m->map, m->size, il, g->selected, T, DS4_N_EXPERT_USED,
                                                  l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset,
                                                  l->ffn_down_exps->abs_offset, l->ffn_gate_exps->type,
                                                  l->ffn_down_exps->type, DS4_N_EXPERT, DS4_N_EMBD,
                                                  DS4_N_FF_EXP, DS4_N_EMBD,
                                                  g->stream_seed_last ? qwen4_stream_seed_tokens() : 0u) != 0;
    }
```
with:
```c
    if (ok && mm_shape && experts_streamed && qwen4_stream_stage_prefill_enabled()) {
        const uint32_t seed = g->stream_seed_last ? qwen4_stream_seed_tokens() : 0u;
#ifdef DS4_HAS_QWEN4_METAL
        const qwen4_prefill_flags pf = qwen4_prefill_policy(qwen4_prefill_mode_env(), g->prefill_role, T, true);
        if (qwen4_prefill_use_pipe(pf, seed)) {
            if (!g->stream_layers_ready) {
                g->n_stream_layers = qwen4_stream_layer_list(w, g->stream_layers, DS4_MAX_LAYER);
                g->stream_layers_ready = true;
            }
            const uint32_t nx = qwen4_stream_next_staged(g->stream_layers, g->n_stream_layers, il,
                                                         g->prefill_role);
            const ds4_layer_weights *ln = nx != UINT32_MAX ? &w->layer[nx] : NULL;
            staged = ds4_gpu_qwen4_stream_stage_layer_pipe(
                m->map, m->size, il, g->selected, T, DS4_N_EXPERT_USED,
                l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset, l->ffn_down_exps->abs_offset,
                l->ffn_gate_exps->type, l->ffn_down_exps->type, DS4_N_EXPERT, DS4_N_EMBD, DS4_N_FF_EXP,
                DS4_N_EMBD, nx,
                ln ? ln->ffn_gate_exps->abs_offset : 0u, ln ? ln->ffn_up_exps->abs_offset : 0u,
                ln ? ln->ffn_down_exps->abs_offset : 0u, ln ? ln->ffn_gate_exps->type : UINT32_MAX,
                ln ? ln->ffn_down_exps->type : UINT32_MAX, pf.pipe_nocache ? 1 : 0) != 0;
        } else
#endif
        staged = ds4_gpu_qwen4_stream_stage_layer(m->map, m->size, il, g->selected, T, DS4_N_EXPERT_USED,
                                                  l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset,
                                                  l->ffn_down_exps->abs_offset, l->ffn_gate_exps->type,
                                                  l->ffn_down_exps->type, DS4_N_EXPERT, DS4_N_EMBD,
                                                  DS4_N_FF_EXP, DS4_N_EMBD, seed) != 0;
    }
```
In `qwen4_graph_forward_tokens`, replace the Task 3 block:
```c
#ifdef DS4_HAS_QWEN4_METAL
    if (pflags.residency) ds4_gpu_prefill_residency_end();
#endif
```
with:
```c
#ifdef DS4_HAS_QWEN4_METAL
    if (pflags.residency) ds4_gpu_prefill_residency_end();
    if (qwen4_prefill_prompt_ends(pmode, g->prefill_role, ok, pstreaming)) {
        ds4_gpu_qwen4_stream_stage_prompt_end();
    }
#endif
```

Just before `static bool qwen4_graph_forward_token(` add:
```c
/* A prompt loop stopped between chunks (cancellation): the pipe may hold a read
 * queued for a next chunk that will not come. */
static void qwen4_prefill_abandon(void) {
#ifdef DS4_HAS_QWEN4_METAL
    if (qwen4_prefill_mode_env() != QWEN4_PREFILL_OFF && ds4_gpu_ssd_streaming_enabled()) {
        ds4_gpu_qwen4_stream_stage_prompt_end();
    }
#endif
}
```
In the server loop, replace:
```c
            if (ds4_session_cancelled(s)) {
                snprintf(err, errlen, "interrupted");
```
with:
```c
            if (ds4_session_cancelled(s)) {
                qwen4_prefill_abandon();
                snprintf(err, errlen, "interrupted");
```
(Check with `grep -c 'snprintf(err, errlen, "interrupted");'` and edit the occurrence inside the qwen4 prompt loop that
sets `s->qwen4_graph.stream_seed_last`.)

- [ ] **Step 9: Build, model-free tests, static check of the off path**

Run:
```bash
make -j10 ds4 ds4-server 2>&1 | grep -E 'error|warning:'; echo "exit ${pipestatus[1]}"
make test-qwen4-prefill-pipe 2>&1 | tail -1
make test-qwen4-kernels 2>&1 | tail -2
```
Expected: no warnings, `exit 0`, `test_qwen4_prefill_pipe: PASS`, qwen4 kernel test PASS lines.

- [ ] **Step 10: [GPU] Exactness across modes and pipe sanity**

Stop oMLX and the watchdogs. Run:
```bash
O=/tmp/claude-501/qpp-task4
speed-bench/qwen-prefill-pipe/cli_exact.sh $O chat p5k p36k p134k
for m in safe max; do DS4_METAL_STREAMING_EXPERT_PREAD_PROFILE=1 DS4_METAL_GPU_IDLE=1 \
  speed-bench/qwen-prefill-pipe/cli_run.sh $O prof-$m $O/prompts/p36k.txt $m 16; \
  grep -a 'gpu run:' $O/prof-$m.err; grep -a -c 'qwen4 pipe layer' $O/prof-$m.err; done
```
Expected:
- `PASS` for all four prompts.
- For `max` at 36K:
  - prefill about +25 % over `off` or more (spike: 592-607 vs ~460);
  - the `gpu run:` idle share is under 20 %;
  - 280+ `qwen4 pipe layer` lines.
- For `safe`, fewer pipe lines: the last chunk uses the union path.

Restore the services.

- [ ] **Step 11: Commit**

```bash
git add ds4.c ds4_gpu.h ds4_metal.m
git commit -F - <<'EOF'
ds4: qwen4 drain-free double-buffered prefill staging pipe

DS4_QWEN4_PREFILL_MODE=safe|max stage streamed layers from a whole-layer read
queued one layer ahead into a second buffer, instead of draining the GPU and
reading the selected experts at every streamed layer. A reader thread waits
for the command buffer that last read the target buffer (tracked by flush and
end_commands hooks). The union path stays for the first streamed layer, read
failures, seeding and the decode fallback. Output is byte-identical.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015npc1LHTBTYaTXLaJt8Uqc
EOF
```

---

### Task 5: Long-prompt paired A/B in `qwen_gate`

**Files:**
- Modify: `speed-bench/qwen-regression/qwen_gate.py`
- Test: `speed-bench/tests/test_qwen_gate.py`

**Interfaces:**
- Produces:
  - `long_prompts(filler, index) -> {"long": str, "medium": str}`
  - `sse_timings(events) -> {"prompt_tokens", "completion_tokens", "ttft_s", "decode_s"}`
  - `request_rates(timing) -> {"prefill_tps", "decode_tps"}`
  - `long_ab(measure, order=AB_ORDER)`
  - `long_ab_summary(ab)`
  - `long_ab_failures(summary)`
  - `server(cmd, cwd, port, log_path, env=None)`
  - CLI `qwen_gate.py longab --bin DIR --out DIR --prefill-mode safe|max`

- [ ] **Step 1: Write the failing tests**

Append to `speed-bench/tests/test_qwen_gate.py` (before `if __name__ == "__main__":` if present, else at the end):
```python
class LongPromptsTest(unittest.TestCase):
    def test_distinct_slices_per_server_and_request(self):
        filler = "".join(str(i) for i in range(200_000))   # ~1.09M chars, no repeating slice
        seen = []
        for index in range(len(qwen_gate.AB_ORDER)):
            prompts = qwen_gate.long_prompts(filler, index)
            self.assertEqual(set(prompts), {"long", "medium"})
            for name, chars in qwen_gate.LONG_REQUESTS:
                self.assertTrue(prompts[name].endswith(qwen_gate.LONG_QUESTION))
                self.assertEqual(len(prompts[name]), chars + len(qwen_gate.LONG_QUESTION))
                seen.append(prompts[name])
        self.assertEqual(len(seen), len(set(seen)))

    def test_short_filler_rejected(self):
        with self.assertRaises(ValueError):
            qwen_gate.long_prompts("x" * 1000, 0)


class SseTimingsTest(unittest.TestCase):
    def test_first_and_last_delta_and_usage(self):
        events = [
            (0.5, {"choices": [{"delta": {"role": "assistant"}}]}),
            (2.0, {"choices": [{"delta": {"reasoning_content": "hm"}}]}),
            (3.0, {"choices": [{"delta": {"content": "ok"}}]}),
            (3.1, {"choices": [], "usage": {"prompt_tokens": 1000, "completion_tokens": 11}}),
        ]
        t = qwen_gate.sse_timings(events)
        self.assertEqual((t["prompt_tokens"], t["completion_tokens"]), (1000, 11))
        self.assertAlmostEqual(t["ttft_s"], 2.0)
        self.assertAlmostEqual(t["decode_s"], 1.0)
        r = qwen_gate.request_rates(t)
        self.assertAlmostEqual(r["prefill_tps"], 500.0)
        self.assertAlmostEqual(r["decode_tps"], 10.0)

    def test_no_delta_is_an_error(self):
        with self.assertRaises(ValueError):
            qwen_gate.sse_timings([(1.0, {"choices": [], "usage": {}})])


def _rates(prefill, decode):
    return {name: {"prefill_tps": prefill, "decode_tps": decode} for name, _ in qwen_gate.LONG_REQUESTS}


class LongAbTest(unittest.TestCase):
    def test_interleaved_order_and_index(self):
        calls = []
        ab = qwen_gate.long_ab(lambda which, index: calls.append((which, index)) or _rates(1, 1))
        self.assertEqual(calls, [(w, i) for i, w in enumerate(qwen_gate.AB_ORDER)])
        self.assertEqual(len(ab["prod"]), 2)
        self.assertEqual(len(ab["branch"]), 2)

    def test_pass_when_decode_holds_and_prefill_gains(self):
        ab = {"prod": [_rates(400, 36), _rates(420, 37)], "branch": [_rates(560, 36), _rates(580, 36.5)]}
        summary = qwen_gate.long_ab_summary(ab)
        self.assertAlmostEqual(summary["long"]["prefill_ratio"], 1140 / 820)
        self.assertEqual(qwen_gate.long_ab_failures(summary), [])

    def test_decode_below_floor_fails_per_request(self):
        ab = {"prod": [_rates(400, 36)] * 2, "branch": [_rates(560, 34)] * 2}
        failures = qwen_gate.long_ab_failures(qwen_gate.long_ab_summary(ab))
        self.assertEqual(len(failures), len(qwen_gate.LONG_REQUESTS))
        self.assertIn("decode", failures[0])

    def test_small_prefill_gain_fails(self):
        ab = {"prod": [_rates(400, 36)] * 2, "branch": [_rates(420, 36)] * 2}
        failures = qwen_gate.long_ab_failures(qwen_gate.long_ab_summary(ab))
        self.assertEqual(len(failures), 1)
        self.assertIn("prefill", failures[0])


class LongRunTest(unittest.TestCase):
    def test_mode_env_only_on_the_branch(self):
        seen = []

        def fake_measure(bin_dir, out, tag, port, index, env=None):
            seen.append((bin_dir, env))
            return _rates(1, 1)

        with tempfile.TemporaryDirectory() as tmp:
            orig = qwen_gate.measure_long_server
            qwen_gate.measure_long_server = fake_measure
            try:
                qwen_gate.run_long("/work", tmp, "max", ds4_running=lambda: "", idle_read=lambda: 1.0)
            finally:
                qwen_gate.measure_long_server = orig
            with open(os.path.join(tmp, "longab.json")) as fp:
                saved = json.load(fp)
        self.assertEqual(seen[0], (None, None))
        self.assertEqual(seen[1], ("/work", {"DS4_QWEN4_PREFILL_MODE": "max"}))
        self.assertEqual(saved["prefill_mode"], "max")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd speed-bench/tests && python3 -m unittest test_qwen_gate 2>&1 | tail -3`
Expected: `FAILED (errors=...)`, with `AttributeError: module 'qwen_gate' has no attribute 'long_prompts'` among them.

- [ ] **Step 3: Implement**

In `qwen_gate.py`, after `WIRED_SLACK_GIB = 0.5`, add:
```python
# Long-prompt paired A/B (docs/superpowers/specs/2026-09-25-qwen4-prefill-residency-pipe-design.md):
# chars of filler per request (~3.4 chars/token: ~32K and ~5K tokens), each a
# slice no other server or request uses, so the prefix cache never skips it.
LONG_REQUESTS = (("long", 110_000), ("medium", 17_000))
LONG_QUESTION = "\n\nSummarize the text above in two sentences."
LONG_MAX_TOKENS = 300
LONG_PREFILL_GAIN = 1.10
```
Change `def server(cmd, cwd, port, log_path):` to `def server(cmd, cwd, port, log_path, env=None):` and its
`Popen` line to:
```python
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                            env=None if env is None else {**os.environ, **env})
```
After `measure_server`, add:
```python
def long_prompts(filler, index):
    """The two prompts of server `index` (0-based position in AB_ORDER)."""
    per_server = sum(chars for _, chars in LONG_REQUESTS)
    start = index * per_server
    if start + per_server > len(filler):
        raise ValueError(f"filler has {len(filler)} chars, need {start + per_server}")
    prompts = {}
    for name, chars in LONG_REQUESTS:
        prompts[name] = filler[start:start + chars] + LONG_QUESTION
        start += chars
    return prompts


def sse_timings(events):
    """events: (seconds since the request, parsed chunk) per SSE data line.
    Time to the first content or reasoning delta, first-to-last delta time,
    and the usage token counts."""
    first = last = None
    usage = {}
    for t, chunk in events:
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                first = t if first is None else first
                last = t
    if first is None:
        raise ValueError("stream had no content delta")
    return {"prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "ttft_s": first, "decode_s": last - first}


def request_rates(timing):
    """Prefill t/s (prompt tokens over time to first token) and decode t/s."""
    prefill = timing["prompt_tokens"] / timing["ttft_s"] if timing["ttft_s"] > 0 else 0.0
    n = timing["completion_tokens"] - 1
    decode = n / timing["decode_s"] if n > 0 and timing["decode_s"] > 0 else 0.0
    return {"prefill_tps": prefill, "decode_tps": decode}


def chat_stream(base, text, max_tokens):
    """One streamed chat request; sse_timings of its chunks."""
    body = json.dumps({"model": "ds4", "messages": [{"role": "user", "content": text}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    events = []
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append((time.time() - t0, json.loads(line[6:])))
    return sse_timings(events)


def measure_long_server(bin_dir, out, tag, port, index, env=None):
    """One fresh server: the long request, then the medium one, streamed."""
    kv = os.path.join(out, f"kv-long-{tag}")
    shutil.rmtree(kv, ignore_errors=True)
    os.makedirs(kv)
    require_idle()
    cmd, cwd = registry_command(REGISTRY, bin_dir, port, kv)
    with open(os.path.join(ROOT, NEEDLE_SOURCE), encoding="utf-8", errors="replace") as fp:
        prompts = long_prompts(fp.read(), index)
    runs = {}
    with server(cmd, cwd, port, os.path.join(out, f"server-long-{tag}.log"), env=env) as base:
        for name, _ in LONG_REQUESTS:
            timing = chat_stream(base, prompts[name], LONG_MAX_TOKENS)
            runs[name] = {**timing, **request_rates(timing)}
    shutil.rmtree(kv, ignore_errors=True)
    return runs


def long_ab(measure, order=AB_ORDER):
    """measure(which, index) -> {request: rates} of one fresh server, interleaved."""
    ab = {"prod": [], "branch": []}
    for index, which in enumerate(order):
        ab[which].append(measure(which, index))
    return ab


def long_ab_summary(ab):
    """Per request: mean prefill and decode t/s per side, branch/PROD ratios."""
    summary = {}
    for name, _ in LONG_REQUESTS:
        row = {}
        for side in ("prod", "branch"):
            runs = [r[name] for r in ab[side]]
            row[side + "_prefill_tps"] = statistics.fmean(r["prefill_tps"] for r in runs)
            row[side + "_decode_tps"] = statistics.fmean(r["decode_tps"] for r in runs)
        row["prefill_ratio"] = row["branch_prefill_tps"] / row["prod_prefill_tps"]
        row["decode_ratio"] = row["branch_decode_tps"] / row["prod_decode_tps"]
        summary[name] = row
    return summary


def long_ab_failures(summary):
    """The default-mode rule: decode >= TPS_FLOOR of PROD on every request,
    and at least LONG_PREFILL_GAIN prefill on the long one."""
    failures = []
    for name, row in summary.items():
        if row["decode_ratio"] < TPS_FLOOR:
            failures.append(f"{name}: decode {row['branch_decode_tps']:.2f} t/s is "
                            f"{row['decode_ratio']:.3f} of PROD {row['prod_decode_tps']:.2f}")
    long_row = summary["long"]
    if long_row["prefill_ratio"] < LONG_PREFILL_GAIN:
        failures.append(f"long: prefill {long_row['branch_prefill_tps']:.1f} t/s is only "
                        f"{long_row['prefill_ratio']:.3f} of PROD {long_row['prod_prefill_tps']:.1f}")
    return failures


def run_long(bin_dir, out, prefill_mode, ds4_running=machine.ds4_running, idle_read=None,
             idle_timeout=wired.IDLE_SETTLE_S, idle_interval=2.0):
    out = os.path.abspath(out)
    busy = ds4_running()
    if busy:
        raise SystemExit("qwen_gate: ds4 is running; the machine must be free:\n" + busy)
    require_idle(idle_read, idle_timeout, idle_interval)
    os.makedirs(out, exist_ok=True)
    port = int(os.environ.get("DS4_GATE_PORT", "18298"))
    env = {"DS4_QWEN4_PREFILL_MODE": prefill_mode}
    ab = long_ab(lambda which, index: measure_long_server(
        None if which == "prod" else bin_dir, out, f"{index}-{which}", port, index,
        None if which == "prod" else env))
    summary = long_ab_summary(ab)
    result = {"prefill_mode": prefill_mode, "ab": ab, "summary": summary,
              "failures": long_ab_failures(summary)}
    with open(os.path.join(out, "longab.json"), "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=1)
    return result
```
In `main()`, after the `chk` parser setup loop (`for p in (rec, chk): ...`), add:
```python
    lng = sub.add_parser("longab", help="paired long-prompt prefill/decode A/B against PROD")
    lng.add_argument("--bin", required=True, help="directory holding the ds4-server to test")
    lng.add_argument("--out", required=True)
    lng.add_argument("--prefill-mode", required=True, choices=("safe", "max"))
```
and replace `args = ap.parse_args()` with:
```python
    args = ap.parse_args()
    if args.mode == "longab":
        result = run_long(args.bin, args.out, args.prefill_mode)
        for name, row in result["summary"].items():
            print(f"qwen_gate: {name}: prefill {row['prod_prefill_tps']:.1f} -> {row['branch_prefill_tps']:.1f} "
                  f"(x{row['prefill_ratio']:.3f}), decode {row['prod_decode_tps']:.2f} -> "
                  f"{row['branch_decode_tps']:.2f} (x{row['decode_ratio']:.3f})")
        for failure in result["failures"]:
            print("qwen_gate: FAIL", failure)
        print("qwen_gate:", "FAIL" if result["failures"] else "PASS")
        return 1 if result["failures"] else 0
```
`run_long` refuses to start when the machine is busy. `LongRunTest` passes a fake `ds4_running` and `idle_read`,
so the test starts no server and needs no GPU.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd speed-bench/tests && python3 -m unittest discover -p 'test_*.py' 2>&1 | tail -3`
Expected: `OK` (the previous count plus the new tests).

- [ ] **Step 5: Commit**

```bash
git add speed-bench/qwen-regression/qwen_gate.py speed-bench/tests/test_qwen_gate.py
git commit -F - <<'EOF'
speed-bench: qwen_gate longab paired long-prompt prefill/decode A/B

Streams a ~32K then a ~5K-token request (distinct filler slices, scratch KV
dir) on fresh PROD and branch servers in AB_ORDER, the branch with
DS4_QWEN4_PREFILL_MODE set; prefill from time to first token, decode from
first-to-last delta. Fails if decode < 97 % of PROD on either request or the
long prefill gains < 10 %.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015npc1LHTBTYaTXLaJt8Uqc
EOF
```

---

### Task 6: Validation, default selection, and results

**Files:**
- Create: `speed-bench/qwen-prefill-pipe/RESULTS.md`
- Modify: auto-memory `qwen4-prefill-launch-latency-spike.md` (outside git)

- [ ] **Step 1: [GPU] Exactness on the servers**

Stop oMLX and the watchdogs. Run:
```bash
make -j10 ds4 ds4-server
DS4_QWEN4_PREFILL_MODE=safe speed-bench/qwen-regression/run.sh fast
DS4_QWEN4_PREFILL_MODE=max speed-bench/qwen-regression/run.sh fast
```
Expected: both end with `qwen_gate: PASS`: byte-identical vi/code replies against the PROD baseline, and all unit
kernels plus `test_qwen4_prefill_pipe: PASS`. The PROD side ignores the variable.

- [ ] **Step 2: [GPU] Long-prompt paired A/B for both modes**

Run:
```bash
python3 speed-bench/qwen-regression/qwen_gate.py longab --bin . --out /tmp/claude-501/qpp-longab-max --prefill-mode max
python3 speed-bench/qwen-regression/qwen_gate.py longab --bin . --out /tmp/claude-501/qpp-longab-safe --prefill-mode safe
```
Expected: two `long`/`medium` summary lines per mode and a PASS or FAIL. Apply the rule:
1. `max` if it passes;
2. otherwise `safe` if it passes;
3. otherwise `off`.

Write the chosen mode down as `CHOSEN`.

- [ ] **Step 3: [GPU] Full tier with the chosen mode (needle, wired memory, short-prompt decode)**

If `CHOSEN` is not `off`, run:
```bash
DS4_QWEN4_PREFILL_MODE=$CHOSEN speed-bench/qwen-regression/run.sh full
```
Expected: `qwen_gate: PASS`:
- the needle is found (~215K-token prompt);
- steady wired is at most baseline + 0.5 GiB;
- the paired short-prompt decode is at least 97 % of PROD.

- [ ] **Step 4: [GPU] Launch latency and GPU idle receipt**

Run:
```bash
O=/tmp/claude-501/qpp-task6
python3 speed-bench/qwen-prefill-pipe/make_prompts.py $O/prompts
for m in off safe max; do DS4_METAL_GPU_IDLE=1 speed-bench/qwen-prefill-pipe/cli_run.sh $O idle-$m $O/prompts/p36k.txt $m 16; grep -a 'gpu run:' $O/idle-$m.err; done
```
Expected: `launch` under 1 s for `safe` and `max`, and 10-20 s for `off`. Restore the services.

- [ ] **Step 5: Write `speed-bench/qwen-prefill-pipe/RESULTS.md`**

Structure (fill every number from Steps 1-4 and Task 3/4 GPU steps; no placeholders left):
```markdown
# qwen4 prefill residency + staging pipe: results

Date: <date>. Branch `feature/qwen4-prefill-pipe` at `<sha>`. Model unc48L (PROD registry flags).

## Knob
`DS4_QWEN4_PREFILL_MODE=off|safe|max` (engine default `off`); see the spec.
Chosen for PROD: `<CHOSEN>` (rule: decode >= 97 % of PROD on both long-prompt requests, prefill >= +10 %).

## Exactness
- cli_exact.sh chat/p5k/p36k/p134k: <PASS lines>
- qwen-regression fast, safe and max: <PASS>
- full tier with <CHOSEN>: needle <hit>, steady wired <GiB> (baseline <GiB>), paired decode <x>

## Long-prompt paired A/B (qwen_gate longab)
| mode | request | PROD prefill | branch prefill | x | PROD decode | branch decode | x | verdict |
|---|---|---|---|---|---|---|---|---|
| max | long | ... |
| max | medium | ... |
| safe | long | ... |
| safe | medium | ... |

## GPU idle at 36K (DS4_METAL_GPU_IDLE)
| mode | prefill t/s | span ms | busy ms | idle % | launch ms |
|---|---|---|---|---|---|

## How to reproduce
<the commands from Tasks 3, 4 and 6>
```
Commit:
```bash
git add speed-bench/qwen-prefill-pipe/RESULTS.md
git commit -F - <<'EOF'
speed-bench: qwen4 prefill residency + pipe results and default choice

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015npc1LHTBTYaTXLaJt8Uqc
EOF
```

- [ ] **Step 6: Update the auto-memory note**

Edit `/Users/dongnh/.claude/projects/-Users-dongnh-Documents-GitHub-ds4-metal/memory/qwen4-prefill-launch-latency-spike.md`:
- the implementation commit range;
- the chosen default and the long-A/B numbers;
- that PROD deployment is pending approval (`prod/<feature>-YYYYMMDD` with `DS4_QWEN4_PREFILL_MODE=<CHOSEN>` in the registry).

Update its `MEMORY.md` line.

- [ ] **Step 7: Hand off**

Use superpowers:requesting-code-review for the whole branch, then superpowers:finishing-a-development-branch.
Do not deploy to PROD in this plan.
