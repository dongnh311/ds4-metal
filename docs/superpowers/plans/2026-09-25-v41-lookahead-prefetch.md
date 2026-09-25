# V4.1 Lookahead Expert Prefetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Warm the OS page cache with the predicted experts of layer L+1 while layer L runs, so V4.1 SSD-streamed decode hides part of its 28.6 ms/token miss path, bit-exact.

**Architecture:**
- After the async worker returns layer L's ids, the main thread copies layer L's FFN input into a one-slot mailbox.
- One process-wide prediction thread scores layer L+1's router on that copy (a dot-product loop, `sqrt(softplus)+bias`) and takes the top k.
- For each predicted expert that is not cached, it issues `F_RDADVISE` for its three byte ranges on a private fd.
- The expert cache, the selection and every kernel are unchanged.

**Tech Stack:** C99 (`ds4.c`, `tests/test_deepseek41_graph.c`), Objective-C (`ds4_metal.m`), pthreads, `fcntl(F_RDADVISE)`, Python 3.9 stdlib (`speed-bench/v41`).

**Spec:** `docs/superpowers/specs/2026-09-25-v41-lookahead-prefetch-design.md` (spike evidence in memory `v41-lookahead-spike`).

## Global Constraints

- **Bit-identical output.** Prefetch only moves file pages into the OS page cache. Switch: `DS4_METAL_DISABLE_V41_LOOKAHEAD`, read per call.
- **V4.1-only.** New code sits behind `#if defined(__APPLE__) && !defined(DS4_NO_GPU)` in the V4.1 section of `ds4.c`, plus one read-only exported query in `ds4_metal.m`. The worker job struct shared with V4-Flash and the shared cache code (pread pool, load, install, evict) are not changed.
- **No critical-path wait.** Per layer, the main thread spends at most a 20 KB copy and a mailbox post; the mailbox never blocks the poster.
- **Machine limits.** No new wired memory; zero swap; wired ≤ 52.48 GiB.
- **Speed.** Judged by interleaved A/B (A, B, B, A) only.
- **GPU gate.** A step marked **[GPU]** runs only after the user confirms the box is free (oMLX and both watchdogs down). Restore them after the last GPU step of a session:
  - `launchctl load ~/Library/LaunchAgents/dev.dongnh.ai-proxy.plist`
  - `launchctl load ~/Library/LaunchAgents/dev.dongnh.gateway-watchdog.plist`
- **GPU run protocol.** Wrap each run with `caffeinate -i -s`, a swap watch and a stall watch; never `kill -9`.
- **Language.** Chat in Vietnamese; code, docs and commits in English. Do not push.
- **Commit trailer:**
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01TwriwxNUTW7k4x9zH3WeQT
  ```
- **Model path:** `M=$HOME/orca/workspaces/ds4-metal-data/gguf/DeepSeek-V4.1-Flash-Q2.gguf`.
- **Prompts:** `P=$HOME/orca/workspaces/ds4-metal-data/v41-phase0/20260924/prompts`.

## Review Focus

1. **Engine close while the thread is mid-advisory.** The thread must be joined and its fd closed before the model is unmapped. A second engine in the same process must start a fresh thread. Pinned by Task 2's start/stop/restart unit test.
2. **Two sessions in one process posting alternately** (`--stream-control` does this). There must be no crash, and no "used" count from the other session's stale prediction. Pinned by Task 2's position check in `ds41_la_note_selected` and by Task 4's stream-control run.
3. **Async load off** (`DS4_METAL_DISABLE_V41_ASYNC_LOAD`). The hook sits inside the async block, so nothing is posted. Pinned in Task 4: the counter line shows `posted 0`.
4. **`DS4_METAL_V41_LOOKAHEAD_K=0` or garbage.** 0 means off; garbage means the default 1; values clamp to 0..6. Pinned by Task 1's parse test.
5. **Last layer and image positions** are never posted: the last layer has no next layer, and image positions use another bias. The last layer is pinned by Task 2's `ds41_la_should_post` unit test; the image skip (`ds41_image_at` in the hook) is a code-review item, since there is no model-free graph with an image position.

---

### Task 1: Lookahead core (pure pieces) and the residency hint

**Files:**
- Modify: `ds4.c`. Add the lookahead core block directly above `static bool ds41_stream_async_load(` (V4.1 section, Apple guard).
- Modify: `ds4_metal.m`. Add `ds4_gpu_stream_expert_cache_resident_hint` right after the definition of `ds4_gpu_stream_expert_cache_configured_count`.
- Modify: `ds4_gpu.h`. Declare it next to `ds4_gpu_stream_expert_cache_configured_count`.
- Modify: `docs/superpowers/specs/2026-09-25-v41-lookahead-prefetch-design.md`. In Data flow step 3, replace "`cblas_sgemv` of layer T's" with "a dot-product loop over layer T's". This keeps the Accelerate header out of `ds4.c`, which is also built for CUDA and Linux; the loop costs about 0.1-0.2 ms per layer.
- Test: `tests/test_deepseek41_graph.c`, new model-free mode `--v41-lookahead-units`.

**Interfaces:**
- Produces:
  - `#define DS41_LA_MAX_EMBD 8192u`, `#define DS41_LA_MAX_K 6u`, `#define DS41_LA_MAX_LAYER 64u`.
  - `typedef struct { uint32_t layer, pos, k, n_embd, n_expert; const float *w; const float *bias; uint64_t gate_offset, up_offset, down_offset, gate_bytes, down_bytes; float x[DS41_LA_MAX_EMBD]; } ds41_la_job;`
  - `typedef struct { pthread_mutex_t mu; pthread_cond_t cv; bool full; ds41_la_job job; uint64_t posted, dropped; } ds41_la_mailbox;`
  - `static uint32_t ds41_la_k(void);` returns 1 by default, 0..6 from `DS4_METAL_V41_LOOKAHEAD_K`, 1 on garbage.
  - `static uint32_t ds41_la_topk(const float *w, const float *bias, const float *x, uint32_t n_expert, uint32_t n_embd, uint32_t k, int32_t *out);` returns the count written, ordered by score desc then index asc.
  - `static void ds41_la_ranges(const ds41_la_job *job, int32_t expert, uint64_t off[3], uint64_t len[3]);` gives gate, up, down.
  - `static void ds41_la_mailbox_init(ds41_la_mailbox *mb);`
  - `static void ds41_la_post(ds41_la_mailbox *mb, const ds41_la_job *job);`
  - `static bool ds41_la_take(ds41_la_mailbox *mb, ds41_la_job *out, bool wait, const volatile bool *stop);`
  - `int ds4_gpu_stream_expert_cache_resident_hint(uint32_t layer, uint32_t expert);`

- [ ] **Step 1: Write the failing test**

In `tests/test_deepseek41_graph.c`, add above `static int check_decode_profile_format(void) {`:

```c
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
/* Lookahead prefetch core: K parsing, the scorer's top-k, the byte ranges
 * (same as graph_stream_expert_table_make), the one-slot mailbox, and the
 * residency hint outside streaming. Model-free. */
static int check_v41_lookahead_units(void) {
    int rc = 1;
    static ds41_la_job a, b, got;
    ds41_la_mailbox mb;
    /* K: default 1, 0..6, garbage -> 1. */
    unsetenv("DS4_METAL_V41_LOOKAHEAD_K");
    REQUIRE(ds41_la_k() == 1);
    setenv("DS4_METAL_V41_LOOKAHEAD_K", "0", 1); REQUIRE(ds41_la_k() == 0);
    setenv("DS4_METAL_V41_LOOKAHEAD_K", "4", 1); REQUIRE(ds41_la_k() == 4);
    setenv("DS4_METAL_V41_LOOKAHEAD_K", "9", 1); REQUIRE(ds41_la_k() == 6);
    setenv("DS4_METAL_V41_LOOKAHEAD_K", "x", 1); REQUIRE(ds41_la_k() == 1);
    unsetenv("DS4_METAL_V41_LOOKAHEAD_K");
    /* Scorer: 5 experts, 2-wide rows. logits = w . x with x = (1, 1). */
    {
        const float w[10] = { 1, 0,   3, 0,   -30, 0,   3, 0,   2, 0 };
        const float bias[5] = { 0, 0, 0, 0, 0.5f };
        const float x[2] = { 1, 1 };
        int32_t out[6];
        /* logits 1, 3, -30, 3, 2; scores sqrt(softplus(l)) + bias:
         * e0 1.146, e1 1.746, e2 ~0, e3 1.746 (tie with e1 -> e1 first),
         * e4 1.458 + 0.5 = 1.958. Order: 4, 1, 3, 0, 2. */
        REQUIRE(ds41_la_topk(w, bias, x, 5, 2, 3, out) == 3);
        REQUIRE(out[0] == 4 && out[1] == 1 && out[2] == 3);
        REQUIRE(ds41_la_topk(w, bias, x, 5, 2, 0, out) == 0);
        REQUIRE(ds41_la_topk(w, bias, x, 5, 2, 6, out) == 5);
        REQUIRE(out[3] == 0 && out[4] == 2);
    }
    /* Ranges: the same offsets graph_stream_expert_table_make produces. */
    {
        ds4_tensor gate = {.abs_offset = 1000}, up = {.abs_offset = 50000}, down = {.abs_offset = 90000};
        ds4_layer_weights l = {0};
        ds4_model m = {.fd = -1};
        l.ffn_gate_exps = &gate; l.ffn_up_exps = &up; l.ffn_down_exps = &down;
        const ds4_gpu_stream_expert_table t = graph_stream_expert_table_make(&m, &l, 3, 120, 70);
        ds41_la_job j = {.gate_offset = t.gate_offset, .up_offset = t.up_offset,
                         .down_offset = t.down_offset, .gate_bytes = t.gate_expert_bytes,
                         .down_bytes = t.down_expert_bytes};
        uint64_t off[3], len[3];
        ds41_la_ranges(&j, 5, off, len);
        REQUIRE(off[0] == 1000 + 5 * 120 && len[0] == 120);
        REQUIRE(off[1] == 50000 + 5 * 120 && len[1] == 120);
        REQUIRE(off[2] == 90000 + 5 * 70 && len[2] == 70);
    }
    /* Mailbox: a second post replaces the first and counts a drop. */
    ds41_la_mailbox_init(&mb);
    a.layer = 7; a.pos = 100; b.layer = 8; b.pos = 100;
    REQUIRE(!ds41_la_take(&mb, &got, false, NULL));
    ds41_la_post(&mb, &a);
    ds41_la_post(&mb, &b);
    REQUIRE(mb.posted == 2 && mb.dropped == 1);
    REQUIRE(ds41_la_take(&mb, &got, false, NULL) && got.layer == 8);
    REQUIRE(!ds41_la_take(&mb, &got, false, NULL));
    /* Residency hint: nothing is resident outside SSD streaming. */
    REQUIRE(ds4_gpu_stream_expert_cache_resident_hint(0, 0) == 0);
    REQUIRE(ds4_gpu_stream_expert_cache_resident_hint(100000, 0) == 0);
    fprintf(stderr, "V4.1 lookahead units PASS\n");
    rc = 0;
done:
    unsetenv("DS4_METAL_V41_LOOKAHEAD_K");
    return rc;
}
#endif

```

In `main`, after the `--v41-moe-fuse-predicates` dispatch (inside its `#if defined(__APPLE__) && !defined(DS4_NO_GPU)` block), add:

```c
    if (argc == 2 && !strcmp(argv[1], "--v41-lookahead-units"))
        return check_v41_lookahead_units();
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `make tests/test_deepseek41_graph 2>&1 | grep -E " error" | head -3`
Expected: compile errors for the undeclared `ds41_la_k`, `ds41_la_topk`, `ds41_la_job` and `ds4_gpu_stream_expert_cache_resident_hint`.

- [ ] **Step 3: Implement the Metal hint**

In `ds4_metal.m`, right after the closing brace of `uint32_t ds4_gpu_stream_expert_cache_configured_count(void) { ... }`:

```objc
/* V4.1 lookahead prefetch asks whether an expert is cached before warming
 * its pages. A relaxed read from another thread: a stale answer costs one
 * redundant or one missing advisory read, never correctness. */
int ds4_gpu_stream_expert_cache_resident_hint(uint32_t layer, uint32_t expert) {
    if (!g_ssd_streaming_mode ||
        layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER ||
        expert >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT) {
        return 0;
    }
    return __atomic_load_n(&g_stream_expert_cache[layer][expert].valid, __ATOMIC_RELAXED) != 0;
}
```

In `ds4_gpu.h`, after the `ds4_gpu_stream_expert_cache_configured_count` declaration:

```c
/* Relaxed "is this expert cached" query for prefetch hints (0 outside SSD streaming). */
int ds4_gpu_stream_expert_cache_resident_hint(uint32_t layer, uint32_t expert);
```

- [ ] **Step 4: Implement the core block in `ds4.c`**

Directly above `static bool ds41_stream_async_load(const ds41_gpu_graph *g) {`, inside a new `#if defined(__APPLE__) && !defined(DS4_NO_GPU)` block:

```c
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
/* Lookahead expert prefetch (docs/superpowers/specs/2026-09-25-v41-lookahead-
 * prefetch-design.md): layer L+1's router on layer L's FFN input predicts
 * L+1's experts; a thread warms their pages in the OS page cache so the demand
 * read finds them resident. Selection and the expert cache are unchanged. */
#define DS41_LA_MAX_EMBD 8192u
#define DS41_LA_MAX_K 6u
#define DS41_LA_MAX_LAYER 64u

typedef struct {
    uint32_t layer, pos, k, n_embd, n_expert;
    const float *w, *bias;
    uint64_t gate_offset, up_offset, down_offset, gate_bytes, down_bytes;
    float x[DS41_LA_MAX_EMBD];
} ds41_la_job;

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    bool full;
    ds41_la_job job;
    uint64_t posted, dropped;
} ds41_la_mailbox;

static uint32_t ds41_la_k(void) {
    const char *v = getenv("DS4_METAL_V41_LOOKAHEAD_K");
    if (!v || !*v) return 1u;
    char *end = NULL;
    const long k = strtol(v, &end, 10);
    if (!end || *end) return 1u;
    return k < 0 ? 0u : k > (long)DS41_LA_MAX_K ? DS41_LA_MAX_K : (uint32_t)k;
}

/* kernel_dsv4_softplus_sqrt_f32_4's expression; ranking only, never output. */
static float ds41_la_score(float v, float bias) {
    const float ex = expf(v);
    const float em = ex < 0.03125f ? ex : 0.03125f;
    const float poly = em * (1.0f - em * (0.5f - em * (1.0f / 3.0f - 0.25f * em)));
    const float sp = v > 20.0f ? v : ex < 0.03125f ? poly : logf(1.0f + ex);
    return sqrtf(sp) + bias;
}

static uint32_t ds41_la_topk(const float *w, const float *bias, const float *x,
                             uint32_t n_expert, uint32_t n_embd, uint32_t k, int32_t *out) {
    float best_s[DS41_LA_MAX_K];
    uint32_t n = 0;
    if (k > DS41_LA_MAX_K) k = DS41_LA_MAX_K;
    if (k > n_expert) k = n_expert;
    for (uint32_t e = 0; e < n_expert && k; e++) {
        const float *row = w + (uint64_t)e * n_embd;
        float dot = 0.0f;
        for (uint32_t i = 0; i < n_embd; i++) dot += row[i] * x[i];
        const float s = ds41_la_score(dot, bias ? bias[e] : 0.0f);
        /* insertion into a sorted top-k: score desc, index asc on ties */
        uint32_t at = n;
        while (at > 0 && s > best_s[at - 1]) at--;
        if (at >= k) continue;
        const uint32_t last = n < k ? n : k - 1;
        for (uint32_t j = last; j > at; j--) { best_s[j] = best_s[j - 1]; out[j] = out[j - 1]; }
        best_s[at] = s; out[at] = (int32_t)e;
        if (n < k) n++;
    }
    return n;
}

static void ds41_la_ranges(const ds41_la_job *job, int32_t expert, uint64_t off[3], uint64_t len[3]) {
    const uint64_t e = (uint64_t)(uint32_t)expert;
    off[0] = job->gate_offset + e * job->gate_bytes; len[0] = job->gate_bytes;
    off[1] = job->up_offset + e * job->gate_bytes;   len[1] = job->gate_bytes;
    off[2] = job->down_offset + e * job->down_bytes; len[2] = job->down_bytes;
}

static void ds41_la_mailbox_init(ds41_la_mailbox *mb) {
    memset(mb, 0, sizeof(*mb));
    pthread_mutex_init(&mb->mu, NULL);
    pthread_cond_init(&mb->cv, NULL);
}

/* Never blocks the poster: a post replaces an unconsumed one. */
static void ds41_la_post(ds41_la_mailbox *mb, const ds41_la_job *job) {
    pthread_mutex_lock(&mb->mu);
    if (mb->full) mb->dropped++;
    mb->job = *job;
    mb->full = true;
    mb->posted++;
    pthread_cond_signal(&mb->cv);
    pthread_mutex_unlock(&mb->mu);
}

static bool ds41_la_take(ds41_la_mailbox *mb, ds41_la_job *out, bool wait, const volatile bool *stop) {
    pthread_mutex_lock(&mb->mu);
    while (wait && !mb->full && !(stop && *stop)) pthread_cond_wait(&mb->cv, &mb->mu);
    const bool got = mb->full;
    if (got) { *out = mb->job; mb->full = false; }
    pthread_mutex_unlock(&mb->mu);
    return got;
}
#endif

```

The scorer's insertion step already handles a new score that falls past k: `if (at >= k) continue;` runs before any shift.

- [ ] **Step 5: Run the tests to verify they pass**

Run:
```bash
make all tests/test_deepseek41_graph 2>&1 | grep -E " error|warning:" | head -3
./tests/test_deepseek41_graph --v41-lookahead-units 2>&1 | grep -E "PASS|\.c:[0-9]+:"
./tests/test_deepseek41_graph --decode-profile-format && ./tests/test_deepseek41_graph --v41-moe-fuse-predicates 2>&1 | grep PASS
```
Expected: a clean build, `V4.1 lookahead units PASS`, and the existing checks PASS.

- [ ] **Step 6: Commit**

```bash
git add ds4.c ds4_metal.m ds4_gpu.h tests/test_deepseek41_graph.c docs/superpowers/specs/2026-09-25-v41-lookahead-prefetch-design.md
git commit -m "ds4: V4.1 lookahead prefetch core (scorer, ranges, mailbox, cache hint)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TwriwxNUTW7k4x9zH3WeQT"
```

---

### Task 2: Prediction thread, hook, switch, counters, stop on engine close

**Files:**
- Modify: `ds4.c`:
  - the lookahead block from Task 1 (thread state, start/stop, post, note, print);
  - `ds41_moe_partial` (the hook inside the `if (async_started)` finish block, after `finish_ok` is known to be true);
  - `ds41_graph_step` (print next to `ds41_decode_profile_print`);
  - `ds4_engine_close` (stop first).
- Test: `tests/test_deepseek41_graph.c`, extending `--v41-lookahead-units`.

**Interfaces:**
- Consumes: Task 1's `ds41_la_*` and `ds4_gpu_stream_expert_cache_resident_hint`.
- Produces:
  - `static bool ds41_la_start(int model_fd);` (idempotent; false marks lookahead failed and prints once)
  - `static void ds41_la_stop(void);` (joins and closes the fd; a later start works again)
  - `static bool ds41_la_should_post(const ds41_gpu_graph *g, uint32_t il);`
  - `static void ds41_la_post_layer(ds41_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *next, uint32_t target, uint64_t gate_bytes, uint64_t down_bytes);`
  - `static void ds41_la_record(uint32_t layer, uint32_t pos, const int32_t *ids, uint32_t n);`
  - `static void ds41_la_note_selected(uint32_t layer, uint32_t pos, const int32_t *ids, uint32_t n);`
  - `static void ds41_la_print(FILE *fp);` prints `ds4: V4.1 lookahead: posted %llu dropped %llu predicted %llu issued %llu used %llu`.

- [ ] **Step 1: Extend the failing test**

In `check_v41_lookahead_units`, before `fprintf(stderr, "V4.1 lookahead units PASS\n");`, add:

```c
    /* Post gating: never the last layer; never without the switch; tp 1 streaming only. */
    {
        ds41_gpu_graph g = {.tp_world = 1, .streaming = true};
        unsetenv("DS4_METAL_DISABLE_V41_LOOKAHEAD");
        REQUIRE(ds41_la_should_post(&g, 0));
        REQUIRE(!ds41_la_should_post(&g, DS4_N_LAYER - 1));
        setenv("DS4_METAL_DISABLE_V41_LOOKAHEAD", "1", 1);
        REQUIRE(!ds41_la_should_post(&g, 0));
        unsetenv("DS4_METAL_DISABLE_V41_LOOKAHEAD");
        setenv("DS4_METAL_V41_LOOKAHEAD_K", "0", 1);
        REQUIRE(!ds41_la_should_post(&g, 0));
        unsetenv("DS4_METAL_V41_LOOKAHEAD_K");
        g.streaming = false;
        REQUIRE(!ds41_la_should_post(&g, 0));
    }
    /* "used" counts only an issued prediction for the same layer and position. */
    {
        const int32_t pred[1] = {7}, sel_hit[6] = {1, 2, 7, 4, 5, 6}, sel_miss[6] = {1, 2, 3, 4, 5, 6};
        const uint64_t used0 = g_ds41_la.used;
        ds41_la_record(3, 10, pred, 1);
        ds41_la_note_selected(3, 11, sel_hit, 6);      /* other position: no */
        REQUIRE(g_ds41_la.used == used0);
        ds41_la_note_selected(3, 10, sel_miss, 6);     /* not selected: no */
        REQUIRE(g_ds41_la.used == used0);
        ds41_la_note_selected(3, 10, sel_hit, 6);
        REQUIRE(g_ds41_la.used == used0 + 1);
    }
    /* Counter line format. */
    {
        char line[256] = "";
        FILE *fp = tmpfile();
        REQUIRE(fp);
        ds41_la_print(fp);
        rewind(fp);
        REQUIRE(fgets(line, sizeof(line), fp));
        fclose(fp);
        unsigned long long a, b, c, d, e;
        REQUIRE(sscanf(line, "ds4: V4.1 lookahead: posted %llu dropped %llu predicted %llu issued %llu used %llu",
                       &a, &b, &c, &d, &e) == 5);
    }
    /* Start, stop, restart: the thread joins and a new one starts. */
    {
        const int fd = open("/dev/null", O_RDONLY);
        REQUIRE(fd >= 0);
        REQUIRE(ds41_la_start(fd));
        REQUIRE(ds41_la_start(fd));                /* idempotent */
        ds41_la_stop();
        REQUIRE(ds41_la_start(fd));
        ds41_la_stop();
        close(fd);
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `make tests/test_deepseek41_graph 2>&1 | grep -E " error" | head -3`
Expected: undeclared `ds41_la_should_post`, `g_ds41_la`, `ds41_la_record`, `ds41_la_start`.

- [ ] **Step 3: Implement the thread, the gating and the counters**

Append inside the Task 1 block, before its `#endif`:

```c
static struct {
    pthread_mutex_t mu;        /* guards started/stop/fd and pred[] */
    pthread_t th;
    bool started, failed;
    volatile bool stop;
    int fd;
    ds41_la_mailbox mb;
    int32_t pred[DS41_LA_MAX_LAYER][DS41_LA_MAX_K];
    uint32_t pred_n[DS41_LA_MAX_LAYER], pred_pos[DS41_LA_MAX_LAYER];
    uint64_t predicted, issued, used;
} g_ds41_la = {.mu = PTHREAD_MUTEX_INITIALIZER, .fd = -1};

static void ds41_la_record(uint32_t layer, uint32_t pos, const int32_t *ids, uint32_t n) {
    if (layer >= DS41_LA_MAX_LAYER) return;
    pthread_mutex_lock(&g_ds41_la.mu);
    g_ds41_la.pred_n[layer] = n > DS41_LA_MAX_K ? DS41_LA_MAX_K : n;
    g_ds41_la.pred_pos[layer] = pos;
    for (uint32_t i = 0; i < g_ds41_la.pred_n[layer]; i++) g_ds41_la.pred[layer][i] = ids[i];
    pthread_mutex_unlock(&g_ds41_la.mu);
}

static void ds41_la_note_selected(uint32_t layer, uint32_t pos, const int32_t *ids, uint32_t n) {
    if (layer >= DS41_LA_MAX_LAYER) return;
    pthread_mutex_lock(&g_ds41_la.mu);
    if (g_ds41_la.pred_pos[layer] == pos) {
        for (uint32_t i = 0; i < g_ds41_la.pred_n[layer]; i++)
            for (uint32_t j = 0; j < n; j++)
                if (ids[j] == g_ds41_la.pred[layer][i]) { g_ds41_la.used++; break; }
        g_ds41_la.pred_n[layer] = 0;              /* count once */
    }
    pthread_mutex_unlock(&g_ds41_la.mu);
}

static void ds41_la_print(FILE *fp) {
    pthread_mutex_lock(&g_ds41_la.mb.mu);
    const unsigned long long posted = g_ds41_la.mb.posted, dropped = g_ds41_la.mb.dropped;
    pthread_mutex_unlock(&g_ds41_la.mb.mu);
    pthread_mutex_lock(&g_ds41_la.mu);
    fprintf(fp, "ds4: V4.1 lookahead: posted %llu dropped %llu predicted %llu issued %llu used %llu\n",
            posted, dropped, (unsigned long long)g_ds41_la.predicted,
            (unsigned long long)g_ds41_la.issued, (unsigned long long)g_ds41_la.used);
    pthread_mutex_unlock(&g_ds41_la.mu);
}

static void *ds41_la_main(void *arg) {
    (void)arg;
    static ds41_la_job job;   /* one thread: 32 KB off the stack */
    while (ds41_la_take(&g_ds41_la.mb, &job, true, &g_ds41_la.stop)) {
        int32_t ids[DS41_LA_MAX_K], issued[DS41_LA_MAX_K];
        uint32_t n_issued = 0;
        const uint32_t n = ds41_la_topk(job.w, job.bias, job.x, job.n_expert, job.n_embd, job.k, ids);
        for (uint32_t i = 0; i < n; i++) {
            if (ds4_gpu_stream_expert_cache_resident_hint(job.layer, (uint32_t)ids[i])) continue;
            uint64_t off[3], len[3];
            ds41_la_ranges(&job, ids[i], off, len);
            for (int r = 0; r < 3; r++) {
                struct radvisory ra = {.ra_offset = (off_t)off[r], .ra_count = (int)len[r]};
                (void)fcntl(g_ds41_la.fd, F_RDADVISE, &ra);
            }
            issued[n_issued++] = ids[i];
        }
        ds41_la_record(job.layer, job.pos, issued, n_issued);
        pthread_mutex_lock(&g_ds41_la.mu);
        g_ds41_la.predicted += n;
        g_ds41_la.issued += n_issued;
        pthread_mutex_unlock(&g_ds41_la.mu);
        if (g_ds41_la.stop) break;
    }
    return NULL;
}

static bool ds41_la_start(int model_fd) {
    pthread_mutex_lock(&g_ds41_la.mu);
    if (g_ds41_la.started || g_ds41_la.failed) {
        const bool ok = g_ds41_la.started;
        pthread_mutex_unlock(&g_ds41_la.mu);
        return ok;
    }
    ds41_la_mailbox_init(&g_ds41_la.mb);
    g_ds41_la.stop = false;
    g_ds41_la.fd = model_fd >= 0 ? dup(model_fd) : -1;
    if (g_ds41_la.fd < 0 || pthread_create(&g_ds41_la.th, NULL, ds41_la_main, NULL) != 0) {
        if (g_ds41_la.fd >= 0) close(g_ds41_la.fd);
        g_ds41_la.fd = -1;
        g_ds41_la.failed = true;
        pthread_mutex_unlock(&g_ds41_la.mu);
        fprintf(stderr, "ds4: V4.1 lookahead prefetch disabled (no fd or thread)\n");
        return false;
    }
    g_ds41_la.started = true;
    pthread_mutex_unlock(&g_ds41_la.mu);
    return true;
}

/* Before the model is unmapped: the thread holds pointers into it. */
static void ds41_la_stop(void) {
    pthread_mutex_lock(&g_ds41_la.mu);
    const bool started = g_ds41_la.started;
    pthread_mutex_unlock(&g_ds41_la.mu);
    if (!started) return;
    pthread_mutex_lock(&g_ds41_la.mb.mu);
    g_ds41_la.stop = true;
    pthread_cond_broadcast(&g_ds41_la.mb.cv);
    pthread_mutex_unlock(&g_ds41_la.mb.mu);
    pthread_join(g_ds41_la.th, NULL);
    pthread_mutex_lock(&g_ds41_la.mu);
    close(g_ds41_la.fd);
    g_ds41_la.fd = -1;
    g_ds41_la.started = false;
    for (uint32_t l = 0; l < DS41_LA_MAX_LAYER; l++) g_ds41_la.pred_n[l] = 0;
    pthread_mutex_unlock(&g_ds41_la.mu);
}

static bool ds41_la_should_post(const ds41_gpu_graph *g, uint32_t il) {
    return g->streaming && !g->quality && !g->imatrix && g->tp_world == 1 &&
        il + 1u < DS4_N_LAYER && il + 1u < DS41_LA_MAX_LAYER &&
        !getenv("DS4_METAL_DISABLE_V41_LOOKAHEAD") && ds41_la_k() > 0;
}

/* Main thread, layer il's ids known: hand layer il+1's router its input. */
static void ds41_la_post_layer(ds41_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *next,
                               uint32_t target, uint64_t gate_bytes, uint64_t down_bytes) {
    static ds41_la_job job;   /* main thread only */
    if (DS4_N_EMBD > DS41_LA_MAX_EMBD || !next->ffn_gate_inp ||
        next->ffn_gate_inp->type != DS4_TENSOR_F32 || !next->ffn_exp_probs_b ||
        !ds41_la_start(m->fd)) return;
    job.layer = target;
    job.pos = g->pos;
    job.k = ds41_la_k();
    job.n_embd = DS4_N_EMBD;
    job.n_expert = DS4_N_EXPERT;
    job.w = (const float *)(const void *)(m->map + next->ffn_gate_inp->abs_offset);
    job.bias = (const float *)(const void *)(m->map + next->ffn_exp_probs_b->abs_offset);
    job.gate_offset = next->ffn_gate_exps->abs_offset;
    job.up_offset = next->ffn_up_exps->abs_offset;
    job.down_offset = next->ffn_down_exps->abs_offset;
    job.gate_bytes = gate_bytes;
    job.down_bytes = down_bytes;
    memcpy(job.x, ds4_gpu_tensor_contents(g->norm), (size_t)DS4_N_EMBD * sizeof(float));
    ds41_la_post(&g_ds41_la.mb, &job);
}
```

`ds4.c` already includes `<fcntl.h>` and `<unistd.h>` at the top (lines 18 and 42), which `F_RDADVISE`, `dup` and `close` need.

- [ ] **Step 4: Hook, print, stop**

In `ds41_moe_partial`, inside `if (async_started) { ... }`, replace:

```c
        if (!flush_ok || !finish_ok) {
            ds41_stream_async_abandon(&async_load);
            return false;
        }
    }
```

with:

```c
        if (!flush_ok || !finish_ok) {
            ds41_stream_async_abandon(&async_load);
            return false;
        }
        /* Lookahead: count a used prediction for this layer, then hand the
         * next layer's router this layer's FFN input (router done on the GPU,
         * nothing that rewrites g->norm is encoded yet). */
        ds41_la_note_selected(il, g->pos, async_load.selected_ids, DS4_N_EXPERT_USED);
        if (ds41_la_should_post(g, il) && !ds41_image_at(g, g->pos))
            ds41_la_post_layer(g, m, l + 1, il + 1u, gate_row * DS4_N_FF_EXP, down_row * DS4_N_EMBD);
    }
```

In `ds41_graph_step`, replace `if (p->tokens % 64u == 0u) ds41_decode_profile_print(stderr, p);` with:

```c
        if (p->tokens % 64u == 0u) {
            ds41_decode_profile_print(stderr, p);
            ds41_la_print(stderr);
        }
```

In `ds4_engine_close`, as the first statement after `if (!e) return;`:

```c
#if defined(DS4_HAS_DEEPSEEK41_GPU) && defined(__APPLE__) && !defined(DS4_NO_GPU)
    ds41_la_stop();   /* before the model is unmapped */
#endif
```

- [ ] **Step 5: Run the tests**

Run:
```bash
make all tests/test_deepseek41_graph 2>&1 | grep -E " error|warning:" | head -3
./tests/test_deepseek41_graph --v41-lookahead-units 2>&1 | grep -E "PASS|\.c:[0-9]+:"
for t in --decode-profile-format --router-log-format --stream-async-abandon --stream-control-env --v41-fuse-switches --v41-moe-fuse-predicates; do ./tests/test_deepseek41_graph $t 2>&1 | grep -E "PASS|\.c:[0-9]+:" | tail -1; done
python3 -m unittest discover -s speed-bench/tests 2>&1 | grep -E "^Ran|^OK|^FAILED"
```
Expected: a clean build, every line PASS, and `OK`.

- [ ] **Step 6: [GPU] Smoke run with counters**

```bash
caffeinate -i -s env DS4_V41_DECODE_PROFILE=1 DS4_METAL_GPU_BUSY_PROFILE=1 DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1 \
  ./ds4-bench --metal -m "$M" --prompt-file $P/switch.txt --ctx-start 8192 --ctx-max 8192 --gen-tokens 128 \
  --teacher-forced-decode --ssd-streaming --csv /tmp/la-smoke.csv 2>&1 | grep -E "V4.1 lookahead:|V4.1 decode profile:" | tail -2
```
Expected:
- posted ≈ 39 × 128;
- issued ≈ 0.13 × predicted or more (predicted = posted − dropped at k = 1);
- used / issued roughly 0.3 (the spike);
- no crash, and the process exits normally (the thread was joined).

- [ ] **Step 7: Commit**

```bash
git add ds4.c tests/test_deepseek41_graph.c
git commit -m "ds4: V4.1 lookahead prefetch thread, hook and counters

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TwriwxNUTW7k4x9zH3WeQT"
```

---

### Task 3: A/B terms for the lookahead counters

**Files:**
- Modify: `speed-bench/v41/phase0.py` (new `LOOKAHEAD_RE`, `parse_lookahead`, fields in `run_one`, `FIELDS`)
- Modify: `speed-bench/v41/ab.py` (`TERMS`)
- Test: `speed-bench/tests/test_phase0.py`, `speed-bench/tests/test_ab.py`

**Interfaces:**
- Produces: `phase0.parse_lookahead(text) -> dict|None` with keys `posted, dropped, predicted, issued, used` from the last counter line. The `run_one` row gains `la_issued_per_tok`, `la_used_per_tok` and `la_dropped` when both the profile and the counter line are present. `ab.TERMS` gains `la_issued_per_tok` and `la_used_per_tok`.

- [ ] **Step 1: Write the failing tests**

Append to `class AbHelpersTest` in `speed-bench/tests/test_phase0.py`:

```python
    def test_parse_lookahead_takes_last_line(self):
        text = ("ds4: V4.1 lookahead: posted 10 dropped 0 predicted 10 issued 3 used 1\n"
                "ds4: V4.1 lookahead: posted 4992 dropped 12 predicted 4980 issued 1500 used 480\n")
        self.assertEqual(phase0.parse_lookahead(text),
                         {"posted": 4992, "dropped": 12, "predicted": 4980, "issued": 1500, "used": 480})
        self.assertIsNone(phase0.parse_lookahead("nothing"))
```

In `speed-bench/tests/test_ab.py`, add to `class AbTest`:

```python
    def test_terms_include_lookahead(self):
        self.assertIn("la_issued_per_tok", ab.TERMS)
        self.assertIn("la_used_per_tok", ab.TERMS)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m unittest discover -s speed-bench/tests 2>&1 | grep -E "^(ERROR|FAIL):|^Ran|^FAILED"`
Expected: FAILED, with no `parse_lookahead` and the TERMS assertions failing.

- [ ] **Step 3: Implement**

In `phase0.py`, after `parse_readahead`:

```python
LOOKAHEAD_RE = re.compile(r"ds4: V4\.1 lookahead: posted (\d+) dropped (\d+) predicted (\d+) "
                          r"issued (\d+) used (\d+)")


def parse_lookahead(text):
    rows = LOOKAHEAD_RE.findall(text)
    if not rows:
        return None
    return dict(zip(("posted", "dropped", "predicted", "issued", "used"), (int(x) for x in rows[-1])))
```

In `run_one`, after the readahead block:

```python
    la = parse_lookahead(stderr)
    tokens = (parse_profile(stderr) or {}).get("tokens")
    if la and tokens:
        row["la_issued_per_tok"] = la["issued"] / tokens
        row["la_used_per_tok"] = la["used"] / tokens
        row["la_dropped"] = la["dropped"]
```

In `FIELDS`, after `"readahead_ms", "host_ms",` add `"la_issued_per_tok", "la_used_per_tok", "la_dropped",`.

In `ab.py`, make `TERMS`:

```python
TERMS = ("step_ms", "gpu_busy_ms", "pread_ms", "readahead_ms", "host_ms",
         "decode_hit_rate", "wired_steady_gib", "la_issued_per_tok", "la_used_per_tok")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest discover -s speed-bench/tests 2>&1 | grep -E "^(ERROR|FAIL):|^Ran|^OK|^FAILED"`
Expected: `OK`, 101 tests.

- [ ] **Step 5: Commit**

```bash
git add speed-bench/v41/phase0.py speed-bench/v41/ab.py speed-bench/tests/test_phase0.py speed-bench/tests/test_ab.py
git commit -m "speed-bench/v41: lookahead counters in the A/B terms

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TwriwxNUTW7k4x9zH3WeQT"
```

---

### Task 4: [GPU] Exactness, async-off gating and the Qwen fast tier

**Files:** none (verification). Results go into the ledger and later into Task 6's write-up.

- [ ] **Step 1: [GPU] Exactness at 8 and 24 GB, and the self-test**

```bash
T=./tests/test_deepseek41_graph; F=speed-bench/promessi_sposi.txt
$T "$M" --stream-control $F DS4_METAL_DISABLE_V41_LOOKAHEAD 8
$T "$M" --stream-control $F DS4_METAL_DISABLE_V41_LOOKAHEAD 24
$T "$M" --stream-control-selftest $F
```
Expected:
- two PASS lines per `--stream-control` run;
- the selftest reports its three planted differences caught (history, logits, state).

- [ ] **Step 2: [GPU] Async load off posts nothing (Review Focus 3)**

```bash
caffeinate -i -s env DS4_METAL_DISABLE_V41_ASYNC_LOAD=1 DS4_V41_DECODE_PROFILE=1 \
  ./ds4-bench --metal -m "$M" --prompt-file $P/switch.txt --ctx-start 4096 --ctx-max 4096 --gen-tokens 64 \
  --teacher-forced-decode --ssd-streaming --csv /tmp/la-async-off.csv 2>&1 | grep "V4.1 lookahead:" | tail -1
```
Expected: `posted 0 dropped 0 predicted 0 issued 0 used 0`.

- [ ] **Step 3: [GPU] Qwen fast tier (`ds4_metal.m` changed)**

Run: `speed-bench/qwen-regression/run.sh fast`
Expected: `qwen_gate: PASS`.

- [ ] **Step 4: Record**

Write the PASS lines and the counter lines into the ledger. There is no code commit in this task.

---

### Task 5: [GPU] A/B, k, GPU stage check, default decision

**Files:**
- Create: `speed-bench/v41/lookahead-<YYYYMMDD>/` (A/B JSON copies and `probes.md`)
- Modify: `ds4.c` only if the A/B says the default must change. That means making lookahead opt-in, or changing the default k in `ds41_la_k`; each is a one-line change covered by the Task 1 test, which must be updated first (TDD).

- [ ] **Step 1: [GPU] Lookahead off vs on (k = 1), auto cache and 24 GB**

```bash
R=$HOME/orca/workspaces/ds4-metal-data/v41-lookahead/$(date +%Y%m%d)
caffeinate -i -s python3 speed-bench/v41/ab.py --label la-k1-auto --model "$M" --prompts "$P" --out "$R" \
  --a-cache auto --a-env DS4_METAL_DISABLE_V41_LOOKAHEAD=1
caffeinate -i -s python3 speed-bench/v41/ab.py --label la-k1-24 --model "$M" --prompts "$P" --out "$R" \
  --a-cache 24 --a-env DS4_METAL_DISABLE_V41_LOOKAHEAD=1
```
Expected: two summaries, with no CONTAMINATED and no swap. Read the B side's `la_used_per_tok` / `la_issued_per_tok` (≈ 0.3 expected) and its `readahead_ms` and `pread_ms` (they should fall).

- [ ] **Step 2: [GPU] k = 2 vs k = 1, auto cache**

```bash
caffeinate -i -s python3 speed-bench/v41/ab.py --label la-k2 --model "$M" --prompts "$P" --out "$R" \
  --a-cache auto --b-env DS4_METAL_V41_LOOKAHEAD_K=2
```
Expected: a summary. k = 2 wins only if its ratio is ≥ 1.01.

- [ ] **Step 3: [GPU] GPU stage check with lookahead on**

```bash
caffeinate -i -s env DS4_METAL_GPU_STAGE_TIMESTAMPS=1 DS4_METAL_GPU_STAGE_TIMESTAMPS_DETAIL=1 DS4_V41_DECODE_PROFILE=1 \
  ./ds4-bench --metal -m "$M" --prompt-file $P/switch.txt --ctx-start 8192 --ctx-max 8192 --gen-tokens 48 \
  --teacher-forced-decode --ssd-streaming --csv $R/stages.csv > $R/stages.stderr 2>&1
python3 speed-bench/v41/stages.py $R/stages.stderr --min-pos 8200
```
Expected: GPU busy within 1 ms of `speed-bench/v41/fusions-20260925/stages-after.txt` (52.65 ms/token). The predictor runs on the CPU, so the GPU should not grow.

- [ ] **Step 4: Decide and record**

Copy `$R/ab-*.json` to `speed-bench/v41/lookahead-<date>/`. Write `probes.md` with each A/B's ratio, the B terms (readahead_ms, pread_ms, la_issued_per_tok, la_used_per_tok, gpu_busy_ms) and the decision:
- **Keep on by default** if `la-k1-auto` has ratio > 1.00; otherwise make it opt-in.
- **Set k = 2** as the default if `la-k2` has ratio ≥ 1.01 (change `ds41_la_k`'s default and the Task 1 test's default assertion).

```bash
git add speed-bench/v41/lookahead-*/ ds4.c tests/test_deepseek41_graph.c
git commit -m "speed-bench/v41: V4.1 lookahead prefetch A/B and default

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TwriwxNUTW7k4x9zH3WeQT"
```

---

### Task 6: [GPU] Final check, results, spec status, memory, restore

**Files:**
- Create: `speed-bench/v41/lookahead-<date>/RESULTS.md`
- Modify: `docs/V41_64GB_BUILD.md` §0 (one status bullet), memory `v41-lookahead-spike.md`

- [ ] **Step 1: [GPU] Final A/B against the Phase-0 configuration**

```bash
caffeinate -i -s python3 speed-bench/v41/ab.py --label final-la --model "$M" --prompts "$P" --out "$R" \
  --a-cache 24 --a-env DS4_METAL_DISABLE_V41_STREAM_DECODE_QUEUE=1 --a-env DS4_METAL_DISABLE_V41_ASYNC_LOAD=1 \
  --a-env DS4_METAL_DISABLE_V41_HC_FUSE=1 --a-env DS4_METAL_DISABLE_V41_LOOKAHEAD=1 --b-cache auto
```
Expected: B is the shipped defaults. Report B t/s against the 12 t/s target.

- [ ] **Step 2: Write the results and update the status**

`RESULTS.md` holds:
- setup (commit, model, env, cache);
- the A/B table from `probes.md`;
- the counters, with precision used/issued against the spike's ~0.33;
- the new per-token decomposition (readahead, pread, GPU, host);
- the final A/B;
- whether 12 t/s is met;
- the next lever (approach 2 staging if the pread term stays large).

In `docs/V41_64GB_BUILD.md` §0, add a bullet "Lookahead prefetch (page cache) <date>: <ratio>, <t/s>".

Update memory `v41-lookahead-spike.md` with the measured result and the decision.

```bash
git add speed-bench/v41/lookahead-*/RESULTS.md docs/V41_64GB_BUILD.md
git commit -m "docs: V4.1 lookahead prefetch results

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TwriwxNUTW7k4x9zH3WeQT"
```

- [ ] **Step 3: Restore oMLX and the watchdogs** (Global Constraints).
