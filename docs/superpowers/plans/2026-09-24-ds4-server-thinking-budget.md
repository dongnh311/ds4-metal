# ds4-server Hard Thinking Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a hard, per-request thinking budget to ds4-server that closes `<think>` gracefully after N generated reasoning tokens and lets the model answer in the same generation, with output byte-identical to today when the budget is off.

**Architecture:** Pure helpers (effective budget, counter, tool-open check, forced-suffix text) plus request/CLI plumbing, then a small change to `generate_job_inner`'s decode loop: when the budget is spent, the current MTP block is cut through the existing `kept < ntok` rewind, and the forced suffix (`"\n\n" + message + "\n</think>\n\n"`) is fed one token per loop iteration through `server_eval_token` and the unchanged per-token body, so streaming, thinking state and the live cache stay consistent.

**Tech Stack:** C (ds4_server.c, ds4_help.c), the `ds4_test --server` unit group, Python 3 stdlib live test and measurement scripts.

**Spec:** docs/superpowers/specs/2026-09-24-ds4-server-thinking-budget-design.md

## Global Constraints

- Budget off (no `--think-budget`, no request budget) = output byte-identical to the pre-change binary.
- Server flags: `--think-budget N` (default 0 = off), `--think-budget-message TEXT` (default Qwen's sentence
  `Considering the limited time by the user, I have to give the solution based on the thinking directly now.`).
- Request fields: Anthropic/Chat `thinking.budget_tokens`; Chat and Responses top-level `thinking_budget`;
  Chat `chat_template_kwargs.thinking_budget`. Effective budget = smaller positive value of server and request.
- Budget applies only with thinking enabled, only to tokens generated inside `<think>`; ignored with
  `--batched-session` (startup warning).
- Forced suffix tokenized with `ds4_tokenize_rendered_chat` so `</think>` is the vocabulary's `think_end_id`.
- No gateway (AI-Gateway-MLX) code changes; no Metal kernel changes.
- Mac only; GPU-using steps (live test, regression gate, measurement) run only when the DS4.1 session is not
  using the GPU and the PROD gateway ds4 slot is stopped, with the user's go-ahead.
- Code, comments, docs and commit messages in English; commits end with
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_017f1tsCGFkVQ3wrmPsJ7FBn`.
- Work on branch `think-budget` in /Users/dongnh/orca/workspaces/ds4-metal/kv-grow; never merge or push without
  the user's approval (local develop also carries the DS4.1 merge 2152f86, which is not pushed yet).

## Review Focus

1. Budget reached while the model is writing a tool call inside `<think>` — expect the close to wait until the
   tool block ends (Task 1 `test_think_budget_tool_open`).
2. `max_tokens` smaller than the budget — expect a normal `length` finish with no forced close (Task 4 check 5).
3. Non-thinking request carrying a budget — expect byte-identical output to the same request without it
   (Task 4 check 6).
4. Malformed budgets (string, negative, zero) — expect invalid JSON for non-numbers and "no budget" for <= 0
   (Task 2 `test_think_budget_request_fields`).
5. Stream vs non-stream and MTP on vs off — expect identical text and a working forced close in both (Task 4
   checks 2 and 7).

---

### Task 1: Budget helpers

**Files:**
- Modify: `ds4_server.c` (insert helpers after `thinking_state_from_prompt`, ~line 12206; tests near
  `test_api_thinking_controls_parse`, ~line 18268; register tests in `ds4_server_unit_tests_run`, ~line 22954)

**Interfaces:**
- Consumes: `find_last_substr`, `find_any_tool_start`, `find_any_tool_end`, `buf`, `buf_puts`, `buf_take`
  (all existing in ds4_server.c).
- Produces:
  - `#define DS4_THINK_BUDGET_DEFAULT_MESSAGE "Considering the limited time by the user, I have to give the solution based on the thinking directly now."`
  - `typedef struct { int limit; int used; bool fired; } think_budget_state;`
  - `static int think_budget_effective(int server_limit, int request_limit);`
  - `static bool think_budget_due(think_budget_state *b, bool inside_after);`
  - `static bool think_budget_tool_open(const char *text);`
  - `static char *think_budget_suffix_text(const char *message);` (caller frees)

- [ ] **Step 1: Write the failing tests**

Insert after the closing `}` of `test_api_thinking_controls_parse` (before `static void test_render_think_max_prompt_prefix(void) {`):

```c
static void test_think_budget_rules(void) {
    TEST_ASSERT(think_budget_effective(0, 0) == 0);
    TEST_ASSERT(think_budget_effective(4096, 0) == 4096);
    TEST_ASSERT(think_budget_effective(0, 512) == 512);
    TEST_ASSERT(think_budget_effective(4096, 512) == 512);
    TEST_ASSERT(think_budget_effective(512, 4096) == 512);

    think_budget_state b = {.limit = 3};
    TEST_ASSERT(!think_budget_due(&b, false));   /* outside <think>: not counted */
    TEST_ASSERT(b.used == 0);
    TEST_ASSERT(!think_budget_due(&b, true));
    TEST_ASSERT(!think_budget_due(&b, true));
    TEST_ASSERT(think_budget_due(&b, true));     /* the third token inside spends it */
    TEST_ASSERT(b.used == 3);
    TEST_ASSERT(think_budget_due(&b, true));     /* still due while the close waits */
    b.fired = true;
    const int used = b.used;
    TEST_ASSERT(!think_budget_due(&b, true));    /* forced tokens never count */
    TEST_ASSERT(b.used == used);

    think_budget_state off = {0};
    for (int i = 0; i < 10; i++) TEST_ASSERT(!think_budget_due(&off, true));
    TEST_ASSERT(off.used == 10);                 /* counted for the log even when off */
}

static void test_think_budget_tool_open(void) {
    TEST_ASSERT(!think_budget_tool_open(NULL));
    TEST_ASSERT(!think_budget_tool_open("Let me think about primes."));
    TEST_ASSERT(think_budget_tool_open("I will call it: <tool_call>{\"name\":\"ls\""));
    TEST_ASSERT(!think_budget_tool_open("<tool_call>{\"name\":\"ls\"}</tool_call> then more"));
    TEST_ASSERT(think_budget_tool_open("<tool_call>a</tool_call> and <tool_call>b"));
    TEST_ASSERT(!think_budget_tool_open("old <tool_call> before <think> new reasoning"));
}

static void test_think_budget_suffix_text(void) {
    char *s = think_budget_suffix_text(NULL);
    TEST_ASSERT(!strcmp(s, "\n\n" DS4_THINK_BUDGET_DEFAULT_MESSAGE "\n</think>\n\n"));
    free(s);
    s = think_budget_suffix_text("");
    TEST_ASSERT(!strcmp(s, "\n\n" DS4_THINK_BUDGET_DEFAULT_MESSAGE "\n</think>\n\n"));
    free(s);
    s = think_budget_suffix_text("Wrap up now.");
    TEST_ASSERT(!strcmp(s, "\n\nWrap up now.\n</think>\n\n"));
    free(s);
}
```

In `ds4_server_unit_tests_run`, after the line `    test_api_thinking_controls_parse();` add:

```c
    test_think_budget_rules();
    test_think_budget_tool_open();
    test_think_budget_suffix_text();
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `make ds4_test 2>&1 | tail -5`
Expected: FAIL — compile errors, `think_budget_effective` / `think_budget_state` undeclared.

- [ ] **Step 3: Implement the helpers**

Insert after the closing `}` of `thinking_state_from_prompt` (before the comment `/* A completed tool block inside unclosed reasoning can be recovered without`):

```c
/* Hard thinking budget (--think-budget, thinking.budget_tokens, thinking_budget).
 * Qwen's documented fallback: when reasoning reaches the budget, append this
 * sentence, close </think>, and let the model answer. */
#define DS4_THINK_BUDGET_DEFAULT_MESSAGE \
    "Considering the limited time by the user, I have to give the solution based on the thinking directly now."

typedef struct {
    int limit;   /* effective budget in generated tokens; 0 = unlimited */
    int used;    /* generated tokens counted inside <think> */
    bool fired;  /* the forced close has been queued */
} think_budget_state;

/* The smaller positive value wins; 0 means no budget on that side. */
static int think_budget_effective(int server_limit, int request_limit) {
    if (server_limit > 0 && request_limit > 0)
        return server_limit < request_limit ? server_limit : request_limit;
    if (server_limit > 0) return server_limit;
    return request_limit > 0 ? request_limit : 0;
}

/* Count one kept token; true while the budget is spent and the close is due.
 * Counting stops once the close fired, so forced tokens never count. */
static bool think_budget_due(think_budget_state *b, bool inside_after) {
    if (!b || b->fired || !inside_after) return false;
    b->used++;
    return b->limit > 0 && b->used >= b->limit;
}

/* True when the reasoning so far opened a tool-call block that has not closed:
 * a forced </think> there would cut tool syntax, so the close waits. */
static bool think_budget_tool_open(const char *text) {
    if (!text) return false;
    const char *start = find_last_substr(text, "<think>");
    if (!start) start = text;
    const char *open = NULL;
    for (const char *p = find_any_tool_start(start); p; p = find_any_tool_start(p + 1))
        open = p;
    return open && !find_any_tool_end(open);
}

/* "\n\n<message>\n</think>\n\n"; the caller frees the result. */
static char *think_budget_suffix_text(const char *message) {
    buf b = {0};
    buf_puts(&b, "\n\n");
    buf_puts(&b, message && message[0] ? message : DS4_THINK_BUDGET_DEFAULT_MESSAGE);
    buf_puts(&b, "\n</think>\n\n");
    return buf_take(&b);
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `make ds4_test 2>&1 | grep -iE "warning|error" ; ./ds4_test --server`
Expected: no warnings or errors; output ends with `server: OK` and `ds4 tests: ok`.

- [ ] **Step 5: Commit**

```bash
git add ds4_server.c
git commit -m "ds4-server: thinking budget helpers

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017f1tsCGFkVQ3wrmPsJ7FBn"
```

---

### Task 2: Request fields, server flags, help and docs

**Files:**
- Modify: `ds4_server.c` (request struct ~line 829; `parse_thinking_control_value` ~1075;
  `parse_chat_template_kwargs` ~1116; `parse_chat_request` ~4099; `parse_anthropic_request` ~4296;
  `parse_responses_request` ~5285; `parse_completion_request` ~5591; `struct server` ~10141;
  `server_config` ~15419; `parse_options` ~15573; server setup in `main` ~15955; unit tests)
- Modify: `ds4_help.c` (`print_server_thinking`)
- Modify: `docs/SERVER.md` (APIs section)

**Interfaces:**
- Consumes: Task 1 helpers (none directly).
- Produces:
  - `request.think_budget` (int, 0 = none)
  - `static bool parse_thinking_control_value(const char **p, bool *thinking_enabled, int *budget_tokens);`
    (`budget_tokens` may be NULL)
  - `static bool parse_chat_template_kwargs(const char **p, bool *thinking_enabled, bool *got_thinking, ds4_think_mode *effort, int *think_budget);`
    (`think_budget` may be NULL)
  - `server.think_budget` (int), `server.think_budget_message` (const char *, NULL = default)

- [ ] **Step 1: Write the failing test**

Insert after `test_think_budget_suffix_text` (from Task 1):

```c
static void test_think_budget_request_fields(void) {
    bool enabled = false;
    int budget = 0;
    const char *p = "{\"type\":\"enabled\",\"budget_tokens\":2048}";
    TEST_ASSERT(parse_thinking_control_value(&p, &enabled, &budget));
    TEST_ASSERT(enabled && budget == 2048);

    p = "{\"type\":\"enabled\",\"budget_tokens\":\"many\"}";
    TEST_ASSERT(!parse_thinking_control_value(&p, &enabled, &budget));

    budget = 7;
    p = "{\"type\":\"enabled\",\"budget_tokens\":-5}";
    TEST_ASSERT(parse_thinking_control_value(&p, &enabled, &budget));
    TEST_ASSERT(budget == 0);                    /* json_int folds negatives to 0 = none */

    p = "{\"type\":\"enabled\",\"budget_tokens\":64}";
    TEST_ASSERT(parse_thinking_control_value(&p, &enabled, NULL));  /* completions ignore it */

    bool got = false;
    ds4_think_mode mode = DS4_THINK_HIGH;
    budget = 0;
    const char *kw = "{\"enable_thinking\":true,\"thinking_budget\":1024}";
    TEST_ASSERT(parse_chat_template_kwargs(&kw, &enabled, &got, &mode, &budget));
    TEST_ASSERT(got && enabled && budget == 1024);
    kw = "{\"thinking_budget\":1024}";
    TEST_ASSERT(parse_chat_template_kwargs(&kw, &enabled, &got, &mode, NULL));

    request r;
    request_init(&r, REQ_CHAT, 128);
    TEST_ASSERT(r.think_budget == 0);
    request_free(&r);
}
```

Register it in `ds4_server_unit_tests_run` right after `test_think_budget_suffix_text();`:

```c
    test_think_budget_request_fields();
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `make ds4_test 2>&1 | tail -5`
Expected: FAIL — `too many arguments to function call` for `parse_thinking_control_value` /
`parse_chat_template_kwargs`, and `no member named 'think_budget'`.

- [ ] **Step 3: Implement the request field and parsers**

In the `request` struct, after `    ds4_think_mode think_mode;` add:

```c
    int think_budget;       /* request thinking cap in tokens; 0 = none */
```

Replace the head and the `type` branch of `parse_thinking_control_value`:

```c
static bool parse_thinking_control_value(const char **p, bool *thinking_enabled,
                                         int *budget_tokens) {
```

and inside its key loop, replace

```c
            if (!strcmp(type, "enabled")) *thinking_enabled = true;
            else if (!strcmp(type, "disabled")) *thinking_enabled = false;
            free(type);
        } else if (!json_skip_value(p)) {
```

with

```c
            if (!strcmp(type, "enabled")) *thinking_enabled = true;
            else if (!strcmp(type, "disabled")) *thinking_enabled = false;
            free(type);
        } else if (!strcmp(key, "budget_tokens") && budget_tokens) {
            if (!json_int(p, budget_tokens)) {
                free(key);
                return false;
            }
        } else if (!json_skip_value(p)) {
```

Change `parse_chat_template_kwargs` to take the budget, and update its comment:

```c
/* chat_template_kwargs as the Qwen3.8 model card documents them: enable_thinking
 * and reasoning_effort are applied, thinking_budget sets the request's thinking
 * cap, other keys are ignored */
static bool parse_chat_template_kwargs(const char **p, bool *thinking_enabled, bool *got_thinking,
                                       ds4_think_mode *effort, int *think_budget) {
```

and in its key loop replace

```c
        } else if (!strcmp(key, "reasoning_effort")) {
            ok = parse_reasoning_effort_value(p, effort);
        } else {
```

with

```c
        } else if (!strcmp(key, "reasoning_effort")) {
            ok = parse_reasoning_effort_value(p, effort);
        } else if (!strcmp(key, "thinking_budget") && think_budget) {
            ok = json_int(p, think_budget);
        } else {
```

Update the call sites:
- `parse_chat_request`: `parse_thinking_control_value(&p, &thinking_enabled, &r->think_budget)` and
  `parse_chat_template_kwargs(&p, &thinking_enabled, &got_thinking, &reasoning_effort, &r->think_budget)`.
- `parse_anthropic_request`: `parse_thinking_control_value(&p, &thinking_enabled, &r->think_budget)`.
- `parse_completion_request`: `parse_thinking_control_value(&p, &thinking_enabled, NULL)`.
- Existing tests: in `test_api_thinking_controls_parse` the two calls become
  `parse_thinking_control_value(&thinking, &enabled, NULL)`; the two `parse_chat_template_kwargs(&kwargs, &enabled, &got, &mode)`
  calls (~line 18442 and 18445) become `parse_chat_template_kwargs(&kwargs, &enabled, &got, &mode, NULL)`.

In `parse_chat_request`, right after the `think` key branch

```c
        } else if (!strcmp(key, "think")) {
            if (!json_bool(&p, &thinking_enabled)) {
                free(key);
                goto bad;
            }
            got_thinking = true;
```

add

```c
        } else if (!strcmp(key, "thinking_budget")) {
            if (!json_int(&p, &r->think_budget)) {
                free(key);
                goto bad;
            }
```

In `parse_responses_request`, right before `        } else if (!strcmp(key, "reasoning")) {` add

```c
        } else if (!strcmp(key, "thinking_budget")) {
            if (!json_int(&p, &r->think_budget)) {
                free(key);
                goto bad;
            }
```

(Check with `grep -n 'key, "think")' ds4_server.c` that the first edit lands inside `parse_chat_request`, whose
body starts at `static bool parse_chat_request(`.)

- [ ] **Step 4: Implement the server flags**

In `server_config`, after `    int mixed_prefill_quantum;` add:

```c
    int think_budget;
    const char *think_budget_message;
```

In `parse_options`, right after the `--mixed-prefill-quantum` branch add:

```c
        } else if (!strcmp(arg, "--think-budget")) {
            c.think_budget = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--think-budget-message")) {
            c.think_budget_message = need_arg(&i, argc, argv, arg);
```

In `struct server`, after its `    bool enable_cors;` (the one inside `struct server {`, ~line 10155) add:

```c
    int think_budget;                   /* --think-budget; 0 = off */
    const char *think_budget_message;   /* NULL = DS4_THINK_BUDGET_DEFAULT_MESSAGE */
```

In `main`, after `    s.enable_cors = cfg.enable_cors;` add:

```c
    s.think_budget = cfg.think_budget;
    s.think_budget_message = cfg.think_budget_message;
    if (s.batched_mode && s.think_budget > 0) {
        server_log(DS4_LOG_WARNING,
                   "ds4-server: --think-budget is ignored with --batched-session");
    }
```

- [ ] **Step 5: Help and docs**

In `ds4_help.c` `print_server_thinking`, after the line
`    para(fp, c, "In thinking mode, client sampling knobs are ignored like the official API.");` add:

```c
    opt(fp, c, "--think-budget N", "Close reasoning after N generated tokens, then answer. 0 disables. Default: 0");
    opt(fp, c, "--think-budget-message TEXT", "Text inserted before the forced </think>. Default: Qwen's fallback sentence.");
    para(fp, c, "Requests may lower the cap with thinking.budget_tokens, thinking_budget, or chat_template_kwargs.thinking_budget.");
```

In `docs/SERVER.md`, after the paragraph ending
`thinking object, or a non-thinking model alias for direct answers.` add a blank line and:

```markdown
A hard reasoning cap is off by default. `--think-budget N` closes the reasoning
after N generated tokens: the server inserts the `--think-budget-message` text
(default: Qwen's "Considering the limited time by the user, I have to give the
solution based on the thinking directly now."), then `</think>`, and the model
answers in the same generation. A request can lower the cap with
`thinking.budget_tokens`, `thinking_budget`, or
`chat_template_kwargs.thinking_budget`; the smaller value wins. Forced tokens
count toward `max_tokens`. The cap is ignored with `--batched-session`.
```

- [ ] **Step 6: Run the tests and builds**

Run: `make ds4_test ds4-server 2>&1 | grep -iE "warning|error" ; ./ds4_test --server ; ./ds4-server --help | grep -A1 think-budget`
Expected: no warnings or errors; `server: OK`, `ds4 tests: ok`; the help shows both `--think-budget` lines.

- [ ] **Step 7: Commit**

```bash
git add ds4_server.c ds4_help.c docs/SERVER.md
git commit -m "ds4-server: thinking budget request fields and flags

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017f1tsCGFkVQ3wrmPsJ7FBn"
```

---

### Task 3: Forced close in the decode loop

**Files:**
- Modify: `ds4_server.c` (`generate_job_inner`, decode loop ~lines 13918-14267)

**Interfaces:**
- Consumes: Task 1 helpers; `request.think_budget`, `server.think_budget`, `server.think_budget_message`
  (Task 2); existing `server_eval_token`, `server_generation_rewind` (via the existing `kept < ntok` path),
  `ds4_tokenize_rendered_chat`, `ds4_tokens_free`, `server_log`, `trace_event`.
- Produces: log lines `ds4-server: chat ctx=<span> thinking budget reached <N> tokens; forced close (<M> tokens)`
  and `ds4-server: chat ctx=<span> thinking closed after <K> tokens[ (budget)]` (read by Tasks 4 and 5).

- [ ] **Step 1: Budget state per decode pass**

Replace

```c
    dsml_decode_tracker_init(&dsml_tracker);
    dsml_tracker.model_syntax = j->req.model_syntax;

    server_generation_enter(s);
```

with

```c
    dsml_decode_tracker_init(&dsml_tracker);
    dsml_tracker.model_syntax = j->req.model_syntax;
    think_budget_state tb = {
        .limit = !s->batched_mode && ds4_think_mode_enabled(j->req.think_mode) ?
            think_budget_effective(s->think_budget, j->req.think_budget) : 0,
    };
    ds4_tokens think_forced = {0};
    int think_forced_next = 0;

    server_generation_enter(s);
```

- [ ] **Step 2: Feed queued forced tokens instead of sampling**

Replace the block from `        const int eos_token = ds4_token_eos(s->engine);` down to and including the plain
eval branch that ends with

```c
        } else {
            if (server_eval_token(s, slot, token, err, sizeof(err)) != 0) {
                finish = "error";
                break;
            }
            toks[0] = token;
            ntok = 1;
        }
```

with

```c
        const int eos_token = ds4_token_eos(s->engine);
        int token = -1;
        int toks[17];
        int ntok = 0;
        const int block_start = ds4_session_pos(slot->session);
        if (think_forced_next < think_forced.len) {
            /* Forced thinking close: feed the queued suffix one token per
             * iteration through the plain eval path; the per-token body below
             * streams it exactly like a sampled token. */
            token = think_forced.v[think_forced_next++];
            if (server_eval_token(s, slot, token, err, sizeof(err)) != 0) {
                finish = "error";
                break;
            }
            toks[0] = token;
            ntok = 1;
        } else {
            token = j->req.ignore_eos ?
                ds4_session_argmax_ignoring_eos(slot->session,
                                                j->req.think_mode) :
                ds4_session_sample(slot->session, temperature, top_k,
                                   top_p, min_p, &rng);
            if (token < 0) {
                finish = "error";
                snprintf(err, sizeof(err), "failed to select a non-EOS token");
                break;
            }
            if (ds4_token_is_stop_for_think_mode(s->engine,
                                                 token,
                                                 j->req.think_mode)) {
                finish = "stop";
                stop_detail = "stop token";
                stop_token = token;
                break;
            }

            if (!s->batched_mode &&
                ds4_engine_mtp_draft_tokens(s->engine) > 1 &&
                getenv("DS4_MTP_SPEC_DISABLE") == NULL)
            {
                if (j->req.ignore_eos) {
                    ntok = ds4_session_eval_speculative_argmax_ignoring_eos(
                        slot->session, token, max_tokens - completion,
                        eos_token, j->req.think_mode,
                        toks, (int)(sizeof(toks) / sizeof(toks[0])),
                        err, sizeof(err));
                } else {
                    ntok = ds4_session_eval_speculative(
                        slot->session, token, max_tokens - completion,
                        eos_token, temperature, top_k, top_p, min_p, &rng,
                        toks, (int)(sizeof(toks) / sizeof(toks[0])),
                        err, sizeof(err));
                }
                if (ntok < 0) {
                    finish = "error";
                    break;
                }
            } else if (s->batched_mode && s->qwen4_batch_mtp &&
                       max_tokens - completion >= 2 && !j->req.ignore_eos &&
                       (!ds4_engine_mtp_exact_sampling(s->engine) || temperature == 0.0f) &&
                       getenv("DS4_MTP_SPEC_DISABLE") == NULL) {
                if (server_eval_tokens(s, slot, token, true, toks, &ntok, err, sizeof(err)) != 0) {
                    finish = "error";
                    break;
                }
            } else {
                if (server_eval_token(s, slot, token, err, sizeof(err)) != 0) {
                    finish = "error";
                    break;
                }
                toks[0] = token;
                ntok = 1;
            }
        }
```

The only behavioural change for budget-off requests is none: `think_forced.len` stays 0, the `else` branch is
the old code re-indented, and `block_start` is read before sampling, which does not move the session position.

- [ ] **Step 3: Count reasoning tokens and log the close**

Replace

```c
                    dsml_decode_tracker_update(&dsml_tracker, text.ptr, text.len);
                }
            }

            size_t stop_pos = 0, stop_len = 0;
```

with

```c
                    dsml_decode_tracker_update(&dsml_tracker, text.ptr, text.len);
                }
            }
            bool think_budget_fire = false;
            if (think_budget_due(&tb, thinking.inside) &&
                !(j->req.has_tools && think_budget_tool_open(text.ptr))) {
                tb.fired = true;
                think_budget_fire = true;
            }
            if (was_thinking && !thinking.inside) {
                server_log(DS4_LOG_GENERATION,
                           "ds4-server: chat ctx=%s thinking closed after %d tokens%s",
                           ctx_span, tb.used, tb.fired ? " (budget)" : "");
            }

            size_t stop_pos = 0, stop_len = 0;
```

- [ ] **Step 4: Queue the forced suffix and cut the block**

Replace

```c
            const bool next_greedy = !thinking.inside &&
```

with

```c
            if (think_budget_fire) {
                /* Cut the rest of this block (the kept < ntok rewind below)
                 * and close the reasoning on the next iterations. */
                char *suffix = think_budget_suffix_text(s->think_budget_message);
                ds4_tokenize_rendered_chat(s->engine, suffix, &think_forced);
                free(suffix);
                think_forced_next = 0;
                server_log(DS4_LOG_GENERATION,
                           "ds4-server: chat ctx=%s thinking budget reached %d tokens; forced close (%d tokens)",
                           ctx_span, tb.used, think_forced.len);
                trace_event(s, trace_id, "thinking budget reached %d tokens; forced close", tb.used);
                break;
            }
            const bool next_greedy = !thinking.inside &&
```

- [ ] **Step 5: Free the queue after the loop**

Replace

```c
    server_generation_leave(s);
```

(the single occurrence right after the decode `while` loop) with

```c
    server_generation_leave(s);
    ds4_tokens_free(&think_forced);
```

Confirm there is no `return` inside the decode `while` loop (`awk 'NR>=13928 && NR<=14290 && /return/' ds4_server.c`
prints nothing for the loop body; adjust the range to the loop's actual lines).

- [ ] **Step 6: Build and run the unit tests**

Run: `make ds4_test ds4-server 2>&1 | grep -iE "warning|error" ; ./ds4_test --server`
Expected: no warnings or errors; `server: OK`, `ds4 tests: ok`.

- [ ] **Step 7: Commit**

```bash
git add ds4_server.c
git commit -m "ds4-server: close reasoning at the thinking budget

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017f1tsCGFkVQ3wrmPsJ7FBn"
```

---

### Task 4: Live test and the byte-identical regression gate (GPU)

**Files:**
- Create: `tests/test_think_budget_live.py`

**Interfaces:**
- Consumes: the `ds4-server` binary and log lines from Task 3; model files
  `~/orca/workspaces/ds4-metal-data/gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q4KDownPad768-DenseQ4Kselimat-MTP.gguf`
  and `~/orca/workspaces/ds4-metal-data/gguf/Qwen3.8-Flash-Next-PLE-Q4_1.gguf`.
- Produces: pass/fail evidence for Review Focus 2, 3, 5 and the off-path gate.

- [ ] **Step 1: Write the live test**

```python
"""Live thinking-budget check; needs the Qwen3.8 GGUF, its PLE sidecar and a free GPU.

python3 tests/test_think_budget_live.py --model MODEL --ple PLE
"""

import argparse
import json
import os
import pathlib
import re
import socket
import subprocess
import tempfile
import time
import urllib.request

PROMPT = "Prove that there are infinitely many prime numbers, then list the first ten primes."
MESSAGE = ("Considering the limited time by the user, I have to give the solution "
           "based on the thinking directly now.")
BUDGET = 64


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Server:
    def __init__(self, root, args, extra_env=None):
        self.out = pathlib.Path(tempfile.mkdtemp(prefix="ds4-think-budget-"))
        self.log = self.out / "server.log"
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        env = dict(os.environ, DS4_QWEN4_STREAM_FULL_LAYERS="32", DS4_QWEN4_PLE_PREFETCH_FULL="0")
        env.update(extra_env or {})
        cmd = [str(root / "ds4-server"), "--metal", "-m", args.model, "--ple", args.ple,
               "-c", "32768", "--prefill-chunk", "2048", "--mtp", "--ssd-streaming",
               "--ssd-streaming-cache-experts", "6GB", "--think-budget", str(BUDGET),
               "--kv-disk-dir", str(self.out / "kv"), "--host", "127.0.0.1",
               "--port", str(self.port)]
        self.fh = open(self.log, "w")
        self.proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=self.fh, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 600
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited {self.proc.returncode}, see {self.log}")
            try:
                urllib.request.urlopen(self.base + "/v1/models", timeout=2).read()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("startup timeout")
                time.sleep(1)
        self.mark = 0

    def new_log(self):
        text = self.log.read_text(errors="replace")
        chunk, self.mark = text[self.mark:], len(text)
        return chunk

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.fh.close()


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=1800)


def chat(base, messages, stream=False, **extra):
    body = {"model": "ds4", "messages": messages, "max_tokens": 400, "temperature": 0,
            "stream": stream}
    body.update(extra)
    resp = post(base, "/v1/chat/completions", body)
    if not stream:
        o = json.loads(resp.read())
        msg = o["choices"][0]["message"]
        return (msg.get("reasoning_content") or "", msg.get("content") or "",
                o["choices"][0].get("finish_reason"))
    reasoning, content, finish = [], [], None
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        choices = json.loads(line[6:]).get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        reasoning.append(delta.get("reasoning_content") or "")
        content.append(delta.get("content") or "")
        finish = choice.get("finish_reason") or finish
    return "".join(reasoning), "".join(content), finish


def anthropic(base, budget):
    body = {"model": "ds4", "max_tokens": 400, "temperature": 0,
            "thinking": {"type": "enabled", "budget_tokens": budget},
            "messages": [{"role": "user", "content": PROMPT}]}
    o = json.loads(post(base, "/v1/messages", body).read())
    thinking = "".join(b.get("thinking", "") for b in o["content"] if b["type"] == "thinking")
    text = "".join(b.get("text", "") for b in o["content"] if b["type"] == "text")
    return thinking, text


def check(cond, what):
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    if not cond:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ple", required=True)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    user = [{"role": "user", "content": PROMPT}]

    srv = Server(root, args)
    try:
        # 1. forced close, non-stream
        r1 = chat(srv.base, user)
        log = srv.new_log()
        check(f"thinking budget reached {BUDGET} tokens" in log, "1 budget log line")
        check(MESSAGE in r1[0], "1 forced sentence in reasoning")
        check(r1[1].strip() != "", "1 answer after the forced close")
        # 2. stream equals non-stream
        r2 = chat(srv.base, user, stream=True)
        srv.new_log()
        check(r2[:2] == r1[:2], "2 stream text == non-stream text")
        # 3. next turn continues from the live state (prefix cached)
        turn2 = user + [{"role": "assistant", "content": r2[1]},
                        {"role": "user", "content": "Now list the next five primes."}]
        chat(srv.base, turn2)
        log = srv.new_log()
        starts = re.findall(r"chat ctx=(\d+)\.\.\d+:\d+ prompt start", log)
        check(bool(starts) and int(starts[0]) > 0 and "live kv cache miss" not in log,
              "3 next turn reuses the live prefix")
        # 4. request budget lower than the server cap (Anthropic)
        thinking, text = anthropic(srv.base, 32)
        log = srv.new_log()
        check("thinking budget reached 32 tokens" in log, "4 anthropic budget_tokens lowers the cap")
        check(MESSAGE in thinking and text.strip() != "", "4 anthropic forced close + answer")
        # 5. max_tokens below the budget: plain length finish, no forced close
        r5 = chat(srv.base, user, max_tokens=40)
        log = srv.new_log()
        check(r5[2] == "length" and "thinking budget reached" not in log, "5 max_tokens wins")
        # 6. thinking disabled: the budget changes nothing
        a = chat(srv.base, user, think=False, thinking_budget=16)
        b = chat(srv.base, user, think=False)
        log = srv.new_log()
        check(a == b and "thinking budget reached" not in log, "6 no-think request unaffected")
    finally:
        srv.stop()

    # 7. MTP speculation off: the forced close still works
    srv = Server(root, args, {"DS4_MTP_SPEC_DISABLE": "1"})
    try:
        r7 = chat(srv.base, user)
        log = srv.new_log()
        check(f"thinking budget reached {BUDGET} tokens" in log and MESSAGE in r7[0]
              and r7[1].strip() != "", "7 forced close with MTP off")
    finally:
        srv.stop()
    print("think-budget live: OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Wait for a free GPU**

Before running: `pgrep -fl "ds4-server|ds4_test|ds4-bench"` prints nothing, the DS4.1 session is not running a
GPU job, and the user agreed. Do not proceed otherwise.

- [ ] **Step 3: Run the live test**

Run:
```bash
python3 tests/test_think_budget_live.py \
  --model ~/orca/workspaces/ds4-metal-data/gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q4KDownPad768-DenseQ4Kselimat-MTP.gguf \
  --ple ~/orca/workspaces/ds4-metal-data/gguf/Qwen3.8-Flash-Next-PLE-Q4_1.gguf
```
Expected: `PASS` lines 1-7 and `think-budget live: OK`. On a FAIL, use superpowers:systematic-debugging with the
printed server log path before changing code.

- [ ] **Step 4: Run the byte-identical regression gate (budget off)**

Run: `speed-bench/qwen-regression/run.sh fast`
Expected: kernel tests pass; vi/code replies byte-identical to `speed-bench/qwen-regression/baseline/`; registry
command unchanged. If it fails, rerun it once on the base commit 2152f86 build to tell a pre-existing difference
from one this branch introduced, and report which.

- [ ] **Step 5: Commit**

```bash
git add tests/test_think_budget_live.py
git commit -m "tests: live thinking-budget check

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017f1tsCGFkVQ3wrmPsJ7FBn"
```

---

### Task 5: Measure and choose N (GPU)

**Files:**
- Create: `speed-bench/think-budget/measure.py`
- Create: `speed-bench/think-budget/RESULTS.md` (written from the run)

**Interfaces:**
- Consumes: registry `~/.local/ai-gateway/runtime-registry.json` (enabled ds4 runtime `process_command`);
  `gsm8k_mini.PROBLEMS/build_prompt/grade` and `humaneval_mini.PROBLEMS/build_prompt/grade` from
  `~/Documents/GitHub/AI-Gateway-MLX/evals/bakeoff/benchmarks` (read-only import); log lines from Task 3.
- Produces: `speed-bench/think-budget/runs/budget-<N>.json` and the chosen N in `RESULTS.md`.

- [ ] **Step 1: Write the measurement script**

```python
"""Choose --think-budget N: thinking-token distribution and code-graded quality per N.

python3 speed-bench/think-budget/measure.py --budgets 0,8192,4096,2048,1024

Runs the PROD ds4 command from the gateway registry, but with this checkout's ds4-server, a scratch port and
KV dir, and --think-budget N (0 = off). Sends GSM8K-mini and HumanEval-mini (vendored in AI-Gateway-MLX) plus
five Vietnamese prompts at temperature 0 with thinking on, and reads the server log for the thinking length.
"""

import argparse
import json
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
REGISTRY = pathlib.Path.home() / ".local/ai-gateway/runtime-registry.json"
PORT = 18299
VI_PROMPTS = [
    "Giải thích ngắn gọn cách bộ nhớ đệm KV giúp mô hình ngôn ngữ sinh văn bản nhanh hơn.",
    "Viết một đoạn văn khoảng 120 chữ giới thiệu vịnh Hạ Long cho khách du lịch.",
    "So sánh ưu và nhược điểm của Python và Go khi viết dịch vụ web.",
    "Tóm tắt ý chính của định luật Ohm và cho một ví dụ tính toán.",
    "Một cửa hàng giảm giá 20% rồi giảm thêm 10% trên giá đã giảm. Tổng cộng giảm bao nhiêu phần trăm?",
]


def prod_command(port, kv_dir, budget):
    reg = json.loads(REGISTRY.read_text())
    rt = next(m["runtimes"]["ds4"] for m in reg["models"].values()
              if m.get("runtimes", {}).get("ds4", {}).get("enabled"))
    cmd = list(rt["process_command"])
    for i, a in enumerate(cmd):
        if a == "--port":
            cmd[i + 1] = str(port)
        elif a == "--kv-disk-dir":
            cmd[i + 1] = str(kv_dir)
        elif os.path.basename(a) == "ds4-server":
            cmd[i] = str(ROOT / "ds4-server")
    if budget > 0:
        cmd += ["--think-budget", str(budget)]
    return cmd


def ask(base, prompt):
    body = {"model": "ds4", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16384, "temperature": 0, "stream": False}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    o = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    msg = o["choices"][0]["message"]
    return msg.get("content") or "", time.time() - t0


def run_budget(budget, bench, out_dir):
    kv = pathlib.Path(tempfile.mkdtemp(prefix="kv-", dir=out_dir))
    log_path = out_dir / f"server-{budget}.log"
    fh = open(log_path, "w")
    proc = subprocess.Popen(prod_command(PORT, kv, budget), cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{PORT}"
    try:
        for _ in range(900):
            if proc.poll() is not None:
                raise RuntimeError(f"server exited, see {log_path}")
            try:
                urllib.request.urlopen(base + "/v1/models", timeout=2).read()
                break
            except OSError:
                time.sleep(1)
        cases = ([("gsm8k", p, bench["gsm8k"]) for p in bench["gsm8k"].PROBLEMS] +
                 [("humaneval", p, bench["humaneval"]) for p in bench["humaneval"].PROBLEMS] +
                 [("vi", p, None) for p in VI_PROMPTS])
        rows = []
        mark = 0
        for kind, problem, mod in cases:
            prompt = mod.build_prompt(problem) if mod else problem
            content, secs = ask(base, prompt)
            text = log_path.read_text(errors="replace")
            new, mark = text[mark:], len(text)
            m = re.findall(r"thinking closed after (\d+) tokens( \(budget\))?", new)
            think = int(m[-1][0]) if m else None
            forced = bool(m and m[-1][1])
            passed = mod.grade(problem, content)[0] if mod else None
            rows.append({"kind": kind, "passed": passed, "think_tokens": think,
                         "forced": forced, "seconds": round(secs, 1),
                         "answer": content if kind == "vi" else None})
            print(f"N={budget} {kind} passed={passed} think={think} forced={forced} {secs:.1f}s", flush=True)
        return rows
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        fh.close()
        shutil.rmtree(kv, ignore_errors=True)
        time.sleep(30)  # let the previous server's wired memory drain before the next start


def summarize(budget, rows):
    think = [r["think_tokens"] for r in rows if r["think_tokens"] is not None]
    graded = [r for r in rows if r["passed"] is not None]
    return {
        "budget": budget,
        "gsm8k_pass": sum(1 for r in rows if r["kind"] == "gsm8k" and r["passed"]),
        "humaneval_pass": sum(1 for r in rows if r["kind"] == "humaneval" and r["passed"]),
        "graded": len(graded),
        "forced": sum(1 for r in rows if r["forced"]),
        "think_median": statistics.median(think) if think else None,
        "think_p90": sorted(think)[int(0.9 * (len(think) - 1))] if think else None,
        "think_max": max(think) if think else None,
        "seconds_total": round(sum(r["seconds"] for r in rows), 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budgets", default="0,8192,4096,2048,1024")
    ap.add_argument("--gateway-repo", default=str(pathlib.Path.home() / "Documents/GitHub/AI-Gateway-MLX"))
    ap.add_argument("--out", default=str(ROOT / "speed-bench/think-budget/runs"))
    args = ap.parse_args()
    sys.path.insert(0, str(pathlib.Path(args.gateway_repo) / "evals/bakeoff/benchmarks"))
    import gsm8k_mini
    import humaneval_mini
    bench = {"gsm8k": gsm8k_mini, "humaneval": humaneval_mini}
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for budget in [int(b) for b in args.budgets.split(",")]:
        rows = run_budget(budget, bench, out_dir)
        (out_dir / f"budget-{budget}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        summaries.append(summarize(budget, rows))
        print(json.dumps(summaries[-1]), flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Wait for a free GPU**

Same condition as Task 4 Step 2. The run takes roughly 2-3 hours (38 prompts x 5 budgets, reasoning-heavy).
Run it under `caffeinate -i -s` and watch the log for stalls (see memory metal-kill9-wedges-gguf-vnode: a thread
stuck in a kernel `pread` does not recover; report and ask for a reboot instead of `kill -9`).

- [ ] **Step 3: Run the measurement**

Run: `caffeinate -i -s python3 speed-bench/think-budget/measure.py --budgets 0,8192,4096,2048,1024`
Expected: one progress line per prompt and a summary line per budget; `runs/summary.json` written.

- [ ] **Step 4: Choose N and write the receipt**

Choose the lowest N whose GSM8K-mini and HumanEval-mini pass sets contain every case that passed at N=0 (no
unlimited-pass case fails). Write `speed-bench/think-budget/RESULTS.md` with: date, commit, model/command, the
summary table (budget, gsm8k_pass/16, humaneval_pass/17, forced, think median/p90/max, total seconds), the
unlimited-run thinking distribution, the five VI answers at N=0 and at the chosen N side by side, and the
chosen N with the one-line reason. If no N below unlimited keeps every case, say so and recommend no cap.

- [ ] **Step 5: Commit**

```bash
git add speed-bench/think-budget/measure.py speed-bench/think-budget/RESULTS.md speed-bench/think-budget/runs/*.json
git commit -m "RESULTS: thinking budget distribution and quality per N

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017f1tsCGFkVQ3wrmPsJ7FBn"
```

---

### Task 6: Finish and deploy (user approval at each gate)

**Files:** none in this repo besides the merge; PROD registry edited per docs/DEPLOY_AI_GATEWAY.md.

- [ ] **Step 1:** Final whole-branch review (superpowers:requesting-code-review), fix findings, re-verify Task 4.
- [ ] **Step 2:** Ask the user to approve merging `think-budget` into develop and pushing. Note in the request that
  local develop carries the unpushed DS4.1 merge 2152f86, so pushing develop publishes it too; get the DS4.1
  session's agreement first.
- [ ] **Step 3:** After approval: merge (`--no-ff`, message `Merge branch 'think-budget' into develop`), push, and
  tell the DS4.1 session that `ds4_server.c` changed.
- [ ] **Step 4:** After a separate PROD approval, follow docs/DEPLOY_AI_GATEWAY.md: record the develop reference
  with the new registry command (`--think-budget <chosen N>` appended to `process_command`, registry backed up
  first as `runtime-registry.json.bak-prod-think-budget-YYYYMMDD`), `deploy-ai-gateway.sh cut think-budget`,
  stop the gateway ds4 slot, `install prod/think-budget-YYYYMMDD`, `smoke --ref` must say "same as ref" for vi and
  code, then hand over.
