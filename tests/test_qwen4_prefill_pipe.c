/* Model-free checks for the qwen4 prefill residency / staging pipe policy
 * (docs/superpowers/specs/2026-09-25-qwen4-prefill-residency-pipe-design.md). */
#include "../ds4.c"
#include "../ds4_qwen4_stage_pipe.h"

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

static void check_role_default(void) {
    ds4_qwen4_gpu_graph *g = xcalloc(1, sizeof(*g));
    CHECK(g->prefill_role == QWEN4_PREFILL_ROLE_NONE);
    free(g);
}

int main(void) {
    check_role_default();
    check_mode_parse();
    check_policy();
    check_prompt_end();
    check_use_pipe();
    check_next_staged();
    check_pipe_order();
    check_pipe_match();
    check_pipe_readers();
    check_pipe_prompt_end();
    if (failures) {
        fprintf(stderr, "test_qwen4_prefill_pipe: %d failure(s)\n", failures);
        return 1;
    }
    puts("test_qwen4_prefill_pipe: PASS");
    return 0;
}
