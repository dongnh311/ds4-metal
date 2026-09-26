/* Real-model Ornith session checks (prefill chunk 512).
 * Usage: test_qwen35_session MODEL
 *  1. the live session extends its prefix to 128, 129, 511, 512, 513, 1024,
 *     2049 and 4096 tokens; at each length it asserts:
 *     - prefix reuse: the sync's first "prefill_chunk" report is
 *       min(previous + 512, length); a reset and replay would report
 *       min(512, length), so this tells them apart from 513 tokens on;
 *     - its logits equal a fresh session replaying the same syncs (< 1e-3);
 *     - its argmax equals a one-pass prefill of the same prefix; the one-pass
 *       logit difference (chunking changes rounding) is printed only;
 *  2. a divergent prompt resets the state: logits bit-identical to a fresh session;
 *  3. a full context stops eval with the error "context is full";
 *  4. a session above the native 262144-token context is refused (no YaRN). */
#define _POSIX_C_SOURCE 200809L
#include "../ds4.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int first, last, calls; } progress_log;

static void on_progress(void *ud, const char *event, int current, int total) {
    (void)total;
    if (strcmp(event, "prefill_chunk")) return;
    progress_log *p = ud;
    if (!p->calls) p->first = current;
    p->last = current;
    p->calls++;
}

static void sync_len(ds4_session *s, const ds4_tokens *tokens, int n) {
    ds4_tokens prefix = *tokens;
    prefix.len = n;
    char err[256] = {0};
    const int rc = ds4_session_sync(s, &prefix, err, sizeof(err));
    if (rc) fprintf(stderr, "sync %d: %s\n", n, err);
    assert(rc == 0 && ds4_session_pos(s) == n);
}

static float max_diff(const float *a, const float *b, int n) {
    float worst = 0.0f;
    for (int i = 0; i < n; i++) {
        assert(isfinite(a[i]) && isfinite(b[i]));
        worst = fmaxf(worst, fabsf(a[i] - b[i]));
    }
    return worst;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL\n", argv[0]);
        return 1;
    }
    const int ctx = 4096 + 64, chunk = 512;
    ds4_engine_options opt = {.model_path = argv[1], .context_size = ctx, .prefill_chunk = chunk,
                              .backend = DS4_BACKEND_METAL};
    ds4_engine *engine = NULL;
    assert(ds4_engine_open(&engine, &opt) == 0 && ds4_engine_is_qwen35moe(engine));
    FILE *fp = fopen("speed-bench/promessi_sposi.txt", "rb");
    assert(fp);
    char *text = calloc(40001, 1);
    assert(fread(text, 1, 40000, fp) > 0);
    fclose(fp);
    ds4_tokens tokens = {0};
    ds4_tokenize_text(engine, text, &tokens);
    free(text);
    assert(tokens.len >= ctx - 2);
    const int vocab = ds4_engine_vocab_size(engine);
    float *a = malloc((size_t)vocab * 4), *b = malloc((size_t)vocab * 4);
    ds4_session *live = NULL, *control = NULL;
    assert(ds4_session_create(&live, engine, ctx) == 0);
    assert(ds4_session_create(&control, engine, ctx) == 0);

    /* 1. prefix extension across chunk boundaries: the live session resumes
     * after its previous length and equals a fresh session replaying the same
     * syncs; a one-pass prefill chunks differently, so only its argmax is
     * asserted and its logit difference is reported */
    const int lengths[] = {128, 129, 511, 512, 513, 1024, 2049, 4096};
    const size_t n_lengths = sizeof(lengths) / sizeof(*lengths);
    for (size_t j = 0; j < n_lengths; j++) {
        const int prev = j ? lengths[j - 1] : 0;
        const int first_expected = prev + chunk < lengths[j] ? prev + chunk : lengths[j];
        progress_log p = {0};
        ds4_session_set_progress(live, on_progress, &p);
        sync_len(live, &tokens, lengths[j]);
        ds4_session_set_progress(live, NULL, NULL);
        if (p.first != first_expected)
            fprintf(stderr, "sync %d after %d: first chunk ends at %d, expected %d\n", lengths[j], prev, p.first,
                    first_expected);
        assert(p.calls > 0 && p.last == lengths[j] && p.first == first_expected);
        assert(ds4_session_copy_logits(live, a, vocab) == vocab);
        ds4_session_invalidate(control);
        for (size_t k = 0; k <= j; k++) sync_len(control, &tokens, lengths[k]);
        assert(ds4_session_copy_logits(control, b, vocab) == vocab);
        const float d = max_diff(a, b, vocab);
        ds4_session_invalidate(control);
        sync_len(control, &tokens, lengths[j]);
        assert(ds4_session_copy_logits(control, b, vocab) == vocab);
        printf("  extend to %5d: first chunk ends at %4d, replay max|d| %.2e, one-pass max|d| %.2e argmax %d/%d\n",
               lengths[j], p.first, d, max_diff(a, b, vocab), ds4_session_argmax(live), ds4_session_argmax(control));
        assert(d < 1e-3f);
        assert(ds4_session_argmax(live) == ds4_session_argmax(control));
    }

    /* 2. a divergent prompt resets the state: bit-identical to a fresh session */
    ds4_tokens other = {0};
    ds4_tokenize_text(engine, "Xin chào! Hôm nay tôi muốn kể cho bạn nghe về Hà Nội.", &other);
    sync_len(live, &other, other.len);
    assert(ds4_session_copy_logits(live, a, vocab) == vocab);
    ds4_session *fresh = NULL;
    assert(ds4_session_create(&fresh, engine, ctx) == 0);
    sync_len(fresh, &other, other.len);
    assert(ds4_session_copy_logits(fresh, b, vocab) == vocab);
    assert(memcmp(a, b, (size_t)vocab * 4) == 0);
    printf("  divergent prompt: bit-identical to a fresh session\n");

    /* 3. context full: eval stops with an error, no crash */
    ds4_session_invalidate(control);
    sync_len(control, &tokens, ctx - 2);
    char err[256] = {0};
    assert(ds4_session_eval(control, 11, err, sizeof(err)) == 0);
    assert(ds4_session_eval(control, 11, err, sizeof(err)) == 0);
    assert(ds4_session_eval(control, 11, err, sizeof(err)) != 0);
    printf("  context full: '%s'\n", err);
    assert(strcmp(err, "context is full") == 0);

    /* 4. above the native context: the graph allocation refuses it */
    ds4_session *big = NULL;
    assert(ds4_session_create(&big, engine, 262145) != 0 && big == NULL);
    printf("  context 262145: session refused\n");

    ds4_session_free(fresh);
    ds4_session_free(control);
    ds4_session_free(live);
    ds4_tokens_free(&other);
    ds4_tokens_free(&tokens);
    free(a);
    free(b);
    ds4_engine_close(engine);
    printf("qwen35 session: ok\n");
    return 0;
}
