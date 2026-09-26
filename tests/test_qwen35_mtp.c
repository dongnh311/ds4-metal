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
