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

/* Cosine similarity of two rows of rowdim floats (both-zero rows count as
 * aligned: 1.0). */
static double row_cosine(const float *a, const float *b, uint32_t rowdim) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (uint32_t i = 0; i < rowdim; i++) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 1.0;
    return dot / (sqrt(na) * sqrt(nb));
}

/* Minimum per-row cosine similarity over `rows` rows of `rowdim` floats. */
static double min_row_cosine(const float *a, const float *b, uint32_t rows, uint32_t rowdim) {
    double mn = 1.0;
    for (uint32_t r = 0; r < rows; r++) {
        const double c = row_cosine(a + (uint64_t)r * rowdim, b + (uint64_t)r * rowdim, rowdim);
        if (c < mn) mn = c;
    }
    return mn;
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
    for (uint64_t i = 0; i < n; i++) CHECK(isfinite(ka[i]));
    /* The trunk seed rows (mtp_h) legitimately differ by a few percent across
     * prefill geometries (ordinary FP rounding accumulated over up to 40
     * layers x 600 tokens grouped into different chunk sizes; measured
     * up to ~15% max|delta| of scale for the K rows themselves), so a
     * max-abs tolerance on the K rows is not discriminating: it would also
     * have to tolerate a genuine row-misalignment bug of comparable size.
     * Per-row cosine similarity does discriminate: ordinary rounding drift
     * keeps each row's direction essentially unchanged (measured min cosine
     * 0.994648 at chunk 64, 0.990599 per-token, over 600 rows), while
     * shifting row p against row p+1 -- a stand-in for a boundary/alignment
     * bug -- collapses the min cosine to 0.301315 (measured on this same
     * run), well below any threshold with headroom over the measured
     * ordinary-drift floor. */
    const uint32_t rowdim = DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    const double cos_b = min_row_cosine(ka, kb, N, rowdim);
    const double cos_c = min_row_cosine(ka, kc, N, rowdim);
    /* Negative control: row p of chunk-512 against row p+1 of chunk-64. A
     * genuine boundary/alignment bug would look like this shift, so this
     * must collapse well below the ordinary-drift floor above. */
    const double cos_shift = min_row_cosine(ka, kb + rowdim, N - 1u, rowdim);
    printf("  catch-up K rows: min row cosine chunk 64 %.6f, per token %.6f, shifted %.6f\n",
           cos_b, cos_c, cos_shift);
    CHECK(cos_b >= 0.95 && cos_c >= 0.95);
    CHECK(cos_shift < 0.95);
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
