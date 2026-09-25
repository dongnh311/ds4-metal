/* GPU kernel tests for Ornith-1.5-35B-A3B (qwen35moe): the kernels in
 * metal/qwen35.metal against double-precision references.
 * Build: make test-qwen35-kernels */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "ds4.h"
#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static uint32_t g_rng = 0x9e3779b9u;

static float frand(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return ((float)(g_rng & 0xffffffu) / 8388608.0f) - 1.0f;
}

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "%s failed\n", what);
        exit(1);
    }
}

static void check_close(const char *what, const float *got, const double *ref, uint64_t n, double tol) {
    double worst = 0.0, scale = 1e-6;
    uint64_t worst_i = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (!isfinite(got[i])) {
            fprintf(stderr, "%s: non-finite value at %llu\n", what, (unsigned long long)i);
            exit(1);
        }
        const double d = fabs((double)got[i] - ref[i]);
        if (d > worst) { worst = d; worst_i = i; }
        if (fabs(ref[i]) > scale) scale = fabs(ref[i]);
    }
    if (worst > tol * scale) {
        fprintf(stderr, "%s: max|d| %.3e (rel %.3e) at %llu: got %.6f ref %.6f\n",
                what, worst, worst / scale, (unsigned long long)worst_i, got[worst_i], ref[worst_i]);
        exit(1);
    }
    printf("  %-44s ok  max|d|=%.2e (scale %.2e)\n", what, worst, scale);
}

static uint16_t f32_to_f16(float f) {
    union { float f; uint32_t u; } v = { f };
    const uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half++;
        return (uint16_t)(sign | half);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    union { uint32_t u; float f; } v;
    if (exp == 0) {
        if (mant == 0) { v.u = sign; return v.f; }
        exp = 127 - 15 + 1;
        while (!(mant & 0x400u)) { mant <<= 1; exp--; }
        mant &= 0x3ffu;
        v.u = sign | (exp << 23) | (mant << 13);
        return v.f;
    }
    if (exp == 31) { v.u = sign | 0x7f800000u | (mant << 13); return v.f; }
    v.u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    return v.f;
}

static double sigmoid_d(double x) { return x >= 0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x)); }
static double silu_d(double x) { return x * sigmoid_d(x); }

/* ---- weight arena ---- */

/* anonymous mmap standing in for the model map; weights are appended, never freed */
typedef struct {
    uint8_t *base;
    uint64_t size;
    uint64_t used;
} arena_t;

static uint64_t arena_alloc(arena_t *a, uint64_t bytes) {
    const uint64_t off = (a->used + 63u) & ~63ull;
    if (off + bytes > a->size) {
        fprintf(stderr, "arena exhausted\n");
        exit(1);
    }
    a->used = off + bytes;
    return off;
}

/* q8_0 rows of `cols` elements (cols % 32 == 0), `rows` rows */
static uint64_t arena_q8_0(arena_t *a, uint64_t rows, uint64_t cols, double **shadow, float scale) {
    const uint64_t blocks = cols / 32;
    const uint64_t off = arena_alloc(a, rows * blocks * 34u);
    uint8_t *w = a->base + off;
    *shadow = malloc(rows * cols * sizeof(double));
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t b = 0; b < blocks; b++) {
            float vals[32];
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                vals[j] = scale * frand();
                if (fabsf(vals[j]) > amax) amax = fabsf(vals[j]);
            }
            const float d = amax / 127.0f;
            const uint16_t dh = f32_to_f16(d);
            const float dq = f16_to_f32(dh);
            uint8_t *blk = w + (r * blocks + b) * 34u;
            memcpy(blk, &dh, 2);
            for (int j = 0; j < 32; j++) {
                int q = (int)lrintf(d > 0 ? vals[j] / d : 0.0f);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                ((int8_t *)blk)[2 + j] = (int8_t)q;
                (*shadow)[r * cols + b * 32 + j] = (double)dq * q;
            }
        }
    }
    return off;
}

static ds4_gpu_tensor *upload(const float *data, uint64_t n) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n * sizeof(float));
    require_ok(t != NULL, "tensor alloc");
    if (data) require_ok(ds4_gpu_tensor_write(t, 0, data, n * sizeof(float)), "tensor write");
    else require_ok(ds4_gpu_tensor_fill_f32(t, 0.0f, n), "tensor fill");
    return t;
}

static float *download(const ds4_gpu_tensor *t, uint64_t n) {
    float *out = malloc(n * sizeof(float));
    require_ok(ds4_gpu_tensor_read(t, 0, out, n * sizeof(float)), "tensor read");
    return out;
}

static float *rand_vec(uint64_t n, float scale) {
    float *v = malloc(n * sizeof(float));
    for (uint64_t i = 0; i < n; i++) v[i] = scale * frand();
    return v;
}

/* q5_K rows: 176-byte super-blocks of 256 (d, dmin, 12 packed 6-bit
 * scale/min bytes, 32 high-bit bytes, 128 nibble bytes; llama.cpp layout).
 * Element j of 32-group g stores its low nibble in qs[(g/2)*32 + j] (low
 * half for even g) and its fifth bit in bit g of qh[j].  The shadow holds the
 * dequantized weights. */
static uint64_t arena_q5_K(arena_t *a, uint64_t rows, uint64_t cols, double **shadow, float scale) {
    const uint64_t blocks = cols / 256;
    const uint64_t off = arena_alloc(a, rows * blocks * 176u);
    uint8_t *w = a->base + off;
    *shadow = malloc(rows * cols * sizeof(double));
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t b = 0; b < blocks; b++) {
            uint8_t *blk = w + (r * blocks + b) * 176u;
            const float d = scale / 63.0f / 31.0f, dmin = d;
            const uint16_t dh = f32_to_f16(d), mh = f32_to_f16(dmin);
            const float dq = f16_to_f32(dh), mq = f16_to_f32(mh);
            memcpy(blk, &dh, 2);
            memcpy(blk + 2, &mh, 2);
            uint8_t sc[8], mn[8];
            for (int g = 0; g < 8; g++) {
                sc[g] = (uint8_t)(1 + (int)(62.0f * (0.5f * frand() + 0.5f)));
                mn[g] = (uint8_t)((int)(63.0f * (0.5f * frand() + 0.5f)));
            }
            uint8_t *s = blk + 4;
            memset(s, 0, 12);
            for (int g = 0; g < 4; g++) { s[g] = sc[g] & 63; s[g + 4] = mn[g] & 63; }
            for (int g = 4; g < 8; g++) {
                s[g + 4] = (uint8_t)((sc[g] & 0xF) | ((mn[g] & 0xF) << 4));
                s[g - 4] |= (uint8_t)((sc[g] >> 4) << 6);
                s[g] |= (uint8_t)((mn[g] >> 4) << 6);
            }
            memset(blk + 16, 0, 160);
            for (int g = 0; g < 8; g++) {
                for (int j = 0; j < 32; j++) {
                    const int q = (int)(31.0f * (0.5f * frand() + 0.5f));
                    blk[48 + (g >> 1) * 32 + j] |= (uint8_t)((q & 15) << ((g & 1) * 4));
                    if (q & 16) blk[16 + j] |= (uint8_t)(1u << g);
                    (*shadow)[r * cols + b * 256 + g * 32 + j] = (double)dq * sc[g] * q - (double)mq * mn[g];
                }
            }
        }
    }
    return off;
}

/* Q5_K routed experts with a Q8_0 shared expert slot: mid and down
 * against the double reference, for decode (T=1), a verify (T=2) and a
 * small prefill batch (T=5). */
static void test_moe_q5k(arena_t *a, uint32_t NE, uint32_t slots, uint32_t E, uint32_t F, uint32_t T) {
    double *gate_w, *up_w, *down_w, *sg_w, *su_w, *sd_w;
    const uint64_t gate_off = arena_q5_K(a, (uint64_t)NE * F, E, &gate_w, 0.05f);
    const uint64_t up_off = arena_q5_K(a, (uint64_t)NE * F, E, &up_w, 0.05f);
    const uint64_t down_off = arena_q5_K(a, (uint64_t)NE * E, F, &down_w, 0.05f);
    const uint64_t sg_off = arena_q8_0(a, F, E, &sg_w, 0.05f);
    const uint64_t su_off = arena_q8_0(a, F, E, &su_w, 0.05f);
    const uint64_t sd_off = arena_q8_0(a, E, F, &sd_w, 0.05f);
    const uint32_t n_out = slots + 1;
    float *x = rand_vec((uint64_t)T * E, 1.0f);
    int32_t *sel = malloc((uint64_t)T * slots * 4);
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t s = 0; s < slots; s++) sel[t * slots + s] = (int32_t)((t * 7u + s * 3u) % NE);
    double *mid = malloc((uint64_t)T * n_out * F * sizeof(double));
    double *part = malloc((uint64_t)T * n_out * E * sizeof(double));
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t s = 0; s < n_out; s++) {
            const bool shared = s == slots;
            const uint32_t e = shared ? 0 : (uint32_t)sel[t * slots + s];
            const double *gw = shared ? sg_w : gate_w + (uint64_t)e * F * E;
            const double *uw = shared ? su_w : up_w + (uint64_t)e * F * E;
            const double *dw = shared ? sd_w : down_w + (uint64_t)e * E * F;
            for (uint32_t f = 0; f < F; f++) {
                double g = 0.0, u = 0.0;
                for (uint32_t i = 0; i < E; i++) {
                    g += gw[(uint64_t)f * E + i] * x[t * E + i];
                    u += uw[(uint64_t)f * E + i] * x[t * E + i];
                }
                mid[((uint64_t)t * n_out + s) * F + f] = silu_d(g) * u;
            }
            for (uint32_t d = 0; d < E; d++) {
                double acc = 0.0;
                for (uint32_t f = 0; f < F; f++) acc += dw[(uint64_t)d * F + f] * mid[((uint64_t)t * n_out + s) * F + f];
                part[((uint64_t)t * n_out + s) * E + d] = acc;
            }
        }
    }
    ds4_gpu_tensor *gx = upload(x, (uint64_t)T * E);
    ds4_gpu_tensor *gsel = ds4_gpu_tensor_alloc((uint64_t)T * slots * 4);
    require_ok(ds4_gpu_tensor_write(gsel, 0, sel, (uint64_t)T * slots * 4), "sel write");
    ds4_gpu_tensor *gmid = upload(NULL, (uint64_t)T * n_out * F);
    ds4_gpu_tensor *gpart = upload(NULL, (uint64_t)T * n_out * E);
    require_ok(ds4_gpu_qwen35_moe_mid_tensor(gmid, gx, gsel, a->base, a->size, gate_off, up_off, 13u, NE, T, slots,
                                             E, F, sg_off, su_off, 8u), "qwen35 moe mid");
    /* The down pass reads the reference mid, so a mid error cannot mask a down error. */
    float *mid_f = malloc((uint64_t)T * n_out * F * sizeof(float));
    for (uint64_t i = 0; i < (uint64_t)T * n_out * F; i++) mid_f[i] = (float)mid[i];
    ds4_gpu_tensor *gmid_ref = upload(mid_f, (uint64_t)T * n_out * F);
    require_ok(ds4_gpu_qwen35_moe_down_tensor(gpart, gmid_ref, gsel, a->base, a->size, down_off, 13u, NE, T, slots,
                                              F, E, sd_off, 8u), "qwen35 moe down");
    char what[64];
    snprintf(what, sizeof(what), "q5_K moe mid T=%u", T);
    {
        float *got = download(gmid, (uint64_t)T * n_out * F);
        check_close(what, got, mid, (uint64_t)T * n_out * F, 2e-4);
        free(got);
    }
    snprintf(what, sizeof(what), "q5_K moe down T=%u", T);
    {
        float *got = download(gpart, (uint64_t)T * n_out * E);
        check_close(what, got, part, (uint64_t)T * n_out * E, 2e-4);
        free(got);
    }
    ds4_gpu_tensor_free(gx); ds4_gpu_tensor_free(gsel); ds4_gpu_tensor_free(gmid);
    ds4_gpu_tensor_free(gmid_ref); ds4_gpu_tensor_free(gpart);
    free(x); free(sel); free(mid); free(part); free(mid_f);
    free(gate_w); free(up_w); free(down_w); free(sg_w); free(su_w); free(sd_w);
}

int main(void) {
    arena_t arena;
    arena.size = (uint64_t)512 << 20;
    arena.base = mmap(NULL, arena.size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    arena.used = 0;
    if (arena.base == MAP_FAILED) { perror("mmap"); return 1; }
    require_ok(ds4_gpu_init(), "GPU initialization");
    require_ok(ds4_gpu_set_model_map(arena.base, arena.size), "model map registration");

    printf("qwen35 moe (q5_K experts, q8_0 shared slot)\n");
    test_moe_q5k(&arena, 16, 8, 512, 256, 1);
    test_moe_q5k(&arena, 16, 8, 512, 256, 2);
    test_moe_q5k(&arena, 16, 8, 512, 512, 5);
    printf("qwen35 kernels: ok\n");
    return 0;
}
