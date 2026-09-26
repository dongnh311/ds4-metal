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

/* f32 weights with a double shadow for the reference */
static uint64_t arena_f32(arena_t *a, uint64_t n, double **shadow, float lo, float hi) {
    const uint64_t off = arena_alloc(a, n * sizeof(float));
    float *w = (float *)(a->base + off);
    *shadow = malloc(n * sizeof(double));
    for (uint64_t i = 0; i < n; i++) {
        w[i] = lo + (hi - lo) * (0.5f * frand() + 0.5f);
        (*shadow)[i] = w[i];
    }
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

/* GDN output: per-head RMSNorm of the scan output times ssm_norm, gated by
 * silu(z) (Qwen3.5 RMSNormGated). */
static void test_gdn_out_silu(arena_t *a, uint32_t H, uint32_t D, uint32_t T) {
    double *w;
    const uint64_t w_off = arena_f32(a, D, &w, 0.5f, 1.5f);
    const uint64_t n = (uint64_t)T * H * D;
    float *o = rand_vec(n, 1.0f), *z = rand_vec(n, 2.0f);
    double *ref = malloc(n * sizeof(double));
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t h = 0; h < H; h++) {
            const uint64_t base = ((uint64_t)t * H + h) * D;
            double ss = 0.0;
            for (uint32_t i = 0; i < D; i++) ss += (double)o[base + i] * o[base + i];
            const double r = 1.0 / sqrt(ss / D + 1e-6);
            for (uint32_t i = 0; i < D; i++) ref[base + i] = o[base + i] * r * w[i] * silu_d(z[base + i]);
        }
    }
    ds4_gpu_tensor *go = upload(o, n), *gz = upload(z, n);
    require_ok(ds4_gpu_qwen35_gdn_out_tensor(go, gz, a->base, a->size, w_off, T, H, D, 1e-6f), "qwen35 gdn out");
    float *got = download(go, n);
    check_close("gdn out silu", got, ref, n, 1e-5);
    free(got); free(o); free(z); free(ref); free(w);
    ds4_gpu_tensor_free(go); ds4_gpu_tensor_free(gz);
}

/* NEOX partial RoPE on the first n_rot dims (pairs i, i + n_rot/2). */
static void rope_ref(double *x, uint32_t n_rot, uint32_t pos, double base) {
    const uint32_t nh = n_rot / 2;
    for (uint32_t i = 0; i < nh; i++) {
        const double th = (double)pos * pow(base, -2.0 * i / n_rot);
        const double c = cos(th), s = sin(th), x0 = x[i], x1 = x[i + nh];
        x[i] = x0 * c - x1 * s;
        x[i + nh] = x0 * s + x1 * c;
    }
}

/* Attention prep without the indexer: q (from the interleaved [q | gate]
 * rows) and k get RMSNorm, their weight and RoPE; the gate passes through; k
 * and v are appended to the F16 caches at pos0.. . */
static void test_attn_prep_noindexer(arena_t *a) {
    const uint32_t T = 3, H = 16, Hkv = 2, D = 256, n_rot = 64, pos0 = 5, cap = 16;
    const double base = 1.0e7;
    double *gq, *gk;
    const uint64_t gq_off = arena_f32(a, D, &gq, 0.5f, 1.5f);
    const uint64_t gk_off = arena_f32(a, D, &gk, 0.5f, 1.5f);
    float *qg = rand_vec((uint64_t)T * H * 2 * D, 1.0f);
    float *kp = rand_vec((uint64_t)T * Hkv * D, 1.0f), *vp = rand_vec((uint64_t)T * Hkv * D, 1.0f);
    uint32_t pos3[16 * 4];
    for (uint32_t p = 0; p < cap; p++) { pos3[p * 4] = pos3[p * 4 + 1] = pos3[p * 4 + 2] = p; pos3[p * 4 + 3] = 0; }
    double *q_ref = malloc((uint64_t)T * H * D * sizeof(double));
    double *g_ref = malloc((uint64_t)T * H * D * sizeof(double));
    double *k_ref = malloc((uint64_t)T * Hkv * D * sizeof(double));
    double row[256];
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t h = 0; h < H; h++) {
            const float *src = qg + ((uint64_t)t * H + h) * 2 * D;
            double ss = 0.0;
            for (uint32_t i = 0; i < D; i++) ss += (double)src[i] * src[i];
            const double r = 1.0 / sqrt(ss / D + 1e-6);
            for (uint32_t i = 0; i < D; i++) row[i] = src[i] * r * gq[i];
            rope_ref(row, n_rot, pos0 + t, base);
            for (uint32_t i = 0; i < D; i++) {
                q_ref[((uint64_t)t * H + h) * D + i] = row[i];
                g_ref[((uint64_t)t * H + h) * D + i] = src[D + i];
            }
        }
        for (uint32_t h = 0; h < Hkv; h++) {
            const float *src = kp + ((uint64_t)t * Hkv + h) * D;
            double ss = 0.0;
            for (uint32_t i = 0; i < D; i++) ss += (double)src[i] * src[i];
            const double r = 1.0 / sqrt(ss / D + 1e-6);
            for (uint32_t i = 0; i < D; i++) row[i] = src[i] * r * gk[i];
            rope_ref(row, n_rot, pos0 + t, base);
            for (uint32_t i = 0; i < D; i++) k_ref[((uint64_t)t * Hkv + h) * D + i] = row[i];
        }
    }
    ds4_gpu_tensor *gqg = upload(qg, (uint64_t)T * H * 2 * D);
    ds4_gpu_tensor *gkp = upload(kp, (uint64_t)T * Hkv * D), *gvp = upload(vp, (uint64_t)T * Hkv * D);
    ds4_gpu_tensor *gq_out = upload(NULL, (uint64_t)T * H * D), *ggate = upload(NULL, (uint64_t)T * H * D);
    ds4_gpu_tensor *kc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *vc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *gpos = ds4_gpu_tensor_alloc(sizeof(pos3));
    require_ok(kc && vc && gpos && ds4_gpu_tensor_write(gpos, 0, pos3, sizeof(pos3)), "prep buffers");
    require_ok(ds4_gpu_qwen35_attn_prep_tensor(gq_out, ggate, kc, vc, gqg, gkp, gvp, gpos, a->base, a->size,
                                               gq_off, gk_off, T, H, Hkv, D, n_rot, pos0, cap,
                                               (float)base, 1e-6f), "qwen35 attn prep");
    float *got = download(gq_out, (uint64_t)T * H * D);
    check_close("attn prep q", got, q_ref, (uint64_t)T * H * D, 2e-5);
    free(got);
    got = download(ggate, (uint64_t)T * H * D);
    check_close("attn prep gate", got, g_ref, (uint64_t)T * H * D, 0.0);
    free(got);
    uint16_t *kh = malloc((uint64_t)cap * Hkv * D * 2u), *vh = malloc((uint64_t)cap * Hkv * D * 2u);
    require_ok(ds4_gpu_tensor_read(kc, 0, kh, (uint64_t)cap * Hkv * D * 2u), "k cache read");
    require_ok(ds4_gpu_tensor_read(vc, 0, vh, (uint64_t)cap * Hkv * D * 2u), "v cache read");
    float *kf = malloc((uint64_t)T * Hkv * D * sizeof(float)), *vf = malloc((uint64_t)T * Hkv * D * sizeof(float));
    double *v_ref = malloc((uint64_t)T * Hkv * D * sizeof(double));
    for (uint64_t i = 0; i < (uint64_t)T * Hkv * D; i++) {
        kf[i] = f16_to_f32(kh[(uint64_t)pos0 * Hkv * D + i]);
        vf[i] = f16_to_f32(vh[(uint64_t)pos0 * Hkv * D + i]);
        v_ref[i] = vp[i];
    }
    check_close("attn prep k cache (f16)", kf, k_ref, (uint64_t)T * Hkv * D, 2e-3);
    check_close("attn prep v cache (f16)", vf, v_ref, (uint64_t)T * Hkv * D, 2e-3);
    free(kh); free(vh); free(kf); free(vf); free(v_ref);
    free(qg); free(kp); free(vp); free(q_ref); free(g_ref); free(k_ref); free(gq); free(gk);
    ds4_gpu_tensor_free(gqg); ds4_gpu_tensor_free(gkp); ds4_gpu_tensor_free(gvp);
    ds4_gpu_tensor_free(gq_out); ds4_gpu_tensor_free(ggate); ds4_gpu_tensor_free(kc);
    ds4_gpu_tensor_free(vc); ds4_gpu_tensor_free(gpos);
}

/* MoE reduce without the residual combine (R = NULL, n_hc = 0): the shared
 * expert comes from a part slot (single rows) or a separate buffer (batches). */
static void test_reduce_nohc(uint32_t T, uint32_t K, uint32_t E, bool shared_slot) {
    const uint32_t stride = K + (shared_slot ? 1u : 0u);
    float *part = rand_vec((uint64_t)T * stride * E, 1.0f);
    float *w = rand_vec((uint64_t)T * K, 1.0f);
    float *sg = rand_vec(T, 2.0f);
    float *sh = rand_vec((uint64_t)T * E, 1.0f);
    double *ref = malloc((uint64_t)T * E * sizeof(double));
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t d = 0; d < E; d++) {
            double acc = 0.0;
            for (uint32_t s = 0; s < K; s++) acc += (double)w[t * K + s] * part[((uint64_t)t * stride + s) * E + d];
            const double shared = shared_slot ? part[((uint64_t)t * stride + K) * E + d] : sh[(uint64_t)t * E + d];
            ref[(uint64_t)t * E + d] = acc + sigmoid_d(sg[t]) * shared;
        }
    }
    ds4_gpu_tensor *gp = upload(part, (uint64_t)T * stride * E), *gw = upload(w, (uint64_t)T * K);
    ds4_gpu_tensor *gsg = upload(sg, T), *gsh = upload(sh, (uint64_t)T * E), *gout = upload(NULL, (uint64_t)T * E);
    require_ok(ds4_gpu_qwen4_moe_reduce_tensor(gout, gp, gw, gsg, shared_slot ? NULL : gsh, NULL, NULL,
                                               T, K, stride, E, 0u), "moe reduce (no hc)");
    float *got = download(gout, (uint64_t)T * E);
    check_close(shared_slot ? "moe reduce, shared slot" : "moe reduce, shared buffer", got, ref, (uint64_t)T * E, 1e-5);
    free(got); free(part); free(w); free(sg); free(sh); free(ref);
    ds4_gpu_tensor_free(gp); ds4_gpu_tensor_free(gw); ds4_gpu_tensor_free(gsg);
    ds4_gpu_tensor_free(gsh); ds4_gpu_tensor_free(gout);
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
    printf("qwen35 gdn out (silu gate)\n");
    test_gdn_out_silu(&arena, 32, 128, 3);
    printf("qwen35 attention prep (no indexer)\n");
    test_attn_prep_noindexer(&arena);
    printf("qwen4 moe reduce without residual (as Ornith calls it)\n");
    test_reduce_nohc(1, 8, 2048, true);
    test_reduce_nohc(12, 8, 2048, false);
    printf("qwen35 kernels: ok\n");
    return 0;
}
