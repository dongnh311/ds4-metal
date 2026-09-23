/* qwen4_moe_kbench: micro-bench + bit dump for the qwen4 decode MoE kernels
 * (mid gate/up, down) on one layer's real expert weights.
 * usage: KB_GGUF=<model.gguf> KB_TENSORS=<tensors.txt> qwen4_moe_kbench <layer> <T> <overlap> <reps> <outfile>
 * tensors.txt comes from kbench_tensors.py.  <overlap> of token t's ten experts
 * repeat token t-1's; 32 selection sets rotate so the timed loop reads from
 * DRAM.  Run from a checkout: the Metal sources load from ./metal.  KB_ONLY=m|d
 * times only mid or down.  Build (Makefile flags):
 *   cc -O3 -ffast-math -mcpu=native -std=c99 -D_GNU_SOURCE -fno-finite-math-only -I. \
 *      -o qwen4_moe_kbench <this file> ds4_metal.o ds4_image.o -lm -pthread \
 *      -framework Foundation -framework Metal -framework Accelerate
 * Copies one layer's routed + shared expert tensors from the GGUF into an
 * anonymous arena registered as the model map, runs mid then down once
 * (outputs written to <outfile>), then times each kernel over <reps> calls. */
#include <stdbool.h>
#include "ds4.h"
#include "ds4_gpu.h"
bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>
#include <math.h>


static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 11); }
static float frnd(void) { /* ~N(0,1) */
    float s = 0; for (int i = 0; i < 12; i++) s += (rnd() & 0xFFFFFF) / 16777216.0f; return s - 6.0f; }

typedef struct { char name[96]; uint32_t type; uint64_t file_off, bytes, arena_off; } tinfo;
static tinfo T_[64]; static int nt;

static tinfo *find(int layer, const char *suffix) {
    char want[96]; snprintf(want, sizeof want, "blk.%d.%s.weight", layer, suffix);
    for (int i = 0; i < nt; i++) if (!strcmp(T_[i].name, want)) return &T_[i];
    fprintf(stderr, "missing %s\n", want); exit(1);
}
static void die(int ok, const char *what) { if (!ok) { fprintf(stderr, "FAIL %s\n", what); exit(1); } }

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: kb layer T overlap reps outfile\n"); return 1; }
    const int layer = atoi(argv[1]); const uint32_t T = (uint32_t)atoi(argv[2]);
    const int ov = atoi(argv[3]); const int reps = atoi(argv[4]);
    FILE *tf = fopen(getenv("KB_TENSORS") ? getenv("KB_TENSORS") : "tensors.txt", "r");
    while (nt < 64 && fscanf(tf, "%95s %u %llu %llu", T_[nt].name, &T_[nt].type,
                             (unsigned long long *)&T_[nt].file_off, (unsigned long long *)&T_[nt].bytes) == 4) nt++;
    fclose(tf);
    const char *sfx[6] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp" };
    tinfo *ti[6]; uint64_t total = 0;
    for (int i = 0; i < 6; i++) { ti[i] = find(layer, sfx[i]); ti[i]->arena_off = total; total += (ti[i]->bytes + 16383) & ~16383ull; }
    uint8_t *base = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    int fd = open(getenv("KB_GGUF"), O_RDONLY);
    for (int i = 0; i < 6; i++) {
        uint64_t done = 0;
        while (done < ti[i]->bytes) {
            ssize_t r = pread(fd, base + ti[i]->arena_off + done, ti[i]->bytes - done, ti[i]->file_off + done);
            die(r > 0, "pread"); done += (uint64_t)r;
        }
    }
    close(fd);
    die(ds4_gpu_init(), "init");
    die(ds4_gpu_set_model_map(base, total), "map");
    const uint32_t E = 2560, F = 640, S = 10, NE = 512;
    float *x = malloc(sizeof(float) * T * E);
    for (uint32_t i = 0; i < T * E; i++) x[i] = frnd();
    enum { NSET = 32 };
    static int32_t selall[NSET][64 * 10];
    for (int set = 0; set < NSET; set++) {
    int32_t *sel = selall[set];
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t s = 0; s < S; s++) {
            int32_t e;
            if (t > 0 && (int)s < ov) e = sel[(t - 1) * S + (s * 3 + 1) % S];   /* shared with the previous token */
            else {
                for (;;) {
                    e = (int32_t)(rnd() % NE); int dup = 0;
                    for (uint32_t k = 0; k < s; k++) if (sel[t * S + k] == e) dup = 1;
                    if (t > 0) for (uint32_t k = 0; k < S; k++) if (sel[(t - 1) * S + k] == e) dup = 1;
                    if (!dup) break;
                }
            }
            sel[t * S + s] = e;
        }
    }
    }
    int32_t *sel = selall[0];
    ds4_gpu_tensor *gx = ds4_gpu_tensor_alloc((uint64_t)T * E * 4);
    ds4_gpu_tensor *gselall = ds4_gpu_tensor_alloc((uint64_t)NSET * T * S * 4);
    for (int set = 0; set < NSET; set++) ds4_gpu_tensor_write(gselall, (uint64_t)set * T * S * 4, selall[set], (uint64_t)T * S * 4);
    ds4_gpu_tensor *gsels[NSET];
    for (int set = 0; set < NSET; set++) gsels[set] = ds4_gpu_tensor_view(gselall, (uint64_t)set * T * S * 4, (uint64_t)T * S * 4);
    ds4_gpu_tensor *gsel = gsels[0];
    ds4_gpu_tensor *gmid = ds4_gpu_tensor_alloc((uint64_t)T * (S + 1) * F * 4);
    ds4_gpu_tensor *gpart = ds4_gpu_tensor_alloc((uint64_t)T * (S + 1) * E * 4);
    ds4_gpu_tensor_write(gx, 0, x, (uint64_t)T * E * 4);
#define MID() ds4_gpu_qwen4_moe_mid_tensor(gmid, gx, gsel, base, total, ti[0]->arena_off, ti[1]->arena_off, ti[0]->type, NE, T, S, E, F, \
                                           ti[3]->arena_off, ti[4]->arena_off, ti[3]->type)
#define DOWN() ds4_gpu_qwen4_moe_down_tensor(gpart, gmid, gsel, base, total, ti[2]->arena_off, ti[2]->type, NE, T, S, F, E, \
                                             ti[5]->arena_off, ti[5]->type)
    die(ds4_gpu_begin_commands(), "begin"); die(MID(), "mid"); die(DOWN(), "down");
    die(ds4_gpu_end_commands(), "end"); die(ds4_gpu_synchronize(), "sync");
    float *mo = malloc((uint64_t)T * (S + 1) * F * 4), *po = malloc((uint64_t)T * (S + 1) * E * 4);
    ds4_gpu_tensor_read(gmid, 0, mo, (uint64_t)T * (S + 1) * F * 4);
    ds4_gpu_tensor_read(gpart, 0, po, (uint64_t)T * (S + 1) * E * 4);
    FILE *of = fopen(argv[5], "wb");
    fwrite(mo, 4, (size_t)T * (S + 1) * F, of); fwrite(po, 4, (size_t)T * (S + 1) * E, of); fclose(of);
    double sm = 0; for (uint64_t i = 0; i < (uint64_t)T * (S + 1) * F; i++) sm += fabs(mo[i]);
    const char *which = getenv("KB_ONLY");
    for (int pass = 0; pass < 2; pass++) {
        if (which && which[0] && which[0] != (pass ? 'd' : 'm')) continue;
        for (int w = 0; w < 2; w++) {   /* warm, then timed */
            const int n = w ? reps : 20;
            die(ds4_gpu_begin_commands(), "begin");
            const double t0 = now();
            for (int r = 0; r < n; r++) { gsel = gsels[r % NSET]; die(pass ? DOWN() : MID(), "k"); }
            gsel = gsels[0];
            die(ds4_gpu_end_commands(), "end"); die(ds4_gpu_synchronize(), "sync");
            if (w) printf("%s L%d T=%u ov=%d: %.2f us/call\n", pass ? "down" : "mid ", layer, T, ov, 1e6 * (now() - t0) / n);
        }
    }
    printf("mid |sum| %.6g\n", sm);
    return 0;
}
