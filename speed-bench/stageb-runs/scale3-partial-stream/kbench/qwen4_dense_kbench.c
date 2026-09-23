/* qwen4_dense_kbench: micro-bench + bit dump for a qwen4 dense decode matvec
 * (Q4_K/Q8_0 via ds4_gpu_matmul_quant_tensor, F16) on a real weight tensor.
 * usage: KB_GGUF=<model.gguf> KB_TENSORS=<tensors.txt> qwen4_dense_kbench <tensor> <in_dim> <out_dim> <T> <reps> <outfile>
 * Build as qwen4_moe_kbench.c.
 * The tensor is copied NCOPY times so the timed loop streams from DRAM. */
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

#define NCOPY 24

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 11); }
static float frnd(void) { float s = 0; for (int i = 0; i < 12; i++) s += (rnd() & 0xFFFFFF) / 16777216.0f; return s - 6.0f; }
static void die(int ok, const char *what) { if (!ok) { fprintf(stderr, "FAIL %s\n", what); exit(1); } }

int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "usage: kbd tensor in out T reps outfile\n"); return 1; }
    const uint32_t in = (uint32_t)atoi(argv[2]), outd = (uint32_t)atoi(argv[3]), T = (uint32_t)atoi(argv[4]);
    const int reps = atoi(argv[5]);
    char name[96]; uint32_t type = 0; unsigned long long foff = 0, bytes = 0; int found = 0;
    FILE *tf = fopen(getenv("KB_TENSORS") ? getenv("KB_TENSORS") : "tensors.txt", "r");
    while (fscanf(tf, "%95s %u %llu %llu", name, &type, &foff, &bytes) == 4) if (!strcmp(name, argv[1])) { found = 1; break; }
    fclose(tf);
    die(found, "tensor");
    const uint64_t stride = (bytes + 16383) & ~16383ull, total = stride * NCOPY;
    uint8_t *base = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    int fd = open(getenv("KB_GGUF"), O_RDONLY);
    uint64_t done = 0;
    while (done < bytes) { ssize_t r = pread(fd, base + done, bytes - done, foff + done); die(r > 0, "pread"); done += (uint64_t)r; }
    close(fd);
    for (int c = 1; c < NCOPY; c++) memcpy(base + c * stride, base, bytes);
    die(ds4_gpu_init(), "init");
    die(ds4_gpu_set_model_map(base, total), "map");
    float *x = malloc(sizeof(float) * T * in);
    for (uint32_t i = 0; i < T * in; i++) x[i] = frnd();
    ds4_gpu_tensor *gx = ds4_gpu_tensor_alloc((uint64_t)T * in * 4);
    ds4_gpu_tensor *go = ds4_gpu_tensor_alloc((uint64_t)T * outd * 4);
    ds4_gpu_tensor_write(gx, 0, x, (uint64_t)T * in * 4);
#define RUN(off) (type == 1u ? ds4_gpu_matmul_f16_tensor(go, base, total, (off), in, outd, gx, T) \
                             : ds4_gpu_matmul_quant_tensor(go, base, total, (off), type, in, outd, gx, T))
    die(ds4_gpu_begin_commands(), "begin"); die(RUN(0), "run");
    die(ds4_gpu_end_commands(), "end"); die(ds4_gpu_synchronize(), "sync");
    float *o = malloc((uint64_t)T * outd * 4);
    ds4_gpu_tensor_read(go, 0, o, (uint64_t)T * outd * 4);
    FILE *of = fopen(argv[6], "wb"); fwrite(o, 4, (size_t)T * outd, of); fclose(of);
    for (int w = 0; w < 2; w++) {
        const int n = w ? reps : 24;
        die(ds4_gpu_begin_commands(), "begin");
        const double t0 = now();
        for (int r = 0; r < n; r++) die(RUN((uint64_t)(r % NCOPY) * stride), "k");
        die(ds4_gpu_end_commands(), "end"); die(ds4_gpu_synchronize(), "sync");
        if (w) printf("%s T=%u: %.2f us/call (floor %.1f us @281GB/s)\n", argv[1], T, 1e6 * (now() - t0) / n, bytes / 281e3);
    }
    return 0;
}
