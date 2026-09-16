#ifndef DS4_EXPERT_PAGER_H
#define DS4_BUGLER_H

#include <stdbool.h>
#include <stdint.h>

#define DS4_EXPERT_PAGER_MAX_BUNDLES 73728  /* 48 layers × 512 experts × 3 tensors */
#define DS4_EXPERT_PAGER_PAGE_SIZE   4096

typedef struct {
    uint32_t layer;
    uint32_t expert;
    uint32_t tensor_idx;  /* 0=gate, 1=up, 2=down */
    uint64_t offset;
    uint64_t size;
    uint32_t type;       /* GGUF type: 16=IQ2_XXS, 10=Q2_K */
    char sha256[65];     /* hex digest */
} ds4_expert_bundle;

typedef struct {
    uint32_t version;
    uint32_t layer_count;
    uint32_t expert_count;
    uint32_t bundle_bytes_gate;
    uint32_t bundle_bytes_up;
    uint32_t bundle_bytes_down;
    uint32_t bundle_count;
    char gguf_sha256[65];
    char sha256[65];
} ds4_expert_index_header;

typedef struct {
    int bin_fd;
    int index_fd;
    ds4_expert_index_header header;
    ds4_expert_bundle *bundles;
    uint64_t bundles_capacity;
    uint64_t bundles_count;

    /* Stats */
    uint64_t total_preads;
    uint64_t total_pread_bytes;
    uint64_t total_misses;
    uint64_t total_hits;
    double total_pread_latency_ms;
} ds4_expert_pager;

/* Open pager — reads bundle file + JSON index */
bool ds4_expert_pager_open(ds4_expert_pager *pager,
                           const char *bin_path,
                           const char *index_path);

/* Close pager */
void ds4_expert_pager_close(ds4_expert_pager *pager);

/* Ensure expert is resident in L1 cache
 * Returns: 0 = all resident, >0 = misses occurred */
int ds4_expert_pager_ensure(ds4_expert_pager *pager,
                            uint32_t layer,
                            const uint32_t *expert_ids,
                            uint32_t n_experts,
                            void **out_pointers);

/* Get stats (call at exit for diagnostic report) */
void ds4_expert_pager_get_stats(const ds4_expert_pager *pager,
                                uint64_t *out_misses,
                                uint64_t *out_hits,
                                double *out_hit_rate,
                                uint64_t *out_pread_bytes,
                                double *out_total_latency_ms);

/* Print diagnostic report to stdout */
void ds4_expert_pager_report_stats(const ds4_expert_pager *pager);

#endif
