#ifndef DS4_EXPERT_PAGER_H
#define DS4_EXPERT_PAGER_H

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
    uint32_t page_size;
    char gguf_sha256[65];
    char sha256[65];
} ds4_expert_index_header;

typedef struct {
    int bin_fd;
    int index_fd;
    void *bundle_map;      /* mmap of qwen38-experts.bin */
    uint64_t bundle_size;
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

/* Return per-tensor bundle sizes from the pager header (gate/up/down).
 * Caller must ensure 0 <= tensor_idx < 3. */
static inline uint64_t ds4_expert_pager_bundle_size(const ds4_expert_pager *p,
                                                    uint32_t tensor_idx) {
    switch (tensor_idx) {
        case 0: return (uint64_t)p->header.bundle_bytes_gate;
        case 1: return (uint64_t)p->header.bundle_bytes_up;
        default: return (uint64_t)p->header.bundle_bytes_down;
    }
}

/* Ensure expert is resident in L1 cache.
 * tensor_idx: 0=gate, 1=up, 2=down — filters bundles to the requested tensor type.
 * Returns: 0 = all resident, >0 = misses occurred */
int ds4_expert_pager_ensure(ds4_expert_pager *pager,
                            uint32_t layer,
                            const uint32_t *expert_ids,
                            uint32_t n_experts,
                            uint32_t tensor_idx,
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

/* ==========================================================================
 * Stage C: Async double-buffer prefetch API
 * ========================================================================== */

/* Start the async worker thread. Call once before graph allocation. */
bool ds4_expert_pager_async_start(void);

/* Stop the async worker thread. Call at shutdown. */
void ds4_expert_pager_async_stop(void);

/* Issue async prefetch for next-layer experts into L2 buffer.
 * Returns: true = job queued, false = worker busy/full */
bool ds4_expert_pager_async_prefetch(ds4_expert_pager *pager,
                                     uint32_t layer,
                                     const uint32_t *expert_ids,
                                     uint32_t n_experts,
                                     uint32_t tensor_idx,
                                     void *target_l2_buf,
                                     uint64_t buf_size);

/* Wait for async job to complete (blocking). Returns success status. */
bool ds4_expert_pager_async_wait(void);

/* Check if async job is done (non-blocking). */
bool ds4_expert_pager_async_is_done(void);

/* Allocate L1/L2 double buffers for all (layer, tensor) combos. */
bool ds4_expert_pager_alloc_double_bufs(ds4_expert_pager *pager,
                                        uint64_t gate_bytes,
                                        uint64_t up_bytes,
                                        uint64_t down_bytes);

/* Get active L1 buffer pointer for (layer, tensor). */
void *ds4_expert_pager_get_l1_buf(uint32_t layer, uint32_t tensor_idx);

/* Swap L1/L2 buffers for next iteration. */
void *ds4_expert_pager_swap_buffer(uint32_t layer, uint32_t tensor_idx);

/* ==========================================================================
 * Stage D: Predictive next-layer prefetch API
 * ========================================================================== */

/* Record expert selection for predictor training.
 * Call after each layer's router completes. */
void ds4_expert_pager_record_selection(ds4_expert_pager *pager,
                                       uint32_t layer,
                                       const uint32_t *expert_ids,
                                       uint32_t n_experts);

/* Predict next layer's experts using simple heuristic:
 * Returns: true = prediction available, false = no prediction
 * Uses: same-expert recurrence (layer N experts ≈ layer N+1) */
bool ds4_expert_pager_predict_next_layer(ds4_expert_pager *pager,
                                         uint32_t next_layer,
                                         uint32_t *out_expert_ids,
                                         uint32_t *out_n_experts);

/* Issue async prefetch for predicted next-layer experts.
 * Returns: true = prefetch issued, false = no prediction/worker busy */
bool ds4_expert_pager_prefetch_predicted(ds4_expert_pager *pager,
                                         uint32_t next_layer,
                                         const uint32_t *predicted_experts,
                                         uint32_t n_experts,
                                         uint32_t tensor_idx,
                                         void *target_l2_buf,
                                         uint64_t buf_size);

/* Get prediction statistics */
void ds4_expert_pager_get_prediction_stats(const ds4_expert_pager *pager,
                                           uint64_t *out_predictions,
                                           uint64_t *out_hits,
                                           double *out_hit_rate);

#endif
