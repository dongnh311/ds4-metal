#ifndef DS4_MEMORY_MANAGER_H
#define DS4_MEMORY_MANAGER_H

#include <stdbool.h>
#include <stdint.h>

/* Unified memory budget manager for 4-way allocation:
 *   1. Model weights (resident in GPU/CPU memory)
 *   2. KV cache (paged, grows with context)
 *   3. Expert cache (SSD-backed via pager)
 *   4. Metal workspace (compute buffers)
 */

#define DS4_MEM_BUDGET_MAX_ROUTES 8

typedef struct {
    /* Total available RAM */
    uint64_t total_ram_bytes;
    
    /* Current allocations */
    uint64_t model_bytes;
    uint64_t kv_bytes;
    uint64_t expert_cache_bytes;
    uint64_t workspace_bytes;
    
    /* Desired allocations (from plan) */
    uint64_t desired_kv_bytes;
    uint64_t desired_expert_bytes;
    
    /* Headroom buffer */
    uint64_t headroom_bytes;
    
    /* Computed budget plan */
    bool plan_computed;
    uint64_t kv_budget_bytes;
    uint64_t expert_budget_bytes;
    uint64_t workspace_budget_bytes;
    uint64_t os_headroom_bytes;
    
    /* Context boundary tracking */
    uint32_t current_ctx_tokens;
    uint32_t last_refit_ctx;
} ds4_memory_manager;

/* Initialize memory manager with system RAM */
bool ds4_memory_manager_init(ds4_memory_manager *mgr, uint64_t total_ram_bytes);

/* Set model bytes (from GGUF load) */
void ds4_memory_manager_set_model_bytes(ds4_memory_manager *mgr, uint64_t model_bytes);

/* Compute budget plan for given context size */
bool ds4_memory_manager_compute_plan(ds4_memory_manager *mgr,
                                      uint32_t ctx_tokens,
                                      uint64_t kv_bytes,
                                      uint64_t expert_bytes,
                                      uint64_t workspace_bytes);

/* Dynamic re-fit when context changes */
bool ds4_memory_manager_refit(ds4_memory_manager *mgr,
                               uint32_t new_ctx_tokens);

/* Check if we can grow by requested amount */
bool ds4_memory_manager_can_grow(ds4_memory_manager *mgr, uint64_t additional_bytes);

/* Get current budget breakdown */
void ds4_memory_manager_get_plan(const ds4_memory_manager *mgr,
                                  uint64_t *out_kv_budget,
                                  uint64_t *out_expert_budget,
                                  uint64_t *out_workspace_budget,
                                  uint64_t *out_headroom);

/* Print diagnostic report */
void ds4_memory_manager_report(const ds4_memory_manager *mgr);

#endif
