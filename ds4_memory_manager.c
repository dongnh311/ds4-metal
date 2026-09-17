/* ds4_memory_manager.c — Unified 4-way memory budget manager

 * Budget allocation strategy:
 *   - Model weights get priority (must be resident)
 *   - KV cache gets dynamic allocation based on context
 *   - Expert cache gets remaining budget (SSD-backed)
 *   - Workspace gets compute buffer requirements
 *   - OS headroom reserved for safety margin
 */

#include "ds4_memory_manager.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
/* macOS uses sysctl, Linux uses sysinfo */
#ifdef __APPLE__
#include <sys/sysctl.h>
#else
#include <sys/sysinfo.h>
#endif
#else
#include <windows.h>
static uint64_t get_system_ram_bytes(void) {
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        return (uint64_t)ms.ullTotalPhys;
    }
    return 0;
}
#endif

/* ---------------------------------------------------------------------------
 * Initialization
 * --------------------------------------------------------------------------- */

bool ds4_memory_manager_init(ds4_memory_manager *mgr, uint64_t total_ram_bytes) {
    if (!mgr) return false;
    
    memset(mgr, 0, sizeof(*mgr));
    mgr->total_ram_bytes = total_ram_bytes;
    mgr->current_ctx_tokens = 0;
    mgr->last_refit_ctx = 0;
    
    /* Reserve 20% OS headroom by default */
    mgr->headroom_bytes = total_ram_bytes / 5;
    
    fprintf(stderr, "ds4_memory_manager: initialized with %llu bytes RAM (%.2f GiB)\n",
            (unsigned long long)total_ram_bytes,
            total_ram_bytes / (1024.0 * 1024.0 * 1024.0));
    
    return true;
}

/* ---------------------------------------------------------------------------
 * Model bytes setter
 * --------------------------------------------------------------------------- */

void ds4_memory_manager_set_model_bytes(ds4_memory_manager *mgr, uint64_t model_bytes) {
    if (!mgr) return;
    mgr->model_bytes = model_bytes;
}

/* ---------------------------------------------------------------------------
 * Budget computation
 * --------------------------------------------------------------------------- */

bool ds4_memory_manager_compute_plan(ds4_memory_manager *mgr,
                                      uint32_t ctx_tokens,
                                      uint64_t kv_bytes,
                                      uint64_t expert_bytes,
                                      uint64_t workspace_bytes) {
    if (!mgr) return false;
    
    mgr->current_ctx_tokens = ctx_tokens;
    mgr->kv_bytes = kv_bytes;
    mgr->expert_cache_bytes = expert_bytes;
    mgr->workspace_bytes = workspace_bytes;
    
    /* Available budget after model and headroom */
    uint64_t available = mgr->total_ram_bytes - mgr->model_bytes - mgr->headroom_bytes;
    
    /* Split available between KV and expert cache (60/40 default) */
    uint64_t kv_budget = (uint64_t)((double)available * 0.6);
    uint64_t expert_budget = available - kv_budget;
    
    /* Cap at requested sizes */
    if (kv_bytes > kv_budget) {
        kv_budget = kv_bytes;  /* Need more than planned - reduce other allocations */
        expert_budget = available - kv_budget;
    }
    
    mgr->kv_budget_bytes = kv_budget;
    mgr->expert_budget_bytes = expert_budget;
    mgr->workspace_budget_bytes = workspace_bytes;
    mgr->os_headroom_bytes = mgr->total_ram_bytes - mgr->model_bytes - kv_budget - expert_budget - workspace_bytes;
    mgr->plan_computed = true;
    
    fprintf(stderr, "ds4_memory_manager: plan computed for %u ctx tokens\n", ctx_tokens);
    fprintf(stderr, "  Model:          %llu bytes\n", (unsigned long long)mgr->model_bytes);
    fprintf(stderr, "  KV budget:      %llu bytes (%.2f GiB)\n", 
            (unsigned long long)kv_budget, kv_budget / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "  Expert budget:  %llu bytes (%.2f GiB)\n",
            (unsigned long long)expert_budget, expert_budget / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "  Workspace:      %llu bytes\n", (unsigned long long)workspace_bytes);
    fprintf(stderr, "  OS headroom:    %llu bytes\n", (unsigned long long)mgr->os_headroom_bytes);
    
    return true;
}

/* ---------------------------------------------------------------------------
 * Dynamic re-fit at context boundaries
 * --------------------------------------------------------------------------- */

bool ds4_memory_manager_refit(ds4_memory_manager *mgr, uint32_t new_ctx_tokens) {
    if (!mgr || !mgr->plan_computed) return false;
    
    /* Only re-fit at 32K boundaries per spec */
    const uint32_t boundary = 32768;
    if ((new_ctx_tokens / boundary) != (mgr->last_refit_ctx / boundary)) {
        mgr->last_refit_ctx = new_ctx_tokens;
        /* Trigger re-computation with new KV size */
        /* For now, just note the change */
        fprintf(stderr, "ds4_memory_manager: context boundary hit at %u tokens, refit scheduled\n", 
                new_ctx_tokens);
        return true;
    }
    return false;
}

/* ---------------------------------------------------------------------------
 * Growth check
 * --------------------------------------------------------------------------- */

bool ds4_memory_manager_can_grow(ds4_memory_manager *mgr, uint64_t additional_bytes) {
    if (!mgr || !mgr->plan_computed) return false;
    
    uint64_t used = mgr->model_bytes + mgr->kv_budget_bytes + 
                    mgr->expert_budget_bytes + mgr->workspace_bytes;
    uint64_t available = mgr->total_ram_bytes - used - mgr->headroom_bytes;
    
    return additional_bytes <= available;
}

/* ---------------------------------------------------------------------------
 * Stats retrieval
 * --------------------------------------------------------------------------- */

void ds4_memory_manager_get_plan(const ds4_memory_manager *mgr,
                                   uint64_t *out_kv_budget,
                                   uint64_t *out_expert_budget,
                                   uint64_t *out_workspace_budget,
                                   uint64_t *out_headroom) {
    if (!mgr) return;
    if (out_kv_budget) *out_kv_budget = mgr->kv_budget_bytes;
    if (out_expert_budget) *out_expert_budget = mgr->expert_budget_bytes;
    if (out_workspace_budget) *out_workspace_budget = mgr->workspace_budget_bytes;
    if (out_headroom) *out_headroom = mgr->os_headroom_bytes;
}

/* ---------------------------------------------------------------------------
 * Diagnostic report
 * --------------------------------------------------------------------------- */

void ds4_memory_manager_report(const ds4_memory_manager *mgr) {
    if (!mgr) return;
    
    fprintf(stderr, "\n=== Memory Manager Report ===\n");
    fprintf(stderr, "Total RAM:        %llu bytes (%.2f GiB)\n",
            (unsigned long long)mgr->total_ram_bytes,
            mgr->total_ram_bytes / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "Model weights:    %llu bytes (%.2f GiB)\n",
            (unsigned long long)mgr->model_bytes,
            mgr->model_bytes / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "Context:          %u tokens\n", mgr->current_ctx_tokens);
    
    if (mgr->plan_computed) {
        fprintf(stderr, "\n--- Computed Budget ---\n");
        fprintf(stderr, "KV budget:        %llu bytes (%.2f GiB)\n",
                (unsigned long long)mgr->kv_budget_bytes,
                mgr->kv_budget_bytes / (1024.0 * 1024.0 * 1024.0));
        fprintf(stderr, "Expert budget:    %llu bytes (%.2f GiB)\n",
                (unsigned long long)mgr->expert_budget_bytes,
                mgr->expert_budget_bytes / (1024.0 * 1024.0 * 1024.0));
        fprintf(stderr, "Workspace:        %llu bytes\n",
                (unsigned long long)mgr->workspace_bytes);
        fprintf(stderr, "OS headroom:      %llu bytes (%.2f GiB)\n",
                (unsigned long long)mgr->os_headroom_bytes,
                mgr->os_headroom_bytes / (1024.0 * 1024.0 * 1024.0));
    } else {
        fprintf(stderr, "\nBudget not yet computed\n");
    }
    
    fprintf(stderr, "===========================\n\n");
}
