/* ds4_kv_cache.c — Paged KV cache for Qwen4 MoE
 * 
 * Architecture:
 *   - Page-based allocation with fixed-size pages (256/512/1024 tokens)
 *   - Page table maps (layer, head, token_pos) → physical page
 *   - No copy-on-grow: grow = allocate new page, never move old ones
 *   - BF16 storage with optional FP8 support (Phase 2)
 */

#include "ds4_kv_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

/* ---------------------------------------------------------------------------
 * Initialization
 * --------------------------------------------------------------------------- */

bool ds4_kv_cache_init(ds4_kv_cache *cache,
                       uint32_t max_tokens,
                       uint32_t n_layers,
                       uint32_t n_heads,
                       uint32_t head_dim,
                       uint32_t page_size_tokens) {
    if (!cache) return false;
    
    /* Validate parameters */
    if (page_size_tokens != 256 && page_size_tokens != 512 && page_size_tokens != 1024) {
        fprintf(stderr, "ds4_kv_cache: invalid page_size_tokens=%u (must be 256/512/1024)\n", 
                page_size_tokens);
        return false;
    }
    
    if (n_layers == 0 || n_heads == 0 || head_dim == 0) {
        fprintf(stderr, "ds4_kv_cache: invalid dimensions\n");
        return false;
    }
    
    memset(cache, 0, sizeof(*cache));
    
    cache->page_size_tokens = page_size_tokens;
    cache->n_layers = n_layers;
    cache->n_heads = n_heads;
    cache->head_dim = head_dim;
    
    /* Allocate page table (conservative estimate: 2x max_tokens pages) */
    uint32_t initial_pages = (max_tokens / page_size_tokens + 1) * 2;
    cache->page_table = calloc(initial_pages, sizeof(ds4_kv_page_entry));
    if (!cache->page_table) {
        fprintf(stderr, "ds4_kv_cache: failed to alloc page table (%u pages)\n", initial_pages);
        return false;
    }
    cache->page_table_capacity = initial_pages;
    
    /* Allocate physical page buffers */
    cache->page_buffers = calloc(initial_pages, sizeof(void *));
    cache->page_refcount = calloc(initial_pages, sizeof(uint32_t));
    if (!cache->page_buffers || !cache->page_refcount) {
        fprintf(stderr, "ds4_kv_cache: failed to alloc page metadata\n");
        free(cache->page_table);
        free(cache->page_buffers);
        free(cache->page_refcount);
        return false;
    }
    
    /* Calculate bytes per token per layer+head */
    /* BF16 = 2 bytes per value, head_dim values per token */
    uint32_t bytes_per_token = 2 * head_dim;
    uint32_t bytes_per_page = page_size_tokens * bytes_per_token;
    
    /* Pre-allocate first batch of pages */
    uint32_t first_batch = initial_pages / 2;
    for (uint32_t i = 0; i < first_batch; i++) {
        void *buf = malloc(bytes_per_page);
        if (buf) {
            cache->page_buffers[i] = buf;
            cache->n_pages++;
            cache->page_allocs++;
            cache->total_bytes += bytes_per_page;
        }
    }
    
    fprintf(stderr, "ds4_kv_cache: initialized %u layers × %u heads × %u dim, "
            "page=%u tokens, pre-alloc %u/%u pages\n",
            n_layers, n_heads, head_dim, page_size_tokens, 
            first_batch, initial_pages);
    
    return true;
}

/* ---------------------------------------------------------------------------
 * Cleanup
 * --------------------------------------------------------------------------- */

void ds4_kv_cache_free(ds4_kv_cache *cache) {
    if (!cache) return;

    if (cache->page_scales) {
        for (uint32_t i = 0; i < cache->n_pages; i++) {
            free(cache->page_scales[i]);
        }
        free(cache->page_scales);
    }
    if (cache->page_buffers) {
        for (uint32_t i = 0; i < cache->n_pages; i++) {
            free(cache->page_buffers[i]);
        }
        free(cache->page_buffers);
    }
    free(cache->page_refcount);
    free(cache->page_table);

    memset(cache, 0, sizeof(*cache));
}

/* ---------------------------------------------------------------------------
 * FP8 storage mode (Phase 2)
 * --------------------------------------------------------------------------- */

bool ds4_kv_cache_enable_fp8(ds4_kv_cache *cache) {
    if (!cache) return false;
    if (cache->storage_mode != 0) return true; /* already FP8 */

    cache->storage_mode = 1;

    /* Allocate per-token scale buffers for each page */
    cache->page_scales = calloc(cache->page_table_capacity, sizeof(uint8_t *));
    if (!cache->page_scales) {
        fprintf(stderr, "ds4_kv_cache: failed to alloc FP8 scales\n");
        cache->storage_mode = 0;
        return false;
    }

    for (uint32_t i = 0; i < cache->n_pages; i++) {
        cache->page_scales[i] = malloc(cache->page_size_tokens);
        if (!cache->page_scales[i]) {
            /* Roll back allocated scales */
            for (uint32_t j = 0; j < i; j++) {
                free(cache->page_scales[j]);
            }
            free(cache->page_scales);
            cache->page_scales = NULL;
            cache->storage_mode = 0;
            return false;
        }
    }

    fprintf(stderr, "ds4_kv_cache: FP8 mode enabled for %u pages\n", cache->n_pages);
    return true;
}

/* ---------------------------------------------------------------------------
 * Page allocation (grow)
 * --------------------------------------------------------------------------- */

bool ds4_kv_cache_grow(ds4_kv_cache *cache, uint32_t n_new_tokens) {
    if (!cache || n_new_tokens == 0) return false;
    
    uint32_t bytes_per_token = 2 * cache->head_dim;
    uint32_t bytes_per_page = cache->page_size_tokens * bytes_per_token;
    
    /* Calculate pages needed */
    uint32_t pages_needed = (n_new_tokens / cache->page_size_tokens) + 1;
    uint32_t pages_available = cache->page_table_capacity - cache->n_pages;
    
    if (pages_available < pages_needed) {
        /* Grow page table */
        uint32_t new_capacity = cache->page_table_capacity * 2;
        while (new_capacity - cache->n_pages < pages_needed) {
            new_capacity *= 2;
        }
        
        ds4_kv_page_entry *new_table = realloc(cache->page_table, 
                                               new_capacity * sizeof(ds4_kv_page_entry));
        void **new_bufs = realloc(cache->page_buffers, 
                                  new_capacity * sizeof(void *));
        uint32_t *new_ref = realloc(cache->page_refcount, 
                                    new_capacity * sizeof(uint32_t));
        if (!new_table || !new_bufs || !new_ref) {
            fprintf(stderr, "ds4_kv_cache: failed to grow page table to %u\n", new_capacity);
            return false;
        }
        
        cache->page_table = new_table;
        cache->page_buffers = new_bufs;
        cache->page_refcount = new_ref;
        cache->page_table_capacity = new_capacity;
    }
    
    /* Allocate new pages */
    uint32_t pages_to_alloc = pages_needed;
    for (uint32_t i = 0; i < pages_to_alloc; i++) {
        uint32_t slot = cache->n_pages;
        cache->page_buffers[slot] = malloc(bytes_per_page);
        if (!cache->page_buffers[slot]) {
            fprintf(stderr, "ds4_kv_cache: failed to allocate page %u\n", slot);
            break;
        }
        cache->page_refcount[slot] = 0;
        cache->n_pages++;
        cache->page_allocs++;
        cache->total_bytes += bytes_per_page;
    }
    
    cache->n_tokens_used += n_new_tokens;
    
    fprintf(stderr, "ds4_kv_cache: grew by %u tokens, now %u/%u pages (%u tokens)\n",
            n_new_tokens, cache->n_pages, cache->page_table_capacity, 
            cache->n_tokens_used);
    
    return true;
}

/* ---------------------------------------------------------------------------
 * Page lookup
 * --------------------------------------------------------------------------- */

void *ds4_kv_cache_get_page(ds4_kv_cache *cache,
                            uint32_t layer,
                            uint32_t head_idx,
                            uint32_t token_pos,
                            uint32_t *out_offset) {
    if (!cache || layer >= cache->n_layers || head_idx >= cache->n_heads) {
        return NULL;
    }

    /* Calculate which page slot this logical position maps to */
    /* Each (layer, head) pair gets its own page ring */
    uint32_t pair_id = layer * cache->n_heads + head_idx;
    uint32_t token_in_sequence = token_pos;

    /* Page index for this token sequence */
    uint32_t page_slot = pair_id * (cache->page_table_capacity / (cache->n_layers * cache->n_heads));

    if (page_slot >= cache->page_table_capacity) {
        return NULL;
    }

    /* Use simple sequential allocation: page_slot = pair_id (modulo available pages) */
    page_slot = pair_id % cache->n_pages;

    /* Ensure we have a buffer at this slot */
    if (!cache->page_buffers[page_slot]) {
        uint32_t bytes_per_page = cache->page_size_tokens * cache->head_dim * 2;
        cache->page_buffers[page_slot] = malloc(bytes_per_page);
        if (!cache->page_buffers[page_slot]) {
            return NULL;
        }
        cache->page_refcount[page_slot] = 1;
        cache->total_bytes += bytes_per_page;
    }

    /* Offset within the page */
    *out_offset = token_in_sequence % cache->page_size_tokens;

    return cache->page_buffers[page_slot];
}

/* ---------------------------------------------------------------------------
 * Data commit
 * --------------------------------------------------------------------------- */

bool ds4_kv_cache_commit(ds4_kv_cache *cache,
                         uint32_t layer,
                         uint32_t head_idx,
                         uint32_t token_pos,
                         const void *data,
                         uint32_t data_bytes) {
    if (!cache || !data) return false;
    
    uint32_t offset;
    void *page = ds4_kv_cache_get_page(cache, layer, head_idx, token_pos, &offset);
    if (!page) return false;
    
    uint32_t bytes_per_token = cache->head_dim * 2;
    if (offset + data_bytes > cache->page_size_tokens * bytes_per_token) {
        fprintf(stderr, "ds4_kv_cache: data overflow at token %u\n", token_pos);
        return false;
    }
    
    memcpy((uint8_t *)page + offset * bytes_per_token, data, data_bytes);
    return true;
}

/* ---------------------------------------------------------------------------
 * Stats
 * --------------------------------------------------------------------------- */

void ds4_kv_cache_get_stats(const ds4_kv_cache *cache,
                            uint64_t *out_total_bytes,
                            uint64_t *out_page_allocs,
                            uint64_t *out_page_frees,
                            uint32_t *out_n_tokens_used) {
    if (!cache) return;
    if (out_total_bytes) *out_total_bytes = cache->total_bytes;
    if (out_page_allocs) *out_page_allocs = cache->page_allocs;
    if (out_page_frees) *out_page_frees = cache->page_frees;
    if (out_n_tokens_used) *out_n_tokens_used = cache->n_tokens_used;
}

void ds4_kv_cache_report_stats(const ds4_kv_cache *cache) {
    if (!cache) return;
    
    uint64_t total_bytes, allocs, frees;
    uint32_t n_tokens;
    ds4_kv_cache_get_stats(cache, &total_bytes, &allocs, &frees, &n_tokens);
    
    fprintf(stderr, "\n=== KV Cache Stats ===\n");
    fprintf(stderr, "Page size:          %u tokens\n", cache->page_size_tokens);
    fprintf(stderr, "Layers × Heads:     %u × %u\n", cache->n_layers, cache->n_heads);
    fprintf(stderr, "Head dim:           %u\n", cache->head_dim);
    fprintf(stderr, "Pages allocated:    %u / %u\n", cache->n_pages, cache->page_table_capacity);
    fprintf(stderr, "Tokens used:        %u\n", n_tokens);
    fprintf(stderr, "Total bytes:        %llu (%.2f MB)\n", 
            (unsigned long long)total_bytes, total_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "Page allocs:        %llu\n", (unsigned long long)allocs);
    fprintf(stderr, "Page frees:         %llu\n", (unsigned long long)frees);
    fprintf(stderr, "======================\n\n");
}
