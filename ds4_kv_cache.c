/* ds4_kv_cache.c — Paged KV cache for Qwen4 MoE
 *
 * Architecture:
 *   - Page-based allocation with fixed-size pages (256/512/1024 tokens)
 *   - Each (layer, head) pair has its own ring of pages for sequential writes
 *   - No copy-on-grow: grow = allocate new pages, never move old ones
 *   - BF16 storage (2 bytes per float); FP8 E4M3 planned for Phase 2
 *
 * Kernel access pattern (wired in ds4.c):
 *   - Before each attention call: materialize paged pages into contiguous GPU staging buffer
 *   - Attention kernels read from the contiguous staging buffer
 */

#include "ds4_kv_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

/* ---------------------------------------------------------------------------
 * Helpers
 * --------------------------------------------------------------------------- */

static uint32_t kv_cache_bytes_per_token(const ds4_kv_cache *cache) {
    return cache->storage_mode == 1 ? cache->head_dim : (2u * cache->head_dim);
}

static uint32_t kv_cache_page_bytes(const ds4_kv_cache *cache) {
    return cache->page_size_tokens * kv_cache_bytes_per_token(cache);
}

static inline uint32_t kv_cache_pages_per_ring(const ds4_kv_cache *cache) {
    uint32_t total_pairs = cache->n_layers * cache->n_heads;
    if (total_pairs == 0 || cache->page_table_capacity == 0) return 1;
    uint32_t pp = cache->page_table_capacity / total_pairs;
    return pp > 0 ? pp : 1;
}

static inline uint32_t kv_cache_ring_start(const ds4_kv_cache *cache, uint32_t layer, uint32_t head_idx) {
    uint32_t pair_id = layer * cache->n_heads + head_idx;
    return pair_id * kv_cache_pages_per_ring(cache);
}

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

    /* Allocate page table with enough entries for all (layer, head) pairs.
     * pages_per_ring MUST cover the full context: get_page() indexes the ring
     * with (token_pos/page_size) % pages_per_ring, so a ring shorter than
     * ceil(max_tokens/page_size) WRAPS and aliases earlier tokens, silently
     * corrupting long contexts. The prior hard-coded "*4" capped every ring at
     * 4*page_size (=2048 tokens @512) -- fine only for tiny gate runs. */
    uint32_t total_pairs = n_layers * n_heads;
    uint32_t ring_pages = 4u;
    if (max_tokens > 0) {
        ring_pages = (max_tokens + page_size_tokens - 1u) / page_size_tokens + 1u;
        if (ring_pages < 4u) ring_pages = 4u;
    }
    if ((uint64_t)total_pairs * ring_pages > DS4_KV_CACHE_MAX_PAGES) {
        uint32_t capped = DS4_KV_CACHE_MAX_PAGES / total_pairs;
        if (capped < 4u) capped = 4u;
        fprintf(stderr, "ds4_kv_cache: WARNING max_tokens=%u wants %u pages/ring but total cap is %u; "
                "clamping to %u pages/ring (~%u tokens/head max)\n",
                max_tokens, ring_pages, DS4_KV_CACHE_MAX_PAGES, capped, capped * page_size_tokens);
        ring_pages = capped;
    }
    uint32_t initial_pages = total_pairs * ring_pages;
    cache->page_table = calloc(initial_pages, sizeof(ds4_kv_page_entry));
    if (!cache->page_table) {
        fprintf(stderr, "ds4_kv_cache: failed to alloc page table (%u pages)\n", initial_pages);
        return false;
    }
    cache->page_table_capacity = initial_pages;

    /* Allocate physical page buffers and refcount. */
    cache->page_buffers = calloc(initial_pages, sizeof(void *));
    cache->page_refcount = calloc(initial_pages, sizeof(uint32_t));
    if (!cache->page_buffers || !cache->page_refcount) {
        fprintf(stderr, "ds4_kv_cache: failed to alloc page metadata\n");
        free(cache->page_table);
        free(cache->page_buffers);
        free(cache->page_refcount);
        return false;
    }

    /* Allocate per-(layer, head) write position trackers. */
    cache->write_pos = calloc((size_t)n_layers * n_heads, sizeof(uint32_t));
    if (!cache->write_pos) {
        fprintf(stderr, "ds4_kv_cache: failed to alloc write_pos\n");
        free(cache->page_table);
        free(cache->page_buffers);
        free(cache->page_refcount);
        return false;
    }

    /* Pre-allocate the first page of each ring (ring start = p*pages_per_ring). */
    uint32_t bytes_per_page = kv_cache_page_bytes(cache);
    uint32_t pages_per_ring = kv_cache_pages_per_ring(cache);
    for (uint32_t p = 0; p < total_pairs; p++) {
        uint32_t slot = p * pages_per_ring;
        if (slot >= initial_pages) break;
        void *buf = malloc(bytes_per_page);
        if (buf) {
            cache->page_buffers[slot] = buf;
            cache->n_pages++;
            cache->page_allocs++;
            cache->total_bytes += bytes_per_page;
        }
    }

    fprintf(stderr, "ds4_kv_cache: initialized %u layers × %u heads × %u dim, "
            "page=%u tokens, %u pages/ring (covers ~%u tokens/head), pre-alloc %u/%u pages\n",
            n_layers, n_heads, head_dim, page_size_tokens, ring_pages,
            ring_pages * page_size_tokens, cache->n_pages, initial_pages);

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
    free(cache->write_pos);

    memset(cache, 0, sizeof(*cache));
}

/* ---------------------------------------------------------------------------
 * FP8 storage mode (Phase 2)
 * --------------------------------------------------------------------------- */

bool ds4_kv_cache_enable_fp8(ds4_kv_cache *cache) {
    if (!cache) return false;
    if (cache->storage_mode != 0) return true; /* already FP8 */

    cache->storage_mode = 1;

    /* Allocate per-token scale buffers for each page. */
    cache->page_scales = calloc(cache->page_table_capacity, sizeof(uint8_t *));
    if (!cache->page_scales) {
        fprintf(stderr, "ds4_kv_cache: failed to alloc FP8 scales\n");
        cache->storage_mode = 0;
        return false;
    }

    for (uint32_t i = 0; i < cache->n_pages; i++) {
        cache->page_scales[i] = malloc(cache->page_size_tokens);
        if (!cache->page_scales[i]) {
            /* Roll back allocated scales. */
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

    uint32_t pages_needed = (n_new_tokens / cache->page_size_tokens) + 1;
    uint32_t pages_per_ring = kv_cache_pages_per_ring(cache);
    uint32_t total_pairs = cache->n_layers * cache->n_heads;
    uint32_t min_pages_needed = total_pairs * pages_per_ring;

    if (cache->n_pages + pages_needed <= cache->page_table_capacity &&
        cache->n_pages >= min_pages_needed) {
        /* We have capacity; just allocate new pages. */
    } else {
        /* Grow page table. */
        uint32_t new_capacity = cache->page_table_capacity * 2;
        while (new_capacity < cache->n_pages + pages_needed) {
            new_capacity *= 2;
        }
        if (new_capacity < min_pages_needed) {
            new_capacity = min_pages_needed;
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

    /* Allocate new pages, filling empty slots in each ring. */
    uint32_t bytes_per_page = kv_cache_page_bytes(cache);
    uint32_t pages_allocated = 0;
    for (uint32_t i = 0; i < pages_needed && cache->n_pages < cache->page_table_capacity; i++) {
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
        pages_allocated++;
    }

    cache->n_tokens_used += n_new_tokens;

    fprintf(stderr, "ds4_kv_cache: grew by %u tokens, now %u/%u pages (%u tokens total)\n",
            n_new_tokens, cache->n_pages, cache->page_table_capacity,
            cache->n_tokens_used);

    return pages_allocated == pages_needed;
}

/* ---------------------------------------------------------------------------
 * Page lookup
 * ---------------------------------------------------------------------------
 *
 * Returns the page buffer and byte offset for a given (layer, head, token_pos).
 * Uses a ring buffer per (layer, head) pair.
 * --------------------------------------------------------------------------- */

void *ds4_kv_cache_get_page(ds4_kv_cache *cache,
                            uint32_t layer,
                            uint32_t head_idx,
                            uint32_t token_pos,
                            uint32_t *out_offset) {
    if (!cache || layer >= cache->n_layers || head_idx >= cache->n_heads) {
        return NULL;
    }

    uint32_t ring_start = kv_cache_ring_start(cache, layer, head_idx);
    uint32_t pages_per_r = kv_cache_pages_per_ring(cache);

    /* Ring index and offset within ring page. */
    uint32_t ring_idx = (token_pos / cache->page_size_tokens) % pages_per_r;
    uint32_t offset_in_page = token_pos % cache->page_size_tokens;
    uint32_t page_slot = ring_start + ring_idx;

    if (page_slot >= cache->page_table_capacity) {
        return NULL;
    }

    /* Ensure buffer exists. */
    if (!cache->page_buffers[page_slot]) {
        uint32_t bpp = kv_cache_page_bytes(cache);
        cache->page_buffers[page_slot] = malloc(bpp);
        if (!cache->page_buffers[page_slot]) {
            return NULL;
        }
        cache->page_refcount[page_slot] = 1;
        cache->total_bytes += bpp;
    }

    *out_offset = offset_in_page * kv_cache_bytes_per_token(cache);
    return cache->page_buffers[page_slot];
}

/* ---------------------------------------------------------------------------
 * Write one token's KV data (auto-advances write position)
 * --------------------------------------------------------------------------- */

bool ds4_kv_cache_write_token(ds4_kv_cache *cache,
                              uint32_t layer,
                              uint32_t head_idx,
                              const void *kv_data,
                              uint32_t kv_bytes) {
    if (!cache || !kv_data) return false;
    if (layer >= cache->n_layers || head_idx >= cache->n_heads) return false;

    uint32_t token_pos = cache->write_pos[layer * cache->n_heads + head_idx];
    uint32_t expected_bytes = kv_cache_bytes_per_token(cache);

    if (kv_bytes != expected_bytes) {
        fprintf(stderr, "ds4_kv_cache: wrong bytes per token: got %u, expected %u\n",
                kv_bytes, expected_bytes);
        return false;
    }

    uint32_t offset;
    void *page = ds4_kv_cache_get_page(cache, layer, head_idx, token_pos, &offset);
    if (!page) return false;

    memcpy((uint8_t *)page + offset, kv_data, kv_bytes);
    cache->write_pos[layer * cache->n_heads + head_idx] = token_pos + 1;
    cache->n_tokens_used = cache->write_pos[layer * cache->n_heads + head_idx];

    return true;
}

/* ---------------------------------------------------------------------------
 * Data commit (legacy API — arbitrary position)
 * --------------------------------------------------------------------------- */

bool ds4_kv_cache_commit(ds4_kv_cache *cache,
                         uint32_t layer,
                         uint32_t head_idx,
                         uint32_t token_pos,
                         const void *data,
                         uint32_t data_bytes) {
    if (!cache || !data) return false;
    if (layer >= cache->n_layers || head_idx >= cache->n_heads) return false;

    uint32_t offset;
    void *page = ds4_kv_cache_get_page(cache, layer, head_idx, token_pos, &offset);
    if (!page) return false;

    uint32_t bpt = kv_cache_bytes_per_token(cache);
    if (offset + data_bytes > cache->page_size_tokens * bpt) {
        fprintf(stderr, "ds4_kv_cache: data overflow at token %u\n", token_pos);
        return false;
    }

    memcpy((uint8_t *)page + offset, data, data_bytes);
    return true;
}

/* ---------------------------------------------------------------------------
 * Materialize: copy paged data into contiguous buffer for GPU kernel
 * ---------------------------------------------------------------------------
 *
 * Reads tokens [pos0, pos0+n_tokens) for (layer, head) and writes them
 * contiguously into staging_buf.
 * --------------------------------------------------------------------------- */

uint64_t ds4_kv_cache_materialize_kv(ds4_kv_cache *cache,
                                     uint32_t layer,
                                     uint32_t head_idx,
                                     uint32_t pos0,
                                     uint32_t n_tokens,
                                     void *staging_buf,
                                     uint64_t staging_bytes) {
    if (!cache || !staging_buf || n_tokens == 0) return 0;
    if (layer >= cache->n_layers || head_idx >= cache->n_heads) return 0;

    uint32_t bpt = kv_cache_bytes_per_token(cache);
    uint64_t total_bytes = (uint64_t)n_tokens * bpt;
    if (total_bytes > staging_bytes) return 0;

    uint8_t *dst = (uint8_t *)staging_buf;
    for (uint32_t t = 0; t < n_tokens; t++) {
        uint32_t abs_pos = pos0 + t;
        uint32_t offset;
        void *page = ds4_kv_cache_get_page(cache, layer, head_idx, abs_pos, &offset);
        if (!page) return 0;

        const uint8_t *src = (const uint8_t *)page + offset;
        memcpy(dst, src, bpt);
        dst += bpt;
    }
    return total_bytes;
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
    fprintf(stderr, "Tokens written:     %u\n", n_tokens);
    fprintf(stderr, "Total bytes:        %llu (%.2f MB)\n",
            (unsigned long long)total_bytes, total_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "Page allocs:        %llu\n", (unsigned long long)allocs);
    fprintf(stderr, "Page frees:         %llu\n", (unsigned long long)frees);
    fprintf(stderr, "======================\n\n");
}

void ds4_kv_cache_reset(ds4_kv_cache *cache) {
    if (!cache) return;
    /* Zero out all page buffers so stale KV data does not leak between runs. */
    for (uint32_t i = 0; i < cache->n_pages; i++) {
        if (cache->page_buffers[i]) {
            memset(cache->page_buffers[i], 0, kv_cache_page_bytes(cache));
        }
    }
    /* Reset write positions. */
    if (cache->write_pos) {
        memset(cache->write_pos, 0,
               (size_t)cache->n_layers * cache->n_heads * sizeof(uint32_t));
    }
    cache->n_tokens_used = 0;
    fprintf(stderr, "ds4_kv_cache: reset done (pages=%u)\n", cache->n_pages);
}
