#ifndef DS4_KV_CACHE_H
#define DS4_KV_CACHE_H

#include <stdbool.h>
#include <stdint.h>

/* Paged KV cache for Qwen4 MoE — BF16 storage with page-based allocation */
#define DS4_KV_CACHE_MAX_PAGES 65536
#define DS4_KV_CACHE_MAX_LAYERS 64
#define DS4_KV_CACHE_DEFAULT_PAGE_TOKENS 512

typedef struct {
    uint32_t layer;          /* transformer layer index */
    uint32_t head_idx;       /* which key/value head pair */
    uint32_t page_id;        /* physical page slot */
    uint32_t token_offset;   /* offset within page (0..page_tokens-1) */
} ds4_kv_page_entry;

typedef struct {
    uint32_t page_size_tokens;   /* tokens per page (256/512/1024) */
    uint32_t n_layers;           /* number of transformer layers */
    uint32_t n_heads;            /* number of KV heads per layer */
    uint32_t head_dim;           /* dimension per head */
    uint32_t n_pages;            /* total pages allocated */
    uint32_t n_tokens_used;      /* total tokens in cache */
    
    /* Page table: maps logical token position to physical page */
    ds4_kv_page_entry *page_table;
    uint32_t page_table_capacity;
    
    /* Physical page storage (GPU-visible) */
    void **page_buffers;         /* array of MTLBuffer pointers */
    uint32_t *page_refcount;     /* reference count for each page */
    
    /* Metadata */
    uint64_t total_bytes;
    uint64_t page_allocs;
    uint64_t page_frees;
} ds4_kv_cache;

/* Initialize paged KV cache */
bool ds4_kv_cache_init(ds4_kv_cache *cache,
                       uint32_t max_tokens,
                       uint32_t n_layers,
                       uint32_t n_heads,
                       uint32_t head_dim,
                       uint32_t page_size_tokens);

/* Free paged KV cache */
void ds4_kv_cache_free(ds4_kv_cache *cache);

/* Allocate new page for token position */
bool ds4_kv_cache_grow(ds4_kv_cache *cache, uint32_t n_new_tokens);

/* Get page pointer for a given (layer, head, token_position) */
void *ds4_kv_cache_get_page(ds4_kv_cache *cache,
                            uint32_t layer,
                            uint32_t head_idx,
                            uint32_t token_pos,
                            uint32_t *out_offset);

/* Commit KV data to page (called after attention computation) */
bool ds4_kv_cache_commit(ds4_kv_cache *cache,
                         uint32_t layer,
                         uint32_t head_idx,
                         uint32_t token_pos,
                         const void *data,
                         uint32_t data_bytes);

/* Get stats for diagnostic reporting */
void ds4_kv_cache_get_stats(const ds4_kv_cache *cache,
                            uint64_t *out_total_bytes,
                            uint64_t *out_page_allocs,
                            uint64_t *out_page_frees,
                            uint32_t *out_n_tokens_used);

/* Print diagnostic report */
void ds4_kv_cache_report_stats(const ds4_kv_cache *cache);

#endif
