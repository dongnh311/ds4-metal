/* tests/test_kv_paged.c — Unit tests for paged KV cache */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "ds4_kv_cache.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { \
    printf("TEST: %s ... ", #name); \
    if (test_##name()) { \
        printf("PASS\n"); \
        tests_passed++; \
    } else { \
        printf("FAIL\n"); \
        tests_failed++; \
    } \
} while (0)

static int test_kv_init_basic(void) {
    ds4_kv_cache cache;
    bool ok = ds4_kv_cache_init(&cache, 1024, 4, 8, 64, 256);
    assert(ok);
    assert(cache.page_size_tokens == 256);
    assert(cache.n_layers == 4);
    assert(cache.n_heads == 8);
    assert(cache.head_dim == 64);
    ds4_kv_cache_free(&cache);
    return 1;
}

static int test_kv_init_invalid_page_size(void) {
    ds4_kv_cache cache;
    bool ok = ds4_kv_cache_init(&cache, 1024, 4, 8, 64, 300);
    assert(!ok);
    return 1;
}

static int test_kv_init_null_cache(void) {
    bool ok = ds4_kv_cache_init(NULL, 1024, 4, 8, 64, 256);
    assert(!ok);
    return 1;
}

static int test_kv_grow(void) {
    ds4_kv_cache cache;
    bool ok = ds4_kv_cache_init(&cache, 2048, 4, 8, 64, 256);
    assert(ok);
    
    uint32_t pages_before = cache.n_pages;
    ok = ds4_kv_cache_grow(&cache, 1024);
    assert(ok);
    assert(cache.n_pages >= pages_before);
    assert(cache.n_tokens_used >= 1024);
    
    ds4_kv_cache_free(&cache);
    return 1;
}

static int test_kv_get_page(void) {
    ds4_kv_cache cache;
    bool ok = ds4_kv_cache_init(&cache, 1024, 4, 8, 64, 256);
    assert(ok);
    
    uint32_t offset;
    void *page = ds4_kv_cache_get_page(&cache, 0, 0, 0, &offset);
    assert(page != NULL);
    assert(offset == 0);
    
    page = ds4_kv_cache_get_page(&cache, 3, 7, 255, &offset);
    assert(page != NULL);
    assert(offset == 255);
    
    page = ds4_kv_cache_get_page(&cache, 4, 0, 0, &offset);
    assert(page == NULL);  /* layer out of bounds */
    
    ds4_kv_cache_free(&cache);
    return 1;
}

static int test_kv_commit(void) {
    ds4_kv_cache cache;
    bool ok = ds4_kv_cache_init(&cache, 1024, 4, 8, 64, 256);
    assert(ok);
    
    uint8_t data[128];
    memset(data, 0xAB, sizeof(data));
    
    ok = ds4_kv_cache_commit(&cache, 0, 0, 0, data, sizeof(data));
    assert(ok);
    
    ds4_kv_cache_free(&cache);
    return 1;
}

static int test_kv_commit_null_data(void) {
    ds4_kv_cache cache;
    bool ok = ds4_kv_cache_init(&cache, 1024, 4, 8, 64, 256);
    assert(ok);
    
    ok = ds4_kv_cache_commit(&cache, 0, 0, 0, NULL, 128);
    assert(!ok);
    
    ds4_kv_cache_free(&cache);
    return 1;
}

static int test_kv_stats(void) {
    ds4_kv_cache cache;
    bool ok = ds4_kv_cache_init(&cache, 1024, 4, 8, 64, 256);
    assert(ok);
    
    uint64_t total_bytes, allocs, frees;
    uint32_t n_tokens;
    ds4_kv_cache_get_stats(&cache, &total_bytes, &allocs, &frees, &n_tokens);
    assert(total_bytes > 0);
    assert(allocs > 0);
    assert(frees == 0);
    assert(n_tokens == 0);
    
    ds4_kv_cache_free(&cache);
    return 1;
}

static int test_kv_page_sizes(void) {
    /* Test all three valid page sizes */
    uint32_t page_sizes[] = {256, 512, 1024};
    
    for (int i = 0; i < 3; i++) {
        ds4_kv_cache cache;
        bool ok = ds4_kv_cache_init(&cache, 2048, 4, 8, 64, page_sizes[i]);
        assert(ok);
        assert(cache.page_size_tokens == page_sizes[i]);
        ds4_kv_cache_free(&cache);
    }
    
    return 1;
}

int main(void) {
    printf("Running ds4_kv_cache paged tests...\n\n");
    
    TEST(kv_init_basic);
    TEST(kv_init_invalid_page_size);
    TEST(kv_init_null_cache);
    TEST(kv_grow);
    TEST(kv_get_page);
    TEST(kv_commit);
    TEST(kv_commit_null_data);
    TEST(kv_stats);
    TEST(kv_page_sizes);
    
    printf("\nResults: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
