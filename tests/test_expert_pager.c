#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>

#include "ds4_expert_pager.h"

/* ---------------------------------------------------------------------------
 * Test fixtures
 * --------------------------------------------------------------------------- */

/* Create a synthetic bundle file for testing */
static int create_test_bundle(const char *path) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("create_test_bundle: open");
        return -1;
    }

    /* Write 10 bundles of 1024 bytes each */
    uint8_t data[1024];
    memset(data, 0xAA, sizeof(data));

    for (int i = 0; i < 10; i++) {
        if (write(fd, data, sizeof(data)) != sizeof(data)) {
            perror("create_test_bundle: write");
            close(fd);
            return -1;
        }
    }

    close(fd);
    return 0;
}

/* Create a synthetic index JSON for testing */
static int create_test_index(const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) {
        perror("create_test_index: fopen");
        return -1;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"version\": 1,\n");
    fprintf(f, "  \"layer_count\": 2,\n");
    fprintf(f, "  \"expert_count\": 5,\n");
    fprintf(f, "  \"bundle_bytes\": {\n");
    fprintf(f, "    \"gate\": 1024,\n");
    fprintf(f, "    \"up\": 1024,\n");
    fprintf(f, "    \"down\": 1024\n");
    fprintf(f, "  },\n");
    fprintf(f, "  \"bundles\": [\n");

    for (int layer = 0; layer < 2; layer++) {
        for (int expert = 0; expert < 5; expert++) {
            uint64_t offset = (layer * 5 + expert) * 1024ULL;
            fprintf(f, "    {\n");
            fprintf(f, "      \"layer\": %d,\n", layer);
            fprintf(f, "      \"expert\": %d,\n", expert);
            fprintf(f, "      \"tensor\": \"gate\",\n");
            fprintf(f, "      \"offset\": %llu,\n", (unsigned long long)offset);
            fprintf(f, "      \"size\": 1024,\n");
            fprintf(f, "      \"type\": 16,\n");
            fprintf(f, "      \"sha256\": \"abcd1234\"\n");
            fprintf(f, "    }%s\n", (layer == 1 && expert == 4) ? "" : ",");
        }
    }

    fprintf(f, "  ],\n");
    fprintf(f, "  \"sha256\": \"test_sha256\"\n");
    fprintf(f, "}\n");

    fclose(f);
    return 0;
}

/* ---------------------------------------------------------------------------
 * Test cases
 * --------------------------------------------------------------------------- */

static void test_pager_open_close() {
    const char *bundle_path = "/tmp/test_expert_pager_bundle.bin";
    const char *index_path = "/tmp/test_expert_pager_index.json";

    /* Setup */
    assert(create_test_bundle(bundle_path) == 0);
    assert(create_test_index(index_path) == 0);

    /* Test open */
    ds4_expert_pager pager;
    assert(ds4_expert_pager_open(&pager, bundle_path, index_path) == true);
    assert(pager.bin_fd >= 0);
    assert(pager.bundles != NULL);
    assert(pager.bundles_count > 0);

    /* Test close */
    ds4_expert_pager_close(&pager);
    assert(pager.bin_fd < 0);
    assert(pager.bundles == NULL);

    /* Cleanup */
    unlink(bundle_path);
    unlink(index_path);

    printf("PASS: test_pager_open_close\n");
}

static void test_pager_ensure() {
    const char *bundle_path = "/tmp/test_expert_pager_bundle2.bin";
    const char *index_path = "/tmp/test_expert_pager_index2.json";

    /* Setup */
    assert(create_test_bundle(bundle_path) == 0);
    assert(create_test_index(index_path) == 0);

    /* Open pager */
    ds4_expert_pager pager;
    assert(ds4_expert_pager_open(&pager, bundle_path, index_path) == true);

    /* Test ensure — request valid experts */
    uint32_t experts[] = {0, 1, 2};
    void *pointers[3] = {NULL};

    int misses = ds4_expert_pager_ensure(&pager, 0, experts, 3, 0, pointers);
    assert(misses == 0);

    /* Verify pointers are valid */
    for (int i = 0; i < 3; i++) {
        assert(pointers[i] != NULL);
        uint8_t *data = (uint8_t *)pointers[i];
        assert(data[0] == 0xAA);  /* Our test pattern */
        free(pointers[i]);
    }

    /* Test ensure — request invalid expert */
    uint32_t invalid_expert = 999;
    misses = ds4_expert_pager_ensure(&pager, 0, &invalid_expert, 1, 0, NULL);
    assert(misses > 0);

    /* Cleanup */
    ds4_expert_pager_close(&pager);
    unlink(bundle_path);
    unlink(index_path);

    printf("PASS: test_pager_ensure\n");
}

static void test_pager_stats() {
    const char *bundle_path = "/tmp/test_expert_pager_bundle3.bin";
    const char *index_path = "/tmp/test_expert_pager_index3.json";

    /* Setup */
    assert(create_test_bundle(bundle_path) == 0);
    assert(create_test_index(index_path) == 0);

    /* Open pager */
    ds4_expert_pager pager;
    assert(ds4_expert_pager_open(&pager, bundle_path, index_path) == true);

    /* Do some operations */
    uint32_t experts[] = {0, 1};
    ds4_expert_pager_ensure(&pager, 0, experts, 2, 0, NULL);
    ds4_expert_pager_ensure(&pager, 1, experts, 2, 0, NULL);

    /* Get stats */
    uint64_t misses, hits, pread_bytes;
    double hit_rate, latency_ms;
    ds4_expert_pager_get_stats(&pager, &misses, &hits, &hit_rate,
                               &pread_bytes, &latency_ms);

    assert(misses == 4);  /* 4 successful reads */
    assert(pread_bytes > 0);
    assert(hit_rate >= 0.0 && hit_rate <= 1.0);

    /* Report stats (just verify it doesn't crash) */
    ds4_expert_pager_report_stats(&pager);

    /* Cleanup */
    ds4_expert_pager_close(&pager);
    unlink(bundle_path);
    unlink(index_path);

    printf("PASS: test_pager_stats\n");
}

static void test_pager_invalid_paths() {
    ds4_expert_pager pager;

    /* Test with non-existent files */
    assert(ds4_expert_pager_open(&pager, "/nonexistent/bundle.bin",
                                 "/nonexistent/index.json") == false);

    printf("PASS: test_pager_invalid_paths\n");
}

/* ---------------------------------------------------------------------------
 * Main
 * --------------------------------------------------------------------------- */

int main(int argc, char *argv[]) {
    printf("Running ds4_expert_pager tests...\n\n");

    test_pager_open_close();
    test_pager_ensure();
    test_pager_stats();
    test_pager_invalid_paths();

    printf("\nAll tests passed!\n");
    return 0;
}
