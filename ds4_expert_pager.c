#include "ds4_expert_pager.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <poll.h>
#endif

/* ---------------------------------------------------------------------------
 * JSON parser (minimal subset for index file)
 * --------------------------------------------------------------------------- */

static char *read_file_to_string(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ds4_expert_pager: failed to open %s: %s\n",
                path, strerror(errno));
        return NULL;
    }

    struct stat st;
    if (fstat(fileno(f), &st) != 0) {
        fclose(f);
        return NULL;
    }

    char *buf = malloc(st.st_size + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }

    size_t n = fread(buf, 1, st.st_size, f);
    buf[n] = '\0';
    fclose(f);

    return buf;
}

/* Simple JSON string extractor — finds "key": "value" pattern */
static const char *json_get_string(const char *json, const char *key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\": \"", key);

    const char *pos = strstr(json, pattern);
    if (!pos) return NULL;

    pos += strlen(pattern);
    const char *end = strchr(pos, '"');
    if (!end) return NULL;

    size_t len = end - pos;
    char *value = malloc(len + 1);
    if (!value) return NULL;

    memcpy(value, pos, len);
    value[len] = '\0';

    /* Return static buffer — caller must copy if needed */
    static char result[256];
    if (len >= sizeof(result)) len = sizeof(result) - 1;
    memcpy(result, value, len);
    result[len] = '\0';
    free(value);

    return result;
}

/* Parse numeric field from JSON — flat key at top level */
static uint64_t json_get_uint(const char *json, const char *key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\": ", key);

    const char *pos = strstr(json, pattern);
    if (!pos) return 0;

    pos += strlen(pattern);
    return strtoull(pos, NULL, 10);
}

/* Parse nested numeric field from JSON — look inside "key": { "subkey": value } */
static uint64_t json_get_nested_uint(const char *json, const char *parent, const char *child) {
    char pattern[512];
    snprintf(pattern, sizeof(pattern), "\"%s\": {\"%s\": ", parent, child);

    const char *pos = strstr(json, pattern);
    if (!pos) return 0;

    pos += strlen(pattern);
    return strtoull(pos, NULL, 10);
}

/* ---------------------------------------------------------------------------
 * Bundle loading
 * --------------------------------------------------------------------------- */

static bool ds4_expert_pager_load_bundles(ds4_expert_pager *pager,
                                          const char *json_str) {
    /* Parse bundles array — simplified extraction */
    const char *arr_start = strstr(json_str, "\"bundles\": [");
    if (!arr_start) {
        fprintf(stderr, "ds4_expert_pager: bundles array not found in index\n");
        return false;
    }

    arr_start += strlen("\"bundles\": [");

    /* Count bundles by counting "{" occurrences */
    uint64_t count = 0;
    const char *p = arr_start;
    while (*p) {
        if (*p == '{') count++;
        p++;
    }

    if (count == 0 || count > DS4_EXPERT_PAGER_MAX_BUNDLES) {
        fprintf(stderr, "ds4_expert_pager: invalid bundle count: %llu\n",
                (unsigned long long)count);
        return false;
    }

    pager->bundles = calloc(count, sizeof(ds4_expert_bundle));
    if (!pager->bundles) {
        fprintf(stderr, "ds4_expert_pager: failed to allocate bundles\n");
        return false;
    }

    pager->bundles_capacity = count;
    pager->bundles_count = 0;

    /* Parse each bundle — look for "layer", "expert", "offset", "size" */
    p = arr_start;
    while (*p && pager->bundles_count < count) {
        if (*p == '{') {
            ds4_expert_bundle *bundle = &pager->bundles[pager->bundles_count];

            /* Extract layer */
            const char *layer_str = strstr(p, "\"layer\": ");
            if (layer_str) {
                bundle->layer = (uint32_t)strtoul(layer_str + 9, NULL, 10);
            }

            /* Extract expert */
            const char *expert_str = strstr(p, "\"expert\": ");
            if (expert_str) {
                bundle->expert = (uint32_t)strtoul(expert_str + 10, NULL, 10);
            }

            /* Extract offset */
            const char *offset_str = strstr(p, "\"offset\": ");
            if (offset_str) {
                bundle->offset = strtoull(offset_str + 11, NULL, 10);
            }

            /* Extract size */
            const char *size_str = strstr(p, "\"size\": ");
            if (size_str) {
                bundle->size = strtoull(size_str + 9, NULL, 10);
            }

            /* Extract type */
            const char *type_str = strstr(p, "\"type\": ");
            if (type_str) {
                bundle->type = (uint32_t)strtoul(type_str + 9, NULL, 10);
            }

            /* Extract tensor */
            const char *tensor_str = strstr(p, "\"tensor\": \"");
            if (tensor_str) {
                tensor_str += 10;
                if (strncmp(tensor_str, "gate", 4) == 0) {
                    bundle->tensor_idx = 0;
                } else if (strncmp(tensor_str, "up", 2) == 0) {
                    bundle->tensor_idx = 1;
                } else if (strncmp(tensor_str, "down", 4) == 0) {
                    bundle->tensor_idx = 2;
                }
            }

            pager->bundles_count++;
        }
        p++;
    }

    return true;
}

/* ---------------------------------------------------------------------------
 * Public API
 * --------------------------------------------------------------------------- */

bool ds4_expert_pager_open(ds4_expert_pager *pager,
                           const char *bin_path,
                           const char *index_path) {
    memset(pager, 0, sizeof(*pager));

    /* Open bundle file */
    pager->bin_fd = open(bin_path, O_RDONLY);
    if (pager->bin_fd < 0) {
        fprintf(stderr, "ds4_expert_pager: failed to open %s: %s\n",
                bin_path, strerror(errno));
        return false;
    }

    /* Read and parse index JSON */
    char *json_str = read_file_to_string(index_path);
    if (!json_str) {
        close(pager->bin_fd);
        return false;
    }

    /* Parse header */
    pager->header.version = json_get_uint(json_str, "version");
    pager->header.layer_count = json_get_uint(json_str, "layer_count");
    pager->header.expert_count = json_get_uint(json_str, "expert_count");
    pager->header.page_size = json_get_uint(json_str, "page_size");
    /* bundle_bytes is nested: "bundle_bytes": {"gate": N, "up": N, "down": N} */
    pager->header.bundle_bytes_gate = json_get_nested_uint(json_str, "bundle_bytes", "gate");
    pager->header.bundle_bytes_up = json_get_nested_uint(json_str, "bundle_bytes", "up");
    pager->header.bundle_bytes_down = json_get_nested_uint(json_str, "bundle_bytes", "down");

    const char *sha = json_get_string(json_str, "sha256");
    if (sha) {
        strncpy(pager->header.sha256, sha, sizeof(pager->header.sha256) - 1);
    }

    const char *gguf_sha = json_get_string(json_str, "gguf_sha256");
    if (gguf_sha) {
        strncpy(pager->header.gguf_sha256, gguf_sha, sizeof(pager->header.gguf_sha256) - 1);
    }

    /* Load bundles */
    if (!ds4_expert_pager_load_bundles(pager, json_str)) {
        free(json_str);
        close(pager->bin_fd);
        return false;
    }

    free(json_str);

    fprintf(stderr, "ds4_expert_pager: opened %llu bundles from %s\n",
            (unsigned long long)pager->bundles_count, index_path);

    return true;
}

void ds4_expert_pager_close(ds4_expert_pager *pager) {
    if (pager->bin_fd >= 0) {
        close(pager->bin_fd);
        pager->bin_fd = -1;
    }

    if (pager->bundles) {
        free(pager->bundles);
        pager->bundles = NULL;
    }
}

int ds4_expert_pager_ensure(ds4_expert_pager *pager,
                            uint32_t layer,
                            const uint32_t *expert_ids,
                            uint32_t n_experts,
                            uint32_t tensor_idx,
                            void **out_pointers) {
    if (!pager || !expert_ids || n_experts == 0) {
        return -1;
    }

    int misses = 0;

    for (uint32_t i = 0; i < n_experts; i++) {
        uint32_t expert = expert_ids[i];

        /* Find bundle in index for this layer + expert + tensor type */
        ds4_expert_bundle *bundle = NULL;
        for (uint64_t j = 0; j < pager->bundles_count; j++) {
            if (pager->bundles[j].layer == layer &&
                pager->bundles[j].expert == expert &&
                pager->bundles[j].tensor_idx == tensor_idx) {
                bundle = &pager->bundles[j];
                break;
            }
        }

        if (!bundle) {
            fprintf(stderr, "ds4_expert_pager: bundle not found for layer=%u expert=%u tensor=%u\n",
                    layer, expert, tensor_idx);
            misses++;
            continue;
        }

        /* Pread bundle into memory */
        void *ptr = malloc(bundle->size);
        if (!ptr) {
            fprintf(stderr, "ds4_expert_pager: failed to allocate %llu bytes\n",
                    (unsigned long long)bundle->size);
            misses++;
            continue;
        }

        ssize_t nread = pread(pager->bin_fd, ptr, bundle->size, bundle->offset);
        if (nread != (ssize_t)bundle->size) {
            fprintf(stderr, "ds4_expert_pager: pread failed: %s\n",
                    strerror(errno));
            free(ptr);
            misses++;
            continue;
        }

        /* Update stats */
        pager->total_preads++;
        pager->total_pread_bytes += bundle->size;
        pager->total_misses++;

        if (out_pointers) {
            out_pointers[i] = ptr;
        }
    }

    return misses;
}

void ds4_expert_pager_get_stats(const ds4_expert_pager *pager,
                                uint64_t *out_misses,
                                uint64_t *out_hits,
                                double *out_hit_rate,
                                uint64_t *out_pread_bytes,
                                double *out_total_latency_ms) {
    if (out_misses) *out_misses = pager->total_misses;
    if (out_hits) *out_hits = pager->total_hits;
    if (out_pread_bytes) *out_pread_bytes = pager->total_pread_bytes;
    if (out_total_latency_ms) *out_total_latency_ms = pager->total_pread_latency_ms;

    uint64_t total = pager->total_misses + pager->total_hits;
    if (out_hit_rate && total > 0) {
        *out_hit_rate = (double)pager->total_hits / total;
    }
}

void ds4_expert_pager_report_stats(const ds4_expert_pager *pager) {
    uint64_t misses, hits, pread_bytes;
    double hit_rate, latency_ms;

    ds4_expert_pager_get_stats(pager, &misses, &hits, &hit_rate,
                               &pread_bytes, &latency_ms);

    uint64_t total = misses + hits;

    fprintf(stderr, "\n=== Expert Pager Stats ===\n");
    fprintf(stderr, "Total requests:  %llu\n", (unsigned long long)total);
    fprintf(stderr, "Hits:            %llu\n", (unsigned long long)hits);
    fprintf(stderr, "Misses:          %llu\n", (unsigned long long)misses);
    fprintf(stderr, "Hit rate:        %.2f%%\n", hit_rate * 100.0);
    fprintf(stderr, "SSD reads:       %llu bytes (%.2f MB)\n",
            (unsigned long long)pread_bytes,
            pread_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "Latency:         %.2f ms\n", latency_ms);
    fprintf(stderr, "==========================\n\n");
}
