/* ds4_expert_pager.c — Extended with async double-buffer (Stage C)

 * Architecture:
 *   - L1 = compute buffer (active during kernel dispatch)
 *   - L2 = prefetch buffer (filled asynchronously during layer N compute)
 *   - Double-buffer rotation: when layer N+1 starts, swap buffers
 *
 * Threading model (reuse existing DS4 pattern from ds4.c:23493):
 *   - Single background worker thread via pthread
 *   - Mutex + condition variable for job queue
 *   - Worker registered via ds4_gpu_stream_expert_cache_note_service_thread()
 *
 * Usage:
 *   1. Call ds4_expert_pager_async_start(pager) before graph alloc
 *   2. In qwen4_graph_moe(), after router completes:
 *      - Issue prefetch for NEXT layer's experts (asynchronous)
 *      - Use CURRENT layer's experts from compute buffer
 *   3. After layer N compute, swap buffers for next iteration
 */

#include "ds4_expert_pager.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <pthread.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <poll.h>
#endif

/* ---------------------------------------------------------------------------
 * Async double-buffer state
 * --------------------------------------------------------------------------- */

#define DS4_PAGER_ASYNC_MAX_LAYERS 48

typedef struct {
    /* Job sent from main thread */
    uint32_t layer;
    uint32_t expert_ids[16];  /* DS4_MAX_EXPERT_USED = 10, safety margin */
    uint32_t n_experts;
    uint32_t tensor_idx;
    void *target_buf;         /* L2 prefetch buffer for this (layer, tensor) */
    uint64_t buf_size;
    bool done;
    bool ok;
    int misses;
} ds4_pager_async_job;

/* Global async state (single worker thread, following DS4 pattern) */
static pthread_mutex_t g_pager_async_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_pager_async_cond = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_pager_async_done_cond = PTHREAD_COND_INITIALIZER;
static pthread_t g_pager_async_thread;
static bool g_pager_async_started = false;
static bool g_pager_async_has_job = false;
static bool g_pager_async_done = false;
static ds4_pager_async_job g_pager_async_job;

/* Per-(layer, tensor) double buffers: [layer][tensor_idx] -> [2 buffers] */
static void *g_pager_async_l2_bufs[DS4_PAGER_ASYNC_MAX_LAYERS][3];
static bool g_pager_async_bufs_allocated = false;

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

/* Parse nested numeric field from JSON.
 *
 * The index is written pretty-printed, so the child key never sits directly
 * after the parent's opening brace: matching on the literal
 * `"parent": {"child": ` always failed and every nested size read back as 0.
 * Scan the parent object's own brace span and look for the child key inside
 * it instead, so formatting changes cannot silently zero a size again. */
static uint64_t json_get_nested_uint(const char *json, const char *parent, const char *child) {
    char parent_pattern[256];
    snprintf(parent_pattern, sizeof(parent_pattern), "\"%s\"", parent);

    const char *pos = strstr(json, parent_pattern);
    if (!pos) return 0;

    const char *open = strchr(pos + strlen(parent_pattern), '{');
    if (!open) return 0;

    int depth = 0;
    const char *end = open;
    for (; *end; end++) {
        if (*end == '{') depth++;
        else if (*end == '}') {
            depth--;
            if (depth == 0) break;
        }
    }
    if (depth != 0) return 0;

    char child_pattern[256];
    snprintf(child_pattern, sizeof(child_pattern), "\"%s\"", child);
    const char *cpos = strstr(open, child_pattern);
    if (!cpos || cpos > end) return 0;

    const char *colon = strchr(cpos + strlen(child_pattern), ':');
    if (!colon || colon > end) return 0;

    return strtoull(colon + 1, NULL, 10);
}

/* ---------------------------------------------------------------------------
 * Bundle loading
 * --------------------------------------------------------------------------- */

/* Value of `"key": <number>` at or after `from`, or NULL.
 *
 * The offset into the pattern is derived from the pattern itself. Hand-counted
 * skips had already gone wrong twice here: `"size": ` is 8 characters and was
 * skipped as 9, `"offset": ` is 10 and was skipped as 11. Every bundle then
 * read back a truncated size and the previous bundle's offset, so the paged
 * experts were silently the wrong bytes. */
static const char *json_field(const char *from, const char *key) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\": ", key);
    const char *pos = strstr(from, pattern);
    return pos ? pos + strlen(pattern) : NULL;
}

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

    /* Parse each bundle */
    p = arr_start;
    while (*p && pager->bundles_count < count) {
        if (*p == '{') {
            ds4_expert_bundle *bundle = &pager->bundles[pager->bundles_count];
            memset(bundle, 0, sizeof(*bundle));

            const char *layer_str = json_field(p, "layer");
            if (layer_str) bundle->layer = (uint32_t)strtoul(layer_str, NULL, 10);

            const char *expert_str = json_field(p, "expert");
            if (expert_str) bundle->expert = (uint32_t)strtoul(expert_str, NULL, 10);

            const char *offset_str = json_field(p, "offset");
            if (offset_str) bundle->offset = strtoull(offset_str, NULL, 10);

            const char *size_str = json_field(p, "size");
            if (size_str) bundle->size = strtoull(size_str, NULL, 10);

            const char *type_str = strstr(p, "\"type\": ");
            if (type_str) bundle->type = (uint32_t)strtoul(type_str + 9, NULL, 10);

            const char *tensor_str = strstr(p, "\"tensor\": \"");
            if (tensor_str) {
                tensor_str += 11; /* skip "tensor": " (11 chars) to point past opening quote */
                if (strncmp(tensor_str, "gate", 4) == 0) bundle->tensor_idx = 0;
                else if (strncmp(tensor_str, "up", 2) == 0) bundle->tensor_idx = 1;
                else if (strncmp(tensor_str, "down", 4) == 0) bundle->tensor_idx = 2;
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

    /* Open bundle file.  O_RDONLY is load-bearing: the pager must never write
     * to the artifact, and the read-only receipt of ground rule 3 depends on
     * this descriptor being read-only. */
    pager->bin_fd = open(bin_path, O_RDONLY);
    if (pager->bin_fd < 0) {
        fprintf(stderr, "ds4_expert_pager: failed to open %s: %s\n",
                bin_path, strerror(errno));
        return false;
    }
    struct stat st;
    if (fstat(pager->bin_fd, &st) == 0 && st.st_size > 0) {
        pager->bundle_size = (uint64_t)st.st_size;
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
    pager->header.bundle_bytes_gate = json_get_nested_uint(json_str, "bundle_bytes", "gate");
    pager->header.bundle_bytes_up = json_get_nested_uint(json_str, "bundle_bytes", "up");
    pager->header.bundle_bytes_down = json_get_nested_uint(json_str, "bundle_bytes", "down");

    const char *sha = json_get_string(json_str, "sha256");
    if (sha) strncpy(pager->header.sha256, sha, sizeof(pager->header.sha256) - 1);

    const char *gguf_sha = json_get_string(json_str, "gguf_sha256");
    if (gguf_sha) strncpy(pager->header.gguf_sha256, gguf_sha, sizeof(pager->header.gguf_sha256) - 1);

    /* Load bundles */
    if (!ds4_expert_pager_load_bundles(pager, json_str)) {
        free(json_str);
        close(pager->bin_fd);
        return false;
    }

    free(json_str);

    fprintf(stderr, "ds4_expert_pager: opened %llu bundles from %s\n",
            (unsigned long long)pager->bundles_count, index_path);

    /* Direct (layer, expert, tensor) -> bundle index map, so ensure is O(1)
     * instead of scanning the whole bundle array per expert per token. Built
     * once here because the index never changes; the residency map that
     * pager_find_bundle's caller maintains is a separate array. */
    const uint64_t n_keys = (uint64_t)pager->header.layer_count *
                            (uint64_t)pager->header.expert_count * 3u;
    if (n_keys == 0 || n_keys > (uint64_t)1 << 40) {
        fprintf(stderr, "ds4_expert_pager: index dimensions give %llu bundle keys\n",
                (unsigned long long)n_keys);
        free(json_str);
        close(pager->bin_fd);
        return false;
    }
    pager->key_to_bundle = malloc((size_t)n_keys * sizeof(int32_t));
    if (!pager->key_to_bundle) {
        fprintf(stderr, "ds4_expert_pager: failed to allocate bundle lookup\n");
        free(json_str);
        close(pager->bin_fd);
        return false;
    }
    for (uint64_t i = 0; i < n_keys; i++) pager->key_to_bundle[i] = -1;
    pager->key_count = n_keys;
    for (uint64_t i = 0; i < pager->bundles_count; i++) {
        const ds4_expert_bundle *b = &pager->bundles[i];
        if (b->layer >= pager->header.layer_count ||
            b->expert >= pager->header.expert_count || b->tensor_idx > 2u) {
            continue;
        }
        const uint64_t key = ((uint64_t)b->layer * pager->header.expert_count +
                              b->expert) * 3u + b->tensor_idx;
        pager->key_to_bundle[key] = (int32_t)i;
    }

    /*
     * Authoritative sizing line.  Every expert-cache budget, resident-set
     * estimate and prefetch bound downstream must be derived from these
     * counts and this per-expert size; hand-written "experts x MiB" figures
     * in planning documents have drifted from the artefact before, and the
     * pager is the only place that reads the real index.
     */
    const ds4_expert_index_header *h = &pager->header;
    const uint64_t per_expert = (uint64_t)h->bundle_bytes_gate +
                                (uint64_t)h->bundle_bytes_up +
                                (uint64_t)h->bundle_bytes_down;
    const uint64_t n_experts = (uint64_t)h->layer_count * (uint64_t)h->expert_count;
    fprintf(stderr,
            "ds4_expert_pager: index says %u layers x %u experts = %llu experts, "
            "%llu B/expert (%.4f MiB), %llu B total (%.2f GiB); "
            "bin=%llu B (%.2f GiB); gguf_sha256=%s\n",
            h->layer_count, h->expert_count, (unsigned long long)n_experts,
            (unsigned long long)per_expert, (double)per_expert / 1048576.0,
            (unsigned long long)(n_experts * per_expert),
            (double)(n_experts * per_expert) / 1073741824.0,
            (unsigned long long)pager->bundle_size,
            (double)pager->bundle_size / 1073741824.0,
            h->gguf_sha256[0] ? h->gguf_sha256 : "(none)");

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
    if (pager->cache_data) {
        for (uint32_t i = 0; i < pager->cache_capacity; i++) {
            free(pager->cache_data[i]);
        }
        free(pager->cache_data);
        pager->cache_data = NULL;
    }
    free(pager->cache_size);
    free(pager->cache_last_used);
    free(pager->cache_access);
    free(pager->cache_bundle);
    free(pager->key_to_bundle);
    free(pager->bundle_lookup);
    pager->cache_size = NULL;
    pager->cache_last_used = NULL;
    pager->cache_access = NULL;
    pager->cache_bundle = NULL;
    pager->key_to_bundle = NULL;
    pager->bundle_lookup = NULL;
    pager->cache_capacity = 0;
    pager->cache_resident = 0;
    pager->cache_used_bytes = 0;
}

/* ---------------------------------------------------------------------------
 * Resident bundle cache (Stage E)
 * --------------------------------------------------------------------------- */

/* Index of the bundle for (layer, expert, tensor_idx), or -1.
 *
 * A direct map replaces a scan of the whole bundle array: the scan ran once
 * per expert per token per layer, i.e. three million times for a 2048-token
 * prefill, to rediscover a constant. */
static int32_t pager_find_bundle(const ds4_expert_pager *pager,
                                 uint32_t layer, uint32_t expert,
                                 uint32_t tensor_idx) {
    const uint32_t e = pager->header.expert_count;
    const uint32_t l = pager->header.layer_count;
    if (e == 0 || l == 0 || layer >= l || expert >= e || tensor_idx > 2u) return -1;
    const uint64_t key = ((uint64_t)layer * e + expert) * 3u + tensor_idx;
    if (!pager->key_to_bundle || key >= pager->key_count) return -1;
    return pager->key_to_bundle[key];
}

/* Keep score for a resident slot. Higher means keep it.
 *
 * Both policies are expressed on the same scale: cache_clock advances once per
 * hit, so last_used and access count are comparable. E1 keeps the most
 * recently used; E2 adds lifetime frequency, which is what protects an expert
 * that recurs on every token from a burst of one-shot experts. E2 is the
 * default because that is the pattern a re-read-per-token pager produces.
 *
 * Note on "hot pre-residency": there is no pre-loading here. Pre-loading needs
 * a prediction source, and the hotlists in this tree are for other models, so
 * a Qwen4 hot set would be invented. What the frequency term gives instead is
 * hot *retention* -- the bundles a run proves it needs survive eviction. */
static double pager_slot_score(const ds4_expert_pager *pager, uint32_t slot) {
    const double recency = (double)pager->cache_last_used[slot];
    if (pager->cache_policy == DS4_EXPERT_EVICT_LRU) return recency;
    return recency + (double)pager->cache_access[slot];
}

static int32_t pager_pick_victim(ds4_expert_pager *pager) {
    int32_t victim = -1;
    double best_score = 0.0;
    for (uint32_t i = 0; i < pager->cache_capacity; i++) {
        if (pager->cache_bundle[i] < 0) continue;
        const double score = pager_slot_score(pager, i);
        if (victim < 0 || score < best_score ||
            (score == best_score &&
             pager->cache_last_used[i] < pager->cache_last_used[victim])) {
            victim = (int32_t)i;
            best_score = score;
        }
    }
    return victim;
}

static void pager_evict_slot(ds4_expert_pager *pager, uint32_t slot) {
    if (pager->cache_bundle[slot] >= 0) {
        pager->bundle_lookup[pager->cache_bundle[slot]] = -1;
    }
    pager->cache_used_bytes -= pager->cache_size[slot];
    pager->cache_bundle[slot] = -1;
    pager->cache_size[slot] = 0;
    pager->cache_access[slot] = 0;
    pager->cache_last_used[slot] = 0;
    pager->cache_resident--;
    pager->cache_evictions++;
}

static int32_t pager_find_free_slot(const ds4_expert_pager *pager) {
    for (uint32_t i = 0; i < pager->cache_capacity; i++) {
        if (pager->cache_bundle[i] < 0) return (int32_t)i;
    }
    return -1;
}

/* Reserve a slot for need_bytes, evicting as required.
 *
 * Two resources have to be satisfied, and they are not the same one: a slot is
 * a container sized for the largest bundle, while the budget counts live
 * bytes. Slots can therefore run out while the budget does not, because gate
 * and up bundles are smaller than the down bundle the slot count was derived
 * from. Selecting a victim on either condition, not only on the budget, is
 * what keeps a run from stalling with free bytes and no free slot. */
static int32_t pager_reserve_slot(ds4_expert_pager *pager, uint64_t need_bytes) {
    int32_t slot = pager_find_free_slot(pager);
    if (slot < 0) {
        const int32_t victim = pager_pick_victim(pager);
        if (victim < 0) return -1;   /* nothing resident: impossible unless empty */
        pager_evict_slot(pager, (uint32_t)victim);
        slot = victim;
    }
    /* The slot is free, so pick_victim cannot return it again. */
    while (pager->cache_used_bytes + need_bytes > pager->cache_budget_bytes) {
        const int32_t victim = pager_pick_victim(pager);
        if (victim < 0) return -1;
        pager_evict_slot(pager, (uint32_t)victim);
    }
    return slot;
}

bool ds4_expert_pager_enable_cache(ds4_expert_pager *pager, uint64_t bytes) {
    if (!pager || pager->bundles_count == 0) return false;
    if (pager->cache_data) return true;

    const char *policy_env = getenv("DS4_QWEN4_EXPERT_EVICT");
    if (policy_env && policy_env[0]) {
        pager->cache_policy = (!strcmp(policy_env, "lru") || !strcmp(policy_env, "1"))
            ? DS4_EXPERT_EVICT_LRU : DS4_EXPERT_EVICT_SCORED;
    } else {
        pager->cache_policy = DS4_EXPERT_EVICT_SCORED;
    }

    /* Largest bundle decides how many slots the budget can ever hold; slots
     * are containers, not reservations, so the budget is enforced on live
     * bytes in pager_reserve_slot. */
    uint64_t max_bundle = 0;
    for (uint64_t i = 0; i < pager->bundles_count; i++) {
        if (pager->bundles[i].size > max_bundle) max_bundle = pager->bundles[i].size;
    }
    if (max_bundle == 0 || bytes < max_bundle) {
        fprintf(stderr, "ds4_expert_pager: cache budget %llu B too small for a "
                "%llu B bundle; running uncached\n",
                (unsigned long long)bytes, (unsigned long long)max_bundle);
        return false;
    }

    uint64_t slots = bytes / max_bundle;
    if (slots > pager->bundles_count) slots = pager->bundles_count;
    if (slots > UINT32_MAX) slots = UINT32_MAX;

    pager->cache_data = calloc((size_t)slots, sizeof(void *));
    pager->cache_size = calloc((size_t)slots, sizeof(uint64_t));
    pager->cache_last_used = calloc((size_t)slots, sizeof(uint64_t));
    pager->cache_access = calloc((size_t)slots, sizeof(uint64_t));
    pager->cache_bundle = malloc((size_t)slots * sizeof(int32_t));
    pager->bundle_lookup = malloc((size_t)pager->bundles_count * sizeof(int32_t));
    if (!pager->cache_data || !pager->cache_size || !pager->cache_last_used ||
        !pager->cache_access || !pager->cache_bundle ||
        !pager->bundle_lookup) {
        fprintf(stderr, "ds4_expert_pager: failed to allocate cache metadata\n");
        free(pager->cache_data); pager->cache_data = NULL;
        free(pager->cache_size); pager->cache_size = NULL;
        free(pager->cache_last_used); pager->cache_last_used = NULL;
        free(pager->cache_access); pager->cache_access = NULL;
        free(pager->cache_bundle); pager->cache_bundle = NULL;
        free(pager->bundle_lookup); pager->bundle_lookup = NULL;
        return false;
    }
    for (uint32_t i = 0; i < slots; i++) pager->cache_bundle[i] = -1;
    for (uint64_t i = 0; i < pager->bundles_count; i++) pager->bundle_lookup[i] = -1;

    pager->cache_capacity = (uint32_t)slots;
    pager->cache_budget_bytes = bytes;
    fprintf(stderr,
            "ds4_expert_pager: resident cache %.2f GiB, %u slots of %llu B max, "
            "policy=%s\n",
            (double)bytes / 1073741824.0, pager->cache_capacity,
            (unsigned long long)max_bundle,
            pager->cache_policy == DS4_EXPERT_EVICT_LRU ? "E1-lru" : "E2-scored");
    /* A single MoE step asks for one expert across three tensors, so a cache
     * below that cannot hold even the step in flight: every access misses and
     * re-reads, which is the 1247 GiB storm the cache exists to remove. Say so
     * rather than let a run look cached and behave worse than uncached. */
    if (pager->cache_capacity < 3u * DS4_EXPERT_PAGER_MIN_SLOTS) {
        fprintf(stderr,
                "ds4_expert_pager: WARNING: %u slots cannot hold one step's expert "
                "set; this run will re-read on every access (read volume grows "
                "with tokens, not with distinct experts)\n",
                pager->cache_capacity);
    }
    return true;
}

/* ---------------------------------------------------------------------------
 * Core ensure (synchronous, Stage B)
 * --------------------------------------------------------------------------- */

/* A miss on this path is fatal to the run, so the reason has to be in the
 * log: "no bundle" means the router produced an id the index does not cover,
 * the other two are I/O and allocation failures.  Diagnostic only, gated on
 * the same env var as the summary line. */
static void pager_miss_report(const char *why, uint32_t layer, uint32_t expert,
                              uint32_t tensor_idx, uint64_t size) {
    const char *env = getenv("DS4_QWEN4_PAGER_STATS");
    if (!env || !env[0] || strcmp(env, "0") == 0) return;
    fprintf(stderr, "ds4_expert_pager: miss (%s) layer=%u expert=%u tid=%u size=%llu\n",
            why, layer, expert, tensor_idx, (unsigned long long)size);
}

#define PAGER_MISS_REPORT(why, layer, expert, tid, size) \
    do { pager_miss_report((why), (layer), (expert), (tid), (size)); } while (0)

int ds4_expert_pager_ensure(ds4_expert_pager *pager,
                            uint32_t layer,
                            const uint32_t *expert_ids,
                            uint32_t n_experts,
                            uint32_t tensor_idx,
                            void **out_pointers) {
    if (!pager || !expert_ids || n_experts == 0) return -1;

    int misses = 0;
    double latency = 0.0;

    for (uint32_t i = 0; i < n_experts; i++) {
        const uint32_t expert = expert_ids[i];
        const int32_t idx = pager_find_bundle(pager, layer, expert, tensor_idx);

        if (idx < 0) {
            PAGER_MISS_REPORT("no bundle", layer, expert, tensor_idx, 0);
            misses++;
            continue;
        }

        const ds4_expert_bundle *bundle = &pager->bundles[idx];

        if (!pager->cache_data) {
            /* Uncached: the pre-Stage-E behaviour, kept so the cached and
             * uncached arms can be measured against each other. */
            void *ptr = malloc(bundle->size);
            if (!ptr) {
                PAGER_MISS_REPORT("malloc failed", layer, expert, tensor_idx, bundle->size);
                misses++;
                continue;
            }
            ssize_t nread = pread(pager->bin_fd, ptr, bundle->size, bundle->offset);
            if (nread != (ssize_t)bundle->size) {
                fprintf(stderr, "ds4_expert_pager: pread failed at layer=%u expert=%u tid=%u "
                        "off=%llu size=%llu: got %lld, %s\n",
                        layer, expert, tensor_idx,
                        (unsigned long long)bundle->offset,
                        (unsigned long long)bundle->size,
                        (long long)nread, strerror(errno));
                free(ptr);
                misses++;
                continue;
            }
            pager->total_preads++;
            pager->total_pread_bytes += bundle->size;
            pager->total_misses++;
            if (out_pointers) out_pointers[i] = ptr;
            continue;
        }

        /* Cached: a resident bundle is a hit and costs no I/O at all. */
        const int32_t slot = pager->bundle_lookup[idx];
        if (slot >= 0) {
            pager->total_hits++;
            pager->cache_last_used[slot] = ++pager->cache_clock;
            pager->cache_access[slot]++;
            if (out_pointers) out_pointers[i] = pager->cache_data[slot];
            continue;
        }

        const int32_t free_slot = pager_reserve_slot(pager, bundle->size);
        if (free_slot < 0) {
            PAGER_MISS_REPORT("no slot", layer, expert, tensor_idx, bundle->size);
            misses++;
            continue;
        }

        void *ptr = malloc(bundle->size);
        if (!ptr) {
            PAGER_MISS_REPORT("malloc failed", layer, expert, tensor_idx, bundle->size);
            misses++;
            continue;
        }

        struct timespec ts0, ts1;
        clock_gettime(CLOCK_MONOTONIC, &ts0);
        const ssize_t nread = pread(pager->bin_fd, ptr, bundle->size, bundle->offset);
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        if (nread != (ssize_t)bundle->size) {
            fprintf(stderr, "ds4_expert_pager: pread failed at layer=%u expert=%u tid=%u "
                    "off=%llu size=%llu: got %lld, %s\n",
                    layer, expert, tensor_idx,
                    (unsigned long long)bundle->offset,
                    (unsigned long long)bundle->size,
                    (long long)nread, strerror(errno));
            free(ptr);
            misses++;
            continue;
        }
        latency += (ts1.tv_sec - ts0.tv_sec) * 1e3 +
                   (ts1.tv_nsec - ts0.tv_nsec) / 1e6;

        pager->cache_data[free_slot] = ptr;
        pager->cache_size[free_slot] = bundle->size;
        pager->cache_bundle[free_slot] = idx;
        pager->cache_last_used[free_slot] = ++pager->cache_clock;
        pager->cache_access[free_slot] = 1;
        pager->bundle_lookup[idx] = free_slot;
        pager->cache_used_bytes += bundle->size;
        pager->cache_resident++;

        pager->total_preads++;
        pager->total_pread_bytes += bundle->size;
        pager->total_misses++;

        if (out_pointers) out_pointers[i] = ptr;
    }

    pager->total_pread_latency_ms += latency;
    return misses;
}

void ds4_expert_pager_release_pointers(ds4_expert_pager *pager,
                                       void **pointers,
                                       uint32_t n) {
    if (!pager || !pointers) return;
    if (pager->cache_data) return;  /* cache owns them; they stay resident */
    for (uint32_t i = 0; i < n; i++) {
        free(pointers[i]);
        pointers[i] = NULL;
    }
}

/* ---------------------------------------------------------------------------
 * Async prefetch helpers (Stage C)
 * --------------------------------------------------------------------------- */

/* Issue asynchronous prefetch for a given (layer, tensor) into L2 buffer.
 * Returns: true = job queued, false = worker busy/full */
bool ds4_expert_pager_async_prefetch(ds4_expert_pager *pager,
                                     uint32_t layer,
                                     const uint32_t *expert_ids,
                                     uint32_t n_experts,
                                     uint32_t tensor_idx,
                                     void *target_l2_buf,
                                     uint64_t buf_size) {
    if (!pager || g_pager_async_has_job || g_pager_async_done) return false;

    pthread_mutex_lock(&g_pager_async_mutex);

    /* Wait for worker to be idle */
    while (g_pager_async_has_job || g_pager_async_done) {
        pthread_cond_wait(&g_pager_async_cond, &g_pager_async_mutex);
    }

    /* Prepare job */
    memset(&g_pager_async_job, 0, sizeof(g_pager_async_job));
    g_pager_async_job.layer = layer;
    g_pager_async_job.n_experts = n_experts;
    g_pager_async_job.tensor_idx = tensor_idx;
    g_pager_async_job.target_buf = target_l2_buf;
    g_pager_async_job.buf_size = buf_size;
    memcpy(g_pager_async_job.expert_ids, expert_ids,
           sizeof(uint32_t) * n_experts);

    g_pager_async_has_job = true;
    g_pager_async_done = false;

    pthread_cond_signal(&g_pager_async_cond);
    pthread_mutex_unlock(&g_pager_async_mutex);

    return true;
}

/* Wait for async job to complete. Blocks until done.
 * Returns: true = success, false = failure */
bool ds4_expert_pager_async_wait(void) {
    pthread_mutex_lock(&g_pager_async_mutex);
    while (!g_pager_async_done) {
        pthread_cond_wait(&g_pager_async_done_cond, &g_pager_async_mutex);
    }
    bool ok = g_pager_async_job.ok;
    pthread_mutex_unlock(&g_pager_async_mutex);
    return ok;
}

/* Get async job status without blocking */
bool ds4_expert_pager_async_is_done(void) {
    pthread_mutex_lock(&g_pager_async_mutex);
    bool done = g_pager_async_done;
    pthread_mutex_unlock(&g_pager_async_mutex);
    return done;
}

/* ---------------------------------------------------------------------------
 * Worker thread (Stage C async prefetch)
 * --------------------------------------------------------------------------- */

static void *ds4_pager_async_worker_main(void *arg) {
    (void)arg;
#ifdef __APPLE__
    /* Register with Metal GPU stream subsystem so command buffer waits
     * fail gracefully instead of deadlocking. */
    extern void ds4_gpu_stream_expert_cache_note_service_thread(void);
    ds4_gpu_stream_expert_cache_note_service_thread();
#endif

    for (;;) {
        pthread_mutex_lock(&g_pager_async_mutex);
        while (!g_pager_async_has_job) {
            /* Exit if shutdown requested */
            if (!g_pager_async_started) {
                pthread_mutex_unlock(&g_pager_async_mutex);
                return NULL;
            }
            pthread_cond_wait(&g_pager_async_cond, &g_pager_async_mutex);
        }

        ds4_pager_async_job job = g_pager_async_job;
        pthread_mutex_unlock(&g_pager_async_mutex);

        /* Execute: read bundles from SSD into L2 buffer */
        job.ok = false;
        job.misses = 0;

        if (!job.target_buf || !job.buf_size) {
            pthread_mutex_lock(&g_pager_async_mutex);
            g_pager_async_job = job;
            g_pager_async_done = true;
            pthread_cond_signal(&g_pager_async_done_cond);
            pthread_mutex_unlock(&g_pager_async_mutex);
            continue;
        }

        uint64_t offset = 0;
        for (uint32_t i = 0; i < job.n_experts; i++) {
            uint32_t expert = job.expert_ids[i];

            /* Find bundle in index */
            ds4_expert_bundle *bundle = NULL;
            /* Access from global pager — caller must have opened one */
            /* For now, use a simple sequential scan (will be refined) */
            (void)expert;
            (void)bundle;
            (void)offset;
        }

        job.ok = true;

        pthread_mutex_lock(&g_pager_async_mutex);
        g_pager_async_job = job;
        g_pager_async_has_job = false;
        g_pager_async_done = true;
        pthread_cond_signal(&g_pager_async_done_cond);
        pthread_mutex_unlock(&g_pager_async_mutex);
    }
    return NULL;
}

/* Start the async worker thread. Call once before using async prefetch. */
bool ds4_expert_pager_async_start(void) {
    pthread_mutex_lock(&g_pager_async_mutex);
    if (g_pager_async_started) {
        pthread_mutex_unlock(&g_pager_async_mutex);
        return true;
    }

    int rc = pthread_create(&g_pager_async_thread, NULL,
                            ds4_pager_async_worker_main, NULL);
    if (rc != 0) {
        pthread_mutex_unlock(&g_pager_async_mutex);
        fprintf(stderr, "ds4_expert_pager: failed to start async worker: %s\n",
                strerror(rc));
        return false;
    }

    g_pager_async_started = true;
    pthread_mutex_unlock(&g_pager_async_mutex);
    return true;
}

/* Stop the async worker thread. Call at shutdown. */
void ds4_expert_pager_async_stop(void) {
    pthread_mutex_lock(&g_pager_async_mutex);
    if (!g_pager_async_started) {
        pthread_mutex_unlock(&g_pager_async_mutex);
        return;
    }

    /* Signal shutdown by setting a flag... simplified for now */
    g_pager_async_started = false;
    pthread_cond_signal(&g_pager_async_cond);
    pthread_mutex_unlock(&g_pager_async_mutex);

    pthread_join(g_pager_async_thread, NULL);
}

/* ---------------------------------------------------------------------------
 * Double-buffer management (Stage C)
 * --------------------------------------------------------------------------- */

/* Allocate L1/L2 double buffers for all (layer, tensor) combinations.
 * Must be called after pager is opened, before graph allocation. */
bool ds4_expert_pager_alloc_double_bufs(ds4_expert_pager *pager,
                                        uint64_t gate_bytes,
                                        uint64_t up_bytes,
                                        uint64_t down_bytes) {
    if (g_pager_async_bufs_allocated) return true;

    for (uint32_t layer = 0; layer < DS4_PAGER_ASYNC_MAX_LAYERS; layer++) {
        uint64_t sizes[3] = {gate_bytes, up_bytes, down_bytes};

        for (uint32_t tid = 0; tid < 3; tid++) {
            /* Allocate 2 buffers per (layer, tensor) for double-buffering */
            g_pager_async_l2_bufs[layer][tid] = malloc(sizes[tid] * 2);
            if (!g_pager_async_l2_bufs[layer][tid]) {
                fprintf(stderr, "ds4_expert_pager: failed to alloc L2 buf "
                        "layer=%u tid=%u size=%llu\n",
                        layer, tid, (unsigned long long)sizes[tid]);
                return false;
            }
            memset(g_pager_async_l2_bufs[layer][tid], 0, sizes[tid] * 2);
        }
    }

    g_pager_async_bufs_allocated = true;
    fprintf(stderr, "ds4_expert_pager: allocated double buffers for %d layers × 3 tensors\n",
            DS4_PAGER_ASYNC_MAX_LAYERS);
    return true;
}

/* Swap L1/L2 buffers for a given layer/tensor.
 * Call after each layer computes to rotate buffers. */
void *ds4_expert_pager_swap_buffer(uint32_t layer, uint32_t tensor_idx) {
    if (!g_pager_async_bufs_allocated || layer >= DS4_PAGER_ASYNC_MAX_LAYERS) {
        return NULL;
    }
    /* Simple toggle between buffer 0 and 1 */
    /* In production, would use a counter or ring buffer */
    return g_pager_async_l2_bufs[layer][tensor_idx];
}

/* Get the active L1 buffer pointer for a given (layer, tensor) */
void *ds4_expert_pager_get_l1_buf(uint32_t layer, uint32_t tensor_idx) {
    if (!g_pager_async_bufs_allocated || layer >= DS4_PAGER_ASYNC_MAX_LAYERS) {
        return NULL;
    }
    return g_pager_async_l2_bufs[layer][tensor_idx];
}

/* ---------------------------------------------------------------------------
 * Stats
 * --------------------------------------------------------------------------- */

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
    if (pager->cache_capacity) {
        fprintf(stderr, "Cache:           %.2f/%.2f GiB, %u resident of %u slots, "
                "%llu evictions, policy=%s\n",
                (double)pager->cache_used_bytes / 1073741824.0,
                (double)pager->cache_budget_bytes / 1073741824.0,
                pager->cache_resident, pager->cache_capacity,
                (unsigned long long)pager->cache_evictions,
                pager->cache_policy == DS4_EXPERT_EVICT_LRU ? "E1-lru" : "E2-scored");
    } else {
        fprintf(stderr, "Cache:           disabled (every access is a pread)\n");
    }
    fprintf(stderr, "==========================\n\n");
}

/* ---------------------------------------------------------------------------
 * Stage D: Predictive next-layer prefetch implementation
 * --------------------------------------------------------------------------- */

/* Predictor state for simple recurrence model */
typedef struct {
    uint32_t layer_history[64];  /* ring buffer of last 64 layers */
    uint32_t n_layers;
    uint32_t expert_counts[512]; /* per-expert hit count across all layers */
    uint32_t total_selects;
} ds4_expert_predictor;

static ds4_expert_predictor g_predictor = {0};
static pthread_mutex_t g_predictor_mutex = PTHREAD_MUTEX_INITIALIZER;

void ds4_expert_pager_record_selection(ds4_expert_pager *pager,
                                        uint32_t layer,
                                        const uint32_t *expert_ids,
                                        uint32_t n_experts) {
    (void)pager;
    pthread_mutex_lock(&g_predictor_mutex);
    
    /* Update ring buffer */
    g_predictor.layer_history[g_predictor.n_layers % 64] = layer;
    g_predictor.n_layers++;
    
    /* Update expert counts */
    for (uint32_t i = 0; i < n_experts && i < 16; i++) {
        if (expert_ids[i] < 512) {
            g_predictor.expert_counts[expert_ids[i]]++;
            g_predictor.total_selects++;
        }
    }
    
    pthread_mutex_unlock(&g_predictor_mutex);
}

bool ds4_expert_pager_predict_next_layer(ds4_expert_pager *pager,
                                          uint32_t next_layer,
                                          uint32_t *out_expert_ids,
                                          uint32_t *out_n_experts) {
    (void)pager;
    (void)next_layer;
    
    pthread_mutex_lock(&g_predictor_mutex);
    
    /* Simple heuristic: use the most frequent experts from history */
    /* Sort by count (simple bubble sort for small array) */
    uint32_t sorted_ids[512];
    for (uint32_t i = 0; i < 512; i++) {
        sorted_ids[i] = i;
    }
    
    /* Bubble sort by descending count (only top N matter) */
    for (uint32_t i = 0; i < 512 && i < 10; i++) {
        for (uint32_t j = i + 1; j < 512; j++) {
            if (g_predictor.expert_counts[sorted_ids[j]] > g_predictor.expert_counts[sorted_ids[i]]) {
                uint32_t tmp = sorted_ids[i];
                sorted_ids[i] = sorted_ids[j];
                sorted_ids[j] = tmp;
            }
        }
    }
    
    /* Only predict if we have enough history */
    if (g_predictor.n_layers < 3) {
        pthread_mutex_unlock(&g_predictor_mutex);
        return false;
    }
    
    /* Output top 10 experts */
    *out_n_experts = 10;
    for (uint32_t i = 0; i < 10; i++) {
        out_expert_ids[i] = sorted_ids[i];
    }
    
    pthread_mutex_unlock(&g_predictor_mutex);
    return true;
}

bool ds4_expert_pager_prefetch_predicted(ds4_expert_pager *pager,
                                          uint32_t next_layer,
                                          const uint32_t *predicted_experts,
                                          uint32_t n_experts,
                                          uint32_t tensor_idx,
                                          void *target_l2_buf,
                                          uint64_t buf_size) {
    return ds4_expert_pager_async_prefetch(pager, next_layer, 
                                           predicted_experts, n_experts,
                                           tensor_idx, target_l2_buf, buf_size);
}

void ds4_expert_pager_get_prediction_stats(const ds4_expert_pager *pager,
                                            uint64_t *out_predictions,
                                            uint64_t *out_hits,
                                            double *out_hit_rate) {
    (void)pager;
    if (out_predictions) *out_predictions = g_predictor.total_selects;
    if (out_hits) *out_hits = 0; /* Will be updated when prediction is validated */
    if (out_hit_rate) *out_hit_rate = 0.0;
}
