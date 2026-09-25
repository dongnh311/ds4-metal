/* Bookkeeping for the qwen4 prefill staging pipe: two whole-layer staging
 * buffers ("slots") and one background reader. Plain C so it can be tested
 * without Metal; ds4_metal.m owns the buffers, the reader thread, the command
 * buffers, and the lock held around every call here.
 *
 * A slot's job reads one streamed layer's gate/up/down tensors. A stage call
 * consumes a finished job (activate); the batch then encoded reads the slot,
 * and the first commit after that is the slot's last reader, which a later
 * read into the slot must wait for. */
#ifndef DS4_QWEN4_STAGE_PIPE_H
#define DS4_QWEN4_STAGE_PIPE_H

#include <stdbool.h>
#include <stdint.h>

enum { QSP_IDLE = 0, QSP_QUEUED, QSP_DONE, QSP_FAILED };

typedef struct {
    int state;
    uint64_t seq;           /* queue order */
    uint32_t layer;
    uint64_t off[3];        /* gate, up, down tensor offsets in the model */
    const void *map;        /* model map the offsets belong to */
    bool nocache;           /* read through the F_NOCACHE fd */
    bool reader_pending;    /* the open batch holds GEMMs reading this slot */
    bool has_reader;        /* a committed batch may still read this slot */
} qsp_slot;

typedef struct {
    qsp_slot slot[2];
    uint64_t seq;
} qsp_state;

/* The slot whose queued, finished or failed job holds this layer, or -1. */
static inline int qsp_find(const qsp_state *p, uint32_t layer, const uint64_t off[3], const void *map) {
    for (int i = 0; i < 2; i++) {
        const qsp_slot *s = &p->slot[i];
        if (s->state != QSP_IDLE && s->layer == layer && s->map == map &&
            s->off[0] == off[0] && s->off[1] == off[1] && s->off[2] == off[2]) {
            return i;
        }
    }
    return -1;
}

/* Queue a read of `layer` into slot i. */
static inline void qsp_queue(qsp_state *p, int i, uint32_t layer, const uint64_t off[3],
                             const void *map, bool nocache) {
    qsp_slot *s = &p->slot[i];
    s->state = QSP_QUEUED;
    s->seq = ++p->seq;
    s->layer = layer;
    s->off[0] = off[0];
    s->off[1] = off[1];
    s->off[2] = off[2];
    s->map = map;
    s->nocache = nocache;
}

/* The oldest queued slot, or -1: the reader serves jobs in queue order. */
static inline int qsp_next_queued(const qsp_state *p) {
    int best = -1;
    for (int i = 0; i < 2; i++) {
        if (p->slot[i].state == QSP_QUEUED &&
            (best < 0 || p->slot[i].seq < p->slot[best].seq)) {
            best = i;
        }
    }
    return best;
}

static inline bool qsp_any_queued(const qsp_state *p) {
    return qsp_next_queued(p) >= 0;
}

static inline void qsp_finish(qsp_state *p, int i, bool ok) {
    p->slot[i].state = ok ? QSP_DONE : QSP_FAILED;
}

/* Slot i is staged for the batch being encoded: its job is consumed and the
 * batch's next commit becomes its last reader.
 *
 * Hazard invariant: the first commit after this call is taken as the slot's
 * last reader (qsp_note_commit below). Nothing may commit between this call
 * and the last GEMM that reads the slot, unless that commit is followed by a
 * full wait for the GPU (which is what makes an earlier, in-between commit
 * safe to treat as the last reader too). A commit that races ahead of the
 * slot's real last-reading GEMM would let a later read into the slot start
 * before that GEMM has actually finished reading it. */
static inline void qsp_activate(qsp_state *p, int i) {
    p->slot[i].state = QSP_IDLE;
    p->slot[i].reader_pending = true;
}

/* A batch was committed. Returns the slots (bit i) that took it as their last
 * reader; a slot keeps the first commit after its activation (see the
 * hazard invariant documented on qsp_activate above: the caller must ensure
 * that first commit really does follow the slot's last reading GEMM). */
static inline unsigned qsp_note_commit(qsp_state *p) {
    unsigned mask = 0;
    for (int i = 0; i < 2; i++) {
        if (p->slot[i].reader_pending) {
            p->slot[i].reader_pending = false;
            p->slot[i].has_reader = true;
            mask |= 1u << i;
        }
    }
    return mask;
}

/* Everything committed so far has completed (the host waited): no readers. */
static inline void qsp_all_complete(qsp_state *p) {
    for (int i = 0; i < 2; i++) {
        p->slot[i].reader_pending = false;
        p->slot[i].has_reader = false;
    }
}

/* Drop slot i's job: its buffer is about to be overwritten or released. */
static inline void qsp_invalidate(qsp_state *p, int i) {
    p->slot[i].state = QSP_IDLE;
}

/* The prompt ended or was abandoned, and the caller waited for queued reads
 * and for the GPU: forget every job and reader. */
static inline void qsp_prompt_end(qsp_state *p) {
    qsp_all_complete(p);
    qsp_invalidate(p, 0);
    qsp_invalidate(p, 1);
}

#endif
