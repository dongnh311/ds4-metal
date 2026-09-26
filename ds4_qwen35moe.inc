/* Ornith-1.5-35B-A3B (llama.cpp qwen35moe) on Metal.
 *
 * Included by ds4.c right after the qwen4 Metal graph, so the family reuses
 * the qwen4 graph struct, GEMV and kernels.  GDN layers call
 * qwen4_graph_linear with the silu output gate (g->gdn_silu).  Attention is
 * the qwen4 gated attention without the QSA indexer: every position is
 * attended densely.  The MoE reuses the router and the Q4_K expert kernels,
 * with the qwen35 row kernels for Q5_K layers.  The residual stream is one
 * n_embd row per token, with no hyper-connections and no PLE layer.  The
 * llama.cpp converter already folded +1 into the norm weights, stores
 * ssm_a = -exp(A_log) and tiles the GDN value heads, which is what the qwen4
 * kernels expect.  The prefill chunk rule (qwen35_prefill_chunk_tokens)
 * lives in ds4.c next to the memory estimate that shares it.  The reused
 * qwen4 helpers keep their DS4_QWEN4_* kernel A/B switches (dense GEMV,
 * decode fusions, attention split/tiled/merge), which pick between
 * arithmetic-equivalent paths; Ornith's own knobs use DS4_QWEN35_*.
 *
 * MTP (graphs allocated with mtp): the MTP block blk.40 follows llama.cpp's
 * draft-mtp.  Its K/V row p comes from token p and h_{p-1}, the trunk's
 * post-output_norm hidden (h_{-1} = 0), so every forward also writes its
 * post-norm rows to mtp_h and the next forward carries the last one over.
 * Catch-up rows need only their K/V; the draft row continues through
 * attention, the MoE, shared_head_norm and the output head.  A speculative
 * verify snapshots the GDN state after its first row and runs its
 * attention one row per dispatch, so each row equals plain decoding bit for
 * bit.  The session cycle is ds4_session_qwen35_spec_cycle in ds4.c. */

static bool qwen35_graph_alloc(ds4_qwen4_gpu_graph *g, uint32_t ctx_cap, uint32_t cap_tokens, bool mtp) {
    memset(g, 0, sizeof(*g));
    if ((uint64_t)ctx_cap > DS4_ROPE_ORIG_CTX) {
        fprintf(stderr, "ds4: Ornith supports up to %llu tokens of context (no YaRN)\n",
                (unsigned long long)DS4_ROPE_ORIG_CTX);
        return false;
    }
    const uint64_t E = DS4_N_EMBD, T = cap_tokens;
    const uint64_t conv_dim = DS4_N_LIN_CONV_DIM;
    const uint64_t v_dim = (uint64_t)DS4_N_LIN_V_HEAD * DS4_N_LIN_HEAD_DIM;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t kv_dim = (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    g->ctx_cap = ctx_cap;
    g->alloc_cap = ctx_cap;
    g->cap_tokens = cap_tokens;
    /* a verify reads both of its rows' logits */
    g->n_logit_rows = mtp ? 2u : 1u;
    g->owns_scratch = true;
    g->gdn_silu = true;
    bool ok = true;
#define QWEN35_ALLOC(field_, n_) do { g->field_ = qwen4_graph_alloc_f32(n_); ok = ok && g->field_; } while (0)
    QWEN35_ALLOC(R, T * E);
    QWEN35_ALLOC(mixed, T * E);
    QWEN35_ALLOC(blk, T * E);
    QWEN35_ALLOC(qkv, T * conv_dim);
    QWEN35_ALLOC(z, T * v_dim);
    QWEN35_ALLOC(ga, T * DS4_N_LIN_V_HEAD);
    QWEN35_ALLOC(gb, T * DS4_N_LIN_V_HEAD);
    QWEN35_ALLOC(lin_o, T * v_dim);
    QWEN35_ALLOC(qg, T * 2u * q_dim);
    QWEN35_ALLOC(kp, T * kv_dim);
    QWEN35_ALLOC(vp, T * kv_dim);
    QWEN35_ALLOC(q, T * q_dim);
    QWEN35_ALLOC(gate, T * q_dim);
    QWEN35_ALLOC(attn_o, T * q_dim);
    QWEN35_ALLOC(attn_part, ds4_gpu_qwen4_attn_part_floats(3u, DS4_N_HEAD, DS4_N_HEAD_DIM));
    QWEN35_ALLOC(router, T * DS4_N_EXPERT);
    QWEN35_ALLOC(selected, T * DS4_N_EXPERT_USED);
    QWEN35_ALLOC(weights, T * DS4_N_EXPERT_USED);
    QWEN35_ALLOC(mid, T * (DS4_N_EXPERT_USED + 1u) * DS4_N_FF_EXP);
    QWEN35_ALLOC(part, T * (DS4_N_EXPERT_USED + 1u) * E);
    QWEN35_ALLOC(sh_gate_logit, T);
    QWEN35_ALLOC(moe_lists, (uint64_t)DS4_N_EXPERT * T);
    QWEN35_ALLOC(moe_counts, DS4_N_EXPERT);
    QWEN35_ALLOC(sh_gate, T * DS4_N_FF_EXP);
    QWEN35_ALLOC(sh_up, T * DS4_N_FF_EXP);
    QWEN35_ALLOC(sh_mid, T * DS4_N_FF_EXP);
    QWEN35_ALLOC(sh_out, T * E);
    QWEN35_ALLOC(logits, (uint64_t)g->n_logit_rows * DS4_N_VOCAB);
    QWEN35_ALLOC(pos3, (uint64_t)ctx_cap * 4u);
    if (mtp) {
        QWEN35_ALLOC(mtp_h, (T + 1u) * E);
        QWEN35_ALLOC(mtp_e, T * E);
        QWEN35_ALLOC(mtp_cat, T * 2u * E);
        QWEN35_ALLOC(mtp_R, T * E);
        QWEN35_ALLOC(mtp_argmax, 1u);
        QWEN35_ALLOC(mtp_argmax_tmp, ((DS4_N_VOCAB + 4095u) / 4096u) * 2u);
    }
#undef QWEN35_ALLOC
    /* F16 K/V per full-attention layer (with MTP, the MTP block's too);
     * recurrent state and conv history per GDN layer, and with MTP the
     * copy a verify takes after its first row. */
    const uint64_t kv_bytes = (uint64_t)ctx_cap * kv_dim * 2u;
    const uint64_t state_n = v_dim * DS4_N_LIN_HEAD_DIM;
    const uint64_t hist_n = (uint64_t)(DS4_N_LIN_CONV - 1u) * conv_dim;
    const uint32_t n_layers = mtp ? DS4_N_LAYER : DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
    for (uint32_t il = 0; il < n_layers && ok; il++) {
        if (ds4_qwen35_layer_is_attention(il)) {
            g->layer_k_cache[il] = ds4_gpu_tensor_alloc(kv_bytes);
            g->layer_v_cache[il] = ds4_gpu_tensor_alloc(kv_bytes);
            ok = g->layer_k_cache[il] && g->layer_v_cache[il];
        } else {
            g->layer_lin_state[il] = qwen4_graph_alloc_f32(state_n);
            g->layer_lin_hist[il] = qwen4_graph_alloc_f32(hist_n);
            ok = g->layer_lin_state[il] && g->layer_lin_hist[il];
            if (ok && mtp) {
                g->snap_lin_state[il] = qwen4_graph_alloc_f32(state_n);
                g->snap_lin_hist[il] = qwen4_graph_alloc_f32(hist_n);
                ok = g->snap_lin_state[il] && g->snap_lin_hist[il];
            }
        }
    }
    g->host_row = xmalloc(T * E * sizeof(float));
    g->host_pos3 = xmalloc(T * 4u * sizeof(uint32_t));
    g->host_logits = xmalloc((uint64_t)DS4_N_VOCAB * sizeof(float));
    if (!ok) {
        fprintf(stderr, "ds4: Ornith graph allocation failed (ctx %u)\n", ctx_cap);
        qwen4_graph_free(g);
        return false;
    }
    return true;
}

static void qwen35_graph_reset(ds4_qwen4_gpu_graph *g) {
    (void)ds4_gpu_synchronize();
    const uint64_t conv_dim = DS4_N_LIN_CONV_DIM;
    const uint64_t v_dim = (uint64_t)DS4_N_LIN_V_HEAD * DS4_N_LIN_HEAD_DIM;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (g->layer_lin_state[il]) ds4_gpu_tensor_fill_f32(g->layer_lin_state[il], 0.0f, v_dim * DS4_N_LIN_HEAD_DIM);
        if (g->layer_lin_hist[il])
            ds4_gpu_tensor_fill_f32(g->layer_lin_hist[il], 0.0f, (uint64_t)(DS4_N_LIN_CONV - 1u) * conv_dim);
    }
    /* the MTP row at position 0 reads h_{-1} = 0 */
    if (g->mtp_h) ds4_gpu_tensor_fill_f32(g->mtp_h, 0.0f, DS4_N_EMBD);
    g->pos = 0;
    g->mtp_pos = 0;
    g->mtp_h_pos0 = 0;
    g->mtp_h_rows = 0;
    g->snap_valid = false;
    g->snap_after_first = false;
    g->verify_rows_exact = false;
}

/* Before a forward at pos0: put h_{pos0-1} into row 0 of the seed rows.  It
 * is row pos0 - mtp_h_pos0 of the previous forward (row 0 itself after a
 * reset); a rejected verify rewinds to its pos0 + 1 and still finds it. */
static bool qwen35_graph_carry_h(ds4_qwen4_gpu_graph *g, uint32_t pos0) {
    if (!g->mtp_h) return true;
    if (pos0 < g->mtp_h_pos0 || pos0 - g->mtp_h_pos0 > g->mtp_h_rows) {
        fprintf(stderr, "ds4: Ornith MTP seed for position %u is not resident\n", pos0);
        return false;
    }
    const uint64_t row = (uint64_t)DS4_N_EMBD * sizeof(float);
    const uint32_t k = pos0 - g->mtp_h_pos0;
    return k == 0 || ds4_gpu_tensor_copy(g->mtp_h, 0, g->mtp_h, (uint64_t)k * row, row) != 0;
}

/* Embedding rows (dequantized on the host from Q8_0) and text rope
 * positions (t, h, w) = (p, p, p). */
static bool qwen35_graph_stage_inputs(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                      const int *tokens, uint32_t T) {
    const uint32_t E = DS4_N_EMBD;
    for (uint32_t t = 0; t < T; t++) {
        qwen4_ref_row(m, w->token_embd, (uint64_t)tokens[t], g->host_row + (uint64_t)t * E);
        uint32_t *p4 = g->host_pos3 + (uint64_t)t * 4u;
        p4[0] = p4[1] = p4[2] = g->pos + t;
        p4[3] = 0;
    }
    return ds4_gpu_tensor_write(g->R, 0, g->host_row, (uint64_t)T * E * sizeof(float)) &&
           ds4_gpu_tensor_write(g->pos3, (uint64_t)g->pos * 16u, g->host_pos3, (uint64_t)T * 16u);
}

/* qwen4_gemv's batched dispatch (qwen4_gemv_rows, ds4.c) picks its Q8_0/F16
 * matvec path by n_tok (dispatch geometry and internal tiling change with
 * the token count), and a T=1 call is not guaranteed to produce the same
 * row bit-identically as that row inside a T=2 call.  A speculative
 * verify's row 0 must equal plain 1-row decoding bit for bit (that is what
 * makes accepting it and continuing decoding correct), so every dense
 * projection reachable from a verify's 2-row forward routes through this
 * helper instead of calling qwen4_gemv directly: under g->verify_rows_exact
 * it dispatches one row at a time, each row taking exactly the T=1 kernel a
 * plain decode would use.  Plain decode, prefill and non-MTP graphs (T<=1 or
 * verify_rows_exact false) fall straight through to qwen4_gemv unchanged. */
static bool qwen35_gemv(ds4_qwen4_gpu_graph *g, ds4_gpu_tensor *out, const ds4_model *m,
                        const ds4_tensor *w, const ds4_gpu_tensor *x, uint32_t T) {
    if (!g->verify_rows_exact || T <= 1u) return qwen4_gemv(out, m, w, x, T);
    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->ndim >= 2 ? w->dim[1] : 1u;
    const uint64_t xrow = in_dim * sizeof(float), orow = out_dim * sizeof(float);
    bool ok = true;
    for (uint32_t t = 0; t < T && ok; t++) {
        ds4_gpu_tensor *xr = ds4_gpu_tensor_view(x, (uint64_t)t * xrow, xrow);
        ds4_gpu_tensor *or_ = ds4_gpu_tensor_view(out, (uint64_t)t * orow, orow);
        ok = xr && or_ && qwen4_gemv(or_, m, w, xr, 1u);
        ds4_gpu_tensor_free(xr);
        ds4_gpu_tensor_free(or_);
    }
    return ok;
}

/* q/gate, k and v projections of the rows in mixed, then the indexer-free
 * prep: q/k norms, RoPE and the F16 K/V append at positions pos0.. */
static bool qwen35_graph_attention_prep(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                                        uint32_t il, uint32_t pos0, uint32_t T) {
    return qwen35_gemv(g, g->qg, m, l->attn_q, g->mixed, T) &&
           qwen35_gemv(g, g->kp, m, l->attn_k, g->mixed, T) &&
           qwen35_gemv(g, g->vp, m, l->attn_v, g->mixed, T) &&
           ds4_gpu_qwen35_attn_prep_tensor(g->q, g->gate, g->layer_k_cache[il], g->layer_v_cache[il],
                                           g->qg, g->kp, g->vp, g->pos3, m->map, m->size,
                                           l->attn_q_norm->abs_offset, l->attn_k_norm->abs_offset,
                                           T, DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT,
                                           pos0, g->alloc_cap, DS4_ROPE_FREQ_BASE, DS4_RMS_EPS) != 0;
}

/* Dense attention (the qwen4 kernels apply sigmoid(gate)) of rows
 * [r0, r0 + n) of q/gate, at positions pos0 + r0.., into the same rows of
 * attn_o.  per_row gives every row its own dispatch: the key split then
 * matches a plain decode at that position, which a speculative verify
 * needs to reproduce plain decoding bit for bit. */
static bool qwen35_graph_attend(ds4_qwen4_gpu_graph *g, uint32_t il, uint32_t pos0,
                                uint32_t r0, uint32_t n, bool per_row) {
    const float scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);
    if (r0 == 0 && !per_row) {
        return ds4_gpu_qwen4_attn_decode_tensor(g->attn_o, g->q, g->gate, g->layer_k_cache[il], g->layer_v_cache[il],
                                                NULL, NULL, n <= 2u ? g->attn_part : NULL, n,
                                                DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, pos0, 0u, 0u, scale,
                                                NULL, NULL, NULL, NULL, 0u) != 0;
    }
    const uint64_t qrow = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM * sizeof(float);
    const uint32_t step = per_row ? 1u : n;
    bool ok = true;
    for (uint32_t t = r0; t < r0 + n && ok; t += step) {
        const uint64_t off = (uint64_t)t * qrow, bytes = (uint64_t)step * qrow;
        ds4_gpu_tensor *q = ds4_gpu_tensor_view(g->q, off, bytes);
        ds4_gpu_tensor *gt = ds4_gpu_tensor_view(g->gate, off, bytes);
        ds4_gpu_tensor *o = ds4_gpu_tensor_view(g->attn_o, off, bytes);
        ok = q && gt && o &&
             ds4_gpu_qwen4_attn_decode_tensor(o, q, gt, g->layer_k_cache[il], g->layer_v_cache[il],
                                              NULL, NULL, step <= 2u ? g->attn_part : NULL, step,
                                              DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, pos0 + t, 0u, 0u, scale,
                                              NULL, NULL, NULL, NULL, 0u) != 0;
        ds4_gpu_tensor_free(q);
        ds4_gpu_tensor_free(gt);
        ds4_gpu_tensor_free(o);
    }
    return ok;
}

/* Gated full attention over every cached position: projections and prep,
 * dense attention, output projection into blk. */
static bool qwen35_graph_attention(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                                   uint32_t il, uint32_t pos0, uint32_t T) {
    return qwen35_graph_attention_prep(g, m, l, il, pos0, T) &&
           qwen35_graph_attend(g, il, pos0, 0u, T, g->verify_rows_exact && T > 1u) &&
           qwen35_gemv(g, g->blk, m, l->attn_output, g->attn_o, T);
}

/* Router (softmax over 256, top 8, renormalized, sigmoid shared gate), the
 * routed experts and the shared expert, reduced into blk without a residual
 * (n_hc = 0); the caller adds blk to R.  Q4_K layers take the tiled expert
 * GEMMs above one tile of tokens, as qwen4 prefill does; Q5_K layers keep
 * the per-token row kernels.  Batches above 8 rows run the shared expert as
 * dense projections, single rows and verifies keep it as an extra slot. */
static bool qwen35_graph_moe(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                             uint32_t T) {
    const uint32_t E = DS4_N_EMBD, F = DS4_N_FF_EXP, NE = DS4_N_EXPERT, K = DS4_N_EXPERT_USED;
    const uint32_t xt = l->ffn_gate_exps->type, dt = l->ffn_down_exps->type;
    bool ok = qwen35_gemv(g, g->router, m, l->ffn_gate_inp, g->mixed, T) &&
        ds4_gpu_qwen4_router_topk_tensor(g->selected, g->weights, g->router, g->mixed, m->map, m->size,
                                         l->ffn_gate_inp_shexp->abs_offset, l->ffn_gate_inp_shexp->type,
                                         E, g->sh_gate_logit, T, NE, K) != 0;
    const bool mm = T > 64u && xt == DS4_TENSOR_Q4_K && dt == DS4_TENSOR_Q4_K;
    const bool shared_dense = mm || T > 8u;
    if (ok && shared_dense) {
        ok = qwen35_gemv(g, g->sh_gate, m, l->ffn_gate_shexp, g->mixed, T) &&
             qwen35_gemv(g, g->sh_up, m, l->ffn_up_shexp, g->mixed, T) &&
             ds4_gpu_swiglu_tensor(g->sh_mid, g->sh_gate, g->sh_up, T * F, 0.0f, 1.0f) &&
             qwen35_gemv(g, g->sh_out, m, l->ffn_down_shexp, g->sh_mid, T);
    }
    if (ok && mm) {
        ok = ds4_gpu_qwen4_moe_build_lists_tensor(g->moe_lists, g->moe_counts, g->selected, T, K, NE,
                                                  g->cap_tokens) &&
             ds4_gpu_qwen4_moe_mm_mid_tensor(g->mid, g->mixed, g->moe_lists, g->moe_counts, m->map, m->size,
                                             l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset, xt,
                                             NE, T, K, K, E, F, g->cap_tokens) &&
             ds4_gpu_qwen4_moe_mm_down_tensor(g->part, g->mid, g->moe_lists, g->moe_counts, m->map, m->size,
                                              l->ffn_down_exps->abs_offset, dt, NE, T, K, K, F, E, g->cap_tokens);
    } else if (ok) {
        const uint64_t sg = shared_dense ? 0u : l->ffn_gate_shexp->abs_offset;
        const uint64_t su = shared_dense ? 0u : l->ffn_up_shexp->abs_offset;
        const uint64_t sd = shared_dense ? 0u : l->ffn_down_shexp->abs_offset;
        const uint32_t st = shared_dense ? UINT32_MAX : l->ffn_gate_shexp->type;
        const uint32_t sdt = shared_dense ? UINT32_MAX : l->ffn_down_shexp->type;
        if (xt == DS4_TENSOR_Q5_K) {
            ok = ds4_gpu_qwen35_moe_mid_tensor(g->mid, g->mixed, g->selected, m->map, m->size,
                                               l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset, xt,
                                               NE, T, K, E, F, sg, su, st) != 0;
        } else {
            ok = ds4_gpu_qwen4_moe_mid_tensor(g->mid, g->mixed, g->selected, m->map, m->size,
                                              l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset, xt,
                                              NE, T, K, E, F, sg, su, st) != 0;
        }
        if (ok && dt == DS4_TENSOR_Q5_K) {
            ok = ds4_gpu_qwen35_moe_down_tensor(g->part, g->mid, g->selected, m->map, m->size,
                                                l->ffn_down_exps->abs_offset, dt, NE, T, K, F, E, sd, sdt) != 0;
        } else if (ok) {
            ok = ds4_gpu_qwen4_moe_down_tensor(g->part, g->mid, g->selected, m->map, m->size,
                                               l->ffn_down_exps->abs_offset, dt, NE, T, K, F, E, sd, sdt) != 0;
        }
    }
    if (ok) {
        ok = ds4_gpu_qwen4_moe_reduce_tensor(g->blk, g->part, g->weights, g->sh_gate_logit,
                                             shared_dense ? g->sh_out : NULL, NULL, NULL, T, K,
                                             K + (shared_dense ? 0u : 1u), E, 0u) != 0;
    }
    return ok;
}

/* qwen4_graph_linear (shared with Qwen3.8, ds4.c) has no x/out parameters
 * of its own: it reads g->mixed and writes g->blk directly, using g->qkv,
 * g->z, g->ga, g->gb and g->lin_o purely as scratch it fills and consumes
 * within the same call (row 0 of that scratch is reused safely on every
 * T=1 call).  g->layer_lin_state[il]/g->layer_lin_hist[il] are the GDN
 * recurrent state: calling it T=1 once per row in order advances that
 * state exactly as plain decoding would.  So under g->verify_rows_exact
 * this points g->mixed/g->blk at per-row views for a T=1 call each, and
 * sets g->snap_after_first only for the row-0 call (qwen4_graph_linear
 * snapshots the GDN state after the call's row 0, so a 1-row call's
 * snapshot is already the state after that row) -- restoring all three
 * fields on every path, success or failure. */
static bool qwen35_graph_linear_rows(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                                     uint32_t il, uint32_t T) {
    if (!g->verify_rows_exact || T <= 1u) return qwen4_graph_linear(g, m, l, il, T);
    ds4_gpu_tensor *const mixed0 = g->mixed, *const blk0 = g->blk;
    const bool snap0 = g->snap_after_first;
    const uint64_t row = (uint64_t)DS4_N_EMBD * sizeof(float);
    bool ok = true;
    for (uint32_t t = 0; t < T && ok; t++) {
        ds4_gpu_tensor *mv = ds4_gpu_tensor_view(mixed0, (uint64_t)t * row, row);
        ds4_gpu_tensor *bv = ds4_gpu_tensor_view(blk0, (uint64_t)t * row, row);
        g->mixed = mv;
        g->blk = bv;
        g->snap_after_first = snap0 && t == 0u;
        ok = mv && bv && qwen4_graph_linear(g, m, l, il, 1u);
        ds4_gpu_tensor_free(mv);
        ds4_gpu_tensor_free(bv);
    }
    g->mixed = mixed0;
    g->blk = blk0;
    g->snap_after_first = snap0;
    return ok;
}

/* One trunk layer: pre-norm mixer and pre-norm MoE, each added to R. */
static bool qwen35_graph_layer(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                               uint32_t il, uint32_t pos0, uint32_t T) {
    const ds4_layer_weights *l = &w->layer[il];
    const uint32_t n = T * DS4_N_EMBD;
    bool ok = ds4_gpu_rms_norm_weight_rows_tensor(g->mixed, g->R, m->map, m->size, l->attn_norm->abs_offset,
                                                  DS4_N_EMBD, T, DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_qwen35_layer_is_attention(il) ? qwen35_graph_attention(g, m, l, il, pos0, T)
                                                    : qwen35_graph_linear_rows(g, m, l, il, T);
    if (ok) ok = ds4_gpu_add_tensor(g->R, g->R, g->blk, n) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->mixed, g->R, m->map, m->size, l->ffn_norm->abs_offset,
                                                     DS4_N_EMBD, T, DS4_RMS_EPS) != 0;
    if (ok) ok = qwen35_graph_moe(g, m, l, T);
    if (ok) ok = ds4_gpu_add_tensor(g->R, g->R, g->blk, n) != 0;
    return ok;
}

/* Forward T tokens at g->pos..; logits_out (optional) receives the last
 * row, or all T rows with all_rows (an MTP verify).  Causal by
 * construction: the recurrent kernels walk tokens in order and attention
 * reads the caches written for the same chunk.  A caller that sets
 * g->snap_after_first gets the GDN state after row 0 in the snapshot set
 * (g->snap_valid, g->snap_pos). */
static bool qwen35_graph_forward_tokens(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                        const int *tokens, uint32_t T, float *logits_out, bool all_rows) {
    if (!g) return false;
    const bool snap = g->snap_after_first;
    g->snap_after_first = false;
    if (T == 0 || T > g->cap_tokens || g->pos + T > g->ctx_cap) return false;
    if (all_rows && (!logits_out || !g->mtp_h || T > g->n_logit_rows)) return false;
    for (uint32_t t = 0; t < T; t++) {
        if (tokens[t] < 0 || tokens[t] >= (int)DS4_N_VOCAB) {
            fprintf(stderr, "ds4: Ornith token id %d is outside the vocabulary\n", tokens[t]);
            return false;
        }
    }
    const uint32_t pos0 = g->pos;
    const uint64_t row = (uint64_t)DS4_N_EMBD * sizeof(float);
    if (!qwen35_graph_stage_inputs(g, m, w, tokens, T)) return false;
    if (!glm_graph_begin_commands_if_needed()) return false;
    g->snap_after_first = snap;
    if (snap) g->snap_pos = pos0 + 1u;
    bool ok = qwen35_graph_carry_h(g, pos0);
    for (uint32_t il = 0; il + DS4_N_NEXTN_PREDICT < DS4_N_LAYER && ok; il++) {
        ok = qwen35_graph_layer(g, m, w, il, pos0, T);
    }
    if (ok && g->mtp_h) {
        /* post-output_norm rows: the MTP seed rows 1..T and the LM head input */
        ds4_gpu_tensor *h = ds4_gpu_tensor_view(g->mtp_h, row, (uint64_t)T * row);
        ok = h && ds4_gpu_rms_norm_weight_rows_tensor(h, g->R, m->map, m->size, w->output_norm->abs_offset,
                                                      DS4_N_EMBD, T, DS4_RMS_EPS) != 0;
        if (ok && logits_out) {
            ds4_gpu_tensor *last = all_rows ? NULL : ds4_gpu_tensor_view(g->mtp_h, (uint64_t)T * row, row);
            ok = (all_rows || last) && qwen35_gemv(g, g->logits, m, w->output, all_rows ? h : last, all_rows ? T : 1u);
            ds4_gpu_tensor_free(last);
        }
        ds4_gpu_tensor_free(h);
    } else if (ok && logits_out) {
        ds4_gpu_tensor *last = ds4_gpu_tensor_view(g->R, (uint64_t)(T - 1u) * row, row);
        ok = last && ds4_gpu_rms_norm_weight_rows_tensor(g->mixed, last, m->map, m->size, w->output_norm->abs_offset,
                                                         DS4_N_EMBD, 1u, DS4_RMS_EPS) != 0 &&
             qwen4_gemv(g->logits, m, w->output, g->mixed, 1u);
        ds4_gpu_tensor_free(last);
    }
    if (!ds4_gpu_end_commands()) ok = false;
    const uint32_t n_rows = all_rows ? T : 1u;
    if (ok && logits_out) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits_out, (uint64_t)n_rows * DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (ok) {
        g->pos += T;
        if (g->mtp_h) {
            g->mtp_h_pos0 = pos0;
            g->mtp_h_rows = T;
        }
    }
    if (snap) g->snap_valid = ok;
    g->snap_after_first = false;
    return ok;
}

/* The MTP block (blk.40) over tokens at positions pos0..pos0+T-1.  Row t
 * reads x0 = eh_proj . [RMSNorm(embed(token_t)) * enorm | RMSNorm(h) * hnorm]
 * with h = h_{pos0+t-1} from mtp_h, and appends its K/V to the block's own
 * cache: the catch-up that keeps that cache complete for every position.
 * Rows run in sub-batches of cap_tokens.  With want_draft the last row
 * continues through attention, the MoE, shared_head_norm and the output
 * head, and *draft_out gets the argmax: the token after tokens[T-1]. */
static DS4_MAYBE_UNUSED bool qwen35_graph_mtp(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                             const int *tokens, uint32_t T, uint32_t pos0, bool want_draft, int *draft_out) {
    const uint32_t il = DS4_N_LAYER - 1u, E = DS4_N_EMBD;
    const ds4_layer_weights *l = &w->layer[il];
    const uint64_t row = (uint64_t)E * sizeof(float);
    const uint64_t qrow = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM * sizeof(float);
    if (!g->mtp_h || T == 0 || pos0 + T > g->ctx_cap || (want_draft && !draft_out) ||
        pos0 < g->mtp_h_pos0 || pos0 - g->mtp_h_pos0 + T - 1u > g->mtp_h_rows) {
        return false;
    }
    for (uint32_t t = 0; t < T; t++) {
        if (tokens[t] < 0 || tokens[t] >= (int)DS4_N_VOCAB) return false;
    }
    bool ok = true;
    uint32_t sub0 = pos0, last = 0;
    for (uint32_t done = 0; done < T && ok;) {
        const uint32_t n = T - done < g->cap_tokens ? T - done : g->cap_tokens;
        const uint32_t p = pos0 + done;
        for (uint32_t t = 0; t < n; t++) {
            qwen4_ref_row(m, w->token_embd, (uint64_t)tokens[done + t], g->host_row + (uint64_t)t * E);
            uint32_t *p4 = g->host_pos3 + (uint64_t)t * 4u;
            p4[0] = p4[1] = p4[2] = p + t;
            p4[3] = 0;
        }
        ok = ds4_gpu_tensor_write(g->mtp_e, 0, g->host_row, (uint64_t)n * row) &&
             ds4_gpu_tensor_write(g->pos3, (uint64_t)p * 16u, g->host_pos3, (uint64_t)n * 16u) &&
             glm_graph_begin_commands_if_needed();
        ds4_gpu_tensor *h = ok ? ds4_gpu_tensor_view(g->mtp_h, (uint64_t)(p - g->mtp_h_pos0) * row,
                                                     (uint64_t)n * row) : NULL;
        ok = ok && h &&
             ds4_gpu_qwen35_mtp_concat_tensor(g->mtp_cat, g->mtp_e, h, m->map, m->size,
                                              l->nextn_enorm->abs_offset, l->nextn_hnorm->abs_offset,
                                              E, n, DS4_RMS_EPS) != 0 &&
             qwen4_gemv(g->mtp_R, m, l->nextn_eh_proj, g->mtp_cat, n) &&
             ds4_gpu_rms_norm_weight_rows_tensor(g->mixed, g->mtp_R, m->map, m->size, l->attn_norm->abs_offset,
                                                 E, n, DS4_RMS_EPS) != 0 &&
             qwen35_graph_attention_prep(g, m, l, il, p, n);
        ds4_gpu_tensor_free(h);
        sub0 = p;
        last = n - 1u;
        done += n;
        /* the next sub-batch rewrites the staged embeddings */
        if (done < T && !ds4_gpu_end_commands()) ok = false;
    }
    if (ok && want_draft) {
        ds4_gpu_tensor *x = ds4_gpu_tensor_view(g->mtp_R, (uint64_t)last * row, row);
        ds4_gpu_tensor *o = ds4_gpu_tensor_view(g->attn_o, (uint64_t)last * qrow, qrow);
        ok = x && o &&
             qwen35_graph_attend(g, il, sub0, last, 1u, false) &&
             qwen4_gemv(g->blk, m, l->attn_output, o, 1u) &&
             ds4_gpu_add_tensor(x, x, g->blk, E) != 0 &&
             ds4_gpu_rms_norm_weight_rows_tensor(g->mixed, x, m->map, m->size, l->ffn_norm->abs_offset,
                                                 E, 1u, DS4_RMS_EPS) != 0 &&
             qwen35_graph_moe(g, m, l, 1u) &&
             ds4_gpu_add_tensor(x, x, g->blk, E) != 0 &&
             ds4_gpu_rms_norm_weight_rows_tensor(g->mixed, x, m->map, m->size,
                                                 l->nextn_shared_head_norm->abs_offset, E, 1u, DS4_RMS_EPS) != 0 &&
             qwen4_gemv(g->logits, m, w->output, g->mixed, 1u) &&
             ds4_gpu_qwen4_argmax_tensor(g->mtp_argmax, g->mtp_argmax_tmp, g->logits, DS4_N_VOCAB) != 0;
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(o);
    }
    if (!ds4_gpu_end_commands()) ok = false;
    if (ok && want_draft) {
        int32_t token = -1;
        ok = ds4_gpu_tensor_read(g->mtp_argmax, 0, &token, sizeof(token)) != 0 &&
             token >= 0 && token < (int32_t)DS4_N_VOCAB;
        if (ok) *draft_out = token;
    }
    if (ok) g->mtp_pos = pos0 + T;
    return ok;
}

/* A rejected draft returns to the state the verify snapshotted after its
 * first row: swap the live and snapshot GDN buffers, as the qwen4 cycle
 * does, without the PLE state Ornith does not have.  K/V rows past the
 * restored position are overwritten by the next forward, and the seed row
 * for it is still resident (the verify's row 1). */
static DS4_MAYBE_UNUSED bool qwen35_graph_state_swap(ds4_qwen4_gpu_graph *g) {
    if (!g->snap_valid) return false;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!g->snap_lin_state[il]) continue;
        ds4_gpu_tensor *t = g->layer_lin_state[il];
        g->layer_lin_state[il] = g->snap_lin_state[il];
        g->snap_lin_state[il] = t;
        t = g->layer_lin_hist[il];
        g->layer_lin_hist[il] = g->snap_lin_hist[il];
        g->snap_lin_hist[il] = t;
    }
    g->pos = g->snap_pos;
    g->snap_valid = false;
    return true;
}

static int generate_qwen35_metal_argmax(
        const ds4_model *model, const ds4_vocab *vocab, const ds4_weights *weights,
        const token_vec *prompt, int n_predict, int ctx_size, uint32_t prefill_chunk,
        ds4_token_emit_fn emit, ds4_generation_done_fn done, void *emit_ud,
        ds4_session_progress_fn progress, void *progress_ud) {
    if (!prompt || prompt->len <= 0 || prompt->len >= ctx_size) {
        fprintf(stderr, "ds4: prompt is empty or exceeds context size\n");
        return 1;
    }
    const uint32_t cap = prefill_chunk && prefill_chunk < (uint32_t)ctx_size ?
        prefill_chunk : qwen35_prefill_chunk_tokens((uint32_t)ctx_size);
    ds4_qwen4_gpu_graph *g = xcalloc(1, sizeof(*g));
    if (!qwen35_graph_alloc(g, (uint32_t)ctx_size, cap, false)) {
        free(g);
        return 1;
    }
    qwen35_graph_reset(g);
    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    const double t_prefill0 = now_sec();
    bool ok = true;
    for (int i = 0; i < prompt->len && ok;) {
        uint32_t chunk = (uint32_t)(prompt->len - i);
        if (chunk > g->cap_tokens) chunk = g->cap_tokens;
        ok = qwen35_graph_forward_tokens(g, model, weights, prompt->v + i, chunk,
                                         i + (int)chunk == prompt->len ? logits : NULL, false);
        i += (int)chunk;
        if (progress) progress(progress_ud, "prefill_chunk", i, prompt->len);
    }
    const double t_prefill1 = now_sec();
    int n_generated = 0;
    const double t_decode0 = now_sec();
    for (int i = 0; ok && i < n_predict && g->pos < g->ctx_cap; i++) {
        const int token = sample_argmax(logits, DS4_N_VOCAB);
        if (vocab_token_is_generation_stop(vocab, token)) break;
        if (emit) emit(emit_ud, token);
        n_generated++;
        if (i == n_predict - 1 || g->pos + 1u >= g->ctx_cap) break;
        ok = qwen35_graph_forward_tokens(g, model, weights, &token, 1, logits, false);
    }
    const double t_decode1 = now_sec();
    if (!ok) fprintf(stderr, "ds4: Ornith forward failed at position %u\n", g->pos);
    if (ok && done) done(emit_ud);
    ds4_log(stderr, DS4_LOG_TIMING, "ds4: Ornith prefill: %.2f t/s, generation: %.2f t/s\n",
            t_prefill1 > t_prefill0 ? (double)prompt->len / (t_prefill1 - t_prefill0) : 0.0,
            t_decode1 > t_decode0 ? (double)n_generated / (t_decode1 - t_decode0) : 0.0);
    free(logits);
    qwen4_graph_free(g);
    free(g);
    return ok ? 0 : 1;
}
