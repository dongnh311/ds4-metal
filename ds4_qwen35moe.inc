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
 * arithmetic-equivalent paths; Ornith's own knobs use DS4_QWEN35_*. */

static bool qwen35_graph_alloc(ds4_qwen4_gpu_graph *g, uint32_t ctx_cap, uint32_t cap_tokens) {
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
    g->n_logit_rows = 1u;
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
    QWEN35_ALLOC(logits, (uint64_t)DS4_N_VOCAB);
    QWEN35_ALLOC(pos3, (uint64_t)ctx_cap * 4u);
#undef QWEN35_ALLOC
    /* F16 K/V per full-attention trunk layer; recurrent state and conv
     * history per GDN layer.  The MTP block's caches arrive with M2. */
    const uint64_t kv_bytes = (uint64_t)ctx_cap * kv_dim * 2u;
    for (uint32_t il = 0; il + DS4_N_NEXTN_PREDICT < DS4_N_LAYER && ok; il++) {
        if (ds4_qwen35_layer_is_attention(il)) {
            g->layer_k_cache[il] = ds4_gpu_tensor_alloc(kv_bytes);
            g->layer_v_cache[il] = ds4_gpu_tensor_alloc(kv_bytes);
            ok = g->layer_k_cache[il] && g->layer_v_cache[il];
        } else {
            g->layer_lin_state[il] = qwen4_graph_alloc_f32(v_dim * DS4_N_LIN_HEAD_DIM);
            g->layer_lin_hist[il] = qwen4_graph_alloc_f32((uint64_t)(DS4_N_LIN_CONV - 1u) * conv_dim);
            ok = g->layer_lin_state[il] && g->layer_lin_hist[il];
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
    g->pos = 0;
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

/* Gated full attention over every cached position: q/gate, k and v
 * projections, indexer-free prep (norms, RoPE, F16 KV append), the qwen4
 * dense attention kernels (which apply sigmoid(gate)), output projection. */
static bool qwen35_graph_attention(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                                   uint32_t il, uint32_t pos0, uint32_t T) {
    const float scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);
    return qwen4_gemv(g->qg, m, l->attn_q, g->mixed, T) &&
           qwen4_gemv(g->kp, m, l->attn_k, g->mixed, T) &&
           qwen4_gemv(g->vp, m, l->attn_v, g->mixed, T) &&
           ds4_gpu_qwen35_attn_prep_tensor(g->q, g->gate, g->layer_k_cache[il], g->layer_v_cache[il],
                                           g->qg, g->kp, g->vp, g->pos3, m->map, m->size,
                                           l->attn_q_norm->abs_offset, l->attn_k_norm->abs_offset,
                                           T, DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT,
                                           pos0, g->alloc_cap, DS4_ROPE_FREQ_BASE, DS4_RMS_EPS) != 0 &&
           ds4_gpu_qwen4_attn_decode_tensor(g->attn_o, g->q, g->gate, g->layer_k_cache[il], g->layer_v_cache[il],
                                            NULL, NULL, T <= 2u ? g->attn_part : NULL, T,
                                            DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, pos0, 0u, 0u, scale,
                                            NULL, NULL, NULL, NULL, 0u) != 0 &&
           qwen4_gemv(g->blk, m, l->attn_output, g->attn_o, T);
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
    bool ok = qwen4_gemv(g->router, m, l->ffn_gate_inp, g->mixed, T) &&
        ds4_gpu_qwen4_router_topk_tensor(g->selected, g->weights, g->router, g->mixed, m->map, m->size,
                                         l->ffn_gate_inp_shexp->abs_offset, l->ffn_gate_inp_shexp->type,
                                         E, g->sh_gate_logit, T, NE, K) != 0;
    const bool mm = T > 64u && xt == DS4_TENSOR_Q4_K && dt == DS4_TENSOR_Q4_K;
    const bool shared_dense = mm || T > 8u;
    if (ok && shared_dense) {
        ok = qwen4_gemv(g->sh_gate, m, l->ffn_gate_shexp, g->mixed, T) &&
             qwen4_gemv(g->sh_up, m, l->ffn_up_shexp, g->mixed, T) &&
             ds4_gpu_swiglu_tensor(g->sh_mid, g->sh_gate, g->sh_up, T * F, 0.0f, 1.0f) &&
             qwen4_gemv(g->sh_out, m, l->ffn_down_shexp, g->sh_mid, T);
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

/* One trunk layer: pre-norm mixer and pre-norm MoE, each added to R. */
static bool qwen35_graph_layer(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                               uint32_t il, uint32_t pos0, uint32_t T) {
    const ds4_layer_weights *l = &w->layer[il];
    const uint32_t n = T * DS4_N_EMBD;
    bool ok = ds4_gpu_rms_norm_weight_rows_tensor(g->mixed, g->R, m->map, m->size, l->attn_norm->abs_offset,
                                                  DS4_N_EMBD, T, DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_qwen35_layer_is_attention(il) ? qwen35_graph_attention(g, m, l, il, pos0, T)
                                                    : qwen4_graph_linear(g, m, l, il, T);
    if (ok) ok = ds4_gpu_add_tensor(g->R, g->R, g->blk, n) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->mixed, g->R, m->map, m->size, l->ffn_norm->abs_offset,
                                                     DS4_N_EMBD, T, DS4_RMS_EPS) != 0;
    if (ok) ok = qwen35_graph_moe(g, m, l, T);
    if (ok) ok = ds4_gpu_add_tensor(g->R, g->R, g->blk, n) != 0;
    return ok;
}

/* Forward T tokens at g->pos..; logits_out (optional) receives the last
 * row.  Causal by construction: the recurrent kernels walk tokens in order
 * and attention reads the caches written for the same chunk. */
static bool qwen35_graph_forward_tokens(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                        const int *tokens, uint32_t T, float *logits_out) {
    if (!g || T == 0 || T > g->cap_tokens || g->pos + T > g->ctx_cap) return false;
    for (uint32_t t = 0; t < T; t++) {
        if (tokens[t] < 0 || tokens[t] >= (int)DS4_N_VOCAB) {
            fprintf(stderr, "ds4: Ornith token id %d is outside the vocabulary\n", tokens[t]);
            return false;
        }
    }
    const uint32_t pos0 = g->pos;
    if (!qwen35_graph_stage_inputs(g, m, w, tokens, T)) return false;
    if (!glm_graph_begin_commands_if_needed()) return false;
    bool ok = true;
    for (uint32_t il = 0; il + DS4_N_NEXTN_PREDICT < DS4_N_LAYER && ok; il++) {
        ok = qwen35_graph_layer(g, m, w, il, pos0, T);
    }
    if (ok && logits_out) {
        ds4_gpu_tensor *last = ds4_gpu_tensor_view(g->R, (uint64_t)(T - 1u) * DS4_N_EMBD * sizeof(float),
                                                   (uint64_t)DS4_N_EMBD * sizeof(float));
        ok = last && ds4_gpu_rms_norm_weight_rows_tensor(g->mixed, last, m->map, m->size, w->output_norm->abs_offset,
                                                         DS4_N_EMBD, 1u, DS4_RMS_EPS) != 0 &&
             qwen4_gemv(g->logits, m, w->output, g->mixed, 1u);
        ds4_gpu_tensor_free(last);
    }
    if (!ds4_gpu_end_commands()) ok = false;
    if (ok && logits_out) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits_out, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (ok) g->pos += T;
    return ok;
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
    if (!qwen35_graph_alloc(g, (uint32_t)ctx_size, cap)) {
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
                                         i + (int)chunk == prompt->len ? logits : NULL);
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
        ok = qwen35_graph_forward_tokens(g, model, weights, &token, 1, logits);
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
