// Ornith-1.5-35B-A3B (llama.cpp qwen35moe) kernels.  This file is appended
// after qwen4.metal in the single Metal library, so the qwen4 helpers and
// argument structs (qwen4_row_dot, qwen4_silu, qwen4_sigmoid,
// ds4_metal_args_qwen4_moe, ds4_metal_args_qwen4_gdn_out) are in scope.
// Everything here is a new entry point: the Qwen3.8 kernels are not touched.

/* --- Q5_K routed experts ------------------------------------------------ */

/* q5_K: 176-byte super-blocks of 256 (d, dmin, 12 packed 6-bit scale/min
 * pairs, 32 high-bit bytes, 128 nibble bytes).  The lane mapping and the
 * accumulation order follow the q4_K case of qwen4_row_dot: group = lane/4,
 * l = (lane%4)*8, eight consecutive elements per lane per block.  Element j
 * of group g takes its fifth bit from bit g of qh[j]. */
static inline float qwen35_row_dot(device const char *row, device const float *x,
                                   uint weight_type, uint in_dim, ushort tiisg) {
    if (weight_type != 13u) return qwen4_row_dot(row, x, weight_type, in_dim, tiisg);
    float acc = 0.0f;
    const uint nb = in_dim / 256u;
    const uint group = tiisg / 4, l = (tiisg % 4) * 8;
    for (uint ib = 0; ib < nb; ib++) {
        device const uchar *blk = (device const uchar *)(row + (uint64_t)ib * 176);
        const float d = (float)(*(device const half *)blk);
        const float dmin = (float)(*(device const half *)(blk + 2));
        device const uchar *sc = blk + 4;
        uint s, mn;
        if (group < 4) { s = sc[group] & 63u; mn = sc[group + 4] & 63u; }
        else { s = (sc[group + 4] & 0xFu) | ((sc[group - 4] & 0xC0u) >> 2); mn = (sc[group + 4] >> 4) | ((sc[group] & 0xC0u) >> 2); }
        const float ds = d * (float)s, dm = dmin * (float)mn;
        device const uchar *qh = blk + 16 + l;
        device const uchar *qs = blk + 48 + (group >> 1) * 32 + l;
        const uint shift = (group & 1u) * 4u;
        device const float *y = x + ib * 256 + group * 32 + l;
        for (uint i = 0; i < 8; i++) {
            const uint q = ((qs[i] >> shift) & 0xFu) | (((qh[i] >> group) & 1u) << 4);
            acc += (ds * (float)q - dm) * y[i];
        }
    }
    return simd_sum(acc);
}

/* kernel_qwen4_moe_mid with qwen35_row_dot: mid[t][s][r] =
 * silu(gate_row . x) * (up_row . x); slot n_slots (when has_shared) is the
 * shared expert from its own bases.  Two rows per SIMD group. */
kernel void kernel_qwen35_moe_mid(
        constant ds4_metal_args_qwen4_moe & args,
        device const char    *gate_base,
        device const char    *up_base,
        device const int32_t *selected,   /* [T][n_slots] */
        device const float   *x,          /* [T][in_dim] */
        device float         *mid,        /* [T][n_slots+has_shared][out_rows] */
        device const char    *sh_gate,
        device const char    *sh_up,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint tok = tgpig.z;
    const uint n_out = args.n_slots + args.has_shared;
    const uint nr = 2u;
    const uint row0 = (tgpig.x * (ntg.x / 32u) + (uint)sgitg) * nr;
    if (row0 >= args.out_rows || slot >= n_out || tok >= args.n_tokens) return;
    const bool shared = slot == args.n_slots;
    const uint type = shared ? args.shared_type : args.weight_type;
    const uint row_bytes = shared ? args.shared_row_bytes : args.row_bytes;
    device const char *gb = shared ? sh_gate : gate_base;
    device const char *ub = shared ? sh_up : up_base;
    const uint64_t ebase = shared ? 0 : (uint64_t)(uint)selected[(uint64_t)tok * args.n_slots + slot] * args.expert_bytes;
    device const float *xt = x + (uint64_t)tok * args.in_dim;
    for (uint r = row0; r < row0 + nr && r < args.out_rows; r++) {
        const uint64_t off = ebase + (uint64_t)r * row_bytes;
        const float g = qwen35_row_dot(gb + off, xt, type, args.in_dim, tiisg);
        const float u = qwen35_row_dot(ub + off, xt, type, args.in_dim, tiisg);
        if (tiisg == 0) mid[((uint64_t)tok * n_out + slot) * args.out_rows + r] = qwen4_silu(g) * u;
    }
}

/* kernel_qwen4_moe_down with qwen35_row_dot. */
kernel void kernel_qwen35_moe_down(
        constant ds4_metal_args_qwen4_moe & args,
        device const char    *down_base,
        device const int32_t *selected,   /* [T][n_slots] */
        device const float   *mid,        /* [T][n_slots+has_shared][in_dim] */
        device float         *part,       /* [T][n_slots+has_shared][out_rows] */
        device const char    *sh_down,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint tok = tgpig.z;
    const uint n_out = args.n_slots + args.has_shared;
    const uint nr = 2u;
    const uint row0 = (tgpig.x * (ntg.x / 32u) + (uint)sgitg) * nr;
    if (row0 >= args.out_rows || slot >= n_out || tok >= args.n_tokens) return;
    const bool shared = slot == args.n_slots;
    const uint type = shared ? args.shared_type : args.weight_type;
    const uint row_bytes = shared ? args.shared_row_bytes : args.row_bytes;
    device const char *db = shared ? sh_down : down_base;
    const uint64_t pair = (uint64_t)tok * n_out + slot;
    const uint64_t ebase = shared ? 0 : (uint64_t)(uint)selected[(uint64_t)tok * args.n_slots + slot] * args.expert_bytes;
    device const float *m = mid + pair * args.in_dim;
    for (uint r = row0; r < row0 + nr && r < args.out_rows; r++) {
        const float v = qwen35_row_dot(db + ebase + (uint64_t)r * row_bytes, m, type, args.in_dim, tiisg);
        if (tiisg == 0) part[pair * args.out_rows + r] = v;
    }
}

/* --- Gated DeltaNet output ---------------------------------------------- */

/* Qwen3.5 RMSNormGated: per-head RMSNorm of the scan output, scaled by
 * ssm_norm and gated by silu(z).  Qwen3.8 gates with sigmoid
 * (kernel_qwen4_gdn_out); the arithmetic is otherwise the same. */
kernel void kernel_qwen35_gdn_out(
        constant ds4_metal_args_qwen4_gdn_out & args,
        device float       *o,        /* [T][H*D], in place */
        device const float *z,        /* [T][H*D] */
        device const float *weight,   /* [D] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint h = tgpig.x;
    const uint tok = tgpig.y;
    if (h >= args.n_head || tok >= args.n_tokens) return;
    const uint D = args.head_dim;
    const uint npt = D / 32;
    const uint64_t base = ((uint64_t)tok * args.n_head + h) * D + tiisg * npt;
    float ss = 0.0f;
    for (uint i = 0; i < npt; i++) ss += o[base + i] * o[base + i];
    ss = simd_sum(ss);
    const float r = rsqrt(ss / (float)D + args.eps);
    for (uint i = 0; i < npt; i++) {
        o[base + i] = o[base + i] * r * weight[tiisg * npt + i] * qwen4_silu(z[base + i]);
    }
}
