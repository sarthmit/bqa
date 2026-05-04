"""
Triton forward kernel for DynamicBasisQueryAttention (bqa_dyn).

Math (per query position t, head h):
    score[t, s] = (1/sqrt(D)) · Σ_j w_k[t, h, j] · ⟨Q[t, h], K_basis[s, j]⟩
    p[t, s]     = causal/window-masked softmax over s of score[t, s]
    y[t, h, d]  = Σ_j w_v[t, h, j] · Σ_s p[t, s] · V_eff[s, h, j, d]
    V_eff[s, h, j, d] = V_basis[s, j, d] + gate[s, h] · VE[s, j, d]   (if HAS_VE)

Single-pass online-softmax kernel that:
- Keeps head_dim = D (no J-fold widening of head_dim like the SDPA path).
- Maintains a single (BLOCK_M, D) accumulator A[m, d] across the s loop, exploiting
  w_v's independence from s: A_new = α·A_old + Σ_j w_v[m, j] · (P @ V_j).
  (3D (J, BLOCK_M, D) state is not needed; the w_v reduction commutes with the s sum.)
- Issues J standard tl.dot calls per s-block for the score, summed with w_k.
- Handles causal + sliding-window masks natively (cheap — block-skip outside the band).
- Bakes the per-source-per-head ve gate into V_eff inside the kernel when HAS_VE=True.

Forward only on this branch — bqa_dyn's J×attention-compute floor still applies, but
this kernel removes the q_eff HBM materialisation and the J*D-wide attention call,
which together account for most of the bqa_dyn-vs-MHA gap on A100 (per
project_bqa_dyn_findings memory). Backward is left for a follow-up; until then,
training paths must keep using DynamicBasisQueryAttention.forward.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _bqa_dyn_attn_fwd_kernel(
    Q, K, V, VE, GATE, WK, WV, O, L,
    sQb, sQt, sQh, sQd,
    sKb, sKt, sKj, sKd,
    sVb, sVt, sVj, sVd,
    sVEb, sVEt, sVEj, sVEd,
    sGb, sGt, sGh,
    sWKb, sWKt, sWKh, sWKj,
    sWVb, sWVt, sWVh, sWVj,
    sOb, sOt, sOh, sOd,
    sLb, sLt, sLh,
    sm_scale,
    H,
    T,
    window_left,
    HAS_VE: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    SAVE_L: tl.constexpr,        # store logsumexp for backward
    J: tl.constexpr,
    J_PAD: tl.constexpr,        # next pow2 ≥ J (tl.arange requires PoT)
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_j = tl.arange(0, J_PAD)
    mask_j = offs_j < J
    mask_m = offs_m < T

    q_ptrs = (
        Q + b * sQb + h * sQh
        + offs_m[:, None] * sQt + offs_d[None, :] * sQd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    wk_ptrs = (
        WK + b * sWKb + h * sWKh
        + offs_m[:, None] * sWKt + offs_j[None, :] * sWKj
    )
    wv_ptrs = (
        WV + b * sWVb + h * sWVh
        + offs_m[:, None] * sWVt + offs_j[None, :] * sWVj
    )
    wk_load_mask = mask_m[:, None] & mask_j[None, :]
    w_k = tl.load(wk_ptrs, mask=wk_load_mask, other=0.0).to(tl.float32)
    w_v = tl.load(wv_ptrs, mask=wk_load_mask, other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    if CAUSAL:
        s_end = tl.minimum((pid_m + 1) * BLOCK_M, T)
    else:
        s_end = T

    if HAS_WINDOW:
        s_start_q = pid_m * BLOCK_M - window_left
        s_start = tl.maximum(0, s_start_q)
    else:
        s_start = 0
    s_start_block = (s_start // BLOCK_N) * BLOCK_N

    for s_blk in range(s_start_block, s_end, BLOCK_N):
        offs_s = s_blk + offs_n_base
        mask_s = offs_s < T

        # score[m, s] = sm_scale · Σ_j w_k[m, j] · (q @ K[s, j, :]^T)
        # Tried a single fused matmul with q_eff = w_k ⊙ q stacked to (BLOCK_M, J·D)
        # — regressed on A100 because the K=512 contracting dim forced BLOCK_M down
        # to 32 and the lost data reuse swamped the bigger-MMA win. J small dots
        # with BLOCK_M=64 wins on A100; revisit on Hopper where TMA + larger SRAM
        # might flip the tradeoff.
        score = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for j in tl.static_range(J):
            k_ptrs = (
                K + b * sKb + j * sKj
                + offs_s[:, None] * sKt + offs_d[None, :] * sKd
            )
            k = tl.load(k_ptrs, mask=mask_s[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k))
            wk_j = tl.sum(w_k * (offs_j == j).to(tl.float32)[None, :], axis=1)
            score += wk_j[:, None] * qk
        score *= sm_scale

        if CAUSAL:
            score = tl.where(offs_m[:, None] >= offs_s[None, :], score, -float("inf"))
        if HAS_WINDOW:
            score = tl.where(offs_m[:, None] - offs_s[None, :] <= window_left, score, -float("inf"))
        score = tl.where(mask_s[None, :], score, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(score, axis=1))
        # NaN-safe: if every score in the block is -inf, m_new stays -inf and exp(-inf+inf)
        # would be NaN. Guard alpha and p so we cleanly skip such rows/blocks.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        p = tl.exp(score - m_safe[:, None])
        p = tl.where(score == float("-inf"), 0.0, p)
        l_i = alpha * l_i + tl.sum(p, axis=1)

        acc = acc * alpha[:, None]

        if HAS_VE:
            g_ptrs = GATE + b * sGb + h * sGh + offs_s * sGt
            gate = tl.load(g_ptrs, mask=mask_s, other=0.0)

        p_dt = p.to(q.dtype)
        for j in tl.static_range(J):
            v_ptrs = (
                V + b * sVb + j * sVj
                + offs_s[:, None] * sVt + offs_d[None, :] * sVd
            )
            v = tl.load(v_ptrs, mask=mask_s[:, None], other=0.0)
            if HAS_VE:
                ve_ptrs = (
                    VE + b * sVEb + j * sVEj
                    + offs_s[:, None] * sVEt + offs_d[None, :] * sVEd
                )
                ve_val = tl.load(ve_ptrs, mask=mask_s[:, None], other=0.0)
                v = v + (gate[:, None] * ve_val).to(v.dtype)
            upd = tl.dot(p_dt, v)
            wv_j = tl.sum(w_v * (offs_j == j).to(tl.float32)[None, :], axis=1)
            acc += wv_j[:, None] * upd

        m_i = m_new

    # Rows with no in-band keys (l_i == 0): leave output zero. This matches the
    # reference where masked rows get -inf softmax → 0/0 NaN, but we explicitly
    # zero them here; FA-style flash kernels do the same.
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    y = acc / l_safe[:, None]

    o_ptrs = (
        O + b * sOb + h * sOh
        + offs_m[:, None] * sOt + offs_d[None, :] * sOd
    )
    tl.store(o_ptrs, y.to(O.dtype.element_ty), mask=mask_m[:, None])

    if SAVE_L:
        # L[b, t, h] = m_i + log(l_i). For empty rows (l_i==0), store -inf so
        # backward can detect & skip them (exp(score - (-inf)) → 0 cleanly).
        l_log = tl.where(l_i == 0.0, float("-inf"), m_i + tl.log(l_safe))
        l_ptrs = L + b * sLb + h * sLh + offs_m * sLt
        tl.store(l_ptrs, l_log, mask=mask_m)


def bqa_dyn_attn_triton_fwd(
    q: torch.Tensor,
    k_basis: torch.Tensor,
    v_basis: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    ve: Optional[torch.Tensor] = None,
    gate: Optional[torch.Tensor] = None,
    causal: bool = True,
    window_size: Optional[Tuple[int, int]] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    num_stages: Optional[int] = None,
    num_warps: int = 4,
    return_L: bool = False,
) -> torch.Tensor:
    """Forward pass of bqa_dyn attention.

    Args:
        q:        (B, T, H, D), already rotary'd + RMS-normed + scaled.
        k_basis:  (B, T, J, D), already rotary'd + RMS-normed + scaled.
        v_basis:  (B, T, J, D)
        w_k:      (B, T, H, J)   softmax probabilities (in compute dtype)
        w_v:      (B, T, H, J)   softmax probabilities (in compute dtype)
        ve:       (B, T, J, D) or None — value-embedding residual.
        gate:     (B, T, H) or None — per-source-per-head sigmoid·3 gate. Required iff ve is.
        causal:   apply causal mask (default True; non-causal not exercised here).
        window_size: (left, right) — only `left` is used; right==0 is implied (causal).
                     None ⇒ no window. left<0 also means no window.

    Returns:
        y: (B, T, H, D), same dtype as q.
    """
    B, T, H, D = q.shape
    Bk, Tk, J, Dk = k_basis.shape
    assert (B, T, D) == (Bk, Tk, Dk), f"Q/K shape mismatch: {q.shape} vs {k_basis.shape}"
    assert v_basis.shape == k_basis.shape
    assert w_k.shape == (B, T, H, J)
    assert w_v.shape == (B, T, H, J)
    assert q.is_cuda and k_basis.is_cuda
    assert q.dtype == k_basis.dtype == v_basis.dtype, "q/k/v dtypes must match"
    has_ve = ve is not None
    if has_ve:
        assert gate is not None, "gate is required when ve is provided"
        assert ve.shape == k_basis.shape, f"ve shape {ve.shape} != k_basis {k_basis.shape}"
        assert gate.shape == (B, T, H), f"gate shape {gate.shape} != {(B, T, H)}"
        assert ve.dtype == q.dtype and gate.dtype == q.dtype

    if window_size is not None and window_size[0] >= 0 and window_size[0] < T - 1:
        has_window = True
        window_left = int(window_size[0])
    else:
        has_window = False
        window_left = T  # ignored

    o = torch.empty_like(q)
    save_L = return_L
    L = torch.empty(B, T, H, device=q.device, dtype=torch.float32) if save_L else q  # dummy ptr if not saved

    def _strides(t):
        return (t.stride(0), t.stride(1), t.stride(2), t.stride(3))

    sQb, sQt, sQh, sQd = _strides(q)
    sKb, sKt, sKj, sKd = _strides(k_basis)
    sVb, sVt, sVj, sVd = _strides(v_basis)
    sOb, sOt, sOh, sOd = _strides(o)
    sWKb, sWKt, sWKh, sWKj = _strides(w_k)
    sWVb, sWVt, sWVh, sWVj = _strides(w_v)
    if save_L:
        sLb, sLt, sLh = L.stride(0), L.stride(1), L.stride(2)
    else:
        sLb = sLt = sLh = 0
    if has_ve:
        sVEb, sVEt, sVEj, sVEd = _strides(ve)
        sGb, sGt, sGh = gate.stride(0), gate.stride(1), gate.stride(2)
    else:
        sVEb = sVEt = sVEj = sVEd = 0
        sGb = sGt = sGh = 0

    # Triton requires power-of-two block sizes for tl.dot. D must be PoT.
    assert (D & (D - 1)) == 0 and D >= 16, f"head_dim must be PoT >= 16, got {D}"

    # Block-size defaults tuned for A100's 164KB shared-mem limit.
    #   bf16 / no-ve : 64×64, num_stages=2 (best — pipelined, large reductions)
    #   bf16 / ve    : 64×32, num_stages=2 (extra ve tile per iter; shrink BLOCK_N
    #                                       to keep pipelining)
    #   fp32         : 32×32, num_stages=1 (every tile doubles vs bf16)
    if block_m is None or block_n is None:
        if q.dtype == torch.float32:
            block_m = block_m or 32
            block_n = block_n or 32
        elif has_ve:
            block_m = block_m or 64
            block_n = block_n or 32
        else:
            block_m = block_m or 64
            block_n = block_n or 64
    if num_stages is None:
        num_stages = 1 if q.dtype == torch.float32 else 2

    assert (block_m & (block_m - 1)) == 0
    assert (block_n & (block_n - 1)) == 0

    sm_scale = 1.0 / (D ** 0.5)

    # tl.arange requires power-of-two; pad J to next pow2 for offs_j.
    J_pad = 1
    while J_pad < J:
        J_pad *= 2

    grid = (triton.cdiv(T, block_m), B * H)

    _bqa_dyn_attn_fwd_kernel[grid](
        q, k_basis, v_basis,
        ve if has_ve else q,            # dummy pointer when not used
        gate if has_ve else q,
        w_k, w_v, o, L,
        sQb, sQt, sQh, sQd,
        sKb, sKt, sKj, sKd,
        sVb, sVt, sVj, sVd,
        sVEb, sVEt, sVEj, sVEd,
        sGb, sGt, sGh,
        sWKb, sWKt, sWKh, sWKj,
        sWVb, sWVt, sWVh, sWVj,
        sOb, sOt, sOh, sOd,
        sLb, sLt, sLh,
        sm_scale,
        H,
        T,
        window_left,
        HAS_VE=has_ve,
        CAUSAL=causal,
        HAS_WINDOW=has_window,
        SAVE_L=save_L,
        J=J,
        J_PAD=J_pad,
        D=D,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if return_L:
        return o, L
    return o


# =====================================================================
# Backward kernels — FlashAttention-style two-pass split, NO ATOMICS.
# =====================================================================
#
# Atomic-heavy backwards (one program per query block, atomic-adding every key-block
# contribution into dK/dV/dVE/dgate) bottleneck on shared HBM lines once H>1 — A100
# benchmarks showed 2.5-6× regression vs the SDPA-autograd path. The standard fix is
# the FA split: separate kernels for dQ-side and dKV-side gradients, each with the
# loop axis chosen so the program owns its output rows.
#
# dQ kernel: program per (b, h, m_block). Owns dq[b,h,m_block,:], dw_k, dw_v.
# dKV kernel: program per (b, h, n_block). Owns dK_per_h[b,h,n_block,j,:],
#             dV_per_h, dVE_per_h, dgate[b,n_block,h].
#
# The wrapper sums dK_per_h / dV_per_h / dVE_per_h across the H axis in PyTorch
# (single fused reduction; trivial cost) to recover the basis-shaped dK/dV/dVE.
# Memory cost of the staging buffers is H× the basis size in fp32 — for d12
# train shapes (B=32, H=6, T=2048, J=3, D=128) ≈ 0.6 GB per tensor, ≈ 1.8 GB total.


@triton.jit
def _bqa_dyn_attn_bwd_dq_kernel(
    Q, K, V, VE, GATE, WK, WV,
    DY, L_IN, Z_AUX,
    DQ, DW_K, DW_V,
    sQb, sQt, sQh, sQd,
    sKb, sKt, sKj, sKd,
    sVb, sVt, sVj, sVd,
    sVEb, sVEt, sVEj, sVEd,
    sGb, sGt, sGh,
    sWKb, sWKt, sWKh, sWKj,
    sWVb, sWVt, sWVh, sWVj,
    sDYb, sDYt, sDYh, sDYd,
    sLb, sLt, sLh,
    sZb, sZt, sZh,
    sDQb, sDQt, sDQh, sDQd,
    sDWKb, sDWKt, sDWKh, sDWKj,
    sDWVb, sDWVt, sDWVh, sDWVj,
    sm_scale,
    H,
    T,
    window_left,
    HAS_VE: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    J: tl.constexpr,
    J_PAD: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_n_base = tl.arange(0, BLOCK_N)
    mask_m = offs_m < T

    q_ptrs = Q + b * sQb + h * sQh + offs_m[:, None] * sQt + offs_d[None, :] * sQd
    dy_ptrs = DY + b * sDYb + h * sDYh + offs_m[:, None] * sDYt + offs_d[None, :] * sDYd
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    dy = tl.load(dy_ptrs, mask=mask_m[:, None], other=0.0)

    L_ptr = L_IN + b * sLb + h * sLh + offs_m * sLt
    Z_ptr = Z_AUX + b * sZb + h * sZh + offs_m * sZt
    L_v = tl.load(L_ptr, mask=mask_m, other=0.0)
    Z_v = tl.load(Z_ptr, mask=mask_m, other=0.0)
    L_finite = L_v != float("-inf")
    L_safe = tl.where(L_finite, L_v, 0.0)

    # Load w_k, w_v as J separate columns directly — avoids the (BLOCK_M, J_PAD)
    # tile + j-onehot-mask trick (which compiles into a multiply+reduce per scalar
    # and is significantly slower in practice).
    dq_acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    # dw_k, dw_v stored as (BLOCK_M, J_PAD) for the final write.
    dw_k_acc = tl.zeros([BLOCK_M, J_PAD], dtype=tl.float32)
    dw_v_acc = tl.zeros([BLOCK_M, J_PAD], dtype=tl.float32)

    if CAUSAL:
        s_end = tl.minimum((pid_m + 1) * BLOCK_M, T)
    else:
        s_end = T
    if HAS_WINDOW:
        s_start = tl.maximum(0, pid_m * BLOCK_M - window_left)
    else:
        s_start = 0
    s_start_block = (s_start // BLOCK_N) * BLOCK_N

    offs_j_pad = tl.arange(0, J_PAD)

    for s_blk in range(s_start_block, s_end, BLOCK_N):
        offs_s = s_blk + offs_n_base
        mask_s = offs_s < T

        # Score recomp. Save qk_j for use in Phase 4 below — for J=3 / BLOCK_M=32
        # / BLOCK_N=32 the J tiles are 12KB total; fits comfortably and removes
        # the need to reload K_j twice.
        score = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk_0 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk_1 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk_2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk_3 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for j in tl.static_range(J):
            k_ptrs = K + b * sKb + j * sKj + offs_s[:, None] * sKt + offs_d[None, :] * sKd
            k = tl.load(k_ptrs, mask=mask_s[:, None], other=0.0)
            qk_j = tl.dot(q, tl.trans(k))
            wk_j_ptrs = WK + b * sWKb + h * sWKh + offs_m * sWKt + j * sWKj
            wk_j = tl.load(wk_j_ptrs, mask=mask_m, other=0.0).to(tl.float32)
            score += wk_j[:, None] * qk_j
            if j == 0:
                qk_0 = qk_j
            if j == 1:
                qk_1 = qk_j
            if j == 2:
                qk_2 = qk_j
            if j == 3:
                qk_3 = qk_j
        score *= sm_scale

        if CAUSAL:
            score = tl.where(offs_m[:, None] >= offs_s[None, :], score, -float("inf"))
        if HAS_WINDOW:
            score = tl.where(offs_m[:, None] - offs_s[None, :] <= window_left, score, -float("inf"))
        score = tl.where(mask_s[None, :], score, -float("inf"))

        p = tl.exp(score - L_safe[:, None])
        p = tl.where(L_finite[:, None], p, 0.0)

        if HAS_VE:
            g_ptrs = GATE + b * sGb + h * sGh + offs_s * sGt
            gate_n = tl.load(g_ptrs, mask=mask_s, other=0.0).to(tl.float32)

        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for j in tl.static_range(J):
            v_ptrs = V + b * sVb + j * sVj + offs_s[:, None] * sVt + offs_d[None, :] * sVd
            v = tl.load(v_ptrs, mask=mask_s[:, None], other=0.0).to(tl.float32)
            if HAS_VE:
                ve_ptrs = VE + b * sVEb + j * sVEj + offs_s[:, None] * sVEt + offs_d[None, :] * sVEd
                ve_val = tl.load(ve_ptrs, mask=mask_s[:, None], other=0.0).to(tl.float32)
                v_eff = v + gate_n[:, None] * ve_val
            else:
                v_eff = v
            vdy_j = tl.dot(dy, tl.trans(v_eff.to(dy.dtype)))
            wv_j_ptrs = WV + b * sWVb + h * sWVh + offs_m * sWVt + j * sWVj
            wv_j = tl.load(wv_j_ptrs, mask=mask_m, other=0.0).to(tl.float32)
            dp += wv_j[:, None] * vdy_j
            dw_v_col = tl.sum(p * vdy_j, axis=1)  # (BLOCK_M,)
            # Scatter into the j-th slot of dw_v_acc.
            dw_v_acc += dw_v_col[:, None] * (offs_j_pad == j).to(tl.float32)[None, :]

        dscore = p * (dp - Z_v[:, None])
        dscore_dt = dscore.to(q.dtype)

        for j in tl.static_range(J):
            # Reload K_j for the dq matmul (we need K, not just qk).
            k_ptrs = K + b * sKb + j * sKj + offs_s[:, None] * sKt + offs_d[None, :] * sKd
            k = tl.load(k_ptrs, mask=mask_s[:, None], other=0.0)

            wk_j_ptrs = WK + b * sWKb + h * sWKh + offs_m * sWKt + j * sWKj
            wk_j_col = tl.load(wk_j_ptrs, mask=mask_m, other=0.0).to(tl.float32)

            # Reuse qk_j saved from Phase 1.
            qk_j = qk_0
            if j == 1:
                qk_j = qk_1
            if j == 2:
                qk_j = qk_2
            if j == 3:
                qk_j = qk_3

            dw_k_col = tl.sum(dscore * qk_j, axis=1) * sm_scale
            dw_k_acc += dw_k_col[:, None] * (offs_j_pad == j).to(tl.float32)[None, :]

            dq_part = tl.dot(dscore_dt, k)
            dq_acc += sm_scale * wk_j_col[:, None] * dq_part

    dq_ptrs = DQ + b * sDQb + h * sDQh + offs_m[:, None] * sDQt + offs_d[None, :] * sDQd
    tl.store(dq_ptrs, dq_acc.to(DQ.dtype.element_ty), mask=mask_m[:, None])

    mask_j = offs_j_pad < J
    dwk_ptrs = (
        DW_K + b * sDWKb + h * sDWKh
        + offs_m[:, None] * sDWKt + offs_j_pad[None, :] * sDWKj
    )
    dwv_ptrs = (
        DW_V + b * sDWVb + h * sDWVh
        + offs_m[:, None] * sDWVt + offs_j_pad[None, :] * sDWVj
    )
    store_mask = mask_m[:, None] & mask_j[None, :]
    tl.store(dwk_ptrs, dw_k_acc.to(DW_K.dtype.element_ty), mask=store_mask)
    tl.store(dwv_ptrs, dw_v_acc.to(DW_V.dtype.element_ty), mask=store_mask)


@triton.jit
def _bqa_dyn_attn_bwd_dkv_kernel(
    Q, K, V, VE, GATE, WK, WV,
    DY, L_IN, Z_AUX,
    DK_PH, DV_PH, DVE_PH, DGATE,
    sQb, sQt, sQh, sQd,
    sKb, sKt, sKj, sKd,
    sVb, sVt, sVj, sVd,
    sVEb, sVEt, sVEj, sVEd,
    sGb, sGt, sGh,
    sWKb, sWKt, sWKh, sWKj,
    sWVb, sWVt, sWVh, sWVj,
    sDYb, sDYt, sDYh, sDYd,
    sLb, sLt, sLh,
    sZb, sZt, sZh,
    sDKPb, sDKPh, sDKPt, sDKPj, sDKPd,
    sDVPb, sDVPh, sDVPt, sDVPj, sDVPd,
    sDVEPb, sDVEPh, sDVEPt, sDVEPj, sDVEPd,
    sDGb, sDGt, sDGh,
    sm_scale,
    H,
    T,
    window_left,
    HAS_VE: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    J: tl.constexpr,
    J_PAD: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Per (b, h, n_block) program. Walks query blocks; owns its (b,h,n_block) slice
    of dK_per_h, dV_per_h, dVE_per_h, dgate (no atomics, single writer)."""
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_m_base = tl.arange(0, BLOCK_M)
    mask_n = offs_n < T

    if HAS_VE:
        g_ptrs_n = GATE + b * sGb + h * sGh + offs_n * sGt
        gate_n = tl.load(g_ptrs_n, mask=mask_n, other=0.0).to(tl.float32)

    # Per-(b, h, n_block, j) accumulators stored as separate Triton tiles. Triton
    # SSA-tracks these across the m loop. We reserve up to J=4 slots; static_range
    # writes only the first J. Using a 3D tile here was 3-4× slower (the broadcast
    # multiply against a J-onehot produces wide ops that the compiler doesn't vectorize).
    dK_0 = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dK_1 = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dK_2 = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dK_3 = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dV_0 = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dV_1 = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dV_2 = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dV_3 = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    if CAUSAL:
        m_start = pid_n * BLOCK_N
    else:
        m_start = 0
    m_start_block = (m_start // BLOCK_M) * BLOCK_M
    if HAS_WINDOW:
        m_end_q = (pid_n + 1) * BLOCK_N + window_left
        m_end = tl.minimum(m_end_q, T)
    else:
        m_end = T

    for m_blk in range(m_start_block, m_end, BLOCK_M):
        offs_m = m_blk + offs_m_base
        mask_m = offs_m < T

        q_ptrs = Q + b * sQb + h * sQh + offs_m[:, None] * sQt + offs_d[None, :] * sQd
        dy_ptrs = DY + b * sDYb + h * sDYh + offs_m[:, None] * sDYt + offs_d[None, :] * sDYd
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        dy = tl.load(dy_ptrs, mask=mask_m[:, None], other=0.0)

        L_ptr = L_IN + b * sLb + h * sLh + offs_m * sLt
        Z_ptr = Z_AUX + b * sZb + h * sZh + offs_m * sZt
        L_v = tl.load(L_ptr, mask=mask_m, other=0.0)
        Z_v = tl.load(Z_ptr, mask=mask_m, other=0.0)
        L_finite = L_v != float("-inf")
        L_safe = tl.where(L_finite, L_v, 0.0)

        # Recompute score (per-j w_k loaded as a column).
        score = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for j in tl.static_range(J):
            k_ptrs = K + b * sKb + j * sKj + offs_n[:, None] * sKt + offs_d[None, :] * sKd
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            qk_j = tl.dot(q, tl.trans(k))
            wk_j_ptrs = WK + b * sWKb + h * sWKh + offs_m * sWKt + j * sWKj
            wk_j = tl.load(wk_j_ptrs, mask=mask_m, other=0.0).to(tl.float32)
            score += wk_j[:, None] * qk_j
        score *= sm_scale

        if CAUSAL:
            score = tl.where(offs_m[:, None] >= offs_n[None, :], score, -float("inf"))
        if HAS_WINDOW:
            score = tl.where(offs_m[:, None] - offs_n[None, :] <= window_left, score, -float("inf"))
        score = tl.where(mask_n[None, :], score, -float("inf"))
        score = tl.where(mask_m[:, None], score, -float("inf"))

        p = tl.exp(score - L_safe[:, None])
        p = tl.where(L_finite[:, None], p, 0.0)

        # dp
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for j in tl.static_range(J):
            v_ptrs = V + b * sVb + j * sVj + offs_n[:, None] * sVt + offs_d[None, :] * sVd
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
            if HAS_VE:
                ve_ptrs = VE + b * sVEb + j * sVEj + offs_n[:, None] * sVEt + offs_d[None, :] * sVEd
                ve_val = tl.load(ve_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
                v_eff = v + gate_n[:, None] * ve_val
            else:
                v_eff = v
            vdy_j = tl.dot(dy, tl.trans(v_eff.to(dy.dtype)))
            wv_j_ptrs = WV + b * sWVb + h * sWVh + offs_m * sWVt + j * sWVj
            wv_j = tl.load(wv_j_ptrs, mask=mask_m, other=0.0).to(tl.float32)
            dp += wv_j[:, None] * vdy_j

        dscore = p * (dp - Z_v[:, None])
        dscore_dt = dscore.to(q.dtype)

        # Accumulate dK_j, dV_eff_j contributions for this m_block.
        for j in tl.static_range(J):
            wk_j_ptrs = WK + b * sWKb + h * sWKh + offs_m * sWKt + j * sWKj
            wk_j_col = tl.load(wk_j_ptrs, mask=mask_m, other=0.0).to(tl.float32)
            wv_j_ptrs = WV + b * sWVb + h * sWVh + offs_m * sWVt + j * sWVj
            wv_j_col = tl.load(wv_j_ptrs, mask=mask_m, other=0.0).to(tl.float32)
            dscore_wk = (dscore_dt * wk_j_col[:, None]).to(q.dtype)
            dK_block = tl.dot(tl.trans(dscore_wk), q) * sm_scale
            p_wv = (p * wv_j_col[:, None]).to(dy.dtype)
            dV_eff_block = tl.dot(tl.trans(p_wv), dy)

            if j == 0:
                dK_0 += dK_block
                dV_0 += dV_eff_block
            if j == 1:
                dK_1 += dK_block
                dV_1 += dV_eff_block
            if j == 2:
                dK_2 += dK_block
                dV_2 += dV_eff_block
            if j == 3:
                dK_3 += dK_block
                dV_3 += dV_eff_block

    # Tail: store dK_per_h[b, h, n_block, j, :], dV_per_h, dVE_per_h, dgate.
    if HAS_VE:
        dgate_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for j in tl.static_range(J):
        if j == 0:
            dK_j = dK_0
            dV_eff_j = dV_0
        if j == 1:
            dK_j = dK_1
            dV_eff_j = dV_1
        if j == 2:
            dK_j = dK_2
            dV_eff_j = dV_2
        if j == 3:
            dK_j = dK_3
            dV_eff_j = dV_3

        dK_ptrs = (
            DK_PH + b * sDKPb + h * sDKPh + j * sDKPj
            + offs_n[:, None] * sDKPt + offs_d[None, :] * sDKPd
        )
        tl.store(dK_ptrs, dK_j.to(DK_PH.dtype.element_ty), mask=mask_n[:, None])

        dV_ptrs = (
            DV_PH + b * sDVPb + h * sDVPh + j * sDVPj
            + offs_n[:, None] * sDVPt + offs_d[None, :] * sDVPd
        )
        tl.store(dV_ptrs, dV_eff_j.to(DV_PH.dtype.element_ty), mask=mask_n[:, None])

        if HAS_VE:
            dVE_j = gate_n[:, None] * dV_eff_j
            dVE_ptrs = (
                DVE_PH + b * sDVEPb + h * sDVEPh + j * sDVEPj
                + offs_n[:, None] * sDVEPt + offs_d[None, :] * sDVEPd
            )
            tl.store(dVE_ptrs, dVE_j.to(DVE_PH.dtype.element_ty), mask=mask_n[:, None])
            ve_ptrs = VE + b * sVEb + j * sVEj + offs_n[:, None] * sVEt + offs_d[None, :] * sVEd
            ve_val = tl.load(ve_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
            dgate_acc += tl.sum(dV_eff_j * ve_val, axis=1)

    if HAS_VE:
        dG_ptrs = DGATE + b * sDGb + h * sDGh + offs_n * sDGt
        tl.store(dG_ptrs, dgate_acc.to(DGATE.dtype.element_ty), mask=mask_n)


def bqa_dyn_attn_triton_bwd(
    dy: torch.Tensor,
    q: torch.Tensor,
    k_basis: torch.Tensor,
    v_basis: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    o: torch.Tensor,
    L: torch.Tensor,
    ve: Optional[torch.Tensor] = None,
    gate: Optional[torch.Tensor] = None,
    causal: bool = True,
    window_size: Optional[Tuple[int, int]] = None,
    dq_block_m: Optional[int] = None,
    dq_block_n: Optional[int] = None,
    dkv_block_m: Optional[int] = None,
    dkv_block_n: Optional[int] = None,
    num_stages: Optional[int] = None,
    num_warps: int = 4,
):
    """Backward pass via two-kernel FA-split. No atomics.

    Returns (dq, dk_basis, dv_basis, dw_k, dw_v, dve, dgate). `dve`/`dgate` are
    None when ve was None in forward. All gradients are in q's dtype.

    The kernels write per-head intermediates dK_per_h / dV_per_h / dVE_per_h,
    summed across H in PyTorch. Memory cost: H× the basis tensor in fp32.
    """
    B, T, H, D = q.shape
    _, _, J, _ = k_basis.shape
    has_ve = ve is not None

    # Backward kernels hardcode up to 4 J accumulators (separate Triton tile
    # variables, since 3D acc tiles are 3-4× slower in practice). For larger J
    # the kernel needs more slots — extend by adding `if j == k: ...` clauses.
    assert J <= 4, f"backward kernels currently support J <= 4 (got J={J})"

    # Z[b, t, h] = <dy, y>_d  ("Δ" in FA backward).
    Z = (dy.float() * o.float()).sum(dim=-1)  # (B, T, H), fp32

    if window_size is not None and window_size[0] >= 0 and window_size[0] < T - 1:
        has_window = True
        window_left = int(window_size[0])
    else:
        has_window = False
        window_left = T

    # dQ-pass outputs (per-program, no atomics).
    dq_out = torch.empty_like(q)
    dw_k_out = torch.empty_like(w_k)
    dw_v_out = torch.empty_like(w_v)

    # dKV-pass per-head intermediates (fp32 for safe summation across H later).
    dK_per_h = torch.empty((B, H, T, J, D), device=q.device, dtype=torch.float32)
    dV_per_h = torch.empty((B, H, T, J, D), device=q.device, dtype=torch.float32)
    if has_ve:
        dVE_per_h = torch.empty((B, H, T, J, D), device=q.device, dtype=torch.float32)
        dgate_fp32 = torch.empty((B, T, H), device=q.device, dtype=torch.float32)
    else:
        dVE_per_h = q       # dummy ptr
        dgate_fp32 = q

    def _strides4(t):
        return (t.stride(0), t.stride(1), t.stride(2), t.stride(3))

    sQb, sQt, sQh, sQd = _strides4(q)
    sKb, sKt, sKj, sKd = _strides4(k_basis)
    sVb, sVt, sVj, sVd = _strides4(v_basis)
    sWKb, sWKt, sWKh, sWKj = _strides4(w_k)
    sWVb, sWVt, sWVh, sWVj = _strides4(w_v)
    sDYb, sDYt, sDYh, sDYd = _strides4(dy)
    sDQb, sDQt, sDQh, sDQd = _strides4(dq_out)
    sDWKb, sDWKt, sDWKh, sDWKj = _strides4(dw_k_out)
    sDWVb, sDWVt, sDWVh, sDWVj = _strides4(dw_v_out)
    sLb, sLt, sLh = L.stride(0), L.stride(1), L.stride(2)
    sZb, sZt, sZh = Z.stride(0), Z.stride(1), Z.stride(2)
    if has_ve:
        sVEb, sVEt, sVEj, sVEd = _strides4(ve)
        sGb, sGt, sGh = gate.stride(0), gate.stride(1), gate.stride(2)
        sDKPb, sDKPh, sDKPt, sDKPj, sDKPd = dK_per_h.stride()
        sDVPb, sDVPh, sDVPt, sDVPj, sDVPd = dV_per_h.stride()
        sDVEPb, sDVEPh, sDVEPt, sDVEPj, sDVEPd = dVE_per_h.stride()
        sDGb, sDGt, sDGh = dgate_fp32.stride(0), dgate_fp32.stride(1), dgate_fp32.stride(2)
    else:
        sVEb = sVEt = sVEj = sVEd = 0
        sGb = sGt = sGh = 0
        sDKPb, sDKPh, sDKPt, sDKPj, sDKPd = dK_per_h.stride()
        sDVPb, sDVPh, sDVPt, sDVPj, sDVPd = dV_per_h.stride()
        sDVEPb = sDVEPh = sDVEPt = sDVEPj = sDVEPd = 0
        sDGb = sDGt = sDGh = 0

    assert (D & (D - 1)) == 0 and D >= 16, f"head_dim must be PoT >= 16, got {D}"

    # Block defaults — small tiles keep shared mem headroom for the many in-flight
    # tiles backward needs (q, dy, score, p, dp, dscore, accumulators…).
    if dq_block_m is None:
        dq_block_m = 32 if q.dtype == torch.float32 else 32
    if dq_block_n is None:
        dq_block_n = 16 if q.dtype == torch.float32 else 32
    if dkv_block_m is None:
        dkv_block_m = 32 if q.dtype == torch.float32 else 32
    if dkv_block_n is None:
        dkv_block_n = 16 if q.dtype == torch.float32 else 32
    if num_stages is None:
        num_stages = 1 if q.dtype == torch.float32 else 2
    for blk in (dq_block_m, dq_block_n, dkv_block_m, dkv_block_n):
        assert (blk & (blk - 1)) == 0

    sm_scale = 1.0 / (D ** 0.5)

    J_pad = 1
    while J_pad < J:
        J_pad *= 2

    # ---- dQ pass ----
    dq_grid = (triton.cdiv(T, dq_block_m), B * H)
    _bqa_dyn_attn_bwd_dq_kernel[dq_grid](
        q, k_basis, v_basis,
        ve if has_ve else q,
        gate if has_ve else q,
        w_k, w_v, dy, L, Z,
        dq_out, dw_k_out, dw_v_out,
        sQb, sQt, sQh, sQd,
        sKb, sKt, sKj, sKd,
        sVb, sVt, sVj, sVd,
        sVEb, sVEt, sVEj, sVEd,
        sGb, sGt, sGh,
        sWKb, sWKt, sWKh, sWKj,
        sWVb, sWVt, sWVh, sWVj,
        sDYb, sDYt, sDYh, sDYd,
        sLb, sLt, sLh,
        sZb, sZt, sZh,
        sDQb, sDQt, sDQh, sDQd,
        sDWKb, sDWKt, sDWKh, sDWKj,
        sDWVb, sDWVt, sDWVh, sDWVj,
        sm_scale,
        H,
        T,
        window_left,
        HAS_VE=has_ve,
        CAUSAL=causal,
        HAS_WINDOW=has_window,
        J=J,
        J_PAD=J_pad,
        D=D,
        BLOCK_M=dq_block_m,
        BLOCK_N=dq_block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # ---- dKV pass ----
    dkv_grid = (triton.cdiv(T, dkv_block_n), B * H)
    _bqa_dyn_attn_bwd_dkv_kernel[dkv_grid](
        q, k_basis, v_basis,
        ve if has_ve else q,
        gate if has_ve else q,
        w_k, w_v, dy, L, Z,
        dK_per_h, dV_per_h,
        dVE_per_h if has_ve else q,
        dgate_fp32 if has_ve else q,
        sQb, sQt, sQh, sQd,
        sKb, sKt, sKj, sKd,
        sVb, sVt, sVj, sVd,
        sVEb, sVEt, sVEj, sVEd,
        sGb, sGt, sGh,
        sWKb, sWKt, sWKh, sWKj,
        sWVb, sWVt, sWVh, sWVj,
        sDYb, sDYt, sDYh, sDYd,
        sLb, sLt, sLh,
        sZb, sZt, sZh,
        sDKPb, sDKPh, sDKPt, sDKPj, sDKPd,
        sDVPb, sDVPh, sDVPt, sDVPj, sDVPd,
        sDVEPb, sDVEPh, sDVEPt, sDVEPj, sDVEPd,
        sDGb, sDGt, sDGh,
        sm_scale,
        H,
        T,
        window_left,
        HAS_VE=has_ve,
        CAUSAL=causal,
        HAS_WINDOW=has_window,
        J=J,
        J_PAD=J_pad,
        D=D,
        BLOCK_M=dkv_block_m,
        BLOCK_N=dkv_block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # Sum per-head intermediates over H to recover the basis-shaped grads.
    dk_basis = dK_per_h.sum(dim=1).to(q.dtype)
    dv_basis = dV_per_h.sum(dim=1).to(q.dtype)
    if has_ve:
        dve_out = dVE_per_h.sum(dim=1).to(q.dtype)
        dgate_out = dgate_fp32.to(q.dtype)
    else:
        dve_out = None
        dgate_out = None
    return dq_out, dk_basis, dv_basis, dw_k_out, dw_v_out, dve_out, dgate_out


# =====================================================================
# torch.autograd.Function wrapper — drop-in for the explicit/SDPA path
# =====================================================================


class _BqaDynAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k_basis, v_basis, w_k, w_v, ve, gate, causal, window_size):
        out, L = bqa_dyn_attn_triton_fwd(
            q, k_basis, v_basis, w_k, w_v,
            ve=ve, gate=gate,
            causal=causal, window_size=window_size,
            return_L=True,
        )
        ctx.save_for_backward(q, k_basis, v_basis, w_k, w_v, ve, gate, out, L)
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.has_ve = ve is not None
        return out

    @staticmethod
    def backward(ctx, dy):
        q, k_basis, v_basis, w_k, w_v, ve, gate, out, L = ctx.saved_tensors
        # If ve is None it's stored as None; saved_tensors handles that.
        dy = dy.contiguous()
        dq, dk, dv, dwk, dwv, dve, dgate = bqa_dyn_attn_triton_bwd(
            dy, q, k_basis, v_basis, w_k, w_v, out, L,
            ve=ve if ctx.has_ve else None,
            gate=gate if ctx.has_ve else None,
            causal=ctx.causal,
            window_size=ctx.window_size,
        )
        return dq, dk, dv, dwk, dwv, dve, dgate, None, None


def bqa_dyn_attn(
    q: torch.Tensor,
    k_basis: torch.Tensor,
    v_basis: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    ve: Optional[torch.Tensor] = None,
    gate: Optional[torch.Tensor] = None,
    causal: bool = True,
    window_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Autograd-wrapped bqa_dyn attention. Backward via Triton kernel.

    See `bqa_dyn_attn_triton_fwd` for argument semantics. Only the kwargs above
    are supported (block sizes/staging are auto-picked).
    """
    return _BqaDynAttnFn.apply(q, k_basis, v_basis, w_k, w_v, ve, gate, causal, window_size)
