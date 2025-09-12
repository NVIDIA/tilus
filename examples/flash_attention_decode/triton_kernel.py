# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0_source"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_gated_delta_rule_update_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    {
        'q': 'torch.Size([1, 1, 4, 128])',
        'k': 'torch.Size([1, 1, 4, 128])',
        'v': 'torch.Size([1, 1, 8, 128])',
        'g': 'torch.Size([1, 1, 8])',
        'beta': 'torch.Size([1, 1, 8])',
        'scale': None,
        'initial_state_source': 'torch.Size([129, 8, 128, 128])',
        'initial_state_indices': 'torch.Size([1])',
        'cu_seqlens': 'torch.Size([2])',
        'use_qk_l2norm_in_kernel': True
    }
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
    p_g = g + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        # Add bounds checking for idx
        if idx >= 0:  # Assuming negative indices are invalid
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q)) + 1e-6)
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k)) + 1e-6)
        b_q = b_q * scale
        # [BK, BV]
        b_h *= tl.exp(b_g)
        # [BV]
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        # [BK, BV]
        b_h += b_k[:, None] * b_v[None, :]
        # [BV]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_g += HV
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)

    # Store final state back to h0_source with bounds checking
    idx = tl.load(h0_indices + i_n)
    if idx >= 0:  # Add bounds checking
        p_h0 = (
            h0_source + idx * HV * K * V + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        )
        tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def fused_recurrent_gated_delta_rule_update_fwd_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Example Inputs:
        q: dtype: torch.bfloat16, shape: torch.Size([1, 1, 4, 128])
        k: dtype: torch.bfloat16, shape: torch.Size([1, 1, 4, 128])
        v: dtype: torch.bfloat16, shape: torch.Size([1, 1, 8, 128])
        g: dtype: torch.float32, shape: torch.Size([1, 1, 8])
        beta: dtype: torch.bfloat16, shape: torch.Size([1, 1, 8])
        scale: None
        initial_state_source: dtype: torch.float32, shape: torch.Size([129, 8, 128, 128])
        initial_state_indices: tensor([126], device='cuda:1', dtype=torch.int32)
        cu_seqlens: tensor([0, 1], device='cuda:1', dtype=torch.int32)
        use_qk_l2norm_in_kernel: True
    """
    B, T, H, K, V = *k.shape, v.shape[-1]  # B=1, T=1, H=4, K=V=128
    HV = v.shape[2]  # HV=8
    N = B if cu_seqlens is None else len(cu_seqlens) - 1  # N=1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)  # BK=128, BV=8
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)  # NK=1, NV=16
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    o = q.new_empty(NK, *v.shape)  # [1, 1, 1, 8, 128]
    grid = (NK, NV, N * HV)  # (1, 16, 8)

    fused_recurrent_gated_delta_rule_update_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o


# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
@triton.jit
def fused_gdn_gating_kernel(
    g,
    A_log,
    a,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    # Handle both 2D (batch, num_heads) and 3D (batch, seq_len, num_heads) inputs
    if a.ndim == 2:
        batch, num_heads = a.shape
        seq_len = 1
    elif a.ndim == 3:
        batch, seq_len, num_heads = a.shape
    else:
        raise ValueError(f"Expected 2D or 3D input tensor, got {a.ndim}D")

    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty_like(a, dtype=torch.float32)
    fused_gdn_gating_kernel[grid](
        g, A_log, a, dt_bias, seq_len, num_heads, beta, threshold, 8, num_warps=1
    )
    return g


def sigmoid_gating_delta_rule_update_triton(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    scale: float,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    beta = b.sigmoid()
    g = fused_gdn_gating(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        beta=softplus_beta,
        threshold=softplus_threshold,
    )
    o = fused_recurrent_gated_delta_rule_update_fwd_triton(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
    )
    return o
