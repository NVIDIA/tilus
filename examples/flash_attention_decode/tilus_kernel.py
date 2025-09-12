# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math
from typing import Optional

import tilus
import torch
from tilus import bfloat16, float32, int32
from tilus.utils import cdiv


@tilus.autotune("num_warps", [1])
@tilus.autotune("block_heads", [4])
class FusedGdnGatingKernel(tilus.Script):
    def __init__(self, num_warps: int, block_heads: int):
        super().__init__()
        self.num_warps = num_warps
        self.block_heads = block_heads

    def __call__(
        self,
        g_ptr: ~float32,
        A_log_ptr: ~float32,
        a_ptr: ~bfloat16,
        dt_bias_ptr: ~float32,
        batch: int,
        num_heads: int,
        beta: float,
        threshold: float,
    ):
        self.attrs.warps = self.num_warps
        self.attrs.blocks = (batch, cdiv(num_heads, self.block_heads))

        # Create global views
        g_g = self.global_view(g_ptr, dtype=float32, shape=[batch, num_heads])
        g_A_log = self.global_view(A_log_ptr, dtype=float32, shape=[num_heads])
        g_a = self.global_view(a_ptr, dtype=bfloat16, shape=[batch, num_heads])
        g_dt_bias = self.global_view(dt_bias_ptr, dtype=float32, shape=[num_heads])

        # Get block indices
        i_batch = self.blockIdx.x
        i_head_block = self.blockIdx.y

        # Load a slice of heads for this block
        head_offset = i_head_block * self.block_heads

        # Load inputs as tensors
        r_A_log = self.load_global(
            g_A_log, offsets=[head_offset], shape=[self.block_heads], dims=[0]
        ).to(float32)
        r_a = self.load_global(
            g_a, offsets=[i_batch, head_offset], shape=[self.block_heads], dims=[1]
        ).to(float32)
        r_dt_bias = self.load_global(
            g_dt_bias, offsets=[head_offset], shape=[self.block_heads], dims=[0]
        ).to(float32)

        # Compute x = a + dt_bias
        r_x = r_a + r_dt_bias

        # Apply softplus with beta and threshold for numerical stability
        r_beta_x = r_x * beta

        # For numerical stability, we'll compute both branches and let the hardware handle it
        # This is a simplified version that should work with Tilus
        r_exp_beta_x = self.exp(r_beta_x)
        r_log_term = self.log(r_exp_beta_x + 1.0)
        r_softplus = self.where(r_x >= threshold, r_x, r_log_term * (1.0 / beta))

        # Compute final result: g = -exp(A_log) * softplus_x
        r_result = -self.exp(r_A_log) * r_softplus

        # Store result
        self.store_global(g_g, r_result, offsets=[i_batch, head_offset], dims=[1])


@tilus.autotune("num_warps", [1, 2, 4])
@tilus.autotune("BV", [2, 4, 8, 16, 32, 64, 128])
class FusedRecurrentGatedDeltaRuleUpdateFwdKernel(tilus.Script):
    def __init__(self, num_warps: int, BV: int):
        super().__init__()
        self.num_warps = num_warps
        self.BV = BV

    def __call__(
        self,
        q_ptr: ~bfloat16,
        k_ptr: ~bfloat16,
        v_ptr: ~bfloat16,
        g_ptr: ~float32,
        o_ptr: ~bfloat16,
        beta_ptr: ~bfloat16,
        scale: float,
        initial_state_source_ptr: ~float32,
        initial_state_indices_ptr: ~int32,
        T: int32,
        B: int,
        H: int,
        HV: int,
        K: int,
        V: int,
        MAX_T: int,
        USE_INITIAL_STATE: bool,
        USE_QK_L2NORM_IN_KERNEL: bool,
    ):
        self.attrs.warps = self.num_warps
        self.attrs.blocks = (T, cdiv(V, self.BV), HV)

        g_q = self.global_view(q_ptr, dtype=bfloat16, shape=[B, T, H, K])
        g_k = self.global_view(k_ptr, dtype=bfloat16, shape=[B, T, H, K])
        g_v = self.global_view(v_ptr, dtype=bfloat16, shape=[B, T, HV, V])
        g_g = self.global_view(g_ptr, dtype=float32, shape=[B, T, HV])
        g_o = self.global_view(o_ptr, dtype=bfloat16, shape=[B, T, HV, V])
        g_beta = self.global_view(beta_ptr, dtype=bfloat16, shape=[B, T, HV])
        initial_state_source = self.global_view(
            initial_state_source_ptr, dtype=float32, shape=[MAX_T, HV, K, V]
        )
        initial_state_indices = self.global_view(
            initial_state_indices_ptr, dtype=int32, shape=[T]
        )

        i_t, i_bv, i_hv = self.blockIdx
        i_h = i_hv // (HV // H)

        r_q = self.load_global(g_q, offsets=[0, i_t, i_h, 0], shape=[K], dims=[3]).to(
            float32
        )
        r_k = self.load_global(g_k, offsets=[0, i_t, i_h, 0], shape=[K], dims=[3]).to(
            float32
        )

        # self.print_tensor('r_q ', r_q)
        # self.print_tensor('r_k ', r_k)

        # normalize q and k
        if USE_QK_L2NORM_IN_KERNEL:
            r_q = r_q / self.sqrt(self.sum(r_q * r_q, dim=0))
            r_k = r_k / self.sqrt(self.sum(r_k * r_k, dim=0))

        r_q = r_q * scale

        # self.print_tensor('normalized r_q ', r_q)
        # self.print_tensor('normalized r_k ', r_k)

        # load initial state
        if USE_INITIAL_STATE:
            state_idx: int32 = initial_state_indices[i_t]
            r_h = self.load_global(
                initial_state_source,
                offsets=[state_idx, i_hv, 0, i_bv * self.BV],
                shape=[K, self.BV],
                dims=[2, 3],
            )
        else:
            state_idx: int32 = -1
            r_h = self.register_tensor(dtype=float32, shape=[K, self.BV], init=0.0)

        # H' = alpha * H : [K, BV] = [] * [K, BV]
        alpha: float32 = math.exp(g_g[0, i_t, i_hv])
        r_h = r_h * alpha

        # p = k * H : [BV] = reduce([K, 1] * [K, BV], dim=0)
        r_p = self.sum(self.unsqueeze(r_k, dim=1) * r_h, dim=0)
        # self.print_tensor('r_p ', r_p)

        # r = beta * (v - p) : [BV] = [] * ([BV] - [BV])
        beta: float32 = float32(g_beta[0, i_t, i_hv])
        r_v = self.load_global(
            g_v, offsets=[0, i_t, i_hv, i_bv * self.BV], shape=[self.BV], dims=[3]
        ).to(float32)
        r_r = beta * (r_v - r_p)

        # H'' = H' + k * r : [K, BV] = [K, BV] + [K] * [BV]
        r_h += self.unsqueeze(r_k, dim=1) * self.unsqueeze(r_r, dim=0)

        # o = q * h : [BV] = [K] * [K, BV]
        r_o = self.sum(self.unsqueeze(r_q, dim=1) * r_h, dim=0).to(bfloat16)

        self.store_global(g_o, r_o, offsets=[0, i_t, i_hv, i_bv * self.BV], dims=[3])
        if state_idx >= 0:
            self.store_global(
                initial_state_source,
                r_h,
                offsets=[state_idx, i_hv, 0, i_bv * self.BV],
                dims=[2, 3],
            )

        self.annotate_layout(
            r_h,
            self.cuda.default_register_layout(
                num_warps=self.num_warps, dtype=float32, shape=[K, self.BV]
            ),
        )


def fused_recurrent_gated_delta_rule_update_fwd_tilus(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    _ = cu_seqlens
    o = torch.empty_like(v)

    B, T, H, K = q.shape
    HV, V = v.shape[-2:]
    MAX_T = initial_state_source.shape[0]

    USE_INITIAL_STATE = True
    USE_QK_L2NORM_IN_KERNEL = use_qk_l2norm_in_kernel

    FusedRecurrentGatedDeltaRuleUpdateFwdKernel()(
        q,
        k,
        v,
        g,
        o,
        beta,
        scale,
        initial_state_source,
        initial_state_indices,
        T,
        B,
        H,
        HV,
        K,
        V,
        MAX_T,
        USE_INITIAL_STATE,
        USE_QK_L2NORM_IN_KERNEL,
    )

    return o


def fused_gdn_gating_tilus(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    """
    Tilus implementation of the fused GDN gating computation.

    Args:
        A_log: Log of A parameter, shape [num_heads]
        a: Input tensor, shape [batch, num_heads]
        dt_bias: Bias parameter, shape [num_heads]
        beta: Beta parameter for softplus (default: 1.0)
        threshold: Threshold for numerical stability (default: 20.0)

    Returns:
        g: Output tensor, shape [batch, num_heads]
    """
    batch, num_heads = a.shape

    # Create output tensor
    g = torch.empty_like(a, dtype=torch.float32)

    # Run the Tilus kernel
    FusedGdnGatingKernel()(
        g,
        A_log,
        a,
        dt_bias,
        batch,
        num_heads,
        beta,
        threshold,
    )

    return g


@tilus.autotune("num_warps", [1, 2, 4])
@tilus.autotune("BV", [2, 4, 8, 16, 32, 64, 128])
class FusedSigmoidGatingDeltaRuleUpdateKernel(tilus.Script):
    def __init__(self, num_warps: int, BV: int):
        super().__init__()
        self.num_warps = num_warps
        self.BV = BV

    def __call__(
        self,
        A_log_ptr: ~float32,
        a_ptr: ~bfloat16,
        dt_bias_ptr: ~float32,
        softplus_beta: float,
        softplus_threshold: float,
        q_ptr: ~bfloat16,
        k_ptr: ~bfloat16,
        v_ptr: ~bfloat16,
        b_ptr: ~bfloat16,
        o_ptr: ~bfloat16,
        scale: float,
        initial_state_source_ptr: ~float32,
        initial_state_indices_ptr: ~int32,
        T: int32,
        B: int,
        H: int,
        HV: int,
        K: int,
        V: int,
        MAX_T: int,
        USE_INITIAL_STATE: bool,
        USE_QK_L2NORM_IN_KERNEL: bool,
    ):
        self.attrs.warps = self.num_warps
        self.attrs.blocks = (T, cdiv(V, self.BV), HV)

        # Create global views
        g_A_log = self.global_view(A_log_ptr, dtype=float32, shape=[HV])
        g_a = self.global_view(a_ptr, dtype=bfloat16, shape=[T, HV])
        g_dt_bias = self.global_view(dt_bias_ptr, dtype=float32, shape=[HV])
        g_q = self.global_view(q_ptr, dtype=bfloat16, shape=[B, T, H, K])
        g_k = self.global_view(k_ptr, dtype=bfloat16, shape=[B, T, H, K])
        g_v = self.global_view(v_ptr, dtype=bfloat16, shape=[B, T, HV, V])
        g_b = self.global_view(b_ptr, dtype=bfloat16, shape=[T, HV])
        g_o = self.global_view(o_ptr, dtype=bfloat16, shape=[B, T, HV, V])
        initial_state_source = self.global_view(
            initial_state_source_ptr, dtype=float32, shape=[MAX_T, HV, K, V]
        )
        initial_state_indices = self.global_view(
            initial_state_indices_ptr, dtype=int32, shape=[T]
        )

        i_t, i_bv, i_hv = self.blockIdx
        i_h = i_hv // (HV // H)

        # Load query and key for this timestep and head
        r_q = self.load_global(g_q, offsets=[0, i_t, i_h, 0], shape=[K], dims=[3]).to(
            float32
        )
        r_k = self.load_global(g_k, offsets=[0, i_t, i_h, 0], shape=[K], dims=[3]).to(
            float32
        )

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            r_q = r_q / self.sqrt(self.sum(r_q * r_q, dim=0))
            r_k = r_k / self.sqrt(self.sum(r_k * r_k, dim=0))

        # Scale query
        r_q = r_q * scale

        # Load gating computation inputs
        r_A_log = self.load_global(g_A_log, offsets=[i_hv], shape=[1], dims=[0]).to(
            float32
        )
        r_a = self.load_global(g_a, offsets=[i_t, i_hv], shape=[1], dims=[0]).to(float32)
        r_dt_bias = self.load_global(g_dt_bias, offsets=[i_hv], shape=[1], dims=[0]).to(
            float32
        )

        # Compute gating: g = -exp(A_log) * softplus(a + dt_bias)
        r_x = r_a + r_dt_bias
        r_beta_x = r_x * softplus_beta

        # Apply softplus with numerical stability
        r_exp_beta_x = self.exp(r_beta_x)
        r_log_term = self.log(r_exp_beta_x + 1.0)
        r_softplus = self.where(
            r_x >= softplus_threshold, r_x, r_log_term * (1.0 / softplus_beta)
        )

        # Compute gating value
        r_g = -self.exp(r_A_log) * r_softplus

        # Load b and compute beta = sigmoid(b)
        r_b = self.load_global(g_b, offsets=[i_t, i_hv], shape=[1], dims=[0]).to(float32)
        r_beta = 1.0 / (1.0 + self.exp(-r_b))  # sigmoid

        # Load initial state
        if USE_INITIAL_STATE:
            state_idx: int32 = initial_state_indices[i_t]
            r_h = self.load_global(
                initial_state_source,
                offsets=[state_idx, i_hv, 0, i_bv * self.BV],
                shape=[K, self.BV],
                dims=[2, 3],
            )
        else:
            state_idx: int32 = -1
            r_h = self.register_tensor(dtype=float32, shape=[K, self.BV], init=0.0)

        # Apply gating to hidden state: H' = alpha * H
        alpha: float32 = self.exp(r_g)
        r_h = r_h * alpha

        # Compute prediction: p = k * H
        r_p = self.sum(self.unsqueeze(r_k, dim=1) * r_h, dim=0)

        # Load value and apply delta rule: r = beta * (v - p)
        r_v = self.load_global(
            g_v, offsets=[0, i_t, i_hv, i_bv * self.BV], shape=[self.BV], dims=[3]
        ).to(float32)
        r_r = r_beta * (r_v - r_p)

        # Update hidden state: H'' = H' + k * r
        r_h += self.unsqueeze(r_k, dim=1) * self.unsqueeze(r_r, dim=0)

        # Compute output: o = q * h
        r_o = self.sum(self.unsqueeze(r_q, dim=1) * r_h, dim=0).to(bfloat16)

        # Store output
        self.store_global(g_o, r_o, offsets=[0, i_t, i_hv, i_bv * self.BV], dims=[3])

        # Store updated state back
        if state_idx >= 0:
            self.store_global(
                initial_state_source,
                r_h,
                offsets=[state_idx, i_hv, 0, i_bv * self.BV],
                dims=[2, 3],
            )

        # Annotate layout for optimization
        self.annotate_layout(
            r_h,
            self.cuda.default_register_layout(
                num_warps=self.num_warps, dtype=float32, shape=[K, self.BV]
            ),
        )


def sigmoid_gating_delta_rule_update_tilus(
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
    cu_seqlens: torch.Tensor = None,
):
    """
    Tilus implementation of sigmoid gating delta rule update using a single fused kernel.

    This function uses a single kernel that combines both sigmoid gating computation
    and the fused recurrent gated delta rule update for better performance.

    Args:
        A_log: Log of A parameter for gating computation
        a: Input tensor for gating computation
        dt_bias: Bias parameter for gating computation
        softplus_beta: Beta parameter for softplus in gating
        softplus_threshold: Threshold for numerical stability in softplus
        q: Query tensor
        k: Key tensor
        v: Value tensor
        b: Tensor to apply sigmoid to get beta
        scale: Scaling factor for queries
        initial_state_source: Initial hidden states
        initial_state_indices: Indices for initial states
        use_qk_l2norm_in_kernel: Whether to apply L2 normalization to q and k
        cu_seqlens: Cumulative sequence lengths (optional, for variable length)

    Returns:
        o: Output tensor from the delta rule update
    """
    _ = cu_seqlens  # Not used in current implementation

    # Get dimensions
    B, T, H, K = q.shape
    HV, V = v.shape[-2:]
    MAX_T = initial_state_source.shape[0]

    # Create output tensor
    o = torch.empty_like(v)

    USE_INITIAL_STATE = True
    USE_QK_L2NORM_IN_KERNEL = use_qk_l2norm_in_kernel

    # Run the fused kernel
    FusedSigmoidGatingDeltaRuleUpdateKernel()(
        A_log,
        a,
        dt_bias,
        softplus_beta,
        softplus_threshold,
        q,
        k,
        v,
        b,
        o,
        scale,
        initial_state_source,
        initial_state_indices,
        T,
        B,
        H,
        HV,
        K,
        V,
        MAX_T,
        USE_INITIAL_STATE,
        USE_QK_L2NORM_IN_KERNEL,
    )

    return o
