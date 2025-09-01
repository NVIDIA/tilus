import torch


def fused_gdn_gating_torch(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    """
    PyTorch implementation of the fused GDN gating computation.

    Computes: g = -exp(A_log) * softplus(a + dt_bias)

    Args:
        A_log: Log of A parameter, shape [num_heads]
        a: Input tensor, shape [batch, num_heads] or [batch, seq_len, num_heads]
        dt_bias: Bias parameter, shape [num_heads]
        beta: Beta parameter for softplus (default: 1.0)
        threshold: Threshold for numerical stability (default: 20.0)

    Returns:
        g: Output tensor, same shape as input a
    """
    # Compute x = a + dt_bias
    x = a.float() + dt_bias.float()

    # Apply softplus with beta and threshold for numerical stability
    # softplus_x = where(beta * x <= threshold, (1/beta) * log(1 + exp(beta * x)), x)
    beta_x = beta * x
    mask = beta_x <= threshold

    # For numerical stability, use the approximation when beta*x > threshold
    softplus_x = torch.where(mask, (1.0 / beta) * torch.log(1.0 + torch.exp(beta_x)), x)

    # Compute final result: g = -exp(A_log) * softplus_x
    g = -torch.exp(A_log.float()) * softplus_x

    return g  # Return as float32 to match Triton implementation


def fused_recurrent_gated_delta_rule_update_fwd_torch(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state_source,
    initial_state_indices,
    use_qk_l2norm_in_kernel=False,
):
    """
    Reference implementation of the fused recurrent gated delta rule update.

    This implements the same computation as the Tilus kernel but in pure PyTorch.
    """
    # Get dimensions
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    device = q.device
    dtype = torch.float32

    # Initialize output
    o = torch.zeros(B, T, HV, V, device=device, dtype=q.dtype)

    # Process each batch element
    for b in range(B):
        # Process each timestep independently (matching kernel approach)
        for t in range(T):
            # Process each head group (HV heads)
            for hv in range(HV):
                # Determine which H head this HV head corresponds to
                h_idx = hv // (HV // H) if HV >= H else 0

                # Initialize hidden state [K, V]
                h_state = torch.zeros(K, V, device=device, dtype=dtype)

                # Load initial state if provided
                if initial_state_source is not None and initial_state_indices is not None:
                    idx = initial_state_indices[t].item()  # Use timestep-specific index
                    if idx >= 0 and idx < initial_state_source.shape[0]:
                        h_state = initial_state_source[idx, hv].clone().to(dtype)
                # Get current inputs
                q_t = q[b, t, h_idx].to(dtype)  # [K]
                k_t = k[b, t, h_idx].to(dtype)  # [K]
                v_t = v[b, t, hv].to(dtype)  # [V]
                g_t = g[b, t, hv].to(dtype)  # scalar

                # Handle beta (can be headwise or scalar)
                if beta.ndim == v.ndim:  # headwise
                    beta_t = beta[b, t, hv].to(dtype)  # [V] or scalar
                else:
                    beta_t = beta[b, t, hv].to(dtype)  # scalar

                # Apply L2 normalization if enabled
                if use_qk_l2norm_in_kernel:
                    q_norm = torch.sqrt(torch.sum(q_t * q_t)) + 1e-6
                    k_norm = torch.sqrt(torch.sum(k_t * k_t)) + 1e-6
                    q_t = q_t / q_norm
                    k_t = k_t / k_norm

                # Scale query
                q_t = q_t * scale

                # Decay hidden state: h *= exp(g)
                h_state = h_state * torch.exp(g_t)

                # Delta rule: v -= sum(h * k, dim=0)
                # h_state is [K, V], k_t is [K], so we want sum over K dimension
                prediction = torch.sum(h_state * k_t[:, None], dim=0)  # [V]
                v_t = v_t - prediction

                # Apply beta gating: v *= beta
                v_t = v_t * beta_t

                # Update hidden state: h += k[:, None] * v[None, :]
                h_state = h_state + k_t[:, None] * v_t[None, :]  # [K, V]

                # Compute output: o = sum(h * q, dim=0)
                o_t = torch.sum(h_state * q_t[:, None], dim=0)  # [V]
                o[b, t, hv] = o_t.to(q.dtype)

                # Store final state back (if needed - the kernel does this)
                if initial_state_source is not None and initial_state_indices is not None:
                    idx = initial_state_indices[t].item()  # Use timestep-specific index
                    if idx >= 0 and idx < initial_state_source.shape[0]:
                        initial_state_source[idx, hv] = h_state.to(
                            initial_state_source.dtype
                        )

    return o


def sigmoid_gating_delta_rule_update_torch(
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
    PyTorch implementation of sigmoid gating delta rule update.

    This function combines sigmoid gating computation with the fused recurrent
    gated delta rule update, similar to the Triton implementation.

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
    # Apply sigmoid to b to get beta
    beta = torch.sigmoid(b)

    # Compute gating values using the fused GDN gating function
    g = fused_gdn_gating_torch(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        beta=softplus_beta,
        threshold=softplus_threshold,
    )

    beta, g = beta.unsqueeze(0), g.unsqueeze(0)  # Add batch dim if needed

    # Apply the fused recurrent gated delta rule update
    o = fused_recurrent_gated_delta_rule_update_fwd_torch(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    return o
