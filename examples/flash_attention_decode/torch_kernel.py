import torch


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
