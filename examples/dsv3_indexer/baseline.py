# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch reference implementation of the DeepSeek V3.2 sparse-attention indexer.

This is the correctness baseline used by Allan's branch (see
``tests/attention/test_dsv3_indexer.py::_dsa_topk_indexer`` in the
``Aalanli/flashinfer`` ``alin_topk`` branch). The Tilus implementation is
verified against this reference.

See ``README.md`` for the math, tensor layouts, and role of the indexer
in DeepSeek V3.2 Sparse Attention.
"""

from typing import List, Tuple

import torch


def dequant_fp8_kv_cache(k_cache: torch.Tensor) -> torch.Tensor:
    """Dequantize the paged FP8 KV cache.

    The cache layout is ``[num_pages, 64, 1, 132]`` uint8. Each page holds
    64 tokens; per token there are 128 bytes of FP8 values followed by
    4 bytes (one FP32 scale, broadcast over the 128-wide head dim).

    Returns ``[num_pages, 64, 128]`` float32.
    """
    num_pages, page_size = k_cache.shape[0], k_cache.shape[1]
    head_dim = 128
    flat = k_cache.view(num_pages, page_size * 132)

    fp8_bytes = flat[:, : page_size * head_dim].contiguous()
    k_fp8 = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    k_fp32 = k_fp8.to(torch.float32)

    scale_bytes = flat[:, page_size * head_dim :].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)

    return k_fp32 * scale


@torch.no_grad()
def dsv3_topk_indexer_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk: int = 2048,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Per-batch indexer logits + top-K key indices.

    For each batch ``b`` with sequence length ``L = seq_lens[b]``::

        K = dequant(k_cache gathered via block_table[b, :ceil(L/64)])[:L]   # [L, 128]
        scores = relu(q[b].float() @ K.T)                                    # [64, L]
        logits = (scores * weights[b][:, None]).sum(dim=0)                   # [L]
        indices = topk(logits, k=min(topk, L))                               # int32

    Args:
        q:           ``[batch, 64, 128]`` FP8 CUDA tensor (64 query heads).
        k_cache:     ``[num_pages, 64, 1, 132]`` uint8 paged FP8 cache.
        weights:     ``[batch, 64]`` float32 per-head weights.
        seq_lens:    ``[batch]`` int32 sequence lengths.
        block_table: ``[batch, max_num_pages]`` int32 paged indices.
        topk:        Number of key tokens to select per batch (default 2048).

    Returns:
        ``(indices, logits)``:
            - ``indices``: ``[batch, topk]`` int32, ``-1`` where ``L < topk``.
            - ``logits``: per-batch list of float32 tensors shaped ``[L]``.
    """
    batch_size = q.shape[0]
    page_size = 64
    device = q.device

    q_fp32 = q.to(torch.float32)
    k_all = dequant_fp8_kv_cache(k_cache)  # [num_pages, 64, 128]

    indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
    logits: List[torch.Tensor] = []

    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        if seq_len == 0:
            logits.append(torch.zeros(0, dtype=torch.float32, device=device))
            continue

        num_pages = (seq_len + page_size - 1) // page_size
        page_ids = block_table[b, :num_pages].to(torch.long)

        k = k_all[page_ids].reshape(-1, 128)[:seq_len]  # [L, 128]

        scores = torch.relu(q_fp32[b] @ k.T)  # [64, L]
        row_logits = (scores * weights[b, :, None]).sum(dim=0)  # [L]
        logits.append(row_logits)

        k_eff = min(topk, seq_len)
        _, top_idx = torch.topk(row_logits, k_eff)
        indices[b, :k_eff] = top_idx.to(torch.int32)

    return indices, logits
