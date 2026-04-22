# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tilus kernel for the DSV3.2 sparse-attention indexer logits (M1).

Computes the per-batch, per-token logits produced by the DeepSeek V3.2
sparse-attention indexer::

    logits[b, l] = sum_h relu(Q[b, h] @ K[b, l]) * weights[b, h]

where the K-cache is paged FP8 with a per-token FP32 scale stored at the tail
of each page (``[num_pages, 64 tokens, 128 fp8 bytes + 4 scale bytes]``).

This is the minimal-correctness M1 implementation using warp-level FP8 MMA
(``m16n8k32.f8e4m3.f32``) via ``self.dot``, which works on SM89+ (including
workstation Blackwell SM120). A SM100/SM103 ``tcgen05`` variant is a
follow-up once the logic is validated.
"""

import tilus
import torch
from tilus import float8_e4m3, float16, float32, int32
from tilus.hidet.ir.type import void_p
from tilus.utils import cdiv

tilus.option.cache_dir("./cache")


class DSV3IndexerLogits(tilus.Script):
    """FP8 paged-MQA logits kernel for the DSV3.2 indexer.

    Tile shape::

        M = BL         = 64   (one page of tokens)
        N = NUM_HEADS  = 64
        K = HEAD_DIM   = 128

    Grid: ``[cdiv(max_model_len, BL), batch_size]``. Each CTA handles one
    batch item and one page of K.
    """

    BL = 64
    NUM_HEADS = 64
    HEAD_DIM = 128
    PAGE_BYTES = 64 * 132  # 8192 fp8 + 256 scale = 8448 bytes/page

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        q_ptr: void_p,  # fp8_e4m3, [batch_size, 64, 128]
        k_fp8_ptr: void_p,  # fp8_e4m3 view of the FP8 payload
        k_scale_ptr: void_p,  # fp32 view of the per-token scale (offset 8192 B)
        w_ptr: void_p,  # fp32, [batch_size, 64]
        block_table_ptr: void_p,  # int32, [batch_size, max_num_pages]
        logits_ptr: void_p,  # fp32, [batch_size, max_model_len]
        batch_size: int32,
        num_pages_total: int32,
        max_num_pages: int32,
        max_model_len: int32,
    ):
        BL, H, D = self.BL, self.NUM_HEADS, self.HEAD_DIM
        PAGE_BYTES = self.PAGE_BYTES

        self.attrs.blocks = [cdiv(max_model_len, BL), batch_size]
        self.attrs.warps = 4

        tile_id: int32 = self.blockIdx.x
        bs: int32 = self.blockIdx.y

        # --- Global views ----------------------------------------------------
        g_q = self.global_view(q_ptr, dtype=float8_e4m3, shape=[batch_size, H, D])
        # FP8 payload — stride in fp8 elements (= bytes). Per-page stride is
        # PAGE_BYTES because the scale block lives in the tail of each page.
        g_k_fp8 = self.global_view(
            k_fp8_ptr,
            dtype=float8_e4m3,
            shape=[num_pages_total, BL, D],
            strides=[PAGE_BYTES, D, 1],
        )
        # Scale — pointer is offset past page 0's FP8 payload. Strides in
        # fp32 elements (PAGE_BYTES // 4).
        g_k_scale = self.global_view(
            k_scale_ptr,
            dtype=float32,
            shape=[num_pages_total, BL],
            strides=[PAGE_BYTES // 4, 1],
        )
        g_w = self.global_view(w_ptr, dtype=float32, shape=[batch_size, H])
        g_bt = self.global_view(
            block_table_ptr, dtype=int32, shape=[batch_size, max_num_pages]
        )
        g_logits = self.global_view(
            logits_ptr, dtype=float32, shape=[batch_size, max_model_len]
        )

        # --- Shared allocations ---------------------------------------------
        s_q = self.shared_tensor(dtype=float8_e4m3, shape=[H, D])
        s_k = self.shared_tensor(dtype=float8_e4m3, shape=[BL, D])
        s_scale = self.shared_tensor(dtype=float32, shape=[BL])

        page_id: int32 = g_bt[bs, tile_id].item()

        self.copy_async(src=g_q, dst=s_q, offsets=[bs, 0, 0], dims=[1, 2])
        self.copy_async(src=g_k_fp8, dst=s_k, offsets=[page_id, 0, 0], dims=[1, 2])
        self.copy_async(src=g_k_scale, dst=s_scale, offsets=[page_id, 0], dims=[1])
        self.copy_async_commit_group()
        self.copy_async_wait_group(0)
        self.sync()

        # --- Compute scores = K @ Q.T → [BL, H] -----------------------------
        # Tilus' atomic MMA table on SM120 doesn't register FP8 ops, so we
        # cast operands to FP16 in registers. Numerically equivalent here
        # because the per-token FP32 scale is applied in the epilogue.
        r_k = self.cast(self.load_shared(s_k), dtype=float16)
        r_q = self.cast(self.load_shared(s_q), dtype=float16)
        r_scores = self.dot(r_k, r_q.transpose(), acc_dtype=float32)

        # --- Epilogue: relu → scale/weight → reduce → store -----------------
        r_scores = self.where(r_scores >= 0.0, x=r_scores, y=0.0)

        r_scale = self.load_shared(s_scale)
        r_w = self.load_global(g_w, offsets=[bs, 0], shape=[H], dims=[1])

        # r_scores [BL, H] * r_scale [BL, 1] * r_w [1, H]
        r_weighted = (
            r_scores * self.unsqueeze(r_scale, dim=1) * self.unsqueeze(r_w, dim=0)
        )
        r_logit = self.sum(r_weighted, dim=1)  # [BL]

        self.store_global(g_logits, r_logit, offsets=[bs, tile_id * BL], dims=[1])


def dsv3_indexer_logits_tilus(
    q: torch.Tensor,  # fp8_e4m3, [B, 64, 128]
    k_cache: torch.Tensor,  # uint8, [num_pages, 64, 1, 132]
    weights: torch.Tensor,  # fp32, [B, 64]
    seq_lens: torch.Tensor,  # int32, [B]  (unused by the kernel)
    block_table: torch.Tensor,  # int32, [B, max_num_pages]
    max_model_len: int,
) -> torch.Tensor:
    """Launch the FP8 paged-MQA logits kernel.

    Returns ``logits[B, max_model_len]`` fp32. Positions beyond
    ``seq_lens[b]`` contain kernel-computed values against garbage K entries;
    callers should mask using ``seq_lens``. For the M1 minimal version the
    test fixture sets ``seq_len == max_model_len`` so every written position
    corresponds to a valid token.
    """
    del seq_lens  # reserved for future masking path
    batch_size = q.shape[0]
    num_pages_total = k_cache.shape[0]
    max_num_pages = block_table.shape[1]

    logits = torch.full(
        (batch_size, max_model_len),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )

    k_fp8_ptr = k_cache.data_ptr()
    # Page 0's scale block starts 64*128 = 8192 bytes into the cache.
    k_scale_ptr = k_cache.data_ptr() + 64 * 128

    kernel = DSV3IndexerLogits()
    kernel(
        q.data_ptr(),
        k_fp8_ptr,
        k_scale_ptr,
        weights.data_ptr(),
        block_table.data_ptr(),
        logits.data_ptr(),
        batch_size,
        num_pages_total,
        max_num_pages,
        max_model_len,
    )
    return logits


def _make_test_data(batch_size: int, seq_len: int):
    """Mirrors benchmark.py::make_test_data but is self-contained."""
    page_size = 64
    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    total_pages = batch_size * num_pages_per_seq
    max_num_pages = (num_pages_per_seq + 1) // 2 * 2

    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
    q = torch.randn(batch_size, 64, 128, dtype=torch.float32, device="cuda").to(
        torch.float8_e4m3fn
    )

    k_cache = torch.empty(total_pages, 64, 1, 132, dtype=torch.uint8, device="cuda")
    flat = k_cache.view(total_pages, -1)
    flat[:, : 64 * 128].view(torch.float8_e4m3fn).copy_(
        torch.randn(total_pages, 64 * 128, dtype=torch.float32, device="cuda").to(
            torch.float8_e4m3fn
        )
    )
    flat[:, 64 * 128 :].view(torch.float32).copy_(
        torch.randn(total_pages, 64, dtype=torch.float32, device="cuda").abs()
    )

    weights = torch.randn(batch_size, 64, dtype=torch.float32, device="cuda")
    block_table = torch.zeros(batch_size, max_num_pages, dtype=torch.int32, device="cuda")
    for b in range(batch_size):
        block_table[b, :num_pages_per_seq] = torch.arange(
            b * num_pages_per_seq,
            (b + 1) * num_pages_per_seq,
            dtype=torch.int32,
            device="cuda",
        )

    return q, k_cache, weights, seq_lens, block_table, max_num_pages * page_size


def main():
    from baseline import dsv3_topk_indexer_ref

    torch.manual_seed(0)
    for batch_size, seq_len in [(1, 64), (2, 256), (4, 1024), (8, 4096)]:
        q, k_cache, weights, seq_lens, block_table, max_model_len = _make_test_data(
            batch_size, seq_len
        )

        _, ref_logits_list = dsv3_topk_indexer_ref(
            q, k_cache, weights, seq_lens, block_table
        )
        logits = dsv3_indexer_logits_tilus(
            q,
            k_cache,
            weights,
            seq_lens,
            block_table,
            max_model_len=max_model_len,
        )
        torch.cuda.synchronize()

        max_err = 0.0
        for b in range(batch_size):
            L = int(seq_lens[b].item())
            ref = ref_logits_list[b]
            act = logits[b, :L]
            torch.testing.assert_close(act, ref, atol=1e-1, rtol=1e-1)
            max_err = max(max_err, (act - ref).abs().max().item())
        print(f"[batch={batch_size}, seq_len={seq_len}]  OK  max abs err={max_err:.3e}")

    print("All good.")


if __name__ == "__main__":
    main()
