# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tilus kernel for the DSV3.2 sparse-attention indexer logits (M1).

Computes the per-batch, per-token logits produced by the DeepSeek V3.2
sparse-attention indexer::

    logits[b, l] = sum_h relu(Q[b, h] @ K[b, l]) * weights[b, h]

where the K-cache is paged FP8 with a per-token FP32 scale stored at the tail
of each page (``[num_pages, 64 tokens, 128 fp8 bytes + 4 scale bytes]``).

SM100+ variant using ``tcgen05`` tensor cores (F8F6F4 MMA). Each CTA
processes one batch item and two consecutive pages of K: ``BL=128`` is
required by F8F6F4 with ``cta_group=1``.
"""

import tilus
import torch
from tilus import boolean, float8_e4m3, float32, int32
from tilus.hidet.ir.type import void_p
from tilus.utils import cdiv

tilus.option.cache_dir("./cache")


@tilus.autotune("num_warps", [4, 8])
class DSV3IndexerLogits(tilus.Script):
    """FP8 paged-MQA logits kernel for the DSV3.2 indexer.

    Tile shape::

        M = BL         = 128  (two pages — F8F6F4 cta_group=1 requires m=128)
        N = NUM_HEADS  = 64
        K = HEAD_DIM   = 128

    Grid: ``[cdiv(max_model_len, BL), batch_size]``. Each CTA handles one
    batch item and two consecutive pages of K.
    """

    BL = 128
    PAGE_BL = 64
    NUM_HEADS = 64
    HEAD_DIM = 128
    PAGE_BYTES = 64 * 132  # 8192 fp8 + 256 scale = 8448 bytes/page

    def __init__(self, num_warps: int):
        super().__init__()
        self.num_warps = num_warps

    def __call__(
        self,
        q_ptr: void_p,  # fp8_e4m3, [batch_size, 64, 128]
        k_fp8_ptr: void_p,  # fp8_e4m3 view of the FP8 payload
        k_scale_ptr: void_p,  # fp32 view of the per-token scale (offset 8192 B)
        w_ptr: void_p,  # fp32, [batch_size, 64]
        seq_lens_ptr: void_p,  # int32, [batch_size]
        block_table_ptr: void_p,  # int32, [batch_size, max_num_pages]
        logits_ptr: void_p,  # fp32, [batch_size, max_model_len]
        batch_size: int32,
        num_pages_total: int32,
        max_num_pages: int32,
        max_model_len: int32,
    ):
        BL, PAGE_BL = self.BL, self.PAGE_BL
        H, D = self.NUM_HEADS, self.HEAD_DIM
        PAGE_BYTES = self.PAGE_BYTES

        self.attrs.blocks = [cdiv(max_model_len, BL), batch_size]
        self.attrs.warps = self.num_warps

        tile_id: int32 = self.blockIdx.x
        bs: int32 = self.blockIdx.y

        # --- Global views ----------------------------------------------------
        g_q = self.global_view(q_ptr, dtype=float8_e4m3, shape=[batch_size, H, D])
        # FP8 payload — stride in fp8 elements (= bytes). Per-page stride is
        # PAGE_BYTES because the scale block lives in the tail of each page.
        g_k_fp8 = self.global_view(
            k_fp8_ptr,
            dtype=float8_e4m3,
            shape=[num_pages_total, PAGE_BL, D],
            strides=[PAGE_BYTES, D, 1],
        )
        # Scale — pointer is offset past page 0's FP8 payload. Strides in
        # fp32 elements (PAGE_BYTES // 4).
        g_k_scale = self.global_view(
            k_scale_ptr,
            dtype=float32,
            shape=[num_pages_total, PAGE_BL],
            strides=[PAGE_BYTES // 4, 1],
        )
        g_w = self.global_view(w_ptr, dtype=float32, shape=[batch_size, H])
        g_seq_lens = self.global_view(
            seq_lens_ptr, dtype=int32, shape=[batch_size]
        )
        g_bt = self.global_view(
            block_table_ptr, dtype=int32, shape=[batch_size, max_num_pages]
        )
        g_logits = self.global_view(
            logits_ptr, dtype=float32, shape=[batch_size, max_model_len]
        )

        # --- Shared allocations ---------------------------------------------
        # Q is shared across all tiles in a batch item. K is loaded two pages
        # at a time into a staged [2, PAGE_BL, D] buffer and later reshaped to
        # [BL, D] for the MMA.
        s_q = self.shared_tensor(dtype=float8_e4m3, shape=[H, D])
        s_k_pages = self.shared_tensor(dtype=float8_e4m3, shape=[2, PAGE_BL, D])
        s_scale_pages = self.shared_tensor(dtype=float32, shape=[2, PAGE_BL])

        # TMEM accumulator: scores [BL, H] fp32.
        t_acc = self.tcgen05.alloc(dtype=float32, shape=[BL, H])

        tma_barrier, mma_barrier = self.mbarrier.alloc(counts=[1, 1]).tolist()

        # --- Page IDs for the two halves of this tile -----------------------
        page_id_0: int32 = g_bt[bs, 2 * tile_id].item()
        page_id_1: int32 = g_bt[bs, 2 * tile_id + 1].item()

        # Scales are small 1D transfers (64 fp32 = 256 B each); cp.async is
        # simpler than TMA here (per report §2.1).
        self.copy_async(
            src=g_k_scale, dst=s_scale_pages[0], offsets=[page_id_0, 0], dims=[1]
        )
        self.copy_async(
            src=g_k_scale, dst=s_scale_pages[1], offsets=[page_id_1, 0], dims=[1]
        )
        self.copy_async_commit_group()

        self.sync()

        # --- Async bulk loads via TMA (Q + two K pages) ---------------------
        with self.single_warp():
            with self.single_thread():
                self.mbarrier.arrive_and_expect_tx(
                    tma_barrier,
                    transaction_bytes=(
                        s_q.nbytes + 2 * s_k_pages[0].nbytes
                    ),
                )
            self.tma.global_to_shared(
                src=g_q, dst=s_q, offsets=[bs, 0, 0], dims=[1, 2],
                mbarrier=tma_barrier,
            )
            self.tma.global_to_shared(
                src=g_k_fp8, dst=s_k_pages[0],
                offsets=[page_id_0, 0, 0], dims=[1, 2],
                mbarrier=tma_barrier,
            )
            self.tma.global_to_shared(
                src=g_k_fp8, dst=s_k_pages[1],
                offsets=[page_id_1, 0, 0], dims=[1, 2],
                mbarrier=tma_barrier,
            )

        self.mbarrier.wait(tma_barrier, phase=0)
        self.copy_async_wait_group(0)
        self.sync()

        # Flatten the two-page buffers into contiguous 2D tiles for downstream
        # consumption (MMA + epilogue).
        s_k = self.reshape_shared(s_k_pages, [BL, D])
        s_scale = self.reshape_shared(s_scale_pages, [BL])

        # --- tcgen05 MMA: scores[BL, H] = K[BL, D] @ Q[H, D].T --------------
        # F8F6F4 kind, cta_group=1, inst_k=32 handled internally by the
        # emitter (k_size=128 → 4 inner k iterations).
        with self.single_warp():
            self.tcgen05.mma(s_k, s_q.transpose(), t_acc, enable_input_d=False)
            self.tcgen05.commit(mbarrier=mma_barrier)
        self.mbarrier.wait(mma_barrier, phase=0)

        # --- Move accumulator TMEM → registers ------------------------------
        r_scores = self.tcgen05.load(t_acc)
        self.tcgen05.wait_load()

        # --- Epilogue: relu → scale/weight → reduce → store -----------------
        r_scores = self.where(r_scores >= 0.0, x=r_scores, y=0.0)

        r_scale = self.load_shared(s_scale)  # [BL]
        r_w = self.load_global(g_w, offsets=[bs, 0], shape=[H], dims=[1])  # [H]

        # [BL, H] * [BL, 1] * [1, H]
        r_weighted = (
            r_scores * self.unsqueeze(r_scale, dim=1) * self.unsqueeze(r_w, dim=0)
        )
        r_logit = self.sum(r_weighted, dim=1)  # [BL]

        # Mask positions >= seq_lens[bs] to -inf. Downstream top-K expects
        # invalid positions to sort to the bottom.
        seq_len_b: int32 = g_seq_lens[bs].item()
        tile_base: int32 = tile_id * BL
        valid_mask = self.register_tensor(
            dtype=boolean,
            shape=[BL],
            init=lambda i: tile_base + i < seq_len_b,
        )
        r_logit = self.where(valid_mask, x=r_logit, y=float("-inf"))

        self.store_global(g_logits, r_logit, offsets=[bs, tile_id * BL], dims=[1])

        self.tcgen05.dealloc(t_acc)


_KERNEL = None


def _get_kernel() -> DSV3IndexerLogits:
    """Cache the autotuned kernel so the tuning pass runs once per process."""
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = DSV3IndexerLogits()
    return _KERNEL


def dsv3_indexer_logits_tilus(
    q: torch.Tensor,  # fp8_e4m3, [B, 64, 128]
    k_cache: torch.Tensor,  # uint8, [num_pages, 64, 1, 132]
    weights: torch.Tensor,  # fp32, [B, 64]
    seq_lens: torch.Tensor,  # int32, [B]  (unused by the kernel)
    block_table: torch.Tensor,  # int32, [B, max_num_pages]
    max_model_len: int,
) -> torch.Tensor:
    """Launch the FP8 paged-MQA logits kernel.

    Returns ``logits[B, max_model_len]`` fp32. Positions ``l >= seq_lens[b]``
    are masked to ``-inf`` so that downstream top-K sorts them to the bottom.
    ``max_model_len`` must be a multiple of ``BL=128`` (two pages per CTA).
    """
    assert max_model_len % DSV3IndexerLogits.BL == 0, (
        f"max_model_len ({max_model_len}) must be a multiple of "
        f"BL={DSV3IndexerLogits.BL} for the two-page tcgen05 kernel."
    )
    batch_size = q.shape[0]
    num_pages_total = k_cache.shape[0]
    max_num_pages = block_table.shape[1]

    logits = torch.empty(
        (batch_size, max_model_len),
        dtype=torch.float32,
        device=q.device,
    )

    k_fp8_ptr = k_cache.data_ptr()
    # Page 0's scale block starts 64*128 = 8192 bytes into the cache.
    k_scale_ptr = k_cache.data_ptr() + 64 * 128

    kernel = _get_kernel()
    kernel(
        q.data_ptr(),
        k_fp8_ptr,
        k_scale_ptr,
        weights.data_ptr(),
        seq_lens.data_ptr(),
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
            # Positions beyond seq_len must be masked to -inf so downstream
            # top-K never picks them.
            if L < max_model_len:
                tail = logits[b, L:max_model_len]
                # Kernel writes float32 min (-3.4e38); tilus truncates
                # constant -inf to the representable minimum.
                assert (tail <= -1e30).all(), (
                    f"positions >= seq_len[{b}]={L} are not masked"
                )
        print(f"[batch={batch_size}, seq_len={seq_len}]  OK  max abs err={max_err:.3e}")

    # Ragged case: different seq_lens per batch, each shorter than max_model_len.
    # Exercises the -inf masking path on real data.
    print("Ragged-seq-lens check:")
    q, k_cache, weights, seq_lens, block_table, max_model_len = _make_test_data(
        batch_size=4, seq_len=1024
    )
    seq_lens = torch.tensor([100, 512, 900, 1024], dtype=torch.int32, device="cuda")
    _, ref_logits_list = dsv3_topk_indexer_ref(
        q, k_cache, weights, seq_lens, block_table
    )
    logits = dsv3_indexer_logits_tilus(
        q, k_cache, weights, seq_lens, block_table, max_model_len=max_model_len
    )
    torch.cuda.synchronize()
    for b in range(4):
        L = int(seq_lens[b].item())
        torch.testing.assert_close(logits[b, :L], ref_logits_list[b], atol=1e-1, rtol=1e-1)
        if L < max_model_len:
            tail = logits[b, L:max_model_len]
            assert (tail <= -1e30).all(), (
                f"positions >= seq_len[{b}]={L} are not masked"
            )
        print(f"  batch={b} seq_len={L}: OK")

    print("All good.")


if __name__ == "__main__":
    main()
