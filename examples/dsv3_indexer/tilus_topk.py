# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tilus radix top-K kernel for the DSV3.2 sparse-attention indexer (M4).

Port of ``knowledge/flashinfer/csrc/dsv3_ops/fast_topk_clusters.cu`` (the
single-cluster, non-PDL, non-fused path).

All 4 radix phases (byte 3 → 0) are implemented. Phase 0 scans the full
logit array; phases 1-3 refine the borderline elements cached from the
previous phase.

NOTE: Uses global_scatter_add instead of shared_scatter_add as a workaround
for a Tilus transpiler hang when shared_scatter_add is called multiple times
with data-dependent (load_global-derived) indices.
"""

import tilus
import torch
from tilus import boolean, float32, int32, uint32
from tilus.hidet.ir.type import void_p

tilus.option.cache_dir("./cache")

# ---------------------------------------------------------------------------
# Phase 0: histogram over top byte of monotonic-uint32 logits
# ---------------------------------------------------------------------------


class DSV3TopKPhase0(tilus.Script):
    """Phase 0: full logit scan → emit bin>threshold, cache bin==threshold."""

    TOP_K = 2048
    RADIX = 256
    THREADS = 1024

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        logits_ptr: void_p,  # uint32 view of fp32 [B, max_model_len]
        seq_lens_ptr: void_p,  # int32 [B]
        indices_ptr: void_p,  # int32 [B * TOP_K] — output
        hist_p0_ptr: void_p,  # int32 [B * RADIX] — phase-0 histogram (temp)
        hist_ptr: void_p,  # int32 [B * RADIX] — phase-1 histogram (built here)
        topk_cnt_ptr: void_p,  # int32 [B] — topk emission counter
        cache_bits_ptr: void_p,  # uint32 [B * NUM_CACHED] — cached mono bits
        cache_idx_ptr: void_p,  # int32  [B * NUM_CACHED] — cached indices
        cache_cnt_ptr: void_p,  # int32 [B] — cached element count
        batch_size: int32,
        max_model_len: int32,
        num_cached: int32,
    ):
        TOP_K = self.TOP_K
        RADIX = self.RADIX
        THREADS = self.THREADS

        self.attrs.blocks = [batch_size]
        self.attrs.warps = THREADS // 32

        bs: int32 = self.blockIdx.x

        g_logits = self.global_view(
            logits_ptr, dtype=uint32, shape=[batch_size, max_model_len]
        )
        g_seq_lens = self.global_view(seq_lens_ptr, dtype=int32, shape=[batch_size])
        # Output buffer has TOP_K + 1 slots per batch; slot TOP_K is the trash
        # bin for overflow/inactive scatter writes (avoids clobbering slot
        # TOP_K - 1 which was the previous garbage target).
        g_indices = self.global_view(
            indices_ptr, dtype=int32, shape=[batch_size * (TOP_K + 1)]
        )
        g_hist_p0 = self.global_view(
            hist_p0_ptr, dtype=int32, shape=[batch_size * RADIX]
        )
        g_hist0 = self.global_view(
            hist_ptr, dtype=int32, shape=[batch_size * RADIX]
        )
        g_topk_cnt = self.global_view(
            topk_cnt_ptr, dtype=int32, shape=[batch_size]
        )
        g_cache_bits = self.global_view(
            cache_bits_ptr, dtype=uint32, shape=[batch_size * num_cached]
        )
        g_cache_idx = self.global_view(
            cache_idx_ptr, dtype=int32, shape=[batch_size * num_cached]
        )
        g_cache_cnt = self.global_view(
            cache_cnt_ptr, dtype=int32, shape=[batch_size]
        )

        seq_len: int32 = g_seq_lens[bs].item()
        hist_base: int32 = bs * RADIX
        idx_base: int32 = bs * (TOP_K + 1)
        cache_base: int32 = bs * num_cached

        # --- Shared state for threshold ---
        s_threshold_bin = self.shared_tensor(dtype=int32, shape=[1])

        # --- Tile register constants ---
        r_ones = self.register_tensor(dtype=int32, shape=[THREADS], init=1)
        r_zeros = self.register_tensor(dtype=int32, shape=[THREADS], init=0)
        r_lane = self.register_tensor(
            dtype=int32, shape=[THREADS], init=lambda i: i
        )

        MSB = uint32(0x80000000)
        ALL = uint32(0xFFFFFFFF)
        BYTE3_DIV = uint32(0x1000000)  # 1 << 24
        BYTE2_DIV = uint32(0x10000)  # 1 << 16
        U256 = uint32(256)

        # --- Phase 0 histogram: top byte (via global_scatter_add) ---
        for offset in self.range(0, seq_len, THREADS):
            r_bits = self.load_global(
                g_logits, offsets=[bs, offset], shape=[THREADS], dims=[1]
            )
            r_is_neg = r_bits >= MSB
            r_mono = self.where(r_is_neg, x=ALL - r_bits, y=r_bits + MSB)
            r_byte3 = self.cast(r_mono / BYTE3_DIV, dtype=int32)
            r_byte3_off = r_byte3 + hist_base

            r_pos = r_lane + offset
            r_in_range = r_pos < seq_len
            r_vals = self.where(r_in_range, x=r_ones, y=0)
            self.atomic.global_scatter_add(
                g_hist_p0, dim=0, indices=r_byte3_off, values=r_vals
            )
        self.sync()

        # --- Threshold ---
        r_hist = self.load_global(
            g_hist_p0, offsets=[hist_base], shape=[RADIX], dims=[0]
        )
        r_total = self.sum(r_hist, keepdim=True)
        r_exc = self.cumsum(r_hist, dim=0, exclusive=True)
        r_down = r_total - r_exc
        r_down_next = r_down - r_hist

        r_k = self.register_tensor(dtype=int32, shape=[RADIX], init=TOP_K)
        r_idx_r = self.register_tensor(
            dtype=int32, shape=[RADIX], init=lambda i: i
        )
        r_false_r = self.register_tensor(
            dtype=boolean, shape=[RADIX], init=False
        )
        r_mask_th = self.where(
            r_down > r_k, x=(r_down_next <= r_k), y=r_false_r
        )
        r_th_cand = self.where(r_mask_th, x=r_idx_r, y=-1)
        r_threshold = self.max(r_th_cand, keepdim=True)

        self.store_shared(s_threshold_bin, r_threshold)
        self.sync()
        threshold_bin: int32 = s_threshold_bin[0].item()

        # --- Phase 0 partition: emit bin>threshold, cache bin==threshold ---
        r_bs_tile = r_zeros + bs
        for offset in self.range(0, seq_len, THREADS):
            r_bits = self.load_global(
                g_logits, offsets=[bs, offset], shape=[THREADS], dims=[1]
            )
            r_is_neg = r_bits >= MSB
            r_mono = self.where(r_is_neg, x=ALL - r_bits, y=r_bits + MSB)
            r_byte3 = self.cast(r_mono / BYTE3_DIV, dtype=int32)

            r_pos = r_lane + offset
            r_in_range = r_pos < seq_len
            r_false_t = self.register_tensor(
                dtype=boolean, shape=[THREADS], init=False
            )

            # --- Emit: bin > threshold ---
            mask_above = self.where(
                r_in_range, x=(r_byte3 > threshold_bin), y=r_false_t
            )
            r_emit_v = self.where(mask_above, x=r_ones, y=r_zeros)
            r_topk_off = self.register_tensor(dtype=int32, shape=[THREADS])
            self.atomic.global_scatter_add(
                g_topk_cnt, dim=0, indices=r_bs_tile, values=r_emit_v,
                output=r_topk_off,
            )
            r_topk_trash = self.register_tensor(
                dtype=int32, shape=[THREADS], init=TOP_K
            )
            r_topk_ok = self.where(
                mask_above, x=(r_topk_off < TOP_K), y=r_false_t
            )
            r_topk_dst = self.where(r_topk_ok, x=r_topk_off, y=r_topk_trash)
            r_topk_dst_off = r_topk_dst + idx_base
            self.store_global_scatter(
                g_indices, dim=0, indices=r_topk_dst_off, values=r_pos
            )

            # --- Cache: bin == threshold ---
            mask_equal = self.where(
                r_in_range, x=(r_byte3 == threshold_bin), y=r_false_t
            )
            r_cache_v = self.where(mask_equal, x=r_ones, y=r_zeros)
            r_cache_off = self.register_tensor(dtype=int32, shape=[THREADS])
            self.atomic.global_scatter_add(
                g_cache_cnt, dim=0, indices=r_bs_tile, values=r_cache_v,
                output=r_cache_off,
            )
            r_cache_lim = r_zeros + (num_cached - 1)
            r_cache_ok = self.where(
                mask_equal, x=(r_cache_off < num_cached), y=r_false_t
            )
            r_cache_dst = self.where(
                r_cache_ok, x=r_cache_off, y=r_cache_lim
            )
            r_cache_dst_off = r_cache_dst + cache_base
            self.store_global_scatter(
                g_cache_idx, dim=0, indices=r_cache_dst_off, values=r_pos
            )
            # Store mono bits (cast pos to uint32 is wrong — need mono bits).
            # We use r_mono which is uint32.
            r_mono_i32 = self.cast(r_mono, dtype=int32)
            r_mono_store = self.where(
                r_cache_ok, x=r_mono_i32, y=0
            )
            # store_global_scatter needs int32 values for int32 global tensor.
            # But cache_bits is uint32. Let's cast the indices for uint32 store.
            # Actually, store mono as uint32 directly.
            self.store_global_scatter(
                g_cache_bits, dim=0, indices=r_cache_dst_off, values=r_mono
            )

            # --- Build phase-1 histogram: byte 2 of cached borderline ---
            r_byte2_u = r_mono / BYTE2_DIV
            r_byte2_masked = r_byte2_u - (r_byte2_u / U256) * U256
            r_byte2 = self.cast(r_byte2_masked, dtype=int32)
            r_byte2_off = r_byte2 + hist_base
            r_hist_v = self.where(r_cache_ok, x=r_ones, y=r_zeros)
            self.atomic.global_scatter_add(
                g_hist0, dim=0, indices=r_byte2_off, values=r_hist_v
            )


# ---------------------------------------------------------------------------
# Phases 1-3: refine cached borderline elements
# ---------------------------------------------------------------------------


class DSV3TopKRefine(tilus.Script):
    """Phase 1/2/3: scan cached elements, emit+cache by next byte."""

    TOP_K = 2048
    RADIX = 256
    THREADS = 1024

    def __init__(self, phase: int):
        super().__init__()
        self.phase = phase

    def __call__(
        self,
        indices_ptr: void_p,  # int32 [B * TOP_K] — output (append)
        hist_in_ptr: void_p,  # int32 [B * RADIX] — current phase histogram
        hist_out_ptr: void_p,  # int32 [B * RADIX] — next phase histogram (built here)
        topk_cnt_ptr: void_p,  # int32 [B] — topk counter (continued)
        cache_bits_in_ptr: void_p,  # uint32 [B * NUM_CACHED] — read cache
        cache_idx_in_ptr: void_p,  # int32  [B * NUM_CACHED] — read cache
        cache_cnt_in_ptr: void_p,  # int32 [B] — read cache count
        cache_bits_out_ptr: void_p,  # uint32 [B * NUM_CACHED] — write cache
        cache_idx_out_ptr: void_p,  # int32  [B * NUM_CACHED] — write cache
        cache_cnt_out_ptr: void_p,  # int32 [B] — write cache count
        batch_size: int32,
        num_cached: int32,
    ):
        TOP_K = self.TOP_K
        RADIX = self.RADIX
        THREADS = self.THREADS
        phase = self.phase  # 1, 2, or 3

        self.attrs.blocks = [batch_size]
        self.attrs.warps = THREADS // 32

        bs: int32 = self.blockIdx.x

        g_indices = self.global_view(
            indices_ptr, dtype=int32, shape=[batch_size * (TOP_K + 1)]
        )
        g_hist_in = self.global_view(
            hist_in_ptr, dtype=int32, shape=[batch_size * RADIX]
        )
        g_hist_out = self.global_view(
            hist_out_ptr, dtype=int32, shape=[batch_size * RADIX]
        )
        g_topk_cnt = self.global_view(
            topk_cnt_ptr, dtype=int32, shape=[batch_size]
        )
        g_bits_in = self.global_view(
            cache_bits_in_ptr, dtype=uint32, shape=[batch_size, num_cached]
        )
        g_idx_in = self.global_view(
            cache_idx_in_ptr, dtype=int32, shape=[batch_size, num_cached]
        )
        g_cnt_in = self.global_view(
            cache_cnt_in_ptr, dtype=int32, shape=[batch_size]
        )
        g_bits_out = self.global_view(
            cache_bits_out_ptr, dtype=uint32, shape=[batch_size * num_cached]
        )
        g_idx_out = self.global_view(
            cache_idx_out_ptr, dtype=int32, shape=[batch_size * num_cached]
        )
        g_cnt_out = self.global_view(
            cache_cnt_out_ptr, dtype=int32, shape=[batch_size]
        )

        hist_base: int32 = bs * RADIX
        idx_base: int32 = bs * (TOP_K + 1)
        cache_base_out: int32 = bs * num_cached

        # Read cached element count from previous phase.
        buf_len: int32 = g_cnt_in[bs].item()

        # --- Threshold from histogram (already built by previous phase) ---
        s_threshold_bin = self.shared_tensor(dtype=int32, shape=[1])

        # Load histogram from global for threshold computation.
        r_hist = self.load_global(
            g_hist_in, offsets=[hist_base], shape=[RADIX], dims=[0]
        )
        r_total = self.sum(r_hist, keepdim=True)
        r_exc = self.cumsum(r_hist, dim=0, exclusive=True)
        r_down = r_total - r_exc
        r_down_next = r_down - r_hist

        # k_remaining = how many more elements we need beyond what's already
        # been emitted.  threshold bin: down[b] > k_rem AND down[b+1] <= k_rem.
        # But we don't know k_remaining easily. Instead, use top_k_remaining
        # which the host can pass. For simplicity, re-derive from histogram:
        # k_remaining = TOP_K - topk_cnt.  But topk_cnt is per-batch and we'd
        # need to read it.  Actually, the reference code accumulates
        # top_k_remaining across phases.  Let me just read topk_cnt.
        topk_so_far: int32 = g_topk_cnt[bs].item()
        k_remaining: int32 = TOP_K - topk_so_far

        r_k_rem = self.register_tensor(
            dtype=int32, shape=[RADIX], init=0
        ) + k_remaining
        r_idx_r = self.register_tensor(
            dtype=int32, shape=[RADIX], init=lambda i: i
        )
        r_false_r = self.register_tensor(
            dtype=boolean, shape=[RADIX], init=False
        )
        r_mask_th = self.where(
            r_down > r_k_rem, x=(r_down_next <= r_k_rem), y=r_false_r
        )
        r_th_cand = self.where(r_mask_th, x=r_idx_r, y=-1)
        r_threshold = self.max(r_th_cand, keepdim=True)

        self.store_shared(s_threshold_bin, r_threshold)
        self.sync()
        threshold_bin: int32 = s_threshold_bin[0].item()

        # --- Byte extraction divisors ---
        # Phase 1: byte 2 → div by 0x10000, mask 0xFF
        # Phase 2: byte 1 → div by 0x100, mask 0xFF
        # Phase 3: byte 0 → mask 0xFF (div by 1)
        if phase == 1:
            CUR_DIV = uint32(0x10000)
            NXT_DIV = uint32(0x100)
        elif phase == 2:
            CUR_DIV = uint32(0x100)
            NXT_DIV = uint32(1)
        else:  # phase == 3
            CUR_DIV = uint32(1)
            NXT_DIV = uint32(1)  # unused
        U256 = uint32(256)

        # --- Tile register constants ---
        r_ones = self.register_tensor(dtype=int32, shape=[THREADS], init=1)
        r_zeros = self.register_tensor(dtype=int32, shape=[THREADS], init=0)
        r_lane = self.register_tensor(
            dtype=int32, shape=[THREADS], init=lambda i: i
        )
        r_bs_tile = r_zeros + bs

        # --- Scan cached elements ---
        for offset in self.range(0, buf_len, THREADS):
            r_pos = r_lane + offset
            r_in_range = r_pos < buf_len
            r_false_t = self.register_tensor(
                dtype=boolean, shape=[THREADS], init=False
            )

            # Load cached bits and indices.
            r_mono = self.load_global(
                g_bits_in, offsets=[bs, offset],
                shape=[THREADS], dims=[1]
            )
            r_orig_idx = self.load_global(
                g_idx_in, offsets=[bs, offset],
                shape=[THREADS], dims=[1]
            )

            # Extract current byte.
            r_shifted = r_mono / CUR_DIV
            r_byte_masked = r_shifted - (r_shifted / U256) * U256
            r_byte = self.cast(r_byte_masked, dtype=int32)

            # --- Emit: bin > threshold ---
            mask_above = self.where(
                r_in_range, x=(r_byte > threshold_bin), y=r_false_t
            )
            r_emit_v = self.where(mask_above, x=r_ones, y=r_zeros)
            r_topk_off = self.register_tensor(dtype=int32, shape=[THREADS])
            self.atomic.global_scatter_add(
                g_topk_cnt, dim=0, indices=r_bs_tile, values=r_emit_v,
                output=r_topk_off,
            )
            r_topk_trash = self.register_tensor(
                dtype=int32, shape=[THREADS], init=TOP_K
            )
            r_topk_ok = self.where(
                mask_above, x=(r_topk_off < TOP_K), y=r_false_t
            )
            r_topk_dst = self.where(r_topk_ok, x=r_topk_off, y=r_topk_trash)
            r_topk_dst_off = r_topk_dst + idx_base
            self.store_global_scatter(
                g_indices, dim=0, indices=r_topk_dst_off, values=r_orig_idx
            )

            # --- Cache: bin == threshold (phases 1-2 only) ---
            if phase < 3:
                mask_equal = self.where(
                    r_in_range, x=(r_byte == threshold_bin), y=r_false_t
                )
                r_cache_v = self.where(mask_equal, x=r_ones, y=r_zeros)
                r_cache_off = self.register_tensor(dtype=int32, shape=[THREADS])
                self.atomic.global_scatter_add(
                    g_cnt_out, dim=0, indices=r_bs_tile, values=r_cache_v,
                    output=r_cache_off,
                )
                r_cache_lim = r_zeros + (num_cached - 1)
                r_cache_ok = self.where(
                    mask_equal, x=(r_cache_off < num_cached), y=r_false_t
                )
                r_cache_dst = self.where(
                    r_cache_ok, x=r_cache_off, y=r_cache_lim
                )
                r_cache_dst_off = r_cache_dst + cache_base_out
                self.store_global_scatter(
                    g_idx_out, dim=0, indices=r_cache_dst_off,
                    values=r_orig_idx
                )
                self.store_global_scatter(
                    g_bits_out, dim=0, indices=r_cache_dst_off,
                    values=r_mono
                )

                # Build next-phase histogram.
                r_nxt_shifted = r_mono / NXT_DIV
                r_nxt_masked = r_nxt_shifted - (r_nxt_shifted / U256) * U256
                r_nxt_byte = self.cast(r_nxt_masked, dtype=int32)
                r_nxt_byte_off = r_nxt_byte + hist_base
                r_hist_v = self.where(r_cache_ok, x=r_ones, y=r_zeros)
                self.atomic.global_scatter_add(
                    g_hist_out, dim=0, indices=r_nxt_byte_off,
                    values=r_hist_v
                )

            # --- Phase 3 final fill: bin == threshold → fill remaining ---
            if phase == 3:
                mask_equal = self.where(
                    r_in_range, x=(r_byte == threshold_bin), y=r_false_t
                )
                r_fill_v = self.where(mask_equal, x=r_ones, y=r_zeros)
                r_fill_off = self.register_tensor(dtype=int32, shape=[THREADS])
                self.atomic.global_scatter_add(
                    g_topk_cnt, dim=0, indices=r_bs_tile, values=r_fill_v,
                    output=r_fill_off,
                )
                r_fill_ok = self.where(
                    mask_equal, x=(r_fill_off < TOP_K), y=r_false_t
                )
                r_fill_dst = self.where(
                    r_fill_ok, x=r_fill_off, y=r_topk_trash
                )
                r_fill_dst_off = r_fill_dst + idx_base
                self.store_global_scatter(
                    g_indices, dim=0, indices=r_fill_dst_off,
                    values=r_orig_idx
                )


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------


def dsv3_topk_tilus(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    topk: int = 2048,
) -> torch.Tensor:
    """Run the full 4-phase Tilus radix top-K kernel."""
    assert topk == DSV3TopKPhase0.TOP_K
    batch_size, max_model_len = logits.shape
    THREADS = DSV3TopKPhase0.THREADS
    RADIX = DSV3TopKPhase0.RADIX
    assert max_model_len % THREADS == 0

    num_cached = max_model_len  # worst case: all elements in one bin
    device = logits.device

    # Allocate TOP_K + 1 slots per batch: slot TOP_K is the scatter trash bin.
    indices = torch.full(
        (batch_size * (topk + 1),), -1, dtype=torch.int32, device=device
    )
    topk_cnt = torch.zeros(batch_size, dtype=torch.int32, device=device)

    # Double-buffered cache (A and B).
    cache_bits_a = torch.zeros(batch_size * num_cached, dtype=torch.int32, device=device)
    cache_idx_a = torch.zeros(batch_size * num_cached, dtype=torch.int32, device=device)
    cache_cnt_a = torch.zeros(batch_size, dtype=torch.int32, device=device)
    cache_bits_b = torch.zeros(batch_size * num_cached, dtype=torch.int32, device=device)
    cache_idx_b = torch.zeros(batch_size * num_cached, dtype=torch.int32, device=device)
    cache_cnt_b = torch.zeros(batch_size, dtype=torch.int32, device=device)

    # Phase-0 histogram (byte 3, temporary).
    hist_p0 = torch.zeros(batch_size * RADIX, dtype=torch.int32, device=device)
    # Phase-1 histogram (built by phase 0 partition).
    hist_1 = torch.zeros(batch_size * RADIX, dtype=torch.int32, device=device)
    # Phase-2 histogram (built by phase 1).
    hist_2 = torch.zeros(batch_size * RADIX, dtype=torch.int32, device=device)
    # Phase-3 histogram (built by phase 2).
    hist_3 = torch.zeros(batch_size * RADIX, dtype=torch.int32, device=device)

    # Phase 0: full scan → emit + cache to buffer A.
    DSV3TopKPhase0()(
        logits.data_ptr(),
        seq_lens.data_ptr(),
        indices.data_ptr(),
        hist_p0.data_ptr(),
        hist_1.data_ptr(),
        topk_cnt.data_ptr(),
        cache_bits_a.data_ptr(),
        cache_idx_a.data_ptr(),
        cache_cnt_a.data_ptr(),
        batch_size,
        max_model_len,
        num_cached,
    )

    # Phase 1: read A → emit + cache to B, build hist_2.
    DSV3TopKRefine(phase=1)(
        indices.data_ptr(),
        hist_1.data_ptr(),
        hist_2.data_ptr(),
        topk_cnt.data_ptr(),
        cache_bits_a.data_ptr(),
        cache_idx_a.data_ptr(),
        cache_cnt_a.data_ptr(),
        cache_bits_b.data_ptr(),
        cache_idx_b.data_ptr(),
        cache_cnt_b.data_ptr(),
        batch_size,
        num_cached,
    )

    # Phase 2: read B → emit + cache to A, build hist_3.
    cache_cnt_a.zero_()
    DSV3TopKRefine(phase=2)(
        indices.data_ptr(),
        hist_2.data_ptr(),
        hist_3.data_ptr(),
        topk_cnt.data_ptr(),
        cache_bits_b.data_ptr(),
        cache_idx_b.data_ptr(),
        cache_cnt_b.data_ptr(),
        cache_bits_a.data_ptr(),
        cache_idx_a.data_ptr(),
        cache_cnt_a.data_ptr(),
        batch_size,
        num_cached,
    )

    # Phase 3: read A → emit + fill remaining. No cache output.
    DSV3TopKRefine(phase=3)(
        indices.data_ptr(),
        hist_3.data_ptr(),
        hist_3.data_ptr(),  # unused output hist
        topk_cnt.data_ptr(),
        cache_bits_a.data_ptr(),
        cache_idx_a.data_ptr(),
        cache_cnt_a.data_ptr(),
        cache_bits_a.data_ptr(),  # unused output cache
        cache_idx_a.data_ptr(),
        cache_cnt_a.data_ptr(),
        batch_size,
        num_cached,
    )

    # Drop the trash slot, reshape to [B, TOP_K].
    return indices.reshape(batch_size, topk + 1)[:, :topk].contiguous()


def _reference_topk(
    logits: torch.Tensor, seq_lens: torch.Tensor, topk: int = 2048
) -> torch.Tensor:
    B = logits.shape[0]
    out = torch.full((B, topk), -1, dtype=torch.int32, device=logits.device)
    for b in range(B):
        L = int(seq_lens[b].item())
        k = min(topk, L)
        _, idx = torch.topk(logits[b, :L], k)
        out[b, :k] = idx.to(torch.int32)
    return out


def main():
    torch.manual_seed(0)

    cases = [
        (1, 4096),
        (1, 8192),
        (1, 16384),
        (1, 32768),
        (2, 8192),
        (4, 16384),
        (8, 16384),
    ]
    for batch_size, seq_len in cases:
        max_model_len = (seq_len + 1023) // 1024 * 1024
        logits = torch.randn(
            batch_size, max_model_len, dtype=torch.float32, device="cuda"
        )
        for b in range(batch_size):
            logits[b, seq_len:] = torch.finfo(torch.float32).min
        seq_lens = torch.full(
            (batch_size,), seq_len, dtype=torch.int32, device="cuda"
        )

        indices = dsv3_topk_tilus(logits, seq_lens, topk=2048)
        ref = _reference_topk(logits, seq_lens, topk=2048)

        for b in range(batch_size):
            got = set(indices[b].cpu().tolist())
            expected = set(ref[b].cpu().tolist())
            got.discard(-1)
            expected.discard(-1)
            overlap = len(got & expected)
            recall = overlap / len(expected) if expected else 1.0
            print(
                f"[batch={batch_size}, seq_len={seq_len}, b={b}]  "
                f"|topk|={len(got)}  overlap={overlap}/{len(expected)}  "
                f"recall={recall:.1%}"
            )


if __name__ == "__main__":
    main()
