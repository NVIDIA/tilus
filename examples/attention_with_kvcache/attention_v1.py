# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math
import numpy as np
import pandas as pd
import tilus
import torch
import einops
from hidet.ir import DataType
from tilus import boolean, f32, f16, int32, void_p
from tilus.utils import benchmark_func, cdiv


class AttentionWithKVCache(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_q = 64
        self.block_kv = 64

    def __call__(
        self,
        q_ptr: ~f16,
        k_cache_ptr: ~f16,
        v_cache_ptr: ~f16,
        cache_seqlens_ptr: ~int32,
        block_table: ~int32,

        batch_size: int,
        seqlen_q: int32,
        num_heads: int,
        head_size: int,
        num_blocks: int32,
        page_block_size: int,
        num_heads_kv: int,
        max_num_blocks_per_seq: int32,
    ):
        self.attrs.blocks = [
            cdiv(seqlen_q, self.block_q),
            num_heads,
            batch_size,
        ]
        self.attrs.warps = 4

        group_heads: int = num_heads // num_heads_kv
        q_offset: int32 = self.blockIdx.x * self.block_q
        head: int32 = self.blockIdx.y
        bs: int32 = self.blockIdx.z

        g_q = self.global_view(q_ptr, dtype=f16, shape=[batch_size, seqlen_q, num_heads, head_size])
        g_k_cache = self.global_view(k_cache_ptr, dtype=f16, shape=[num_blocks, page_block_size, num_heads_kv, head_size])
        g_v_cache = self.global_view(v_cache_ptr, dtype=f16, shape=[num_blocks, page_block_size, num_heads_kv, head_size])
        g_cache_seqlens = self.global_view(cache_seqlens_ptr, dtype=int32, shape=[batch_size])
        g_block_table = self.global_view(block_table, dtype=int32, shape=[batch_size, max_num_blocks_per_seq])

        # load query to register
        s_q = self.shared_tensor(dtype=f16, shape=[self.block_q, head_size])
        self.store_shared(
            dst=s_q,
            src=self.load_global(
                g_q,
                offsets=[bs, q_offset, head, 0],
                shape=[self.block_q, head_size],
                dims=[1, 3]
            )
        )
        self.sync()
        r_q = self.load_shared(s_q)
        self.sync()
        self.free_shared(s_q)

        # accumulators
        r_o = self.register_tensor(dtype=f16, shape=[self.block_q, head_size], init=0.0)
        r_m = self.register_tensor(dtype=f16, shape=[self.block_q], init=-1e6)    # rowmax(score)
        r_l = self.register_tensor(dtype=f16, shape=[self.block_q], init=0.0)     # rowsum(exp(score - m))

        s_k = self.shared_tensor(dtype=f16, shape=[self.block_kv, head_size])
        s_v = self.shared_tensor(dtype=f16, shape=[self.block_kv, head_size])

        self.copy_async(src=g_k_cache, dst=s_k, offsets=[g_block_table[bs, 0], 0, head // group_heads, 0], dims=[1, 3])
        self.copy_async_commit_group()

        # we split the kv sequence into two parts, the first part do not need to apply causal mask
        kv_offset_no_causal_end: int32 = g_cache_seqlens[bs] - seqlen_q
        kv_offset_end: int32 = g_cache_seqlens[bs]


def attention_with_kvcache_tilus(
    q: torch.Tensor,  # fp16[batch_size, seqlen, num_heads, head_size]
    k_cache: torch.Tensor,  # fp16[num_blocks, page_block_size, num_heads_kv, head_size]
    v_cache: torch.Tensor,  # fp16[num_blocks, page_block_size, num_heads_kv, head_size]
    cache_seqlens: torch.Tensor,  # int32[batch_size]
    block_table: torch.Tensor,  # int32[batch_size, max_num_blocks_per_seq]
):
    pass

def attention_with_kvcache_reference(
    q: torch.Tensor,    # fp16[batch_size, seqlen, num_heads, head_size]
    k_cache: torch.Tensor,  # fp16[num_blocks, page_block_size, num_heads_kv, head_size]
    v_cache: torch.Tensor,  # fp16[num_blocks, page_block_size, num_heads_kv, head_size]
    cache_seqlens: torch.Tensor,  # int32[batch_size]
    block_table: torch.Tensor,  # int32[batch_size, max_num_blocks_per_seq]
):
    original_dtype = q.dtype

    q, k_cache, v_cache = q.float(), k_cache.float(), v_cache.float()

    head_size = q.size(3)
    batch_size = q.size(0)
    groups = q.size(2) // k_cache.size(2)
    k = einops.rearrange(k_cache[block_table.flatten()], pattern='(b nblocks) block_size ... -> b (nblocks block_size) ...', b=batch_size)
    v = einops.rearrange(v_cache[block_table.flatten()], pattern='(b nblocks) block_size ... -> b (nblocks block_size) ...', b=batch_size)
    k = einops.repeat(k, 'b s h d -> b s (h g) d', g=groups)
    v = einops.repeat(v, 'b s h d -> b s (h g) d', g=groups)
    scores = torch.einsum('bthd,bshd->bhts', q / math.sqrt(head_size), k)   # [batch_size, heads, seqlen_q, seqlen_k]

    seqlen_q, seqlen_k = q.size(1), k.size(1)
    col_idx = torch.arange(seqlen_k, dtype=torch.long, device='cuda').unsqueeze(0)  # [1, seqlen_k]
    mask = einops.rearrange(col_idx < cache_seqlens.unsqueeze(1), 'b s -> b 1 1 s')  # [batch_size, 1, 1, seqlen_k]
    scores = scores.masked_fill(~mask, float('-inf'))
    attention = torch.softmax(scores, dim=-1)   # [batch_size, heads, seqlen_q, seqlen_k]

    output = torch.einsum('bhts,bshd->bthd', attention, v)
    output = output.to(original_dtype)

    return output


def attention_with_kvcache_flash_attention(
    q: torch.Tensor,    # [batch_size, seqlen, num_heads, head_size]
    k_cache: torch.Tensor,  # [num_blocks, page_block_size, num_heads_kv, head_size]
    v_cache: torch.Tensor,  # [num_blocks, page_block_size, num_heads_kv, head_size]
    cache_seqlens: torch.Tensor,  # [1]
    block_table: torch.Tensor,  # int32[batch_size, max_num_blocks_per_seq]
):
    from flash_attn.flash_attn_interface import flash_attn_with_kvcache
    return flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True
    )



def main():
    dtype = torch.float16
    block_size = 256

    for batch_size, seqlen_q, seqlen_kv, num_heads, head_size, num_heads_kv in [
        [16, 1, 512, 32, 128, 8],
    ]:
        num_blocks = cdiv(seqlen_kv, block_size) * batch_size * 2

        q = torch.randn(batch_size, seqlen_q, num_heads, head_size, dtype=dtype, device='cuda')
        k_cache = torch.randn(num_blocks, block_size, num_heads_kv, head_size, dtype=dtype, device='cuda')
        v_cache = torch.randn(num_blocks, block_size, num_heads_kv, head_size, dtype=dtype, device='cuda')
        cache_seqlens = torch.randint(1, seqlen_kv + 1, (batch_size,), dtype=torch.int32, device='cuda')    # [batch_size]
        block_table = einops.rearrange(
            torch.randperm(num_blocks, dtype=torch.int32, device='cuda'),
            "(b nblocks) -> b nblocks",
            b=batch_size,
        )   # [batch_size, max_num_blocks_per_seq]

        for name, runner in [
            ("flash-attn", attention_with_kvcache_flash_attention),
            # ("tilus", attention_with_kvcache_tilus),
        ]:
            actual = runner(q, k_cache, v_cache, cache_seqlens, block_table)
            expected = attention_with_kvcache_reference(q, k_cache, v_cache, cache_seqlens, block_table)

            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    main()
