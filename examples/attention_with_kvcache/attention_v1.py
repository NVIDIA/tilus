# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import einops
import pandas as pd
import tilus
import torch
from tilus import boolean, f16, f32, int32
from tilus.utils import benchmark_func, cdiv

tilus.option.cache_dir("./cache")

pd.options.display.max_columns = None
pd.options.display.width = 1000


@tilus.autotune("block_q", [16, 32, 64])
@tilus.autotune("block_kv", [64, 128])
@tilus.autotune("num_warps", [4])
class AttentionWithKVCache(tilus.Script):
    def __init__(self, head_size: int, block_q: int, block_kv: int, num_warps: int):
        super().__init__()
        self.num_warps = num_warps
        self.block_q = block_q
        self.block_kv = block_kv
        self.head_size = head_size

        self.r_o_layout = self.cuda.resolve_dot_config(
            operand_dtype=f16,
            acc_dtype=f32,
            m=self.block_q,
            n=self.head_size,
            k=self.block_kv,
            warp_m=num_warps,
            warp_n=1,
        ).lc

    def __call__(
        self,
        q_ptr: ~f16,
        k_cache_ptr: ~f16,
        v_cache_ptr: ~f16,
        o_ptr: ~f16,
        cache_seqlens_ptr: ~int32,
        block_table: ~int32,
        batch_size: int,
        seqlen_q: int32,
        num_heads: int,
        num_blocks: int32,
        page_block_size: int,
        num_heads_kv: int,
        max_num_blocks_per_seq: int32,
    ):
        self.static_assert(
            page_block_size % self.block_kv == 0,
            "page_block_size must be multiple of block_kv",
        )

        self.attrs.blocks = [
            cdiv(seqlen_q, self.block_q),
            num_heads,
            batch_size,
        ]
        self.attrs.warps = self.num_warps

        group_heads: int = num_heads // num_heads_kv
        q_offset: int32 = self.blockIdx.x * self.block_q
        head: int32 = self.blockIdx.y
        bs: int32 = self.blockIdx.z

        g_q = self.global_view(
            q_ptr, dtype=f16, shape=[batch_size, seqlen_q, num_heads, self.head_size]
        )
        g_k_cache = self.global_view(
            k_cache_ptr,
            dtype=f16,
            shape=[num_blocks, page_block_size, num_heads_kv, self.head_size],
        )
        g_v_cache = self.global_view(
            v_cache_ptr,
            dtype=f16,
            shape=[num_blocks, page_block_size, num_heads_kv, self.head_size],
        )
        g_cache_seqlens = self.global_view(
            cache_seqlens_ptr, dtype=int32, shape=[batch_size]
        )
        g_block_table = self.global_view(
            block_table, dtype=int32, shape=[batch_size, max_num_blocks_per_seq]
        )
        g_o = self.global_view(
            o_ptr, dtype=f16, shape=[batch_size, seqlen_q, num_heads, self.head_size]
        )

        # load query to register
        s_q = self.shared_tensor(dtype=f16, shape=[self.block_q, self.head_size])
        self.store_shared(
            dst=s_q,
            src=self.load_global(
                g_q,
                offsets=[bs, q_offset, head, 0],
                shape=[self.block_q, self.head_size],
                dims=[1, 3],
            ),
        )
        self.sync()
        r_q = self.load_shared(s_q)
        self.sync()
        self.free_shared(s_q)

        # accumulators
        r_o = self.register_tensor(
            dtype=f32, shape=[self.block_q, self.head_size], init=0.0
        )
        r_m = self.register_tensor(
            dtype=f32, shape=[self.block_q, 1], init=-1e6
        )  # rowmax(score)
        r_l = self.register_tensor(
            dtype=f32, shape=[self.block_q, 1], init=0.0
        )  # rowsum(exp(score - m))

        s_k = self.shared_tensor(dtype=f16, shape=[self.block_kv, self.head_size])
        s_v = self.shared_tensor(dtype=f16, shape=[self.block_kv, self.head_size])

        self.copy_async(
            src=g_k_cache,
            dst=s_k,
            offsets=[g_block_table[bs, 0].item(), 0, head // group_heads, 0],
            dims=[1, 3],
        )
        self.copy_async_commit_group()

        q_left_len = g_cache_seqlens[bs].item() - seqlen_q
        kv_offset_end: int32 = q_left_len + q_offset + self.block_q
        for kv_offset in self.range(0, kv_offset_end, self.block_kv):
            self.copy_async_wait_group(0)
            self.sync()
            self.copy_async(
                src=g_v_cache,
                dst=s_v,
                offsets=[
                    g_block_table[bs, kv_offset // page_block_size].item(),
                    kv_offset % page_block_size,
                    head // group_heads,
                    0,
                ],
                dims=[1, 3],
            )
            self.copy_async_commit_group()

            r_k = self.load_shared(s_k)  # [block_kv, head_size]
            score = self.dot(r_q, r_k.transpose(), acc_dtype=f32) * math.sqrt(
                1.0 / self.head_size
            )  # [block_q, block_kv]
            mask = self.register_tensor(
                dtype=boolean,
                shape=[self.block_q, self.block_kv],
                init=lambda i, j: i + q_offset + q_left_len >= j + kv_offset,
            )
            score = score + self.where(mask, x=0.0, y=-1e6)

            self.copy_async_wait_group(0)
            self.sync()
            preload_kv_offset = kv_offset + self.block_kv
            self.copy_async(
                src=g_k_cache,
                dst=s_k,
                offsets=[
                    g_block_table[bs, preload_kv_offset // page_block_size].item(),
                    preload_kv_offset % page_block_size,
                    head // group_heads,
                    0,
                ],
                dims=[1, 3],
            )
            self.copy_async_commit_group()

            r_v = self.load_shared(s_v)

            # online softmax
            r_cur_m = self.max(score, dim=1, keepdim=True)  # [block_q, 1]
            r_new_m = self.maximum(r_m, r_cur_m)  # [block_q, 1]
            r_p = self.exp(score - r_new_m)  # [block_q, block_kv]
            r_cur_o = self.dot(
                r_p.to(r_v.dtype), r_v, acc_dtype=f32
            )  # [block_q, head_size]
            r_o = r_o * self.exp(r_m - r_new_m) + r_cur_o  # [block_q, head_size]
            r_l = r_l * self.exp(r_m - r_new_m) + self.sum(
                r_p, dim=1, keepdim=True
            )  # [block_q, 1]
            r_m = r_new_m  # [block_q, 1]

            self.annotate_layout(r_o, self.r_o_layout)

        self.copy_async_wait_group(0)
        self.sync()
        self.free_shared(s_k)
        self.free_shared(s_v)

        r_o = r_o / r_l
        r_o_f16 = self.cast(r_o, dtype=f16)
        s_o = self.shared_tensor(dtype=f16, shape=[self.block_q, self.head_size])
        self.store_shared(s_o, r_o_f16)
        self.sync()
        self.store_global(
            g_o, self.load_shared(s_o), offsets=[bs, q_offset, head, 0], dims=[1, 3]
        )
        self.free_shared(s_o)


def attention_with_kvcache_tilus(
    q: torch.Tensor,  # fp16[batch_size, seqlen, num_heads, head_size]
    k_cache: torch.Tensor,  # fp16[num_blocks, page_block_size, num_heads_kv, head_size]
    v_cache: torch.Tensor,  # fp16[num_blocks, page_block_size, num_heads_kv, head_size]
    cache_seqlens: torch.Tensor,  # int32[batch_size]
    block_table: torch.Tensor,  # int32[batch_size, max_num_blocks_per_seq]
):
    o = torch.empty_like(q)
    batch_size, seqlen_q, num_heads, head_size = q.size()
    num_blocks, page_block_size, num_heads_kv, _ = k_cache.size()
    max_num_blocks_per_seq = block_table.size(1)
    AttentionWithKVCache(head_size)(
        q,
        k_cache,
        v_cache,
        o,
        cache_seqlens,
        block_table,
        batch_size,
        seqlen_q,
        num_heads,
        num_blocks,
        page_block_size,
        num_heads_kv,
        max_num_blocks_per_seq,
    )
    return o


def attention_with_kvcache_reference(
    q: torch.Tensor,  # fp16[batch_size, seqlen, num_heads, head_size]
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
    k = einops.rearrange(
        k_cache[block_table.flatten()],
        pattern="(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )
    v = einops.rearrange(
        v_cache[block_table.flatten()],
        pattern="(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )
    k = einops.repeat(k, "b s h d -> b s (h g) d", g=groups)
    v = einops.repeat(v, "b s h d -> b s (h g) d", g=groups)
    scores = torch.einsum(
        "bthd,bshd->bhts", q / math.sqrt(head_size), k
    )  # [batch_size, heads, seqlen_q, seqlen_k]

    seqlen_q, seqlen_k = q.size(1), k.size(1)

    # Sequence length mask: only attend to valid tokens in cache
    col_idx = torch.arange(seqlen_k, dtype=torch.long, device="cuda").unsqueeze(
        0
    )  # [1, seqlen_k]
    seq_mask = einops.rearrange(
        col_idx < cache_seqlens.unsqueeze(1), "b s -> b 1 1 s"
    )  # [batch_size, 1, 1, seqlen_k]

    # Causal mask: query position i can attend to all cache tokens plus query tokens up to position i
    # In KV cache, new query tokens are appended after existing cache tokens
    row_idx = torch.arange(seqlen_q, dtype=torch.long, device="cuda").unsqueeze(
        1
    )  # [seqlen_q, 1]
    col_idx = torch.arange(seqlen_k, dtype=torch.long, device="cuda").unsqueeze(
        0
    )  # [1, seqlen_k]

    # Query token i (at absolute position cache_seqlens - seqlen_q + i) can attend to:
    # - All cache tokens (positions 0 to cache_seqlens - seqlen_q - 1)
    # - Query tokens up to position i (positions cache_seqlens - seqlen_q to cache_seqlens - seqlen_q + i)
    cache_start_pos = (
        cache_seqlens.unsqueeze(1).unsqueeze(1) - seqlen_q
    )  # [batch_size, 1, 1]
    query_abs_pos = cache_start_pos + row_idx.unsqueeze(0)  # [batch_size, seqlen_q, 1]
    causal_mask = col_idx <= query_abs_pos  # [batch_size, seqlen_q, seqlen_k]
    causal_mask = causal_mask.unsqueeze(1)  # [batch_size, 1, seqlen_q, seqlen_k]

    # Combine both masks
    combined_mask = seq_mask & causal_mask
    scores = scores.masked_fill(~combined_mask, float("-inf"))
    attention = torch.softmax(scores, dim=-1)  # [batch_size, heads, seqlen_q, seqlen_k]
    attention = attention.to(original_dtype)

    output = torch.einsum("bhts,bshd->bthd", attention, v.to(original_dtype))
    output = output.to(original_dtype)

    return output


def attention_with_kvcache_flash_attention(
    q: torch.Tensor,  # [batch_size, seqlen, num_heads, head_size]
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
        causal=True,
    )


def main():
    headers = [
        "batch_size",
        "seqlen_q",
        "sum_seqlen_kv",
        "num_heads",
        "head_size",
        "num_heads_kv",
        "name",
        "latency (ms)",
        "tflops",
    ]
    data = []

    dtype = torch.float16
    block_size = 256
    for batch_size, seqlen_q, max_seqlen_kv, num_heads, head_size, num_heads_kv in [
        [1, 4096, 4096, 32, 128, 8],
        [1, 1024, 4096, 32, 128, 8],
        [16, 1, 4096, 32, 128, 8],
    ]:
        num_blocks = cdiv(max_seqlen_kv, block_size) * batch_size

        q = torch.rand(
            batch_size, seqlen_q, num_heads, head_size, dtype=dtype, device="cuda"
        )
        k_cache = torch.rand(
            num_blocks, block_size, num_heads_kv, head_size, dtype=dtype, device="cuda"
        )
        v_cache = torch.rand(
            num_blocks, block_size, num_heads_kv, head_size, dtype=dtype, device="cuda"
        )
        # Set up realistic KV cache scenario: existing cache + new query tokens
        existing_cache_len = max_seqlen_kv - seqlen_q  # tokens already in cache
        cache_seqlens = torch.tensor(
            [existing_cache_len + seqlen_q] * batch_size, dtype=torch.int32, device="cuda"
        )  # total after adding query
        block_table = einops.rearrange(
            torch.randperm(num_blocks, dtype=torch.int32, device="cuda"),
            "(b nblocks) -> b nblocks",
            b=batch_size,
        )  # [batch_size, max_num_blocks_per_seq]

        for name, runner in [
            ("flash-attn", attention_with_kvcache_flash_attention),
            ("tilus", attention_with_kvcache_tilus),
        ]:
            print(name)
            actual = runner(q, k_cache, v_cache, cache_seqlens, block_table)
            expected = attention_with_kvcache_reference(
                q, k_cache, v_cache, cache_seqlens, block_table
            )

            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

            latency = benchmark_func(
                lambda: runner(q, k_cache, v_cache, cache_seqlens, block_table),
                warmup=10,
                repeat=100,
            )
            num_tflops = (
                (2 * seqlen_q * num_heads * head_size)
                * torch.sum(cache_seqlens).item()
                / (latency * 1e9)
            )
            data.append(
                [
                    batch_size,
                    seqlen_q,
                    torch.sum(cache_seqlens).item(),
                    num_heads,
                    head_size,
                    num_heads_kv,
                    name,
                    latency,
                    num_tflops,
                ]
            )

    df = pd.DataFrame(data, columns=headers)
    print(df)


if __name__ == "__main__":
    main()
