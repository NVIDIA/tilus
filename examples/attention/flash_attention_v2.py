import importlib.util

import numpy as np
import pandas as pd
import tilus
import torch
from hidet.ir import DataType
from tilus import boolean, f32, int32, void_p
from tilus.ir import RegisterTensor, SharedTensor
from tilus.ir.tensor import GlobalTensor
from tilus.utils import benchmark_func, cdiv

tilus.option.cache_dir("./cache")
# tilus.option.debug.dump_ir()
# tilus.logging.set_logging_level("debug")
# tilus.utils.clear_cache()
pd.options.display.max_columns = None
pd.options.display.width = 1000


@tilus.autotune("num_warps", [2, 4, 8])
@tilus.autotune("block_q", [32, 64, 128])
@tilus.autotune("block_kv", [32, 64, 128])
class FlashAttention(tilus.Script):
    def __init__(
        self,
        dtype: DataType,
        num_heads: int,
        num_heads_kv: int,
        head_size: int,
        num_warps: int,
        block_q: int,
        block_kv: int,
    ):
        super().__init__()
        self.dtype: DataType = dtype
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.head_size = head_size
        self.num_warps = num_warps
        self.block_q = block_q
        self.block_kv = block_kv
        self.score_scale = float(1.0 / np.sqrt(head_size))
        self.group_heads = num_heads // num_heads_kv

        # determine layout
        self.qk_config = self.cuda.resolve_dot_config(
            dtype,
            f32,
            m=block_q,
            n=block_kv,
            k=head_size,
            warp_m=num_warps,
            warp_n=1,
        )
        self.sv_config = self.cuda.resolve_dot_config(
            dtype,
            f32,
            m=block_q,
            n=head_size,
            k=block_kv,
            warp_m=num_warps,
            warp_n=1,
        )
        assert self.qk_config.lc == self.sv_config.la

    def apply_mask(self, score: RegisterTensor, q_offset: int32, kv_offset: int32):
        mask = self.register_tensor(
            dtype=boolean,
            shape=[self.block_q, self.block_kv],
            init=lambda ij: ij[0] + q_offset >= ij[1] + kv_offset,
        )
        self.assign(score, score + self.where(mask, x=0.0, y=-1e6))

    def softmax_rescale(
        self,
        score: RegisterTensor,
        m: RegisterTensor,
        l: RegisterTensor,
        o: RegisterTensor,
    ) -> RegisterTensor:
        """
        o: f32[block_q, head_size]
        m: f32[block_q, 1]  # rowmax(score)
        l: f32[block_q, 1]  # rowsum(exp(score - m))
        score: f32[block_q, block_kv]
        """
        scale = self.score_scale * 1.4426950408889634  # log2(e) * score_scale
        cur_m = self.max(score, dim=1, keepdim=True) * scale  # [block_q, 1]
        new_m = self.maximum(m, cur_m)  # [block_q, 1]
        rp = self.exp2(score * scale - new_m)  # [block_q, block_kv]
        m_scale = self.exp2(m - new_m)
        self.assign(o, o * m_scale)
        self.assign(l, l * m_scale + self.sum(rp, dim=1, keepdim=True))
        self.assign(m, new_m)
        return rp.to(self.dtype)

    def attention_iteration(
        self,
        bs: int32,
        kv_offset: int32,
        q_offset: int32,
        head: int32,
        gk: GlobalTensor,
        gv: GlobalTensor,
        rq: RegisterTensor,  # f16[block_q, head_size]
        sk: SharedTensor,  # f16[block_kv, head_size],
        sv: SharedTensor,  # f16[block_kv, head_size],
        o: RegisterTensor,  # f32[block_q, head_size]
        m: RegisterTensor,  # f32[block_q, 1]
        l: RegisterTensor,  # f32[block_q, 1]
        check_bounds: bool,
    ):
        # wait for the async copy of k to finish
        self.copy_async_wait_group(0)
        self.sync()
        self.copy_async(
            gv,
            sv,
            offsets=[bs, kv_offset, head // self.group_heads, 0],
            dims=[1, 3],
            check_bounds=check_bounds,
        )
        self.copy_async_commit_group()

        # issue the async copy for v and perform dot(q, k)
        rk = self.load_shared(sk)  # [block_kv, head_size]
        score = self.dot(rq, rk.transpose(), acc_dtype=f32)  # [block_q, block_kv]
        self.annotate_layout(score, self.qk_config.lc)

        if check_bounds:
            self.apply_mask(score, q_offset, kv_offset)  # apply causal mask

        # wait for the async copy of v to finish
        self.copy_async_wait_group(0)
        self.sync()
        self.copy_async(
            gk,
            sk,
            offsets=[
                bs,
                kv_offset + self.block_kv,
                head // self.group_heads,
                0,
            ],
            dims=[1, 3],
            check_bounds=check_bounds,
        )
        self.copy_async_commit_group()

        # load v to register
        rv = self.load_shared(sv)  # [block_kv, head_size]

        # online softmax
        rp = self.softmax_rescale(score, m=m, l=l, o=o)

        # pv
        cur_o = self.dot(rp, rv, acc_dtype=f32)  # [block_q, head_size]
        self.annotate_layout(cur_o, self.sv_config.lc)
        self.assign(o, o + cur_o)

    def __call__(
        self,
        batch_size: int,
        seqlen: int32,
        q_ptr: void_p,
        k_ptr: void_p,
        v_ptr: void_p,
        o_ptr: void_p,
    ):
        """
        ```
            load query to register
            cp_async k
            cp_async_fence
            for kv tile:
                cp_async_wait(0)
                sync()
                cp_async v
                cp_async_fence

                score = mma(q, k)
                apply mask to score

                cp_async_wait(0)
                sync()
                cp_async k
                cp_async_fence

                p = apply online softmax to score
                o = mma(p, v)
        ```
        """
        self.attrs.warps = self.num_warps
        self.attrs.blocks = (
            cdiv(seqlen, self.block_q),
            self.num_heads,
            batch_size,
        )

        q_offset = self.blockIdx.x * self.block_q
        head = self.blockIdx.y
        bs = self.blockIdx.z

        gq = self.global_view(
            q_ptr,
            dtype=self.dtype,
            shape=[batch_size, seqlen, self.num_heads, self.head_size],
        )
        gk = self.global_view(
            k_ptr,
            dtype=self.dtype,
            shape=[batch_size, seqlen, self.num_heads_kv, self.head_size],
        )
        gv = self.global_view(
            v_ptr,
            dtype=self.dtype,
            shape=[batch_size, seqlen, self.num_heads_kv, self.head_size],
        )
        go = self.global_view(
            o_ptr,
            dtype=self.dtype,
            shape=[batch_size, seqlen, self.num_heads, self.head_size],
        )

        # load query to register
        sq = self.shared_tensor(dtype=self.dtype, shape=[self.block_q, self.head_size])
        ldq = self.load_global(
            gq,
            offsets=[bs, q_offset, head, 0],
            shape=[self.block_q, self.head_size],
            dims=[1, 3],
        )
        self.store_shared(sq, ldq)
        self.sync()
        rq = self.load_shared(sq)  # [block_q, head_size]
        self.sync()
        self.free_shared(sq)

        # accumulators
        o = self.register_tensor(dtype=f32, shape=[self.block_q, self.head_size], init=0.0)
        m = self.register_tensor(dtype=f32, shape=[self.block_q, 1], init=-1e6)  # rowmax(score)
        l = self.register_tensor(
            dtype=f32, shape=[self.block_q, 1], init=0.0
        )  # rowsum(exp(score - m))

        sk = self.shared_tensor(dtype=self.dtype, shape=[self.block_kv, self.head_size])
        sv = self.shared_tensor(dtype=self.dtype, shape=[self.block_kv, self.head_size])

        self.copy_async(gk, sk, offsets=[bs, 0, head // self.group_heads, 0], dims=[1, 3])
        self.copy_async_commit_group()

        kv_offset_inner_end = (q_offset + 1) // self.block_kv * self.block_kv
        for kv_offset in range(0, kv_offset_inner_end, self.block_kv):
            self.attention_iteration(
                bs,
                kv_offset,
                q_offset,
                head,
                gk,
                gv,
                rq,
                sk,
                sv,
                o,
                m,
                l,
                check_bounds=False,
            )

        kv_offset_end = q_offset + self.block_q
        for kv_offset in range(kv_offset_inner_end, kv_offset_end, self.block_kv):
            self.attention_iteration(
                bs,
                kv_offset,
                q_offset,
                head,
                gk,
                gv,
                rq,
                sk,
                sv,
                o,
                m,
                l,
                check_bounds=True,
            )

        self.copy_async_wait_group(0)
        self.sync()
        self.free_shared(sk)
        self.free_shared(sv)

        o = o / l
        o_f16 = self.cast(o, dtype=self.dtype)  # [block_q, head_size]
        so = self.shared_tensor(dtype=self.dtype, shape=[self.block_q, self.head_size])
        self.store_shared(so, o_f16)
        self.sync()
        self.store_global(
            go,
            self.load_shared(so),
            offsets=[bs, q_offset, head, 0],
            dims=[1, 3],
        )
        self.free_shared(so)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    """
    Flash attention function for variable length sequences.

    Parameters
    ----------
    q: torch.Tensor
        The query tensor of shape (bs, seqlen, num_heads, head_size).

    k: torch.Tensor
        The key tensor of shape (bs, seqlen, num_heads_kv, head_size).

    v: torch.Tensor
        The value tensor of shape (bs, seqlen, num_heads_kv, head_size).

    Returns
    -------
    o: torch.Tensor
        The output tensor of shape (bs, seqlen, num_heads, head_size).
    """
    out = torch.empty_like(q)
    FlashAttention(
        dtype=tilus.float16,
        num_heads=q.size(2),
        num_heads_kv=k.size(2),
        head_size=q.size(3),
    )(q.size(0), q.size(1), q, k, v, out)
    return out


def flash_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    bs, seqlen, num_heads, head_size = q.size()
    _, _, num_heads_kv, _ = k.size()
    assert q.size(0) == k.size(0) == v.size(0), "Batch size must match for q, k, and v."
    assert q.size(1) == k.size(1) == v.size(1), "Sequence length must match for q, k, and v."
    assert q.size(3) == k.size(3) == v.size(3), "Head size must match for q, k, and v."
    assert k.size(2) == v.size(2), "Number of heads in k and v must match."
    assert num_heads % num_heads_kv == 0, "Number of heads must be divisible by number of kv heads."

    k = torch.repeat_interleave(k, num_heads // num_heads_kv, dim=2)
    v = torch.repeat_interleave(v, num_heads // num_heads_kv, dim=2)

    q = torch.transpose(q, 1, 2).reshape(bs * num_heads, seqlen, head_size)
    k = torch.transpose(k, 1, 2).reshape(bs * num_heads, seqlen, head_size)
    v = torch.transpose(v, 1, 2).reshape(bs * num_heads, seqlen, head_size)

    score = torch.bmm(q, k.mT) / np.sqrt(head_size)  # [bs * num_heads, seqlen, seqlen]
    causal_mask = torch.tril(torch.ones(seqlen, seqlen, dtype=torch.bool), diagonal=0).to(q.device)
    causal_mask = causal_mask.unsqueeze(0)  # [1, seqlen, seqlen]
    causal_mask = causal_mask.expand(
        bs * num_heads, seqlen, seqlen
    ).contiguous()  # [bs * num_heads, seqlen, seqlen]
    score = score.masked_fill(causal_mask == 0, float("-inf"))

    o = torch.bmm(
        torch.softmax(score.float(), dim=-1).to(q.dtype), v
    )  # [bs * num_heads, seqlen, head_size]
    o = o.reshape(bs, num_heads, seqlen, head_size).transpose(1, 2).contiguous()
    return o


def has_flash_attn():
    return importlib.util.find_spec("flash_attn") is not None


def flash_attention_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    from flash_attn import flash_attn_func

    return flash_attn_func(q, k, v, causal=True)


def demo_flash_attention():
    for bs, seqlen, num_heads, head_size, num_heads_kv in [
        # [1, 8, 1, 64, 1],
        [1, 4096, 32, 128, 8]
    ]:
        q = torch.rand(bs, seqlen, num_heads, head_size, dtype=torch.float16).cuda()
        k = torch.rand(bs, seqlen, num_heads_kv, head_size, dtype=torch.float16).cuda()
        v = torch.rand(bs, seqlen, num_heads_kv, head_size, dtype=torch.float16).cuda()
        flash_attention(q, k, v)
        torch.cuda.synchronize()


def main(bench=True):
    headers = [
        "batch_size",
        "seqlen",
        "num_heads",
        "head_size",
        "num_heads_kv",
        "name",
        "latency (ms)",
        "gflops",
    ]
    data = []
    for batch_size, seqlen, num_heads, head_size, num_heads_kv in [
        # llama 3.1 8B
        [1, 1024, 32, 128, 8],
        [1, 2048, 32, 128, 8],
        [1, 4096, 32, 128, 8],
        [1, 8192, 32, 128, 8],
        # llama 3.1 70B
        [1, 1024, 64, 128, 8],
        [1, 2048, 64, 128, 8],
        [1, 4096, 64, 128, 8],
        [1, 8192, 64, 128, 8],
    ]:
        q = torch.rand(batch_size, seqlen, num_heads, head_size, dtype=torch.float16).cuda()
        k = torch.rand(batch_size, seqlen, num_heads_kv, head_size, dtype=torch.float16).cuda()
        v = torch.rand(batch_size, seqlen, num_heads_kv, head_size, dtype=torch.float16).cuda()
        # q = torch.ones(batch_size, seqlen, num_heads, head_size, dtype=torch.float16).cuda()
        # k = torch.ones(batch_size, seqlen, num_heads_kv, head_size, dtype=torch.float16).cuda()
        # v = torch.ones(batch_size, seqlen, num_heads_kv, head_size, dtype=torch.float16).cuda()
        # for i in range(seqlen):
        #     for j in range(head_size):
        #         v[0, i, 0, j] = i
        for name, runner in [
            ("flash-attn", flash_attention_flash_attn),
            ("tilus", flash_attention),
        ]:
            if name == "flash-attn" and not has_flash_attn():
                print("flash-attn is not available, skipping...")
                continue
            print(
                f"Running {name} with batch_size={batch_size}, seqlen={seqlen}, num_heads={num_heads}, head_size={head_size}, num_heads_kv={num_heads_kv}"
            )
            actual = runner(q, k, v)
            try:
                expected = flash_attention_reference(q, k, v)
                torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
            except torch.OutOfMemoryError:
                pass
            # print(expected)
            # print(actual)
            # print(actual - expected)
            # print the top 10 differences and their indices (4 dimensions)
            # diff = actual - expected
            # top_diff_indices = torch.topk(diff.abs().flatten(), 10).indices
            # top_diff_values = diff.flatten()[top_diff_indices]
            # top_diff_indices = np.unravel_index(top_diff_indices.cpu().numpy(), diff.shape)
            # for i in range(len(top_diff_indices[0])):
            #     print(
            #         f"Top diff {i}: index={top_diff_indices[0][i], top_diff_indices[1][i], top_diff_indices[2][i], top_diff_indices[3][i]}, value={top_diff_values[i]}"
            #     )

            latency = (
                benchmark_func(
                    lambda: runner(q, k, v),
                    warmup=5,
                    repeat=20,
                )
                if bench
                else float("nan")
            )
            gflops = 2 * batch_size * num_heads * seqlen * head_size * seqlen / latency * 1e-9
            data.append(
                [
                    batch_size,
                    seqlen,
                    num_heads,
                    head_size,
                    num_heads_kv,
                    name,
                    latency,
                    gflops,
                ]
            )
    df = pd.DataFrame(data, columns=headers)
    df_pivot = df.pivot(
        index=[
            "batch_size",
            "seqlen",
            "num_heads",
            "head_size",
            "num_heads_kv",
        ],
        columns="name",
        values=["latency (ms)", "gflops"],
    ).reset_index()
    print(df_pivot)


if __name__ == "__main__":
    main()
    # ncu_run(main, bench=False)
    # ncu_run(main, bench=False, kernel_regex="flash_fwd|flash_attention")
