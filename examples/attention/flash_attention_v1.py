import numpy as np
import pandas as pd
import tilus
import torch
from hidet.ir import DataType
from tilus import boolean, f32, int32, void_p
from tilus.utils import benchmark_func, cdiv

tilus.option.cache_dir("./cache")
tilus.option.debug.dump_ir()
pd.options.display.max_columns = None
pd.options.display.width = 1000


@tilus.autotune("num_warps", [4])
@tilus.autotune("block_q", [64])
@tilus.autotune("block_kv", [64])
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
            dtype, f32, m=block_q, n=block_kv, k=head_size, warp_m=num_warps, warp_n=1
        )
        self.sv_config = self.cuda.resolve_dot_config(
            dtype, f32, m=block_q, n=head_size, k=block_kv, warp_m=num_warps, warp_n=1
        )
        assert self.qk_config.lc == self.sv_config.la

    def __call__(self, batch_size: int, seqlen: int32, q_ptr: void_p, k_ptr: void_p, v_ptr: void_p, o_ptr: void_p):
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
        self.attrs.blocks = cdiv(seqlen, self.block_q), self.num_heads, batch_size

        q_offset = self.blockIdx.x * self.block_q
        head = self.blockIdx.y
        bs = self.blockIdx.z

        gq = self.global_view(q_ptr, dtype=self.dtype, shape=[batch_size, seqlen, self.num_heads, self.head_size])
        gk = self.global_view(k_ptr, dtype=self.dtype, shape=[batch_size, seqlen, self.num_heads_kv, self.head_size])
        gv = self.global_view(v_ptr, dtype=self.dtype, shape=[batch_size, seqlen, self.num_heads_kv, self.head_size])
        go = self.global_view(o_ptr, dtype=self.dtype, shape=[batch_size, seqlen, self.num_heads, self.head_size])

        # load query to register
        sq = self.shared_tensor(dtype=self.dtype, shape=[self.block_q, self.head_size])
        ldq = self.load_global(gq, offsets=[bs, q_offset, head, 0], shape=[self.block_q, self.head_size], dims=[1, 3])
        self.store_shared(sq, ldq)
        self.sync()
        rq = self.load_shared(sq)  # [block_q, head_size]
        self.sync()
        self.free_shared(sq)

        # accumulators
        o = self.register_tensor(dtype=f32, shape=[self.block_q, self.head_size], init=0.0)
        m = self.register_tensor(dtype=f32, shape=[self.block_q, 1], init=-1e6)  # rowmax(score)
        l = self.register_tensor(dtype=f32, shape=[self.block_q, 1], init=0.0)  # rowsum(exp(score - m))

        sk = self.shared_tensor(dtype=self.dtype, shape=[self.block_kv, self.head_size])
        sv = self.shared_tensor(dtype=self.dtype, shape=[self.block_kv, self.head_size])

        self.copy_async(gk, sk, offsets=[bs, 0, head // self.group_heads, 0], dims=[1, 3])
        self.copy_async_commit_group()

        kv_offset_end = q_offset + self.block_q
        for kv_offset in range(0, kv_offset_end, self.block_kv):
            # wait for the async copy of k to finish
            self.copy_async_wait_group(0)
            self.sync()
            self.copy_async(gv, sv, offsets=[bs, kv_offset, head // self.group_heads, 0], dims=[1, 3])
            self.copy_async_commit_group()

            # issue the async copy for v and perform dot(q, k)
            rk = self.load_shared(sk)  # [block_kv, head_size]
            score = self.mma_dot(rq, rk.transpose(), acc_dtype=f32) * self.score_scale  # [block_q, block_kv]
            self.annotate_layout(score, self.qk_config.lc)
            mask = self.register_tensor(
                dtype=boolean,
                shape=[self.block_q, self.block_kv],
                f_init=lambda ij: ij[0] + q_offset >= ij[1] + kv_offset,
            )
            score = score + self.where(mask, x=0.0, y=-1e6)

            # wait for the async copy of v to finish
            self.copy_async_wait_group(0)
            self.sync()
            self.copy_async(gk, sk, offsets=[bs, kv_offset + self.block_kv, head // self.group_heads, 0], dims=[1, 3])
            self.copy_async_commit_group()

            # load v to register
            rv = self.load_shared(sv)  # [block_kv, head_size]

            # online softmax
            cur_m = self.max(score, dim=1, keepdim=True)  # [block_q, 1]
            new_m = self.maximum(m, cur_m)  # [block_q, 1]
            p = self.exp(score - new_m)  # [block_q, block_kv]
            cur_o = self.mma_dot(p.to(self.dtype), rv, acc_dtype=f32)  # [block_q, head_size]
            self.annotate_layout(cur_o, self.sv_config.lc)
            o = o * self.exp(m - new_m) + cur_o  # [block_q, head_size]
            l = l * self.exp(m - new_m) + self.sum(p, dim=1, keepdim=True)  # [block_q, 1]
            m = new_m  # [block_q, 1]

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
            slice_dims=[1, 3],
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
    FlashAttention(dtype=tilus.float16, num_heads=q.size(2), num_heads_kv=k.size(2), head_size=q.size(3))(
        q.size(0), q.size(1), q, k, v, out
    )
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
    causal_mask = causal_mask.expand(bs * num_heads, seqlen, seqlen).contiguous()  # [bs * num_heads, seqlen, seqlen]
    score = score.masked_fill(causal_mask == 0, float("-inf"))

    o = torch.bmm(torch.softmax(score.float(), dim=-1).to(q.dtype), v)  # [bs * num_heads, seqlen, head_size]
    o = o.reshape(bs, num_heads, seqlen, head_size).transpose(1, 2).contiguous()
    return o


def flash_attention_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    try:
        from flash_attn import flash_attn_func

        return flash_attn_func(q, k, v, causal=True)
    except ImportError:
        return flash_attention_reference(q, k, v)


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
    headers = ["batch_size", "seqlen", "num_heads", "head_size", "num_heads_kv", "name", "latency (ms)", "gflops"]
    data = []
    for batch_size, seqlen, num_heads, head_size, num_heads_kv in [
        [1, 4096, 32, 128, 8],
    ]:
        q = torch.rand(batch_size, seqlen, num_heads, head_size, dtype=torch.float16).cuda()
        k = torch.rand(batch_size, seqlen, num_heads_kv, head_size, dtype=torch.float16).cuda()
        v = torch.rand(batch_size, seqlen, num_heads_kv, head_size, dtype=torch.float16).cuda()
        for name, runner in [
            ("flash-attn", flash_attention_flash_attn),
            ("tilus", flash_attention),
        ]:
            print(
                f"Running {name} with batch_size={batch_size}, seqlen={seqlen}, num_heads={num_heads}, head_size={head_size}, num_heads_kv={num_heads_kv}"
            )
            actual = runner(q, k, v)
            expected = flash_attention_reference(q, k, v)

            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
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
            data.append([batch_size, seqlen, num_heads, head_size, num_heads_kv, name, latency, gflops])
    df = pd.DataFrame(data, columns=headers)
    df_pivot = df.pivot(
        index=["batch_size", "seqlen", "num_heads", "head_size", "num_heads_kv"],
        columns="name",
        values=["latency (ms)", "gflops"],
    ).reset_index()
    print(df_pivot)


if __name__ == "__main__":
    main()
    # ncu_run(main, bench=False)
    # ncu_run(main, bench=False, kernel_regex='flash_fwd|flash_attention')
