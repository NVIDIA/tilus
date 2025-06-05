import numpy as np
import pandas as pd
import tilus
import torch
from hidet.ir import DataType
from tilus import boolean, f32, int32, void_p
from tilus.utils import benchmark_func, cdiv

# tilus.logging.set_logging_level("DEBUG")
tilus.option.cache_dir("./cache")
tilus.option.debug.dump_ir()
tilus.utils.clear_cache()
pd.options.display.max_columns = None
pd.options.display.width = 1000


@tilus.autotune("num_warps", [2])
@tilus.autotune("block_q", [16])
@tilus.autotune("block_kv", [16])
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
        self.head_group_size = num_heads // num_heads_kv

    def __call__(self, batch_size: int, seqlen: int32, q_ptr: void_p, k_ptr: void_p, v_ptr: void_p, o_ptr: void_p):
        self.attrs.warps = self.num_warps
        self.attrs.blocks = cdiv(seqlen, self.block_q), self.num_heads, batch_size

        q_offset = self.blockIdx.x * self.block_q
        head = self.blockIdx.y
        bs = self.blockIdx.z

        gq = self.global_view(ptr=q_ptr, dtype=self.dtype, shape=[batch_size, seqlen, self.num_heads, self.head_size])
        gk = self.global_view(
            ptr=k_ptr, dtype=self.dtype, shape=[batch_size, seqlen, self.num_heads_kv, self.head_size]
        )
        gv = self.global_view(
            ptr=v_ptr, dtype=self.dtype, shape=[batch_size, seqlen, self.num_heads_kv, self.head_size]
        )
        go = self.global_view(ptr=o_ptr, dtype=self.dtype, shape=[batch_size, seqlen, self.num_heads, self.head_size])

        sk = self.shared_tensor(dtype=self.dtype, shape=[self.block_kv, self.head_size])
        sv = self.shared_tensor(dtype=self.dtype, shape=[self.block_kv, self.head_size])

        # load query
        sq = self.shared_tensor(dtype=self.dtype, shape=[self.block_q, self.head_size])
        self.store_shared(
            sq,
            self.load_global(
                gq, offsets=[bs, q_offset, head, 0], shape=[self.block_q, self.head_size], slice_dims=[1, 3]
            ),
        )
        self.sync()
        rq = self.load_shared(sq)
        self.sync()
        self.free_shared(sq)

        # accumulators
        o = self.register_tensor(dtype=f32, shape=[self.block_q, self.head_size], init=0.0)
        m = self.register_tensor(dtype=f32, shape=[self.block_q, 1], init=-1e6)  # rowmax(score)
        l = self.register_tensor(dtype=f32, shape=[self.block_q, 1], init=0.0)  # rowsum(exp(score - m))

        kv_offset_end = q_offset + self.block_q
        for kv_offset in range(0, kv_offset_end, self.block_kv):
            self.store_shared(
                sk,
                self.load_global(
                    gk,
                    offsets=[bs, kv_offset, head // self.head_group_size, 0],
                    shape=[self.block_kv, self.head_size],
                    slice_dims=[1, 3],
                ),
            )
            self.store_shared(
                sv,
                self.load_global(
                    gv,
                    offsets=[bs, kv_offset, head // self.head_group_size, 0],
                    shape=[self.block_kv, self.head_size],
                    slice_dims=[1, 3],
                ),
            )
            self.sync()

            rk = self.load_shared(sk)  # [block_kv, head_size]
            rv = self.load_shared(sv)  # [block_kv, head_size]
            score = self.mma_dot(rq, rk.transpose(), acc_dtype=f32) * self.score_scale  # [block_q, block_kv]
            mask = self.register_tensor(
                dtype=boolean,
                shape=[self.block_q, self.block_kv],
                f_init=lambda ij: ij[0] + q_offset >= ij[1] + kv_offset,
            )
            score = score + self.where(mask, x=0.0, y=-1e6)

            # online softmax
            cur_m = self.max(score, dim=1, keepdim=True)  # [block_q, 1]
            new_m = self.maximum(m, cur_m)  # [block_q, 1]
            p = self.exp(score - new_m)  # [block_q, block_kv]
            p_f16 = self.cast(p, dtype=self.dtype)  # [block_q, block_kv]
            cur_o = self.mma_dot(p_f16, rv, acc_dtype=f32)  # [block_q, head_size]
            o = o * self.exp(m - new_m) + cur_o  # [block_q, head_size]
            l = l * self.exp(m - new_m) + self.sum(p, dim=1, keepdim=True)  # [block_q, 1]
            m = new_m  # [block_q, 1]
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


def flash_attention_baseline(
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


def main():
    headers = ["bs", "seqlen", "num_heads", "head_size", "num_heads_kv", "name", "latency (ms)", "gflops"]
    data = []
    for bs, seqlen, num_heads, head_size, num_heads_kv in [
        [1, 4096, 32, 128, 8],
        # [1, 16, 1, 16, 1]
    ]:
        q = torch.rand(bs, seqlen, num_heads, head_size, dtype=torch.float16).cuda()
        k = torch.rand(bs, seqlen, num_heads_kv, head_size, dtype=torch.float16).cuda()
        v = torch.rand(bs, seqlen, num_heads_kv, head_size, dtype=torch.float16).cuda()
        # q = torch.ones(bs, seqlen, num_heads, head_size, dtype=torch.float16).cuda()
        # k = torch.ones(bs, seqlen, num_heads_kv, head_size, dtype=torch.float16).cuda()
        # v = torch.ones(bs, seqlen, num_heads_kv, head_size, dtype=torch.float16).cuda()

        for name, runner in [
            ("flash-attn", flash_attention_baseline),
            ("tilus", flash_attention),
        ]:
            print(
                f"Running {name} with bs={bs}, seqlen={seqlen}, num_heads={num_heads}, head_size={head_size}, num_heads_kv={num_heads_kv}"
            )
            actual = runner(q, k, v)
            expected = flash_attention_reference(q, k, v)

            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
            latency = benchmark_func(
                lambda: runner(q, k, v),
                warmup=5,
                repeat=20,
            )
            gflops = 2 * bs * num_heads * seqlen * head_size * seqlen / latency * 1e-9
            data.append([bs, seqlen, num_heads, head_size, num_heads_kv, name, latency, gflops])
    df = pd.DataFrame(data, columns=headers)
    df_pivot = df.pivot(
        index=["bs", "seqlen", "num_heads", "head_size", "num_heads_kv"],
        columns="name",
        values=["latency (ms)", "gflops"],
    ).reset_index()
    print(df_pivot)


if __name__ == "__main__":
    # flash_attention(0, 0, 0)
    main()
    # demo_flash_attention()
    # sanitizer_run(demo_flash_attention)
