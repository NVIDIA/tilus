"""
Matmul with Software Pipelining
===============================
"""

import math

import pandas
import tilus
import torch
from tilus import float16, float32, int32
from tilus.utils import benchmark_func


@tilus.autotune("num_warps", [4, 8])
@tilus.autotune("block_m, block_n", [(128, 128), (128, 64), (64, 128)])
@tilus.autotune("block_k", [16, 32])
@tilus.autotune("num_stages", [3, 4, 5])
class MatmulV4(tilus.Script):
    def __init__(self, num_warps, block_m, block_n, block_k, num_stages):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps
        self.num_stages = num_stages

    def __call__(
        self,
        m_size: int32,
        n_size: int,
        k_size: int,
        a_ptr: ~float16,
        b_ptr: ~float16,
        c_ptr: ~float16,
    ):
        self.attrs.blocks = [
            self.utils.ceil_div(m_size, self.block_m),
            self.utils.ceil_div(n_size, self.block_n),
        ]
        self.attrs.warps = self.num_warps

        block_m, block_n, block_k = self.block_m, self.block_n, self.block_k
        offset_m: int32 = block_m * self.blockIdx.x
        offset_n: int32 = block_n * self.blockIdx.y

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[k_size, n_size])
        sa = self.shared_tensor(dtype=float16, shape=[self.num_stages, block_m, block_k])
        sb = self.shared_tensor(dtype=float16, shape=[self.num_stages, block_k, block_n])
        acc = self.register_tensor(dtype=float32, shape=[block_m, block_n], init=0.0)

        for stage in range(self.num_stages - 1):
            offset_k = stage * self.block_k
            self.copy_async(src=ga, dst=sa[stage], offsets=[offset_m, offset_k])
            self.copy_async(src=gb, dst=sb[stage], offsets=[offset_k, offset_n])
            self.copy_async_commit_group()

        self.copy_async_wait_group(n=self.num_stages - 2)
        self.sync()

        current_stage: int32 = 0
        preload_stage: int32 = self.num_stages - 1
        for offset_k in self.range(0, k_size, block_k, unroll=self.num_stages):
            # computation for current tile
            a = self.load_shared(sa[current_stage])
            b = self.load_shared(sb[current_stage])
            self.dot(a, b, acc, out=acc)

            # preload the next tile of A and B into shared memory
            preload_offset_k = offset_k + (self.num_stages - 1) * block_k
            self.copy_async(
                src=ga,
                dst=sa[preload_stage],
                offsets=[offset_m, preload_offset_k],
            )
            self.copy_async(
                src=gb,
                dst=sb[preload_stage],
                offsets=[preload_offset_k, offset_n],
            )
            self.copy_async_commit_group()

            # update the stage
            current_stage = (current_stage + 1) % self.num_stages
            preload_stage = (preload_stage + 1) % self.num_stages
            self.copy_async_wait_group(n=self.num_stages - 2)
            self.sync()

        self.free_shared(sa)
        self.free_shared(sb)

        casted_acc = self.cast(acc, dtype=float16)
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        self.store_global(gc, casted_acc, offsets=[offset_m, offset_n])


def main():
    headers = ["m", "n", "k", "name", "latency (ms)", "gflops"]
    workloads = [
        [4096, 4096, 4096],
    ]

    rows = []
    for m, n, k in workloads:
        matmul = MatmulV4()

        a = (torch.rand(m, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
        b = (torch.rand(k, n, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
        c_actual = torch.empty(m, n, dtype=torch.float16).cuda()
        c_expect = a @ b
        matmul(m, n, k, a, b, c_actual)

        # check correctness
        torch.testing.assert_close(c_expect, c_actual)

        # benchmark
        for name, func in [
            ("torch", lambda: torch.matmul(a, b, out=c_expect)),
            ("tilus", lambda: matmul(m, n, k, a, b, c_actual)),
        ]:
            latency = benchmark_func(func, warmup=5, repeat=20)
            flops = 2 * m * n * k / latency * 1e-9
            rows.append([m, n, k, name, latency, flops])

    df = pandas.DataFrame(rows, columns=headers)
    print(df)


if __name__ == "__main__":
    main()
