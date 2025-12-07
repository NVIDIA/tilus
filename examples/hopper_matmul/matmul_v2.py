# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pandas
import tilus
import torch
from tilus import float16, float32, int32, uint32
from tilus.utils import benchmark_func, cdiv


@tilus.autotune("num_stages", [2, 3, 4])
@tilus.autotune(
    "block_m, block_n", [(64, 128), (128, 128), (128, 256), (256, 128), (256, 256)]
)
@tilus.autotune("block_k", [16, 32, 64])
class MatmulWGMMAV2(tilus.Script):
    def __init__(
        self,
        num_stages,
        block_m,
        block_n,
        block_k,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k

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
            cdiv(m_size, self.block_m),
            cdiv(n_size, self.block_n),
        ]
        self.attrs.warps = 4

        block_m, block_n, block_k = self.block_m, self.block_n, self.block_k
        offset_m: int32 = block_m * self.blockIdx.x
        offset_n: int32 = block_n * self.blockIdx.y

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[n_size, k_size])
        sa = self.shared_tensor(dtype=float16, shape=[self.num_stages, block_m, block_k])
        sb = self.shared_tensor(dtype=float16, shape=[self.num_stages, block_n, block_k])
        acc = self.register_tensor(dtype=float32, shape=[block_m, block_n], init=0.0)

        tma_barriers = self.mbarrier.alloc(count=[2 for _ in range(self.num_stages)])
        phase = self.register_tensor(dtype=uint32, shape=[self.num_stages], init=0)
        
        num_iters: int32 = cdiv(k_size, block_k)
        max_num_stages: int32 = min(num_iters, self.num_stages)

        for stage in range(max_num_stages):
            offset_k = stage * self.block_k
            with self.single_thread():
                self.tma.global_to_shared(
                    src=ga, dst=sa[stage], offsets=[offset_m, offset_k], mbarrier=tma_barriers[stage]
                )
                self.tma.global_to_shared(
                    src=gb, dst=sb[stage], offsets=[offset_n, offset_k], mbarrier=tma_barriers[stage]
                )

        for iter in range(num_iters):
            stage = iter % self.num_stages
            self.mbarrier.wait(tma_barriers[stage], phase=phase[stage])
            # self.sync()

            self.wgmma.fence()
            self.wgmma.mma(sa[stage], sb[stage].transpose(), acc)
            self.wgmma.commit_group()
            self.wgmma.wait_group(0)
            phase[stage] ^= 1

            preload_iter = iter + self.num_stages
            if preload_iter < num_iters:
                preload_stage = preload_iter % self.num_stages
                offset_k = preload_iter * self.block_k
                with self.single_thread():
                    self.tma.global_to_shared(
                        src=ga, dst=sa[preload_stage], offsets=[offset_m, offset_k], mbarrier=tma_barriers[preload_stage]
                    )
                    self.tma.global_to_shared(
                        src=gb, dst=sb[preload_stage], offsets=[offset_n, offset_k], mbarrier=tma_barriers[preload_stage]
                    )
            self.sync()

        self.free_shared(sa)
        self.free_shared(sb)

        casted_acc = self.cast(acc, dtype=float16)
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        self.store_global(gc, casted_acc, offsets=[offset_m, offset_n])


def main():
    headers = ["m", "n", "k", "name", "latency (ms)", "tflops"]
    workloads = [
        [4096, 4096, 4096],
        # [128, 48, 16],
    ]

    rows = []
    for m, n, k in workloads:
        matmul = MatmulWGMMAV2()

        a = (torch.rand(m, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
        b = (torch.rand(n, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
        c_actual = torch.empty(m, n, dtype=torch.float16).cuda()
        c_expect = a @ b.T
        matmul(m, n, k, a, b, c_actual)
        torch.cuda.synchronize()

        # check correctness
        torch.testing.assert_close(c_expect, c_actual)

        # benchmark
        for name, func in [
            ("torch", lambda: torch.matmul(a, b.T, out=c_expect)),
            ("tilus", lambda: matmul(m, n, k, a, b, c_actual)),
        ]:
            latency = benchmark_func(func, warmup=5, repeat=20)
            tflops = 2 * m * n * k / latency * 1e-9
            rows.append([m, n, k, name, latency, tflops])

    df = pandas.DataFrame(rows, columns=headers)
    print(df)


# %%

if __name__ == "__main__":
    main()
