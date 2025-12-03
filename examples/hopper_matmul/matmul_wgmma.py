# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pandas
import tilus
import torch
from tilus import float16, float32, int32, uint32
from tilus.utils import benchmark_func, cdiv


tilus.option.cache_dir("./cache")
tilus.option.debug.dump_ir(True)

@tilus.autotune("block_m, block_n", [(128, 128), (128, 256), (128, 64)])
@tilus.autotune("block_k", [16, 32, 16])
class MatmulTMA(tilus.Script):
    def __init__(
        self,
        block_m,
        block_n,
        block_k,
    ):
        super().__init__()
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
        sa = self.shared_tensor(dtype=float16, shape=[block_m, block_k])
        sb = self.shared_tensor(dtype=float16, shape=[block_n, block_k])
        acc = self.register_tensor(dtype=float32, shape=[block_m, block_n], init=0.0)

        tma_barrier = self.mbarrier.alloc(count=[2])
        phase: uint32 = 0

        for offset_k in range(0, k_size, block_k):
            # issue asynchronous copy instructions to load tiles of A and B
            with self.single_thread():
                self.tma.global_to_shared(src=ga, dst=sa, offsets=[offset_m, offset_k], mbarrier=tma_barrier)
                self.tma.global_to_shared(src=gb, dst=sb, offsets=[offset_n, offset_k], mbarrier=tma_barrier)
                self.mbarrier.wait(tma_barrier, phase=phase)

            # synchronize threads in the block to ensure data is available in shared memory
            self.sync()

            # a = self.load_shared(sa)
            # b = self.load_shared(sb)
            self.wgmma.fence()
            self.wgmma.mma(sa, sb, acc)
            self.wgmma.commit_group()
            self.wgmma.wait_group(1)
            self.sync()
            phase ^= 1

        self.free_shared(sa)
        self.free_shared(sb)

        casted_acc = self.cast(acc, dtype=float16)
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        self.store_global(gc, casted_acc, offsets=[offset_m, offset_n])


def main():
    headers = ["m", "n", "k", "name", "latency (ms)", "tflops"]
    workloads = [
        [4096, 4096, 4096],
    ]

    rows = []
    for m, n, k in workloads:
        matmul = MatmulTMA()

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
            ("torch", lambda: torch.matmul(a, b, out=c_expect)),
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
