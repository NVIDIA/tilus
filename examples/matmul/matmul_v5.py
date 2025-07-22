# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Split-K
=======

This example demonstrates how to implement a matrix multiplication kernel using split-K optimization in tilus.

In previous examples, we use a single thread block to compute a tile of the output matrix C. This approach works well
for workloads with large m and n dimensions since there are enough C tiles to saturate the GPU. However, for workloads
with small m and n dimensions and large k dimension, it's more efficient to split the k dimension into multiple segments
and assign each segment to a separate thread block. After that, we can aggregate the results from these thread blocks
that compute the same C tile to get the final result.

There are mainly two ways to implement the split-K optimization: 1) using a separate kernel to perform the aggregation,
or 2) implementing the aggregation logic in the same kernel that computes the C tile with semaphores. In this example,
we will implement the second approach.

We will use several new tilus instructions:

.. currentmodule:: tilus.Script

.. autosummary::

    global_tensor
    lock_semaphore
    release_semaphore

We use :meth:`global_tensor` to create a global tensor that will be used to store the semaphores for each C tile.
Its shape is ``[cdiv(m_size, block_m), cdiv(n_size, block_n)]``, where ``block_m`` and ``block_n`` are the dimensions
of the C tile. All thread blocks that compute the same C tile will use the same semaphore to synchronize the aggregation
of the results.

After we launched the kernel, all thread blocks will compute their accumulated result for the C tile. If ``k_size``
equals to 1024 and we split it into 4 segments (i.e., ``split_k_factor=4``), then each thread block will compute over
a k segment of size 256. We name the 4 blocks as ``0, 1, 2, 3``. The first thread block directly stores its result
to the C matrix, while the other thread blocks will wait until the semaphore becomes to their block index. After
the first thread block stores its result, it releases the semaphore with the value of 1, allowing the second thread
block to aggregate its result with the first one. The second thread block will then store the aggregated result
to the C matrix and release the semaphore with the value of 2, allowing the third thread block to aggregate its result
with the first two. This process continues until all thread blocks have aggregated their results and stored the final
result to the C matrix. The last thread block will release the semaphore with the value of 0, to satisfy the requirement
of :meth:`global_tensor` with ``requires_clean=True``.

The following example implements above logic in the matrix multiplication kernel:
"""

import tilus
from tilus import float16, float32, int32
from tilus.utils import cdiv


@tilus.autotune("num_warps", [4, 8])
@tilus.autotune("block_m, block_n", [(128, 128), (128, 64), (64, 128), (32, 256)])
@tilus.autotune("block_k", [16, 32])
@tilus.autotune("num_stages", [3, 4, 5])
@tilus.autotune("split_k_factor", [1, 4, 12, 16])
class MatmulV5(tilus.Script):
    def __init__(
        self,
        num_warps,
        block_m,
        block_n,
        block_k,
        num_stages,
        split_k_factor,
    ):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.split_k_factor = split_k_factor

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
            self.split_k_factor,
        ]
        self.attrs.warps = self.num_warps

        # the k_size for each thread block
        block_k_size = (
            cdiv(cdiv(k_size, self.split_k_factor), self.block_k) * self.block_k
        )
        start_offset_k = self.blockIdx.z * block_k_size
        end_offset_k = min(start_offset_k + block_k_size, k_size)

        block_m, block_n, block_k = self.block_m, self.block_n, self.block_k
        offset_m: int32 = block_m * self.blockIdx.x
        offset_n: int32 = block_n * self.blockIdx.y

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[k_size, n_size])
        sa = self.shared_tensor(dtype=float16, shape=[self.num_stages, block_m, block_k])
        sb = self.shared_tensor(dtype=float16, shape=[self.num_stages, block_k, block_n])
        acc = self.register_tensor(dtype=float32, shape=[block_m, block_n], init=0.0)

        for stage in range(self.num_stages - 1):
            offset_k = start_offset_k + stage * self.block_k
            self.copy_async(src=ga, dst=sa[stage], offsets=[offset_m, offset_k])
            self.copy_async(src=gb, dst=sb[stage], offsets=[offset_k, offset_n])
            self.copy_async_commit_group()

        self.copy_async_wait_group(n=self.num_stages - 2)
        self.sync()

        current_stage: int32 = 0
        preload_stage: int32 = self.num_stages - 1
        for offset_k in self.range(
            start_offset_k, end_offset_k, block_k, unroll=self.num_stages
        ):
            # computation for current tile
            a = self.load_shared(sa[current_stage])
            b = self.load_shared(sb[current_stage])
            self.dot(a, b, acc, out=acc)

            # preload the next tile of A and B into shared memory
            preload_offset_k = offset_k + (self.num_stages - 1) * block_k
            if preload_offset_k < end_offset_k:
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

        # free the shared memory tensors for A and B
        self.free_shared(sa)
        self.free_shared(sb)

        # cast the accumulator to float16 and change the register tensor's layout
        sc = self.shared_tensor(dtype=float16, shape=[block_m, block_n])
        casted_acc = self.cast(acc, dtype=float16)
        self.store_shared(sc, casted_acc)
        self.sync()
        rc = self.load_shared(sc)
        self.free_shared(sc)

        m_blocks, n_blocks = cdiv(m_size, block_m), cdiv(n_size, block_n)
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        if self.split_k_factor == 0:
            self.store_global(gc, rc, offsets=[offset_m, offset_n])
        else:
            semaphores = self.global_tensor(
                dtype=int32, shape=[m_blocks, n_blocks], requires_clean=True
            )
            semaphore: ~int32 = ~semaphores[self.blockIdx.x, self.blockIdx.y]

            # load and accumulate the partial result in global memory
            if self.blockIdx.z > 0:
                self.lock_semaphore(semaphore, value=self.blockIdx.z)
                partial_rc = self.load_global(
                    gc, offsets=[offset_m, offset_n], shape=[block_m, block_n]
                )
                self.add(rc, partial_rc, out=rc)

            # store the result to global memory and release the semaphore
            self.store_global(gc, rc, offsets=[offset_m, offset_n])

            # release the semaphore
            self.sync()  # we need to make sure the previous store_global is finished
            self.release_semaphore(
                semaphore, value=(self.blockIdx.z + 1) % self.split_k_factor
            )


# %%

import math

import pandas
import torch
from tilus.utils import benchmark_func


def main():
    headers = ["m", "n", "k", "name", "latency (ms)", "gflops"]
    workloads = [
        [4096, 4096, 4096],
        [4096, 4096, 14336],
    ]

    rows = []
    for m, n, k in workloads:
        matmul = MatmulV5()

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


# %%

if __name__ == "__main__":
    main()
