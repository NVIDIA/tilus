# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Optimizations from v4:
# - Use mbarrier for synchronization between producer and consumer WGs, instead of using shared memory as flags.
# - Use two consumer WGs to consume the produced tiles in parallel, and use WGMMA commit/wait group to synchronize between them, instead of using a single consumer WG to consume all tiles.

import math

import pandas
import tilus
import torch
from tilus import RegisterTensor, float16, float32, int32, uint32
from tilus.utils import benchmark_func, cdiv


class Pipeline(tilus.Class):
    def __init__(
        self,
        num_stages: int,
        producer_arrive_count: int = 1,
        consumer_arrive_count: int = 1,
    ):
        self.num_stages: int = num_stages
        self.empty_barriers = self.mbarrier.alloc(
            [consumer_arrive_count for _ in range(num_stages)]
        )
        self.full_barriers = self.mbarrier.alloc(
            [producer_arrive_count for _ in range(num_stages)]
        )
        self.producer_stage: int32 = 0
        self.consumer_stage: int32 = 0
        self.producer_phase: uint32 = self.mbarrier.producer_initial_phase
        self.consumer_phase: uint32 = self.mbarrier.consumer_initial_phase

    def producer_acquire(self):
        self.mbarrier.wait(
            barrier=self.empty_barriers[self.producer_stage],
            phase=self.producer_phase,
            sem="relaxed",
            scope="cta",
        )

    def producer_barrier(self) -> RegisterTensor:
        return self.full_barriers[self.producer_stage]

    def producer_advance(self):
        self.producer_stage = (self.producer_stage + 1) % self.num_stages
        self.producer_phase = self.producer_phase ^ (self.producer_stage == 0)

    def consumer_acquire(self):
        self.mbarrier.wait(
            barrier=self.full_barriers[self.consumer_stage],
            phase=self.consumer_phase,
            sem="relaxed",
            scope="cta",
        )

    def consumer_barrier(self) -> RegisterTensor:
        return self.empty_barriers[self.consumer_stage]

    def consumer_advance(self):
        self.consumer_stage = (self.consumer_stage + 1) % self.num_stages
        self.consumer_phase = self.consumer_phase ^ (self.consumer_stage == 0)

    def prev_consumer_barrier(self) -> RegisterTensor:
        prev_stage = (self.consumer_stage + (self.num_stages - 1)) % self.num_stages
        return self.empty_barriers[prev_stage]


# block_m must be >= 128 so each WG's WGMMA M = block_m/2 >= 64.
@tilus.autotune("num_stages", [3, 4, 5, 6])
@tilus.autotune("block_m, block_n", [[128, 128], [128, 256], [256, 128], [256, 256]])
@tilus.autotune("block_k", [16, 32, 64])
@tilus.autotune("swizzle_size", [4, 8])
class MatmulWGMMAV5(tilus.Script):
    def __init__(self, num_stages, block_m, block_n, block_k, swizzle_size):
        super().__init__()
        self.num_stages = num_stages
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.swizzle_size = swizzle_size

    def compute_block_coord(
        self, linear_idx: int32, num_m_blocks: int32, num_n_blocks: int
    ):
        swizzle_size = self.swizzle_size
        tiles_per_group = num_m_blocks * swizzle_size
        group_idx, in_group_idx = self.fast_divmod(linear_idx, tiles_per_group)
        first_n = group_idx * swizzle_size
        m_block: int32 = 0
        n_block: int32 = 0
        remainder = num_n_blocks - num_n_blocks // swizzle_size * swizzle_size
        last_group_width = remainder if remainder > 0 else swizzle_size
        if first_n + swizzle_size <= num_n_blocks:
            m_block, r = self.fast_divmod(in_group_idx, swizzle_size)
            n_block = first_n + r
        else:
            m_block, r = self.fast_divmod(in_group_idx, last_group_width)
            n_block = first_n + r
        return m_block, n_block

    def __call__(
        self,
        m_size: int32,
        n_size: int,
        k_size: int,
        a_ptr: ~float16,
        b_ptr: ~float16,
        c_ptr: ~float16,
    ):
        num_stages = self.num_stages
        block_m, block_n, block_k = self.block_m, self.block_n, self.block_k
        block_m_half = block_m // 2

        num_m_blocks = cdiv(m_size, block_m)
        num_n_blocks = cdiv(n_size, block_n)
        self.attrs.blocks = num_m_blocks * num_n_blocks
        self.attrs.warps = 9  # 1 producer + 2 consumer WGs (4 warps each)

        m_block, n_block = self.compute_block_coord(
            self.blockIdx.x, num_m_blocks, num_n_blocks
        )
        offset_m: int32 = m_block * block_m
        offset_n: int32 = n_block * block_n

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[n_size, k_size])
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        # Per-WG A slab: index as sa[stage, wg_idx].
        sa = self.shared_tensor(
            dtype=float16, shape=[num_stages, 2, block_m_half, block_k]
        )
        sb = self.shared_tensor(dtype=float16, shape=[num_stages, block_n, block_k])

        tma_pipe = Pipeline(
            num_stages, producer_arrive_count=1, consumer_arrive_count=256
        )

        with self.thread_group(thread_begin=256, num_threads=32):  # TMA producer
            for offset_k in self.range(0, k_size, block_k, unroll=num_stages):
                tma_pipe.producer_acquire()
                with self.single_thread():
                    self.mbarrier.arrive_and_expect_tx(
                        tma_pipe.producer_barrier(),
                        transaction_bytes=sa[tma_pipe.producer_stage, 0].nbytes
                        + sa[tma_pipe.producer_stage, 1].nbytes
                        + sb[tma_pipe.producer_stage].nbytes,
                    )
                self.tma.global_to_shared(
                    src=ga,
                    dst=sa[tma_pipe.producer_stage, 0],
                    offsets=[offset_m, offset_k],
                    mbarrier=tma_pipe.producer_barrier(),
                )
                self.tma.global_to_shared(
                    src=ga,
                    dst=sa[tma_pipe.producer_stage, 1],
                    offsets=[offset_m + block_m_half, offset_k],
                    mbarrier=tma_pipe.producer_barrier(),
                )
                self.tma.global_to_shared(
                    src=gb,
                    dst=sb[tma_pipe.producer_stage],
                    offsets=[offset_n, offset_k],
                    mbarrier=tma_pipe.producer_barrier(),
                )
                tma_pipe.producer_advance()

            for _ in self.range(min(num_stages, cdiv(k_size, block_k))):
                tma_pipe.producer_acquire()
                tma_pipe.producer_advance()

        with self.thread_group(thread_begin=0, num_threads=128):  # consumer WG0
            acc0 = self.register_tensor(
                dtype=float32, shape=[block_m_half, block_n], init=0.0
            )
            tma_pipe.consumer_acquire()
            self.wgmma.fence()
            self.wgmma.mma(
                sa[tma_pipe.consumer_stage, 0],
                sb[tma_pipe.consumer_stage].transpose(),
                acc0,
            )
            self.wgmma.commit_group()
            tma_pipe.consumer_advance()

            for offset_k in self.range(block_k, k_size, block_k, unroll=num_stages):
                tma_pipe.consumer_acquire()
                self.wgmma.fence()
                self.wgmma.mma(
                    sa[tma_pipe.consumer_stage, 0],
                    sb[tma_pipe.consumer_stage].transpose(),
                    acc0,
                )
                self.wgmma.commit_group()
                self.wgmma.wait_group(1)
                self.mbarrier.arrive(tma_pipe.prev_consumer_barrier())
                tma_pipe.consumer_advance()

            self.wgmma.wait_group(0)
            self.mbarrier.arrive(tma_pipe.prev_consumer_barrier())

            casted0 = self.cast(acc0, dtype=float16)
            self.store_global(gc, casted0, offsets=[offset_m, offset_n])

        with self.thread_group(thread_begin=128, num_threads=128):  # consumer WG1
            acc1 = self.register_tensor(
                dtype=float32, shape=[block_m_half, block_n], init=0.0
            )
            tma_pipe.consumer_acquire()
            self.wgmma.fence()
            self.wgmma.mma(
                sa[tma_pipe.consumer_stage, 1],
                sb[tma_pipe.consumer_stage].transpose(),
                acc1,
            )
            self.wgmma.commit_group()
            tma_pipe.consumer_advance()

            for offset_k in self.range(block_k, k_size, block_k, unroll=num_stages):
                tma_pipe.consumer_acquire()
                self.wgmma.fence()
                self.wgmma.mma(
                    sa[tma_pipe.consumer_stage, 1],
                    sb[tma_pipe.consumer_stage].transpose(),
                    acc1,
                )
                self.wgmma.commit_group()
                self.wgmma.wait_group(1)
                self.mbarrier.arrive(tma_pipe.prev_consumer_barrier())
                tma_pipe.consumer_advance()

            self.wgmma.wait_group(0)
            self.mbarrier.arrive(tma_pipe.prev_consumer_barrier())

            casted1 = self.cast(acc1, dtype=float16)
            self.store_global(gc, casted1, offsets=[offset_m + block_m_half, offset_n])


def main():
    headers = ["m", "n", "k", "name", "latency (ms)", "tflops"]
    workloads = [
        [4096, 4096, 4096],
        [4096, 4096, 14336],
        [8192, 8192, 8192],
        [10240, 10240, 10240],
    ]

    rows = []
    for m, n, k in workloads:
        matmul = MatmulWGMMAV5()

        a = (torch.rand(m, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
        b = (torch.rand(n, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
        c_actual = torch.empty(m, n, dtype=torch.float16).cuda()
        c_expect = a @ b.T
        matmul(m, n, k, a, b, c_actual)
        torch.cuda.synchronize()

        torch.testing.assert_close(c_expect, c_actual, atol=1e-2, rtol=1e-2)

        for name, func in [
            ("torch", lambda: torch.matmul(a, b.T, out=c_expect)),
            ("tilus", lambda: matmul(m, n, k, a, b, c_actual)),
        ]:
            latency = benchmark_func(func, warmup=5, repeat=20)
            tflops = 2 * m * n * k / latency * 1e-9
            rows.append([m, n, k, name, latency, tflops])

    df = pandas.DataFrame(rows, columns=headers)
    print(df)


if __name__ == "__main__":
    main()
