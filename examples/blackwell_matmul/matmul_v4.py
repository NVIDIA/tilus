# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import pandas
import tilus
import torch
from tilus import RegisterTensor, float16, float32, int32, uint32
from tilus.utils import benchmark_func, cdiv

tilus.option.cache_dir("./cache")


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
        # wait until the current stage is free (consumer has finished with it)
        self.mbarrier.wait(
            barrier=self.empty_barriers[self.producer_stage],
            phase=self.producer_phase,
            sem="relaxed",
            scope="cta",
        )

    def producer_barrier(self) -> RegisterTensor:
        # return the barrier to signal when the producer has filled this stage
        return self.full_barriers[self.producer_stage]

    def producer_advance(self):
        # advance to the next stage; flip phase when wrapping around
        self.producer_stage = (self.producer_stage + 1) % self.num_stages
        self.producer_phase = self.producer_phase ^ (self.producer_stage == 0)

    def consumer_acquire(self):
        # wait until the current stage is filled (producer has loaded data)
        self.mbarrier.wait(
            barrier=self.full_barriers[self.consumer_stage],
            phase=self.consumer_phase,
            sem="relaxed",
            scope="cta",
        )

    def consumer_barrier(self) -> RegisterTensor:
        # return the barrier to signal when the consumer has consumed this stage
        return self.empty_barriers[self.consumer_stage]

    def consumer_advance(self):
        # advance to the next stage; flip phase when wrapping around
        self.consumer_stage = (self.consumer_stage + 1) % self.num_stages
        self.consumer_phase = self.consumer_phase ^ (self.consumer_stage == 0)


@tilus.autotune("block_m, block_n, e_block_n", [[128, 128, 16], [128, 256, 16]])
@tilus.autotune("block_k", [32, 64])
@tilus.autotune("stages", [2, 3, 4])
@tilus.autotune("swizzle_size", [4, 8])
class BlackwellMatmulV4(tilus.Script):
    def __init__(
        self,
        block_m: int,
        block_n: int,
        block_k: int,
        stages: int,
        e_block_n: int,
        swizzle_size: int,
    ):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.stages = stages
        self.e_block_n = e_block_n
        self.swizzle_size = swizzle_size

    def compute_block_coord(
        self, linear_idx: int32, num_m_blocks: int32, num_n_blocks: int
    ):
        """Map a 1D linear block index to 2D (m_block, n_block) with swizzle grouping.

        Tiles within a swizzle group share N-columns, improving L2 cache reuse
        for the B matrix.
        """
        swizzle_size = self.swizzle_size
        tiles_per_group = num_m_blocks * swizzle_size
        group_idx, in_group_idx = self.fast_divmod(linear_idx, tiles_per_group)
        first_n = group_idx * swizzle_size
        m_block: int32 = 0
        n_block: int32 = 0
        # When num_n_blocks is divisible by swizzle_size, all groups are full and
        # last_group_width is never used. Use swizzle_size as a safe fallback to
        # avoid division-by-zero in the precompute.
        remainder = num_n_blocks - num_n_blocks // swizzle_size * swizzle_size
        last_group_width = remainder if remainder > 0 else swizzle_size
        if first_n + swizzle_size <= num_n_blocks:
            # Full group: swizzle_size is a compile-time constant
            m_block, r = self.fast_divmod(in_group_idx, swizzle_size)
            n_block = first_n + r
        else:
            # Last group: divisor is num_n_blocks % swizzle_size, which is grid-constant
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
        block_m = self.block_m
        block_n = self.block_n
        block_k = self.block_k
        stages = self.stages
        e_block_n = self.e_block_n

        num_m_blocks = cdiv(m_size, block_m)
        num_n_blocks = cdiv(n_size, block_n)
        # 1D grid: tile rasterization maps linear index to 2D coordinates
        self.attrs.blocks = num_m_blocks * num_n_blocks
        self.attrs.warps = 4

        # tile rasterization: swizzle for better L2 cache reuse of B columns
        m_block, n_block = self.compute_block_coord(
            self.blockIdx.x, num_m_blocks, num_n_blocks
        )
        offset_m: int32 = m_block * block_m
        offset_n: int32 = n_block * block_n

        g_a = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        g_b = self.global_view(b_ptr, dtype=float16, shape=[n_size, k_size])
        g_c = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        s_a = self.shared_tensor(dtype=float16, shape=[stages, block_m, block_k])
        s_b = self.shared_tensor(dtype=float16, shape=[stages, block_n, block_k])
        t_acc = self.tcgen05.alloc(dtype=float32, shape=[block_m, block_n])

        # Pipeline class encapsulates barrier/phase/stage management from V3
        tma_pipe = Pipeline(stages)
        flush_barrier = self.mbarrier.alloc(1)

        with self.thread_group(thread_begin=0, num_threads=32):
            for offset_k in self.range(0, k_size, block_k, unroll=stages):
                tma_pipe.producer_acquire()
                with self.single_thread():
                    self.mbarrier.arrive_and_expect_tx(
                        tma_pipe.producer_barrier(),
                        transaction_bytes=s_a[tma_pipe.producer_stage].nbytes
                        + s_b[tma_pipe.producer_stage].nbytes,
                    )
                self.tma.global_to_shared(
                    src=g_a,
                    dst=s_a[tma_pipe.producer_stage],
                    offsets=[offset_m, offset_k],
                    mbarrier=tma_pipe.producer_barrier(),
                )
                self.tma.global_to_shared(
                    src=g_b,
                    dst=s_b[tma_pipe.producer_stage],
                    offsets=[offset_n, offset_k],
                    mbarrier=tma_pipe.producer_barrier(),
                )
                tma_pipe.producer_advance()

        with self.thread_group(thread_begin=32, num_threads=32):
            for offset_k in self.range(0, k_size, block_k, unroll=stages):
                tma_pipe.consumer_acquire()
                self.tcgen05.mma(
                    s_a[tma_pipe.consumer_stage],
                    s_b[tma_pipe.consumer_stage].transpose(),
                    t_acc,
                    enable_input_d=offset_k != 0,
                )
                self.tcgen05.commit(mbarrier=tma_pipe.consumer_barrier())
                tma_pipe.consumer_advance()

            self.tcgen05.commit(mbarrier=flush_barrier)
            self.mbarrier.wait(flush_barrier, phase=0)

        self.sync()

        # TMA epilogue: tmem -> register -> shared -> global (via TMA)
        s_c = self.shared_tensor(dtype=float16, shape=[block_m, e_block_n])
        for e_offset_n in range(0, block_n, e_block_n):
            # slice a e_block_n-wide column from the accumulator
            t_acc_slice = self.tcgen05.slice(
                t_acc,
                offsets=[0, e_offset_n],
                shape=[block_m, e_block_n],
                dims=[0, 1],
            )
            r_acc = self.tcgen05.load(t_acc_slice)
            self.tcgen05.wait_load()
            self.store_shared(s_c, r_acc.to(float16))
            # fence: make generic-proxy writes visible to async-proxy (TMA)
            self.fence.proxy_async(space="shared")
            self.sync()
            with self.single_warp():
                # TMA bulk store from shared to global
                self.tma.shared_to_global(
                    s_c,
                    g_c,
                    offsets=[offset_m, offset_n + e_offset_n],
                    dims=[0, 1],
                )
                self.tma.commit_group()
                self.tma.wait_group(n=0, read=True)
            self.sync()

        self.sync()
        self.tcgen05.dealloc(t_acc)


def main(bench=True):
    matmul = BlackwellMatmulV4()

    headers = ["m", "n", "k", "name", "latency (ms)", "tflops"]
    rows = []

    for m_size, n_size, k_size in [
        [8192, 8192, 8192],
    ]:
        print(f"Running with m_size={m_size}, n_size={n_size}, k_size={k_size}")
        a = torch.randn(m_size, k_size, dtype=torch.float16, device="cuda")
        b = torch.randn(n_size, k_size, dtype=torch.float16, device="cuda")
        c = torch.empty(m_size, n_size, dtype=torch.float16, device="cuda")

        c_ref = a @ b.T
        torch.cuda.synchronize()

        matmul(m_size, n_size, k_size, a, b, c)
        torch.cuda.synchronize()

        torch.testing.assert_close(c, c_ref, atol=1e-2, rtol=1e-2)

        # benchmark
        if bench:
            for name, func in [
                ("torch", lambda: a @ b.T),
                ("tilus", lambda: matmul(m_size, n_size, k_size, a, b, c)),
            ]:
                latency = benchmark_func(func, warmup=5, repeat=100)
                tflops = 2 * m_size * n_size * k_size / latency * 1e-9
                rows.append([m_size, n_size, k_size, name, latency, tflops])
                time.sleep(3)  # sleep 3s to cool down the GPU between runs

    if bench:
        df = pandas.DataFrame(rows, columns=headers)
        print(df)


if __name__ == "__main__":
    main(bench=True)
    # tilus.utils.ncu_run(main, bench=False, kernel_regex="tilus|nvjet")
