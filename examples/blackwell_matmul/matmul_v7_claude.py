# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Persistent 2SM matmul with Cluster Launch Control (CLC).

Built on v6 (2SM cluster with cta_group=2) + v7_clc (persistent blocks via CLC).

Each cluster of 2 CTAs processes its initial tile, then uses CLC to steal work
from unstarted clusters. CTA0 loads top half of A and left half of B; CTA1 loads
bottom half of A and right half of B. MMA is issued on CTA0 with cta_group=2.

Workers:
- LoadWorker (warp 0, threads 0-31): TMA loads A and B, issues CLC fetch_next
- MmaWorker (warp 1, thread 32, CTA0 only): MMA with cta_group=2
- Epilogue (all threads): store results via TMA, reset accumulator
"""
import os

import pandas
import tilus
import torch
from tilus import float16, float32, int32, uint32
from tilus.ir.tensor import GlobalTensor, RegisterTensor
from tilus.utils import benchmark_func, cdiv

tilus.option.cache_dir(os.path.join(os.path.dirname(__file__), "cache"))


class Pipeline(tilus.Class):
    def __init__(
        self, num_stages: int, producer_arrive_count: int, consumer_arrive_count: int
    ):
        self.num_stages: int = num_stages
        self.full_barriers = self.mbarrier.alloc(
            [consumer_arrive_count for _ in range(num_stages)]
        )
        self.empty_barriers = self.mbarrier.alloc(
            [producer_arrive_count for _ in range(num_stages)]
        )
        self.producer_stage: int32 = 0
        self.consumer_stage: int32 = 0
        self.producer_phase: uint32 = self.mbarrier.producer_initial_phase
        self.consumer_phase: uint32 = self.mbarrier.consumer_initial_phase

    def producer_acquire(self):
        self.mbarrier.wait(
            barrier=self.full_barriers[self.producer_stage], phase=self.producer_phase
        )

    def producer_advance(self):
        self.producer_stage = (self.producer_stage + 1) % self.num_stages
        self.producer_phase = self.producer_phase ^ (self.producer_stage == 0)

    def producer_release_barrier(self) -> RegisterTensor:
        return self.empty_barriers[self.producer_stage]

    def consumer_acquire(self):
        self.mbarrier.wait(
            barrier=self.empty_barriers[self.consumer_stage], phase=self.consumer_phase
        )

    def consumer_advance(self):
        self.consumer_stage = (self.consumer_stage + 1) % self.num_stages
        self.consumer_phase = self.consumer_phase ^ (self.consumer_stage == 0)

    def consumer_release_barrier(self) -> RegisterTensor:
        return self.full_barriers[self.consumer_stage]


class Params(tilus.Class):
    def __init__(
        self,
        m_size: int32,
        n_size: int,
        k_size: int,
        block_m: int,
        block_n: int,
        block_k: int,
        g_a: GlobalTensor,
        g_b: GlobalTensor,
        g_c: GlobalTensor,
    ):
        self.m_size: int32 = m_size
        self.n_size: int = n_size
        self.k_size: int = k_size
        self.block_m: int = block_m
        self.block_n: int = block_n
        self.block_k: int = block_k
        self.g_a: GlobalTensor = g_a
        self.g_b: GlobalTensor = g_b
        self.g_c: GlobalTensor = g_c


class LoadPipeline(Pipeline):
    def __init__(self, num_stages: int, params: Params):
        super().__init__(
            num_stages=num_stages, producer_arrive_count=1, consumer_arrive_count=1
        )
        block_m, block_n, block_k = params.block_m, params.block_n, params.block_k
        self.params: Params = params
        self.s_a = self.shared_tensor(
            dtype=float16, shape=[num_stages, block_m // 2, block_k]
        )
        self.s_b = self.shared_tensor(
            dtype=float16, shape=[num_stages, block_n // 2, block_k]
        )


class Scheduler(tilus.Class):
    """CLC-based tile scheduler. Uses multicast to broadcast response to all CTAs."""

    def __init__(self):
        self.barrier = self.mbarrier.alloc(count=1)
        self.phase: uint32 = self.mbarrier.consumer_initial_phase
        self.s_response = self.shared_tensor(dtype=int32, shape=[4])

    def fetch_next(self):
        """Issue async CLC cancel request. Must be called from >=32 threads for multicast."""
        self.clc.try_cancel(self.s_response, self.barrier, multicast=True)

    def query_next(self):
        """Wait for CLC response and decode. Returns (is_valid, blockIdx)."""
        self.mbarrier.wait(self.barrier, phase=self.phase)
        self.phase = self.phase ^ uint32(1)
        is_valid, blockIdx = self.clc.query_response(self.s_response)
        return is_valid, blockIdx


@tilus.autotune(
    "block_m, block_n, e_block_n", [[256, 64, 16], [256, 128, 16], [256, 256, 16]]
)
@tilus.autotune("block_k", [16, 32, 64])
@tilus.autotune("stages", [2, 3, 4, 5, 6])
class BlackwellMatmulV7(tilus.Script):
    debug_schedule = dict(
        block_m=256,
        block_n=256,
        block_k=64,
        stages=5,
        e_block_n=16,
    )

    def __init__(
        self, block_m: int, block_n: int, block_k: int, stages: int, e_block_n: int
    ):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.stages = stages
        self.e_block_n = e_block_n

    def __call__(
        self,
        m_size: int32,
        n_size: int,
        k_size: int,
        a_ptr: ~float16,
        b_ptr: ~float16,
        c_ptr: ~float16,
    ):
        """
        Persistent 2SM matmul with CLC scheduling.

                            Input B (K, N)
                          +-------+-------+
                          |  b0   |  b1   |
                          |(K,N/2)|(K,N/2)|
                          |[CTA0] |[CTA1] |
                          +-------+-------+
        +--------------+  +---------------+
        |  a0 (M/2, K) |  |  d0 (M/2, N)  |
        |  [CTA0]      |  |  [CTA0]       |
        +--------------+  +---------------+
        |  a1 (M/2, K) |  |  d1 (M/2, N)  |
        |  [CTA1]      |  |  [CTA1]       |
        +--------------+  +---------------+
         Input A (M, K)     Output D (M, N)
        """
        self.attrs.blocks = [cdiv(m_size, self.block_m) * 2, cdiv(n_size, self.block_n)]
        self.attrs.cluster_blocks = [2, 1]
        self.attrs.warps = 4

        params = Params(
            m_size=m_size,
            n_size=n_size,
            k_size=k_size,
            block_m=self.block_m,
            block_n=self.block_n,
            block_k=self.block_k,
            g_a=self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size]),
            g_b=self.global_view(b_ptr, dtype=float16, shape=[n_size, k_size]),
            g_c=self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size]),
        )

        scheduler = Scheduler()
        pipe = LoadPipeline(num_stages=self.stages, params=params)

        # Allocate tensor memory (once, reused across tiles)
        t_acc = self.tcgen05.alloc(
            dtype=float32,
            shape=[self.block_m // 2, self.block_n],
            init=0.0,
            cta_group=2,
        )
        flush_barrier = self.mbarrier.alloc(1)
        s_c = self.shared_tensor(
            dtype=float16, shape=[self.block_m // 2, self.e_block_n]
        )

        cta_rank = self.cluster.blockRank

        # Initial tile offsets from grid launch blockIdx
        offset_m_a: int32 = self.blockIdx.x * (self.block_m // 2)
        offset_n_b: int32 = (
            self.blockIdx.y * self.block_n + cta_rank * (self.block_n // 2)
        )
        offset_m_c: int32 = self.blockIdx.x * (self.block_m // 2)
        offset_n_c: int32 = self.blockIdx.y * self.block_n
        flush_phase: uint32 = self.mbarrier.consumer_initial_phase

        self.cluster.sync()

        while True:
            # === Load phase (warp 0, both CTAs) ===
            with self.thread_group(thread_begin=0, num_threads=32):
                for offset_k in self.range(
                    0, k_size, self.block_k, unroll=self.stages
                ):
                    pipe.producer_acquire()
                    mbarrier = pipe.producer_release_barrier()
                    if cta_rank == 0:
                        with self.single_thread():
                            transaction_bytes = (
                                pipe.s_a[0].nbytes + pipe.s_b[0].nbytes
                            ) * 2
                            self.mbarrier.arrive_and_expect_tx(
                                mbarrier, transaction_bytes
                            )
                    else:
                        mbarrier = self.cluster.map_shared_addr(
                            mbarrier, target_rank=0
                        )
                    with self.single_thread():
                        self.tma.global_to_shared(
                            src=params.g_a,
                            dst=pipe.s_a[pipe.producer_stage],
                            offsets=[offset_m_a, offset_k],
                            mbarrier=mbarrier,
                            cta_group=2,
                        )
                        self.tma.global_to_shared(
                            src=params.g_b,
                            dst=pipe.s_b[pipe.producer_stage],
                            offsets=[offset_n_b, offset_k],
                            mbarrier=mbarrier,
                            cta_group=2,
                        )
                    pipe.producer_advance()

                pass  # CLC fetch moved to after epilogue

            # === MMA phase (thread 32 on CTA0 only) ===
            if cta_rank == 0:
                with self.thread_group(thread_begin=32, num_threads=1):
                    for _ in self.range(
                        0, k_size, self.block_k, unroll=self.stages
                    ):
                        pipe.consumer_acquire()
                        self.tcgen05.mma(
                            pipe.s_a[pipe.consumer_stage],
                            pipe.s_b[pipe.consumer_stage].transpose(),
                            t_acc,
                            cta_group=2,
                        )
                        self.tcgen05.commit(
                            mbarrier=pipe.consumer_release_barrier(),
                            cta_group=2,
                            multicast_mask=0b11,
                        )
                        pipe.consumer_advance()
                    self.tcgen05.commit(
                        mbarrier=flush_barrier,
                        cta_group=2,
                        multicast_mask=0b11,
                    )

            # === Wait for MMA completion ===
            self.mbarrier.wait(flush_barrier, phase=flush_phase)
            self.sync()

            # === Epilogue: store result to global memory ===
            for e_offset_n in range(0, self.block_n, self.e_block_n):
                t_acc_slice = self.tcgen05.slice(
                    t_acc,
                    offsets=[0, e_offset_n],
                    shape=[self.block_m // 2, self.e_block_n],
                    dims=[0, 1],
                )
                r_acc = self.tcgen05.load(t_acc_slice)
                self.tcgen05.wait_load()
                self.store_shared(s_c, r_acc.to(float16))
                self.sync()
                with self.single_thread():
                    self.tma.shared_to_global(
                        s_c,
                        params.g_c,
                        offsets=[offset_m_c, offset_n_c + e_offset_n],
                        dims=[0, 1],
                    )
                    self.tma.commit_group()
                    self.tma.wait_group(n=0)
                self.sync()

            # === Reset accumulator for next tile ===
            self.tcgen05.store(
                t_acc,
                src=self.register_tensor(
                    dtype=float32,
                    shape=[self.block_m // 2, self.block_n],
                    init=0.0,
                ),
            )
            self.tcgen05.wait_store()
            self.sync()

            # === Fetch next tile via CLC (after all computation is done) ===
            with self.thread_group(thread_begin=0, num_threads=32):
                if cta_rank == 0:
                    scheduler.fetch_next()

            # === Query CLC for next tile ===
            is_valid, new_blockIdx = scheduler.query_next()
            # break  # DEBUG: always break to test CLC overhead
            if is_valid:
                offset_m_a = (new_blockIdx.x + cta_rank) * (self.block_m // 2)
                offset_n_b = (
                    new_blockIdx.y * self.block_n
                    + cta_rank * (self.block_n // 2)
                )
                offset_m_c = (new_blockIdx.x + cta_rank) * (self.block_m // 2)
                offset_n_c = new_blockIdx.y * self.block_n
            else:
                break

            flush_phase = flush_phase ^ uint32(1)

        # Cleanup
        self.sync()
        self.tcgen05.dealloc(t_acc)


def main(bench=True):
    matmul = BlackwellMatmulV7()

    headers = ["m", "n", "k", "name", "latency (ms)", "tflops"]
    rows: list = []

    for m_size, n_size, k_size in [
        # [4096, 4096, 4096],
        # [4096, 4096, 14336],
        # [8192, 8192, 8192],
        [10240, 10240, 10240],
    ]:
        print(f"Running with m_size={m_size}, n_size={n_size}, k_size={k_size}")
        a = torch.randn(m_size, k_size, dtype=torch.float16, device="cuda")
        b = torch.randn(n_size, k_size, dtype=torch.float16, device="cuda")
        c_actual = torch.empty(m_size, n_size, dtype=torch.float16, device="cuda")
        c_expected = torch.empty(m_size, n_size, dtype=torch.float16, device="cuda")

        matmul(m_size, n_size, k_size, a, b, c_actual)
        torch.matmul(a, b.T, out=c_expected)
        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

        # benchmark
        if bench:
            for name, func in [
                ("torch", lambda: torch.matmul(a, b.T, out=c_expected)),
                ("tilus", lambda: matmul(m_size, n_size, k_size, a, b, c_actual)),
            ]:
                latency = benchmark_func(func, warmup=5, repeat=20)
                tflops = 2 * m_size * n_size * k_size / latency * 1e-9
                rows.append([m_size, n_size, k_size, name, latency, tflops])

    if bench:
        df = pandas.DataFrame(rows, columns=headers)
        print(df)


if __name__ == "__main__":
    main(bench=True)
