# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import pandas
import tilus
import torch
from tilus import float16, float32, int32, uint32
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedTensor
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
    def __init__(
        self,
        num_stages: int,
        params: Params,
    ):
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


class LoadWorker(tilus.Class):
    def __init__(self, pipe: LoadPipeline, params: Params):
        self.pipe: LoadPipeline = pipe
        self.params: Params = params

    def async_run(self):
        pipe, params = self.pipe, self.params
        s_a: SharedTensor = pipe.s_a
        s_b: SharedTensor = pipe.s_b
        num_stages: int = pipe.num_stages
        k_size, block_k = params.k_size, params.block_k
        cta_rank = self.cluster.blockRank
        offset_m = self.blockIdx.x * (params.block_m // 2)
        offset_n = self.blockIdx.y * params.block_n + cta_rank * (params.block_n // 2)

        with self.thread_group(thread_begin=0, num_threads=32):
            for offset_k in self.range(0, k_size, block_k, unroll=num_stages):
                self.pipe.producer_acquire()
                mbarrier = pipe.producer_release_barrier()
                if cta_rank == 0:
                    with self.single_thread():
                        # the mbarrier on CTA0 will track the completion of both CTAs' loading
                        transaction_bytes = (s_a[0].nbytes + s_b[0].nbytes) * 2
                        self.mbarrier.arrive_and_expect_tx(mbarrier, transaction_bytes)
                else:
                    # get the mbarrier address in the CTA0 to signal
                    mbarrier = self.cluster.map_shared_addr(mbarrier, target_rank=0)
                with self.single_thread():
                    self.tma.global_to_shared(
                        src=params.g_a,
                        dst=s_a[pipe.producer_stage],
                        offsets=[offset_m, offset_k],
                        mbarrier=mbarrier,
                        cta_group=2,
                    )
                    self.tma.global_to_shared(
                        src=params.g_b,
                        dst=s_b[pipe.producer_stage],
                        offsets=[offset_n, offset_k],
                        mbarrier=mbarrier,
                        cta_group=2,
                    )
                pipe.producer_advance()


class MmaWorker(tilus.Class):
    def __init__(self, pipe: LoadPipeline, params: Params):
        self.pipe: LoadPipeline = pipe
        self.params: Params = params
        self.t_acc = self.tcgen05.alloc(
            dtype=float32,
            shape=[params.block_m // 2, params.block_n],
            init=0.0,
            cta_group=2,
        )
        self.flush_barrier = self.mbarrier.alloc(1)

    def async_run(self):
        pipe = self.pipe
        s_a, s_b = pipe.s_a, pipe.s_b
        num_stages: int = pipe.num_stages
        cta_rank = self.cluster.blockRank
        if cta_rank == 0:
            with self.thread_group(thread_begin=32, num_threads=1):
                for offset_k in self.range(
                    0, self.params.k_size, self.params.block_k, unroll=num_stages
                ):
                    pipe.consumer_acquire()
                    self.tcgen05.mma(
                        s_a[pipe.consumer_stage],
                        s_b[pipe.consumer_stage].transpose(),
                        self.t_acc,
                        cta_group=2,
                    )
                    self.tcgen05.commit(
                        mbarrier=pipe.consumer_release_barrier(),
                        cta_group=2,
                        multicast_mask=0b11,
                    )
                    pipe.consumer_advance()
                self.tcgen05.commit(
                    mbarrier=self.flush_barrier, cta_group=2, multicast_mask=0b11
                )
        self.mbarrier.wait(self.flush_barrier, phase=0)


@tilus.autotune(
    "block_m, block_n, e_block_n", [[256, 64, 16], [256, 128, 16], [256, 256, 16]]
)
@tilus.autotune("block_k", [16, 32, 64])
@tilus.autotune("stages", [2, 3, 4, 5, 6])
class BlackwellMatmulV6(tilus.Script):
    # debug_schedule = dict(
    #     block_m=256,
    #     block_n=256,
    #     block_k=64,
    #     stages=5,
    #     e_block_n=16,
    # )
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
        Each CTA provides its own slice of A, B, and D.
        CTA0 = CTA with last bit of cluster rank = 0
        CTA1 = CTA with last bit of cluster rank = 1

                            Input B (K, N)
                          ┌───────┬───────┐
                          │  b0   │  b1   │
                          │(K,N/2)│(K,N/2)│
                          │[CTA0] │[CTA1] │
                          └───────┴───────┘
        ┌──────────────┐  ┌───────────────┐
        │  a0 (M/2, K) │  │  d0 (M/2, N)  │
        │  [CTA0]      │  │  [CTA0]       │
        ├──────────────┤  ├───────────────┤
        │  a1 (M/2, K) │  │  d1 (M/2, N)  │
        │  [CTA1]      │  │  [CTA1]       │
        └──────────────┘  └───────────────┘
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

        pipe = LoadPipeline(num_stages=self.stages, params=params)
        load_worker = LoadWorker(pipe, params)
        mma_worker = MmaWorker(pipe, params)

        # producer
        load_worker.async_run()

        # consumer
        mma_worker.async_run()

        self.sync()

        # store the result back to global memory
        offset_m: int32 = (self.block_m // 2) * self.blockIdx.x
        offset_n: int32 = self.block_n * self.blockIdx.y
        s_c = self.shared_tensor(dtype=float16, shape=[self.block_m // 2, self.e_block_n])

        for e_offset_n in range(0, self.block_n, self.e_block_n):
            t_acc = self.tcgen05.slice(
                mma_worker.t_acc,
                offsets=[0, e_offset_n],
                shape=[self.block_m // 2, self.e_block_n],
                dims=[0, 1],
            )
            r_acc = self.tcgen05.load(t_acc)
            self.tcgen05.wait_load()
            self.store_shared(s_c, r_acc.to(float16))
            self.sync()
            with self.single_thread():
                self.tma.shared_to_global(
                    s_c,
                    params.g_c,
                    offsets=[offset_m, offset_n + e_offset_n],
                    dims=[0, 1],
                )
                self.tma.commit_group()
                self.tma.wait_group(n=0)
            self.sync()

        # all allocated tensor memory must be deallocated
        self.sync()
        self.tcgen05.dealloc(mma_worker.t_acc)


def main(bench=True):
    matmul = BlackwellMatmulV6()

    headers = ["m", "n", "k", "name", "latency (ms)", "tflops"]
    rows: list = []

    for m_size, n_size, k_size in [
        [4096, 4096, 4096],
        [4096, 4096, 14336],
        [8192, 8192, 8192],
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
    # ncu_run(main, bench=False, kernel_regex="hidet|nvjet")
