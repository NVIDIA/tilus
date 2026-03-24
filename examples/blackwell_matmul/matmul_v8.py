# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import pandas
import tilus
import torch
from tilus import RegisterTensor, SharedTensor, float16, float32, int32, uint32
from tilus.utils.ncu_utils import ncu_run
from tilus.utils import benchmark_func, cdiv

tilus.option.cache_dir(os.path.join(os.path.dirname(__file__), "cache"))
tilus.option.debug.dump_ir()


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

    def producer_acquire(self, scope: str = "cta"):
        self.mbarrier.wait(
            barrier=self.empty_barriers[self.producer_stage],
            phase=self.producer_phase,
            scope=scope,
        )

    def producer_barrier(self) -> RegisterTensor:
        return self.full_barriers[self.producer_stage]

    def producer_advance(self):
        self.producer_stage = (self.producer_stage + 1) % self.num_stages
        self.producer_phase = self.producer_phase ^ (self.producer_stage == 0)

    def consumer_acquire(self, scope: str = "cta"):
        self.mbarrier.wait(
            barrier=self.full_barriers[self.consumer_stage],
            phase=self.consumer_phase,
            scope=scope,
        )

    def consumer_barrier(self) -> RegisterTensor:
        return self.empty_barriers[self.consumer_stage]

    def consumer_advance(self):
        self.consumer_stage = (self.consumer_stage + 1) % self.num_stages
        self.consumer_phase = self.consumer_phase ^ (self.consumer_stage == 0)


@tilus.autotune("block_m", [256])
@tilus.autotune("block_n, e_block_n", [[256, 16], [256, 32]])
@tilus.autotune("block_k", [64])
@tilus.autotune("tma_stages", [5, 6])
@tilus.autotune("mma_stages", [2])
@tilus.autotune("swizzle_size", [4, 8, 16])
class BlackwellMatmulV8(tilus.Script):
    debug_schedule = dict(
        block_m=256,
        block_n=256,
        block_k=64,
        tma_stages=6,
        mma_stages=2,
        e_block_n=32,
        swizzle_size=8,
    )

    def __init__(
        self,
        block_m: int,
        block_n: int,
        block_k: int,
        tma_stages: int,
        mma_stages: int,
        e_block_n: int,
        swizzle_size: int,
    ):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.e_block_n = e_block_n
        self.tma_stages = tma_stages
        self.mma_stages = mma_stages
        self.swizzle_size = swizzle_size
        self.clc_stages = 1

    def compute_block_coord(
        self, linear_idx: int32, num_m_blocks: int32, num_n_blocks: int
    ):
        swizzle_size = self.swizzle_size
        tiles_per_group = num_m_blocks * swizzle_size
        group_idx = linear_idx // tiles_per_group
        in_group_idx = linear_idx % tiles_per_group
        first_n = group_idx * swizzle_size
        group_width = num_n_blocks - first_n
        if group_width > swizzle_size:
            group_width = swizzle_size
        m_block = in_group_idx // group_width
        n_block = first_n + in_group_idx % group_width
        return m_block, n_block

    def query_clc_response(self, s_clc_response: SharedTensor, pipe: Pipeline):
        pipe.consumer_acquire(scope="cluster")
        response = s_clc_response[pipe.consumer_stage]
        is_valid, new_blockIdx = self.clc.query_response(response)
        self.fence.async_view(space="shared")
        self.mbarrier.arrive_and_expect_tx_remote(
            pipe.consumer_barrier(), transaction_bytes=0, target_rank=0
        )
        pipe.consumer_advance()
        return is_valid, new_blockIdx

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
        num_m_blocks = cdiv(m_size, self.block_m)
        num_n_blocks = cdiv(n_size, self.block_n)
        self.attrs.blocks = num_m_blocks * num_n_blocks * 2, 1
        self.attrs.cluster_blocks = 2
        self.attrs.warps = 8

        block_m = self.block_m
        block_n = self.block_n
        block_k = self.block_k
        e_block_n = self.e_block_n
        tma_stages = self.tma_stages
        mma_stages = self.mma_stages
        clc_stages = self.clc_stages

        g_a = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        g_b = self.global_view(b_ptr, dtype=float16, shape=[n_size, k_size])
        g_c = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])

        s_a = self.shared_tensor(dtype=float16, shape=[tma_stages, block_m // 2, block_k])
        s_b = self.shared_tensor(dtype=float16, shape=[tma_stages, block_n // 2, block_k])
        t_acc = self.tcgen05.alloc(
            dtype=float32, shape=[mma_stages, block_m // 2, block_n], cta_group=2
        )

        s_clc_response = self.shared_tensor(dtype=int32, shape=[clc_stages, 4])

        tma_pipe = Pipeline(tma_stages)
        mma_pipe = Pipeline(
            mma_stages, consumer_arrive_count=128
        )  # 4 warps (epilogue warps)
        clc_pipe = Pipeline(
            clc_stages, consumer_arrive_count=224 * 2
        )  # 7 warps * 2 blocks

        cta_rank = self.cluster.blockRank

        self.cluster_sync()

        with self.single_warp(0):  # tma worker (gmem -> smem)
            m_block_0, n_block_0 = self.compute_block_coord(
                self.blockIdx.x // 2, num_m_blocks, num_n_blocks
            )
            offset_m_a = (m_block_0 * 2 + cta_rank) * (block_m // 2)
            offset_n_b = n_block_0 * block_n + cta_rank * (block_n // 2)
            while True:
                for offset_k in range(0, k_size, block_k):
                    tma_pipe.producer_acquire()
                    mbarrier = tma_pipe.producer_barrier()
                    if cta_rank == 0:
                        with self.single_thread():
                            # the mbarrier on CTA0 will track the completion of both CTAs' loading
                            transaction_bytes = (s_a[0].nbytes + s_b[0].nbytes) * 2
                            self.mbarrier.arrive_and_expect_tx(
                                mbarrier, transaction_bytes
                            )
                    else:
                        # get the mbarrier address in the CTA0 to signal
                        mbarrier = self.cluster.map_shared_addr(mbarrier, target_rank=0)
                    with self.single_thread():
                        self.tma.global_to_shared(
                            src=g_a,
                            dst=s_a[tma_pipe.producer_stage],
                            offsets=[offset_m_a, offset_k],
                            mbarrier=mbarrier,
                            cta_group=2,
                        )
                        self.tma.global_to_shared(
                            src=g_b,
                            dst=s_b[tma_pipe.producer_stage],
                            offsets=[offset_n_b, offset_k],
                            mbarrier=mbarrier,
                            cta_group=2,
                        )
                    tma_pipe.producer_advance()

                is_valid, new_blockIdx = self.query_clc_response(s_clc_response, clc_pipe)
                if not is_valid:
                    break
                m_block_0, n_block_0 = self.compute_block_coord(
                    new_blockIdx.x // 2, num_m_blocks, num_n_blocks
                )
                offset_m_a = (m_block_0 * 2 + cta_rank) * (block_m // 2)
                offset_n_b = n_block_0 * block_n + cta_rank * (block_n // 2)

        with self.single_warp(1):  # mma worker (smem -> tmem)
            while True:
                with self.single_thread():
                    if cta_rank == 0:
                        mma_pipe.producer_acquire()
                        for offset_k in range(0, k_size, block_k):
                            tma_pipe.consumer_acquire()
                            self.tcgen05.mma(
                                s_a[tma_pipe.consumer_stage],
                                s_b[tma_pipe.consumer_stage].transpose(),
                                t_acc[mma_pipe.producer_stage],
                                enable_input_d=offset_k != 0,
                                cta_group=2,
                            )
                            self.tcgen05.commit(
                                mbarrier=tma_pipe.consumer_barrier(),
                                cta_group=2,
                                multicast_mask=0b11,
                            )
                            tma_pipe.consumer_advance()
                        self.tcgen05.commit(
                            mbarrier=mma_pipe.producer_barrier(),
                            cta_group=2,
                            multicast_mask=0b11,
                        )
                        mma_pipe.producer_advance()

                is_valid, new_blockIdx = self.query_clc_response(s_clc_response, clc_pipe)
                if not is_valid:
                    break

        with self.single_warp(2):  # scheduler
            while True:
                if cta_rank == 0:
                    clc_pipe.producer_acquire(
                        scope="cluster"
                    )  # peer cta will arrive this barrier, need 'cluster'scoped acquire
                    self.mbarrier.arrive_and_expect_tx_multicast(
                        clc_pipe.producer_barrier(),
                        transaction_bytes=16,
                        multicast_mask=0b11,
                    )
                    with self.single_thread():
                        self.clc.try_cancel(
                            s_clc_response[clc_pipe.producer_stage],
                            mbarrier=clc_pipe.producer_barrier(),
                            multicast=True,
                        )
                    clc_pipe.producer_advance()

                is_valid, new_blockIdx = self.query_clc_response(s_clc_response, clc_pipe)
                if not is_valid:
                    break

        with self.warp_group(warp_begin=4, num_warps=4):  # epilogue (tmem -> gmem)
            s_c = self.shared_tensor(dtype=float16, shape=[block_m // 2, self.e_block_n])
            m_block_e, n_block_e = self.compute_block_coord(
                self.blockIdx.x // 2, num_m_blocks, num_n_blocks
            )
            offset_m_c = (m_block_e * 2 + cta_rank) * (block_m // 2)
            offset_n_c = n_block_e * block_n
            while True:
                mma_pipe.consumer_acquire()

                for e_offset_n in range(0, block_n, e_block_n):
                    t_acc_slice = self.tcgen05.slice(
                        t_acc[mma_pipe.consumer_stage],
                        offsets=[0, e_offset_n],
                        shape=[block_m // 2, e_block_n],
                        dims=[0, 1],
                    )
                    r_acc = self.tcgen05.load(t_acc_slice)
                    self.tcgen05.wait_load()
                    self.store_shared(s_c, r_acc.to(float16))
                    self.fence.async_view(space="shared")
                    self.sync()
                    with self.single_thread():
                        self.tma.shared_to_global(
                            s_c,
                            g_c,
                            offsets=[offset_m_c, offset_n_c + e_offset_n],
                            dims=[0, 1],
                        )
                        self.tma.commit_group()
                        self.tma.wait_group(n=0)
                    self.sync()

                self.mbarrier.arrive(mma_pipe.consumer_barrier())
                mma_pipe.consumer_advance()

                is_valid, new_blockIdx = self.query_clc_response(s_clc_response, clc_pipe)
                if not is_valid:
                    break
                m_block_e, n_block_e = self.compute_block_coord(
                    new_blockIdx.x // 2, num_m_blocks, num_n_blocks
                )
                offset_m_c = (m_block_e * 2 + cta_rank) * (block_m // 2)
                offset_n_c = n_block_e * block_n

        # all allocated tensor memory must be deallocated
        self.sync()
        self.tcgen05.dealloc(t_acc)


def main(bench=True):
    matmul = BlackwellMatmulV8()

    headers = ["m", "n", "k", "name", "latency (ms)", "tflops"]
    rows: list = []

    for m_size, n_size, k_size in [
        # [4096, 4096, 4096],
        # [4096, 4096, 14336],
        # [8192, 8192, 8192],
        [10240, 10240, 10240],
    ]:
        print(f"Running with m_size={m_size}, n_size={n_size}, k_size={k_size}")
        a = torch.randint(0, 2, size=(m_size, k_size), dtype=torch.float16, device="cuda")
        b = torch.randint(0, 2, size=(n_size, k_size), dtype=torch.float16, device="cuda")
        c_actual = torch.empty(m_size, n_size, dtype=torch.float16, device="cuda")
        c_expected = torch.empty(m_size, n_size, dtype=torch.float16, device="cuda")

        matmul(m_size, n_size, k_size, a, b, c_actual)
        torch.cuda.synchronize()

        torch.matmul(a, b.T, out=c_expected)
        torch.cuda.synchronize()

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
    # main(bench=False)
    # ncu_run(main, bench=False, kernel_regex="hidet|nvjet")
