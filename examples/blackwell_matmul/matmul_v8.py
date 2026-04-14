# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pandas
import tilus
import torch
from tilus import RegisterTensor, SharedTensor, float16, float32, int32, uint32
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


@tilus.autotune("block_m", [256])
@tilus.autotune("block_n, e_block_n", [[256, 16], [256, 32]])
@tilus.autotune("block_k", [64])
@tilus.autotune("tma_stages", [5, 6])
@tilus.autotune("mma_stages", [2])
@tilus.autotune("swizzle_size", [4, 8, 16])
class BlackwellMatmulV8(tilus.Script):
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

    def query_clc_response(self, s_clc_response: SharedTensor, pipe: Pipeline):
        pipe.consumer_acquire()
        response = s_clc_response[pipe.consumer_stage]
        is_valid, new_blockIdx = self.clc.query_response(response)
        self.mbarrier.arrive_and_expect_tx_remote(
            pipe.consumer_barrier(),
            transaction_bytes=0,
            target_rank=0,
            sem="relaxed",
            scope="cluster",
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

        self.cluster.sync()

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
                    # TMA loads are warp-cooperative: all 32 threads participate,
                    # predication is handled internally by the emitter
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
                # tcgen05.mma and tcgen05.commit are warp-cooperative:
                # all 32 threads participate, predication handled internally
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
                    clc_pipe.producer_acquire()
                    self.mbarrier.arrive_and_expect_tx_multicast(
                        clc_pipe.producer_barrier(),
                        transaction_bytes=16,
                        multicast_mask=0b11,
                        sem="relaxed",
                        scope="cluster",
                    )
                    # clc.try_cancel is warp-cooperative: predication handled internally
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
                    self.fence.proxy_async(space="shared")
                    self.sync()
                    # TMA store is warp-cooperative: use single_warp() to select
                    # one warp, predication handled internally by the emitter
                    with self.single_warp():
                        self.tma.shared_to_global(
                            s_c,
                            g_c,
                            offsets=[offset_m_c, offset_n_c + e_offset_n],
                            dims=[0, 1],
                        )
                        self.tma.commit_group()
                        self.tma.wait_group(n=0, read=True)
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
        self.cluster.sync()
        self.tcgen05.dealloc(t_acc)


def main(bench=True):
    matmul = BlackwellMatmulV8()

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

    if bench:
        df = pandas.DataFrame(rows, columns=headers)
        print(df)


if __name__ == "__main__":
    main(bench=True)
    # tilus.utils.ncu_run(main, bench=False, kernel_regex="tilus|nvjet")
