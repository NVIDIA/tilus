# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# v6: production Hopper matmul. Warp-specialized producer/consumer pipeline,
# TMA async loads, WGMMA, and swizzle-rasterized tile order for L2 reuse.
#
# Optimizations from v5, all with the same fp32 accumulation as v0-v5:
# - Rebalance the register file with setmaxnreg: the producer warp group drops
#   to 40 registers so the two consumer groups can hold 232 each, enough for a
#   64x320 fp32 accumulator apiece on a 128x320 CTA tile.
# - Pair CTAs in a 2-CTA cluster along M. Both need the same B tile, so each
#   loads half of it and multicasts it to both, cutting L2 traffic.
# - Make the kernel persistent: one CTA per SM loops over tiles, so the producer
#   prefetches the next tile while the consumers run the epilogue.
# - Stage the epilogue through shared memory for coalesced TMA stores.

import math

import pandas
import tilus
import torch
from tilus import (
    GlobalTensor,
    RegisterTensor,
    SharedTensor,
    float16,
    float32,
    int32,
    uint32,
)
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


@tilus.autotune("num_stages", [3])
@tilus.autotune("block_m, block_n", [[128, 320]])
@tilus.autotune("block_k", [64])
@tilus.autotune("swizzle_size", [8])
class MatmulWGMMAV6(tilus.Script):
    def __init__(self, num_stages, block_m, block_n, block_k, swizzle_size):
        super().__init__()
        self.num_stages = num_stages
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.swizzle_size = swizzle_size
        # Persistent kernel: launch one CTA per SM.
        self.num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count

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

    def tile_offsets(self, tile: int32, num_m_pairs: int32, num_n_blocks: int):
        # A tile index names a pair of vertically adjacent CTA tiles; each CTA of
        # the cluster takes the one matching its rank.
        m_pair, n_block = self.compute_block_coord(tile, num_m_pairs, num_n_blocks)
        offset_m: int32 = (m_pair * 2 + self.cluster.blockRank) * self.block_m
        offset_n: int32 = n_block * self.block_n
        return offset_m, offset_n

    def release_stage(self, barrier: RegisterTensor):
        # Both CTAs multicast half of B into this stage, so both producers must
        # see it released before either refills it.
        for rank in range(2):
            self.mbarrier.arrive_and_expect_tx_remote(
                barrier, transaction_bytes=0, target_rank=rank, sem="relaxed"
            )

    def mma_stage(
        self,
        sa: SharedTensor,
        sb: SharedTensor,
        tma_pipe: Pipeline,
        acc0: RegisterTensor,
        acc1: RegisterTensor,
        consumer_idx: int,
    ):
        # Two m64n160k16 WGMMA streams cover the 320 columns; one commit group.
        self.wgmma.fence()
        self.wgmma.mma(
            sa[tma_pipe.consumer_stage, consumer_idx],
            sb[tma_pipe.consumer_stage, 0].transpose(),
            acc0,
        )
        self.wgmma.mma(
            sa[tma_pipe.consumer_stage, consumer_idx],
            sb[tma_pipe.consumer_stage, 1].transpose(),
            acc1,
        )
        self.wgmma.commit_group()

    def store_half(
        self,
        sc: SharedTensor,
        gc: GlobalTensor,
        acc: RegisterTensor,
        offset_m: int32,
        offset_n: int32,
    ):
        self.store_shared(sc, self.cast(acc, dtype=float16))
        self.fence.proxy_async(space="shared")
        self.sync()
        with self.single_thread():
            self.tma.shared_to_global(sc, gc, offsets=[offset_m, offset_n])
            self.tma.commit_group()

    def wait_store_read(self):
        # The staging slot may be rewritten once the TMA engine has read it.
        with self.single_thread():
            self.tma.wait_group(n=0, read=True)
        self.sync()

    def consume(
        self,
        sa: SharedTensor,
        sb: SharedTensor,
        sc: SharedTensor,
        gc: GlobalTensor,
        tma_pipe: Pipeline,
        consumer_idx: int,
        k_size: int,
        num_m_pairs: int32,
        num_n_blocks: int,
    ):
        self.setmaxnreg(232, "inc")
        half_n = self.block_n // 2
        num_tiles = num_m_pairs * num_n_blocks
        for tile in self.range(self.blockIdx.x // 2, num_tiles, self.gridDim.x // 2):
            offset_m, offset_n = self.tile_offsets(tile, num_m_pairs, num_n_blocks)
            acc0 = self.register_tensor(dtype=float32, shape=[64, half_n], init=0.0)
            acc1 = self.register_tensor(dtype=float32, shape=[64, half_n], init=0.0)

            tma_pipe.consumer_acquire()
            self.mma_stage(sa, sb, tma_pipe, acc0, acc1, consumer_idx)
            tma_pipe.consumer_advance()
            for offset_k in self.range(
                self.block_k, k_size, self.block_k, unroll=self.num_stages
            ):
                tma_pipe.consumer_acquire()
                self.mma_stage(sa, sb, tma_pipe, acc0, acc1, consumer_idx)
                self.wgmma.wait_group(1)
                with self.single_thread():
                    self.release_stage(tma_pipe.prev_consumer_barrier())
                tma_pipe.consumer_advance()
            self.wgmma.wait_group(0)
            with self.single_thread():
                self.release_stage(tma_pipe.prev_consumer_barrier())

            # Epilogue: each group stages its two 64x160 halves in turn through
            # its own slot of sc, so the two groups never wait on each other.
            row = offset_m + consumer_idx * 64
            self.wait_store_read()
            self.store_half(sc[consumer_idx], gc, acc0, row, offset_n)
            self.wait_store_read()
            self.store_half(sc[consumer_idx], gc, acc1, row, offset_n + half_n)

        with self.single_thread():
            self.tma.wait_group(n=0)

    def produce(
        self,
        ga: GlobalTensor,
        gb: GlobalTensor,
        sa: SharedTensor,
        sb: SharedTensor,
        tma_pipe: Pipeline,
        k_size: int,
        num_m_pairs: int32,
        num_n_blocks: int,
    ):
        half_n = self.block_n // 2
        cta_rank = self.cluster.blockRank
        num_tiles = num_m_pairs * num_n_blocks
        for tile in self.range(self.blockIdx.x // 2, num_tiles, self.gridDim.x // 2):
            offset_m, offset_n = self.tile_offsets(tile, num_m_pairs, num_n_blocks)
            for offset_k in self.range(0, k_size, self.block_k, unroll=self.num_stages):
                tma_pipe.producer_acquire()
                with self.single_thread():
                    # Expect the whole stage: this CTA's A plus both halves of
                    # B, one loaded here and one multicast by the peer.
                    self.mbarrier.arrive_and_expect_tx(
                        tma_pipe.producer_barrier(),
                        transaction_bytes=sa[tma_pipe.producer_stage].nbytes
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
                    offsets=[offset_m + 64, offset_k],
                    mbarrier=tma_pipe.producer_barrier(),
                )
                self.tma.global_to_shared(
                    src=gb,
                    dst=sb[tma_pipe.producer_stage, cta_rank],
                    offsets=[offset_n + cta_rank * half_n, offset_k],
                    mbarrier=tma_pipe.producer_barrier(),
                    multicast_mask=0b11,
                )
                tma_pipe.producer_advance()

        for _ in self.range(self.num_stages):
            tma_pipe.producer_acquire()
            tma_pipe.producer_advance()

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
        half_n = block_n // 2

        num_m_pairs = cdiv(cdiv(m_size, block_m), 2)
        num_n_blocks = cdiv(n_size, block_n)
        self.attrs.blocks = self.num_sms // 2 * 2
        self.attrs.cluster_blocks = 2
        self.attrs.warps = 12  # 2 consumer WGs + 1 producer WG

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[n_size, k_size])
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        # A as one 64-row slab per consumer; B as two 160-row halves (TMA boxes
        # are at most 256 rows); sc as one 64x160 staging slot per consumer.
        sa = self.shared_tensor(dtype=float16, shape=[num_stages, 2, 64, block_k])
        sb = self.shared_tensor(dtype=float16, shape=[num_stages, 2, half_n, block_k])
        sc = self.shared_tensor(dtype=float16, shape=[2, 64, half_n])

        # Each stage is released by both consumers of both CTAs.
        tma_pipe = Pipeline(num_stages, producer_arrive_count=1, consumer_arrive_count=4)
        # The peer must not multicast into this CTA before its barriers exist.
        self.cluster.sync()

        # The producer branch is placed after the consumers: with it first,
        # ptxas serializes the WGMMAs around the setmaxnreg in the divergent path.
        with self.thread_group(thread_begin=0, num_threads=128):
            self.consume(sa, sb, sc, gc, tma_pipe, 0, k_size, num_m_pairs, num_n_blocks)

        with self.thread_group(thread_begin=128, num_threads=128):
            self.consume(sa, sb, sc, gc, tma_pipe, 1, k_size, num_m_pairs, num_n_blocks)

        with self.thread_group(thread_begin=256, num_threads=128):
            self.setmaxnreg(40, "dec")
            with self.thread_group(thread_begin=0, num_threads=32):
                self.produce(ga, gb, sa, sb, tma_pipe, k_size, num_m_pairs, num_n_blocks)

        # Keep both CTAs resident until neither will touch the other's memory.
        self.cluster.sync()


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
        matmul = MatmulWGMMAV6()

        a = torch.randn(m, k, dtype=torch.float16).cuda() / math.sqrt(k)
        b = torch.randn(n, k, dtype=torch.float16).cuda()
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
