# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import pandas
import tilus
import torch
from tilus import float16, float32, int32, uint32
from tilus.utils import benchmark_func, cdiv

tilus.option.cache_dir("./cache")


@tilus.autotune(
    "block_m, block_n, e_block_n", [[128, 64, 16], [128, 128, 16], [128, 256, 16]]
)
@tilus.autotune("block_k", [16, 32, 64])
@tilus.autotune("stages", [2, 3, 4])
class BlackwellMatmulV3(tilus.Script):
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
        self.attrs.blocks = [cdiv(m_size, self.block_m), cdiv(n_size, self.block_n)]
        self.attrs.warps = 4

        offset_m: int32 = self.block_m * self.blockIdx.x
        offset_n: int32 = self.block_n * self.blockIdx.y

        g_a = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        g_b = self.global_view(b_ptr, dtype=float16, shape=[n_size, k_size])
        s_a = self.shared_tensor(
            dtype=float16, shape=[self.stages, self.block_m, self.block_k]
        )
        s_b = self.shared_tensor(
            dtype=float16, shape=[self.stages, self.block_n, self.block_k]
        )

        t_acc = self.tcgen05.alloc(dtype=float32, shape=[self.block_m, self.block_n])

        # full_barriers: signaled when TMA has filled the stage (data ready)
        full_barriers = self.mbarrier.alloc(counts=[1] * self.stages)
        # empty_barriers: signaled when MMA has consumed the stage (slot free)
        empty_barriers = self.mbarrier.alloc(counts=[1] * self.stages)

        # TMA warp (producer): loads tiles from global to shared memory
        with self.thread_group(thread_begin=0, num_threads=32):
            stage: int32 = 0
            # phase=1: mbarrier starts at phase 0, so waiting for phase 1
            # passes immediately (slot is empty, ready to fill)
            producer_phase: uint32 = 1
            for offset_k in self.range(0, k_size, self.block_k, unroll=self.stages):
                # wait for the MMA warp to free this stage
                self.mbarrier.wait(
                    empty_barriers[stage],
                    phase=producer_phase,
                    sem="relaxed",
                    scope="cta",
                )
                with self.single_thread():
                    self.mbarrier.arrive_and_expect_tx(
                        full_barriers[stage],
                        transaction_bytes=s_a[stage].nbytes + s_b[stage].nbytes,
                    )
                self.tma.global_to_shared(
                    src=g_a,
                    dst=s_a[stage],
                    offsets=[offset_m, offset_k],
                    mbarrier=full_barriers[stage],
                )
                self.tma.global_to_shared(
                    src=g_b,
                    dst=s_b[stage],
                    offsets=[offset_n, offset_k],
                    mbarrier=full_barriers[stage],
                )
                # advance stage; flip phase when wrapping to stage 0
                stage = (stage + 1) % self.stages
                producer_phase ^= stage == 0

        # MMA warp (consumer): computes on tiles loaded by the TMA warp
        with self.thread_group(thread_begin=32, num_threads=32):
            # phase=0: mbarrier starts at phase 0, so waiting for phase 0
            # blocks until the producer signals (slot is not yet filled)
            consumer_phase: uint32 = 0
            stage: int32 = 0
            for offset_k in self.range(0, k_size, self.block_k, unroll=self.stages):
                # wait for the TMA warp to fill this stage
                self.mbarrier.wait(
                    full_barriers[stage],
                    phase=consumer_phase,
                    sem="relaxed",
                    scope="cta",
                )
                self.tcgen05.mma(
                    s_a[stage],
                    s_b[stage].transpose(),
                    t_acc,
                    enable_input_d=offset_k != 0,
                )
                # commit signals empty_barriers: marks this stage as consumed
                self.tcgen05.commit(mbarrier=empty_barriers[stage])
                # advance stage; flip phase when wrapping to stage 0
                stage = (stage + 1) % self.stages
                consumer_phase ^= stage == 0

            # drain: wait for all in-flight MMA to finish
            flush_barrier = self.mbarrier.alloc(1)
            self.tcgen05.commit(mbarrier=flush_barrier)
            self.mbarrier.wait(flush_barrier, phase=0)

        self.sync()

        # TMA epilogue: tmem -> register -> shared -> global (via TMA)
        g_c = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        s_c = self.shared_tensor(dtype=float16, shape=[self.block_m, self.e_block_n])
        for e_offset_n in range(0, self.block_n, self.e_block_n):
            t_acc_slice = self.tcgen05.slice(
                t_acc,
                offsets=[0, e_offset_n],
                shape=[self.block_m, self.e_block_n],
                dims=[0, 1],
            )
            r_acc = self.tcgen05.load(t_acc_slice)
            self.tcgen05.wait_load()
            self.store_shared(s_c, r_acc.to(float16))
            self.fence.proxy_async(space="shared")
            self.sync()
            with self.single_warp():
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
    matmul = BlackwellMatmulV3()

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
