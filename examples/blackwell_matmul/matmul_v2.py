# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
class BlackwellMatmulV2(tilus.Script):
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
        # multi-stage shared memory: leading dimension indexes the pipeline stage
        s_a = self.shared_tensor(
            dtype=float16, shape=[self.stages, self.block_m, self.block_k]
        )
        s_b = self.shared_tensor(
            dtype=float16, shape=[self.stages, self.block_n, self.block_k]
        )

        t_acc = self.tcgen05.alloc(dtype=float32, shape=[self.block_m, self.block_n])

        # one TMA barrier per stage
        tma_barriers = self.mbarrier.alloc(counts=[1 for _ in range(self.stages)])
        mma_barrier = self.mbarrier.alloc(counts=1)
        # per-role phase: tracks the expected phase for the next wait
        tma_phase: uint32 = 0
        mma_phase: uint32 = 0

        # prefill: issue TMA loads for the first (stages - 1) tiles without waiting
        for i in range(self.stages - 1):
            offset_k = i * self.block_k
            with self.single_warp():
                with self.single_thread():
                    self.mbarrier.arrive_and_expect_tx(
                        tma_barriers[i], transaction_bytes=s_a[i].nbytes + s_b[i].nbytes
                    )
                self.tma.global_to_shared(
                    src=g_a,
                    dst=s_a[i],
                    offsets=[offset_m, offset_k],
                    mbarrier=tma_barriers[i],
                )
                self.tma.global_to_shared(
                    src=g_b,
                    dst=s_b[i],
                    offsets=[offset_n, offset_k],
                    mbarrier=tma_barriers[i],
                )

        self.sync()

        current_stage: int32 = 0
        preload_stage: int32 = self.stages - 1

        # unroll by stages so the compiler can resolve stage indices to constants
        for offset_k in self.range(0, k_size, self.block_k, unroll=self.stages):
            with self.single_warp():
                # preload: issue TMA for a future tile into the next free stage
                preload_offset_k = offset_k + (self.stages - 1) * self.block_k
                with self.single_thread():
                    self.mbarrier.arrive_and_expect_tx(
                        tma_barriers[preload_stage],
                        transaction_bytes=s_a[preload_stage].nbytes
                        + s_b[preload_stage].nbytes,
                    )
                self.tma.global_to_shared(
                    src=g_a,
                    dst=s_a[preload_stage],
                    offsets=[offset_m, preload_offset_k],
                    mbarrier=tma_barriers[preload_stage],
                )
                self.tma.global_to_shared(
                    src=g_b,
                    dst=s_b[preload_stage],
                    offsets=[offset_n, preload_offset_k],
                    mbarrier=tma_barriers[preload_stage],
                )
                # wait for the current stage's TMA data to arrive
                self.mbarrier.wait(
                    tma_barriers[current_stage],
                    phase=tma_phase,
                    sem="relaxed",
                    scope="cta",
                )
                # compute on the current stage
                self.tcgen05.mma(
                    s_a[current_stage],
                    s_b[current_stage].transpose(),
                    t_acc,
                    enable_input_d=offset_k != 0,
                )
                self.tcgen05.commit(mbarrier=mma_barrier)
                self.mbarrier.wait(
                    mma_barrier, phase=mma_phase, sem="relaxed", scope="cta"
                )

            # advance stage indices (ring buffer); flip phase when wrapping to stage 0
            preload_stage = (preload_stage + 1) % self.stages
            current_stage = (current_stage + 1) % self.stages
            tma_phase ^= current_stage == 0
            mma_phase ^= 1
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
    matmul = BlackwellMatmulV2()

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
