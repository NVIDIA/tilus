# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas
import tilus
import torch
from tilus import float16, float32, int32, uint32
from tilus.utils import benchmark_func, cdiv

tilus.option.cache_dir("./cache")


@tilus.autotune("block_m, block_n", [[128, 64], [128, 128], [128, 256]])
@tilus.autotune("block_k", [16, 32, 64])
@tilus.autotune("stages", [2, 3, 4])
class BlackwellMatmulV3(tilus.Script):
    def __init__(self, block_m: int, block_n: int, block_k: int, stages: int):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.stages = stages

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

        # full_barriers: signaled when data is ready (TMA done)
        full_barriers = self.mbarrier.alloc(counts=[1] * self.stages)
        # empty_barriers: signaled when slot is free (MMA done)
        empty_barriers = self.mbarrier.alloc(counts=[1] * self.stages)

        # TMA warp (producer): loads tiles from global to shared memory
        with self.thread_group(thread_begin=0, num_threads=32):
            stage: int32 = 0
            # init=1: all stages start empty (ready to be filled)
            empty_phases = self.register_tensor(dtype=uint32, shape=[self.stages], init=1)
            for offset_k in self.range(0, k_size, self.block_k, unroll=self.stages):
                # wait for the MMA warp to free this stage
                self.mbarrier.wait(empty_barriers[stage], phase=empty_phases[stage])
                empty_phases[stage] ^= 1
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
                stage = (stage + 1) % self.stages

        # MMA warp (consumer): computes on tiles loaded by the TMA warp
        with self.thread_group(thread_begin=32, num_threads=32):
            # init=0: no stages are full yet (waiting for TMA)
            full_phases = self.register_tensor(dtype=uint32, shape=[self.stages], init=0)
            stage: int32 = 0
            for offset_k in self.range(0, k_size, self.block_k, unroll=self.stages):
                # wait for the TMA warp to fill this stage
                self.mbarrier.wait(full_barriers[stage], phase=full_phases[stage])
                full_phases[stage] ^= 1
                self.tcgen05.mma(
                    s_a[stage],
                    s_b[stage].transpose(),
                    t_acc,
                    enable_input_d=offset_k != 0,
                )
                # commit signals empty_barriers: frees this stage for TMA reuse
                self.tcgen05.commit(mbarrier=empty_barriers[stage])
                stage = (stage + 1) % self.stages

            # drain: wait for all in-flight MMA to finish
            flush_barrier = self.mbarrier.alloc(1)
            self.tcgen05.commit(mbarrier=flush_barrier)
            self.mbarrier.wait(flush_barrier, phase=0)

        self.sync()

        r_acc = self.tcgen05.load(t_acc)

        g_c = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        self.store_global(g_c, r_acc.to(float16), offsets=[offset_m, offset_n])

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

    if bench:
        df = pandas.DataFrame(rows, columns=headers)
        print(df)


if __name__ == "__main__":
    main(bench=True)
    # tilus.utils.ncu_run(main, bench=False, kernel_regex="tilus|nvjet")
