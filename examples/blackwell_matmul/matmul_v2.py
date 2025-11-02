# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import pandas
import tilus
import torch
from tilus import float16, float32, int32, uint32
from tilus.utils import benchmark_func, cdiv

if not tilus.target.get_current_target().supports(tilus.target.nvgpu_sm100a):
    # skip this example if the current target does not support nvgpu_sm100a
    exit(0)

tilus.option.cache_dir(os.path.join(os.path.dirname(__file__), "cache"))
tilus.option.debug.dump_ir()

# tilus.target.set_current_target(tilus.target.nvgpu_sm100a)


@tilus.autotune("block_m, block_n", [[128, 64], [128, 128], [128, 256]])
@tilus.autotune("block_k", [16, 32, 64])
@tilus.autotune("stages", [2, 3, 4])
class BlackwellMatmul(tilus.Script):
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
        s_a = self.shared_tensor(dtype=float16, shape=[self.stages, self.block_m, self.block_k])
        s_b = self.shared_tensor(dtype=float16, shape=[self.stages, self.block_n, self.block_k])

        # allocate a tensor in tensor memory (tmem)
        t_acc = self.tcgen05.alloc(
            dtype=float32, shape=[self.block_m, self.block_n], init=0.0
        )

        # allocate barriers
        tma_barriers = self.mbarrier.alloc(count=[1 for _ in range(self.stages)])
        mma_barrier = self.mbarrier.alloc(count=1)
        tma_phases = self.register_tensor(dtype=uint32, shape=[self.stages], init=0)
        mma_phase: uint32 = 0

        for i in range(self.stages - 1):
            with self.single_thread():
                self.tma.global_to_shared(
                    src=g_a,
                    dst=s_a[i],
                    offsets=[offset_m, i * self.block_k],
                    mbarrier=tma_barriers[i],
                )
                self.tma.global_to_shared(
                    src=g_b,
                    dst=s_b[i],
                    offsets=[offset_n, i * self.block_k],
                    mbarrier=tma_barriers[i],
                )
                self.mbarrier.arrive(tma_barriers[i])

        self.sync()

        current_stage: int32 = 0
        preload_stage: int32 = self.stages - 1

        for offset_k in range(0, k_size, self.block_k):
            with self.single_thread():  # we use a single thread to issue the TMA copy
                # preload 
                self.tma.global_to_shared(
                    src=g_a,
                    dst=s_a[preload_stage],
                    offsets=[offset_m, offset_k],
                    mbarrier=tma_barriers[preload_stage],
                )
                self.tma.global_to_shared(
                    src=g_b,
                    dst=s_b[preload_stage],
                    offsets=[offset_n, offset_k],
                    mbarrier=tma_barriers[preload_stage],
                )
                self.mbarrier.arrive(tma_barriers[preload_stage])
                self.mbarrier.wait(tma_barriers[current_stage], phase=tma_phases[current_stage].item())

                self.tcgen05.mma(s_a[current_stage], s_b[current_stage].transpose(), t_acc)
                self.tcgen05.commit(mbarrier=mma_barrier)
                self.mbarrier.wait(mma_barrier, phase=mma_phase)

            tma_phases[current_stage] ^= 1
            mma_phase ^= 1
            preload_stage = (preload_stage + 1) % self.stages
            current_stage = (current_stage + 1) % self.stages
            self.sync()

        # load the result from tensor memory to register
        r_acc = self.tcgen05.load(
            t_acc, offsets=[0, 0], shape=[self.block_m, self.block_n]
        )

        g_c = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        self.store_global(g_c, r_acc.to(float16), offsets=[offset_m, offset_n])

        # all allocated tensor memory must be deallocated
        self.sync()
        self.tcgen05.dealloc(t_acc)


def main(bench=True):
    matmul = BlackwellMatmul()

    headers = ["m", "n", "k", "name", "latency (ms)", "tflops"]
    rows = []

    for m_size, n_size, k_size in [
        [4096, 4096, 4096],
        [4096, 4096, 14336],
    ]:
        print(f"Running with m_size={m_size}, n_size={n_size}, k_size={k_size}")
        a = torch.randn(m_size, k_size, dtype=torch.float16, device="cuda")
        b = torch.randn(n_size, k_size, dtype=torch.float16, device="cuda")
        c = torch.empty(m_size, n_size, dtype=torch.float16, device="cuda")

        matmul(m_size, n_size, k_size, a, b, c)
        torch.cuda.synchronize()

        c_ref = a @ b.T

        torch.testing.assert_close(c, c_ref, atol=1e-2, rtol=1e-2)

        # benchmark
        if bench:
            for name, func in [
                ("torch", lambda: a @ b.T),
                ("tilus", lambda: matmul(m_size, n_size, k_size, a, b, c)),
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
