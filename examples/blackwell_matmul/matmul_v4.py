# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import pandas
import tilus
import torch
from tilus import float16, float32, int32, uint32
from tilus.ir.tensor import RegisterTensor
from tilus.utils import benchmark_func, cdiv

if not tilus.target.get_current_target().supports(tilus.target.nvgpu_sm100a):
    # skip this example if the current target does not support nvgpu_sm100a
    exit(0)

tilus.option.cache_dir(os.path.join(os.path.dirname(__file__), "cache"))


class PipelineState(tilus.State):
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


@tilus.autotune("block_m, block_n", [[128, 64], [128, 128], [128, 256]])
@tilus.autotune("block_k", [16, 32, 64])
@tilus.autotune("stages", [2, 3, 4])
class BlackwellMatmul(tilus.Script):
    debug_schedule = dict(
        block_m=128,
        block_n=128,
        block_k=16,
        stages=3,
    )

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

        # allocate a tensor in tensor memory (tmem)
        t_acc = self.tcgen05.alloc(
            dtype=float32, shape=[self.block_m, self.block_n], init=0.0
        )

        state = PipelineState(
            num_stages=self.stages,
            producer_arrive_count=2,
            consumer_arrive_count=1,
        )

        with self.thread_group(group_index=0, group_size=32):
            # producer
            for offset_k in self.range(0, k_size, self.block_k, unroll=self.stages):
                # self.printf("[%d][%d] producer acquring offset_k=%d\n", self.blockIdx.x, state.producer_stage, offset_k)
                state.producer_acquire()
                # self.printf("[%d][%d] producer acquired offset_k=%d\n", self.blockIdx.x, state.producer_stage, offset_k)
                with self.single_thread():
                    self.tma.global_to_shared(
                        src=g_a,
                        dst=s_a[state.producer_stage],
                        offsets=[offset_m, offset_k],
                        mbarrier=state.producer_release_barrier(),
                    )
                    self.tma.global_to_shared(
                        src=g_b,
                        dst=s_b[state.producer_stage],
                        offsets=[offset_n, offset_k],
                        mbarrier=state.producer_release_barrier(),
                    )
                # self.printf("[%d][%d] producer produced offset_k=%d\n", self.blockIdx.x, state.producer_stage, offset_k)
                state.producer_advance()

            # remaining mma stages to wait for completion
            for _ in self.range(min(self.stages, cdiv(k_size, self.block_k))):
                state.producer_acquire()
                state.producer_advance()

        with self.thread_group(group_index=1, group_size=32):
            for offset_k in self.range(0, k_size, self.block_k, unroll=self.stages):
                # self.printf("[%d][%d] consumer acquring offset_k=%d\n", self.blockIdx.x, state.consumer_stage, offset_k)
                state.consumer_acquire()
                # self.printf("[%d][%d] consumer acquired offset_k=%d\n", self.blockIdx.x, state.consumer_stage, offset_k)
                with self.single_thread():
                    self.tcgen05.mma(
                        s_a[state.consumer_stage],
                        s_b[state.consumer_stage].transpose(),
                        t_acc,
                    )
                    self.tcgen05.commit(mbarrier=state.consumer_release_barrier())
                # self.printf("[%d][%d] consumer consumed offset_k=%d\n", self.blockIdx.x, state.consumer_stage, offset_k)
                state.consumer_advance()

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
    rows: list = []

    for m_size, n_size, k_size in [
        [128, 128, 16 * 6],
        # [40],
        # [4096, 4096, 4096],
        # [4096, 4096, 14336],
        # [8192, 8192, 8192],
        # [10240, 10240, 10240],
    ]:
        print(f"Running with m_size={m_size}, n_size={n_size}, k_size={k_size}")
        a = torch.randn(m_size, k_size, dtype=torch.float16, device="cuda")
        b = torch.randn(n_size, k_size, dtype=torch.float16, device="cuda")
        c = torch.empty(m_size, n_size, dtype=torch.float16, device="cuda")

        print(matmul.jit_instance_for(m_size, n_size, k_size, a, b, c))
        jit_instance = matmul.jit_instance_for(m_size, n_size, k_size, a, b, c)
        print(jit_instance.programs()[0])
        exit(0)

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
