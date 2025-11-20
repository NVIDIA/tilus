# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas
import tilus
import torch
from tilus import float16, float32, int32, uint32
from tilus.ir.tensor import GlobalTensor, RegisterTensor
from tilus.utils import benchmark_func, cdiv


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


class LoadPipeline(Pipeline):
    def __init__(
        self,
        num_stages: int,
        block_m: int,
        block_n: int,
        block_k: int,
    ):
        super().__init__(
            num_stages=num_stages, producer_arrive_count=2, consumer_arrive_count=1
        )
        self.s_a = self.shared_tensor(dtype=float16, shape=[num_stages, block_m, block_k])
        self.s_b = self.shared_tensor(dtype=float16, shape=[num_stages, block_n, block_k])


class LoadWorker(tilus.Class):
    def __init__(self, pipe: LoadPipeline, g_a: GlobalTensor, g_b: GlobalTensor):
        self.pipe: LoadPipeline = pipe
        self.g_a: GlobalTensor = g_a
        self.g_b: GlobalTensor = g_b

    def run(self, k_size: int32, offset_m: int32, offset_n: int32, block_k: int):
        pipe, g_a, g_b = self.pipe, self.g_a, self.g_b
        s_a, s_b = pipe.s_a, pipe.s_b
        num_stages: int = pipe.num_stages

        for offset_k in self.range(0, k_size, block_k, unroll=num_stages):
            self.pipe.producer_acquire()
            with self.single_thread():
                self.tma.global_to_shared(
                    src=g_a,
                    dst=s_a[pipe.producer_stage],
                    offsets=[offset_m, offset_k],
                    mbarrier=pipe.producer_release_barrier(),
                )
                self.tma.global_to_shared(
                    src=g_b,
                    dst=s_b[pipe.producer_stage],
                    offsets=[offset_n, offset_k],
                    mbarrier=pipe.producer_release_barrier(),
                )
            pipe.producer_advance()

        # remaining mma stages to wait for completion
        for _ in self.range(min(num_stages, cdiv(k_size, block_k))):
            pipe.producer_acquire()
            pipe.producer_advance()


class MmaWorker(tilus.Class):
    def __init__(self, pipe: LoadPipeline, block_m: int, block_n: int):
        self.pipe: LoadPipeline = pipe
        self.t_acc = self.tcgen05.alloc(dtype=float32, shape=[block_m, block_n], init=0.0)

    def run(self, k_size: int, block_k: int):
        pipe = self.pipe
        s_a, s_b = pipe.s_a, pipe.s_b
        num_stages: int = pipe.num_stages
        for offset_k in self.range(0, k_size, block_k, unroll=num_stages):
            pipe.consumer_acquire()
            with self.single_thread():
                self.tcgen05.mma(
                    s_a[pipe.consumer_stage],
                    s_b[pipe.consumer_stage].transpose(),
                    self.t_acc,
                )
                self.tcgen05.commit(mbarrier=pipe.consumer_release_barrier())
            pipe.consumer_advance()

    def dealloc(self):
        self.tcgen05.dealloc(self.t_acc)


@tilus.autotune("block_m, block_n", [[128, 64], [128, 128], [128, 256]])
@tilus.autotune("block_k", [16, 32, 64])
@tilus.autotune("stages", [2, 3, 4])
class BlackwellMatmulV4(tilus.Script):
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

        pipe = LoadPipeline(
            num_stages=self.stages,
            block_m=self.block_m,
            block_n=self.block_n,
            block_k=self.block_k,
        )
        load_worker = LoadWorker(pipe, g_a, g_b)
        mma_worker = MmaWorker(pipe, self.block_m, self.block_n)

        with self.thread_group(thread_begin=0, num_threads=32):
            # producer
            load_worker.run(
                k_size=k_size, offset_m=offset_m, offset_n=offset_n, block_k=self.block_k
            )

        with self.thread_group(thread_begin=32, num_threads=32):
            # consumer
            mma_worker.run(k_size=k_size, block_k=self.block_k)

        self.sync()

        # load the result from tensor memory to register
        r_acc = self.tcgen05.load(
            mma_worker.t_acc, offsets=[0, 0], shape=[self.block_m, self.block_n]
        )

        g_c = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        self.store_global(g_c, r_acc.to(float16), offsets=[offset_m, offset_n])

        # all allocated tensor memory must be deallocated
        self.sync()
        mma_worker.dealloc()


def main(bench=True):
    matmul = BlackwellMatmulV4()

    headers = ["m", "n", "k", "name", "latency (ms)", "tflops"]
    rows: list = []

    for m_size, n_size, k_size in [
        # [128, 128, 16 * 6],
        # [40],
        [4096, 4096, 4096],
        [4096, 4096, 14336],
        [8192, 8192, 8192],
        [10240, 10240, 10240],
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
