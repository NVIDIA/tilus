# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas
import tilus
import torch
from tilus import float16, float32, int32, uint32, boolean, Dim3
from tilus.ir.tensor import GlobalTensor, RegisterTensor, TMemoryTensor
from tilus.utils import benchmark_func, cdiv

"""
Pipelines:
- LoadPipeline: load A and B from global memory to shared memory
- MmaPipeline: compute MMA from shared memory to tensor memory

Workers:
- LoadWorker (warp 0): producer of LoadPipeline
- MmaWorker (warp 1): consumer of LoadPipeline, producer of MmaPipeline
- EpilogueWorker (warp 4-7): consumer of MmaPipeline
"""

class Scheduler(tilus.Class):
    def __init__(
        self,
        m_size: int32,
        n_size: int,
        k_size: int,
        block_m: int,
        block_n: int,
        block_k: int,
        initial_block: Dim3
    ):
        self.m_size: int32 = m_size
        self.n_size: int = n_size
        self.k_size: int = k_size
        self.block_m: int = block_m
        self.block_n: int = block_n
        self.block_k: int = block_k
        self.offset_m: int32 = initial_block.x * self.block_m
        self.offset_n: int32 = initial_block.y * self.block_n

        self.response = self.shared_tensor(dtype=int32, shape=[4])
        self.barrier = self.mbarrier.alloc(1)
        self.phase: uint32 = 0
    
    def fetch_block(self) -> boolean:
        self.clc.try_cancel(response=self.response, mbarrier=self.barrier, multicast=False)
        self.mbarrier.wait(barrier=self.barrier, phase=self.phase)
        is_valid, blockIdx = self.clc.query_response(response=self.response)

        if is_valid:
            self.offset_m = blockIdx.x * self.block_m
            self.offset_n = blockIdx.y * self.block_n
            self.phase = self.phase ^ uint32(1)
        return is_valid


class LoadPipeline(tilus.Pipeline):
    def __init__(self, s: Scheduler, num_stages: int):
        super().__init__(num_stages=num_stages, producer_arrive_count=2, consumer_arrive_count=1)
        self.s_a = self.shared_tensor(dtype=float16, shape=[num_stages, s.block_m, s.block_k])
        self.s_b = self.shared_tensor(dtype=float16, shape=[num_stages, s.block_n, s.block_k])


class MmaPipeline(tilus.Pipeline):
    def __init__(self, s: Scheduler, num_stages: int):
        super().__init__(
            num_stages=num_stages, 
            producer_arrive_count=1, 
            consumer_arrive_count=1
        )
        self.t_acc: TMemoryTensor = self.tcgen05.alloc(float32, shape=[num_stages, s.block_m, s.block_n])


class LoadWorker(tilus.Class):
    def __init__(
        self, load_pipe: LoadPipeline, s: Scheduler, g_a: GlobalTensor, g_b: GlobalTensor, 
    ):
        self.load_pipe: LoadPipeline = load_pipe
        self.schduler: Scheduler = s
        self.g_a: GlobalTensor = g_a
        self.g_b: GlobalTensor = g_b

    def async_run(self):
        load_pipe, g_a, g_b, s = self.load_pipe, self.g_a, self.g_b, self.schduler
        s_a, s_b = load_pipe.s_a, load_pipe.s_b
        num_stages: int = load_pipe.num_stages
        with self.thread_group(thread_begin=0, num_threads=32):
            for offset_k in self.range(0, s.k_size, s.block_k, unroll=num_stages):
                self.load_pipe.producer_acquire()
                with self.single_thread():
                    self.tma.global_to_shared(
                        src=g_a,
                        dst=s_a[load_pipe.producer_stage],
                        offsets=[s.offset_m, offset_k],
                        mbarrier=load_pipe.producer_release_barrier(),
                    )
                    self.tma.global_to_shared(
                        src=g_b,
                        dst=s_b[load_pipe.producer_stage],
                        offsets=[s.offset_n, offset_k],
                        mbarrier=load_pipe.producer_release_barrier(),
                    )
                load_pipe.producer_advance()

            # remaining mma stages to wait for completion
            for _ in self.range(min(num_stages, cdiv(s.k_size, s.block_k))):
                load_pipe.producer_acquire()
                load_pipe.producer_advance()


class MmaWorker(tilus.Class):
    def __init__(self, load_pipe: LoadPipeline, mma_pipe: MmaPipeline, scheduler: Scheduler):
        self.load_pipe: LoadPipeline = load_pipe
        self.mma_pipe: MmaPipeline = mma_pipe
        self.scheduler: Scheduler = scheduler

    def async_run(self):
        load_pipe = self.load_pipe
        mma_pipe = self.mma_pipe
        with self.thread_group(thread_begin=32, num_threads=32):
            for offset_k in self.range(
                0, self.scheduler.k_size, self.scheduler.block_k, unroll=load_pipe.num_stages
            ):
                load_pipe.consumer_acquire()
                with self.single_thread():
                    self.tcgen05.mma(
                        load_pipe.s_a[load_pipe.consumer_stage],
                        load_pipe.s_b[load_pipe.consumer_stage].transpose(),
                        mma_pipe.t_acc[mma_pipe.producer_stage]
                    )
                    self.tcgen05.commit(mbarrier=load_pipe.consumer_release_barrier())
                load_pipe.consumer_advance()


class EpilogueWorker(tilus.Class):
    def __init__(self, mma_pipe: MmaPipeline, scheduler: Scheduler):
        super().__init__()

    
    def async_run(self):
        pass

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

        info = Scheduler(
            m_size=m_size,
            n_size=n_size,
            k_size=k_size,
            block_m=self.block_m,
            block_n=self.block_n,
            block_k=self.block_k,
            offset_m=self.block_m * self.blockIdx.x,
            offset_n=self.block_n * self.blockIdx.y,
            num_stages=self.stages,
        )

        load_pipe = LoadPipeline(s=info)
        mma_pipe = MmaPipeline(s=info)

        load_worker = LoadWorker(load_pipe, g_a, g_b, info)
        mma_worker = MmaWorker(load_pipe, mma_pipe, info)

        # producer
        load_worker.async_run()

        # consumer
        mma_worker.async_run()

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
