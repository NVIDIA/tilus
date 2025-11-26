# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas
import tilus
import torch
from hidet.ir.expr import Expr
from tilus import float16, float32, int32, uint32, boolean, Dim3
from tilus.ir.tensor import GlobalTensor, RegisterTensor, TMemoryTensor
from tilus.utils import benchmark_func, cdiv

"""
Pipelines:
- BlockPipeline: thread blocks to work on
- LoadPipeline: load A and B from global memory to shared memory
- MmaPipeline: compute MMA from shared memory to tensor memory

Workers:
- BlockScheduler (warp 0): producer of BlockPipeline
- LoadWorker (warp 1): producer of LoadPipeline, consumer of BlockPipeline
- MmaWorker (warp 2): consumer of LoadPipeline, producer of MmaPipeline, consumer of BlockPipeline
- EpilogueWorker (warp 4-7): consumer of MmaPipeline, consumer of BlockPipeline
"""

# class Scheduler(tilus.Class):
#     def __init__(
#         self,
#         m_size: int32,
#         n_size: int,
#         k_size: int,
#         block_m: int,
#         block_n: int,
#         block_k: int,
#         initial_block: Dim3
#     ):
#         self.m_size: int32 = m_size
#         self.n_size: int = n_size
#         self.k_size: int = k_size
#         self.block_m: int = block_m
#         self.block_n: int = block_n
#         self.block_k: int = block_k
#         self.offset_m: int32 = initial_block.x * self.block_m
#         self.offset_n: int32 = initial_block.y * self.block_n

#         self.response = self.shared_tensor(dtype=int32, shape=[4])
#         self.barrier = self.mbarrier.alloc(1)
#         self.phase: uint32 = 0
    
#     def fetch_block(self) -> boolean:
#         self.clc.try_cancel(response=self.response, mbarrier=self.barrier, multicast=False)
#         self.mbarrier.wait(barrier=self.barrier, phase=self.phase)
#         is_valid, blockIdx = self.clc.query_response(response=self.response)

#         if is_valid:
#             self.offset_m = blockIdx.x * self.block_m
#             self.offset_n = blockIdx.y * self.block_n
#             self.phase = self.phase ^ uint32(1)
#         return is_valid


class Params(tilus.Class):
    def __init__(
        self,
        m_size: int32,
        n_size: int,
        k_size: int,
        block_m: int,
        block_n: int,
        block_k: int,
        g_a: GlobalTensor,
        g_b: GlobalTensor,
        g_c: GlobalTensor,
    ):
        self.m_size: int32 = m_size
        self.n_size: int = n_size
        self.k_size: int = k_size
        self.block_m: int = block_m
        self.block_n: int = block_n
        self.block_k: int = block_k
        self.g_a: GlobalTensor = g_a
        self.g_b: GlobalTensor = g_b
        self.g_c: GlobalTensor = g_c


class BlockPipeline(tilus.Pipeline):
    def __init__(self):
        super().__init__(
            num_stages=1, 
            producer_arrive_count=1, # producer: block_scheduler
            consumer_arrive_count=32 * 7  # consumer: block_scheduler, load_worker, mma_worker, epilogue_worker
        )
        self.s_blocks = self.shared_tensor(dtype=int32, shape=[1, 4])

    def fetch_next(self) -> tuple[Expr, Dim3]:
        """ Utility function can be used by consumers. Need to be executed within consumer's thread group. """
        self.consumer_acquire()
        is_valid, blockIdx = self.clc.query_response(self.s_blocks[self.consumer_stage])
        self.mbarrier.arrive(barrier=self.consumer_release_barrier())
        self.consumer_advance()
        return is_valid, blockIdx

class LoadPipeline(tilus.Pipeline):
    def __init__(self, num_stages: int, params: Params):
        super().__init__(num_stages=num_stages, producer_arrive_count=2, consumer_arrive_count=1)
        self.s_a = self.shared_tensor(dtype=float16, shape=[num_stages, params.block_m, params.block_k])
        self.s_b = self.shared_tensor(dtype=float16, shape=[num_stages, params.block_n, params.block_k])
    


class MmaPipeline(tilus.Pipeline):
    def __init__(self, params: Params):
        super().__init__(
            num_stages=2, 
            producer_arrive_count=1, 
            consumer_arrive_count=1
        )
        self.t_acc: TMemoryTensor = self.tcgen05.alloc(float32, shape=[2, params.block_m, params.block_n], init=0.0)


class BlockScheduler(tilus.Class):
    def __init__(self, block_pipe: BlockPipeline):
        super().__init__()
        self.block_pipe: BlockPipeline = block_pipe
    
    def async_run(self):
        with self.thread_group(thread_begin=0, num_threads=32):
            pipe = self.block_pipe

            while True:
                # try to cancel a pending thread block in hardware block scheduler
                self.block_pipe.producer_acquire()
                self.clc.try_cancel(
                    response=pipe.s_blocks[pipe.producer_phase],
                    mbarrier=pipe.producer_release_barrier(),
                    multicast=False,
                )
                self.block_pipe.producer_advance()

                # wait for the response
                is_valid, blockIdx = self.block_pipe.fetch_next()
                if not is_valid:
                    break


class LoadWorker(tilus.Class):
    def __init__(
        self, params: Params, load_pipe: LoadPipeline, block_pipe: BlockPipeline
    ):
        self.params: Params = params
        self.load_pipe: LoadPipeline = load_pipe
        self.block_pipe: BlockPipeline = block_pipe

    def async_run(self):
        params, load_pipe, block_pipe = self.params, self.load_pipe, self.block_pipe
        s_a, s_b = load_pipe.s_a, load_pipe.s_b
        num_stages: int = load_pipe.num_stages
        offset_m: int32 = self.blockIdx.x * params.block_m
        offset_n: int32 = self.blockIdx.y * params.block_n
        with self.thread_group(thread_begin=32, num_threads=32):
            while True:
                for offset_k in self.range(0, params.k_size, params.block_k, unroll=num_stages):
                    load_pipe.producer_acquire()
                    with self.single_thread():
                        self.tma.global_to_shared(
                            src=params.g_a,
                            dst=s_a[load_pipe.producer_stage],
                            offsets=[offset_m, offset_k],
                            mbarrier=load_pipe.producer_release_barrier(),
                        )
                        self.tma.global_to_shared(
                            src=params.g_b,
                            dst=s_b[load_pipe.producer_stage],
                            offsets=[offset_n, offset_k],
                            mbarrier=load_pipe.producer_release_barrier(),
                        )
                    load_pipe.producer_advance()
                
                is_valid, blockIdx = block_pipe.fetch_next()
                if is_valid:
                    offset_m = blockIdx.x * params.block_m
                    offset_n = blockIdx.y * params.block_n
                else:
                    break


class MmaWorker(tilus.Class):
    def __init__(self, params: Params, load_pipe: LoadPipeline, mma_pipe: MmaPipeline, block_pipe: BlockPipeline):
        self.params: Params = params
        self.load_pipe: LoadPipeline = load_pipe
        self.mma_pipe: MmaPipeline = mma_pipe
        self.block_pipe: BlockPipeline = block_pipe

    def async_run(self):
        params, load_pipe, mma_pipe, block_pipe = self.params, self.load_pipe, self.mma_pipe, self.block_pipe
        with self.thread_group(thread_begin=32, num_threads=32):
            while True:
                mma_pipe.producer_acquire()
                for _ in self.range(0, params.k_size, params.block_k, unroll=load_pipe.num_stages):
                    load_pipe.consumer_acquire()
                    with self.single_thread():
                        self.tcgen05.mma(
                            load_pipe.s_a[load_pipe.consumer_stage],
                            load_pipe.s_b[load_pipe.consumer_stage].transpose(),
                            mma_pipe.t_acc[mma_pipe.producer_stage]
                        )
                        self.tcgen05.commit(mbarrier=load_pipe.consumer_release_barrier())
                    load_pipe.consumer_advance()
                self.tcgen05.commit(mbarrier=mma_pipe.producer_release_barrier())
                mma_pipe.producer_advance()

                # check if there is a new block to process
                is_valid, blockIdx = block_pipe.fetch_next()
                if not is_valid:
                    break


class EpilogueWorker(tilus.Class):
    def __init__(self, mma_pipe: MmaPipeline):
        super().__init__()
        self.mma_pipe: MmaPipeline = mma_pipe

    def async_run(self):
        mma_pipe = self.mma_pipe
        with self.thread_group(thread_begin=128, num_threads=128):
            while True:
                mma_pipe.consumer_acquire()
                with self.single_thread():
                    r_acc = self.tcgen05.load(
                        mma_pipe.t_acc[mma_pipe.consumer_stage], offsets=[0, 0], shape=mma_pipe.t_acc.shape[1:]
                    )
                    # Here we can do epilogue operations such as store to global memory, activation, etc.
                    # For simplicity, we skip these operations in this example.
                mma_pipe.consumer_advance()


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

        scheduler = Scheduler(
            m_size=m_size,
            n_size=n_size,
            k_size=k_size,
            block_m=self.block_m,
            block_n=self.block_n,
            block_k=self.block_k,
            initial_block=self.blockIdx
        )
        load_pipe = LoadPipeline(scheduler, num_stages=self.stages)
        mma_pipe = MmaPipeline(scheduler, num_stages=2)
        load_worker = LoadWorker(scheduler, load_pipe, g_a, g_b)
        mma_worker = MmaWorker(scheduler, load_pipe, mma_pipe)

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
