# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pandas
import tilus
import torch
from hidet.ir.expr import Expr
from tilus import Dim3, float16, float32, int32, uint32
from tilus.ir.tensor import GlobalTensor, TMemoryTensor
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


class Scheduler(tilus.Class):
    def __init__(self):
        self.barrier = self.mbarrier.alloc(count=1)
        self.phase: uint32 = self.mbarrier.consumer_initial_phase
        self.s_response = self.shared_tensor(dtype=int32, shape=[4])

    # def fetch_next(self) -> tuple[Expr, Dim3]:
    #     return boolean.false, self.blockIdx

    def fetch_next(self):
        self.clc.try_cancel(self.s_response, self.barrier, multicast=False)

    def query_next(self) -> tuple[Expr, Dim3]:
        """Utility function can be used by consumers. Need to be executed within consumer's thread group."""
        # wait for the response
        self.mbarrier.wait(self.barrier, phase=self.phase)
        self.phase = self.phase ^ uint32(1)

        # decode the response
        is_valid, blockIdx = self.clc.query_response(self.s_response)
        return is_valid, blockIdx


class LoadPipeline(tilus.Pipeline):
    def __init__(self, num_stages: int, params: Params):
        super().__init__(
            num_stages=num_stages,
            producer_arrive_count=2,  # two tma loads
            consumer_arrive_count=1,  # one commit in MmaWorker
        )
        self.s_a = self.shared_tensor(
            dtype=float16, shape=[num_stages, params.block_m, params.block_k]
        )
        self.s_b = self.shared_tensor(
            dtype=float16, shape=[num_stages, params.block_n, params.block_k]
        )


class MmaPipeline(tilus.Pipeline):
    def __init__(self, num_stages: int, params: Params):
        super().__init__(
            num_stages=num_stages,
            producer_arrive_count=1,  # one commit in MmaWorker
            consumer_arrive_count=128,  # epilogue has 128 threads
        )
        self.t_acc: TMemoryTensor = self.tcgen05.alloc(
            float32, shape=[num_stages, params.block_m, params.block_n], init=0.0
        )

    def finalize(self):
        self.sync()
        self.tcgen05.dealloc(self.t_acc)


class LoadWorker(tilus.Class):
    def __init__(self, params: Params, load_pipe: LoadPipeline, scheduler: Scheduler):
        self.params: Params = params
        self.load_pipe: LoadPipeline = load_pipe
        self.scheduler: Scheduler = scheduler

    def async_run(self):
        params, load_pipe, scheduler = self.params, self.load_pipe, self.scheduler
        s_a, s_b = load_pipe.s_a, load_pipe.s_b
        num_stages: int = load_pipe.num_stages
        offset_m: int32 = self.blockIdx.x * params.block_m
        offset_n: int32 = self.blockIdx.y * params.block_n
        with self.thread_group(thread_begin=32, num_threads=32):
            while True:
                for offset_k in self.range(
                    0, params.k_size, params.block_k, unroll=num_stages
                ):
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

                scheduler.fetch_next()
                is_valid, blockIdx = scheduler.query_next()
                if is_valid:
                    offset_m = blockIdx.x * params.block_m
                    offset_n = blockIdx.y * params.block_n
                else:
                    break


class MmaWorker(tilus.Class):
    def __init__(
        self,
        params: Params,
        load_pipe: LoadPipeline,
        mma_pipe: MmaPipeline,
        scheduler: Scheduler,
    ):
        self.params: Params = params
        self.load_pipe: LoadPipeline = load_pipe
        self.mma_pipe: MmaPipeline = mma_pipe
        self.scheduler: Scheduler = scheduler

    def async_run(self):
        params, load_pipe, mma_pipe, scheduler = (
            self.params,
            self.load_pipe,
            self.mma_pipe,
            self.scheduler,
        )
        with self.thread_group(thread_begin=64, num_threads=32):
            while True:
                mma_pipe.producer_acquire()
                for _ in self.range(
                    0, params.k_size, params.block_k, unroll=load_pipe.num_stages
                ):
                    load_pipe.consumer_acquire()
                    with self.single_thread():
                        self.tcgen05.mma(
                            load_pipe.s_a[load_pipe.consumer_stage],
                            load_pipe.s_b[load_pipe.consumer_stage].transpose(),
                            mma_pipe.t_acc[mma_pipe.producer_stage],
                        )
                        self.tcgen05.commit(mbarrier=load_pipe.consumer_release_barrier())
                    load_pipe.consumer_advance()
                with self.single_thread():
                    self.tcgen05.commit(mbarrier=mma_pipe.producer_release_barrier())
                mma_pipe.producer_advance()

                # check if there is a new block to process
                is_valid, blockIdx = scheduler.query_next()
                if not is_valid:
                    break


class EpilogueWorker(tilus.Class):
    def __init__(self, params: Params, mma_pipe: MmaPipeline, scheduler: Scheduler):
        super().__init__()
        self.params: Params = params
        self.mma_pipe: MmaPipeline = mma_pipe
        self.scheduler: Scheduler = scheduler

    def async_run_stg(self):
        params, mma_pipe, block_pipe = self.params, self.mma_pipe, self.scheduler
        with self.thread_group(thread_begin=128, num_threads=128):
            offset_m: int32 = self.blockIdx.x * params.block_m
            offset_n: int32 = self.blockIdx.y * params.block_n
            while True:
                mma_pipe.consumer_acquire()
                t_acc = mma_pipe.t_acc[mma_pipe.consumer_stage]

                # tmem to smem
                r_acc = self.tcgen05.load(t_acc)
                self.tcgen05.wait_load()
                self.store_global(
                    params.g_c, r_acc.to(float16), offsets=[offset_m, offset_n]
                )

                # reset tmem to 0.0 for next accumulation
                self.tcgen05.store(
                    t_acc,
                    src=self.register_tensor(
                        dtype=float32, shape=[params.block_m, params.block_n], init=0.0
                    ),
                )
                self.tcgen05.wait_store()
                self.sync()

                self.mbarrier.arrive(mma_pipe.consumer_release_barrier())
                mma_pipe.consumer_advance()

                is_valid, blockIdx = block_pipe.query_next()
                if is_valid:
                    offset_m = blockIdx.x * params.block_m
                    offset_n = blockIdx.y * params.block_n
                else:
                    break

    def async_run_tma(self):
        params, mma_pipe, block_pipe = self.params, self.mma_pipe, self.scheduler
        s_acc = self.shared_tensor(dtype=float16, shape=[params.block_m, params.block_n])
        with self.thread_group(thread_begin=128, num_threads=128):
            offset_m: int32 = self.blockIdx.x * params.block_m
            offset_n: int32 = self.blockIdx.y * params.block_n
            while True:
                mma_pipe.consumer_acquire()
                t_acc = mma_pipe.t_acc[mma_pipe.consumer_stage]

                # tmem to smem
                r_acc = self.tcgen05.load(t_acc)
                self.tcgen05.wait_load()
                self.store_shared(s_acc, r_acc.to(float16))
                self.sync()

                # smem to gmem
                self.tma.fence_proxy_copy_async()
                with self.single_thread():
                    self.tma.shared_to_global(
                        src=s_acc,
                        dst=params.g_c,
                        offsets=[offset_m, offset_n],
                    )
                    self.tma.commit_group()
                    self.tma.wait_group(n=0)
                self.sync()

                # reset tmem to 0.0 for next accumulation
                self.tcgen05.store(
                    t_acc,
                    src=self.register_tensor(
                        dtype=float32, shape=[params.block_m, params.block_n], init=0.0
                    ),
                )
                self.tcgen05.wait_store()
                self.sync()

                self.mbarrier.arrive(mma_pipe.consumer_release_barrier())
                mma_pipe.consumer_advance()

                is_valid, blockIdx = block_pipe.query_next()
                if is_valid:
                    offset_m = blockIdx.x * params.block_m
                    offset_n = blockIdx.y * params.block_n
                else:
                    break


@tilus.autotune("block_m, block_n", [[128, 64], [128, 128], [128, 256]])
@tilus.autotune("block_k", [16, 32, 64])
@tilus.autotune("load_stages", [2, 3, 4])
class BlackwellMatmulV5(tilus.Script):
    def __init__(self, block_m: int, block_n: int, block_k: int, load_stages: int):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.load_stages = load_stages

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
        self.attrs.warps = 8

        params = Params(
            m_size=m_size,
            n_size=n_size,
            k_size=k_size,
            block_m=self.block_m,
            block_n=self.block_n,
            block_k=self.block_k,
            g_a=self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size]),
            g_b=self.global_view(b_ptr, dtype=float16, shape=[n_size, k_size]),
            g_c=self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size]),
        )

        scheduler = Scheduler()
        load_pipe = LoadPipeline(num_stages=self.load_stages, params=params)
        mma_pipe = MmaPipeline(num_stages=1, params=params)

        load_worker = LoadWorker(params, load_pipe=load_pipe, scheduler=scheduler)
        mma_worker = MmaWorker(params, load_pipe, mma_pipe, scheduler)
        epilogue_worker = EpilogueWorker(params, mma_pipe, scheduler)

        load_worker.async_run()
        mma_worker.async_run()
        epilogue_worker.async_run_stg()

        self.sync()
        mma_pipe.finalize()


def main(bench=True):
    matmul = BlackwellMatmulV5()
    torch.manual_seed(0)

    headers = ["m", "n", "k", "name", "latency (ms)", "tflops"]
    rows: list = []

    for m_size, n_size, k_size in [
        # [128, 128, 4096],
        # [256, 256, 4096],
        # [1024, 1024, 4096],
        [4096, 4096, 4096],
        [4096, 4096, 14336],
        [8192, 8192, 8192],
        [10240, 10240, 10240],
    ]:
        print(f"Running with m_size={m_size}, n_size={n_size}, k_size={k_size}")
        a = torch.randn(m_size, k_size, dtype=torch.float16, device="cuda")
        b = torch.randn(n_size, k_size, dtype=torch.float16, device="cuda")
        c_actual = torch.empty(m_size, n_size, dtype=torch.float16, device="cuda")
        c_expected = torch.empty(m_size, n_size, dtype=torch.float16, device="cuda")

        matmul(m_size, n_size, k_size, a, b, c_actual)
        torch.matmul(a, b.T, out=c_expected)
        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

        # benchmark
        if bench:
            for name, func in [
                ("torch", lambda: torch.matmul(a, b.T, out=c_expected)),
                ("tilus", lambda: matmul(m_size, n_size, k_size, a, b, c_actual)),
            ]:
                latency = benchmark_func(func, warmup=5, repeat=20)
                tflops = 2 * m_size * n_size * k_size / latency * 1e-9
                rows.append([m_size, n_size, k_size, name, latency, tflops])

    if bench:
        df = pandas.DataFrame(rows, columns=headers)
        print(df)


if __name__ == "__main__":
    # tilus.utils.clear_cache()
    main(bench=True)
    # ncu_run(main, bench=False, kernel_regex="hidet|nvjet")
