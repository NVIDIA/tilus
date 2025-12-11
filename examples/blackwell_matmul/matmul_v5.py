# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas
import tilus
import torch
from tilus import float16, float32, int32, uint32
from tilus.ir.tensor import GlobalTensor, RegisterTensor
from tilus.utils import benchmark_func, cdiv
from tilus.extensions.hidet.utils.ncu_utils import ncu_run
from hidet.ir.primitives.cuda.vars import threadIdx

tilus.option.cache_dir("./cache")
# tilus.option.debug.dump_ir()


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


class LoadPipeline(Pipeline):
    def __init__(
        self,
        num_stages: int,
        params: Params,
    ):
        super().__init__(
            num_stages=num_stages, producer_arrive_count=4, consumer_arrive_count=1
        )
        self.params: Params = params
        self.s_a = self.shared_tensor(
            dtype=float16, shape=[num_stages, params.block_m, params.block_k]
        )
        self.s_b = self.shared_tensor(
            dtype=float16, shape=[num_stages, params.block_n, params.block_k]
        )


class LoadWorker(tilus.Class):
    def __init__(self, pipe: LoadPipeline, params: Params):
        self.pipe: LoadPipeline = pipe
        self.params: Params = params

    def async_run(self):
        """
             |  B0  |  B1  |  B2  |  B3  |
        ---- +-------------+-------------+
         A0  |             |             |
        ---- |     cta0    |     cta2    |
         A1  |             |             |
        ---- +-------------+-------------+
         A2  |             |             |
        ---- |     cta1    |     cta3    |
         A3  |             |             |
        ---- +-------------+-------------+

        cta0: load A0 and B0
        cta1: load A2 and B1
        cta2: load A1 and B2
        cta3: load A3 and B3
        """
        pipe, params = self.pipe, self.params
        num_stages, block_m, block_n, block_k = (
            pipe.num_stages,
            params.block_m,
            params.block_n,
            params.block_k,
        )
        s_a = self.reshape_shared(pipe.s_a, [num_stages, 2, block_m // 2, block_k])
        s_b = self.reshape_shared(pipe.s_b, [num_stages, 2, block_n // 2, block_k])
        offset_m = self.blockIdx.x * params.block_m + self.cluster.blockIdx.y * (block_m // 2)
        offset_n = self.blockIdx.y * params.block_n + self.cluster.blockIdx.x * (block_n // 2)
        # self.printf("[%d, %d][Loader]\n", self.blockIdx.x, self.blockIdx.y)
        with self.thread_group(thread_begin=0, num_threads=32):
            for offset_k in self.range(
                0, params.k_size, params.block_k, unroll=num_stages
            ):
                # if self.blockIdx.x == 0 and self.blockIdx.y == 0 and offset_k // params.block_k % 2 == 0:
                #     self.printf("[%d, %d][Loader] offset_k: %d\n", self.blockIdx.x, self.blockIdx.y, offset_k)
                # self.sync()
                self.pipe.producer_acquire()
                self.tma.global_to_shared(
                    src=params.g_a,
                    dst=s_a[pipe.producer_stage, self.cluster.blockIdx.y],
                    offsets=[offset_m, offset_k],
                    mbarrier=pipe.producer_release_barrier(),
                    multicast_mask=0b0101 if self.cluster.blockIdx.x == 0 else 0b1010,
                )
                self.tma.global_to_shared(
                    src=params.g_b,
                    dst=s_b[pipe.producer_stage, self.cluster.blockIdx.x],
                    offsets=[offset_n, offset_k],
                    mbarrier=pipe.producer_release_barrier(),
                    multicast_mask=0b0011 if self.cluster.blockIdx.y == 0 else 0b1100,
                )
                pipe.producer_advance()


class MmaWorker(tilus.Class):
    def __init__(self, pipe: LoadPipeline, params: Params):
        self.pipe: LoadPipeline = pipe
        self.params: Params = params
        self.t_acc = self.tcgen05.alloc(
            dtype=float32, shape=[params.block_m, params.block_n], init=0.0
        )
        self.flush_barrier = self.mbarrier.alloc(1)

    def async_run(self):
        pipe = self.pipe
        s_a, s_b = pipe.s_a, pipe.s_b
        num_stages: int = pipe.num_stages
        with self.thread_group(thread_begin=32, num_threads=32):
            for offset_k in self.range(
                0, self.params.k_size, self.params.block_k, unroll=num_stages
            ):
                # self.printf("[%d, %d][MMA] offset_k: %d\n", self.blockIdx.x, self.blockIdx.y, offset_k)
                pipe.consumer_acquire()
                with self.single_thread():
                    self.tcgen05.mma(
                        s_a[pipe.consumer_stage],
                        s_b[pipe.consumer_stage].transpose(),
                        self.t_acc,
                    )
                    self.tcgen05.commit(mbarrier=pipe.consumer_release_barrier())
                pipe.consumer_advance()

            with self.single_thread():
                self.tcgen05.commit(mbarrier=self.flush_barrier)
            self.mbarrier.wait(self.flush_barrier, phase=0)


@tilus.autotune("block_m, block_n", [[128, 64], [128, 128], [128, 256]])
@tilus.autotune("block_k", [16, 32, 64])
@tilus.autotune("stages", [2, 3, 4])
class BlackwellMatmulV5(tilus.Script):
    debug_schedule = dict(
        block_m=128,
        block_n=64,
        block_k=16,
        stages=2,
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
        self.attrs.blocks = [
            cdiv(m_size, self.block_m * 2) * 2,
            cdiv(n_size, self.block_n * 2) * 2,
        ]
        self.attrs.cluster_blocks = (2, 2)
        self.attrs.warps = 4

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
        self.cluster.sync()

        pipe = LoadPipeline(num_stages=self.stages, params=params)
        load_worker = LoadWorker(pipe, params)
        mma_worker = MmaWorker(pipe, params)

        # make sure all 
        self.sync()
        self.cluster.sync()

        # producer
        load_worker.async_run()

        # consumer
        mma_worker.async_run()

        self.sync()

        # store the result back to global memory
        offset_m: int32 = self.block_m * self.blockIdx.x
        offset_n: int32 = self.block_n * self.blockIdx.y
        r_acc = self.tcgen05.load(mma_worker.t_acc)
        self.tcgen05.wait_load()
        self.store_global(params.g_c, r_acc.to(float16), offsets=[offset_m, offset_n])

        # all allocated tensor memory must be deallocated
        self.sync()
        self.tcgen05.dealloc(mma_worker.t_acc)


def main(bench=True):
    matmul = BlackwellMatmulV5()

    headers = ["m", "n", "k", "name", "latency (ms)", "tflops"]
    rows: list = []

    for m_size, n_size, k_size in [
        # [128, 128, 16 * 6],
        # [40],
        # [256 * 10, 128, 256],
        [256, 128, 1024],
        # [4096, 4096, 4096],
        # [4096, 4096, 14336],
        # [8192, 8192, 8192],
        # [10240, 10240, 10240],
    ]:
        print(f"Running with m_size={m_size}, n_size={n_size}, k_size={k_size}")
        a = torch.randn(m_size, k_size, dtype=torch.float16, device="cuda")
        b = torch.randn(n_size, k_size, dtype=torch.float16, device="cuda")
        c_actual = torch.empty(m_size, n_size, dtype=torch.float16, device="cuda")
        c_expected = torch.empty(m_size, n_size, dtype=torch.float16, device="cuda")

        matmul(m_size, n_size, k_size, a, b, c_actual)
        torch.cuda.synchronize()

        torch.matmul(a, b.T, out=c_expected)
        torch.cuda.synchronize()

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
    # main(bench=False)
    main(bench=True)
    # ncu_run(main, bench=False, kernel_regex="hidet|nvjet")
