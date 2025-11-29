# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas
import tilus
import torch
from tilus import float16, float32, int32, uint32
from tilus.ir.tensor import GlobalTensor, RegisterTensor
from tilus.utils import benchmark_func, cdiv
from tilus.extensions.hidet.utils.ncu_utils import ncu_run


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


class LoadPipeline(tilus.Pipeline):
    def __init__(
        self,
        num_stages: int,
        params: Params,
    ):
        super().__init__(
            num_stages=num_stages, producer_arrive_count=2, consumer_arrive_count=1
        )
        self.params: Params = params
        self.s_a = self.shared_tensor(
            dtype=float16, shape=[num_stages, params.block_m, params.block_k]
        )
        self.s_b = self.shared_tensor(
            dtype=float16, shape=[num_stages, params.block_n, params.block_k]
        )


class LoadWorker(tilus.Class):
    def __init__(
        self, pipe: LoadPipeline, params: Params
    ):
        self.pipe: LoadPipeline = pipe
        self.params: Params = params

    def async_run(self):
        pipe, params = self.pipe, self.params
        s_a, s_b = pipe.s_a, pipe.s_b
        num_stages: int = pipe.num_stages
        offset_m = self.blockIdx.x * params.block_m
        offset_n = self.blockIdx.y * params.block_n
        with self.thread_group(thread_begin=0, num_threads=32):
            for offset_k in self.range(0, params.k_size, params.block_k, unroll=num_stages):
                self.pipe.producer_acquire()
                with self.single_thread():
                    self.tma.global_to_shared(
                        src=params.g_a,
                        dst=s_a[pipe.producer_stage],
                        offsets=[offset_m, offset_k],
                        mbarrier=pipe.producer_release_barrier(),
                    )
                    self.tma.global_to_shared(
                        src=params.g_b,
                        dst=s_b[pipe.producer_stage],
                        offsets=[offset_n, offset_k],
                        mbarrier=pipe.producer_release_barrier(),
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


        params = Params(
            m_size=m_size,
            n_size=n_size,
            k_size=k_size,
            block_m=self.block_m,
            block_n=self.block_n,
            block_k=self.block_k,
            g_a=self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size]),
            g_b=self.global_view(b_ptr, dtype=float16, shape=[n_size, k_size]),
            g_c=self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        )

        pipe = LoadPipeline(num_stages=self.stages, params=params)
        load_worker = LoadWorker(pipe, params)
        mma_worker = MmaWorker(pipe, params)

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
    main(bench=True)
    # ncu_run(main, bench=False, kernel_regex="hidet|nvjet")
