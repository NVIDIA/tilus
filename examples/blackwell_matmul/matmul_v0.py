# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import pandas
import tilus
import torch
from tilus import float16, float32, int32, uint32
from tilus.utils import benchmark_func, cdiv

tilus.option.cache_dir("./cache")


@tilus.autotune("block_m, block_n", [[128, 64], [128, 128], [128, 256]])
@tilus.autotune("block_k", [16, 32, 64])
class BlackwellMatmulV0(tilus.Script):
    def __init__(self, block_m: int, block_n: int, block_k: int):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k

    def __call__(
        self,
        m_size: int32,
        n_size: int,
        k_size: int,
        a_ptr: ~float16,
        b_ptr: ~float16,
        c_ptr: ~float16,
    ):
        # set the number of blocks and warps for the kernel launch
        self.attrs.blocks = [cdiv(m_size, self.block_m), cdiv(n_size, self.block_n)]
        self.attrs.warps = 4

        # compute the tile offset from the block index
        offset_m: int32 = self.block_m * self.blockIdx.x
        offset_n: int32 = self.block_n * self.blockIdx.y

        # create global tensor views from raw pointers
        g_a = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        g_b = self.global_view(b_ptr, dtype=float16, shape=[n_size, k_size])

        # allocate shared memory tiles for A and B
        s_a = self.shared_tensor(dtype=float16, shape=[self.block_m, self.block_k])
        s_b = self.shared_tensor(dtype=float16, shape=[self.block_n, self.block_k])

        # allocate a tensor in tensor memory (tmem) as the MMA accumulator
        t_acc = self.tcgen05.alloc(dtype=float32, shape=[self.block_m, self.block_n])

        # allocate one mbarrier to track MMA completion
        mbarriers = self.mbarrier.alloc(counts=[1])

        # mbarrier phase flips between 0 and 1 after each wait
        phase: uint32 = 0

        # synchronize all threads before entering the main loop
        self.sync()

        for offset_k in range(0, k_size, self.block_k):
            # async copy tiles from global to shared memory (legacy, non-TMA)
            self.copy_async(src=g_a, dst=s_a, offsets=[offset_m, offset_k])
            self.copy_async(src=g_b, dst=s_b, offsets=[offset_n, offset_k])
            self.copy_async_wait_all()
            self.sync()

            # tcgen05 instructions are warp-cooperative (issued by a single warp)
            with self.single_warp():
                # D = A @ B (first iter) or D = A @ B + D (subsequent iters)
                self.tcgen05.mma(
                    s_a, s_b.transpose(), t_acc, enable_input_d=offset_k != 0
                )
                # make the mbarrier track completion of prior async tcgen05 ops
                self.tcgen05.commit(mbarrier=mbarriers[0])
                # wait until the MMA writes to tmem are complete
                self.mbarrier.wait(mbarriers[0], phase=phase)
            self.sync()

            phase ^= 1

        # load the result from tensor memory to registers
        r_acc = self.tcgen05.load(t_acc)

        # cast to float16 and store to global memory
        g_c = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        self.store_global(g_c, r_acc.to(float16), offsets=[offset_m, offset_n])

        # all allocated tensor memory must be deallocated before kernel exits
        self.sync()
        self.tcgen05.dealloc(t_acc)


def main(bench=True):
    matmul = BlackwellMatmulV0()

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
