"""
Matmul with Shared Memory
=========================

On modern GPUs, shared memory is a limited resource that can be used to store data that is frequently accessed by
threads within the same block. This example demonstrates how to implement matrix multiplication using shared memory
to optimize performance.
"""

import tilus
from tilus import float16, float32, int32


class MatmulV1(tilus.Script):
    def __init__(self, num_warps=4, block_m=64, block_n=64, block_k=16):
        super().__init__()
        self.num_warps = num_warps
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
        self.attrs.blocks = [
            self.utils.ceil_div(m_size, self.block_m),
            self.utils.ceil_div(n_size, self.block_n),
        ]
        self.attrs.warps = self.num_warps

        offset_m: int32 = self.block_m * self.blockIdx.x
        offset_n: int32 = self.block_n * self.blockIdx.y

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[k_size, n_size])

        # allocate shared memory for the tiles for A and B
        sa = self.shared_tensor(dtype=float16, shape=[self.block_m, self.block_k])
        sb = self.shared_tensor(dtype=float16, shape=[self.block_k, self.block_n])

        acc = self.register_tensor(dtype=float32, shape=[self.block_m, self.block_n], init=0.0)

        for offset_k in range(0, k_size, self.block_k):
            # load a tile of A matrix from global memory to shared memory
            lda = self.load_global(
                ga,
                offsets=[offset_m, offset_k],
                shape=[self.block_m, self.block_k],
            )

            # store the loaded tile in shared memory
            self.store_shared(sa, lda)

            # load a tile of B matrix from global memory to shared memory
            ldb = self.load_global(
                gb,
                offsets=[offset_k, offset_n],
                shape=[self.block_k, self.block_n],
            )

            # store the loaded tile in shared memory
            self.store_shared(sb, ldb)

            # synchronize threads to ensure all have stored their data in shared memory
            self.sync()

            # load the tiles from shared memory to registers
            a = self.load_shared(sa)
            b = self.load_shared(sb)

            acc = self.dot(a, b, acc)
            self.sync()

        self.free_shared(sa)
        self.free_shared(sb)

        casted_acc = self.cast(acc, dtype=float16)
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        self.store_global(gc, casted_acc, offsets=[offset_m, offset_n])


# %%
# There are several new instructions used in this example:
#
# .. currentmodule:: tilus.Script
#
# - :meth:`shared_tensor`: to create a shared tensor used to store the tiles of A and B matrices.
# - :meth:`load_shared`: to load the tiles from shared memory to registers.
# - :meth:`store_shared`: to store the tiles in shared memory.
# - :meth:`free_shared`: to free the shared memory allocated for the tiles so that the precious shared memory can be
#   reused. Every shared memory allocation must be freed before the end of the kernel.
# - :meth:`sync`: to synchronize all threads in the thread block.
#
# In the main loop, we load tiles of A and B matrices from global memory to shared memory and perform a synchronization.
# After that, we load the tiles from shared memory to registers and perform the dot product. Another synchronization
# is performed to ensure all threads have completed their computations before proceeding to the next iteration. The
# loading and computation steps require two synchronizations since they access the same shared tensors and instructions
# in tilus
#
#

import math

import pandas
import torch
from tilus.utils import benchmark_func


def main():
    headers = ["m", "n", "k", "name", "latency (ms)", "gflops"]
    workloads = [
        [4096, 4096, 4096],
    ]

    rows = []
    for m, n, k in workloads:
        matmul = MatmulV1()

        a = (torch.rand(m, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
        b = (torch.rand(k, n, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
        c_actual = torch.empty(m, n, dtype=torch.float16).cuda()
        c_expect = a @ b
        matmul(m, n, k, a, b, c_actual)

        torch.cuda.synchronize()

        # check correctness
        torch.testing.assert_close(c_expect, c_actual)

        # benchmark
        for name, func in [
            ("torch", lambda: torch.matmul(a, b, out=c_expect)),
            ("tilus", lambda: matmul(m, n, k, a, b, c_actual)),
        ]:
            latency = benchmark_func(func, warmup=5, repeat=20)
            flops = 2 * m * n * k / latency * 1e-9
            rows.append([m, n, k, name, latency, flops])

    df = pandas.DataFrame(rows, columns=headers)
    print(df)


# %%

if __name__ == "__main__":
    main()
