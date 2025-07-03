"""
Use Async Copy
==============

On NVIDIA Ampere and newer architectures, NVIDIA introduced hardware support for asynchronous copy from global memory
to shared memory without using register as an intermediate buffer. In tilus, we introduced block-level instructions
to support this feature, allowing us to implement a more efficient matrix multiplication kernel.

.. currentmodule:: tilus.Script

.. autosummary::

    copy_async
    copy_async_wait_all


.. currentmodule:: tilus.Script

The :meth:`copy_async` instruction copies a tile from global memory to shared memory asynchronously. This instruction
is asynchronous, meaning that the copy operation will not block the execution of the kernel. We can use :meth:`copy_async_wait_all`
to wait for all asynchronous copy operations to complete. The completion of this wait instruction guarantees that
the data is available in shared memory for subsequent computations. However, this instruction itself does not synchronize
the threads in the block, so we still need to use the :meth:`sync` instruction to make sure :meth:`copy_async_wait_all` is completed before
we proceed with the computation.

We used both instructions in the following matmul kernel implementation. It is similar to the previous version but
uses asynchronous copy to load the tiles of matrices A and B into shared memory. It does not use registers as an
intermediate buffer.
"""

import math

import pandas
import tilus
import torch
from tilus import float16, float32, int32
from tilus.utils import benchmark_func


@tilus.autotune("num_warps", [4, 8])
@tilus.autotune("block_m, block_n", [(128, 128), (128, 64), (64, 128)])
@tilus.autotune("block_k", [16, 32])
class MatmulV3(tilus.Script):
    def __init__(
        self,
        num_warps,
        block_m,
        block_n,
        block_k,
    ):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps

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

        block_m, block_n, block_k = self.block_m, self.block_n, self.block_k
        offset_m: int32 = block_m * self.blockIdx.x
        offset_n: int32 = block_n * self.blockIdx.y

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[k_size, n_size])
        sa = self.shared_tensor(dtype=float16, shape=[block_m, block_k])
        sb = self.shared_tensor(dtype=float16, shape=[block_k, block_n])
        acc = self.register_tensor(dtype=float32, shape=[block_m, block_n], init=0.0)

        for offset_k in range(0, k_size, block_k):
            # issue asynchronous copy instructions to load tiles of A and B
            self.copy_async(src=ga, dst=sa, offsets=[offset_m, offset_k])
            self.copy_async(src=gb, dst=sb, offsets=[offset_k, offset_n])

            # wait for all asynchronous copy operations to complete
            self.copy_async_wait_all()

            # synchronize threads in the block to ensure data is available in shared memory
            self.sync()

            a = self.load_shared(sa)
            b = self.load_shared(sb)
            self.dot(a, b, acc, out=acc)
            self.sync()

        self.free_shared(sa)
        self.free_shared(sb)

        casted_acc = self.cast(acc, dtype=float16)
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        self.store_global(gc, casted_acc, offsets=[offset_m, offset_n])


def main():
    headers = ["m", "n", "k", "name", "latency (ms)", "gflops"]
    workloads = [
        [4096, 4096, 4096],
    ]

    rows = []
    for m, n, k in workloads:
        matmul = MatmulV3()

        a = (torch.rand(m, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
        b = (torch.rand(k, n, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
        c_actual = torch.empty(m, n, dtype=torch.float16).cuda()
        c_expect = a @ b
        matmul(m, n, k, a, b, c_actual)

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
