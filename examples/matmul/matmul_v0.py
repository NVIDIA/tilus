"""
Naive Matmul
************

This tutorial demonstrates a simple implementation of matrix multiplication using Tilus. We hope to use this example
to illustrate the basic concepts of writing a kernel in Tilus, including kernel definition, data types, tensors, and
usage of tilus kernel.

"""

# %%
# Tilus Script
# ============
# In Tilus, we define a kernel by subclassing the :class:`tilus.Script` class. There are two methods
# that we need to implement: ``__init__`` and ``__call__``.
#
# - The ``__init__`` method is used to initialize the compilation-time known hyperparameters of the script.
# - The ``__call__`` method is the main entry point of the script, where we define the computation logic of the kernel.
import tilus


class TilusScriptKernel(tilus.Script):
    def __init__(
        self,
        # compilation-time known hyperparameters
    ):
        super().__init__()
        ...  # process the hyperparameters

    def __call__(
        self,
        # kernel parameters
    ): ...  # define the computation logic of the kernel


# %%
# Naive Matmul Implementation
# ===========================
# In this section, we implement a naive matrix multiplication kernel using Tilus.
# This naive implementation is not optimized for performance, but it serves as a good starting point to understand the
# basic concepts of Tilus.
from tilus import float16, float32, int32
from tilus.utils import cdiv


class MatmulV0(tilus.Script):
    def __init__(self):
        super().__init__()
        # we define three hyperparameters: ``block_m``, ``block_n``, and ``block_k`` to determine the tile size on
        # m, n, and k dimensions for each `thread block` of the kernel.
        self.block_m = 16
        self.block_n = 16
        self.block_k = 16

    def __call__(
        self,
        m_size: int32,  # the size of the m dimension of the input matrix A and output matrix C
        n_size: int,  # the size of the n dimension of the input matrix B and output matrix C
        k_size: int,  # the size of the k dimension of the input matrix A and B
        a_ptr: ~float16,  # the pointer to the input matrix A, which is a 2D tensor of shape [m_size, k_size]
        b_ptr: ~float16,  # the pointer to the input matrix B, which is a 2D tensor of shape [k_size, n_size]
        c_ptr: ~float16,  # the pointer to the output matrix C, which is a 2D tensor of shape [m_size, n_size]
    ):
        self.attrs.blocks = [
            cdiv(m_size, self.block_m),  # the x dimension size of the grid
            cdiv(n_size, self.block_n),  # the y dimension size of the grid
        ]
        self.attrs.warps = (
            1  # the number of warps per thread block, must be a compile-time known integer
        )

        # define two int32 variables to store the offsets of the m and n dimensions for the current thread block.
        offset_m: int32 = self.block_m * self.blockIdx.x
        offset_n: int32 = self.block_n * self.blockIdx.y

        # create two global tensors `ga` and `gb` to represent the input matrices A and B, respectively.
        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[k_size, n_size])

        # create a register tensor `acc` to accumulate the results of the matrix multiplication.
        acc = self.register_tensor(dtype=float32, shape=[self.block_m, self.block_n], init=0.0)

        # iterate over the k dimension in blocks of size `block_k`.
        for k in range(cdiv(k_size, self.block_k)):
            # calculate the offset for the current block in the k dimension
            offset_k = k * self.block_k

            # load a block of matrix A and B into register tensors `a` and `b`.
            a = self.load_global(
                ga, offsets=[offset_m, offset_k], shape=[self.block_m, self.block_k]
            )
            b = self.load_global(
                gb, offsets=[offset_k, offset_n], shape=[self.block_k, self.block_n]
            )

            # perform the dot product: acc = a @ b + acc
            self.dot(a, b, acc, out=acc)

        # after the loop, we cast the accumulated result `acc` to float16 type and store it back to the output matrix C.
        acc_f16 = self.cast(acc, dtype=float16)
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        self.store_global(gc, acc_f16, offsets=[offset_m, offset_n])


# %%
# We used three type annotations for the kernel parameters:
#
# - ``int32``: a runtime-known 32-bit integer, which is used for m_size.
# - ``int``: a compile-time known integer, which is used for n_size and k_size. Different values of n_size and k_size
#   will trigger Just-In-Time (JIT) compilation of the kernel.
# - ``~float16``: a pointer to a float16 array, which is used for a_ptr, b_ptr, and c_ptr. It is equivalent to
#   `float16*` in C/C++.
#
# .. currentmodule:: tilus.Script
#
# We also used the following tilus instructions in the kernel:
#
# - :meth:`global_view`: to create a global tensor view of the input matrices A and B.
# - :meth:`register_tensor`: to create a register tensor to accumulate the results of the matrix multiplication.
# - :meth:`load_global`: to load a block of matrix A and B into register tensors.
# - :meth:`dot`: to perform the dot product of two register tensors.
# - :meth:`cast`: to cast the accumulated result to float16 type.
# - :meth:`store_global`: to store the accumulated result back to the output matrix C.
#
# All of these instructions are defined with **block** semantics, which means that they are executed by all threads
# in a thread block.

# %%
# Launching the Kernel
# ====================
# To launch the kernel, we need to create an instance of the `MatmulV0` class and call it with the appropriate
# parameters. The following code demonstrates how to do this:
import math

import pandas
import torch
from tilus.utils import benchmark_func


def main():
    headers = ["m", "n", "k", "name", "latency (ms)", "gflops"]
    workloads = [[4096, 4096, 4096]]

    rows = []
    for m, n, k in workloads:
        # create an instance of the kernel we have just defined
        matmul = MatmulV0()

        a = (torch.rand(m, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
        b = (torch.rand(k, n, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
        c_actual = torch.empty(m, n, dtype=torch.float16).cuda()
        c_expect = a @ b
        torch.cuda.synchronize()

        # launch the kernel by passing required arguments
        matmul(m, n, k, a, b, c_actual)
        torch.cuda.synchronize()

        # check correctness
        torch.testing.assert_close(c_expect, c_actual, atol=1e-2, rtol=1e-2)

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
# The above code creates an instance of the `MatmulV0` class and calls it with input sizes ``m``, ``n``, ``k``, and
# torch tensors `a`, `b`, and `c_actual`. The kernel is launched with the specified parameters,
# and the results are checked for correctness.

if __name__ == "__main__":
    main()

# %%
# The output of the above code will be a pandas DataFrame that contains the performance results of the naive matrix
# multiplication kernel. This naive kernel serves as a good starting point to understand the basic concepts of Tilus,
# but it is not optimized yet. In the subsequent versions of the matmul kernel, we will introduce various optimizations
# that can significantly improve the performance of the matrix multiplication and finally achieve the performance
# similar to the vendor libraries like cuBLAS.
