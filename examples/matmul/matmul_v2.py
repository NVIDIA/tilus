# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Auto-tuning
===========

In previous versions of the matmul kernel, we manually set the hyperparameters such as `block_m`, `block_n`, and
`block_k`. However, these hyperparameters can significantly affect the performance of the kernel, and finding the
optimal values for them can be a tedious and time-consuming process.

In tilus, we can use the :meth:`tilus.autotune` decorator to annotate the search space of the hyperparameters and let tilus
automatically search for the best configuration:

.. code-block:: python

   @tilus.autotune("arg_name1", [v11, v12, v12])
   @tilus.autotune("arg_name2, arg_name3", [(v21, v31), (v22, v32)])
   class AwesomeKernel(tilus.Script):
       def __init__(self, user_arg, arg_name1, arg_name2, arg_name3):
           super().__init__()
           ... # use the hyperparameters to perform compilation-time preparations

The above example defines a space contains 5 configurations for (`arg_name1`, `arg_name2`, `arg_name3`):

- (v11, v21, v31)
- (v11, v22, v32)
- (v12, v21, v31)
- (v12, v22, v32)
- (v13, v21, v31)
- (v13, v22, v32)

When we instantiate the `AwesomeKernel` class, we only need to provide the `user_arg` and the rest of the parameters
will be automatically tuned by tilus:

.. code-block:: python

    kernel = AwesomeKernel(user_arg)

Tilus will use all configurations to instantiate the kernel, run each of them, and
automatically select the best configuration based on the performance of the kernel for the given input data.
"""

import tilus
from tilus import float16, float32, int32
from tilus.utils import cdiv

# %%
# Annotate Schedule Space
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# Resuing the same kernel implementation as in the previous example, we can use the :meth:`tilus.autotune` decorator to
# annotate the search space of the hyperparameters like `num_warps`, `block_m`, `block_n`, and `block_k`, as shown below:


@tilus.autotune("num_warps", [4, 8])
@tilus.autotune("block_m, block_n", [(128, 128), (128, 64), (64, 128)])
@tilus.autotune("block_k", [16, 32])
class MatmulV2(tilus.Script):
    def __init__(
        self,
        num_warps,
        block_m,
        block_n,
        block_k,
    ):
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
            cdiv(m_size, self.block_m),
            cdiv(n_size, self.block_n),
        ]
        self.attrs.warps = self.num_warps

        offset_m: int32 = self.block_m * self.blockIdx.x
        offset_n: int32 = self.block_n * self.blockIdx.y

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[k_size, n_size])
        sa = self.shared_tensor(dtype=float16, shape=[self.block_m, self.block_k])
        sb = self.shared_tensor(dtype=float16, shape=[self.block_k, self.block_n])
        acc = self.register_tensor(
            dtype=float32, shape=[self.block_m, self.block_n], init=0.0
        )

        for offset_k in range(0, k_size, self.block_k):
            lda = self.load_global(
                ga,
                offsets=[offset_m, offset_k],
                shape=[self.block_m, self.block_k],
            )
            self.store_shared(sa, lda)
            ldb = self.load_global(
                gb,
                offsets=[offset_k, offset_n],
                shape=[self.block_k, self.block_n],
            )
            self.store_shared(sb, ldb)
            self.sync()

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
# Launch the Kernel
# ~~~~~~~~~~~~~~~~~

import math

import pandas
import torch
from tilus.utils import benchmark_func


def main():
    headers = ["m", "n", "k", "name", "latency (ms)", "tflops"]
    workloads = [
        [4096, 4096, 4096],
    ]

    rows = []
    for m, n, k in workloads:
        matmul = MatmulV2()

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
            tflops = 2 * m * n * k / latency * 1e-9
            rows.append([m, n, k, name, latency, tflops])

    df = pandas.DataFrame(rows, columns=headers)
    print(df)


# %%
# In the main function, we define the kernel by instantiating the `MatmulV2` class. There is no compilation in this
# step. We invoke the kernel by calling it like ``matmul(m, n, k, a, b, c_actual)``, the kernel will trigger the
# autotuning process, which will compile the kernel with all the configurations defined in the :meth:`tilus.autotune`
# decorator, run the kernel of each configuration on the given arguments, and automatically select the best
# configuration. The autotuning happens only once, and the compiled kernel will be cached for future invocations.

if __name__ == "__main__":
    main()
