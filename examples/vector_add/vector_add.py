# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Elementwise vector addition on the GPU.

Each thread block loads a contiguous tile of ``block_elems`` values from ``a`` and ``b``,
computes ``c = a + b``, and stores the result. This is a minimal Tilus example: one
:class:`tilus.Script`, :meth:`global_view`, :meth:`load_global`, elementwise ``+``, and
:meth:`store_global`.

``n`` must be divisible by ``block_elems`` (enforced in :func:`main`).
"""

import pandas
import tilus
import torch
from tilus import float32, int32
from tilus.utils import benchmark_func, cdiv


class VectorAdd(tilus.Script):
    """``c[i] = a[i] + b[i]`` for ``i in range(n)``."""

    def __init__(self):
        super().__init__()
        self.block_elems = 1024

    def __call__(
        self,
        n: int32,
        a_ptr: ~float32,
        b_ptr: ~float32,
        c_ptr: ~float32,
    ):
        self.attrs.blocks = (cdiv(n, self.block_elems),)
        self.attrs.warps = 4

        offset: int32 = self.block_elems * self.blockIdx.x

        ga = self.global_view(a_ptr, dtype=float32, shape=[n])
        gb = self.global_view(b_ptr, dtype=float32, shape=[n])
        gc = self.global_view(c_ptr, dtype=float32, shape=[n])

        ra = self.load_global(ga, offsets=[offset], shape=[self.block_elems])
        rb = self.load_global(gb, offsets=[offset], shape=[self.block_elems])
        rc = ra + rb
        self.store_global(gc, rc, offsets=[offset])


def _nbytes_fp32_vector_add(n_elts: int) -> int:
    # 3 x fp32: read a, read b, write c
    return n_elts * 4 * 3


def main():
    headers = ["n", "name", "latency (ms)", "GB/s"]
    workloads = [1 << 20, 1 << 24]

    rows = []
    for n in workloads:
        assert n % 1024 == 0, "n must be divisible by block_elems (1024)"

        kernel = VectorAdd()
        a = torch.randn(n, dtype=torch.float32, device="cuda")
        b = torch.randn(n, dtype=torch.float32, device="cuda")
        c_actual = torch.empty(n, dtype=torch.float32, device="cuda")
        c_expect = a + b
        torch.cuda.synchronize()

        kernel(n, a, b, c_actual)
        torch.cuda.synchronize()

        torch.testing.assert_close(c_expect, c_actual)

        for name, func in [
            ("torch", lambda: torch.add(a, b, out=c_actual)),
            ("tilus", lambda: kernel(n, a, b, c_actual)),
        ]:
            latency = benchmark_func(func, warmup=5, repeat=20)
            gbps = _nbytes_fp32_vector_add(n) / (latency * 1e-3) / 1e9
            rows.append([n, name, latency, gbps])

    df = pandas.DataFrame(rows, columns=headers)
    print(df)


if __name__ == "__main__":
    main()
