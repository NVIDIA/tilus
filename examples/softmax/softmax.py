# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused softmax kernel.

Computes row-wise softmax: y[i, j] = exp(x[i,j] - max_j x[i,j])
                                      / sum_j exp(x[i,j] - max_j x[i,j])

The kernel fuses the entire softmax into a single launch, avoiding
redundant global memory round-trips that a naive PyTorch implementation
would incur.  Each thread block processes `block_m` rows, iterating
over the column dimension in tiles of `block_n`.

Three passes over the row data:
  1. Compute row-wise max (for numerical stability).
  2. Compute exp(x - max) and accumulate the row-wise sum.
  3. Divide by the sum and store the result.
"""

import pandas
import tilus
import torch
from tilus import float16, float32, int32
from tilus.utils import benchmark_func, cdiv


@tilus.autotune("block_m", [1, 4, 8])
@tilus.autotune("block_n", [128, 256, 512, 1024])
@tilus.autotune("warps", [4, 8])
class FusedSoftmax(tilus.Script):
    """Row-wise fused softmax kernel."""

    def __init__(self, block_m: int, block_n: int, warps: int):
        super().__init__()
        self.block_m: int = block_m
        self.block_n: int = block_n
        self.warps: int = warps

    def __call__(
        self,
        m_size: int,
        n_size: int32,
        x_ptr: ~float16,
        y_ptr: ~float16,
    ):
        self.attrs.blocks = (cdiv(m_size, self.block_m),)
        self.attrs.warps = self.warps

        offset_m = self.blockIdx.x * self.block_m

        g_x = self.global_view(
            x_ptr, dtype=float16, shape=[m_size, n_size]
        )
        g_y = self.global_view(
            y_ptr, dtype=float16, shape=[m_size, n_size]
        )

        # --- Pass 1: row-wise max for numerical stability ---
        r_max = self.register_tensor(
            dtype=float32,
            shape=[self.block_m, self.block_n],
            init=float("-inf"),
        )
        for offset_n in range(0, n_size, self.block_n):
            r_x = self.load_global(
                g_x,
                offsets=[offset_m, offset_n],
                shape=[self.block_m, self.block_n],
            ).to(float32)
            r_max = self.maximum(r_max, r_x)

        r_row_max = self.max(
            r_max, dim=1, keepdim=True
        )  # [block_m, 1]

        # --- Pass 2: exp(x - max) and row-wise sum ---
        r_sum = self.register_tensor(
            dtype=float32,
            shape=[self.block_m, self.block_n],
            init=0.0,
        )
        for offset_n in range(0, n_size, self.block_n):
            r_x = self.load_global(
                g_x,
                offsets=[offset_m, offset_n],
                shape=[self.block_m, self.block_n],
            ).to(float32)
            r_exp = self.exp(r_x - r_row_max)
            r_sum = r_sum + r_exp

        r_row_sum = self.sum(
            r_sum, dim=1, keepdim=True
        )  # [block_m, 1]

        # --- Pass 3: normalize and store ---
        for offset_n in range(0, n_size, self.block_n):
            r_x = self.load_global(
                g_x,
                offsets=[offset_m, offset_n],
                shape=[self.block_m, self.block_n],
            ).to(float32)
            r_exp = self.exp(r_x - r_row_max)
            r_y = r_exp / r_row_sum
            self.store_global(
                g_y,
                r_y.to(float16),
                offsets=[offset_m, offset_n],
            )


def main():
    headers = [
        "m_size",
        "n_size",
        "dtype",
        "torch (ms)",
        "tilus (ms)",
    ]
    rows = []

    for m_size, n_size in [
        (1823, 781),
        (4096, 1024),
        (4096, 4096),
        (4096, 8192),
    ]:
        softmax_kernel = FusedSoftmax()

        x = (
            torch.rand(
                m_size, n_size, dtype=torch.float16, device="cuda"
            )
            - 0.5
        ) * 2.0
        y_actual = torch.empty_like(x)

        softmax_kernel(m_size, n_size, x, y_actual)
        y_expected = torch.softmax(x.float(), dim=1).half()

        torch.testing.assert_close(
            y_actual, y_expected, atol=1e-2, rtol=1e-2
        )

        rows.append(
            [
                m_size,
                n_size,
                "float16",
                benchmark_func(
                    lambda: torch.softmax(x, dim=1)
                ),
                benchmark_func(
                    lambda: softmax_kernel(
                        m_size, n_size, x, y_actual
                    )
                ),
            ]
        )
        print(
            f"Softmax matches reference for "
            f"size ({m_size}, {n_size})"
        )

    df = pandas.DataFrame(rows, columns=headers)
    print(df)


if __name__ == "__main__":
    main()
