# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pandas
import tilus
import torch
from tilus import float16, float32, int32
from tilus.utils import benchmark_func, cdiv


@tilus.autotune("block_m", [1, 8])
@tilus.autotune("block_n", [128, 256, 512, 1024])
@tilus.autotune("warps", [2, 4, 8])
class LayerNorm(tilus.Script):
    """Forward-only layer normalization tilus kernel.

    This implements the per-row LayerNorm used in many transformer blocks. It
    computes: y = (x - mean) / sqrt(var + eps) * gamma + beta

    Only the forward is provided; gamma and beta are optional (can be None).
    """

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
        gamma_ptr: ~float16,
        beta_ptr: ~float16,
        y_ptr: ~float16,
        eps: float,
    ):
        self.attrs.blocks = (cdiv(m_size, self.block_m),)
        self.attrs.warps = self.warps

        offset_m = self.blockIdx.x * self.block_m

        g_x = self.global_view(x_ptr, dtype=float16, shape=[m_size, n_size])
        g_y = self.global_view(y_ptr, dtype=float16, shape=[m_size, n_size])
        g_gamma = self.global_view(gamma_ptr, dtype=float16, shape=[n_size])
        g_beta = self.global_view(beta_ptr, dtype=float16, shape=[n_size])

        # Register accumulators for mean and variance (computed in float32)
        r_sum = self.register_tensor(
            dtype=float32, shape=[self.block_m, self.block_n], init=0.0
        )
        r_square = self.register_tensor(
            dtype=float32, shape=[self.block_m, self.block_n], init=0.0
        )

        # first pass: compute mean and variance
        for offset_n in range(0, n_size, self.block_n):
            r_x = self.load_global(
                g_x, offsets=[offset_m, offset_n], shape=[self.block_m, self.block_n]
            ).to(float32)  # [block_m, block_n]
            r_sum = r_sum + r_x
            r_square = r_square + self.square(r_x)

        # finalize mean and variance
        r_mean = self.sum(r_sum, dim=1, keepdim=True) / n_size  # [block_m, 1]
        r_var = (
            self.sum(r_square, dim=1, keepdim=True) / n_size - r_mean * r_mean
        )  # [block_m, 1], var = E[x^2] - (E[x])^2
        r_rstd = self.rsqrt(r_var + eps)

        # second pass: y = (x - mean) * rstd * gamma + beta
        for offset_n in range(0, n_size, self.block_n):
            r_x = self.load_global(
                g_x, offsets=[offset_m, offset_n], shape=[self.block_m, self.block_n]
            ).to(float32)  # [block_m, block_n]
            r_gamma = self.load_global(
                g_gamma, offsets=[offset_n], shape=[self.block_n]
            ).to(float32)  # [block_n]
            r_beta = self.load_global(
                g_beta, offsets=[offset_n], shape=[self.block_n]
            ).to(float32)  # [block_n]
            r_x_hat = (r_x - r_mean) * r_rstd
            r_y = r_x_hat * r_gamma + r_beta
            self.store_global(g_y, r_y.to(float16), offsets=[offset_m, offset_n])


def main():
    headers = ["m_size", "n_size", "dtype", "torch", "tilus"]
    rows = []
    for i in [1, 2, 4, 8]:
        m_size = n_size = 1024 * i

        tilus_layer_norm = LayerNorm()

        x = (torch.rand(m_size, n_size, dtype=torch.float16).cuda() - 0.5) * 2.0
        gamma = torch.rand(n_size, dtype=torch.float16).cuda()
        beta = torch.rand(n_size, dtype=torch.float16).cuda()
        y_actual = torch.empty_like(x)

        tilus_layer_norm(m_size, n_size, x, gamma, beta, y_actual, 1e-5)
        y_expected = torch.nn.functional.layer_norm(
            x, normalized_shape=[n_size], weight=gamma, bias=beta, eps=1e-5
        )

        torch.testing.assert_close(y_actual, y_expected, atol=1e-2, rtol=1e-2)

        rows.append(
            [
                m_size,
                n_size,
                "float16",
                benchmark_func(
                    lambda: torch.nn.functional.layer_norm(
                        x, normalized_shape=[n_size], weight=gamma, bias=beta, eps=1e-5
                    )
                ),
                benchmark_func(
                    lambda: tilus_layer_norm(
                        m_size, n_size, x, gamma, beta, y_actual, 1e-5
                    )
                ),
            ]
        )
        print(f"LayerNorm forward matches reference for size ({m_size}, {n_size})")

    df = pandas.DataFrame(rows, columns=headers)
    print(df)


if __name__ == "__main__":
    main()
