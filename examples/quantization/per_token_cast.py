# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-token FP8 cast with scale factors.

This is a Tilus translation of DeepSeek TileKernels'
``per_token_cast_kernel.py`` for the common FP16 -> FP8 e4m3 path.  Each CTA
processes one token and one channel group, computes the absolute maximum within
that group, stores a float32 scale factor, and writes the scaled FP8 output.
"""

import pandas
import tilus
import torch
from tilus import float8_e4m3, float16, float32, int32
from tilus.utils import benchmark_func, cdiv


@tilus.autotune("block_n", [128])
@tilus.autotune("warps", [4, 8])
class PerTokenCast(tilus.Script):
    def __init__(self, block_n: int, warps: int, num_per_channels: int = 128):
        super().__init__()
        self.block_m = 1
        self.block_n = block_n
        self.warps = warps
        self.num_per_channels = num_per_channels

    def __call__(
        self,
        num_tokens: int,
        hidden: int32,
        x_ptr: ~float16,
        out_ptr: ~float8_e4m3,
        out_sf_ptr: ~float32,
    ):
        self.attrs.blocks = (
            cdiv(num_tokens, self.block_m),
            cdiv(hidden, self.block_n),
        )
        self.attrs.warps = self.warps

        offset_m = self.blockIdx.x * self.block_m
        offset_n = self.blockIdx.y * self.block_n
        sf_col = offset_n // self.num_per_channels

        g_x = self.global_view(
            x_ptr,
            dtype=float16,
            shape=[num_tokens, hidden],
        )
        g_out = self.global_view(
            out_ptr,
            dtype=float8_e4m3,
            shape=[num_tokens, hidden],
        )
        g_out_sf = self.global_view(
            out_sf_ptr,
            dtype=float32,
            shape=[num_tokens, cdiv(hidden, self.num_per_channels)],
        )

        r_x = self.load_global(
            g_x,
            offsets=[offset_m, offset_n],
            shape=[self.block_m, self.block_n],
        ).to(float32)

        r_absmax = self.max(self.abs(r_x), dim=1, keepdim=True)
        r_fp8_max = self.register_tensor(
            dtype=float32,
            shape=[self.block_m, 1],
            init=448.0,
        )
        r_scale = self.where(r_absmax > 0.0, x=r_absmax / 448.0, y=1.0)
        r_inv_scale = self.where(r_absmax > 0.0, x=r_fp8_max / r_absmax, y=1.0)

        self.store_global(g_out_sf, r_scale, offsets=[offset_m, sf_col])
        self.store_global(
            g_out,
            (r_x * r_inv_scale).to(float8_e4m3),
            offsets=[offset_m, offset_n],
        )


def per_token_cast_reference(
    x: torch.Tensor,
    num_per_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, hidden = x.shape
    grouped = x.float().reshape(num_tokens, hidden // num_per_channels, num_per_channels)
    scales = grouped.abs().amax(dim=2) / 448.0
    scales = torch.where(scales > 0.0, scales, torch.ones_like(scales))
    out = (grouped / scales[:, :, None]).reshape_as(x).to(torch.float8_e4m3fn)
    return out, scales


def dequantized_sum(
    out: torch.Tensor, scales: torch.Tensor, num_per_channels: int
) -> torch.Tensor:
    grouped = out.float().reshape(
        out.shape[0],
        out.shape[1] // num_per_channels,
        num_per_channels,
    )
    return (grouped * scales[:, :, None]).sum()


def main():
    rows = []
    headers = [
        "tokens",
        "hidden",
        "torch (ms)",
        "tilus (ms)",
        "speedup",
        "sum diff",
    ]

    for num_tokens, hidden in [
        (128, 1024),
        (256, 2048),
        (257, 4096),
    ]:
        num_per_channels = 128
        kernel = PerTokenCast(num_per_channels=num_per_channels)

        x = (
            torch.randn(
                num_tokens,
                hidden,
                device="cuda",
                dtype=torch.float16,
            )
            * 2.0
        ).contiguous()
        out = torch.empty((num_tokens, hidden), device="cuda", dtype=torch.float8_e4m3fn)
        out_sf = torch.empty(
            (num_tokens, hidden // num_per_channels),
            device="cuda",
            dtype=torch.float32,
        )

        kernel(num_tokens, hidden, x, out, out_sf)
        expected_out, expected_sf = per_token_cast_reference(x, num_per_channels)

        max_code_diff = (out.float() - expected_out.float()).abs().max().item()
        assert max_code_diff <= 32.0, f"max decoded FP8 code diff is {max_code_diff}"
        torch.testing.assert_close(out_sf, expected_sf, atol=1e-5, rtol=1e-5)

        actual_sum = dequantized_sum(out, out_sf, num_per_channels)
        expected_sum = dequantized_sum(expected_out, expected_sf, num_per_channels)
        torch.testing.assert_close(actual_sum, expected_sum, atol=2.0, rtol=2e-2)
        sum_diff = (actual_sum - expected_sum).abs().item()

        torch_ms = benchmark_func(lambda: per_token_cast_reference(x, num_per_channels))
        tilus_ms = benchmark_func(lambda: kernel(num_tokens, hidden, x, out, out_sf))
        rows.append(
            [
                num_tokens,
                hidden,
                torch_ms,
                tilus_ms,
                f"{torch_ms / tilus_ms:.2f}x",
                sum_diff,
            ]
        )
        print(
            "Per-token FP8 cast matches reference for size "
            f"({num_tokens}, {hidden}); max code diff={max_code_diff:.6g}; "
            f"dequantized sum diff={sum_diff:.6g}"
        )

    print(pandas.DataFrame(rows, columns=headers))


if __name__ == "__main__":
    main()
