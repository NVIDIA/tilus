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
from tile_kernels.quant.per_token_cast_kernel import per_token_cast
from tilus import float8_e4m3, float16, float32, int32
from tilus.utils import benchmark_func, cdiv


@tilus.autotune("block_m", [1, 2, 4, 8])
@tilus.autotune("groups_per_block", [1, 2, 4, 8])
@tilus.autotune("warps", [4, 8])
class PerTokenCast(tilus.Script):
    def __init__(
        self,
        block_m: int,
        groups_per_block: int,
        warps: int,
        num_per_channels: int = 128,
    ):
        super().__init__()
        self.block_m = block_m
        self.num_per_channels = num_per_channels
        self.groups_per_block = groups_per_block
        self.block_n = num_per_channels
        self.warps = warps

    def __call__(
        self,
        num_tokens: int,
        hidden: int32,
        x_ptr: ~float16,
        out_ptr: ~float8_e4m3,
        out_sf_ptr: ~float32,
    ):
        n_step = self.block_n * self.groups_per_block
        self.attrs.blocks = (
            cdiv(num_tokens, self.block_m),
            cdiv(hidden, n_step),
        )
        self.attrs.warps = self.warps
        self.assume(hidden % self.num_per_channels == 0)

        offset_m = self.blockIdx.x * self.block_m
        base_offset_n = self.blockIdx.y * n_step

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

        for gi in range(self.groups_per_block):
            offset_n = base_offset_n + gi * self.block_n
            sf_col = offset_n // self.num_per_channels

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


def tilekernels_per_token_cast_reference(
    x: torch.Tensor,
    num_per_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return per_token_cast(x, "e4m3", num_per_channels)


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
        "tilekernels (ms)",
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
        x_tilekernels = x.float()

        kernel(num_tokens, hidden, x, out, out_sf)
        expected_out, expected_sf = tilekernels_per_token_cast_reference(
            x_tilekernels,
            num_per_channels,
        )

        max_code_diff = (out.float() - expected_out.float()).abs().max().item()
        assert max_code_diff <= 32.0, f"max decoded FP8 code diff is {max_code_diff}"
        torch.testing.assert_close(out_sf, expected_sf, atol=1e-5, rtol=1e-5)

        actual_sum = dequantized_sum(out, out_sf, num_per_channels)
        expected_sum = dequantized_sum(expected_out, expected_sf, num_per_channels)
        torch.testing.assert_close(actual_sum, expected_sum, atol=2.0, rtol=2e-2)
        sum_diff = (actual_sum - expected_sum).abs().item()

        tilekernels_ms = benchmark_func(
            lambda: tilekernels_per_token_cast_reference(
                x_tilekernels,
                num_per_channels,
            )
        )
        tilus_ms = benchmark_func(lambda: kernel(num_tokens, hidden, x, out, out_sf))
        rows.append(
            [
                num_tokens,
                hidden,
                tilekernels_ms,
                tilus_ms,
                f"{tilekernels_ms / tilus_ms:.2f}x",
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
