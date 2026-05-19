# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused SwiGLU forward with per-token FP8 cast.

This is a Tilus translation of DeepSeek TileKernels'
``swiglu_forward_and_per_token_cast_kernel.py``.  It computes

    out = silu(x[:, :hidden]) * x[:, hidden:]

optionally applies a routing weight and expert mask, then quantizes each
``num_per_channels`` group to FP8 e4m3 with one float32 scale factor per
token/group.
"""

import pandas
import tilus
import torch
from tile_kernels.quant.swiglu_forward_and_per_token_cast_kernel import (
    swiglu_forward_and_per_token_cast,
)
from tilus import float8_e4m3, float16, float32, int32
from tilus.utils import benchmark_func, cdiv


@tilus.autotune("block_m", [1])
@tilus.autotune("groups_per_block", [1, 2, 4, 8, 16])
@tilus.autotune("warps", [1, 2, 4, 8])
class SwiGLUForwardAndPerTokenCast(tilus.Script):
    def __init__(
        self,
        block_m: int,
        groups_per_block: int,
        warps: int,
        with_weight: bool = True,
        with_pos_to_expert: bool = True,
        use_clamp: bool = True,
        num_per_channels: int = 128,
    ):
        super().__init__()
        self.block_m = block_m
        self.num_per_channels = num_per_channels
        self.groups_per_block = groups_per_block
        self.block_n = num_per_channels
        self.warps = warps
        self.with_weight = with_weight
        self.with_pos_to_expert = with_pos_to_expert
        self.use_clamp = use_clamp

    def __call__(
        self,
        num_expanded_tokens: int,
        hidden: int32,
        num_topk_values: int32,
        x_ptr: ~float16,
        out_ptr: ~float8_e4m3,
        out_sf_ptr: ~float32,
        pos_to_token_topk_ptr: ~int32,
        topk_weights_ptr: ~float32,
        pos_to_expert_ptr: ~int32,
        swiglu_clamp_value: float32,
    ):
        n_step = self.block_n * self.groups_per_block
        self.attrs.blocks = (
            cdiv(num_expanded_tokens, self.block_m),
            cdiv(hidden, n_step),
        )
        self.attrs.warps = self.warps
        self.assume(hidden % self.num_per_channels == 0)

        offset_m = self.blockIdx.x * self.block_m
        base_offset_n = self.blockIdx.y * n_step

        g_x = self.global_view(
            x_ptr,
            dtype=float16,
            shape=[num_expanded_tokens, hidden * 2],
        )
        g_out = self.global_view(
            out_ptr,
            dtype=float8_e4m3,
            shape=[num_expanded_tokens, hidden],
        )
        g_out_sf = self.global_view(
            out_sf_ptr,
            dtype=float32,
            shape=[num_expanded_tokens, cdiv(hidden, self.num_per_channels)],
        )
        g_pos_to_token_topk = self.global_view(
            pos_to_token_topk_ptr,
            dtype=int32,
            shape=[num_expanded_tokens],
        )
        g_topk_weights = self.global_view(
            topk_weights_ptr,
            dtype=float32,
            shape=[num_topk_values],
        )
        g_pos_to_expert = self.global_view(
            pos_to_expert_ptr,
            dtype=int32,
            shape=[num_expanded_tokens],
        )

        if (not self.with_pos_to_expert) or g_pos_to_expert[offset_m].item() >= 0:
            base_sf_col = base_offset_n // self.num_per_channels

            # Wide load: full n_step at once so layout-inference vectorises.
            r_l = self.load_global(
                g_x,
                offsets=[offset_m, base_offset_n],
                shape=[self.block_m, n_step],
            ).to(float32)
            r_r = self.load_global(
                g_x,
                offsets=[offset_m, base_offset_n + hidden],
                shape=[self.block_m, n_step],
            ).to(float32)

            if self.use_clamp:
                negative_swiglu_clamp_value = 0.0 - swiglu_clamp_value
                r_l = self.where(r_l > swiglu_clamp_value, x=swiglu_clamp_value, y=r_l)
                r_r = self.where(r_r > swiglu_clamp_value, x=swiglu_clamp_value, y=r_r)
                r_r = self.where(
                    r_r < negative_swiglu_clamp_value,
                    x=negative_swiglu_clamp_value,
                    y=r_r,
                )

            r_silu = r_l / (self.exp(-r_l) + 1.0)
            r_value = r_silu * r_r

            if self.with_weight:
                topk_pos = g_pos_to_token_topk[offset_m].item()
                if topk_pos >= 0:
                    topk_weight = g_topk_weights[topk_pos].item()
                    r_value = r_value * topk_weight

            # Reshape into [block_m, groups_per_block, num_per_channels] so the
            # per-group absmax is a single reduce on dim=2.
            r_value_grouped = self.reshape(
                r_value,
                shape=[self.block_m, self.groups_per_block, self.num_per_channels],
            )
            r_absmax = self.max(
                self.abs(r_value_grouped), dim=2, keepdim=True
            )  # [block_m, groups_per_block, 1]
            r_fp8_max = self.register_tensor(
                dtype=float32,
                shape=[self.block_m, self.groups_per_block, 1],
                init=448.0,
            )
            r_scale = self.where(r_absmax > 0.0, x=r_absmax / 448.0, y=1.0)
            r_inv_scale = self.where(r_absmax > 0.0, x=r_fp8_max / r_absmax, y=1.0)

            # Store one fp32 scale per group.
            r_scale_2d = self.reshape(
                r_scale, shape=[self.block_m, self.groups_per_block]
            )
            self.store_global(g_out_sf, r_scale_2d, offsets=[offset_m, base_sf_col])

            # Apply scaling, flatten back, cast to fp8, bulk store.
            r_out_grouped = (r_value_grouped * r_inv_scale).to(float8_e4m3)
            r_out = self.reshape(r_out_grouped, shape=[self.block_m, n_step])
            self.store_global(g_out, r_out, offsets=[offset_m, base_offset_n])


def tilekernels_swiglu_reference(
    x: torch.Tensor,
    pos_to_token_topk: torch.Tensor,
    topk_weights: torch.Tensor,
    pos_to_expert: torch.Tensor,
    clamp_value: float,
    num_per_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return swiglu_forward_and_per_token_cast(
        x,
        "e4m3",
        num_per_channels,
        pos_to_token_topk=pos_to_token_topk,
        topk_weights=topk_weights,
        pos_to_expert=pos_to_expert,
        swiglu_clamp_value=clamp_value,
    )


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

    for num_expanded_tokens, hidden, num_tokens, num_topk in [
        (128, 1024, 64, 2),
        (256, 2048, 128, 2),
        (257, 4096, 128, 2),
        (1024, 4096, 512, 2),
    ]:
        num_per_channels = 128
        kernel = SwiGLUForwardAndPerTokenCast(num_per_channels=num_per_channels)

        x = (
            torch.randn(
                num_expanded_tokens,
                hidden * 2,
                device="cuda",
                dtype=torch.float16,
            )
            * 2.0
        ).contiguous()
        pos_to_token_topk = torch.arange(
            num_expanded_tokens,
            device="cuda",
            dtype=torch.int32,
        ) % (num_tokens * num_topk)
        topk_weights = torch.rand(
            num_tokens,
            num_topk,
            device="cuda",
            dtype=torch.float32,
        )
        pos_to_expert = torch.ones(num_expanded_tokens, device="cuda", dtype=torch.int32)
        pos_to_expert[::17] = -1

        out = torch.empty(
            (num_expanded_tokens, hidden),
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        out_sf = torch.empty(
            (num_expanded_tokens, hidden // num_per_channels),
            device="cuda",
            dtype=torch.float32,
        )
        x_tilekernels = x.float()

        clamp_value = 6.0
        kernel(
            num_expanded_tokens,
            hidden,
            num_tokens * num_topk,
            x,
            out,
            out_sf,
            pos_to_token_topk,
            topk_weights,
            pos_to_expert,
            clamp_value,
        )

        expected_out, expected_sf = tilekernels_swiglu_reference(
            x_tilekernels,
            pos_to_token_topk,
            topk_weights,
            pos_to_expert,
            clamp_value,
            num_per_channels,
        )
        valid = pos_to_expert >= 0
        max_code_diff = (
            (out[valid].float() - expected_out[valid].float()).abs().max().item()
        )
        assert max_code_diff <= 32.0, f"max decoded FP8 code diff is {max_code_diff}"
        torch.testing.assert_close(
            out_sf[valid],
            expected_sf[valid],
            atol=1e-5,
            rtol=1e-5,
        )
        actual_sum = dequantized_sum(out[valid], out_sf[valid], num_per_channels)
        expected_sum = dequantized_sum(
            expected_out[valid],
            expected_sf[valid],
            num_per_channels,
        )
        torch.testing.assert_close(actual_sum, expected_sum, atol=2.0, rtol=2e-2)
        sum_diff = (actual_sum - expected_sum).abs().item()

        tilekernels_ms = benchmark_func(
            lambda: tilekernels_swiglu_reference(
                x_tilekernels,
                pos_to_token_topk,
                topk_weights,
                pos_to_expert,
                clamp_value,
                num_per_channels,
            )
        )
        tilus_ms = benchmark_func(
            lambda: kernel(
                num_expanded_tokens,
                hidden,
                num_tokens * num_topk,
                x,
                out,
                out_sf,
                pos_to_token_topk,
                topk_weights,
                pos_to_expert,
                clamp_value,
            )
        )
        rows.append(
            [
                num_expanded_tokens,
                hidden,
                tilekernels_ms,
                tilus_ms,
                f"{tilekernels_ms / tilus_ms:.2f}x",
                sum_diff,
            ]
        )
        print(
            "SwiGLU FP8 cast matches reference for size "
            f"({num_expanded_tokens}, {hidden}); max code diff={max_code_diff:.6g}; "
            f"dequantized sum diff={sum_diff:.6g}"
        )

    print(pandas.DataFrame(rows, columns=headers))


if __name__ == "__main__":
    main()
