# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-token FP8 cast with scale factors.

This is a Tilus translation of DeepSeek TileKernels'
``per_token_cast_kernel.py``.  Each CTA processes one token and one channel
group, computes the absolute maximum within that group, stores a float32 scale
factor, and writes the scaled FP8 e4m3 output.

The input is bfloat16.  That is not a free choice: TileKernels'
``get_cast_input_and_config`` asserts the unquantized input is bfloat16 or
float32, so bf16 is the only 16-bit type both implementations accept.  Since
this kernel is purely DRAM-bound on its input read, handing the reference fp32
while Tilus reads 16-bit halves the reference's achievable bandwidth-limited
runtime and makes the comparison meaningless.  Both sides here read the same
bf16 tensor.

Scale-factor arithmetic follows TileKernels' ``get_sf_and_inv`` exactly: the
group absmax is clamped from below to 1e-4, the scale is ``absmax / 448`` and
its reciprocal is ``448 / absmax``.
"""

import pandas
import tilus
import torch
from fp8_check import (
    E4M3_MAX,
    SF_CLAMP_MIN,
    check_fp8_close,
    check_scales_close,
    dequantize,
    quantization_snr_db,
    torch_per_token_cast,
)
from tile_kernels.quant.per_token_cast_kernel import per_token_cast
from tilus import bfloat16, float8_e4m3, float32, int32
from tilus.utils import benchmark_func, cdiv


@tilus.autotune("block_m", [1, 2, 4])
@tilus.autotune("groups_per_block", [1, 4, 16, 64])
@tilus.autotune("warps", [4, 8, 16])
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
        x_ptr: ~bfloat16,
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
            dtype=bfloat16,
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

        # One wide load of the whole tile rather than a loop of per-group
        # loads.  The loop form does not get unrolled, so each iteration
        # exposed the full global-load latency with nothing to overlap it;
        # issuing every load up front and reshaping for the reduction is worth
        # ~5% at the DRAM-bound shape.
        r_x = self.load_global(
            g_x,
            offsets=[offset_m, base_offset_n],
            shape=[self.block_m, n_step],
        ).to(float32)

        # Reshape into [block_m, groups_per_block, num_per_channels] so the
        # per-group absmax is a single reduce on dim=2.
        r_x_grouped = self.reshape(
            r_x,
            shape=[self.block_m, self.groups_per_block, self.num_per_channels],
        )

        # Clamp the absmax from below exactly as TileKernels does, so an
        # all-zero group produces a tiny scale instead of a division by zero,
        # and so both sides agree bit-for-bit on the clamped value.
        r_absmax = self.max(self.abs(r_x_grouped), dim=2, keepdim=True)
        r_amax = self.where(r_absmax > SF_CLAMP_MIN, x=r_absmax, y=SF_CLAMP_MIN)
        r_fp8_max = self.register_tensor(
            dtype=float32,
            shape=[self.block_m, self.groups_per_block, 1],
            init=E4M3_MAX,
        )
        r_scale = r_amax / E4M3_MAX
        r_inv_scale = r_fp8_max / r_amax

        # Store one fp32 scale per group.
        r_scale_2d = self.reshape(r_scale, shape=[self.block_m, self.groups_per_block])
        self.store_global(
            g_out_sf,
            r_scale_2d,
            offsets=[offset_m, base_offset_n // self.num_per_channels],
        )

        # Apply scaling, flatten back, cast to fp8, bulk store.
        r_out_grouped = (r_x_grouped * r_inv_scale).to(float8_e4m3)
        r_out = self.reshape(r_out_grouped, shape=[self.block_m, n_step])
        self.store_global(g_out, r_out, offsets=[offset_m, base_offset_n])


def main():
    rows = []
    headers = [
        "tokens",
        "hidden",
        "tilekernels (ms)",
        "tilus (ms)",
        "speedup",
        "code mismatch",
        "sf rel err",
        "snr (dB)",
    ]

    # The first three shapes move only a few hundred KB, so at ~6 us they are
    # dominated by launch and dispatch overhead rather than by the kernel.  The
    # last one reads 128 MB and is genuinely DRAM-bound, which is the regime
    # this kernel is written for and the only one where the speedup column
    # says anything about code quality.
    for num_tokens, hidden in [
        (128, 1024),
        (256, 2048),
        (257, 4096),
        (8192, 8192),
    ]:
        num_per_channels = 128
        kernel = PerTokenCast(num_per_channels=num_per_channels)

        # One bf16 tensor, read by both implementations.
        x = (
            torch.randn(
                num_tokens,
                hidden,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 2.0
        ).contiguous()

        def run_tilus():
            # Allocate here rather than reusing preallocated buffers, because
            # the TileKernels entry point allocates its returns on every call.
            out = torch.empty(
                (num_tokens, hidden), device="cuda", dtype=torch.float8_e4m3fn
            )
            out_sf = torch.empty(
                (num_tokens, hidden // num_per_channels),
                device="cuda",
                dtype=torch.float32,
            )
            kernel(num_tokens, hidden, x, out, out_sf)
            return out, out_sf

        def run_tilekernels():
            return per_token_cast(x, "e4m3", num_per_channels)

        out, out_sf = run_tilus()
        expected_out, expected_sf = run_tilekernels()
        torch_out, torch_sf = torch_per_token_cast(x, num_per_channels)

        # Tilus against TileKernels: same inputs, same arithmetic, so they may
        # differ only by the approximate fp32 division Tilus compiles with.
        sf_rel_err = check_scales_close(
            out_sf, expected_sf, label=f"({num_tokens}, {hidden}) sf vs tilekernels"
        )
        stats = check_fp8_close(
            out,
            expected_out,
            label=f"({num_tokens}, {hidden}) out vs tilekernels",
        )

        # Both against an independent fp32 PyTorch cast, so that agreeing with
        # each other is not mistaken for being correct.
        check_scales_close(
            out_sf, torch_sf, label=f"({num_tokens}, {hidden}) sf vs torch"
        )
        check_fp8_close(out, torch_out, label=f"({num_tokens}, {hidden}) out vs torch")
        check_fp8_close(
            expected_out,
            torch_out,
            label=f"({num_tokens}, {hidden}) tilekernels out vs torch",
        )

        snr_db = quantization_snr_db(x, dequantize(out, out_sf, num_per_channels))

        tilekernels_ms = benchmark_func(run_tilekernels)
        tilus_ms = benchmark_func(run_tilus)
        rows.append(
            [
                num_tokens,
                hidden,
                tilekernels_ms,
                tilus_ms,
                f"{tilekernels_ms / tilus_ms:.2f}x",
                f"{stats.mismatch_frac:.4%}",
                f"{sf_rel_err:.2e}",
                f"{snr_db:.1f}",
            ]
        )
        print(
            "Per-token FP8 cast matches reference for size "
            f"({num_tokens}, {hidden}): every code within 1 of TileKernels and "
            f"of fp32 torch, {stats.mismatch_frac:.4%} of codes differ at all, "
            f"scale factors agree to {sf_rel_err:.2e} relative, "
            f"quantization SNR {snr_db:.1f} dB"
        )

    print(pandas.DataFrame(rows, columns=headers).to_string(index=False))
    print(
        "\nBoth implementations read the same bf16 input and allocate their own "
        "outputs.\nThe Tilus kernel is autotuned per shape; the TileKernels "
        "kernel picks its tiling\nanalytically from `hidden`, so it is not tuned "
        "against this measurement."
    )


if __name__ == "__main__":
    main()
