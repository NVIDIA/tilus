# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused SwiGLU forward with per-token FP8 cast.

This is a Tilus translation of DeepSeek TileKernels'
``swiglu_forward_and_per_token_cast_kernel.py``.  It computes

    out = silu(clamp(x[:, :hidden])) * clamp(x[:, hidden:])

optionally applies a routing weight and expert mask, then quantizes each
``num_per_channels`` group to FP8 e4m3 with one float32 scale factor per
token/group.

The input is bfloat16 on both sides.  The kernel is DRAM-bound on its two
input reads, so giving the reference fp32 while Tilus reads 16-bit would double
the reference's traffic and produce a speedup number that measures the dtype
rather than the kernel.  bf16 is also what the companion ``per_token_cast``
example must use, since TileKernels' cast entry point accepts only bf16 or
fp32 for unquantized input.

Scale-factor arithmetic follows TileKernels' ``get_sf_and_inv`` exactly: the
group absmax is clamped from below to 1e-4, the scale is ``absmax / 448`` and
its reciprocal is ``448 / absmax``.
"""

from typing import NamedTuple

import pandas
import tilus
import torch
from tile_kernels.quant.swiglu_forward_and_per_token_cast_kernel import (
    swiglu_forward_and_per_token_cast,
)
from tilus import bfloat16, float8_e4m3, float32, int32
from tilus.utils import benchmark_func, cdiv

E4M3_MAX = 448.0
SF_CLAMP_MIN = 1e-4


class CodeLadderStats(NamedTuple):
    max_code_diff: int
    mismatch_frac: float


def fp8_ordinal(x: torch.Tensor) -> torch.Tensor:
    """Map e4m3 values to consecutive signed integer codes."""
    assert x.dtype == torch.float8_e4m3fn
    bits = x.view(torch.uint8).to(torch.int32)
    magnitude = bits & 0x7F
    return torch.where(bits & 0x80 != 0, -magnitude, magnitude)


def check_fp8_close(actual: torch.Tensor, expected: torch.Tensor, *, label: str, max_mismatch_frac: float = 0.01) -> CodeLadderStats:
    code_diff = (fp8_ordinal(actual) - fp8_ordinal(expected)).abs()
    max_code_diff = int(code_diff.max().item())
    mismatch_frac = float((code_diff != 0).to(torch.float64).mean().item())
    assert max_code_diff <= 1, f"{label}: {max_code_diff} e4m3 codes apart at worst; expected at most 1"
    assert mismatch_frac <= max_mismatch_frac, f"{label}: {mismatch_frac:.4%} of elements differ"
    return CodeLadderStats(max_code_diff, mismatch_frac)


def check_scales_close(actual: torch.Tensor, expected: torch.Tensor, *, label: str, rtol: float = 1e-6) -> float:
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=0.0, msg=label)
    return float(((actual - expected).abs() / expected.abs()).max().item())


def torch_per_token_cast(values: torch.Tensor, num_per_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, hidden = values.shape
    grouped = values.float().reshape(num_tokens, hidden // num_per_channels, -1)
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(SF_CLAMP_MIN)
    scale = (amax / E4M3_MAX).squeeze(-1)
    out = (grouped * (E4M3_MAX / amax)).clamp(-E4M3_MAX, E4M3_MAX).reshape(num_tokens, hidden).to(torch.float8_e4m3fn)
    return out, scale


def dequantize(out: torch.Tensor, scales: torch.Tensor, num_per_channels: int) -> torch.Tensor:
    grouped = out.float().reshape(out.shape[0], out.shape[1] // num_per_channels, num_per_channels)
    return (grouped * scales[:, :, None]).reshape(out.shape)


def quantization_snr_db(reference: torch.Tensor, dequantized: torch.Tensor) -> float:
    noise = dequantized.float() - reference.float()
    return float(10.0 * torch.log10(reference.float().square().sum() / noise.square().sum().clamp_min(1e-30)))



@tilus.autotune("block_m", [1])
@tilus.autotune("groups_per_block", [1, 4, 16, 64])
@tilus.autotune("warps", [2, 4, 8, 16])
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
        x_ptr: ~bfloat16,
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
            dtype=bfloat16,
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
            # Clamp the absmax from below exactly as TileKernels does.
            r_amax = self.where(r_absmax > SF_CLAMP_MIN, x=r_absmax, y=SF_CLAMP_MIN)
            r_fp8_max = self.register_tensor(
                dtype=float32,
                shape=[self.block_m, self.groups_per_block, 1],
                init=E4M3_MAX,
            )
            r_scale = r_amax / E4M3_MAX
            r_inv_scale = r_fp8_max / r_amax

            # Store one fp32 scale per group.
            r_scale_2d = self.reshape(
                r_scale, shape=[self.block_m, self.groups_per_block]
            )
            self.store_global(g_out_sf, r_scale_2d, offsets=[offset_m, base_sf_col])

            # Apply scaling, flatten back, cast to fp8, bulk store.
            r_out_grouped = (r_value_grouped * r_inv_scale).to(float8_e4m3)
            r_out = self.reshape(r_out_grouped, shape=[self.block_m, n_step])
            self.store_global(g_out, r_out, offsets=[offset_m, base_offset_n])


def torch_swiglu(
    x: torch.Tensor,
    pos_to_token_topk: torch.Tensor,
    topk_weights: torch.Tensor,
    clamp_value: float,
) -> torch.Tensor:
    """Reference SwiGLU activation in fp32, before quantization.

    Mirrors TileKernels' operation order: clamp, silu, multiply by the gate,
    then apply the routing weight.
    """
    hidden = x.shape[1] // 2
    left = x[:, :hidden].float().clamp(max=clamp_value)
    right = x[:, hidden:].float().clamp(min=-clamp_value, max=clamp_value)
    value = left / (1.0 + torch.exp(-left)) * right
    weight = topk_weights.reshape(-1)[pos_to_token_topk.long()]
    return value * weight[:, None]


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

    # The first four shapes are dominated by launch and dispatch overhead: they
    # move at most a few MB and run in under 25 us.  The last one reads 128 MB
    # and is genuinely DRAM-bound, which is the regime this kernel is written
    # for and the only one where the speedup column says anything about code
    # quality.
    for num_expanded_tokens, hidden, num_tokens, num_topk in [
        (128, 1024, 64, 2),
        (256, 2048, 128, 2),
        (257, 4096, 128, 2),
        (1024, 4096, 512, 2),
        (4096, 8192, 2048, 2),
    ]:
        num_per_channels = 128
        kernel = SwiGLUForwardAndPerTokenCast(num_per_channels=num_per_channels)

        # One bf16 tensor, read by both implementations.
        x = (
            torch.randn(
                num_expanded_tokens,
                hidden * 2,
                device="cuda",
                dtype=torch.bfloat16,
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
        clamp_value = 6.0

        def run_tilus():
            # Allocate here rather than reusing preallocated buffers, because
            # the TileKernels entry point allocates its returns on every call.
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
            return out, out_sf

        def run_tilekernels():
            return swiglu_forward_and_per_token_cast(
                x,
                "e4m3",
                num_per_channels,
                pos_to_token_topk=pos_to_token_topk,
                topk_weights=topk_weights,
                pos_to_expert=pos_to_expert,
                swiglu_clamp_value=clamp_value,
            )

        out, out_sf = run_tilus()
        expected_out, expected_sf = run_tilekernels()

        # Masked-out rows are never written by either kernel, so they hold
        # whatever the allocator handed back; compare only the live rows.
        valid = pos_to_expert >= 0
        activations = torch_swiglu(
            x[valid], pos_to_token_topk[valid], topk_weights, clamp_value
        )
        torch_out, torch_sf = torch_per_token_cast(activations, num_per_channels)

        # Tilus against TileKernels: same inputs, same arithmetic, so they may
        # differ only by the approximate fp32 division Tilus compiles with.
        label = f"({num_expanded_tokens}, {hidden})"
        sf_rel_err = check_scales_close(
            out_sf[valid], expected_sf[valid], label=f"{label} sf vs tilekernels"
        )
        stats = check_fp8_close(
            out[valid], expected_out[valid], label=f"{label} out vs tilekernels"
        )

        # Both against an independent fp32 PyTorch SwiGLU and cast, so that
        # agreeing with each other is not mistaken for being correct.  The
        # tolerance on the scale factors is looser than in `per_token_cast`
        # because the absmax is taken over a transcendental (`exp`), where the
        # device and PyTorch implementations may differ by an ulp.
        check_scales_close(
            out_sf[valid], torch_sf, label=f"{label} sf vs torch", rtol=1e-4
        )
        check_fp8_close(out[valid], torch_out, label=f"{label} out vs torch")
        check_fp8_close(
            expected_out[valid],
            torch_out,
            label=f"{label} tilekernels out vs torch",
        )

        snr_db = quantization_snr_db(
            activations, dequantize(out[valid], out_sf[valid], num_per_channels)
        )

        tilekernels_ms = benchmark_func(run_tilekernels)
        tilus_ms = benchmark_func(run_tilus)
        rows.append(
            [
                num_expanded_tokens,
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
            f"SwiGLU FP8 cast matches reference for size {label}: every code "
            f"within 1 of TileKernels and of fp32 torch, "
            f"{stats.mismatch_frac:.4%} of codes differ at all, scale factors "
            f"agree to {sf_rel_err:.2e} relative, "
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
