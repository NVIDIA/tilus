# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common BF16 128x128 path of TileKernels' per-channel E4M3 cast."""

import tilus
import torch
from tile_kernels.quant.per_channel_cast_kernel import per_channel_cast
from tilus import bfloat16, float8_e4m3, float32, int32
from tilus.ir.layout.ops import spatial
from tilus.utils import benchmark_func, cdiv

E4M3_MAX = 448.0
SF_CLAMP_MIN = 1.0e-4

def check_fp8_close(actual: torch.Tensor, expected: torch.Tensor, *, label: str, max_mismatch_frac: float = 0.01) -> None:
    """Check e4m3 values on their integer code ladder."""
    actual_bits = actual.view(torch.uint8).to(torch.int32)
    expected_bits = expected.view(torch.uint8).to(torch.int32)
    actual_codes = torch.where(actual_bits & 0x80 != 0, -(actual_bits & 0x7F), actual_bits & 0x7F)
    expected_codes = torch.where(expected_bits & 0x80 != 0, -(expected_bits & 0x7F), expected_bits & 0x7F)
    code_diff = (actual_codes - expected_codes).abs()
    max_code_diff = int(code_diff.max().item())
    mismatch_frac = float((code_diff != 0).to(torch.float64).mean().item())
    assert max_code_diff <= 1, f"{label}: {max_code_diff} e4m3 codes apart at worst; expected at most 1"
    assert mismatch_frac <= max_mismatch_frac, f"{label}: {mismatch_frac:.4%} of elements differ"


class PerChannelCast(tilus.Script):
    """BF16 input, 128-token groups, and hidden sizes divisible by 128 only."""

    def __init__(self):
        super().__init__()
        self.block_m, self.block_n, self.warps = 128, 128, 8

    def __call__(self, num_tokens: int32, hidden: int32, x_ptr: ~bfloat16, out_ptr: ~float8_e4m3, sf_ptr: ~float32):
        self.attrs.blocks = (cdiv(num_tokens, self.block_m), cdiv(hidden, self.block_n))
        self.attrs.warps = self.warps
        self.assume(num_tokens % self.block_m == 0)
        self.assume(hidden % self.block_n == 0)
        x = self.global_view(x_ptr, dtype=bfloat16, shape=[num_tokens, hidden])
        out = self.global_view(out_ptr, dtype=float8_e4m3, shape=[num_tokens, hidden])
        sf = self.global_view(sf_ptr, dtype=float32, shape=[cdiv(num_tokens, self.block_m), hidden])
        offset_m, offset_n = self.blockIdx.x * self.block_m, self.blockIdx.y * self.block_n
        # Keep the BF16 tile in shared memory across the reduction.  Reloading
        # it for the FP8 epilogue avoids keeping both the complete BF16 and
        # FP32 tiles live in registers, matching TileKernels' lifetime.
        shared_x = self.shared_tensor(dtype=bfloat16, shape=[self.block_m, self.block_n])
        input_values = self.load_global(x, offsets=[offset_m, offset_n], shape=[self.block_m, self.block_n])
        self.annotate_layout(input_values, spatial(8, 32).local(16, self.block_n // 32))
        self.store_shared(shared_x, input_values)
        self.sync()
        values_for_reduce = self.abs(self.load_shared(shared_x)).to(float32)
        # TileKernels maps 8 warps over [8, 32] and gives each thread a
        # [16, 4] micro-tile.  The generic inferred layout instead made every
        # thread retain an entire 128-value column; make this mapping explicit.
        self.annotate_layout(values_for_reduce, spatial(8, 32).local(16, self.block_n // 32))
        amax = self.max(values_for_reduce, dim=0, keepdim=True)
        amax = self.where(amax > SF_CLAMP_MIN, x=amax, y=SF_CLAMP_MIN)
        scale = amax / E4M3_MAX
        inv_scale = self.register_tensor(dtype=float32, shape=[1, self.block_n], init=E4M3_MAX) / amax
        self.store_global(sf, scale, offsets=[self.blockIdx.x, offset_n])
        output_values = self.load_shared(shared_x)
        self.annotate_layout(output_values, spatial(8, 32).local(16, self.block_n // 32))
        output_values_f32 = output_values.to(float32) * inv_scale
        output_values_fp8 = output_values_f32.to(float8_e4m3)
        self.store_global(out, output_values_fp8, offsets=[offset_m, offset_n])
        self.free_shared(shared_x)


def main():
    tokens, hidden = 8192, 8192
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16).contiguous()
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    sf = torch.empty(tokens // 128, hidden, device="cuda", dtype=torch.float32)
    kernel = PerChannelCast()
    kernel(tokens, hidden, x, out, sf)
    ref_amax = x.float().abs().reshape(-1, 128, hidden).amax(1).clamp_min(SF_CLAMP_MIN)
    torch.testing.assert_close(sf, ref_amax / E4M3_MAX, rtol=1e-5, atol=1e-7)
    tk_out, tk_sf = per_channel_cast(x, "e4m3", 128)
    torch.testing.assert_close(sf, tk_sf, rtol=1e-5, atol=1e-7)
    check_fp8_close(out, tk_out, label="per-channel Tilus vs TileKernels")
    def run_tilus():
        out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        sf = torch.empty(tokens // 128, hidden, device="cuda", dtype=torch.float32)
        kernel(tokens, hidden, x, out, sf)
        return out, sf
    tilus_ms = benchmark_func(run_tilus, warmup=10, repeat=50)
    tilekernels_ms = benchmark_func(lambda: per_channel_cast(x, "e4m3", 128), warmup=10, repeat=50)
    print(f"Per-channel FP8 cast: Tilus {tilus_ms:.4f} ms, TileKernels {tilekernels_ms:.4f} ms")


if __name__ == "__main__":
    main()
