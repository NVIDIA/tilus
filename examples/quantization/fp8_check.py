# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared reference and correctness checks for the per-token FP8 cast examples.

FP8 e4m3 has a three-bit mantissa, so adjacent representable values are up to
6.25% apart and the ladder tops out at 448.  Comparing *decoded* e4m3 values
with an absolute float tolerance is therefore nearly vacuous: a tolerance loose
enough to absorb a single rounding step near the top of the range is +/-32,
which is also loose enough to accept a kernel that is quantizing against a
completely wrong scale factor.

These helpers compare on the code ladder instead.  e4m3 is stored
sign-magnitude, so ranking the seven magnitude bits and re-applying the sign
gives an integer ordinal in which *any* two adjacent representable values
differ by exactly one, uniformly across the dynamic range.  "Agrees to within
one rounding step" is then ``|ordinal(a) - ordinal(b)| <= 1``.

One rounding step is the tightest claim that holds here.  Tilus compiles every
kernel with ``-prec-div=false`` (see ``python/tilus/hidet/backend/build.py``),
so the fp32 divisions that produce the scale factor and its reciprocal are
approximate to about one ulp.  TileLang uses exact division.  With identical
inputs and otherwise identical arithmetic the two therefore still straddle an
e4m3 rounding boundary on a small fraction of elements; everything else is
bit-identical.  A real bug moves the mismatch rate or the code distance well
outside those bounds.
"""

from typing import NamedTuple

import torch

# Largest finite magnitude representable in e4m3 (TileKernels' ``T.max_value``).
E4M3_MAX = 448.0

# TileKernels clamps the group absmax from below before dividing, so that an
# all-zero group yields a tiny scale rather than a division by zero.  See
# ``CastOutputConfig.clamp_min_value`` in ``tile_kernels/quant/common.py``.
SF_CLAMP_MIN = 1e-4


class CodeLadderStats(NamedTuple):
    """Result of comparing two e4m3 tensors on the code ladder."""

    max_code_diff: int
    mismatch_frac: float


def fp8_ordinal(x: torch.Tensor) -> torch.Tensor:
    """Rank each e4m3 value on the ladder of representable values.

    e4m3 is sign-magnitude, so the seven magnitude bits are already a monotone
    rank within one sign.  Negating them for the negative half gives a single
    signed ordinal where consecutive representable values always differ by one.
    """
    assert x.dtype == torch.float8_e4m3fn
    bits = x.view(torch.uint8).to(torch.int32)
    magnitude = bits & 0x7F
    return torch.where(bits & 0x80 != 0, -magnitude, magnitude)


def check_fp8_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    max_mismatch_frac: float = 0.01,
) -> CodeLadderStats:
    """Assert two e4m3 tensors agree to within one rounding step.

    Fails if any element is more than one representable value away, or if more
    than ``max_mismatch_frac`` of elements disagree at all.  The second bound
    matters: a systematic error that happens to be small still shows up as a
    mismatch rate far above the ~0.1% produced by fp32 division rounding.
    """
    code_diff = (fp8_ordinal(actual) - fp8_ordinal(expected)).abs()
    max_code_diff = int(code_diff.max().item())
    mismatch_frac = float((code_diff != 0).to(torch.float64).mean().item())

    assert max_code_diff <= 1, (
        f"{label}: {max_code_diff} e4m3 codes apart at worst; expected at most "
        f"1 (a single rounding step)"
    )
    assert mismatch_frac <= max_mismatch_frac, (
        f"{label}: {mismatch_frac:.4%} of elements differ, above the "
        f"{max_mismatch_frac:.4%} budget for fp32 division rounding"
    )
    return CodeLadderStats(max_code_diff, mismatch_frac)


def check_scales_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    rtol: float = 1e-6,
) -> float:
    """Assert two fp32 scale-factor tensors agree to a few fp32 ulps.

    The scale is ``max(absmax, 1e-4) / 448`` on both sides, computed from an
    absmax that is a max over identical inputs and so is bit-identical.  Only
    the division differs, hence a tolerance in ulps (1 ulp ~ 1.2e-7 relative)
    rather than the 1e-5 that would also pass a wrong reduction.
    """
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=0.0, msg=label)
    return float(((actual - expected).abs() / expected.abs()).max().item())


def torch_per_token_cast(
    values: torch.Tensor,
    num_per_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ground-truth per-token FP8 cast in fp32 PyTorch.

    ``values`` holds the activations to quantize (for SwiGLU, the post-SwiGLU
    values).  Checking both kernels against this catches the case where they
    agree with each other but are both wrong.
    """
    num_tokens, hidden = values.shape
    grouped = values.float().reshape(num_tokens, hidden // num_per_channels, -1)
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(SF_CLAMP_MIN)
    scale = (amax / E4M3_MAX).squeeze(-1)
    scaled = (grouped * (E4M3_MAX / amax)).clamp(-E4M3_MAX, E4M3_MAX)
    out = scaled.reshape(num_tokens, hidden).to(torch.float8_e4m3fn)
    return out, scale


def dequantize(
    out: torch.Tensor,
    scales: torch.Tensor,
    num_per_channels: int,
) -> torch.Tensor:
    """Decode an e4m3 tensor and its per-group scales back to fp32."""
    grouped = out.float().reshape(
        out.shape[0],
        out.shape[1] // num_per_channels,
        num_per_channels,
    )
    return (grouped * scales[:, :, None]).reshape(out.shape)


def quantization_snr_db(reference: torch.Tensor, dequantized: torch.Tensor) -> float:
    """Signal-to-noise ratio of a dequantized tensor against its fp32 source.

    Reported rather than asserted on: it is the sanity check that the cast is
    carrying real information.  Round-to-nearest into a three-bit mantissa
    lands near 32 dB for Gaussian activations, and is stable across shapes, so
    a value well below that means the scale factors are wrong even if the code
    ladder checks pass.
    """
    reference = reference.float()
    noise = dequantized.float() - reference
    return float(
        10.0
        * torch.log10(reference.square().sum() / noise.square().sum().clamp_min(1e-30))
    )
