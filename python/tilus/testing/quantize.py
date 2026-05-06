# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quantization helpers for testing low-precision kernels.

These are reference implementations meant for tests / examples; they prefer
clarity over performance. Use them to generate quantized inputs and to
dequantize outputs back to FP32 for comparison against a high-precision
reference matmul.
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch

from tilus.hidet.ir.dtypes import float4_e2m1, float32

# NVFP4 spec: per-block scale factor covers 16 K-elements.
NVFP4_SF_BLOCK_K = 16

# Range of FP4 E2M1: {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}. Max magnitude = 6.
_FP4_E2M1_MAX = 6.0


def shuffle_sf_to_block_scaled_layout(
    sf: torch.Tensor,
    sf_block_size: Literal[16, 32] = 16,
) -> torch.Tensor:
    """Pre-shuffle a natural ``(rows, K_blocks)`` SF tensor into the canonical
    HBM layout consumed by ``tcgen05.cp.32x128b.warpx4`` + block-scaled
    ``tcgen05.mma.block_scale`` (see :meth:`tilus.lang.cuda.Tcgen05Module.mma_scaled`).

    The canonical HBM layout is a 5-mode tiling::

        ((lane=32, MN_fold=4, K_inner=4), (MN_outer = rows / 128, K_outer = K_blocks / 4))
        :( (16,           4,         1),  (K_blocks/4 * 512,             512))   (bytes)

    Each `(MN_outer, K_outer)` outer tile is a 512-byte contiguous atom that
    maps 1:1 to one MMA inst's SF region in TMEM (32 lanes × 4 cells × 4 bytes
    per cell = 512 bytes).

    Within an atom, the byte order is `(lane → MN_fold → K_inner)` row-major:

    * ``lane`` (= ``rows % 32``) → 16-byte stride.
    * ``MN_fold`` (= ``(rows // 32) % 4``) → 4-byte stride.
    * ``K_inner`` (= ``K_blocks % 4``) → 1-byte stride.

    The function works for both SFA (``rows == M``) and SFB (``rows == N``).
    The atom shape `(32, 4, 4)` is identical for ``scale_vec::1X``, ``2X``,
    and ``4X`` — only the *interpretation* of the K_inner axis differs (the
    kernel selects which byte slot via ``sfa_id``/``sfb_id``).

    Parameters
    ----------
    sf : torch.Tensor
        SF tensor in natural row-major layout. Shape ``(rows, K_blocks)`` with
        ``rows`` a multiple of 128 and ``K_blocks`` a multiple of 4. Dtype is
        any 1-byte type (typically ``torch.float8_e4m3fn`` for UE4M3 SFs or a
        ``uint8`` view for UE8M0 SFs).
    sf_block_size : int, default 16
        The K-elements covered per SF block (``16`` for NVFP4 / MXFP4 with
        ``scale_vec::4X``; ``32`` for MXFP8/MXFP4 with ``scale_vec::1X`` or
        ``2X``). The shuffle layout itself is the same for all values; this
        argument is kept for documentation and to match the kernel's
        ``mma_scaled(sf_block_size=...)`` choice.

    Returns
    -------
    torch.Tensor
        Contiguous 1-D tensor of bytes (logically same shape as ``sf``,
        viewed as ``rows × K_blocks`` bytes), with the byte order rearranged
        to the canonical layout. Pass its ``.data_ptr()`` to a tilus kernel
        whose SFA / SFB SMEM tile is declared with shape
        ``[block_rows, block_K_blocks]`` of the same SF dtype.

    Notes
    -----
    The output tensor has the same total byte count as the input but its
    byte order is shuffled. To reconstruct the natural layout from a shuffled
    tensor, reverse the index-permutation in the same pattern.
    """
    if sf.ndim != 2:
        raise ValueError(f"sf must be 2D (rows, K_blocks), got shape {tuple(sf.shape)}")
    rows, k_blocks = sf.shape
    if rows % 128 != 0:
        raise ValueError(f"sf.shape[0] (= {rows}) must be a multiple of 128 for the canonical SF layout")
    if k_blocks % 4 != 0:
        raise ValueError(f"sf.shape[1] (= {k_blocks}) must be a multiple of 4 for the canonical SF layout")
    if sf.element_size() != 1:
        raise ValueError(
            f"sf must be a 1-byte dtype (e.g. float8_e4m3fn or uint8 view); got dtype {sf.dtype}"
        )
    _ = sf_block_size  # only kept for caller documentation; layout is the same for 16/32

    rows_per_atom = 128
    k_blocks_per_atom = 4
    M_outer = rows // rows_per_atom
    K_outer = k_blocks // k_blocks_per_atom

    # Reshape natural (rows, K_blocks) into the 5-mode canonical layout.
    #   (rows, K_blocks) ─ logical
    #   = (M_outer, M_fold=4, lane=32, K_outer, K_inner=4)        [logical view]
    # because rows = M_outer*4*32, K_blocks = K_outer*4, and within each
    # 128-row block the indexing is `M_fold*32 + lane`.
    sf_view = sf.contiguous().view(M_outer, 4, 32, K_outer, 4)
    # Permute to (M_outer, K_outer, lane=32, M_fold=4, K_inner=4) — the
    # canonical order. M_fold and lane are swapped relative to the logical
    # view; the permutation is `(0, 3, 2, 1, 4)`.
    sf_perm = sf_view.permute(0, 3, 2, 1, 4).contiguous()
    # Flatten back to a 2D byte tensor of the same total size (so the
    # caller can pass `.data_ptr()` to a tilus global_view).
    return sf_perm.view(rows, k_blocks)


def quantize_nvfp4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize an FP32 tensor to NVFP4 with per-16-K-element UE4M3 scales.

    Quantization is applied along the last dimension only; any leading
    dimensions are preserved. Each block of 16 contiguous last-axis elements
    shares one UE4M3 scale factor; values within the block are FP4 (E2M1)
    after dividing by that scale.

    Parameters
    ----------
    x: torch.Tensor
        FP32 tensor of shape ``[..., K]`` with ``ndim >= 1``. ``K`` must be a
        multiple of :data:`NVFP4_SF_BLOCK_K` (= 16).

    Returns
    -------
    a_packed: torch.Tensor
        ``[..., K // 2]`` ``torch.uint8`` storage holding two packed FP4
        (E2M1) elements per byte. Pass ``a_packed.data_ptr()`` to a tilus
        kernel that views it as ``float4_e2m1``; or
        ``a_packed.view(torch.float4_e2m1fn_x2)`` for ``torch._scaled_mm``.
    sf: torch.Tensor
        ``[..., K // 16]`` ``torch.float8_e4m3fn`` per-block scale factors
        (UE4M3 from the kernel's perspective).

    Raises
    ------
    ValueError
        If ``x.ndim < 1`` or ``x.shape[-1]`` is not a multiple of 16.
    """
    import tilus  # local import to avoid a cycle

    if x.ndim < 1:
        raise ValueError(
            f"quantize_nvfp4 expects ndim >= 1, got shape {tuple(x.shape)}"
        )
    K = x.shape[-1]
    if K % NVFP4_SF_BLOCK_K != 0:
        raise ValueError(
            f"x.shape[-1] (= {K}) must be a multiple of "
            f"NVFP4_SF_BLOCK_K (= {NVFP4_SF_BLOCK_K})"
        )
    leading = x.shape[:-1]
    n_blocks = K // NVFP4_SF_BLOCK_K

    # Per-block amax → choose UE4M3 scale s.t. dequant(quant(v)) ≈ v.
    blocks = x.reshape(*leading, n_blocks, NVFP4_SF_BLOCK_K)
    block_amax = blocks.abs().amax(dim=-1).clamp(min=1e-12)        # [..., K/16]
    sf_fp32 = block_amax / _FP4_E2M1_MAX                           # scale per block
    sf = sf_fp32.to(torch.float8_e4m3fn)                           # round to UE4M3

    # Re-read sf as FP32 (gives us the *quantized* scale we'll actually use).
    sf_dequant = sf.to(torch.float32).clamp(min=1e-12)

    # Quantize: v / sf → round to FP4 set. Use tilus' FP32→FP4 cast so the
    # representable set matches what the kernel sees.
    scaled = blocks / sf_dequant.unsqueeze(-1)                     # [..., K/16, 16]
    scaled_fp4 = (
        tilus.from_torch(scaled.reshape(*leading, K))
        .to(float4_e2m1).to(float32).torch()
    )                                                              # [..., K] fp32
    a_packed = (
        tilus.from_torch(scaled_fp4).to(float4_e2m1).storage
    )
    return a_packed, sf


def dequantize_nvfp4(a_packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`quantize_nvfp4` — reconstruct an FP32 reference tensor.

    Operates along the last dimension; any leading dimensions are preserved
    and must match between ``a_packed`` and ``sf``.

    Parameters
    ----------
    a_packed: torch.Tensor
        ``[..., K // 2]`` ``torch.uint8`` storage holding two packed FP4
        (E2M1) elements per byte. Must have ``ndim >= 1``.
    sf: torch.Tensor
        ``[..., K // 16]`` ``torch.float8_e4m3fn`` per-block scale factors,
        as returned by :func:`quantize_nvfp4`. Leading dims must equal
        those of ``a_packed``.

    Returns
    -------
    x: torch.Tensor
        ``[..., K]`` FP32 tensor where each block of 16 contiguous last-axis
        elements has been scaled by its corresponding block scale factor.

    Raises
    ------
    ValueError
        If either input has ``ndim < 1`` or their shapes are inconsistent.
    """
    import tilus  # local import to avoid a cycle

    if a_packed.ndim < 1 or sf.ndim < 1:
        raise ValueError(
            f"dequantize_nvfp4 expects ndim >= 1, got "
            f"a_packed.shape={tuple(a_packed.shape)}, sf.shape={tuple(sf.shape)}"
        )
    leading = a_packed.shape[:-1]
    K_half = a_packed.shape[-1]
    K = K_half * 2
    expected_sf_shape = (*leading, K // NVFP4_SF_BLOCK_K)
    if tuple(sf.shape) != expected_sf_shape:
        raise ValueError(
            f"SF shape mismatch: expected {expected_sf_shape}, "
            f"got {tuple(sf.shape)}"
        )

    a_fp32 = (
        tilus.from_torch(a_packed.view(torch.uint8))
        .view(float4_e2m1)
        .to(float32).torch()
    )                                                              # [..., K]
    sf_fp32 = sf.to(torch.float32)                                 # [..., K/16]
    return (
        a_fp32.reshape(*leading, K // NVFP4_SF_BLOCK_K, NVFP4_SF_BLOCK_K)
        * sf_fp32.unsqueeze(-1)
    ).reshape(*leading, K)
