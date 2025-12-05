# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, cast

import numpy as np
from hidet.ir.type import DataType
from tilus.extensions.hidet.ir.primitives.cuda.tcgen05 import Tcgen05SwizzleMode
from tilus.ir.layout.shared_layout import SharedLayout
from tilus.ir.layout.utils.cute import CuteLayout, CuteSwizzle, IntTuple, SwizzledCuteLayout, cute_layout, tuple_product
from tilus.utils import floor_log2

# class Tcgen05SwizzleMode(Enum):
#     """TCGen05 swizzle modes corresponding to cute Swizzle parameters"""

#     NO_SWIZZLE = (0, 0, 0)  # No swizzling or Interleaved
#     B32_SWIZZLE = (1, 4, 3)  # 32B Swizzling: Swizzle<1, 4, 3>
#     B64_SWIZZLE = (2, 4, 3)  # 64B Swizzling: Swizzle<2, 4, 3>
#     B128_SWIZZLE = (3, 4, 3)  # 128B Swizzling: Swizzle<3, 4, 3>

#     def encode(self) -> int:
#         # see https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-desc-layout
#         return {
#             Tcgen05SwizzleMode.NO_SWIZZLE: 0,
#             Tcgen05SwizzleMode.B32_SWIZZLE: 6,
#             Tcgen05SwizzleMode.B64_SWIZZLE: 4,
#             Tcgen05SwizzleMode.B128_SWIZZLE: 2,
#         }[self]

#     @property
#     def bbits(self) -> int:
#         return self.value[0]

#     @property
#     def mbase(self) -> int:
#         return self.value[1]

#     @property
#     def sshift(self) -> int:
#         return self.value[2]

#     def as_cute_swizzle(self) -> CuteSwizzle:
#         bbits, mbase, sshift = self.value
#         return CuteSwizzle(bbits=bbits, mbase=mbase, sshift=sshift)


def as_cute_swizzle(swizzle_mode: Tcgen05SwizzleMode) -> CuteSwizzle:
    if swizzle_mode == Tcgen05SwizzleMode.NO_SWIZZLE:
        return CuteSwizzle(bbits=0, mbase=0, sshift=0)
    elif swizzle_mode == Tcgen05SwizzleMode.B32_SWIZZLE:
        return CuteSwizzle(bbits=1, mbase=4, sshift=3)
    elif swizzle_mode == Tcgen05SwizzleMode.B64_SWIZZLE:
        return CuteSwizzle(bbits=2, mbase=4, sshift=3)
    elif swizzle_mode == Tcgen05SwizzleMode.B128_SWIZZLE:
        return CuteSwizzle(bbits=3, mbase=4, sshift=3)
    else:
        raise ValueError(f"Unsupported swizzle mode: {swizzle_mode}")


@dataclass(order=True, eq=True, unsafe_hash=True)
class CanonicalSharedLayout:
    """
    The canonical layout of tcgen05 cp instructions.

    +----------------+--------------------------+--------------------------------------+-------------------------------------+
    | Major-ness     | Swizzling mode           | Canonical Layout without swizzling   | Swizzling on the previous column    |
    +================+==========================+======================================+=====================================+
    | MN-major       | No-swizzling or          | ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO)) | Swizzle<0, 4, 3>                    |
    |                | Interleaved              |                                      |                                     |
    +----------------+--------------------------+--------------------------------------+-------------------------------------+
    |                | 32B Swizzling            | ((T,2,m),(8,k)):((1,T,LBO),(2T,SBO)) | Swizzle<1, 4, 3>                    |
    +----------------+--------------------------+--------------------------------------+-------------------------------------+
    |                | 64B Swizzling            | ((T,4,m),(8,k)):((1,T,LBO),(4T,SBO)) | Swizzle<2, 4, 3>                    |
    +----------------+--------------------------+--------------------------------------+-------------------------------------+
    |                | 128B Swizzling           | ((T,8,m),(8,k)):((1,T,LBO),(8T,SBO)) | Swizzle<3, 4, 3>                    |
    +================+==========================+======================================+=====================================+
    | K-major        | No-swizzling or          | ((8,m),(T,1,k)):((1T,SBO),(1,T,LBO)) | Swizzle<0, 4, 3>                    |
    |                | Interleaved              |                                      |                                     |
    +----------------+--------------------------+--------------------------------------+-------------------------------------+
    |                | 32B Swizzling            | ((8,m),(T,2,k)):((2T,SBO),(1,T,LBO)) | Swizzle<1, 4, 3>                    |
    +----------------+--------------------------+--------------------------------------+-------------------------------------+
    |                | 64B Swizzling            | ((8,m),(T,4,k)):((4T,SBO),(1,T,LBO)) | Swizzle<2, 4, 3>                    |
    +----------------+--------------------------+--------------------------------------+-------------------------------------+
    |                | 128B Swizzling           | ((8,m),(T,8,k)):((8T,SBO),(1,T,LBO)) | Swizzle<3, 4, 3>                    |
    +----------------+--------------------------+--------------------------------------+-------------------------------------+
    where
    - T = 128 / sizeof-elements-in-bits T represents scale factor which normalizes matrix element types to 128-bits.
    - m represents the number of repeating patterns across rows.
    - k represents the number of repeating patterns across columns.
    (The table is a generalization of the table in https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-canonical-layouts.)
    """

    major_kind: Literal["MN", "K"]
    swizzle_mode: Tcgen05SwizzleMode
    SBO: int
    LBO: int
    m: int
    k: int
    T: int

    def __post_init__(self):
        atom_size = 2 ** as_cute_swizzle(self.swizzle_mode).bbits * 8 * self.T
        if self.major_kind == "MN":
            if self.swizzle_mode == Tcgen05SwizzleMode.NO_SWIZZLE:
                m_stride = self.SBO
                k_stride = self.LBO
            else:
                m_stride = self.LBO
                k_stride = self.SBO
        else:
            m_stride = self.SBO
            k_stride = self.LBO
        if (self.m > 1 and m_stride % atom_size != 0) or (self.k > 1 and k_stride % atom_size != 0):
            raise ValueError(f"m_stride {m_stride} and k_stride {k_stride} must be divisible by atom size: {atom_size}")

    @property
    def S(self) -> int:
        return 2 ** as_cute_swizzle(self.swizzle_mode).bbits

    @property
    def dtype_nbits(self) -> int:
        return 128 // self.T

    @property
    def swizzled_cute_layout(self) -> SwizzledCuteLayout:
        shape: IntTuple
        strides: IntTuple
        if self.major_kind == "MN":
            shape = ((self.T, self.S, self.m), (8, self.k))
            if self.swizzle_mode == Tcgen05SwizzleMode.NO_SWIZZLE:
                strides = ((1, self.T, self.SBO), (self.T, self.LBO))
            else:
                strides = ((1, self.T, self.LBO), (self.T, self.SBO))
        else:
            shape = ((8, self.m), (self.T, self.S, self.k))
            strides = ((self.S * self.T, self.SBO), (1, self.T, self.LBO))
        swizzle = as_cute_swizzle(self.swizzle_mode)
        return SwizzledCuteLayout(CuteLayout(shape, strides), swizzle)

    def as_shared_layout(self) -> SharedLayout:
        return get_shared_layout_from_canonical(self)


def _generate_atom_grid(major_kind: Literal["MN", "K"], swizzle_mode: Tcgen05SwizzleMode, t: int) -> np.ndarray:
    canonical_layout = CanonicalSharedLayout(
        major_kind=major_kind, swizzle_mode=swizzle_mode, SBO=0, LBO=0, m=1, k=1, T=t
    )
    atom_layout = get_shared_layout_from_canonical(canonical_layout)
    return atom_layout.as_numpy_grid()


def canonicalize_shared_layout(shared_layout: SharedLayout, dtype: DataType) -> Optional[CanonicalSharedLayout]:
    """
    Convert a SharedLayout to its canonical TCGen05 form using direct pattern analysis.

    See the docstring of get_shared_layout_from_canonical for the canonical layout pattern.

    Parameters
    ----------
    shared_layout : SharedLayout
        The shared layout to canonicalize
    dtype : DataType
        The data type of the tensor elements, used to determine T = 128 / dtype.nbits. What only matters is the number of bits of the data type.

    Returns
    -------
    ret: Optional[CanonicalSharedLayout]
        The canonical form if found, None otherwise
    """
    if len(shared_layout.shape) != 2:
        return None

    # Calculate T from dtype: T = 128 / dtype.nbits
    if 128 % dtype.nbits != 0:
        return None
    T = 128 // dtype.nbits

    # Create meshgrid for the entire layout
    entire_grid = shared_layout.as_numpy_grid()
    entire_shape = shared_layout.shape

    # Try each swizzle mode and majorness using direct pattern analysis
    for major_kind in ["MN", "K"]:
        for swizzle_mode in [
            Tcgen05SwizzleMode.NO_SWIZZLE,
            Tcgen05SwizzleMode.B32_SWIZZLE,
            Tcgen05SwizzleMode.B64_SWIZZLE,
            Tcgen05SwizzleMode.B128_SWIZZLE,
        ]:
            # Generate atom layout and get its offset grid
            atom_grid = _generate_atom_grid(cast(Literal["MN", "K"], major_kind), swizzle_mode, T)
            atom_shape = atom_grid.shape

            # Check if the entire grid can be divided into atoms
            if entire_shape[0] % atom_shape[0] != 0 or entire_shape[1] % atom_shape[1] != 0:
                continue

            # Check if the top-left corner matches the atom pattern
            if not np.array_equal(entire_grid[: atom_shape[0], : atom_shape[1]], atom_grid):
                continue

            m = entire_shape[0] // atom_shape[0]
            k = entire_shape[1] // atom_shape[1]

            entire_grid = entire_grid.reshape(m, atom_shape[0], k, atom_shape[1]).transpose(0, 2, 1, 3)
            level_grid = entire_grid - atom_grid

            # check if all elements in the last two dimensions are the same
            if not np.array_equal(np.min(level_grid, axis=(2, 3)), np.max(level_grid, axis=(2, 3))):
                return None

            # Extract SBO and LBO from box differences
            ROW_STRIDE = int(level_grid[1, 0, 0, 0]) if m > 1 else 1
            COL_STRIDE = int(level_grid[0, 1, 0, 0]) if k > 1 else 1

            if major_kind == "MN":
                if swizzle_mode == Tcgen05SwizzleMode.NO_SWIZZLE:
                    SBO = ROW_STRIDE
                    LBO = COL_STRIDE
                else:
                    SBO = COL_STRIDE
                    LBO = ROW_STRIDE
            else:
                SBO = ROW_STRIDE
                LBO = COL_STRIDE

            return CanonicalSharedLayout(
                major_kind=cast(Literal["MN", "K"], major_kind),
                swizzle_mode=swizzle_mode,
                SBO=SBO,
                LBO=LBO,
                m=m,
                k=k,
                T=T,
            )

    return None


def get_shared_layout_from_canonical(canonical_layout: CanonicalSharedLayout) -> SharedLayout:
    """
    Construct the shared layout specified by the canonical layout of tcgen05 cp instructions.

    Parameters
    ----------
    canonical_layout : CanonicalSharedLayout
        The canonical layout to construct the shared layout from.

    Returns
    -------
    ret: SharedLayout
        The shared memory layout of Tilus corresponding to the canonical layout.
    """
    swizzle_mode = canonical_layout.swizzle_mode
    cute_swizzle = as_cute_swizzle(swizzle_mode)
    bbits, mbase, sshift = cute_swizzle.bbits, cute_swizzle.mbase, cute_swizzle.sshift
    T, m, k, SBO, LBO = (
        canonical_layout.T,
        canonical_layout.m,
        canonical_layout.k,
        canonical_layout.SBO,
        canonical_layout.LBO,
    )

    # Determine swizzle factor based on bbits (from the canonical layout table)
    S = 2**bbits

    # Calculate the logical shape based on cute layout interpretation and major-ness
    if canonical_layout.major_kind == "MN":
        if swizzle_mode == Tcgen05SwizzleMode.NO_SWIZZLE:
            layout = cute_layout(shape=((T, m), (8, k)), strides=((1, SBO), (T, LBO)))
        else:
            layout = cute_layout(shape=((T, S, m), (8, k)), strides=((1, T, LBO), (S * T, SBO)))
    elif canonical_layout.major_kind == "K":
        if swizzle_mode == Tcgen05SwizzleMode.NO_SWIZZLE:
            layout = cute_layout(shape=((8, m), (T, k)), strides=((T, SBO), (1, LBO)))
        else:
            layout = cute_layout(shape=((8, m), (T, S, k)), strides=((S * T, SBO), (1, T, LBO)))
    else:
        raise ValueError(f"Unsupported major_kind: {canonical_layout.major_kind}")

    nbytes = 16 // canonical_layout.T
    swizzle = CuteSwizzle(bbits=bbits, mbase=mbase - floor_log2(nbytes), sshift=sshift)
    swizzled_cute_layout = SwizzledCuteLayout(layout, swizzle)

    assert isinstance(layout.shape, Sequence)
    shape = [int(tuple_product(item)) for item in layout.shape]

    return swizzled_cute_layout.as_shared_layout(shape)


def generate_canonical_layout(
    shape: tuple[int, int], dtype: DataType, major_kind: Literal["MN", "K"], swizzle_mode: Tcgen05SwizzleMode
) -> CanonicalSharedLayout:
    if 128 % dtype.nbits != 0:
        raise ValueError(f"dtype {dtype.name} is not supported")
    T = 128 // dtype.nbits
    S = 2 ** as_cute_swizzle(swizzle_mode).bbits
    if major_kind == "MN":
        if shape[0] % (T * S) != 0 or shape[1] % 8 != 0:
            raise ValueError(f"shape {shape} is not supported")
        m, k = shape[0] // (T * S), shape[1] // 8
        atom_size = T * S * 8
        if swizzle_mode == Tcgen05SwizzleMode.NO_SWIZZLE:
            SBO = atom_size
            LBO = atom_size * m
        else:
            SBO = atom_size
            LBO = atom_size * k
    else:
        if shape[0] % 8 != 0 or shape[1] % (T * S) != 0:
            raise ValueError(f"shape {shape} is not supported")
        m, k = shape[0] // 8, shape[1] // (T * S)
        atom_size = T * S * 8
        SBO = atom_size
        LBO = atom_size * m

    return CanonicalSharedLayout(major_kind=major_kind, swizzle_mode=swizzle_mode, SBO=SBO, LBO=LBO, m=m, k=k, T=T)
