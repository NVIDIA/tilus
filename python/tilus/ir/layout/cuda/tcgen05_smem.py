from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Sequence

import numpy as np
from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, Var
from hidet.ir.type import DataType
from hidet.utils import prod

from tilus.extensions.hidet.ir.utils.index_transform import index_deserialize
from tilus.ir.layout.shared_layout import SharedLayout
from tilus.ir.layout.utils.visualize import visualize_layout
from tilus.ir.utils.veceval import meshgrid, vectorized_evaluate
from tilus.ir.utils.vector import vector


class Tcgen05SwizzleMode(Enum):
    """TCGen05 swizzle modes corresponding to cute Swizzle parameters"""

    NO_SWIZZLE = (0, 0, 0)  # No swizzling or Interleaved
    B32_SWIZZLE = (1, 4, 3)  # 32B Swizzling: Swizzle<1, 4, 3>
    B64_SWIZZLE = (2, 4, 3)  # 64B Swizzling: Swizzle<2, 4, 3>
    B128_SWIZZLE = (3, 4, 3)  # 128B Swizzling: Swizzle<3, 4, 3>

    def __init__(self, bbits: int, mbase: int, sshift: int):
        self.bbits = bbits
        self.mbase = mbase
        self.sshift = sshift


@dataclass
class CanonicalSharedLayout:
    shape: tuple[int, int]  # (mn-size, k-size)
    major_kind: Literal["MN", "K"]
    swizzle_mode: Tcgen05SwizzleMode
    sbo: int
    lbo: int
    m: int
    k: int
    t: int

    def __post_init__(self):
        atom_size = 2**self.swizzle_mode.bbits * 8 * self.t
        if (self.m > 1 and self.sbo % atom_size != 0) or (self.k > 1 and self.lbo % atom_size != 0):
            raise ValueError(f"SBO {self.sbo} and LBO {self.lbo} must be divisible by atom size: {atom_size}")

    def __eq__(self, other: CanonicalSharedLayout) -> bool:
        return (
            self.shape == other.shape
            and self.major_kind == other.major_kind
            and self.swizzle_mode == other.swizzle_mode
            and self.sbo == other.sbo
            and self.lbo == other.lbo
            and self.m == other.m
            and self.k == other.k
            and self.t == other.t
        )


def _generate_atom_grid(major_kind: Literal["MN", "K"], swizzle_mode: Tcgen05SwizzleMode, t: int) -> SharedLayout:
    m = k = 1
    swizzle_factor = 2**swizzle_mode.bbits
    if major_kind == "MN":
        shape = (t * swizzle_factor * m, 8 * k)
    else:
        shape = (8 * m, t * swizzle_factor * k)
    canonical_layout = CanonicalSharedLayout(
        shape=shape, major_kind=major_kind, swizzle_mode=swizzle_mode, sbo=1, lbo=1, m=m, k=k, t=t
    )
    return get_shared_layout_from_canonical(canonical_layout)


def canonicalize_shared_layout(shared_layout: SharedLayout, dtype: DataType) -> Optional[CanonicalSharedLayout]:
    """
    Reverse engineer a SharedLayout to determine its canonical TCGen05 form using direct pattern analysis.

    This function uses your proposed approach:
    1. Determine swizzle mode and majorness by analyzing stride patterns
    2. Extract SBO/LBO by analyzing box patterns after identifying the atom structure

    Parameters
    ----------
    shared_layout : SharedLayout
        The shared layout to canonicalize
    dtype : DataType
        The data type of the tensor elements, used to determine T = 128 / dtype.nbits

    Returns
    -------
    Optional[CanonicalSharedLayout]
        The canonical form if found, None otherwise
    """
    if len(shared_layout.shape) != 2:
        return None

    shape_m, shape_k = shared_layout.shape

    # Calculate T from dtype: T = 128 / dtype.nbits
    if 128 % dtype.nbits != 0:
        return None
    t = 128 // dtype.nbits

    # Create meshgrid for the entire layout
    grid = meshgrid(shared_layout.shape)
    entire_grid = vectorized_evaluate(
        expr=shared_layout.offset, var2value={axis: grid[i] for i, axis in enumerate(shared_layout.axes)}
    )

    # Try each swizzle mode and majorness using direct pattern analysis
    swizzle_modes = [
        Tcgen05SwizzleMode.NO_SWIZZLE,
        Tcgen05SwizzleMode.B32_SWIZZLE,
        Tcgen05SwizzleMode.B64_SWIZZLE,
        Tcgen05SwizzleMode.B128_SWIZZLE,
    ]

    for major_kind in ["MN", "K"]:
        for swizzle_mode in swizzle_modes:
            # Generate atom layout and get its offset grid
            atom_layout = _generate_atom_grid(major_kind, swizzle_mode, t)
            atom_axes_grid = meshgrid(atom_layout.shape)
            atom_grid = vectorized_evaluate(
                expr=atom_layout.offset, var2value={axis: atom_axes_grid[i] for i, axis in enumerate(atom_layout.axes)}
            )

            atom_shape = atom_layout.shape
            entire_shape = shared_layout.shape

            # Check if the entire grid can be divided into atoms
            if entire_shape[0] % atom_shape[0] != 0 or entire_shape[1] % atom_shape[1] != 0:
                continue

            # Check if the top-left corner matches the atom pattern
            if not np.array_equal(entire_grid[: atom_shape[0], : atom_shape[1]], atom_grid):
                continue

            m = entire_shape[0] // atom_shape[0]
            k = entire_shape[1] // atom_shape[1]

            entire_grid = entire_grid.reshape(m, atom_shape[0], k, atom_shape[1]).permute(0, 2, 1, 3)
            level_grid = entire_grid - atom_grid

            # check if all elements in the last two dimensions are the same
            if np.any(np.min(level_grid, axis=(2, 3)) != np.max(level_grid, axis=(2, 3))):
                return None

            # Extract SBO and LBO from box differences
            sbo = level_grid[1, 0, 0, 0] if m > 1 else 1
            lbo = level_grid[0, 1, 0, 0] if k > 1 else 1

            return CanonicalSharedLayout(
                shape=(shape_m, shape_k),
                major_kind=major_kind,
                swizzle_mode=swizzle_mode,
                sbo=sbo,
                lbo=lbo,
                m=m,
                k=k,
                t=t,
            )

    return None


def get_shared_layout_from_canonical(canonical_layout: CanonicalSharedLayout) -> SharedLayout:
    """
    Construct the shared layout specified by the canonical layout of tcgen05 cp instructions.

    This table provides a reference for different swizzling modes and their corresponding
    canonical layouts, categorized by major-ness.

    +----------------+--------------------------+--------------------------------------+-------------------------------------+
    | Major-ness     | Swizzling mode           | Canonical Layout without swizzling   | Swizzling on the previous column    |
    +================+==========================+======================================+=====================================+
    | MN- major      | No-swizzling or          | ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO)) | Swizzle<0, 4, 3>                    |
    |                | Interleaved              |                                      |                                     |
    +----------------+--------------------------+--------------------------------------+-------------------------------------+
    |                | 32B Swizzling            | ((T,2,m),(8,k)):((1,T,LBO),(2T,SBO)) | Swizzle<1, 4, 3>                    |
    +----------------+--------------------------+--------------------------------------+-------------------------------------+
    |                | 64B Swizzling            | ((T,4,m),(8,k)):((1,T,LBO),(4T,SBO)) | Swizzle<2, 4, 3>                    |
    +----------------+--------------------------+--------------------------------------+-------------------------------------+
    |                | 128B Swizzling           | ((T,8,m),(8,k)):((1,T,LBO),(8T,SBO)) | Swizzle<3, 4, 3>                    |
    +================+==========================+======================================+=====================================+
    | K- major       | No-swizzling or          | ((8,m),(T,k)):((1T,SBO),(1,LBO))     | Swizzle<0, 4, 3>                    |
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
    (The table can be found at https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-canonical-layouts)

    Returns
    -------
    SharedLayout
        The shared memory layout for TCGen05
    """
    bbits, mbase, sshift = (
        canonical_layout.swizzle_mode.bbits,
        canonical_layout.swizzle_mode.mbase,
        canonical_layout.swizzle_mode.sshift,
    )
    t, m, k, sbo, lbo = (
        canonical_layout.t,
        canonical_layout.m,
        canonical_layout.k,
        canonical_layout.sbo,
        canonical_layout.lbo,
    )

    # Determine swizzle factor based on bbits (from the canonical layout table)
    swizzle_factor = 2**bbits

    # Calculate the logical shape based on cute layout interpretation and major-ness
    if canonical_layout.major_kind == "MN":
        # MN-major: Shape = ((T, swizzle_factor, m), (8, k))
        # This flattens to (T * swizzle_factor * m, 8 * k)
        shape = (t * swizzle_factor * m, 8 * k)
    elif canonical_layout.major_kind == "K":
        # K-major: Shape = ((8, m), (T, swizzle_factor * k))
        # This flattens to (8 * m, T * swizzle_factor * k)
        shape = (8 * m, t * swizzle_factor * k)
    else:
        raise ValueError(f"Unsupported major_kind: {canonical_layout.major_kind}")

    def f_offset(axes: Sequence[Var]) -> Expr:
        i, j = axes[0], axes[1]

        if canonical_layout.major_kind == "MN":
            # ((T,sf,m),(8,k)):((1,T,SBO),(sf*T,LBO))
            i0, i1, i2 = index_deserialize(i, (t, swizzle_factor, m), ranks=[2, 1, 0])
            j0, j1 = index_deserialize(j, (8, k), ranks=[1, 0])
            base_offset = sum(vector([i0, i1, i2]) * vector([1, t, sbo])) + sum(
                vector([j0, j1]) * vector([swizzle_factor * t, lbo])
            )
        elif canonical_layout.major_kind == "K":
            # ((8,m),(T,sf*k)):((sf*T,SBO),(1,T, LBO))

            i0 = i % 8  # 8 dimension
            i1 = i // 8  # m dimension

            j0 = j % t  # T dimension
            j_temp = j // t
            j1 = j_temp % swizzle_factor  # swizzle_factor dimension
            j2 = j_temp // swizzle_factor  # k dimension, ignore for k-major

            assert swizzle_factor * t == shape[1]

            # Apply stride patterns from the canonical layout table for K-major
            if bbits == 0:
                # ((8,m),(T,k)):((1T,SBO),(1,LBO))
                # Strides: ((T*SBO, 1), (1, LBO))
                base_offset = i0 * t * sbo + i1 * 1 + j0 * 1 + j1 * lbo
            else:
                # For swizzled layouts: ((8,m),(T,swizzle_factor*k)):((swizzle_factor*T,SBO),(1,T))
                # Strides: ((swizzle_factor*T*SBO, 1), (1, T))
                base_offset = i0 * swizzle_factor * t + i1 * 1 + j0 * 1 + j1 * t + j2 * t

        # Apply cute swizzle transformation if needed
        if bbits > 0:
            # Apply cute swizzle: offset ^ shiftr(offset & yyy_msk, msk_sft)
            # where yyy_msk = bit_msk << (mbase + sshift)
            # and bit_msk = (1 << bbits) - 1
            bit_msk = (1 << bbits) - 1
            yyy_msk = bit_msk << (mbase + sshift)

            # Implement the swizzle transformation
            # offset ^ shiftr(offset & yyy_msk, sshift)
            swizzle_offset = base_offset ^ ((base_offset & yyy_msk) >> sshift)
            return swizzle_offset
        else:
            return base_offset

    return SharedLayout.create(shape=shape, size=prod(shape), f_offset=f_offset)


def main():
    original_canonical = CanonicalSharedLayout(
        shape=(16, 16), major_kind="MN", swizzle_mode=Tcgen05SwizzleMode.NO_SWIZZLE, sbo=64, lbo=128, m=2, k=2, t=8
    )

    layout = get_shared_layout_from_canonical(original_canonical)
    print(visualize_layout(layout))

    recovered_canonical = canonicalize_shared_layout(layout, int32)

    assert recovered_canonical is not None
    assert recovered_canonical == original_canonical


if __name__ == "__main__":
    main()
