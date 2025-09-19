from enum import Enum
from typing import Sequence

from hidet.ir.expr import Expr, Var

from tilus.ir.layout.shared_layout import SharedLayout


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


def get_tcgen05_smem_layout(T: int, m: int, k: int, sbo: int, lbo: int, swizzle: tuple[int, int, int]) -> SharedLayout:
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
    | K- major       | No-swizzling or          | ((8,m),(T,2k)):((1T,SBO),(1,LBO))    | Swizzle<0, 4, 3>                    |
    |                | Interleaved              |                                      |                                     |
    +----------------+--------------------------+--------------------------------------+-------------------------------------+
    |                | 32B Swizzling            | ((8,m),(T,2k)):((2T,SBO),(1,T))      | Swizzle<1, 4, 3>                    |
    +----------------+--------------------------+--------------------------------------+-------------------------------------+
    |                | 64B Swizzling            | ((8,m),(T,2k)):((4T,SBO),(1,T))      | Swizzle<2, 4, 3>                    |
    +----------------+--------------------------+--------------------------------------+-------------------------------------+
    |                | 128B Swizzling           | ((8,m),(T,2k)):((8T,SBO),(1,T))      | Swizzle<3, 4, 3>                    |
    +----------------+--------------------------+--------------------------------------+-------------------------------------+
    where
    - T = 128 / sizeof-elements-in-bits T represents scale factor which normalizes matrix element types to 128-bits.
    - m represents the number of repeating patterns across rows.
    - k represents the number of repeating patterns across columns.
    (The table can be found at https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-canonical-layouts)

    Parameters
    ----------
    T : int
        Scale factor (128 / sizeof-elements-in-bits)
    m : int
        Number of repeating patterns across rows
    k : int
        Number of repeating patterns across columns
    sbo : int
        Small batch offset
    lbo : int
        Large batch offset
    swizzle : tuple[int, int, int]
        Swizzle parameters (BBits, MBase, SShift) from cute Swizzle

    Returns
    -------
    SharedLayout
        The shared memory layout for TCGen05
    """
    bbits, mbase, sshift = swizzle

    # Determine swizzle factor based on bbits (from the canonical layout table)
    if bbits == 0:
        # No swizzling: ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO))
        swizzle_factor = 1
    elif bbits == 1:
        # 32B Swizzling: ((T,2,m),(8,k)):((1,T,LBO),(2T,SBO))
        swizzle_factor = 2
    elif bbits == 2:
        # 64B Swizzling: ((T,4,m),(8,k)):((1,T,LBO),(4T,SBO))
        swizzle_factor = 4
    elif bbits == 3:
        # 128B Swizzling: ((T,8,m),(8,k)):((1,T,LBO),(8T,SBO))
        swizzle_factor = 8
    else:
        raise ValueError(f"Unsupported swizzle bbits: {bbits}")

    # Calculate the logical shape based on cute layout interpretation
    # For MN-major: Shape = ((T, swizzle_factor, m), (8, k))
    # This flattens to (T * swizzle_factor * m, 8 * k)
    shape = (T * swizzle_factor * m, 8 * k)

    # Calculate total size
    size = T * swizzle_factor * m * 8 * k

    def f_offset(axes: Sequence[Var]) -> Expr:
        """
        Compute offset based on cute layout canonical patterns.

        The cute layout is interpreted as:
        Shape:  ((T, swizzle_factor, m), (8, k))
        Stride: ((stride_patterns from table), (stride_patterns from table))

        We need to map the 2D coordinate (i, j) to the hierarchical coordinate
        and then compute the inner product with strides.
        """
        i, j = axes[0], axes[1]

        # Convert flat coordinates to hierarchical coordinates
        # i -> (i0, i1, i2) where i = i0 + i1*T + i2*T*swizzle_factor
        # j -> (j0, j1) where j = j0 + j1*8

        i0 = i % T  # T dimension
        i_temp = i // T
        i1 = i_temp % swizzle_factor  # swizzle_factor dimension
        i2 = i_temp // swizzle_factor  # m dimension

        j0 = j % 8  # 8 dimension
        j1 = j // 8  # k dimension

        # Apply stride patterns from the canonical layout table
        if bbits == 0:
            # ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO))
            # Strides: ((1, T, SBO), (T, LBO))
            base_offset = i0 * 1 + i1 * T + i2 * (T * sbo) + j0 * (T * lbo) + j1 * lbo
        else:
            # For swizzled layouts: ((T,swizzle_factor,m),(8,k)):((1,T,LBO),(swizzle_factor*T,SBO))
            # Strides: ((1, T, LBO), (swizzle_factor*T, SBO))
            base_offset = i0 * 1 + i1 * T + i2 * (T * lbo) + j0 * (swizzle_factor * T * sbo) + j1 * sbo

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

    return SharedLayout.create(shape=shape, size=size, f_offset=f_offset)
