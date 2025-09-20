from enum import Enum
from typing import Sequence, Literal, Optional
from dataclasses import dataclass
import numpy as np
from hidet.ir.expr import Expr, Var
from hidet.ir.type import DataType
from hidet.utils import prod

from tilus.ir.layout.shared_layout import SharedLayout
from tilus.ir.utils.veceval import vectorized_evaluate, meshgrid
from tilus.extensions.hidet.ir.expr import index_vars


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


def canonicalize_shared_layout(shared_layout: SharedLayout, dtype: DataType) -> Optional[CanonicalSharedLayout]:
    """
    Reverse engineer a SharedLayout to determine its canonical TCGen05 form.
    
    This function uses a brute-force approach to find the canonical parameters
    that produce the same offset grid as the input layout. The dtype parameter
    is used to determine T = 128 / dtype.nbits, making the canonical form unique.
    
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
    # This makes the canonical form unique
    if 128 % dtype.nbits != 0:
        # T must be an integer, so dtype.nbits must divide 128
        return None
    t = 128 // dtype.nbits
    
    # Create meshgrid for the entire layout
    grid = meshgrid(shared_layout.shape)
    target_offset_grid = vectorized_evaluate(
        expr=shared_layout.offset, 
        var2value={axis: grid[i] for i, axis in enumerate(shared_layout.axes)}
    )
    
    # Try all possible canonical parameter combinations with the determined T
    swizzle_modes = [
        Tcgen05SwizzleMode.NO_SWIZZLE,
        Tcgen05SwizzleMode.B32_SWIZZLE,
        Tcgen05SwizzleMode.B64_SWIZZLE, 
        Tcgen05SwizzleMode.B128_SWIZZLE,
    ]
    
    for major_kind in ["MN", "K"]:
        for swizzle_mode in swizzle_modes:
            swizzle_factor = 2 ** swizzle_mode.bbits if swizzle_mode.bbits > 0 else 1
            
            # Calculate possible m and k values based on shape constraints
            if major_kind == "MN":
                # MN-major: shape = (t * swizzle_factor * m, 8 * k)
                if shape_m % (t * swizzle_factor) != 0 or shape_k % 8 != 0:
                    continue
                m = shape_m // (t * swizzle_factor)
                k = shape_k // 8
            else:  # K-major
                # K-major: shape = (8 * m, t * swizzle_factor * k)
                if shape_m % 8 != 0 or shape_k % (t * swizzle_factor) != 0:
                    continue
                m = shape_m // 8
                k = shape_k // (t * swizzle_factor)
            
            # Compute SBO and LBO directly from the atom layout and target grid
            sbo, lbo = _compute_sbo_lbo(major_kind, swizzle_mode, t, m, k, target_offset_grid)
            
            if sbo is not None and lbo is not None:
                # Create candidate canonical layout with computed SBO/LBO
                candidate = CanonicalSharedLayout(
                    shape=(shape_m, shape_k),
                    major_kind=major_kind,
                    swizzle_mode=swizzle_mode,
                    sbo=sbo,
                    lbo=lbo,
                    m=m,
                    k=k,
                    t=t
                )
                
                # Verify the candidate produces the same offset grid
                if _test_canonical_match(candidate, target_offset_grid):
                    return candidate
    
    return None


def _compute_sbo_lbo(major_kind: str, swizzle_mode: Tcgen05SwizzleMode, t: int, m: int, k: int, target_offset_grid: np.ndarray) -> tuple[int | None, int | None]:
    """
    Compute SBO and LBO directly by analyzing the offset differences in the target grid.
    
    The approach:
    1. Use specific coordinate points to extract the SBO and LBO contributions
    2. Based on the stride patterns from the canonical layout table, we can identify
       which coordinates will isolate the SBO and LBO terms
    
    Parameters
    ----------
    major_kind : str
        "MN" or "K"
    swizzle_mode : Tcgen05SwizzleMode
        The swizzle mode
    t : int
        The T parameter
    m : int
        Number of replications in the first dimension
    k : int
        Number of replications in the second dimension
    target_offset_grid : np.ndarray
        The target offset grid to match
        
    Returns
    -------
    tuple[int | None, int | None]
        (sbo, lbo) if successfully computed, (None, None) otherwise
    """
    bbits = swizzle_mode.bbits
    swizzle_factor = 2 ** bbits if bbits > 0 else 1
    
    try:
        if major_kind == "MN":
            # For MN-major layouts, we need to find coordinates that isolate SBO and LBO
            # From the stride patterns:
            # No swizzle: base_offset = i0*1 + i1*T + i2*(T*SBO) + j0*(T*LBO) + j1*LBO
            # Swizzled:   base_offset = i0*1 + i1*T + i2*(T*LBO) + j0*(swizzle_factor*T*SBO) + j1*SBO
            
            if bbits == 0:
                # No swizzle case: SBO affects i2 term, LBO affects j0 and j1 terms
                if m > 1:
                    # To isolate SBO: compare offsets where i2 differs by 1
                    # Use coordinates (T*swizzle_factor, 0) vs (0, 0)
                    coord1 = (t * swizzle_factor, 0)  # i2=1, others=0
                    coord2 = (0, 0)                   # i2=0, others=0
                    if coord1[0] < target_offset_grid.shape[0] and coord1[1] < target_offset_grid.shape[1]:
                        sbo = target_offset_grid[coord1] - target_offset_grid[coord2]
                        sbo = sbo // t  # Divide by T since stride is T*SBO
                    else:
                        sbo = 1
                else:
                    sbo = 1
                    
                if k > 1:
                    # To isolate LBO: compare offsets where j1 differs by 1
                    # Use coordinates (0, 8) vs (0, 0)
                    coord1 = (0, 8)  # j1=1, others=0
                    coord2 = (0, 0)  # j1=0, others=0
                    if coord1[0] < target_offset_grid.shape[0] and coord1[1] < target_offset_grid.shape[1]:
                        lbo = target_offset_grid[coord1] - target_offset_grid[coord2]
                    else:
                        lbo = 1
                else:
                    lbo = 1
                    
            else:
                # Swizzled case: LBO affects i2 term, SBO affects j0 and j1 terms
                if m > 1:
                    # To isolate LBO: compare offsets where i2 differs by 1
                    coord1 = (t * swizzle_factor, 0)  # i2=1, others=0
                    coord2 = (0, 0)                   # i2=0, others=0
                    if coord1[0] < target_offset_grid.shape[0] and coord1[1] < target_offset_grid.shape[1]:
                        lbo = target_offset_grid[coord1] - target_offset_grid[coord2]
                        lbo = lbo // t  # Divide by T since stride is T*LBO
                    else:
                        lbo = 1
                else:
                    lbo = 1
                    
                if k > 1:
                    # To isolate SBO: compare offsets where j1 differs by 1
                    coord1 = (0, 8)  # j1=1, others=0
                    coord2 = (0, 0)  # j1=0, others=0
                    if coord1[0] < target_offset_grid.shape[0] and coord1[1] < target_offset_grid.shape[1]:
                        sbo = target_offset_grid[coord1] - target_offset_grid[coord2]
                    else:
                        sbo = 1
                else:
                    sbo = 1
                    
        else:  # K-major
            # For K-major layouts:
            # No swizzle: base_offset = i0*(T*SBO) + i1*1 + j0*1 + j1*T + j2*LBO
            # Swizzled:   base_offset = i0*(swizzle_factor*T*SBO) + i1*1 + j0*1 + j1*T + j2*T
            
            if bbits == 0:
                # No swizzle case: SBO affects i0 term, LBO affects j2 term
                # base_offset = i0*(T*SBO) + i1*1 + j0*1 + j1*T + j2*LBO
                if m > 1:
                    # To isolate SBO: we need i0 to differ by 1, but coordinates are (i,j)
                    # where i = i0 + i1*8, so to get i0=1,i1=0 we use coordinate (1,0)
                    coord1 = (1, 0)  # i0=1, i1=0, others=0
                    coord2 = (0, 0)  # i0=0, i1=0, others=0
                    if coord1[0] < target_offset_grid.shape[0] and coord1[1] < target_offset_grid.shape[1]:
                        sbo = target_offset_grid[coord1] - target_offset_grid[coord2]
                        sbo = sbo // t  # Divide by T since stride is T*SBO
                    else:
                        sbo = 1
                else:
                    sbo = 1
                    
                if k > 1:
                    # To isolate LBO: we need j2 to differ by 1, but coordinates are (i,j)
                    # where j = j0 + j1*T + j2*T*swizzle_factor, so to get j2=1,j1=0,j0=0 we use coordinate (0, T*swizzle_factor)
                    coord1 = (0, t * swizzle_factor)  # j2=1, others=0
                    coord2 = (0, 0)                   # j2=0, others=0
                    if coord1[0] < target_offset_grid.shape[0] and coord1[1] < target_offset_grid.shape[1]:
                        lbo = target_offset_grid[coord1] - target_offset_grid[coord2]
                    else:
                        lbo = 1
                else:
                    lbo = 1
                    
            else:
                # Swizzled case: SBO affects i0 term, j2 term has fixed stride T
                if m > 1:
                    # To isolate SBO: we need i0 to differ by 1
                    coord1 = (1, 0)  # i0=1, i1=0, others=0
                    coord2 = (0, 0)  # i0=0, i1=0, others=0
                    if coord1[0] < target_offset_grid.shape[0] and coord1[1] < target_offset_grid.shape[1]:
                        sbo = target_offset_grid[coord1] - target_offset_grid[coord2]
                        sbo = sbo // (swizzle_factor * t)  # Divide by the stride coefficient
                    else:
                        sbo = 1
                else:
                    sbo = 1
                    
                # For swizzled K-major, LBO is not used in the stride pattern, set to 1
                lbo = 1
        
        return int(sbo), int(lbo)
        
    except Exception:
        return None, None


def _test_canonical_match(candidate: CanonicalSharedLayout, target_offset_grid: np.ndarray) -> bool:
    """
    Test if a candidate canonical layout produces the same offset grid as the target.
    
    Parameters
    ----------
    candidate : CanonicalSharedLayout
        The candidate canonical layout to test
    target_offset_grid : np.ndarray
        The target offset grid to match
        
    Returns
    -------
    bool
        True if the candidate produces the same offset grid, False otherwise
    """
    try:
        # Generate layout from candidate canonical form
        candidate_layout = get_shared_layout_from_canonical(candidate)
        
        # Generate offset grid from candidate layout
        candidate_grid = meshgrid(candidate_layout.shape)
        candidate_offset_grid = vectorized_evaluate(
            expr=candidate_layout.offset,
            var2value={axis: candidate_grid[i] for i, axis in enumerate(candidate_layout.axes)}
        )
        
        # Compare offset grids
        return np.array_equal(target_offset_grid, candidate_offset_grid)
    except:
        return False


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
    | K- major       | No-swizzling or          | ((8,m),(T, k)):((1T,SBO),(1,LBO))    | Swizzle<0, 4, 3>                    |
    |                | Interleaved              |                                      |                                     |
    +----------------+--------------------------+--------------------------------------+-------------------------------------+
    |                | 32B Swizzling            | ((8,m),(T,2 )):((2T,SBO),(1,T))      | Swizzle<1, 4, 3>                    |
    +----------------+--------------------------+--------------------------------------+-------------------------------------+
    |                | 64B Swizzling            | ((8,m),(T,4 )):((4T,SBO),(1,T))      | Swizzle<2, 4, 3>                    |
    +----------------+--------------------------+--------------------------------------+-------------------------------------+
    |                | 128B Swizzling           | ((8,m),(T,8 )):((8T,SBO),(1,T))      | Swizzle<3, 4, 3>                    |
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
    bbits, mbase, sshift = canonical_layout.swizzle_mode.bbits, canonical_layout.swizzle_mode.mbase, canonical_layout.swizzle_mode.sshift
    t, m, k, sbo, lbo = canonical_layout.t, canonical_layout.m, canonical_layout.k, canonical_layout.sbo, canonical_layout.lbo

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
        """
        Compute offset based on cute layout canonical patterns.

        The cute layout is interpreted differently for MN-major vs K-major:
        MN-major: Shape = ((T, swizzle_factor, m), (8, k))
        K-major:  Shape = ((8, m), (T, swizzle_factor * k))

        We need to map the 2D coordinate (i, j) to the hierarchical coordinate
        and then compute the inner product with strides.
        """
        i, j = axes[0], axes[1]

        if canonical_layout.major_kind == "MN":
            # MN-major coordinate decomposition
            # i -> (i0, i1, i2) where i = i0 + i1*T + i2*T*swizzle_factor
            # j -> (j0, j1) where j = j0 + j1*8
            
            i0 = i % t  # T dimension
            i_temp = i // t
            i1 = i_temp % swizzle_factor  # swizzle_factor dimension
            i2 = i_temp // swizzle_factor  # m dimension

            j0 = j % 8  # 8 dimension
            j1 = j // 8  # k dimension

            # Apply stride patterns from the canonical layout table for MN-major
            if bbits == 0:
                # ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO))
                # Strides: ((1, T, SBO), (T, LBO))
                base_offset = i0 * 1 + i1 * t + i2 * (t * sbo) + j0 * (t * lbo) + j1 * lbo
            else:
                # For swizzled layouts: ((T,swizzle_factor,m),(8,k)):((1,T,LBO),(swizzle_factor*T,SBO))
                # Strides: ((1, T, LBO), (swizzle_factor*T, SBO))
                base_offset = i0 * 1 + i1 * t + i2 * (t * lbo) + j0 * (swizzle_factor * t * sbo) + j1 * sbo

        elif canonical_layout.major_kind == "K":
            # K-major coordinate decomposition
            # i -> (i0, i1) where i = i0 + i1*8
            # j -> (j0, j1, j2) where j = j0 + j1*T + j2*T*swizzle_factor
            
            i0 = i % 8   # 8 dimension
            i1 = i // 8  # m dimension
            
            j0 = j % t  # T dimension
            j_temp = j // t
            j1 = j_temp % swizzle_factor  # swizzle_factor dimension
            j2 = j_temp // swizzle_factor  # k dimension

            # Apply stride patterns from the canonical layout table for K-major
            if bbits == 0:
                # ((8,m),(T,k)):((1T,SBO),(1,LBO))
                # Strides: ((T*SBO, 1), (1, LBO))
                base_offset = i0 * (t * sbo) + i1 * 1 + j0 * 1 + j1 * t + j2 * lbo
            else:
                # For swizzled layouts: ((8,m),(T,swizzle_factor*k)):((swizzle_factor*T,SBO),(1,T))
                # Strides: ((swizzle_factor*T*SBO, 1), (1, T))
                base_offset = i0 * (swizzle_factor * t * sbo) + i1 * 1 + j0 * 1 + j1 * t + j2 * t

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
