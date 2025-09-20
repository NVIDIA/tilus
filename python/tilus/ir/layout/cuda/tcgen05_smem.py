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
    target_offset_grid = vectorized_evaluate(
        expr=shared_layout.offset, 
        var2value={axis: grid[i] for i, axis in enumerate(shared_layout.axes)}
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
            result = _analyze_pattern_direct(
                target_offset_grid, major_kind, swizzle_mode, t, shape_m, shape_k
            )
            
            if result is not None:
                sbo, lbo, m, k = result
                
                # Create candidate canonical layout
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


def _analyze_pattern_direct(
    target_offset_grid: np.ndarray,
    major_kind: str,
    swizzle_mode: Tcgen05SwizzleMode,
    t: int,
    shape_m: int,
    shape_k: int
) -> Optional[tuple[int, int, int, int]]:
    """
    Directly analyze the target offset grid to determine if it matches the given
    swizzle mode and majorness, and extract SBO/LBO parameters.
    
    This implements your approach:
    1. Check if the shape is compatible with the swizzle mode and majorness
    2. Analyze stride patterns to confirm the pattern type
    3. Extract SBO/LBO from box differences
    
    Parameters
    ----------
    target_offset_grid : np.ndarray
        The target offset grid to analyze
    major_kind : str
        "MN" or "K"
    swizzle_mode : Tcgen05SwizzleMode
        The swizzle mode to test
    t : int
        The T parameter
    shape_m : int
        Target shape first dimension
    shape_k : int
        Target shape second dimension
        
    Returns
    -------
    Optional[tuple[int, int, int, int]]
        (sbo, lbo, m, k) if pattern matches, None otherwise
    """
    try:
        swizzle_factor = 2 ** swizzle_mode.bbits if swizzle_mode.bbits > 0 else 1
        
        # Step 1: Check shape compatibility and calculate m, k
        if major_kind == "MN":
            # MN-major: shape = (t * swizzle_factor * m, 8 * k)
            atom_h, atom_w = t * swizzle_factor, 8
            if shape_m % atom_h != 0 or shape_k % atom_w != 0:
                return None
            m, k = shape_m // atom_h, shape_k // atom_w
        else:  # K-major
            # K-major: shape = (8 * m, t * swizzle_factor * k)
            atom_h, atom_w = 8, t * swizzle_factor
            if shape_m % atom_h != 0 or shape_k % atom_w != 0:
                return None
            m, k = shape_m // atom_h, shape_k // atom_w
            
        # Step 2: Analyze stride patterns to confirm this is the right pattern type
        if not _verify_stride_pattern(target_offset_grid, major_kind, swizzle_mode, t, atom_h, atom_w):
            return None
            
        # Step 3: Extract SBO/LBO from box patterns
        sbo, lbo = _extract_sbo_lbo_from_grid(
            target_offset_grid, major_kind, swizzle_mode, t, m, k, atom_h, atom_w
        )
        
        if sbo is not None and lbo is not None:
            return sbo, lbo, m, k
            
        return None
        
    except Exception:
        return None


def _verify_stride_pattern(
    target_offset_grid: np.ndarray,
    major_kind: str,
    swizzle_mode: Tcgen05SwizzleMode,
    t: int,
    atom_h: int,
    atom_w: int
) -> bool:
    """
    Verify that the target grid follows the expected stride pattern for the given
    swizzle mode and majorness by checking key stride relationships.
    """
    try:
        target_h, target_w = target_offset_grid.shape
            
        # For MN-major layouts, check column-wise stride pattern
        if major_kind == "MN":
            # Check if consecutive rows have stride 1 (characteristic of MN-major)
            if target_h >= 2:
                row_stride = target_offset_grid[1, 0] - target_offset_grid[0, 0]
                if row_stride != 1:
                    return False
                    
        # For K-major layouts, check row-wise stride pattern  
        else:  # K-major
            # Check if consecutive columns have stride 1 (characteristic of K-major)
            if target_w >= 2:
                col_stride = target_offset_grid[0, 1] - target_offset_grid[0, 0]
                if col_stride != 1:
                    return False
                    
        return True
        
    except Exception:
        return False


def _extract_sbo_lbo_from_grid(
    target_offset_grid: np.ndarray,
    major_kind: str,
    swizzle_mode: Tcgen05SwizzleMode,
    t: int,
    m: int,
    k: int,
    atom_h: int,
    atom_w: int
) -> tuple[int | None, int | None]:
    """
    Extract SBO and LBO by analyzing the differences between atom repetitions.
    
    This follows your box approach: create a grid of box values where each box
    represents an atom repetition, then extract SBO/LBO from the box value patterns.
    """
    try:
        # Create box grid: each box represents one atom repetition
        box_grid = np.zeros((m, k), dtype=np.int64)
        
        for i in range(m):
            for j in range(k):
                # Get the top-left corner of each box (atom repetition)
                box_row = i * atom_h
                box_col = j * atom_w
                
                if box_row < target_offset_grid.shape[0] and box_col < target_offset_grid.shape[1]:
                    box_grid[i, j] = target_offset_grid[box_row, box_col]
                    
        # Subtract the base offset (box[0,0]) to get relative offsets
        if m > 0 and k > 0:
            base_offset = box_grid[0, 0]
            box_grid = box_grid - base_offset
            
        # Extract SBO/LBO based on the canonical layout patterns
        return _extract_sbo_lbo_from_boxes(box_grid, major_kind, swizzle_mode, t, m, k)
        
    except Exception:
        return None, None




def _extract_sbo_lbo_from_boxes(
    box_grid: np.ndarray, 
    major_kind: str, 
    swizzle_mode: Tcgen05SwizzleMode, 
    t: int, 
    m: int, 
    k: int
) -> tuple[int | None, int | None]:
    """
    Extract SBO and LBO from the box value patterns.
    
    The box values follow specific patterns based on the canonical layout:
    - MN-major no swizzle: box_value = i * (T * SBO) + j * LBO
    - MN-major swizzled: box_value = i * (T * LBO) + j * SBO  
    - K-major no swizzle: box_value = i * SBO + j * LBO
    - K-major swizzled: box_value = i * SBO (LBO not used)
    
    Parameters
    ----------
    box_grid : np.ndarray
        Grid of box values with shape (m, k)
    major_kind : str
        "MN" or "K"
    swizzle_mode : Tcgen05SwizzleMode
        The swizzle mode
    t : int
        The T parameter
    m : int
        Number of boxes in first dimension
    k : int
        Number of boxes in second dimension
        
    Returns
    -------
    tuple[int | None, int | None]
        (sbo, lbo) if successfully extracted, None otherwise
    """
    try:
        bbits = swizzle_mode.bbits
        
        if major_kind == "MN":
            if bbits == 0:
                # No swizzle: box_value = i * (T * SBO) + j * LBO
                if m > 1 and k > 1:
                    # Use differences to extract coefficients
                    # box_grid[1,0] - box_grid[0,0] = T * SBO
                    # box_grid[0,1] - box_grid[0,0] = LBO
                    sbo_coeff = box_grid[1, 0] - box_grid[0, 0]
                    lbo = box_grid[0, 1] - box_grid[0, 0]
                    sbo = sbo_coeff // t
                elif m > 1:
                    # Only i dimension available
                    sbo_coeff = box_grid[1, 0] - box_grid[0, 0]
                    sbo = sbo_coeff // t
                    lbo = 1  # Default
                elif k > 1:
                    # Only j dimension available
                    lbo = box_grid[0, 1] - box_grid[0, 0]
                    sbo = 1  # Default
                else:
                    sbo, lbo = 1, 1  # Single box
            else:
                # Swizzled: box_value = i * (T * LBO) + j * SBO
                if m > 1 and k > 1:
                    # box_grid[1,0] - box_grid[0,0] = T * LBO
                    # box_grid[0,1] - box_grid[0,0] = SBO
                    lbo_coeff = box_grid[1, 0] - box_grid[0, 0]
                    sbo = box_grid[0, 1] - box_grid[0, 0]
                    lbo = lbo_coeff // t
                elif m > 1:
                    lbo_coeff = box_grid[1, 0] - box_grid[0, 0]
                    lbo = lbo_coeff // t
                    sbo = 1  # Default
                elif k > 1:
                    sbo = box_grid[0, 1] - box_grid[0, 0]
                    lbo = 1  # Default
                else:
                    sbo, lbo = 1, 1  # Single box
                    
        else:  # K-major
            if bbits == 0:
                # No swizzle: base_offset = i0 * (T * SBO) + i1 * 1 + j0 * 1 + j1 * T + j2 * LBO
                # The box grid represents (m, k) where each box is an (8, T) atom
                # box_grid[i,j] corresponds to the offset at the top-left of atom (i,j)
                # 
                # From the canonical layout, moving between boxes:
                # - box[1,0] vs box[0,0]: i1 increases by 1 (since we move 8 rows), so difference = 1 (from i1 term)
                # - box[0,1] vs box[0,0]: j2 increases by 1 (since we move T cols), so difference = LBO
                if m > 1 and k > 1:
                    # For K-major no swizzle: base_offset = i0 * (T * SBO) + i1 * 1 + j0 * 1 + j1 * T + j2 * LBO
                    # 
                    # From the box grid pattern [[0, 8, 16], [1, 9, 17]]:
                    # - LBO can be extracted from box[0,1] - box[0,0] = 8 (this is j2 difference)
                    # - For SBO, we need to look at the stride within a single atom
                    #   The stride from (0,0) to (1,0) in the target grid is T*SBO = 64
                    #   So SBO = 64 / T = 64 / 4 = 16
                    
                    lbo = box_grid[0, 1] - box_grid[0, 0]
                    
                    # Extract SBO from the stride pattern within the grid
                    # target_offset_grid[1, 0] - target_offset_grid[0, 0] = T * SBO
                    sbo_coeff = target_offset_grid[1, 0] - target_offset_grid[0, 0]
                    sbo = sbo_coeff // t
                elif m > 1:
                    if target_offset_grid.shape[0] > 1:
                        sbo_coeff = target_offset_grid[1, 0] - target_offset_grid[0, 0]
                        sbo = sbo_coeff // t
                    else:
                        sbo = 1
                    lbo = 1  # Default
                elif k > 1:
                    lbo_coeff = box_grid[0, 1] - box_grid[0, 0]
                    lbo = lbo_coeff
                    sbo = 1  # Default
                else:
                    sbo, lbo = 1, 1  # Single box
            else:
                # Swizzled: box_value = i * SBO, LBO not used
                if m > 1:
                    sbo = box_grid[1, 0] - box_grid[0, 0]
                else:
                    sbo = 1  # Default
                lbo = 1  # Not used in swizzled K-major
                
        return int(sbo), int(lbo)
        
    except Exception:
        return None, None


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
