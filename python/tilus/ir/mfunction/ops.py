from __future__ import annotations

from typing import Sequence

from tilus.ir.layout.utils import LayoutOperationError, get_mode_groups
from tilus.ir.mfunction.mfunction import MultiFunction, multi_function
from tilus.utils import gcd, prod


def identity(shape: Sequence[int]) -> MultiFunction:
    """Create an identity multi-function with the given shape.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of the multi-function.

    Returns
    -------
    ret: MultiFunction
        The identity multi-function with the given shape.
    """
    return multi_function(shape=shape, mode_shape=shape, modes=range(len(shape)))


def collapse(
    func: MultiFunction,
    dims: Sequence[int],
) -> MultiFunction:
    """Collapse the given multi-function along the specified dimensions.

    Given a multi-function and a sequence of dimensions, this function will collapse the multi-function along these
    dimensions to make all X where only the specified dimensions are different, to be mapped to the same Y. The image
    of the multi-function will be reduced accordingly.

    Parameters
    ----------
    func: MultiFunction
        The multi-function to be collapsed.
    dims: Sequence[int]
        The dimensions to collapse.

    Returns
    -------
    ret: MultiFunction
        The sliced multi-function. It has the same shape as the original multi-function.
    """
    if not all(0 <= dim < len(func.shape) for dim in dims):
        raise ValueError(f"Invalid dimensions {dims} for multi-function with shape {func.shape}.")
    modes_to_remove = [mode for dim, group in enumerate(func.mode_groups) for mode in group if dim in dims]
    new_modes = [mode for mode in func.modes if mode not in modes_to_remove]
    return multi_function(shape=func.shape, mode_shape=func.mode_shape, modes=new_modes)


def collapse_by_shape(func: MultiFunction, shape: Sequence[int]) -> MultiFunction:
    """Collapse the given multi-function according to the specified shape.

    The specified shape must be able to be broadcasted to the shape of the multi-function, that is, the shape of the
    multi-function must be a multiple of the specified shape in every dimension after prepending 1s to the specified
    shape to match the rank of the multi-function. When one dimension has different sizes, the specified shape must
    be 1 in that dimension.

    Parameters
    ----------
    func: MultiFunction
        The multi-function to be collapsed.

    shape: Sequence[int]
        The shape to collapse the multi-function according to.

    Returns
    -------
    ret: MultiFunction
        The collapsed multi-function. It has the same shape as the original multi-function.
    """
    dims_keep = []
    dims_no_keep = []

    diff_rank = len(func.shape) - len(shape)
    for i in range(len(func.shape)):
        if i < diff_rank:
            dims_no_keep.append(i)
        elif func.shape[i] != shape[i - diff_rank] and shape[i - diff_rank] == 1:
            dims_keep.append(i)
        elif func.shape[i] == shape[i - diff_rank]:
            # matched dimension
            continue
        else:
            raise ValueError(f"Cannot slice multi-function from shape {func.shape} to {shape}.")
    if dims_keep:
        func = collapse(func, dims=dims_keep)
    if dims_no_keep:
        func = collapse(func, dims=dims_no_keep)
    return func


def _multi_function_with_mode_shape(
    func: MultiFunction,
    shape: Sequence[int],
) -> MultiFunction:
    """Create a multi-function with fine-grained mode shape.

    Parameters
    ----------
    func: MultiFunction
        The multi-function to be converted.

    shape: Sequence[int]
        The fine-grained mode shape to be used for the multi-function.

    Returns
    -------
    ret: MultiFunction
        The multi-function with the given mode shape.

    Raises
    ------
    LayoutOperationError:
        If the mode shape is not compatible with the multi-function.
    """
    if prod(shape) != func.size:
        raise LayoutOperationError(
            f"The mode shape {shape} is not compatible with the multi-function of size {func.size}."
        )
    mode_groups = get_mode_groups(func.mode_shape, shape)
    new_modes = []
    for mode in func.modes:
        if mode < 0:
            new_modes.append(mode)  # replication mode
        else:
            new_modes.extend(mode_groups[mode])

    # do not canonicalize the multi-function here, as it may change the mode shape and modes
    # since it might fuse some modes
    return MultiFunction.create(shape=func.shape, mode_shape=shape, modes=new_modes)


def cover(
    fa: MultiFunction,
    fb: MultiFunction,
) -> bool:
    """Check whether the multi-function fa covers the multi-function fb.

    The size and image size of both multi-functions must be the same. For every x in the domain, if we have
    fb(x) \subseteq fa(x),
    then we say that fa covers fb. In other words, for every x in the domain of fb, the image of fb at x is a subset
    of the image of fa at x.

    Parameters
    ----------
    fa: MultiFunction
        The multi-function that is supposed to cover the other multi-function.
    fb: MultiFunction
        The multi-function that is supposed to be covered by the other multi-function.

    Returns
    -------
    ret: bool
        True if fa covers fb, False otherwise.
    """
    # 1. check if the size and image size of both multi-functions are the same, if not, return False
    # 2. calculate the fine-grained shape and mode_remap for both multi-functions
    if fa.size != fb.size or fa.image_size != fb.image_size:
        return False

    # get the fine-grained shape
    mode_shape = []
    a_shape, b_shape = list(fa.mode_shape), list(fb.mode_shape)
    while a_shape and b_shape:
        a_head = a_shape[0]
        b_head = b_shape[0]
        factor = gcd(a_head, b_head)
        if factor == 1:
            return False
        mode_shape.append(factor)
        a_shape[0] //= factor
        b_shape[0] //= factor
        if a_shape[0] == 1:
            a_shape.pop(0)
        if b_shape[0] == 1:
            b_shape.pop(0)

    if a_shape or b_shape:  # now, it should be empty
        return False

    # use the fine-grained shape to transform the multi-functions
    fa = _multi_function_with_mode_shape(fa, mode_shape)
    fb = _multi_function_with_mode_shape(fb, mode_shape)

    # check if all non-negative modes in fa matches the corresponding modes in fb
    a_modes, b_modes = list(fa.modes), list(fb.modes)
    while a_modes and b_modes:
        a_mode = a_modes[-1]
        b_mode = b_modes[-1]

        if a_mode < 0 and b_mode < 0:
            factor = gcd(-a_mode, -b_mode)
            if factor == 1:
                return False
            a_modes[-1] //= factor
            b_modes[-1] //= factor
            if a_modes[-1] == -1:
                a_modes.pop()
            if b_modes[-1] == -1:
                b_modes.pop()
        elif a_mode < 0:
            if (-a_mode) % fb.mode_shape[b_mode] != 0:
                return False
            a_modes[-1] = -(-a_modes[-1]) // fb.mode_shape[b_mode]
            if a_modes[-1] == -1:
                a_modes.pop()
            b_modes.pop()
        elif b_mode < 0:
            return False
        else:
            if a_mode != b_mode:
                return False
            a_modes.pop()
            b_modes.pop()
    if a_modes or b_modes:  # now, it should be empty
        return False

    return True
