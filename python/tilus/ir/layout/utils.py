from typing import Sequence


class LayoutOperationError(Exception):
    """
    Exception raised when a layout operation fails.
    """

    pass


def get_mode_groups(shape: Sequence[int], mode_shape: Sequence[int]) -> list[list[int]]:
    """
    Get the groups of modes based on the shape and mode_shape.

    Parameters
    ----------
    shape: Sequence[int]
        A sequence of integers representing the shape.

    mode_shape: Sequence[int]
        The shape of the modes. All elements must be greater than 1.

    Returns
    -------
    grouped_modes: list[list[int]]
        The groups of modes. Each group corresponds to one dimension of the shape. Thus,
        len(grouped_modes) == len(shape) and sum(len(group) for group in grouped_modes) == len(mode_shape).
    """
    if any(s <= 1 for s in mode_shape):
        raise LayoutOperationError(f"All elements in mode_shape must be greater than 1, got {mode_shape}.")
    i = 0
    grouped_modes = []
    for s in shape:
        group_modes = []
        remaining = s
        while remaining > 1:
            if i >= len(mode_shape) or remaining % mode_shape[i] != 0:
                raise LayoutOperationError(f"Cannot group the modes for shape {shape} with mode_shape {mode_shape}. ")
            remaining //= mode_shape[i]
            group_modes.append(i)
            i += 1
        grouped_modes.append(group_modes)
    if i != len(mode_shape):
        raise LayoutOperationError(f"Cannot group the modes for shape {shape} with mode_shape {mode_shape}. ")
    return grouped_modes
