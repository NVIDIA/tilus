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
from typing import Sequence


class LayoutOperationError(Exception):
    """Exception raised when a layout operation fails."""

    pass


def get_mode_groups(shape: Sequence[int], mode_shape: Sequence[int]) -> list[list[int]]:
    """
    Get the groups of modes based on the shape and mode_shape.

    Example:
    >>> shape = [64, 32]
    >>> mode_shape = [8, 8, 16, 2]
    >>> get_mode_groups(shape, mode_shape)
    [[0, 1], [2, 3]]

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
