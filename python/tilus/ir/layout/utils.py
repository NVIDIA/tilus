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
import tabulate
from tilus.ir.utils.veceval import meshgrid, vectorized_evaluate
from tilus.utils import prod

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


def visualize_layout(layout, tablefmt: str = "simple_grid") -> str:
    """
    Visualize the layout in a human-readable format.

    Parameters
    ----------
    layout: RegisterLayout
        The layout to be converted.

    tablefmt: str
        The table format to use. It should be a valid format specifier in tabulate.tabulate function.
        Candidates:

        - simple_grid
        - plain
        - grid
        - rounded_grid
        - mixed_grid
        - double_grid
        - fancy_grid
        - outline
        - simple_outline
        - mixed_outline
        - presto

    Returns
    -------
    ret: str
        The string representation of the layout that is human-readable.
    """
    from tilus.ir.layout.register_layout import RegisterLayout
    from tilus.ir.layout.shared_layout import SharedLayout

    head = str(layout)

    if isinstance(layout, RegisterLayout):
        shape = list(layout.shape)
        if len(shape) > 3:
            # normalize the shape into 3-dimension
            shape = [s for s in shape if s > 1]  # prune 1s
            while len(shape) > 3:
                shape = [prod(shape[0:2])] + shape[2:]
        while len(shape) < 3:
            shape.insert(0, 1)
        layout = layout.with_shape(shape)

        tables: list[str] = []
        for batch in range(shape[0]):
            table: list[list[str]] = []
            for i in range(shape[1]):
                row = []
                for j in range(shape[2]):
                    local_index = layout.get_local(global_indices=[batch, i, j])
                    thread_indices = layout.get_spatial(global_indices=[batch, i, j])
                    thread_indices.sort()
                    if len(thread_indices) == 1:
                        row.append(f"{thread_indices[0]}: {local_index}")
                    else:
                        row.append(f"{thread_indices}: {local_index}")
                table.append(row)
            tables.append(tabulate.tabulate(table, tablefmt=tablefmt))

        return head + "\n" + "\n".join(tables)
    elif isinstance(layout, SharedLayout):
        if len(layout.shape) != 2:
            raise LayoutOperationError(f"Shared layout with shape {layout.shape} is not supported for visualization.")
        grid = meshgrid(layout.shape)
        offset_grid = vectorized_evaluate(layout.offset, var2value={axis: grid[i] for i, axis in enumerate(layout.axes)})
        table = []
        for i in range(layout.shape[0]):
            row = []
            for j in range(layout.shape[1]):
                row.append(f"{offset_grid[i, j]}")
            table.append(row)
        return head + "\n" + tabulate.tabulate(table, tablefmt=tablefmt)
    else:
        raise ValueError(f"Unsupported layout type: {type(layout)}")
