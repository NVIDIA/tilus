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

from typing import List, Sequence

import tabulate
from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, Var
from hidet.ir.utils.index_transform import index_serialize, index_deserialize
from hidet.utils import prod

from tilus.extensions.hidet.ir.utils.index_transform import vector_mul
from tilus.ir.layout.ops.utils import LayoutOperationError
from tilus.ir.layout.shared_layout import SharedLayout
from tilus.ir.utils.veceval import meshgrid, vectorized_evaluate


def _generic_repeat(shape: List[int], ranks: List[int]) -> SharedLayout:
    assert len(shape) == len(ranks)
    assert len(ranks) == len(set(ranks)) and all(0 <= d < len(shape) for d in ranks)
    strides: List[int] = [prod([s for j, s in enumerate(shape) if ranks[j] > ranks[i]]) for i in range(len(shape))]

    def f_offset(axes: Sequence[Var]) -> Expr:
        return sum([axes[i] * strides[i] for i in range(len(shape))], start=int32.zero)

    return SharedLayout.create(shape=shape, size=prod(shape), f_offset=f_offset)


def _shared_compose(lhs: SharedLayout, rhs: SharedLayout) -> SharedLayout:
    assert len(lhs.shape) == len(rhs.shape)
    ndims = len(lhs.shape)

    def f_offset(axes: Sequence[Var]) -> Expr:
        lhs_axes = [axes[i] // rhs.shape[i] for i in range(ndims)]
        rhs_axes = [axes[i] % rhs.shape[i] for i in range(ndims)]
        lhs_offset = lhs(*lhs_axes)
        rhs_offset = rhs(*rhs_axes)
        return lhs_offset * rhs.size + rhs_offset

    shape = vector_mul(lhs.shape, rhs.shape)
    size = lhs.size * rhs.size

    return SharedLayout.create(shape=shape, size=size, f_offset=f_offset)


def shared_row_major(*shape: int) -> SharedLayout:
    """Create a shared layout with row-major order.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of the shared tensor. Each dimension is a constant integer.

    Returns
    -------
    ret: SharedLayout
        A shared layout with the specified shape in row-major order.
    """
    return _generic_repeat(shape=list(shape), ranks=list(range(len(shape))))


def shared_column_major(*shape: int) -> SharedLayout:
    """Create a shared layout with column-major order.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of the shared tensor. Each dimension is a constant integer.

    Returns
    -------
    ret: SharedLayout
        A shared layout with the specified shape in column-major order.
    """
    return _generic_repeat(shape=list(shape), ranks=list(reversed(range(len(shape)))))


def shared_compose(lhs: SharedLayout, rhs: SharedLayout, *others: SharedLayout) -> SharedLayout:
    """Compose multiple shared layouts together.

    Parameters
    ----------
    lhs: SharedLayout
        The first shared layout to compose.
    rhs: SharedLayout
        The second shared layout to compose.
    others: Sequence[SharedLayout]
        The additional shared layouts to compose with the first two. It can be empty.

    Returns
    -------
    ret: SharedLayout
        The composed shared layout.
    """
    if len(others) == 0:
        return _shared_compose(lhs, rhs)
    else:
        return shared_compose(_shared_compose(lhs, rhs), *others)


def shared_permute(layout: SharedLayout, dims: Sequence[int]) -> SharedLayout:
    """Permute the dimensions of the shared layout.

    Parameters
    ----------
    layout: SharedLayout
        The layout to permute.

    dims: Sequence[int]
        The permutation order of the dimensions. The length of dims must be equal to the number of dimensions of the
        layout.

    Returns
    -------
    ret: SharedLayout
        The permuted layout.
    """
    if set(dims) != set(range(len(layout.shape))):
        raise LayoutOperationError("Dims must be a permutation of {}, got {}".format(range(len(layout.shape)), dims))
    shape = tuple(layout.shape[d] for d in dims)
    axes = tuple(layout.axes[d] for d in dims)
    return SharedLayout(shape=shape, size=layout.size, axes=axes, offset=layout.offset)

def shared_reshape(layout: SharedLayout, new_shape: Sequence[int]) -> SharedLayout:
    """Reshape the shared layout to a new shape.

    Parameters
    ----------
    layout: SharedLayout
        The layout to reshape.

    new_shape: Sequence[int]
        The new shape of the layout. The product of the dimensions in the new shape must be equal to the product of the
        dimensions in the original shape.

    Returns
    -------
    ret: SharedLayout
        The reshaped layout.
    """
    if prod(new_shape) != prod(layout.shape):
        raise LayoutOperationError(f"Cannot reshape shared layout with shape {layout.shape} to new shape {new_shape} due to size mismatch.")

    # partition the original shape and new shape into chunks with the same size
    # for example:
    # [4, 8, 5, 6, 20] => [[4, 8, 5], [6], [20]]
    # [20, 8, 6, 4, 5] => [[20, 8], [6], [4, 5]]
    original_chunks: List[List[int]] = []
    new_chunks: List[List[int]] = []
    o_start = 0
    n_start = 0
    while o_start < len(layout.shape) or n_start < len(new_shape):
        o_size = 1
        n_size = 1
        o_end = o_start
        n_end = n_start
        while (o_end - o_start == 0) or (n_end - n_start == 0) or (o_size != n_size):
            if o_size < n_size:
                o_size *= layout.shape[o_end]
                o_end += 1
            else:
                n_size *= new_shape[n_end]
                n_end += 1
        original_chunks.append(list(layout.shape[o_start:o_end]))
        new_chunks.append(list(new_shape[n_start:n_end]))
        o_start = o_end
        n_start = n_end
    assert o_start == len(layout.shape) and n_start == len(new_shape)

    def f_offset(axes: Sequence[Var]) -> Expr:
        # first get the linear index for each chunk
        chunk_indices: List[Expr] = []
        cur = 0
        for chunk in new_chunks:
            chunk_indices.append(index_serialize(axes[cur:cur + len(chunk)], chunk))
            cur += len(chunk)
        
        # then deserialize the linear index to original layout axes
        original_axes: list[Expr] = []
        for chunk_index, original_chunk in zip(chunk_indices, original_chunks):
            original_axes.extend(index_deserialize(chunk_index, original_chunk))
        
        # finally compute the offset using original layout
        return layout(*original_axes)

    return SharedLayout.create(shape=new_shape, size=layout.size, f_offset=f_offset)

def visualize_layout(layout: SharedLayout, tablefmt: str = "simple_grid") -> str:
    """
    Visualize the layout in a human-readable format.

    Parameters
    ----------
    layout: SharedLayout
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
    head = str(layout)
    assert isinstance(layout, SharedLayout)

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
