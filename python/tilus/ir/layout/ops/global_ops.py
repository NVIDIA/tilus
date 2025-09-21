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

from typing import Sequence

from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, Var

from tilus.extensions.hidet.ir.utils.index_transform import index_multiply
from tilus.ir.layout.global_layout import GlobalLayout
from tilus.utils import prod


def _generic_repeat(shape: Sequence[Expr | int], ranks: Sequence[int]) -> GlobalLayout:
    assert len(shape) == len(ranks)
    assert len(ranks) == len(set(ranks)) and all(0 <= d < len(shape) for d in ranks)
    strides: list[Expr] = [prod([s for j, s in enumerate(shape) if ranks[j] > ranks[i]]) for i in range(len(shape))]

    def f_offset(axes: Sequence[Var]) -> Expr:
        return sum([axes[i] * strides[i] for i in range(len(shape))], start=int32.zero)

    return GlobalLayout.create(shape=shape, size=prod(shape), f_offset=f_offset)


def _global_compose(lhs: GlobalLayout, rhs: GlobalLayout) -> GlobalLayout:
    assert len(lhs.shape) == len(rhs.shape)
    ndims = len(lhs.shape)

    def f_offset(axes: Sequence[Var]) -> Expr:
        lhs_indices = [axes[i] // rhs.shape[i] for i in range(ndims)]
        rhs_indices = [axes[i] // rhs.shape[i] for i in range(ndims)]
        lhs_offset = lhs(*lhs_indices)
        rhs_offset = rhs(*rhs_indices)
        return lhs_offset * rhs.size + rhs_offset

    shape = index_multiply(lhs.shape, rhs.shape)
    size = lhs.size * rhs.size

    return GlobalLayout.create(shape=shape, size=size, f_offset=f_offset)


def global_row_major(*shape: Expr | int) -> GlobalLayout:
    """Create a global layout with row-major order.

    Parameters
    ----------
    shape: Sequence[Expr | int]
        The shape of the global tensor. Each dimension can be an expression of grid-invariant expression, or a
        constant integer.

    Returns
    -------
    ret: GlobalLayout
        A global layout with the specified shape in row-major order.
    """
    return _generic_repeat(shape=shape, ranks=list(range(len(shape))))


def global_column_major(*shape: Expr | int) -> GlobalLayout:
    """Create a global layout with column-major order.

    Parameters
    ----------
    shape: Sequence[Expr | int]
        The shape of the global tensor. Each dimension can be an expression of grid-invariant expression, or a
        constant integer.

    Returns
    -------
    ret: GlobalLayout
        A global layout with the specified shape in column-major order.
    """
    return _generic_repeat(shape=shape, ranks=list(reversed(range(len(shape)))))


def global_compose(lhs: GlobalLayout, rhs: GlobalLayout, *others: GlobalLayout) -> GlobalLayout:
    """Compose multiple global layouts.

    This function composes two or more global layouts into a single global layout.
    Please refer to our research paper `Tilus <https://arxiv.org/pdf/2504.12984>`_, Section 4.2 for more details on layout composition.

    Parameters
    ----------
    lhs: GlobalLayout
        The left-hand side global layout.
    rhs: GlobalLayout
        The right-hand side global layout.
    others: Sequence[GlobalLayout]
        The additional global layouts to be composed with the first two. It's optional and can be empty.

    Returns
    -------
    ret: GlobalLayout
        The composed global layout that combines the effects of all input layouts.
    """
    if len(others) == 0:
        return _global_compose(lhs, rhs)
    else:
        return global_compose(_global_compose(lhs, rhs), *others)


def global_strides(shape: Sequence[Expr | int], strides: Sequence[Expr | int]) -> GlobalLayout:
    """Create a global layout with specified strides.

    This function creates a global layout with the given shape and strides. Given the axes and strides, we map the
    axes to ``sum(axes[i] * strides[i])`` to compute the offset of the global tensor.

    Parameters
    ----------
    shape: Sequence[Expr | int]
        The shape of the global tensor. Each dimension can be an expression of grid-invariant expression, or a
        constant integer.
    strides: Sequence[Expr | int]
        The strides of the global tensor. Each stride corresponds to the step size in each dimension when traversing
        the global tensor. It should have the same length as the shape. Each stride can be an expression of grid-invariant
        expression, or a constant integer.

    Returns
    -------
    ret: GlobalLayout
        A global layout with the specified shape and strides. The offset is computed as the sum of the product of each
        axis and its corresponding stride.
    """
    assert len(shape) == len(strides)

    def f_offset(axes: Sequence[Var]) -> Expr:
        return sum([axes[i] * strides[i] for i in range(len(shape))], start=int32.zero)

    return GlobalLayout.create(shape=shape, size=prod(shape), f_offset=f_offset)


def global_slice(
    layout: GlobalLayout, offsets: Sequence[Expr | int], dims: Sequence[int], shape: Sequence[Expr | int]
) -> GlobalLayout:
    """Create a sliced global layout from an existing layout.

    This function creates a new global layout by slicing an existing global layout. The slicing is defined by the
    specified offsets, dimensions to slice, and the shape of the resulting layout. The new layout retains the mapping
    function of the original layout, adjusted for the specified offsets and dimensions.

    Parameters
    ----------
    layout: GlobalLayout
        The original global layout to be sliced.
    offsets: Sequence[Expr | int]
        The offsets for each dimension of the original layout. It should have the same length as the original layout's
        shape.
    dims: Sequence[int]
        The dimensions to be sliced from the original layout. Each dimension should be a valid index in the original
        layout's shape.
    shape: Sequence[Expr | int]
        The shape of the resulting sliced global layout. It should have the same length as the number of dimensions
        specified in `dims`.

    Returns
    -------
    ret: GlobalLayout
        A new global layout that represents the sliced version of the original layout, with the specified shape and
        adjusted mapping function.
    """
    assert len(dims) == len(shape) <= len(layout.shape) == len(offsets)

    def f_offset(axes: Sequence[Var]) -> Expr:
        indices = list(offsets)
        for dim, axis in zip(dims, axes):
            indices[dim] = axis + offsets[dim]
        return layout(*indices) - layout(*offsets)  # type: ignore[arg-type]

    return GlobalLayout.create(shape=shape, size=prod(shape), f_offset=f_offset)
