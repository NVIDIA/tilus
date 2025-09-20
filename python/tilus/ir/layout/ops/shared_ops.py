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

from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, Var
from hidet.utils import prod

from tilus.extensions.hidet.ir.utils.index_transform import vector_mul
from tilus.ir.layout.shared_layout import SharedLayout


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
