# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from dataclasses import dataclass
from typing import Callable, Sequence

from tilus.hidet.ir.expr import Expr, Var, as_expr, index_vars
from tilus.ir.node import IRNode


@dataclass(frozen=True, eq=False)
class GlobalLayout(IRNode):
    """The layout for global tensor.

    Attributes
    ----------
    shape: tuple[Expr, ...]
        The shape of the global tensor. Each dimension can be an expression of grid-invariant expression, or a constant
        integer.
    size: Expr
        The storage size of the global tensor, in number of elements. If the layout is a `compact` layout, size
        should be equal to the product of the shape dimensions. Otherwise, it can be either larger (in case of padding)
        or smaller (in case of sharing data for different elements) than the product of the shape dimensions.
    axes: tuple[Var, ...]
        The axes of the global tensor. Each axis is a variable that represents the index of the corresponding dimension.
        It should have the same length as the shape.
    offset: Expr
        An expression that computes the offset of the global tensor based on the axes. Only the axes and grid-invariant
        variables can be used in the expression.
    """

    shape: tuple[Expr, ...]
    size: Expr
    axes: tuple[Var, ...]
    offset: Expr

    def __call__(self, *indices: Expr) -> Expr:
        """Compute the offset on given indices.

        This method computes the offset of an element in the global tensor with the given indices.

        Parameters
        ----------
        indices: Sequence[Expr]
            The indices of the global tensor. The length of the indices should match the number of axes in the layout.

        Returns
        -------
        ret: Expr
            The computed offset of the global tensor element at the given indices.
        """
        assert len(indices) == len(self.axes)
        from tilus.hidet.ir.tools import rewrite

        return rewrite(self.offset, rewrite_map={axis: index for axis, index in zip(self.axes, indices)})

    @staticmethod
    def create(shape: Sequence[Expr | int], size: Expr, f_offset: Callable[[Sequence[Var]], Expr]) -> GlobalLayout:
        """Create a global layout with custom mapping.

        This method creates a global layout with the given shape, size, and the mapping function for the offset.

        Parameters
        ----------
        shape: Sequence[Expr | int]
            The shape of the global tensor. Each dimension can be an expression of grid-invariant expression, or a
            constant integer.
        size: Expr
            The storage size of the global tensor, in number of elements. If the layout is a `compact` layout, size
            should be equal to the product of the shape dimensions. Otherwise, it can be either larger (in case of
            padding) or smaller (in case of sharing data for different elements) than the product of the shape dimensions.
        f_offset: Callable[[Sequence[Var]], Expr]
            A function that takes a sequence of axes (variables) and returns an expression that computes the offset
            of the global tensor based on those axes. The axes are variables that represent the index of the corresponding
            dimension. The function should not use any variables that are not grid-invariant.

        Returns
        -------
        ret: GlobalLayout
            A new instance of GlobalLayout with the specified shape, size, axes, and offset function.
        """
        expr_shape = tuple(as_expr(s) for s in shape)
        axes: list[Var] = index_vars(num_vars=len(shape))
        return GlobalLayout(shape=expr_shape, size=size, axes=tuple(axes), offset=f_offset(axes))
