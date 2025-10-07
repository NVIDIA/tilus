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

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from hidet.ir.expr import Expr, Var, as_expr

from tilus.extensions.hidet.ir.expr import index_vars
from tilus.ir.node import IRNode


@dataclass(frozen=True, eq=False)
class SharedLayout(IRNode):
    """The layout for shared tensor.

    Attributes
    ----------
    shape: tuple[int, ...]
        The shape of the shared tensor. Each dimension is a constant integer.
    size: int
        The storage size of the shared tensor, in number of elements. If the layout is a `compact` layout, size
        should be equal to the product of the shape dimensions. Otherwise, it can be either larger (in case of padding)
        or smaller (in case of sharing data for different elements) than the product of the shape dimensions. The
        size must be a constant integer.
    axes: tuple[Var, ...]
        The axes of the shared tensor. Each axis is a variable that represents the index of the corresponding dimension.
        It should have the same length as the shape.
    offset: Expr
        The offset expression of the shared tensor based on the axes. It is an expression that computes the offset
        of the shared tensor based on the axes. Only the axes and variables that are invariant in the lifetime of the
        given corresponding shared tensor with this layout can be used in the expression.
    """

    shape: tuple[int, ...]
    size: int
    axes: tuple[Var, ...]
    offset: Expr

    def __call__(self, *indices: Expr) -> Expr:
        """Compute the offset on given indices.

        This method computes the offset of an element in the shared tensor with the given indices.

        Parameters
        ----------
        indices: Sequence[Expr]
            The indices of the shared tensor. The length of the indices should match the number of axes in the layout.

        Returns
        -------
        ret: Expr
            The computed offset of the shared tensor element at the given indices.
        """
        assert len(indices) == len(self.axes)
        from hidet.ir.tools import rewrite

        return rewrite(self.offset, rewrite_map={axis: index for axis, index in zip(self.axes, indices)})

    @staticmethod
    def create(shape: Sequence[int], size: int, f_offset: Callable[[Sequence[Var]], Expr | int]) -> SharedLayout:
        """Create a shared layout.

        This method creates a shared layout with the given shape, size, and a function to compute the offset based on
        the axes. The shape must be a sequence of constant integers, and the size must be a constant integer that is
        larger than the maximum possible offset computed by the `f_offset` function.

        Parameters
        ----------
        shape: Sequence[int]
            The shape of the shared tensor. Each dimension is a constant integer.
        size: int
            The storage size of the shared tensor, in number of elements.
        f_offset: Callable[[Sequence[Var]], Expr]
            The function that computes the offset of the shared tensor based on the axes. It takes a sequence of
            axes (variables) and returns an expression that computes the offset. The function must ensure that the
            size is larger than the maximum possible offset computed by this function.

        Returns
        -------
        ret: SharedLayout
            A shared layout with the specified shape, size, axes, and offset.
        """
        axes: List[Var] = index_vars(num_vars=len(shape))
        return SharedLayout(shape=tuple(shape), size=size, axes=tuple(axes), offset=as_expr(f_offset(axes)))

    def slice(self, offsets: Sequence[Expr], slice_dims: Sequence[int], slice_shape: Sequence[int]) -> SharedLayout:
        assert len(set(slice_dims)) == len(slice_dims), "slice_dims must be unique"
        assert len(slice_shape) == len(slice_dims), "slice_dims and slice_shape must have the same length"
        assert len(slice_dims) <= len(self.shape), "slice_dims must be less than or equal to the number of dimensions"

        def f_offset(axes: Sequence[Var]) -> Expr:
            indices: List[Expr] = list(offsets)
            for dim, axis in zip(slice_dims, axes):
                indices[dim] = indices[dim] + axis
            return self(*indices) - self(*offsets)

        return SharedLayout.create(shape=slice_shape, size=self.size, f_offset=f_offset)

    def simplify(self) -> SharedLayout:
        from tilus.extensions.hidet.transforms.rule_based_simplifier import BoundInfo, RuleBasedSimplifier

        var2bound: Dict[Var, BoundInfo] = {
            axis: BoundInfo(min_value=0, max_value=extent - 1) for axis, extent in zip(self.axes, self.shape)
        }
        simplifier = RuleBasedSimplifier(var2bound=var2bound)
        return SharedLayout(shape=self.shape, size=self.size, axes=self.axes, offset=simplifier(self.offset))

    def swizzle(self, dim: int, regards_dim: int, log_step: int) -> SharedLayout:
        ndims = len(self.shape)
        assert 0 <= dim < ndims and 0 <= regards_dim < ndims and dim != regards_dim

        def get_xor_index(indices: Sequence[Expr]) -> Expr:
            indices = list(indices)  # copy
            step = 2**log_step
            regards_index = indices[regards_dim] // step
            regards_extent = self.shape[regards_dim] // step
            if regards_extent > self.shape[dim]:
                regards_index = regards_index % self.shape[dim]
            return regards_index

        def f_offset(axes: Sequence[Var]) -> Expr:
            swizzled_indices: List[Expr] = [axis for axis in axes]
            swizzled_indices[dim] = swizzled_indices[dim] ^ get_xor_index(axes)
            return self(*swizzled_indices)

        return SharedLayout.create(shape=self.shape, size=self.size, f_offset=f_offset)

    def prepend_dim(self, extent: int) -> SharedLayout:
        def f_offset(axes: Sequence[Var]) -> Expr:
            tile_offset = axes[0] * self.size
            return tile_offset + self(*axes[1:])

        return SharedLayout.create(shape=(extent,) + self.shape, size=extent * self.size, f_offset=f_offset)

    def transpose(self) -> SharedLayout:
        assert len(self.shape) == 2
        return self.permute(dims=[1, 0])

    def permute(self, dims: Sequence[int]) -> SharedLayout:
        from tilus.ir.layout.ops.shared_ops import shared_permute

        return shared_permute(self, dims)

    def unsqueeze(self, dims: Sequence[int]) -> SharedLayout:
        shape = []
        cur_dim = 0
        for i in range(len(self.shape) + len(dims)):
            if i in dims:
                shape.append(1)
            else:
                shape.append(self.shape[cur_dim])
                cur_dim += 1

        def f_offset(axes: Sequence[Var]) -> Expr:
            base_axes = [axis for i, axis in enumerate(axes) if i not in dims]
            return self(*base_axes)

        return SharedLayout.create(shape=shape, size=self.size, f_offset=f_offset)

    def visualize(self, tablefmt: str = "simple_grid") -> str:
        from tilus.ir.layout.ops.shared_ops import visualize_layout

        return visualize_layout(self, tablefmt=tablefmt)
