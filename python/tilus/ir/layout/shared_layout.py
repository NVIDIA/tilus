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
from typing import Callable, Dict, List, Sequence, Optional

from hidet.ir.expr import Expr, Var, as_expr
from hidet.ir.utils.index_transform import index_deserialize

from tilus.extensions.hidet.ir.expr import index_vars
from tilus.ir.node import IRNode
from tilus.ir.layout.ops.utils import get_mode_groups

@dataclass(frozen=True, eq=True)
class Swizzle:
    """
    A swizzle function.

    0xxxxYYYxxZZZxxxx
    z_mask = ((1 << self.bbits) - 1) << self.mbase
    y_mask = ((1 << self.bbits) - 1) << (self.mbase + self.sshift)
    return offset ^ ((offset & y_mask) >> self.sshift)
    """
    base: int
    bits: int
    shift: int

    def __call__(self, index: Expr) -> Expr:
        # we use a primitive function to here
        # todo: use general computation after refactor to cute-like shared layout
        from tilus.extensions.hidet.ir.primitives.swizzle import swizzle

        if self.bits == 0:
            return index
        return swizzle(index, self.base, self.bits, self.shift)
    

@dataclass(frozen=True, eq=False)
class SharedLayout(IRNode):
    """The layout for shared tensor.

    We use three components to describe a shared tensor layout: the shape, the mode shape, and the mode strides.
    
    The mode shape and mode strides are used to describe how to split each dimension into multiple sub-dimensions (modes), 
    and the strides of each mode. 

    For example, consider a shape of (64, 32), we can split the first dimension into two sub-dimensions (modes) of size 8 and 8,
    and the second dimension into two sub-dimensions (modes) of size 16 and 2. The mode shape would be (8, 8, 16, 2). We can 
    have strides for each mode, for example, (256, 2, 16, 1). Then given the indices (i, j), we can compute the indices in the
    sub-dimensions (i1, i2, j1, j2) where i1 = i // 8, i2 = i % 8, j1 = j // 2, j2 = j % 2. The offset can be computed as:
    offset = i1 * 256 + i2 * 2 + j1 * 16 + j2 * 1. To get the final offset in the shared tensor, we can use the formula:
    (i, j) => ((i // 8) * 256) + ((i % 8) * 2) + ((j // 2) * 16) + ((j % 2) * 1).

    Attributes
    ----------
    shape: tuple[int, ...]
        The shape of the shared tensor. Each dimension is a constant integer.
    mode_shape: tuple[int, ...]
        We can split each dimension into multiple sub-dimensions (modes). 
    mode_strides: tuple[int, ...]
        The strides of each mode.
    swizzle: Optional[Swizzle]
        The swizzle function to apply on the final offset. If None, no swizzling is applied.
    """

    shape: tuple[int, ...]
    mode_shape: tuple[int, ...]
    mode_strides: tuple[int, ...]
    swizzle: Optional[Swizzle]

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
        # get the stride-based index
        group_modes = get_mode_groups(self.shape, self.mode_shape)
        mode_indices: list[Expr] = []
        for index, modes in zip(indices, group_modes):
            mode_indices.extend(index_deserialize(index, shape=[self.mode_shape[m] for m in modes]))
        total_index: Expr = as_expr(sum(index * stride for index, stride in zip(mode_indices, self.mode_strides)))
        
        # apply swizzle if exists
        if self.swizzle is not None:
            total_index = self.swizzle(total_index)
        
        return total_index

    @staticmethod
    def create(shape: Sequence[int], mode_shape: Sequence[int], mode_strides: Sequence[int], swizzle: Optional[Swizzle]) -> SharedLayout:
        """
        Create a SharedLayout from shape, mode_shape, and mode_strides.
        
        Parameters
        ----------
        shape: Sequence[int]
            The shape of the shared tensor.
        mode_shape: Sequence[int]
            The mode shape of the shared tensor.
        mode_strides: Sequence[int]
            The mode strides of the shared tensor.
        swizzle: Optional[Swizzle]
            The swizzle function to apply on the final offset. If None, no swizzling is applied.

        Returns
        -------
        ret: SharedLayout
            The created SharedLayout.
        """
        if any(s < 1 for s in shape):
            raise ValueError("All dimensions in shape must be positive integers.")
        if len(mode_shape) != len(mode_strides):
            raise ValueError("mode_shape and mode_strides must have the same length.")
        return SharedLayout(shape=tuple(shape), mode_shape=tuple(mode_shape), mode_strides=tuple(mode_strides), swizzle=swizzle)

    @property
    def size(self) -> int:
        """Get the total size of the shared layout.

        It is the minimum number of elements required to store the tensor in shared memory.

        Returns
        -------
        ret: int
            The total size of the shared layout.
        """
        indices = [extent - 1 for extent in self.mode_shape]
        max_index = sum(a * b for a, b in zip(indices, self.mode_strides))
        return max_index + 1

    def slice(self, offsets: Sequence[Expr], slice_dims: Sequence[int], slice_shape: Sequence[int]) -> SharedLayout:
        raise RuntimeError("No slice anymore.")

    def simplify(self) -> SharedLayout:
        raise RuntimeError("No need to simplify anymore.")

    def with_swizzle(self, dim: int, regards_dim: int, log_step: int) -> SharedLayout:
        raise RuntimeError("Update swizzle.")
        # ndims = len(self.shape)
        # assert 0 <= dim < ndims and 0 <= regards_dim < ndims and dim != regards_dim
        # def get_xor_index(indices: Sequence[Expr]) -> Expr:
        #     indices = list(indices)  # copy
        #     step = 2**log_step
        #     regards_index = indices[regards_dim] // step
        #     regards_extent = self.shape[regards_dim] // step
        #     if regards_extent > self.shape[dim]:
        #         regards_index = regards_index % self.shape[dim]
        #     return regards_index
        # def f_offset(axes: Sequence[Var]) -> Expr:
        #     swizzled_indices: List[Expr] = [axis for axis in axes]
        #     swizzled_indices[dim] = swizzled_indices[dim] ^ get_xor_index(axes)
        #     return self(*swizzled_indices)
        # return SharedLayout.create(shape=self.shape, size=self.size, f_offset=f_offset)

    def prepend_dim(self, extent: int) -> SharedLayout:
        shape = (extent,) + self.shape
        if extent > 1:
            mode_shape = (extent,) + self.mode_shape
            mode_strides = (self.size,) + self.mode_strides
        else:
            mode_shape = self.mode_shape
            mode_strides = self.mode_strides

        return SharedLayout.create(
            shape=shape,
            mode_shape=mode_shape,
            mode_strides=mode_strides,
            swizzle=self.swizzle,
        )

    def transpose(self) -> SharedLayout:
        assert len(self.shape) == 2
        return self.permute(dims=[1, 0])

    def permute(self, dims: Sequence[int]) -> SharedLayout:
        from tilus.ir.layout.ops.shared_ops import shared_permute

        return shared_permute(self, dims)

    def unsqueeze(self, dims: Sequence[int]) -> SharedLayout:
        from tilus.ir.layout.ops.shared_ops import shared_unsqueeze

        return shared_unsqueeze(self, dims)


    def visualize(self, tablefmt: str = "simple_grid") -> str:
        from tilus.ir.layout.ops.shared_ops import visualize_layout

        return visualize_layout(self, tablefmt=tablefmt)
