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
from typing import Optional, Sequence

import numpy as np
from hidet.ir.expr import Expr, Var, as_expr
from hidet.ir.utils.index_transform import index_deserialize
from hidet.utils import prod

from tilus.extensions.hidet.ir.expr import index_vars
from tilus.ir.node import IRNode
from tilus.ir.utils.veceval import vectorized_evaluate


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

    def __str__(self):
        return f"Swizzle(base={self.base}, bits={self.bits}, shift={self.shift})"
    
    def __eq__(self, value):
        if value is None:
            return False
        assert isinstance(value, Swizzle)
        return self.base == value.base and self.bits == value.bits and self.shift == value.shift
    
    def __hash__(self):
        return hash((self.base, self.bits, self.shift))


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
    optional_swizzle: Optional[Swizzle]

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
        from tilus.ir.layout.ops.utils import get_mode_groups

        # get the stride-based index
        group_modes = get_mode_groups(self.shape, self.mode_shape)
        mode_indices: list[Expr] = []
        for index, modes in zip(indices, group_modes):
            mode_indices.extend(index_deserialize(index, shape=[self.mode_shape[m] for m in modes]))
        total_index: Expr = as_expr(sum(index * stride for index, stride in zip(mode_indices, self.mode_strides)))

        # apply swizzle if exists
        if self.optional_swizzle is not None:
            total_index = self.optional_swizzle(total_index)

        return total_index
    
    def __eq__(self, other):
        if not isinstance(other, SharedLayout):
            return False
        return (
            self.shape == other.shape
            and self.mode_shape == other.mode_shape
            and self.mode_strides == other.mode_strides
            and self.optional_swizzle == other.optional_swizzle
        )
    
    def __hash__(self):
        return hash((self.shape, self.mode_shape, self.mode_strides, self.optional_swizzle))

    @property
    def swizzle(self) -> Swizzle:
        if self.optional_swizzle is None:
            raise ValueError("No swizzle is applied on this layout.")
        return self.optional_swizzle

    @staticmethod
    def create(
        shape: Sequence[int],
        mode_shape: Sequence[int],
        mode_strides: Sequence[int],
        optional_swizzle: Optional[Swizzle],
    ) -> SharedLayout:
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
        if prod(mode_shape) != prod(shape):
            raise ValueError("The product of mode_shape must equal to the product of shape.")
        return SharedLayout(
            shape=tuple(shape),
            mode_shape=tuple(mode_shape),
            mode_strides=tuple(mode_strides),
            optional_swizzle=optional_swizzle,
        )

    def as_numpy_grid(self) -> np.ndarray:
        grid_axes = np.meshgrid(*[np.arange(extent) for extent in self.shape], indexing="ij")
        axes = index_vars(num_vars=len(self.shape))
        offset = self(*axes)
        atom_grid = vectorized_evaluate(expr=offset, var2value={axis: grid_axes[i] for i, axis in enumerate(axes)})
        return atom_grid

    def as_axes_mapping(self) -> tuple[list[Var], Expr]:
        axes = index_vars(num_vars=len(self.shape))
        offset = self(*axes)
        return axes, offset

    def count_size(self) -> int:
        """Count the total size of the shared layout.

        It is the minimum number of elements required to store the tensor in shared memory.

        Returns
        -------
        ret: int
            The total size of the shared layout.
        """
        indices = [extent - 1 for extent in self.mode_shape]
        max_index = sum(a * b for a, b in zip(indices, self.mode_strides))
        return max_index + 1

    def slice(self, retain_dims: Sequence[int]) -> SharedLayout:
        from tilus.ir.layout.ops.shared_ops import shared_slice

        return shared_slice(self, retain_dims)

    def apply_swizzle(self, swizzle: Swizzle) -> SharedLayout:
        if self.optional_swizzle is not None:
            raise RuntimeError("Chained swizzle is not supported.")
        return SharedLayout.create(
            shape=self.shape,
            mode_shape=self.mode_shape,
            mode_strides=self.mode_strides,
            optional_swizzle=swizzle,
        )

    def prepend_dim(self, extent: int) -> SharedLayout:
        shape = (extent,) + self.shape
        if extent > 1:
            mode_shape = (extent,) + self.mode_shape
            mode_strides = (self.count_size(),) + self.mode_strides
        else:
            mode_shape = self.mode_shape
            mode_strides = self.mode_strides

        return SharedLayout.create(
            shape=shape,
            mode_shape=mode_shape,
            mode_strides=mode_strides,
            optional_swizzle=self.optional_swizzle,
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

def canonicalize_shared_layout(
    layout: SharedLayout
) -> SharedLayout:
    """Canonicalize a SharedLayout.

    This function merges consecutive modes that belong to the same dimension if they have compatible strides.
    It also standardizes the swizzle representation by setting it to None if it swizzle 0 bits. After canonicalization,
    two SharedLayouts that are functionally equivalent will have the same representation.
    
    Parameters
    ----------
    layout: SharedLayout
        The SharedLayout to be canonicalized.
    Returns
    -------
    ret: SharedLayout
        The canonicalized SharedLayout.
    """
    from tilus.ir.layout.ops.utils import get_mode_groups

    # merge consecutive modes that belong to the same dimension
    mode_groups: list[list[int]] = get_mode_groups(layout.shape, layout.mode_shape)
    grouped_mode_shape: list[list[int]] = []
    grouped_mode_strides: list[list[int]] = []
    for modes in mode_groups:
        grouped_mode_shape.append([])
        grouped_mode_strides.append([])
        i = 0
        while i < len(modes):
            j = i
            while j + 1 < len(modes) and layout.mode_strides[modes[i]] == layout.mode_strides[modes[j + 1]] * layout.mode_shape[modes[j + 1]]:
                j += 1
            grouped_mode_shape[-1].append(prod(layout.mode_shape[modes[k]] for k in range(i, j + 1)))
            grouped_mode_strides[-1].append(layout.mode_strides[modes[j]])
            i = j + 1
    mode_shape: tuple[int, ...] = tuple(shape for group in grouped_mode_shape for shape in group)
    mode_strides: tuple[int, ...] = tuple(strides for group in grouped_mode_strides for strides in group)

    # canonicalize swizzle: if swizzle has 0 bits, set it to None (both mean no swizzle)
    if layout.optional_swizzle is not None and layout.optional_swizzle.bits == 0:
        optional_swizzle = None
    else:
        optional_swizzle = layout.optional_swizzle
        

    if (
        mode_shape == layout.mode_shape
        and mode_strides == layout.mode_strides
        and optional_swizzle is layout.optional_swizzle
    ):
        return layout
    else:
        return SharedLayout.create(
            shape=layout.shape, mode_shape=mode_shape, mode_strides=mode_strides, optional_swizzle=optional_swizzle
        )

def shared_layout(
    shape: Sequence[int],
    mode_shape: Sequence[int],
    mode_strides: Sequence[int],
    optional_swizzle: Optional[Swizzle] = None,
) -> SharedLayout:
    """Create a SharedLayout from shape, mode_shape, and mode_strides.

    The created SharedLayout is canonicalized.

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
    layout = SharedLayout.create(
        shape=shape,
        mode_shape=mode_shape,
        mode_strides=mode_strides,
        optional_swizzle=optional_swizzle,
    )
    layout = canonicalize_shared_layout(layout)
    return layout

