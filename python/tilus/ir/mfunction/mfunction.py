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

import itertools
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import tabulate
from hidet.ir.expr import Expr

from tilus.extensions.hidet.ir.utils.index_transform import index_deserialize, index_serialize
from tilus.utils import prod

Int = int | Expr


@dataclass(frozen=True, eq=False)
class MultiFunction:
    """A multi-function represents a function that maps (x0, x1, ..., x_{n-1}) to a set of integers Y.

    Let f be the multi-function, we have

    Y = f(x0, ..., x_{n-1})

    where (x0, x1, ..., x_{n-1}) is in a grid of specific shape.

    The computation of the multi-function is defined through `mode_shape` and `modes`:

    mode_shape is a fine-grained shape of `shape`, where one dimension of the shape might be represented by a factor
    decomposition of multiple dimensions in the mode shape, and each dimension in the mode shape is called a mode.
    We use `modes` to represent an ordered list of modes that represent the image.

    For example, we can have a multi-function: f = MF(shape=(4, 5, 6), mode_shape=(2, 2, 5, 3, 2), modes=(0, 2, 3))

    f(x0, x1, x2)  # where 0 <= x0 < 4, 0 <= x1 < 5, 0 <= x2 < 6
    => mode indices: (x0 // 2, x0 % 2, x1, x2 // 2, x2 % 2)
    => selected mode indices: (x0 // 2, x2, x2 // 2)  # in shape (2, 5, 3)
    => x0 // 2 * 15 + x2 * 3 + x2 // 2

    If one mode is negative, it represents a replication mode, which means that the mode is replicated by the size of
    the absolute value of the mode. For example, if we have a multi-function:
    f = MF(shape=(4, 5, 6), mode_shape=(2, 2, 5, 3, 2), modes=(0, -3, 2))

    f(x0, x1, x2)  # where 0 <= x0 < 4, 0 <= x1 < 5, 0 <= x2 < 6
    => mode indices: (x0 // 2, x0 % 2, x1, x2 // 2, x2 % 2)
    => selected mode indices: (x0 // 2, v, x2 // 2)  # where v is in [0, 1, 2] (replicated by -3)
    => {x0 // 2 * 15 + v * 3 + x2 // 2 for v in [0, 1, 2]}

    Attributes
    ----------
    shape: tuple[int, ...]
        The shape of the multi-function.

    mode_shape: tuple[int, ...]
        The mode shape of the multi-function, which is a fine-grained shape of the multi-function.

    modes: tuple[int, ...]
        The modes of the image. Negative integers represent replication modes, while non-negative integers represent
        the indices of the modes.

    Methods
    -------
    __str__():
        Returns a string representation of the multi-function.

    __mul__(other: MultiFunction) -> MultiFunction:
        Multiplies two multi-functions together, returning a new multi-function that represents the composition of the
        two. If the composition can not be represented in by a (mode_shape, modes) pair, it raises a
        `LayoutOperationError`.
    """

    shape: tuple[int, ...]
    mode_shape: tuple[int, ...]
    modes: tuple[int, ...]

    @staticmethod
    def create(shape: Sequence[int], mode_shape: Sequence[int], modes: Sequence[int]) -> MultiFunction:
        return MultiFunction(
            shape=tuple(shape) if not isinstance(shape, tuple) else shape,
            mode_shape=tuple(mode_shape) if not isinstance(mode_shape, tuple) else mode_shape,
            modes=tuple(modes) if not isinstance(modes, tuple) else modes,
        )

    @cached_property
    def _image_shape(self) -> tuple[int, ...]:
        return tuple(self.mode_shape[mode] if mode >= 0 else -mode for mode in self.modes)

    @cached_property
    def size(self) -> int:
        """Returns the size of the multi-function, which is the product of the mode shape."""
        return prod(self.mode_shape)

    @cached_property
    def image_size(self) -> int:
        """Returns the size of the image of the multi-function, which is the product of the image shape."""
        return prod(self._image_shape)

    @cached_property
    def mode_groups(self) -> list[list[int]]:
        """Returns the mode groups of the multi-function.

        The mode groups are the groups of modes that are replicated together. For example, if we have a multi-function
        with modes (0, -3, 2), the mode groups will be [[0], [1, 2]].
        """
        from tilus.ir.layout.utils import get_mode_groups

        return get_mode_groups(self.shape, self.mode_shape)

    def __call__(self, x: Sequence[Int]) -> list[Expr]:
        """Returns the image of the multi-function for the given input x."""
        replicate_dims = [i for i, mode in enumerate(self.modes) if mode < 0]
        replicate_sizes = [-mode for mode in self.modes if mode < 0]
        mode_indices = index_deserialize(index_serialize(x, self.shape), self.mode_shape)
        image_indices: list[Int] = [mode_indices[mode] if mode >= 0 else 0 for mode in self.modes]

        results: list[Expr] = []
        for items in itertools.product(*[range(size) for size in replicate_sizes]):
            for i, item in enumerate(items):
                image_indices[replicate_dims[i]] = item
            results.append(index_serialize(image_indices, self._image_shape))

        return results

    def __str__(self):
        items = {
            "shape": list(self.shape),
            "mode_shape": list(self.mode_shape),
            "modes": list(self.modes),
        }
        return "multi_function(" + ", ".join(f"{k}={v}" for k, v in items.items()) + ")"

    def __eq__(self, other):
        if not isinstance(other, MultiFunction):
            return NotImplemented
        return self.shape == other.shape and self.mode_shape == other.mode_shape and self.modes == other.modes

    def __mul__(self, other: MultiFunction) -> MultiFunction:
        from tilus.ir.layout.utils import get_mode_groups

        image_shape = self._image_shape
        image_mode_groups = get_mode_groups(image_shape, other.mode_shape)
        composed_mode_shape: list[int] = []
        mode_remap: dict[int, int] = {}
        for mode, size in enumerate(self.mode_shape):
            if mode in self.modes:
                for image_mode in image_mode_groups[self.modes.index(mode)]:
                    mode_remap[image_mode] = len(composed_mode_shape)
                    composed_mode_shape.append(other.mode_shape[image_mode])
            else:
                composed_mode_shape.append(size)

        composed_modes = []
        for image_mode in other.modes:
            if image_mode < 0:
                composed_modes.append(image_mode)
            else:
                if image_mode in mode_remap:
                    composed_modes.append(mode_remap[image_mode])
                else:
                    # Replication mode, we need to add it as a negative integer
                    composed_modes.append(-other.mode_shape[image_mode])

        return multi_function(
            shape=self.shape,
            mode_shape=composed_mode_shape,
            modes=composed_modes,
        )

    def collapse(self, dims: Sequence[int]) -> MultiFunction:
        from tilus.ir.mfunction.ops import collapse

        return collapse(self, dims)

    def collapse_by_shape(self, shape: Sequence[int]) -> MultiFunction:
        from tilus.ir.mfunction.ops import collapse_by_shape

        return collapse_by_shape(self, shape)

    def cover(self, other: MultiFunction) -> bool:
        """Check whether this multi-function covers another multi-function."""
        from tilus.ir.mfunction.ops import cover

        return cover(self, other)


def canonicalize(func: MultiFunction) -> MultiFunction:
    """
    Canonicalizes the multi-function.

    This function perform the following steps to canonicalize the multi-function:
    1. filters out dimension with size 1 in the mode shape.
       Like ([2, 1, 3], [0, 2]) -> ([2, 3], [0, 1])
    2. merge consecutive modes.
       Like ([2, 3, 4, 5, 6], [0, 1, 2, 4]) -> ([24, 5, 6], [0, 2])

    Parameters
    ----------
    func: MultiFunction
        The multi-function to be canonicalized.

    Returns
    -------
    ret: MultiFunction
        The canonicalized multi-function.
    """
    if any(s == 1 for s in func.mode_shape):
        mode_remap = {}
        new_mode_shape: list[int] = []
        for mode, size in enumerate(func.mode_shape):
            if size > 1:
                mode_remap[mode] = len(new_mode_shape)
                new_mode_shape.append(size)
        new_modes = [mode_remap[mode] if mode >= 0 else mode for mode in func.modes if mode < 0 or mode in mode_remap]
        func = MultiFunction.create(shape=func.shape, mode_shape=new_mode_shape, modes=new_modes)
    if any(a + 1 == b and a >= 0 for a, b in zip(func.mode_shape[:-1], func.mode_shape[1:])):
        mode2dim = {mode: i for i, mode in enumerate(func.modes) if mode >= 0}
        mode_remap = {}
        new_mode_shape = []
        mode_count = 0
        for mode, size in enumerate(func.mode_shape):
            if mode >= 1 and mode - 1 in mode2dim and mode in mode2dim and mode2dim[mode - 1] + 1 == mode2dim[mode]:
                # merge this mode with the previous one
                new_mode_shape[-1] *= size
                mode_remap[mode] = -1  # -1 indicates that this mode is merged with the previous one
            else:
                new_mode_shape.append(size)
                mode_remap[mode] = mode_count
                mode_count += 1

        new_modes = [
            mode_remap[mode] if mode >= 0 else mode for mode in func.modes if mode < 0 or mode_remap[mode] != -1
        ]
        func = MultiFunction.create(shape=func.shape, mode_shape=new_mode_shape, modes=new_modes)
    return func


def multi_function(shape: Sequence[int], mode_shape: Sequence[int], modes: Sequence[int]) -> MultiFunction:
    """
    Create a multi-function with the given shape, mode shape, and modes.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of the multi-function.

    mode_shape: Sequence[int]
        The mode shape of the multi-function, which is a fine-grained shape of the multi-function.

    modes: Sequence[int]
        The modes of the image. Negative integers represent replication modes, while non-negative integers represent
        the indices of the modes.

    Returns
    -------
    ret: MultiFunction
        The multi-function with the given shape, mode shape, and modes.
    """
    return canonicalize(MultiFunction.create(shape=shape, mode_shape=mode_shape, modes=modes))


def visualize(func: MultiFunction) -> str:
    """Visualizes the multi-function as a string.

    The output is a table with two columns: x and Y, where x is the input to the multi-function and Y is the
    corresponding output set of integers.

    Parameters
    ----------
    func: MultiFunction
        The multi-function to be visualized.

    Returns
    -------
    ret: str
        A string representation of the multi-function in a table format.
    """
    headers = ["x", "Y"]
    rows = []
    for x in range(prod(func.mode_shape)):
        y_set = func(index_deserialize(x, func.shape))
        rows.append([int(x), str(y_set)])
    return tabulate.tabulate(rows, headers=headers, tablefmt="simple_grid")
