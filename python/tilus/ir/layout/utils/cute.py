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

from typing import Sequence, Union

from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr
from hidet.utils import prod
from tilus.extensions.hidet.ir.primitives.swizzle import swizzle
from tilus.extensions.hidet.ir.utils.index_transform import index_deserialize
from tilus.ir.layout.shared_layout import SharedLayout, Swizzle, shared_layout

Int = Union[Expr, int]
IntTuple = Int | Sequence[Union[Int, "IntTuple"]]


def is_congruent(a: IntTuple, b: IntTuple) -> bool:
    if not isinstance(a, Sequence) and not isinstance(b, Sequence):
        return True
    if len(a) != len(b):  # type: ignore[arg-type]
        return False
    return all(is_congruent(a_item, b_item) for a_item, b_item in zip(a, b))  # type: ignore[arg-type]


def tuple_multiply(a: IntTuple, b: IntTuple) -> IntTuple:
    if isinstance(a, Sequence) and isinstance(b, Sequence):
        return tuple(tuple_multiply(a_item, b_item) for a_item, b_item in zip(a, b))
    elif not isinstance(a, Sequence) and not isinstance(b, Sequence):
        return a * b
    else:
        raise ValueError(f"Invalid a or b: {a} or {b}")


def tuple_product(a: IntTuple) -> Int:
    return prod(tuple_product(item) for item in a) if isinstance(a, Sequence) else a


def tuple_sum(a: IntTuple) -> Int:
    return sum(tuple_sum(item) for item in a) if isinstance(a, Sequence) else a


def specialize(coords: IntTuple, shape: IntTuple) -> IntTuple:
    while isinstance(coords, Sequence) and len(coords) == 1:
        coords = coords[0]
    if isinstance(coords, Sequence) and isinstance(shape, Sequence):
        if len(coords) != len(shape):
            raise ValueError(f"Invalid coords or shape: {coords} or {shape}")
        return tuple(specialize(coord, shape) for coord, shape in zip(coords, shape))
    elif not isinstance(coords, Sequence) and not isinstance(shape, Sequence):
        return coords
    elif not isinstance(coords, Sequence) and isinstance(shape, Sequence):
        sizes = [tuple_product(item) for item in shape]
        return index_deserialize(coords, sizes, ranks=list(reversed(range(len(sizes)))))
    else:
        raise ValueError(f"Invalid coords or shape: {coords} or {shape}")


class CuteLayout:
    def __init__(self, shape: IntTuple, strides: IntTuple):
        self.shape: IntTuple = shape
        self.strides: IntTuple = strides

        if not is_congruent(shape, strides):
            raise ValueError("Shape and strides must be congruent")

    def __str__(self) -> str:
        return f"{self.shape}:{self.strides}"

    def __call__(self, *coords: IntTuple) -> Int:
        ret = tuple_sum(tuple_multiply(specialize(coords, self.shape), self.strides))
        return ret

    @property
    def flattened_shape(self) -> tuple[Int, ...]:
        if not isinstance(self.shape, Sequence):
            return (self.shape,)
        else:
            return tuple(tuple_product(item) for item in self.shape)


class CuteSwizzle:
    def __init__(self, bbits: int, mbase: int, sshift: int):
        self.bbits: int = bbits
        self.mbase: int = mbase
        self.sshift: int = sshift

    def __str__(self) -> str:
        return f"Swizzle<{self.bbits}, {self.mbase}, {self.sshift}>"

    def __call__(self, offset: Int) -> Int:
        if self.bbits == 0:
            return offset
        else:
            # 0xxxxYYYxxZZZxxxx
            # z_mask = ((1 << self.bbits) - 1) << self.mbase
            # y_mask = ((1 << self.bbits) - 1) << (self.mbase + self.sshift)
            # return offset ^ ((offset & y_mask) >> self.sshift)
            return swizzle(int32(offset), self.mbase, self.bbits, self.sshift)

    def as_swizzle(self) -> Swizzle:
        return Swizzle(base=self.mbase, bits=self.bbits, shift=self.sshift)


class SwizzledCuteLayout:
    def __init__(self, layout: CuteLayout, swizzle: CuteSwizzle):
        self.layout: CuteLayout = layout
        self.swizzle: CuteSwizzle = swizzle

    def __str__(self) -> str:
        return str(self.swizzle) + " ○ " + str(self.layout)

    def __call__(self, *coords: IntTuple) -> Int:
        return self.swizzle(self.layout(*coords))

    def as_shared_layout(self, tensor_shape: Sequence[int]) -> SharedLayout:
        # since cute layout use column-major order when splitting modes, we need to reverse the shape and strides
        def reverse_int_tuple(t: IntTuple) -> IntTuple:
            if isinstance(t, Sequence):
                return tuple(reverse_int_tuple(item) for item in reversed(t))
            else:
                return t

        rev_shape = reverse_int_tuple(self.layout.shape)
        rev_strides = reverse_int_tuple(self.layout.strides)

        # then, we flatten them into 1D lists
        def flatten_int_tuple(t: IntTuple) -> list[Int]:
            if isinstance(t, Sequence):
                result = []
                for item in t:
                    result.extend(flatten_int_tuple(item))
                return result
            else:
                return [t]

        flat_shape = flatten_int_tuple(rev_shape)
        flat_strides = flatten_int_tuple(rev_strides)

        mode_shape = [int(s) for s in flat_shape]
        mode_strides = [int(s) for s in flat_strides]

        return shared_layout(
            shape=tensor_shape, mode_shape=mode_shape, mode_strides=mode_strides, swizzle=self.swizzle.as_swizzle()
        )


def cute_layout(shape: IntTuple, strides: IntTuple) -> CuteLayout:
    return CuteLayout(shape, strides)
