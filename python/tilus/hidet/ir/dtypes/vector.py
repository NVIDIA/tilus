# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Sequence

from tvm_ffi.dataclasses import py_class

from tilus.hidet.ir.type import DataType

from .boolean import boolean
from .floats import bfloat16, float16, float32
from .integer import int8, uint8, uint32
from .integer_subbyte import int4b, uint4b


@py_class("tilus.hidet.ir.dtypes.VectorType", frozen=True, structural_eq="tree")
class VectorType(DataType):
    lane_type: DataType
    num_lanes: int

    def is_float(self) -> bool:
        return False

    def is_integer(self) -> bool:
        return False

    def is_vector(self) -> bool:
        return True

    def is_complex(self) -> bool:
        return False

    def is_boolean(self) -> bool:
        return False

    def constant(self, value: Sequence[Any]):
        from tilus.hidet.ir.expr import constant  # noqa: PLC0415

        lanes = [self.lane_type.constant(v) for v in value]
        if len(lanes) != self.num_lanes:
            raise ValueError("Invalid vector constant, expect {} elements, got {}".format(self.num_lanes, len(lanes)))
        return constant(tuple(lanes), self)

    @property
    def one(self):
        return self.constant([self.lane_type.one] * self.num_lanes)

    @property
    def zero(self):
        return self.constant([self.lane_type.zero] * self.num_lanes)

    @property
    def min_value(self):
        return self.constant([self.lane_type.min_value] * self.num_lanes)

    @property
    def max_value(self):
        return self.constant([self.lane_type.max_value] * self.num_lanes)


def _make_vector(lane: DataType, num: int) -> VectorType:
    name = "{}x{}".format(lane.name, num)
    short_name = "{}x{}".format(lane.short_name, num)
    nbytes = lane.nbytes * num if not lane.is_integer_subbyte() else lane.nbits * num // 8
    return VectorType(name=name, short_name=short_name, nbytes=nbytes, lane_type=lane, num_lanes=num)


int8x4 = _make_vector(int8, 4)
i8x4 = int8x4

uint8x4 = _make_vector(uint8, 4)
u8x4 = uint8x4

float32x1 = _make_vector(float32, 1)
f32x1 = float32x1

float32x2 = _make_vector(float32, 2)
f32x2 = float32x2

float32x4 = _make_vector(float32, 4)
f32x4 = float32x4

float32x8 = _make_vector(float32, 8)
f32x8 = float32x8

float16x1 = _make_vector(float16, 1)
f16x1 = float16x1

float16x2 = _make_vector(float16, 2)
f16x2 = float16x2

float16x4 = _make_vector(float16, 4)
f16x4 = float16x4

float16x8 = _make_vector(float16, 8)
f16x8 = float16x8

int4bx2 = _make_vector(int4b, 2)
i4x2 = int4bx2

uint4bx2 = _make_vector(uint4b, 2)
u4x2 = uint4bx2

int4bx8 = _make_vector(int4b, 8)
i4x8 = int4bx8

uint4bx8 = _make_vector(uint4b, 8)
u4x8 = uint4bx8

bfloat16x2 = _make_vector(bfloat16, 2)

uint32x1 = _make_vector(uint32, 1)
uint32x2 = _make_vector(uint32, 2)
uint32x4 = _make_vector(uint32, 4)


def vectorize(base_dtype: DataType, num_lanes: int) -> VectorType:
    table = {
        (float32, 1): float32x1,
        (float32, 2): float32x2,
        (float32, 4): float32x4,
        (float32, 8): float32x8,
        (float16, 1): float16x1,
        (float16, 2): float16x2,
        (float16, 4): float16x4,
        (float16, 8): float16x8,
        (int8, 4): int8x4,
        (uint8, 4): uint8x4,
        (boolean, 4): int8x4,
        (bfloat16, 2): bfloat16x2,
    }
    if (base_dtype, num_lanes) in table:
        return table[(base_dtype, num_lanes)]
    raise ValueError("Cannot vectorize {}x{}".format(base_dtype, num_lanes))
