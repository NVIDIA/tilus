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

import warnings
from typing import Any

import tvm_ffi
from tvm_ffi.dataclasses import py_class

from tilus.hidet.ir.type import DataType


@py_class("tilus.hidet.ir.dtypes.IntInfo", frozen=True, structural_eq="tree")
class IntInfo(tvm_ffi.Object):
    bits: int
    max: int
    min: int
    dtype: DataType


@py_class("tilus.hidet.ir.dtypes.IntegerType", frozen=True, structural_eq="tree")
class IntegerType(DataType):
    imin: int
    imax: int

    def is_float(self) -> bool:
        return False

    def is_integer(self) -> bool:
        return True

    def is_complex(self) -> bool:
        return False

    def is_vector(self) -> bool:
        return False

    def is_boolean(self) -> bool:
        return False

    def constant(self, value: Any):
        from tilus.hidet.ir.expr import Constant, constant  # noqa: PLC0415

        if isinstance(value, Constant):
            value = value.value
        if isinstance(value, float):
            warnings.warn(
                "Converting float to integer when creating {} constant: {}.".format(self.name, value),
                stacklevel=2,
            )
        value = int(value)
        if not self.imin <= value <= self.imax:
            raise ValueError("Value {} is out of range for {}.".format(value, self.name))
        return constant(value, self)

    def signedness(self) -> bool:
        return self.imin < 0

    @property
    def one(self):
        return self.constant(1)

    @property
    def zero(self):
        return self.constant(0)

    @property
    def min_value(self):
        return self.constant(self.imin)

    @property
    def max_value(self):
        return self.constant(self.imax)

    def iinfo(self) -> IntInfo:
        return IntInfo(bits=self.nbytes * 8, max=self.imax, min=self.imin, dtype=self)


int8 = IntegerType(name="int8", short_name="i8", nbytes=1, imin=-128, imax=127)
int16 = IntegerType(name="int16", short_name="i16", nbytes=2, imin=-32768, imax=32767)
int32 = IntegerType(name="int32", short_name="i32", nbytes=4, imin=-2147483648, imax=2147483647)
int64 = IntegerType(name="int64", short_name="i64", nbytes=8, imin=-9223372036854775808, imax=9223372036854775807)

uint8 = IntegerType(name="uint8", short_name="u8", nbytes=1, imin=0, imax=255)
uint16 = IntegerType(name="uint16", short_name="u16", nbytes=2, imin=0, imax=65535)
uint32 = IntegerType(name="uint32", short_name="u32", nbytes=4, imin=0, imax=4294967295)
# imax is stored as a C int64_t by the FFI layer; uint64's theoretical max
# (2**64 - 1) overflows. Cap at int64 max — range checks for values in the
# upper half of uint64 will need a different path if they become important.
uint64 = IntegerType(name="uint64", short_name="u64", nbytes=8, imin=0, imax=9223372036854775807)

i8 = int8
i16 = int16
i32 = int32
i64 = int64

u8 = uint8
u16 = uint16
u32 = uint32
u64 = uint64
