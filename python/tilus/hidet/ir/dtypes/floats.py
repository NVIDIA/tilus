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

import numpy as np
import tvm_ffi
from tvm_ffi.dataclasses import py_class

from tilus.hidet.ir.type import DataType


@py_class("tilus.hidet.ir.dtypes.FloatInfo", frozen=True, structural_eq="tree")
class FloatInfo(tvm_ffi.Object):
    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float
    dtype: DataType


@py_class("tilus.hidet.ir.dtypes.FloatType", frozen=True, structural_eq="tree")
class FloatType(DataType):
    fmin: float
    fmax: float
    eps: float | None = None
    smallest_normal: float | None = None
    mantissa_nbits: int | None = None
    exponent_nbits: int | None = None

    def is_float(self) -> bool:
        return True

    def is_integer(self) -> bool:
        return False

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
        value = float(value)
        if value > self.fmax:
            warnings.warn(
                "Constant value {} is larger than the maximum value {} of data type {}. "
                "Truncated to maximum value of {}.".format(value, self.fmax, self.name, self.name),
                stacklevel=2,
            )
            value = self.fmax
        if value < self.fmin:
            warnings.warn(
                "Constant value {} is smaller than the minimum value {} of data type {}. "
                "Truncated to minimum value of {}.".format(value, self.fmin, self.name, self.name),
                stacklevel=2,
            )
            value = self.fmin
        return constant(value, self)

    @property
    def one(self):
        return self.constant(1.0)

    @property
    def zero(self):
        return self.constant(0.0)

    @property
    def min_value(self):
        return self.constant(self.fmin)

    @property
    def max_value(self):
        return self.constant(self.fmax)

    def finfo(self) -> FloatInfo:
        return FloatInfo(
            bits=self.nbytes * 8,
            eps=self.eps if self.eps is not None else float("nan"),
            max=self.fmax,
            min=self.fmin,
            smallest_normal=self.smallest_normal if self.smallest_normal is not None else float("nan"),
            dtype=self,
        )


float8_e4m3 = FloatType(
    name="float8_e4m3",
    short_name="f8e4m3",
    nbytes=1,
    fmin=float(-448),
    fmax=float(448),
    eps=2 ** (-2),
    smallest_normal=2 ** (-6),
    mantissa_nbits=3,
    exponent_nbits=4,
)
float8_e5m2 = FloatType(
    name="float8_e5m2",
    short_name="f8e5m2",
    nbytes=1,
    fmin=float(-57344),
    fmax=float(57344),
    eps=2 ** (-2),
    smallest_normal=2 ** (-14),
    mantissa_nbits=2,
    exponent_nbits=5,
)
float16 = FloatType(
    name="float16",
    short_name="f16",
    nbytes=2,
    fmin=float(np.finfo(np.float16).min),
    fmax=float(np.finfo(np.float16).max),
    eps=float(np.finfo(np.float16).eps),
    smallest_normal=float(np.finfo(np.float16).tiny),
    mantissa_nbits=10,
    exponent_nbits=5,
)
float32 = FloatType(
    name="float32",
    short_name="f32",
    nbytes=4,
    fmin=float(np.finfo(np.float32).min),
    fmax=float(np.finfo(np.float32).max),
    eps=float(np.finfo(np.float32).eps),
    smallest_normal=float(np.finfo(np.float32).tiny),
    mantissa_nbits=23,
    exponent_nbits=8,
)
float64 = FloatType(
    name="float64",
    short_name="f64",
    nbytes=8,
    fmin=float(np.finfo(np.float64).min),
    fmax=float(np.finfo(np.float64).max),
    eps=float(np.finfo(np.float64).eps),
    smallest_normal=float(np.finfo(np.float64).tiny),
    mantissa_nbits=52,
    exponent_nbits=11,
)
bfloat16 = FloatType(
    name="bfloat16",
    short_name="bf16",
    nbytes=2,
    fmin=-3.4e38,
    fmax=3.4e38,
    mantissa_nbits=7,
    exponent_nbits=8,
)
tfloat32 = FloatType(
    name="tfloat32",
    short_name="tf32",
    nbytes=4,
    fmin=-3.4e38,
    fmax=3.4e38,
    mantissa_nbits=10,
    exponent_nbits=8,
)

f8e4m3 = float8_e4m3
f8e5m2 = float8_e5m2
f16 = float16
f32 = float32
f64 = float64
bf16 = bfloat16
tf32 = tfloat32
