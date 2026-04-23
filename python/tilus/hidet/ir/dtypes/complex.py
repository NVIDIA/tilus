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

from typing import Any

from tvm_ffi.dataclasses import py_class

from tilus.hidet.ir.dtypes.floats import float32, float64
from tilus.hidet.ir.type import DataType


@py_class("tilus.hidet.ir.dtypes.ComplexType", frozen=True, structural_eq="tree")
class ComplexType(DataType):
    base_dtype: DataType

    def is_float(self) -> bool:
        return False

    def is_integer(self) -> bool:
        return False

    def is_vector(self) -> bool:
        return False

    def is_complex(self) -> bool:
        return True

    def is_boolean(self) -> bool:
        return False

    def constant(self, value: Any):
        from tilus.hidet.ir.expr import Constant, constant  # noqa: PLC0415

        if isinstance(value, Constant):
            value = value.value
        if isinstance(value, complex):
            return constant(value, const_type=self)
        if isinstance(value, (int, float)):
            return constant(complex(value, 0.0), const_type=self)
        raise RuntimeError("Invalid constant value for complex type: {}".format(value))

    @property
    def one(self):
        return self.constant(1.0 + 0.0j)

    @property
    def zero(self):
        return self.constant(0.0 + 0.0j)

    @property
    def min_value(self):
        raise RuntimeError("Complex type has no minimum value")

    @property
    def max_value(self):
        raise RuntimeError("Complex type has no maximum value")


complex64 = ComplexType(name="complex64", short_name="c64", nbytes=2 * float32.nbytes, base_dtype=float32)
complex128 = ComplexType(name="complex128", short_name="c128", nbytes=2 * float64.nbytes, base_dtype=float64)

c64 = complex64
c128 = complex128
