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

import math

from tvm_ffi.dataclasses import py_class

from tilus.hidet.ir.dtypes.floats import FloatInfo, FloatType


@py_class("tilus.hidet.ir.dtypes.FloatSubbyteType", frozen=True, structural_eq="tree")
class FloatSubbyteType(FloatType):
    subbyte_nbits: int
    exponent_bias: int

    @property
    def nbits(self) -> int:
        return self.subbyte_nbits

    def finfo(self) -> FloatInfo:
        return FloatInfo(
            bits=self.subbyte_nbits,
            eps=self.eps if self.eps is not None else float("nan"),
            max=self.fmax,
            min=self.fmin,
            smallest_normal=self.smallest_normal if self.smallest_normal is not None else float("nan"),
            dtype=self,
        )


def _finfo_for(e_bits: int, m_bits: int, e_bias: int):
    if e_bits == 5:
        # reserve all-ones exponent for 'inf' in 5-exponent-bit variants
        max_value = math.pow(2.0, (1 << e_bits) - 1 - 1 - e_bias) * (2.0 - math.pow(2.0, -m_bits))
    else:
        max_value = math.pow(2.0, (1 << e_bits) - 1 - e_bias) * (2.0 - math.pow(2.0, -m_bits))
    return {
        "fmax": max_value,
        "fmin": -max_value,
        "eps": math.pow(2.0, -m_bits),
        "smallest_normal": math.pow(2.0, 1 - e_bias),
    }


def _make_float_subbyte(name: str, short_name: str, nbits: int, e_bits: int, m_bits: int) -> FloatSubbyteType:
    e_bias = (1 << (e_bits - 1)) - 1
    limits = _finfo_for(e_bits, m_bits, e_bias)
    return FloatSubbyteType(
        name=name,
        short_name=short_name,
        nbytes=-1,
        fmin=limits["fmin"],
        fmax=limits["fmax"],
        eps=limits["eps"],
        smallest_normal=limits["smallest_normal"],
        mantissa_nbits=m_bits,
        exponent_nbits=e_bits,
        subbyte_nbits=nbits,
        exponent_bias=e_bias,
    )


# float7
f7e5m1 = float7_e5m1 = _make_float_subbyte("float7_e5m1", "f7e5m1", 7, 5, 1)
f7e4m2 = float7_e4m2 = _make_float_subbyte("float7_e4m2", "f7e4m2", 7, 4, 2)
f7e3m3 = float7_e3m3 = _make_float_subbyte("float7_e3m3", "f7e3m3", 7, 3, 3)
f7e2m4 = float7_e2m4 = _make_float_subbyte("float7_e2m4", "f7e2m4", 7, 2, 4)

# float6
f6e4m1 = float6_e4m1 = _make_float_subbyte("float6_e4m1", "f6e4m1", 6, 4, 1)
f6e3m2 = float6_e3m2 = _make_float_subbyte("float6_e3m2", "f6e3m2", 6, 3, 2)
f6e2m3 = float6_e2m3 = _make_float_subbyte("float6_e2m3", "f6e2m3", 6, 2, 3)

# float5
f5e3m1 = float5_e3m1 = _make_float_subbyte("float5_e3m1", "f5e3m1", 5, 3, 1)
f5e2m2 = float5_e2m2 = _make_float_subbyte("float5_e2m2", "f5e2m2", 5, 2, 2)
f5e1m3 = float5_e1m3 = _make_float_subbyte("float5_e1m3", "f5e1m3", 5, 1, 3)

# float4
f4e2m1 = float4_e2m1 = _make_float_subbyte("float4_e2m1", "f4e2m1", 4, 2, 1)
f4e1m2 = float4_e1m2 = _make_float_subbyte("float4_e1m2", "f4e1m2", 4, 1, 2)

# float3
f3e1m1 = float3_e1m1 = _make_float_subbyte("float3_e1m1", "f3e1m1", 3, 1, 1)
