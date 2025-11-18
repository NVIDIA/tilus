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
from typing import no_type_check

from hidet.ir.dtypes import DataType, f32, float32, uint8
from hidet.ir.expr import Expr
from hidet.ir.primitives.func import call_primitive_func, register_primitive_function
from hidet.utils import initialize

from tilus.extensions.hidet.ir.dtypes.floats_subbyte import FloatSubbyteType
from tilus.extensions.hidet.ir.expr import reinterpret
from tilus.extensions.hidet.ir.primitives.cuda.float32 import mul as f32_mul


def register_float_cast_functions(dtype: FloatSubbyteType) -> None:
    from hidet.lang import attrs, script  # pylint: disable=import-outside-toplevel
    from hidet.lang.types import uint32

    nbits = dtype.nbits
    exponent_nbits = dtype.exponent_nbits
    mantissa_nbits = dtype.mantissa_nbits

    assert nbits == 1 + exponent_nbits + mantissa_nbits
    assert mantissa_nbits >= 1 and exponent_nbits >= 1 and nbits <= 8

    def pow2_of_float_as_uint32(p: int) -> int:
        return (p + 127) << 23

    def ones(n: int) -> int:
        return (1 << n) - 1

    @no_type_check
    @script
    def cast_from_f32_(src: f32) -> uint8:
        attrs.func_kind = "cuda_internal"
        attrs.func_name = "cast_f32_to_{}".format(dtype.short_name)

        src_uint32: uint32 = reinterpret(src, uint32)
        sign: uint8 = ((src_uint32 >> 31) & 1) << (nbits - 1)
        exponents = (src_uint32 >> 23) & 0xFF
        mantissa = src_uint32 & 0x7FFFFF

        if exponents == 0xFF:
            # NaN or inf, set to maximum value
            return uint8(sign | ((1 << nbits) - 1))
        if exponents == 0x00:
            # underflow to 0
            return sign

        # the normalized number = 1.mantissa * 2^(exponents - 127)
        # update exp_value = exponents - 127 and mantissa = 1.mantissa
        exp_value = exponents - 127
        mantissa = mantissa | 0x800000  # add implicit 1 in the normalized number

        # Rounding preparation
        # 1        8 23
        # . ........ .LRSSSSSSSSSSSSSSSSSSSS
        # .    ..... ..
        # 1        e m
        # shift = 23 - m
        # L: lsb (least significant bit)
        # R: round bit
        # S: sticky bits (all bits after R)
        shift = 23 - mantissa_nbits
        round_bit = (mantissa >> (shift - 1)) & 1
        sticky_bits = (mantissa & ones(shift - 1)) != 0
        lsb = (mantissa >> shift) & 1

        # Perform rounding to nearest, ties to even
        if round_bit != 0 and (sticky_bits != 0 or lsb != 0):
            mantissa += 1 << shift
            if mantissa & 0x01000000:
                mantissa >>= 1
                exp_value += 1

        # adjust exponent and mantissa to the sub-byte float format
        mantissa >>= shift
        exp_value += (1 << (exponent_nbits - 1)) - 1

        if exp_value >= (1 << exponent_nbits):
            # Handle overflow
            return uint8(sign | ones(nbits - 1))
        elif exp_value <= 0:
            # handle underflow
            if exp_value >= 1 - mantissa_nbits:
                # underflow to denormalized number
                mantissa >>= 1 - exp_value
                exp_value = 0
            else:
                # underflow to 0
                return sign
        else:
            # normalized number, remove the implicit 1
            mantissa ^= uint8(1 << mantissa_nbits)

        # assemble the sub-byte float number
        return uint8(sign | (exp_value << mantissa_nbits) | mantissa)

    @no_type_check
    @script
    def cast_to_f32_(src: uint8) -> f32:
        attrs.func_kind = "cuda_internal"
        attrs.func_name = "cast_{}_to_f32".format(dtype.short_name)

        sign: uint32 = (src & uint8(1 << (nbits - 1))) << (32 - nbits)
        exponent_mantissa: uint32 = (src & uint8((1 << (nbits - 1)) - 1)) << (23 - mantissa_nbits)
        dst_uint32: uint32 = sign | exponent_mantissa
        dst_f32: f32 = reinterpret(dst_uint32, float32)
        e_adjust_pow_uint32: uint32 = uint32(pow2_of_float_as_uint32(128 - (1 << (exponent_nbits - 1))))
        # we can not flush the subnormal to zero
        dst_f32 = f32_mul(dst_f32, reinterpret(e_adjust_pow_uint32, float32), ftz=False)

        return dst_f32

    functions = [cast_from_f32_, cast_to_f32_]

    for func in functions:
        register_primitive_function(func.name, func)


@initialize()
def register_functions():
    from tilus.extensions.hidet.ir.dtypes import (
        float3_e1m1,
        float4_e2m1,
        float5_e2m2,
        float5_e3m1,
        float6_e2m3,
        float6_e3m2,
        float6_e4m1,
        float7_e2m4,
        float7_e3m3,
        float7_e4m2,
        float7_e5m1,
    )

    for dtype in [
        float7_e5m1,
        float7_e4m2,
        float7_e3m3,
        float7_e2m4,
        float6_e4m1,
        float6_e3m2,
        float6_e2m3,
        float5_e3m1,
        float5_e2m2,
        float4_e2m1,
        float3_e1m1,
    ]:
        register_float_cast_functions(dtype)


def cast_subbyte_float_from_f32(src: Expr, dst_dtype: DataType) -> Expr:
    """Cast f32 to a sub-byte float number (represented in the low bits of uint8). Rounding to nearest, ties to even."""
    func_name = "cast_f32_to_{}".format(dst_dtype.short_name)
    return call_primitive_func(func_name, [src])


def cast_subbyte_float_to_f32(src: Expr, src_dtype: DataType) -> Expr:
    """Cast a sub-byte float number (represented in the low bits of uint8) to f32."""
    func_name = "cast_{}_to_f32".format(src_dtype.short_name)
    return call_primitive_func(func_name, [src])
