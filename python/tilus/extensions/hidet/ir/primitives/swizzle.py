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

from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr
from hidet.ir.primitives.func import call_primitive_func, register_primitive_function
from hidet.utils import initialize


@initialize()
def register_swizzle_primitive():
    from hidet.lang import attrs, script

    @no_type_check
    @script
    def swizzle_impl(x: int32, mbase: int32, bbits: int32, sshift: int32) -> int32:
        attrs.func_kind = "cuda_internal"
        attrs.func_name = "swizzle"

        return (x ^ ((x & (((1 << bbits) - 1) << (mbase + sshift))) >> sshift)) if bbits > 0 else x

    register_primitive_function(swizzle_impl.name, swizzle_impl)


def swizzle(x: Expr, mbase: Expr | int, bbits: Expr | int, sshift: Expr | int) -> Expr:
    """
    Using the swizzle from cute:

    0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
                                  ^--^ MBase is the number of least-sig bits to keep constant
                     ^-^       ^-^     BBits is the number of bits in the mask
                       ^---------^     SShift is the distance to shift the YYY mask
                                          (pos shifts YYY to the right, neg shifts YYY to the left)

    e.g. Given
    0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx

    the result is
    0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ `xor` YY
    """
    return call_primitive_func("swizzle", [x, int32(mbase), int32(bbits), int32(sshift)])
