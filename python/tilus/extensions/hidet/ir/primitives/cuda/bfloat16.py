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

from hidet.ir.expr import Expr
from hidet.ir.func import Function
from hidet.ir.primitives.func import call_primitive_func, register_primitive_function
from hidet.ir.stmt import BlackBoxStmt
from hidet.utils import initialize


@initialize()
def register_functions():
    from hidet.lang import attrs, script  # pylint: disable=import-outside-toplevel
    from hidet.lang.types import uint32, void_p

    template = r"__nv_bfloat162 out = __hmul2(*reinterpret_cast<__nv_bfloat162*>({}), *reinterpret_cast<const __nv_bfloat162*>({})); *reinterpret_cast<__nv_bfloat162*>({}) = out;"

    @no_type_check
    @script
    def mul_bf16x2_(d: void_p, a: uint32, b: uint32):
        attrs.func_kind = "cuda_internal"
        attrs.func_name = "mul_bf16x2"

        # the following inst only supports for sm_90 and later
        # asm('mul.rn.bf16x2 %0, %1, %2;', outputs=[cast(d, ~uint32)[0]], inputs=[a, b], is_volatile=True)

        BlackBoxStmt(template, ~a, ~b, d)

    funcs = [mul_bf16x2_]
    for func in funcs:
        assert isinstance(func, Function)
        register_primitive_function(name=func.name, func_or_type=func)


def mul_bf16x2(d: Expr, a: Expr, b: Expr) -> Expr:
    """
    Multiply two bf16x2 values and store the result in `d`.

    Expect `d` to be an uint32 pointer while `a` an `b` are uint32 values, all of them will be interpreted as bf16x2.

    Parameters
    ----------
    d: Expr
        The pointer to the bf16x2 result, stored with uint32 data type.
    a: Expr
        The first bf16x2 operand stored with uint32 data type.
    b: Expr
        The second bf16x2 operand stored with uint32 data type.
    """
    return call_primitive_func("mul_bf16x2", args=[d, a, b])
