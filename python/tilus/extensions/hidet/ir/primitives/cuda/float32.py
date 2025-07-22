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
from hidet.utils import initialize


def resolve_ex2_name(ftz: bool) -> str:
    """
    Resolve the function name based on whether FTZ (Flush-to-Zero) is enabled.
    """
    return "cuda_ex2_approx{ftz}_f32".format(ftz="_ftz" if ftz else "")


def resolve_mul_name(rnd: str, ftz: bool) -> str:
    """
    Resolve the function name for multiplication based on whether FTZ (Flush-to-Zero) is enabled.
    """
    return "cuda_mul_{rnd}{ftz}".format(rnd="_" + rnd, ftz="_ftz" if ftz else "")


@initialize()
def register_functions():
    from hidet.lang import asm, attrs, script  # pylint: disable=import-outside-toplevel
    from hidet.lang.types import f32

    for ftz in [False, True]:
        func_name = resolve_ex2_name(ftz)

        template = "ex2.approx{ftz}.f32 %0, %1;".format(ftz=".ftz" if ftz else "")

        @no_type_check
        @script
        def f32_ex2(a: f32) -> f32:
            attrs.func_kind = "cuda_internal"
            attrs.func_name = func_name
            # the following inst only supports for sm_90 and later
            ret: f32 = 0.0
            asm(template, outputs=[ret], inputs=[a], is_volatile=False)
            return ret

        assert isinstance(f32_ex2, Function)

        register_primitive_function(name=f32_ex2.name, func_or_type=f32_ex2)

    for rnd in ["rn", "rz", "rm", "rp"]:
        for ftz in [False, True]:
            func_name = resolve_mul_name(rnd, ftz)

            template = "mul{rnd}{ftz}.f32 %0, %1, %2;".format(
                rnd="." + rnd if rnd != "rn" else "",
                ftz=".ftz" if ftz else "",
            )

            @no_type_check
            @script
            def f32_mul(a: f32, b: f32) -> f32:
                attrs.func_kind = "cuda_internal"
                attrs.func_name = func_name
                ret: f32 = 0.0
                asm(template, outputs=[ret], inputs=[a, b], is_volatile=False)
                return ret

            assert isinstance(f32_mul, Function)

            register_primitive_function(name=f32_mul.name, func_or_type=f32_mul)


def ex2(a: Expr, ftz: bool = True) -> Expr:
    """
    Compute the approximate exponential of a float32 value `a` using the `ex2.approx` instruction.

    Parameters
    ----------
    a: Expr
        The input float32 value for which the exponential is computed.
    ftz: bool
        Whether to use the fast-approximate version of the exponential function.
        Defaults to True, which uses the `ex2.approx.ftz` instruction.

    Returns
    -------
    ret: Expr
        The result of the exponential computation as a float32 expression.
    """
    func_name = resolve_ex2_name(ftz)
    return call_primitive_func(func_name, args=[a])


def mul(a: Expr, b: Expr, rnd: str = "rn", ftz: bool = True) -> Expr:
    """
    Compute the product of two float32 values `a` and `b` using the `mul{.ftz}` instruction.

    Parameters
    ----------
    a: Expr
        The first float32 value.
    b: Expr
        The second float32 value.
    rnd: str
        The rounding mode for the multiplication.
        - 'rn' for round to nearest even (default).
        - 'rz' for round towards zero.
        - 'rm' for round towards minus infinity.
        - 'rp' for round towards plus infinity.
    ftz: bool
        Whether to use the fast-approximate version of the multiplication function.
        Defaults to True, which uses the `.ftz` modifier.

    Returns
    -------
    ret: Expr
        The result of the multiplication as a float32 expression.
    """
    if rnd not in ["rn", "rz", "rm", "rp"]:
        raise ValueError(f"Invalid rounding mode: {rnd}. Must be one of 'rn', 'rz', 'rm', or 'rp'.")
    func_name = resolve_mul_name(rnd, ftz)
    return call_primitive_func(func_name, args=[a, b])
