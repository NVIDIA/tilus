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
from typing import Optional

from tilus.hidet.ir.expr import Expr
from tilus.hidet.ir.func import Function
from tilus.hidet.ir.primitives.func import call_primitive_func, register_primitive_function
from tilus.hidet.utils import initialize


def resolve_func_name(mode: Optional[str] = None) -> str:
    if mode is None:
        return "prmt_b32"
    else:
        return "prmt_b32_{}".format(mode)


def resolve_inst_template(mode: Optional[str] = None) -> str:
    if mode is None:
        return "prmt.b32 %0, %1, %2, %3;"
    else:
        return "prmt.b32.{} %0, %1, %2, %3;".format(mode)


@initialize()
def register_functions():
    from tilus.hidet.lang import asm, attrs, cast, script  # pylint: disable=import-outside-toplevel
    from tilus.hidet.lang.types import uint32, void_p

    for mode in [None, "f4e", "b4e", "rc8", "ecl", "ecr", "rc16"]:
        template = resolve_inst_template(mode)

        @script
        def prmt_primitive(d: void_p, a: uint32, b: uint32, c: uint32):
            attrs.func_kind = "cuda_internal"
            attrs.func_name = resolve_func_name(mode)

            asm(template, outputs=[cast(d, ~uint32)[0]], inputs=[a, b, c], is_volatile=True)

        assert isinstance(prmt_primitive, Function)
        register_primitive_function(name=prmt_primitive.name, func_or_type=prmt_primitive)


def prmt(d: Expr, a: Expr, b: Expr, c: Expr, *, mode: Optional[str] = None):
    """
    Perform a byte-level permutation operation on two 32-bit values and store the result in `d`.

    The permutation operation is determined by the permutation mode `mode`.

    See Also the PTX ISA documentation for the `prmt` instruction for more information:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt

    Parameters
    ----------
    d: Expr
        The pointer to the 32-bit result.
    a: Expr
        The first uint32 operand.
    b: Expr
        The second uint32 operand.
    c: Expr
        The control operand.
    mode: Optional[str]
        The permutation mode. If not provided, the default mode is used.
    """
    assert mode in [None, "f4e", "b4e", "rc8", "ecl", "ecr", "rc16"]
    return call_primitive_func(resolve_func_name(mode), args=[d, a, b, c])
