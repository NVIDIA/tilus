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


@initialize()
def register_functions():
    from hidet.lang import asm, attrs, script  # pylint: disable=import-outside-toplevel
    from hidet.lang.types import int32

    @no_type_check
    @script
    def cuda_popc(a: int32) -> int32:
        attrs.func_kind = "cuda_internal"

        ret: int32 = 0
        asm("popc.b32 %0, %1;", outputs=[ret], inputs=[a])
        return ret

    funcs = [cuda_popc]
    for func in funcs:
        assert isinstance(func, Function)
        register_primitive_function(name=func.name, func_or_type=func)


def popc(a: Expr) -> Expr:
    """
    Count the number of set bits (population count) in a 32-bit integer.

    Parameters
    ----------
    a: Expr
        The 32-bit integer input.

    Returns
    -------
    ret: Expr
        The population count of the input integer.
    """
    return call_primitive_func("cuda_popc", args=[a])
