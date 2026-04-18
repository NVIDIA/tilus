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
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional, Sequence, Union

from tilus.hidet.ir.dtypes import (
    bf16,
    bfloat16,
    bfloat16x2,
    boolean,
    f16,
    f16x2,
    f32,
    f32x2,
    f64,
    float8_e4m3,
    float8_e5m2,
    float16,
    float16x2,
    float32,
    float32x2,
    float64,
    i1,
    i2,
    i4,
    i8,
    i16,
    i32,
    i64,
    int8,
    int16,
    int32,
    int64,
    tf32,
    tfloat32,
    u1,
    u2,
    u4,
    u8,
    u16,
    u32,
    u64,
    uint8,
    uint16,
    uint32,
    uint64,
)
from tilus.hidet.ir.expr import Dereference, Expr, Var, address, bitwise_not, cast, deref, view
from tilus.hidet.ir.func import Function
from tilus.hidet.ir.stmt import DeclareScope, ForStmtAttr, asm
from tilus.hidet.ir.type import BaseType, DataType, PointerType, TensorType, VoidType, data_type, void_p
from tilus.hidet.lang.constructs import meta
from tilus.hidet.lang.constructs.declare import (
    as_tensor_pointer,
    register_tensor,
    shared_tensor,
    tensor,
    tensor_pointer,
)
from tilus.hidet.lang.constructs.loops import grid, range
from tilus.hidet.lang.script import script, script_module

void = VoidType()

# def var_of_function(func: Function) -> Var:
#     # pylint: disable=import-outside-toplevel
#     from hidet.lang.script import ScriptModuleContext
#
#     if not isinstance(func, Function):
#         raise ValueError('Expect a hidet.ir.Function, got {}.'.format(type(func).__name__))
#     ctx = ScriptModuleContext.current_context()
#     func_var: Optional[Var] = ctx.lookup(func.name)
#     if func_var is None:
#         raise ValueError('Function has not been defined in current script module.')
#     return func_var
