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
"""
Insert implicit casts into binary expressions whose operands have different dtypes.

Historically this pass also lowered `generic_*` primitive function calls to device/dtype-specific
variants. Now that primitives are registered with concrete types and dispatched at call-site,
only the binary implicit-cast rewriting remains.
"""

from typing import List

from tilus.hidet.ir.expr import BinaryExpr, Expr, cast
from tilus.hidet.ir.func import Function
from tilus.hidet.ir.functors import IRRewriter
from tilus.hidet.ir.tools import TypeInfer
from tilus.hidet.ir.type import DataType
from tilus.hidet.ir.utils.type_utils import numeric_promotion_for_all
from tilus.hidet.transforms.base import FunctionPass


def _cast_args(args: List[Expr], arg_dtypes: List[DataType], target_dtype: DataType) -> List[Expr]:
    return [a if d.name == target_dtype.name else cast(a, target_dtype) for a, d in zip(args, arg_dtypes)]


class InsertImplicitBinaryCastRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()

    def visit_Binary(self, e: BinaryExpr):
        lhs = self.visit(e.a)
        rhs = self.visit(e.b)
        lhs_dtype = self.type_infer(lhs)
        rhs_dtype = self.type_infer(rhs)
        if isinstance(lhs_dtype, DataType) and isinstance(rhs_dtype, DataType) and lhs_dtype.name != rhs_dtype.name:
            target = numeric_promotion_for_all(lhs_dtype, rhs_dtype)
            lhs, rhs = _cast_args([lhs, rhs], [lhs_dtype, rhs_dtype], target)
            if lhs is e.a and rhs is e.b:
                return e
            else:
                return e.__class__(lhs, rhs)
        else:
            return IRRewriter.visit_Binary(self, e)


class ResolveGenericPrimitiveFuncPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        rewriter = InsertImplicitBinaryCastRewriter()
        return rewriter.visit(func)


def resolve_primitive_func_pass():
    return ResolveGenericPrimitiveFuncPass()
