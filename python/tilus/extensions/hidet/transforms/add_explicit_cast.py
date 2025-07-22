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
from typing import List, Sequence

from hidet.ir import BufferStoreStmt, TensorPointerType
from hidet.ir.expr import Call, Expr, cast
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import AssignStmt, LaunchKernelStmt, LetStmt, Stmt
from hidet.ir.tools import TypeInfer
from hidet.ir.type import BaseType, DataType, FuncType, PointerType, type_equal
from hidet.transforms.base import FunctionPass
from hidet.utils import same_list

from tilus.extensions.hidet.ir.type import get_base_type


class AddExplicitCastRewriter(IRRewriter):
    def __init__(self):
        super().__init__(use_memo=False)
        self.type_infer = TypeInfer()

    def process(self, expr: Expr, target_type: BaseType) -> Expr:
        source_type = self.type_infer(expr)

        # If we are doing pointer cast and the two types are different
        cond1 = isinstance(target_type, (TensorPointerType, PointerType)) and not type_equal(source_type, target_type)

        # If we are assigning one data type to another data type
        cond2 = (
            isinstance(source_type, DataType)
            and target_type.is_data_type()
            and not type_equal(source_type, target_type)
        )

        perform_explicit_cast = cond1 or cond2

        if perform_explicit_cast:
            processed = cast(expr, target_type)
        else:
            processed = expr
        return processed

    def process_list(self, exprs: Sequence[Expr], target_types: List[BaseType]) -> list[Expr]:
        return [self.process(expr, target_type) for expr, target_type in zip(exprs, target_types)]

    def visit_Call(self, e: Call) -> Expr:
        func_type = e.func_var.type
        assert isinstance(func_type, FuncType)
        args = self.process_list(self.visit(e.args), func_type.param_types)
        if same_list(args, e.args):
            return e
        else:
            return Call(e.func_var, tuple(args))

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt) -> Stmt:
        func_type = stmt.func_var.type
        assert isinstance(func_type, FuncType) and len(func_type.param_types) == len(stmt.args)
        args = self.process_list(self.visit(stmt.args), func_type.param_types)
        if same_list(args, stmt.args):
            return stmt
        else:
            return LaunchKernelStmt(
                func_var=stmt.func_var,
                args=args,
                grid_dim=stmt.grid_dim,
                block_dim=stmt.block_dim,
                cluster_dim=stmt.cluster_dim,
                shared_mem=stmt.shared_mem_bytes,
                target=stmt.target,
            )

    def visit_AssignStmt(self, stmt: AssignStmt) -> Stmt:
        value = self.process(stmt.value, stmt.var.type)
        if value is stmt.value:
            return stmt
        else:
            return AssignStmt(stmt.var, value)

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt) -> Stmt:
        value = stmt.value
        assert isinstance(value, Expr)
        value = self.process(value, get_base_type(self.type_infer(stmt.buf)))
        if value is stmt.value:
            return stmt
        else:
            return BufferStoreStmt(self.visit(stmt.buf), self.visit(stmt.indices), value)

    def visit_LetStmt(self, stmt: LetStmt) -> Stmt:
        bind_values = self.process_list(stmt.bind_values, [bind_var.type for bind_var in stmt.bind_vars])
        body = self.visit(stmt.body)
        if same_list(bind_values, stmt.bind_values) and body is stmt.body:
            return stmt
        else:
            return LetStmt(stmt.bind_vars, bind_values, body)


class AddExplicitPointerCastPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        rewriter = AddExplicitCastRewriter()
        return rewriter.visit_Function(func)


def add_explicit_cast_pass() -> FunctionPass:
    return AddExplicitPointerCastPass()
