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

from hidet.ir.expr import Var

from tilus.ir.func import Function
from tilus.ir.functors import IRRewriter
from tilus.ir.stmt import AssignStmt, LetStmt, Stmt
from tilus.transforms.base import Pass
from tilus.utils import same_list


class LetPropogationRewriter(IRRewriter):
    def __init__(self) -> None:
        super().__init__()
        self.params: list[Var] = []
        self.let_vars: set[Var] = set()

    def visit_Function(self, func: Function) -> Function:
        self.params.extend(func.params)
        self.let_vars.update(func.params)  # we assume function parameters are immutable in this pass
        body = self.visit(func.body)
        grid_blocks = self.visit(func.metadata.grid_blocks)
        if func.metadata.analysis:
            analysis = func.metadata.analysis
            lower_bound = self.visit(analysis.lower_bound)
            upper_bound = self.visit(analysis.upper_bound)
            divisibility = self.visit(analysis.divisibility)
            if (
                lower_bound is analysis.lower_bound
                and upper_bound is analysis.upper_bound
                and divisibility is analysis.divisibility
            ):
                new_analysis = analysis
            else:
                new_analysis = type(analysis)(divisibility, lower_bound, upper_bound)
            analysis = new_analysis
        else:
            analysis = None
        if body is func.body and grid_blocks is func.metadata.grid_blocks and analysis is func.metadata.analysis:
            return func
        else:
            metadata = func.metadata.with_analysis(analysis).with_grid_blocks(grid_blocks)
            func = func.with_body(body).with_metadata(metadata)
            return func

    def visit_LetStmt(self, stmt: LetStmt) -> Stmt:
        bind_vars = []
        bind_values = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            bind_value = self.visit(bind_value)
            if isinstance(bind_value, Var) and bind_value in self.let_vars:
                self.memo[bind_var] = bind_value
                continue
            bind_vars.append(bind_var)
            bind_values.append(bind_value)
            self.let_vars.add(bind_var)
        body = self.visit(stmt.body)

        if same_list(bind_vars, stmt.bind_vars) and same_list(bind_values, stmt.bind_values) and body is stmt.body:
            return stmt
        else:
            if len(bind_vars) == len(bind_values) == 0:
                return body
            else:
                return LetStmt(tuple(bind_vars), tuple(bind_values), body)

    def visit_AssignStmt(self, stmt: AssignStmt) -> Stmt:
        if stmt.var in self.params:
            raise NotImplementedError("This pass assume that function parameters are immutable.")
        return super().visit_AssignStmt(stmt)


class LetPropogationPass(Pass):
    def __call__(self, prog: Function) -> Function:
        rewriter = LetPropogationRewriter()
        return rewriter(prog)


def let_propagation_pass() -> Pass:
    return LetPropogationPass()
