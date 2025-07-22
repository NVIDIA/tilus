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
"""
Convert DeclareStmt with initialized value to LetStmt if the declared variable satisfy the following conditions:
    1. has never been modified with AssignStmt statement, and
    2. has never been addressed with Address expression, and
    3. has never been referenced with Reference expression, and
    4. has never appeared in outputs of AsmStmt statement

"""

from collections import defaultdict
from typing import Dict, List

from hidet.ir import SeqStmt
from hidet.ir.expr import Address, Expr, Reference, Var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import AsmStmt, AssignStmt, DeclareStmt, LetStmt, Stmt
from hidet.ir.tools import collect
from hidet.transforms.base import FunctionPass, Pass
from torch.fx.tensor_type import TensorType


class DeclareToLetRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.assigns: Dict[Var, int] = defaultdict(int)

    def rewrite(self, func: Function) -> Function:
        for potential_usage in collect(func, (DeclareStmt, AssignStmt, AsmStmt, Address, Reference)):
            if isinstance(potential_usage, Stmt):
                stmt = potential_usage
                if isinstance(stmt, DeclareStmt):
                    if stmt.init is not None:
                        self.assigns[stmt.var] += 1
                elif isinstance(stmt, AssignStmt):
                    self.assigns[stmt.var] += 1
                elif isinstance(stmt, AsmStmt):
                    for output_expr in stmt.output_exprs:
                        if isinstance(output_expr, Var):
                            self.assigns[output_expr] += 1
                else:
                    assert False
            elif isinstance(potential_usage, Expr):
                expr = potential_usage
                if isinstance(expr, Address):
                    if isinstance(expr.expr, Var):
                        self.assigns[expr.expr] += 1
                elif isinstance(expr, Reference):
                    if isinstance(expr.expr, Var):
                        self.assigns[expr.expr] += 1
                else:
                    assert False
            else:
                assert False
        for param in func.params:
            self.assigns[param] += 1
        return self.visit(func)

    def visit_SeqStmt(self, seq_stmt: SeqStmt) -> Stmt:
        seq = [self.visit(stmt) for stmt in seq_stmt.seq]
        for i in range(len(seq) - 1, -1, -1):
            stmt = seq[i]
            if isinstance(stmt, DeclareStmt) and self.assigns[stmt.var] == 1 and stmt.init is not None:
                # declare var = init (var is only assigned when declared and never modified later)
                let_stmt = LetStmt(bind_vars=[stmt.var], bind_values=[stmt.init], body=self.concat(seq[i + 1 :]))
                seq = seq[:i] + [let_stmt]
            elif isinstance(stmt, AssignStmt) and self.assigns[stmt.var] == 1:
                # declare var
                # ...
                # assign var = value (var is only assigned here and never modified later)
                #   (there is no addressing or referencing either)
                let_stmt = LetStmt(bind_vars=[stmt.var], bind_values=[stmt.value], body=self.concat(seq[i + 1 :]))
                seq = seq[:i] + [let_stmt]
            elif (
                isinstance(stmt, DeclareStmt)
                and self.assigns[stmt.var] == 1
                and stmt.init is None
                and not isinstance(stmt.var.type, TensorType)
            ):
                # declare var (var is never assigned)
                # we can safely remove this declare statement
                seq = seq[:i] + seq[i + 1 :]
        return self.concat(seq)

    def concat(self, seq: List[Stmt]) -> Stmt:
        if len(seq) == 1:
            return seq[0]
        else:
            return SeqStmt(seq)


class DeclareToLetPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        rewriter = DeclareToLetRewriter()
        return rewriter.rewrite(func)


def declare_to_let_pass() -> Pass:
    return DeclareToLetPass()
