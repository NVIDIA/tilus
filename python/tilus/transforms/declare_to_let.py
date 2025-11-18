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
Convert DeclareStmt with initialized value to LetStmt when possible.

The declared variable should satisfy the following conditions to be converted:
    1. has never been modified with AssignStmt statement, and
    2. has never been addressed with Address expression, and
    3. has never been referenced with Reference expression, and.

This pass is adopted from the pass with the same name from Hidet.
"""

from collections import defaultdict
from typing import Dict

from hidet.ir.expr import Address, Expr, Reference, Var

from tilus.ir.func import Function
from tilus.ir.functors import IRRewriter
from tilus.ir.stmt import AssignStmt, DeclareStmt, LetStmt, SeqStmt, Stmt
from tilus.ir.tools import collect
from tilus.transforms.base import Pass


class DeclareToLetRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.assigns: Dict[Var, int] = defaultdict(int)

    def rewrite(self, func: Function) -> Function:
        for potential_usage in collect(func, (DeclareStmt, AssignStmt, Address, Reference)):
            if isinstance(potential_usage, Stmt):
                stmt = potential_usage
                if isinstance(stmt, DeclareStmt):
                    if stmt.init is not None:
                        self.assigns[stmt.var] += 1
                elif isinstance(stmt, AssignStmt):
                    self.assigns[stmt.var] += 1
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
        return self.visit(func)

    def visit_SeqStmt(self, seq_stmt: SeqStmt) -> Stmt:
        seq = [self.visit(stmt) for stmt in seq_stmt.seq]
        for i in range(len(seq) - 1, -1, -1):
            stmt = seq[i]
            if isinstance(stmt, DeclareStmt):
                if self.assigns[stmt.var] == 1 and stmt.init is not None:
                    let_stmt = LetStmt.create(
                        bind_vars=[stmt.var], bind_values=[stmt.init], body=SeqStmt.create(seq[i + 1 :])
                    )
                    seq = seq[:i] + [let_stmt]
        if len(seq) == len(seq_stmt.seq) and all(a is b for a, b in zip(seq, seq_stmt.seq)):
            return seq_stmt
        else:
            return SeqStmt.create(seq)


class DeclareToLetPass(Pass):
    def process_function(self, func: Function) -> Function:
        rewriter = DeclareToLetRewriter()
        return rewriter.rewrite(func)


def declare_to_let_pass() -> Pass:
    return DeclareToLetPass()
