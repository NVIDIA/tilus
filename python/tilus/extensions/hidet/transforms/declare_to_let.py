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
from enum import Enum
from typing import Dict, List

from hidet.ir import ForMappingStmt, ForStmt, SeqStmt, TensorType
from hidet.ir.expr import Address, Reference, Var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter, IRVisitor
from hidet.ir.stmt import AsmStmt, AssignStmt, DeclareStmt, LetStmt, Stmt
from hidet.ir.type import ArrayType
from hidet.transforms.base import FunctionPass, Pass


class DefinitionKind(Enum):
    DECLARE_WITHOUT_INIT = "declare_without_init"
    DECLARE_WITH_INIT = "declare_with_init"
    LET = "let"
    PARAM = "param"
    FOR = "for"


class UsageAnalyzer(IRVisitor):
    def __init__(self):
        super().__init__()
        self.defined_by: dict[Var, DefinitionKind] = {}
        self.explicit_assign_count: Dict[Var, int] = defaultdict(
            int
        )  # number of explicit assignment usage (i.e., AssignStmt)
        self.assign_count: Dict[Var, int] = defaultdict(int)  # number of assignment-like usage
        self.address_count: Dict[Var, int] = defaultdict(int)  # number of addressing-like usage

    def analyze(self, func: Function) -> None:
        self.assign_count.clear()
        self.address_count.clear()
        self.memo.clear()
        self.visit(func)

    def visit_LetStmt(self, stmt: LetStmt) -> None:
        super().visit_LetStmt(stmt)
        for var in stmt.bind_vars:
            self.defined_by[var] = DefinitionKind.LET

    def visit_DeclareStmt(self, stmt: DeclareStmt) -> None:
        super().visit_DeclareStmt(stmt)
        if stmt.init is not None:
            self.defined_by[stmt.var] = DefinitionKind.DECLARE_WITH_INIT
        else:
            self.defined_by[stmt.var] = DefinitionKind.DECLARE_WITHOUT_INIT

    def visit_ForStmt(self, stmt: ForStmt) -> None:
        super().visit_ForStmt(stmt)
        self.defined_by[stmt.loop_var] = DefinitionKind.FOR

    def visit_ForTaskStmt(self, stmt: ForMappingStmt) -> None:
        super().visit_ForTaskStmt(stmt)
        for loop_var in stmt.loop_vars:
            self.defined_by[loop_var] = DefinitionKind.FOR

    def visit_AssignStmt(self, stmt: AssignStmt) -> None:
        super().visit_AssignStmt(stmt)
        self.assign_count[stmt.var] += 1
        self.explicit_assign_count[stmt.var] += 1

    def visit_AsmStmt(self, stmt: AsmStmt) -> None:
        super().visit_AsmStmt(stmt)
        for output_expr in stmt.output_exprs:
            if isinstance(output_expr, Var):
                self.assign_count[output_expr] += 1

    def visit_Address(self, e: Address) -> None:
        super().visit_Address(e)
        if isinstance(e.expr, Var):
            self.address_count[e.expr] += 1

    def visit_Reference(self, e: Reference) -> None:
        super().visit_Reference(e)
        if isinstance(e.expr, Var):
            self.address_count[e.expr] += 1


class DeclareToLetRewriter(IRRewriter):
    """
    Transform some declare/assign statements to let statements.

    There are different cases:

    1. var is only assigned when declared and never modified later
        ```
          declare var = init
          ...  # var is not modified (i.e., assigned or addressed)
          =>
          let var = init
          ...
        ```
    2. var is declared without initialization and only assigned once later, and never modified later
        ```
           declare var
           ...
           assign var = value
           ...  # var is not modified (i.e., assigned or addressed)
           =>
           ...
           let var = value
           ...
        ```
        ONLY for non-tensor type
    3. var is declared without initialization and never modified later
        ```
            declare var
            ...
            =>
            ...
        ```
        ONLY for non-tensor type
    """

    def __init__(self):
        super().__init__()
        self.analyzer: UsageAnalyzer = UsageAnalyzer()
        self.declare_to_remove: set[Var] = set()

    def rewrite(self, func: Function) -> Function:
        self.analyzer.analyze(func)
        return self.visit(func)

    @staticmethod
    def concat(seq: List[Stmt]) -> Stmt:
        if len(seq) == 1:
            return seq[0]
        else:
            return SeqStmt(seq)

    def visit_SeqStmt(self, seq_stmt: SeqStmt) -> Stmt:
        seq = [self.visit(stmt) for stmt in seq_stmt.seq]
        for i in range(len(seq) - 1, -1, -1):
            stmt = seq[i]

            if (
                isinstance(stmt, DeclareStmt)
                and stmt.init is not None
                and self.analyzer.assign_count[stmt.var] == 0
                and self.analyzer.address_count[stmt.var] == 0
            ):
                # case 1
                let_stmt = LetStmt(bind_vars=[stmt.var], bind_values=[stmt.init], body=self.concat(seq[i + 1 :]))
                seq = seq[:i] + [let_stmt]
            elif (
                isinstance(stmt, DeclareStmt)
                and not isinstance(stmt.var.type, (TensorType, ArrayType))
                and stmt.init is None
                and self.analyzer.explicit_assign_count[stmt.var] == 1
                and self.analyzer.assign_count[stmt.var] == 1
                and self.analyzer.address_count[stmt.var] == 0
            ):
                # case 2 (remove declare)
                seq = seq[:i] + seq[i + 1 :]
            elif (
                isinstance(stmt, AssignStmt)
                and not isinstance(stmt.var.type, (TensorType, ArrayType))
                and self.analyzer.defined_by.get(stmt.var, None) == DefinitionKind.DECLARE_WITHOUT_INIT
                and self.analyzer.explicit_assign_count[stmt.var] == 1
                and self.analyzer.assign_count[stmt.var] == 1
                and self.analyzer.address_count[stmt.var] == 0
            ):
                # case 2 (convert assign to let)
                let_stmt = LetStmt(bind_vars=[stmt.var], bind_values=[stmt.value], body=self.concat(seq[i + 1 :]))
                seq = seq[:i] + [let_stmt]
            elif (
                isinstance(stmt, DeclareStmt)
                and not isinstance(stmt.var.type, (TensorType, ArrayType))
                and stmt.init is None
                and self.analyzer.assign_count[stmt.var] == 0
                and self.analyzer.address_count[stmt.var] == 0
            ):
                # case 3 (remove declare)
                seq = seq[:i] + seq[i + 1 :]

        return self.concat(seq)


class DeclareToLetPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        rewriter = DeclareToLetRewriter()
        return rewriter.rewrite(func)


def declare_to_let_pass() -> Pass:
    return DeclareToLetPass()
