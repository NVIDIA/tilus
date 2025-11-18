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
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from hidet.ir.expr import Expr, Var

from tilus.ir.inst import Instruction
from tilus.ir.node import IRNode
from tilus.ir.tensor import Tensor


@dataclass(frozen=True, eq=False)
class Stmt(IRNode):
    pass


@dataclass(frozen=True, eq=False)
class SeqStmt(Stmt):
    seq: tuple[Stmt, ...]

    @staticmethod
    def create(seq: Sequence[Stmt]) -> SeqStmt:
        return SeqStmt(tuple(seq))


@dataclass(frozen=True, eq=False)
class ForStmt(Stmt):
    iter_var: Var
    extent: Expr
    body: Stmt

    # candidates:
    # - None (no annotation),
    # - -1 (unroll all),
    # - n (n >= 1, unroll with factor n)
    unroll_factor: Optional[int]


@dataclass(frozen=True, eq=False)
class ThreadGroupStmt(Stmt):
    thread_begin: int
    num_threads: int
    body: Stmt

    @staticmethod
    def create(thread_begin: int, num_threads: int, body: Stmt) -> ThreadGroupStmt:
        return ThreadGroupStmt(thread_begin, num_threads, body)


@dataclass(frozen=True, eq=False)
class IfStmt(Stmt):
    cond: Expr
    then_body: Stmt
    else_body: Optional[Stmt]

    def with_else_body(self, else_body: Stmt) -> IfStmt:
        return IfStmt(self.cond, self.then_body, else_body)


@dataclass(frozen=True, eq=False)
class WhileStmt(Stmt):
    cond: Expr
    body: Stmt


@dataclass(frozen=True, eq=False)
class BreakStmt(Stmt):
    pass


@dataclass(frozen=True, eq=False)
class ReturnStmt(Stmt):
    pass


@dataclass(frozen=True, eq=False)
class DeclareStmt(Stmt):
    var: Var
    init: Optional[Expr]


@dataclass(frozen=True, eq=False)
class AssignStmt(Stmt):
    var: Var
    value: Expr


@dataclass(frozen=True, eq=False)
class LetStmt(Stmt):
    bind_vars: tuple[Var, ...]
    bind_values: tuple[Expr, ...]
    body: Stmt

    @staticmethod
    def create(bind_vars: Sequence[Var], bind_values: Sequence[Expr], body: Stmt) -> LetStmt:
        return LetStmt(tuple(bind_vars), tuple(bind_values), body)


@dataclass(frozen=True, eq=False)
class EvaluateStmt(Stmt):
    expr: Expr
    pred: Optional[Expr]


@dataclass(frozen=True, eq=False)
class TensorItemPtrStmt(Stmt):
    ptr_var: Var
    tensor: Tensor
    space: str  # 'generic', 'shared', 'global', 'local'


@dataclass(frozen=True, eq=False)
class TensorItemValueStmt(Stmt):
    var: Var
    tensor: Tensor


@dataclass(frozen=True, eq=False)
class InstStmt(Stmt):
    inst: Instruction


def seq_stmt(seq: Sequence[Stmt | Instruction]) -> Stmt:
    stmt_seq: List[Stmt] = [InstStmt(item) if isinstance(item, Instruction) else item for item in seq]
    if len(stmt_seq) == 1:
        return stmt_seq[0]
    else:
        return SeqStmt(tuple(stmt_seq))
