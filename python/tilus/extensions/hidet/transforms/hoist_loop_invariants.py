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
"""
Hoist the loop invariant expressions out of the loop.

Algorithm:
1. Get the depth for each loop
2. Iterate the loops from the innermost to the outermost, for each loop:
  2.1 identify the loop invariant expressions.
  2.2 sort the loop invariants, and select HOIST_NUM of them.
      use NumOccurrences * NumOperandsPerExpr as the key for sorting.
  2.3 pick the first HOIST_XOR_NUM expressions
  2.4 visit the loop body to replace the selected expressions with the variables that hold the hoisted values.
"""

from __future__ import annotations

from collections import defaultdict

from hidet.ir import AssignStmt, BinaryExpr, BitwiseAnd, Constant, ForStmt, FuncType, LetStmt, StmtBuilder
from hidet.ir.expr import Add, BitwiseXor, Div, Expr, LeftShift, Mod, Multiply, RightShift, Sub, Var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter, IRVisitor
from hidet.ir.stmt import Stmt
from hidet.ir.tools import ExprHash, TypeInfer
from hidet.ir.utils.hash_sum import HashSum
from hidet.transforms.base import FunctionPass

from tilus.logging import get_logger

logger = get_logger(__name__)


"""
The number of loop invariant expressions to hoist at most for each loop.
"""
HOIST_NUM = 32


class LoopDepthAnalyzer(IRVisitor):
    def __init__(self):
        super().__init__()
        self.loop2depth: dict[Var, int] = {}
        self.current_depth: int = 0

    def visit_ForStmt(self, stmt: ForStmt) -> None:
        self.current_depth += 1
        self.loop2depth[stmt.loop_var] = self.current_depth
        super().visit_ForStmt(stmt)
        self.current_depth -= 1


class LoopInvariantsAnalyzer(IRVisitor):
    """
    Analyze the loop body to get the invariants and their number of occurrences.

    A list of loop invariant variables is provided to help identify the loop invariant expressions.
    """

    def __init__(self, loop_invariant_vars: list[Var], hasher: ExprHash):
        super().__init__()
        self.hasher: ExprHash = hasher
        self.loop_invariant_vars = loop_invariant_vars
        self.invariants: set[Expr] = set()
        self.invariant_ops: dict[Expr, int] = {}
        self.invariant_count: dict[HashSum, int] = defaultdict(int)

    def visit_Var(self, e: Var) -> None:
        if (e.name is not None and not isinstance(e.type, FuncType)) or e in self.loop_invariant_vars:
            self.invariants.add(e)
            self.invariant_ops[e] = 0
            self.invariant_count[self.hasher(e)] += 1
        else:
            return super().visit_Var(e)

    def visit_Constant(self, e: Constant) -> None:
        self.invariants.add(e)
        self.invariant_ops[e] = 0
        self.invariant_count[self.hasher(e)] += 1

    def visit_binary(self, e: BinaryExpr) -> None:
        self.visit(e.a)
        self.visit(e.b)
        if e.a in self.invariants and e.b in self.invariants:
            self.invariants.add(e)
            self.invariant_ops[e] = self.invariant_ops[e.a] + self.invariant_ops[e.b] + 1
            self.invariant_count[self.hasher(e)] += 1

    def visit_Add(self, e: Add) -> None:
        self.visit_binary(e)

    def visit_Sub(self, e: Sub) -> None:
        self.visit_binary(e)

    def visit_Multiply(self, e: Multiply) -> None:
        self.visit_binary(e)

    def visit_Div(self, e: Div) -> None:
        self.visit_binary(e)

    def visit_Mod(self, e: Mod) -> None:
        self.visit_binary(e)

    def visit_BitwiseXor(self, e: BitwiseXor) -> None:
        self.visit_binary(e)

    def visit_BitwiseAnd(self, e: BitwiseAnd) -> None:
        self.visit_binary(e)

    def visit_RightShift(self, e: RightShift) -> None:
        self.visit_binary(e)

    def visit_LeftShift(self, e: LeftShift) -> None:
        self.visit_binary(e)


class LoopBodyRewriter(IRRewriter):
    def __init__(self, remap: dict[Expr, Expr]):
        super().__init__()
        self.memo.update(remap)


class HoistLoopInvariantsRewriter(IRRewriter):
    def __init__(self, depth: int, loop2depth: dict[Var, int], holist_limit: int):
        super().__init__()
        self.type_infer: TypeInfer = TypeInfer()
        self.hasher: ExprHash = ExprHash()
        self.depth: int = depth
        self.loop2depth: dict[Var, int] = loop2depth
        self.invariant_vars: list[Var] = []
        self.hoist_limit: int = holist_limit

    def visit_Function(self, func: Function) -> Function:
        # add parameters as invariant variables
        for param in func.params:
            self.invariant_vars.append(param)

        body = self.visit(func.body)

        if body is func.body:
            return func
        else:
            return Function(func.name, func.params, body, func.ret_type, func.kind, func.attrs)

    def visit_LetStmt(self, stmt: LetStmt) -> Stmt:
        self.invariant_vars.extend(stmt.bind_vars)
        ret = super().visit_LetStmt(stmt)
        self.invariant_vars = self.invariant_vars[: -len(stmt.bind_vars)]
        return ret

    def visit_AssignStmt(self, stmt: AssignStmt) -> Stmt:
        if stmt.var in self.invariant_vars:
            assert False, "Invariant variable should never be assigned."
        return super().visit_AssignStmt(stmt)

    def visit_ForStmt(self, stmt: ForStmt) -> Stmt:
        if (self.loop2depth[stmt.loop_var] != self.depth) or isinstance(stmt.extent, Constant):
            # If the loop is small, we do not hoist the loop invariant expressions.
            self.invariant_vars.append(stmt.loop_var)
            ret = super().visit_ForStmt(stmt)
            self.invariant_vars.pop()
            return ret

        # 2.1 Identify the loop invariant expressions
        loop_invariants_analyzer = LoopInvariantsAnalyzer(loop_invariant_vars=self.invariant_vars, hasher=self.hasher)
        loop_invariants_analyzer.visit(stmt.body)

        # 2.2 Sort the loop invariant expressions
        hash2count: dict[HashSum, int] = loop_invariants_analyzer.invariant_count
        hash2ops: dict[HashSum, int] = {}
        for expr, ops in loop_invariants_analyzer.invariant_ops.items():
            hash2ops[self.hasher(expr)] = ops
        hashes = hash2count.keys()

        hashes = sorted(hashes, key=lambda h: hash2count[h] * hash2ops[h], reverse=True)

        # 2.3 pick the first HOIST_NUM expressions
        hashes = hashes[: self.hoist_limit]

        # 2.4 rewrite the loop body with the hoisted variables with the values of the loop invariants
        hash2var: dict[HashSum, Var] = {}
        hash2value: dict[HashSum, Expr] = {}
        remap: dict[Expr, Expr] = {}
        for invariant in loop_invariants_analyzer.invariants:
            invariant_hash = self.hasher(invariant)
            if invariant_hash not in hashes:
                continue
            if hash2ops[invariant_hash] == 0:
                continue
            if invariant_hash not in hash2var:
                var_name = "loop_invariant"
                hash2var[invariant_hash] = Var(var_name, type=self.type_infer(invariant))
                hash2value[invariant_hash] = invariant
            remap[invariant] = hash2var[invariant_hash]

        body_rewriter = LoopBodyRewriter(remap=remap)
        body = body_rewriter(stmt.body)

        if body is stmt.body:
            return stmt
        else:
            sb = StmtBuilder()
            bind_vars, bind_values = [], []
            for var, value in hash2var.items():
                bind_vars.append(value)
                bind_values.append(hash2value[var])
            with sb.lets(bind_vars, bind_values):
                with sb.for_loop(stmt.loop_var, stmt.extent, str(stmt.attr)):
                    sb += body
            return sb.finish()


class HoistLoopInvariantsPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        # 1. Analyze the loop depth
        depth_analyzer = LoopDepthAnalyzer()
        depth_analyzer.visit(func)

        # 2. Iterate the loops from the innermost to the outermost
        depths = sorted(set(depth_analyzer.loop2depth.values()), reverse=True)
        for depth in depths:
            hoist_repeat = 4
            hoist_limit = 8
            for _ in range(hoist_repeat):
                rewriter = HoistLoopInvariantsRewriter(depth, depth_analyzer.loop2depth, hoist_limit)
                func = rewriter(func)

        return func


def hoist_loop_invariants_pass() -> FunctionPass:
    """Hoist the loop invariant XOR expressions out of the loop."""
    return HoistLoopInvariantsPass()
