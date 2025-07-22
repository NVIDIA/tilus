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
Affine to recurrence lowering pass.

This pass transforms affine expressions towards a iterative variable into recurrence form.

For example, we can transform the following code:
```
for i in range(n):
    access(gmem + blockIdx.x * 32 + threadIdx.x * 16 + i * 8 + 4)
```
into
```
addr = gmem + blockIdx.x * 32 + threadIdx.x * 16 + 4
for i in range(n):
    access(addr)
    addr += 8
```

Algorithm:
1. get the loop-depth of all for-loops
2. iterate over the loop-depth from the innermost to the outermost, and perform the following steps for each loop with
  the current loop-depth:
  2.1. before entering the loop, mark all variables that are let-bound outside the loop as loop-invariant variables
  2.2. analyze the expressions used in the loop body. Given an expression inside the loop body, we decompose it into the
    sum of the following parts:
      - loop-invariant part: the part that will not change during the loop iterations.
        e.g., `gmem + blockIdx.x * 32` in the example above.
      - coefficient part: the part that is multiplied by the loop index, which must be a loop-invariant expression.
        e.g., `8` in the example above.
      - constant part: the part that is a constant value.
        e.g., `4` in the example above.
      It is possible that an expression can not be decomposed into above form, in this case, we do not record it.
  2.3. after the analysis in 2.2. revisit the loop body, and for each expression that can be decomposed into the
    above form, we transform it into a recurrence form:
    ```
    declare expr = loop-invariant part
    for i in range(n):  # this is the loop that we are currently processing
        usage(expr + constant part)
        assign expr = expr + coefficient part
    ```
    We might need to have multiple expressions to be rewritten for each loop, so we need to reuse the same expr
    if they have the same loop-invariant part and coefficient part.
  2.4. after the loop body is processed, we need to add a prologue for the loop and an epilogue for the loop body.
"""

from __future__ import annotations

import operator
from typing import Optional

from hidet import int32
from hidet.ir import Add, BinaryExpr, Cast, Constant, Div, LetStmt, Mod, Multiply, Sub, Var
from hidet.ir.builders import StmtBuilder
from hidet.ir.expr import BitwiseXor, Expr
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter, IRVisitor
from hidet.ir.stmt import ForStmt, Stmt
from hidet.ir.tools import TypeInfer
from hidet.ir.tools.hasher import ExprHash, HashSum
from hidet.transforms.base import FunctionPass

from tilus.ir.tools.printer import use_standalone_printer
from tilus.logging import get_logger

logger = get_logger(__name__)


class OptionalExpr:
    def __init__(self, expr: Optional[Expr]):
        # None indicates 0
        self.expr: Optional[Expr] = expr

    def empty(self):
        return self.expr is None

    def __add__(self, other):
        if self.expr is None:
            return other
        elif other.expr is None:
            return self
        else:
            return OptionalExpr(self.expr + other.expr)

    def __sub__(self, other):
        if self.expr is None and other.expr is None:
            return OptionalExpr(None)
        elif self.expr is None:
            return OptionalExpr(-other.expr)
        elif other.expr is None:
            return self
        else:
            return OptionalExpr(self.expr - other.expr)

    def __mul__(self, other):
        if self.expr is None or other.expr is None:
            return OptionalExpr(None)
        else:
            return OptionalExpr(self.expr * other.expr)

    def __floordiv__(self, other):
        if self.expr is None or other.expr is None:
            return OptionalExpr(None)
        else:
            return OptionalExpr(self.expr // other.expr)

    def __mod__(self, other):
        if self.expr is None or other.expr is None:
            return OptionalExpr(None)
        else:
            return OptionalExpr(self.expr % other.expr)


class AffineExpr:
    def __init__(
        self,
        invariant_expr: OptionalExpr = OptionalExpr(None),
        coefficient: OptionalExpr = OptionalExpr(None),
        constant: OptionalExpr = OptionalExpr(None),
    ):
        self.invariant_expr: OptionalExpr = invariant_expr
        self.coefficient: OptionalExpr = coefficient
        self.constant: OptionalExpr = constant

    def is_loop_invariant(self) -> bool:
        return self.coefficient.empty()

    def as_tuple(self) -> tuple[OptionalExpr, OptionalExpr, OptionalExpr]:
        return self.invariant_expr, self.coefficient, self.constant

    def __str__(self):
        items = []
        if self.invariant_expr.expr is not None:
            items.append(str(self.invariant_expr.expr))
        if self.coefficient.expr is not None:
            items.append(f"{self.coefficient.expr} * i")
        if self.constant.expr is not None:
            items.append(str(self.constant.expr))
        if len(items) == 0:
            return "0"
        return " + ".join(items)

    def __add__(self, other: AffineExpr) -> AffineExpr:
        # (e1 + e2 * i + c1) + (e3 + e4 * i + c2) = (e1 + e3) + (e2 + e4) * i + (c1 + c2)
        e1, e2, c1 = self.as_tuple()
        e3, e4, c2 = other.as_tuple()
        return AffineExpr(invariant_expr=e1 + e3, coefficient=e2 + e4, constant=c1 + c2)

    def __sub__(self, other: AffineExpr) -> AffineExpr:
        # (e1 + e2 * i + c1) - (e3 + e4 * i + c2) = (e1 - e3) + (e2 - e4) * i + (c1 - c2)
        e1, e2, c1 = self.as_tuple()
        e3, e4, c2 = other.as_tuple()
        return AffineExpr(invariant_expr=e1 - e3, coefficient=e2 - e4, constant=c1 - c2)

    def __mul__(self, other: AffineExpr) -> AffineExpr:
        # (e1 + e2 * i + c1) * (e3 + e4 * i + c2)
        # = (e1 * e3 + e1 * c2 + e3 * c1) + (e2 * e2 + e2 * c2 + e1 * e4 + e4 * c1) * i + (c1 * c2)
        e1, e2, c1 = self.as_tuple()
        e3, e4, c2 = other.as_tuple()
        if not e2.empty() and not e4.empty():
            return None
        return AffineExpr(
            invariant_expr=e1 * e3 + e1 * c2 + e3 * c1,
            coefficient=e2 * e4 + e2 * c2 + e1 * e4 + e4 * c1,
            constant=c1 * c2,
        )

    def __floordiv__(self, other: AffineExpr) -> AffineExpr:
        # (e1 + e2 * i + c1) // (e3 + e4 * i + c2)  when e2 = e4 = 0
        # = (e1 + c1) // (e3 + c2)
        e1, e2, c1 = self.as_tuple()
        e3, e4, c2 = other.as_tuple()
        if not e2.empty() or not e4.empty():
            # if either the coefficient of self or other is not empty, we cannot perform floor division
            return None
        return AffineExpr(
            invariant_expr=(e1 + c1) // (e3 + c2), coefficient=OptionalExpr(None), constant=OptionalExpr(None)
        )

    def __mod__(self, other: AffineExpr) -> AffineExpr:
        # (e1 + e2 * i + c1) % (e3 + e4 * i + c2)  when e2 = e4 = 0
        # = (e1 + c1) % (e3 + c2)
        e1, e2, c1 = self.as_tuple()
        e3, e4, c2 = other.as_tuple()
        if not e2.empty() or not e4.empty():
            # if either the coefficient of self or other is not empty, we cannot perform modulo
            return None
        return AffineExpr(
            invariant_expr=(e1 + c1) % (e3 + c2), coefficient=OptionalExpr(None), constant=OptionalExpr(None)
        )


class LoopDepthAnalyzer(IRVisitor):
    """
    Analyze the loop depth of all for-loops in the function body.
    """

    def __init__(self):
        super().__init__()
        self.current_depth: int = 0
        self.loop_depth: dict[Var, int] = {}

    def visit_ForStmt(self, stmt: ForStmt) -> None:
        self.loop_depth[stmt.loop_var] = self.current_depth
        self.current_depth += 1
        self.visit(stmt.body)
        self.current_depth -= 1


class AffineExprAnalyzer(IRVisitor):
    """
    Analyze affine expression decomposition given the loop variable and loop-invariant variables.
    """

    def __init__(self, loop_var: Var, loop_invariant_vars: list[Var]):
        super().__init__()
        self.loop_var = loop_var
        self.loop_invariant_vars = loop_invariant_vars
        self.expr2affine: dict[Expr, AffineExpr] = {}

    def visit_Var(self, e: Var) -> None:
        if e in self.expr2affine:
            return
        if e in self.loop_invariant_vars or e.name is not None:
            self.expr2affine[e] = AffineExpr(invariant_expr=OptionalExpr(e))
        elif e is self.loop_var:
            self.expr2affine[e] = AffineExpr(coefficient=OptionalExpr(int32.one))

    def visit_Constant(self, e: Constant) -> None:
        if e in self.expr2affine:
            return
        self.expr2affine[e] = AffineExpr(constant=OptionalExpr(e))

    def visit_binary(self, e: BinaryExpr) -> None:
        self.visit(e.a)
        self.visit(e.b)
        op_dict = {
            Add: operator.add,
            Sub: operator.sub,
            Multiply: operator.mul,
            Div: operator.floordiv,
            Mod: operator.mod,
        }
        lhs = self.expr2affine.get(e.a, None)
        rhs = self.expr2affine.get(e.b, None)
        if lhs is None or rhs is None:
            ret = None
        else:
            assert isinstance(lhs, AffineExpr)
            assert isinstance(rhs, AffineExpr)
            if type(e) in op_dict:
                op = op_dict[type(e)]
                ret = op(lhs, rhs)
            elif lhs.coefficient.expr is None and rhs.coefficient.expr is None:
                ret = AffineExpr(invariant_expr=OptionalExpr(e))
            else:
                ret = None
        if ret is not None:
            self.expr2affine[e] = ret
        logger.debug("%s: %s", e, ret if ret is not None else "None")

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

    def visit_Cast(self, e: Cast) -> None:
        base = self.visit(e.expr)
        if base in self.expr2affine and self.expr2affine[base].is_loop_invariant():
            # if the base expression is loop-invariant, we can cast it to the target type
            self.expr2affine[e] = AffineExpr(invariant_expr=OptionalExpr(e))

    def visit_LetStmt(self, stmt: LetStmt) -> None:
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            self.visit(bind_value)
            if bind_value in self.expr2affine:
                self.expr2affine[bind_var] = self.expr2affine[bind_value]
        self.visit(stmt.body)


class AffineToRecurrenceRewriter(IRRewriter):
    """
    Rewrites affine expressions towards a iterative variable into recurrence form.
    """

    def __init__(self, depth: int, loop_depth: dict[Var, int]):
        super().__init__()
        self.depth: int = depth
        self.loop_depth: dict[Var, int] = loop_depth

        self.type_infer = TypeInfer()

        # the currently bound variables that will not change
        self.var2value: dict[Var, Expr] = {}

        # loop-specific context
        self.outside_loop: bool = True
        self.hasher: ExprHash = ExprHash()
        self.affine_analyzer: Optional[AffineExprAnalyzer] = None
        self.affine_exprs: dict[HashSum, tuple[Var, AffineExpr]] = {}

    def visit(self, node):
        if isinstance(node, Expr):
            expr: Expr = node
            if self.affine_analyzer is not None and expr in self.affine_analyzer.expr2affine:
                affine_expr = self.affine_analyzer.expr2affine[expr]
                if not affine_expr.coefficient.empty() and not affine_expr.invariant_expr.empty():
                    # invariant_expr + coefficient * loop_var
                    hash_expr = (
                        affine_expr.invariant_expr
                        + affine_expr.coefficient * OptionalExpr(self.affine_analyzer.loop_var)
                    ).expr
                    hashsum = self.hasher(hash_expr)
                    if hashsum not in self.affine_exprs:
                        expr_var = Var(hint="affine_expr", type=self.type_infer(expr))
                        self.affine_exprs[hashsum] = (expr_var, affine_expr)
                    expr_var = self.affine_exprs[hashsum][0]
                    ret = expr_var
                    if affine_expr.constant.expr is not None:
                        ret = ret + affine_expr.constant.expr
                    return ret
        return super().visit(node)

    @use_standalone_printer
    def visit_ForStmt(self, stmt: ForStmt) -> Stmt:
        if self.loop_depth[stmt.loop_var] != self.depth:
            # we are not processing the loop with the current depth
            return super().visit_ForStmt(stmt)

        logger.debug("Processing loop with depth %d", self.depth)
        logger.debug("%s", stmt)

        # 2.1. before entering the loop, mark all variables that are let-bound outside the loop as loop-invariant variables
        self.hasher = ExprHash()
        self.affine_analyzer = AffineExprAnalyzer(
            loop_var=stmt.loop_var, loop_invariant_vars=list(self.var2value.keys())
        )
        self.affine_exprs.clear()
        self.affine_analyzer.visit(stmt.body)

        for expr, affine_expr in self.affine_analyzer.expr2affine.items():
            if affine_expr.invariant_expr.expr is not None and affine_expr.coefficient.expr is not None:
                logger.debug("%s", expr)
                logger.debug("  -> %s", affine_expr)

        # 2.2. visit the loop body
        self.outside_loop = False
        body = self.visit(stmt.body)
        self.outside_loop = True

        self.affine_analyzer = None

        # 2.3. after the loop body is processed, we need to add a prologue for the loop and an epilogue for the loop body
        if body is stmt.body:
            return stmt
        else:
            sb = StmtBuilder()
            for expr_var, expr_affine in self.affine_exprs.values():
                # declare the loop-invariant variable
                optional_expr = expr_affine.invariant_expr
                invariant_part = optional_expr.expr if not optional_expr.empty() else int32.zero
                sb.declare(expr_var, invariant_part)
            with sb.for_loop(v=stmt.loop_var, extent=stmt.extent, attr=str(stmt.attr)):
                sb.append(body)
                for expr_var, expr_affine in self.affine_exprs.values():
                    # emit the recurrence update
                    assert not expr_affine.coefficient.empty()
                    sb.assign(expr_var, expr_var + expr_affine.coefficient.expr)
            return sb.finish()

    def visit_LetStmt(self, stmt: LetStmt) -> Stmt:
        if self.outside_loop:
            for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
                self.var2value[bind_var] = bind_value
            body = self.visit(stmt.body)
            for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
                if bind_var in self.var2value:
                    del self.var2value[bind_var]
            if body is stmt.body:
                return stmt
            else:
                return LetStmt(bind_vars=stmt.bind_vars, bind_values=stmt.bind_values, body=body)
        else:
            return super().visit_LetStmt(stmt)


class LowerAffineToRecurrencePass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        # 1. get the loop-depth of all for-loops
        depth_analyzer = LoopDepthAnalyzer()
        depth_analyzer.visit(func.body)

        # 2. iterate over the loop-depth from the innermost to the outermost
        all_depths = sorted(set(depth_analyzer.loop_depth.values()), reverse=True)
        body = func.body
        for depth in all_depths:
            rewriter = AffineToRecurrenceRewriter(depth, depth_analyzer.loop_depth)
            body = rewriter(body)

        if body is func.body:
            return func
        else:
            return Function(
                name=func.name,
                params=func.params,
                body=body,
                ret_type=func.ret_type,
                kind=func.kind,
                attrs=func.attrs,
            )


def lower_affine_to_recurrence_pass() -> FunctionPass:
    return LowerAffineToRecurrencePass()
