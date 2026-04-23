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
import operator
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

from tilus.hidet.ir.dtypes import int32
from tilus.hidet.ir.expr import (
    Add,
    BinaryExpr,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    Call,
    Cast,
    Constant,
    Div,
    Equal,
    Expr,
    IfThenElse,
    LeftShift,
    LessEqual,
    LessThan,
    LogicalAnd,
    LogicalNot,
    LogicalOr,
    Mod,
    Multiply,
    Neg,
    NotEqual,
    RightShift,
    Sub,
    Var,
    cast,
    constant,
    convert,
    is_false,
    is_one,
    is_true,
    is_zero,
)
from tilus.hidet.ir.functors import BaseRewriter, ExprRewriter, StmtRewriter
from tilus.hidet.ir.node import Node
from tilus.hidet.ir.stmt import ForStmt, IfStmt, SeqStmt, Stmt
from tilus.hidet.ir.tools import rewrite
from tilus.hidet.ir.type import DataType
from tilus.hidet.utils import same_list


class Simplifier(StmtRewriter, ExprRewriter, BaseRewriter):
    def __init__(self, skip_node_types: Optional[Sequence[Type[Expr]]] = None):
        super().__init__()
        self.skip_node_types = skip_node_types

    def visit(self, node: Union[Node, Tuple, List, Dict[str, Any], str, int, float]):
        if self.skip_node_types and isinstance(node, tuple(self.skip_node_types)):
            return node
        return super().visit(node)

    def visit_Binary(self, e: BinaryExpr):  # pylint: disable=too-many-branches
        a = self(e.a)
        b = self(e.b)
        if isinstance(e, Add):
            if is_zero(a):
                return b
            if is_zero(b):
                return a
        elif isinstance(e, Sub):
            if is_zero(b):
                return a
        elif isinstance(e, Multiply):
            if is_one(a):
                return b
            if is_one(b):
                return a
            if is_zero(a) or is_zero(b):
                return convert(0)
        elif isinstance(e, Div):
            if is_one(b):
                return a
        elif isinstance(e, Mod):
            if is_one(e.b):
                return convert(0)
        elif isinstance(e, (LessThan, LessEqual, Equal, NotEqual)):
            pass
        elif isinstance(e, LogicalAnd):
            if is_false(a) or is_false(b):
                return convert(False)
            if is_true(a):
                return b
            if is_true(b):
                return a
        elif isinstance(e, LogicalOr):
            if is_true(a) or is_true(b):
                return convert(True)
            if is_false(a):
                return b
            if is_false(b):
                return a
        elif isinstance(e, BitwiseAnd):
            pass
        elif isinstance(e, BitwiseOr):
            pass
        elif isinstance(e, BitwiseXor):
            pass
        elif isinstance(e, LeftShift):
            pass
        elif isinstance(e, RightShift):
            pass
        else:
            raise ValueError()

        if isinstance(a, Constant) and isinstance(b, Constant):
            # a op b
            op_dict = {
                Add: operator.add,
                Sub: operator.sub,
                Multiply: operator.mul,
                Div: operator.truediv,
                Mod: operator.mod,
                LessThan: operator.lt,
                LessEqual: operator.le,
                Equal: operator.eq,
                NotEqual: operator.ne,
                BitwiseAnd: operator.and_,
                BitwiseOr: operator.or_,
                BitwiseXor: operator.xor,
                LeftShift: operator.lshift,
                RightShift: operator.rshift,
            }
            if e.__class__ in op_dict:
                if a.type.name == "int32" and b.type.name == "int32" and isinstance(e, Div):
                    # the Div for int32 will use floordiv. Override the native behavior of python
                    return convert(a.value // b.value, "int32")
                else:
                    return convert(op_dict[e.__class__](a.value, b.value))
            elif isinstance(e, LogicalAnd):
                return convert(a.value and b.value)
            elif isinstance(e, LogicalOr):
                return convert(a.value or b.value)
            else:
                raise ValueError()
        elif isinstance(a, Constant) and not isinstance(b, Constant):
            # aa op bb (aa is constant)
            op = e.__class__
            mapping: Dict[Type, Tuple[Callable, Callable]] = {
                Add: (lambda: True, lambda: b + a),
                Multiply: (lambda: True, lambda: b * a),
            }
            if op in mapping:
                condition, result = mapping[op]
                if condition():
                    return result()
        elif isinstance(a, BinaryExpr) and isinstance(a.b, Constant) and isinstance(b, Constant):
            # (aa op1 bb) op2 cc
            op1 = a.__class__
            op2 = e.__class__
            aa, bb, cc = a.a, a.b, b
            btype, ctype = bb.type, cc.type
            mapping: Dict[Tuple, Tuple[Callable, Callable]] = {  # (op1, op2) -> (condition, result)
                (Add, Add): (lambda: True, lambda: aa + (bb + cc) if bb + cc > 0 else aa - (-(bb + cc))),
                (Add, Sub): (lambda: True, lambda: aa + (bb - cc) if bb - cc > 0 else aa - (-(bb - cc))),
                (Sub, Add): (lambda: True, lambda: aa - (bb - cc) if bb - cc > 0 else aa + (-(bb - cc))),
                (Sub, Sub): (lambda: True, lambda: aa - (bb + cc) if bb + cc > 0 else aa + (-(bb + cc))),
                (Div, Div): (lambda: btype.is_integer() and ctype.is_integer(), lambda: aa // (bb * cc)),
            }
            if (op1, op2) in mapping:
                condition, result = mapping[(op1, op2)]
                if condition():
                    return result()

        if a is e.a and b is e.b:
            return e
        return e.__class__(a, b)

    def visit_Not(self, e: LogicalNot):
        a = self(e.a)
        if isinstance(a, Constant):
            return convert(not a.value)
        if a is e.a:
            return e
        else:
            return LogicalNot(a)

    def visit_Neg(self, e: Neg):
        a = self(e.a)
        if isinstance(a, Constant):
            return convert(-a.value)
        if a is e.a:
            return e
        else:
            return Neg(a)

    def visit_Cast(self, e: Cast):
        expr = self.visit(e.expr)
        if isinstance(expr, Constant) and expr.type.is_data_type():
            assert isinstance(e.target_type, DataType)
            return constant(expr.value, e.target_type)
        elif expr is e.expr:
            return e
        else:
            return cast(expr, e.target_type)

    def visit_IfStmt(self, stmt: IfStmt):
        cond = self.visit(stmt.cond)
        then_body = self.visit(stmt.then_body)
        else_body = self.visit(stmt.else_body) if stmt.else_body else None
        if is_true(cond):
            return then_body
        elif is_false(cond):
            if else_body:
                return else_body
            else:
                return SeqStmt.create([])
        else:
            if cond is stmt.cond and then_body is stmt.then_body and else_body is stmt.else_body:
                return stmt
            else:
                return IfStmt.create(cond, then_body, else_body)

    def visit_ForStmt(self, stmt: ForStmt):
        loop_var = self(stmt.loop_var)
        extent = self(stmt.extent)
        body = self(stmt.body)
        if is_one(extent):
            return rewrite(stmt.body, {loop_var: convert(0)})
        else:
            if loop_var is stmt.loop_var and body is stmt.body:
                return stmt
            else:
                return ForStmt.create(loop_var, extent, body=body, attr=stmt.attr)

    def visit_IfThenElse(self, e: IfThenElse):
        cond = self.visit(e.cond)
        then_expr = self.visit(e.then_expr)
        else_expr = self.visit(e.else_expr)
        if is_true(cond):
            return then_expr
        elif is_false(cond):
            return else_expr
        else:
            if cond is e.cond and then_expr is e.then_expr and else_expr is e.else_expr:
                return e
            else:
                return IfThenElse(cond, then_expr, else_expr)


def simplify(
    node: Union[Stmt, Expr, int, float, list, tuple],
    *,
    repeat_limit=10,
    enable_rules=False,
    skip_node_types: Optional[Sequence[Type[Expr]]] = None,
):
    if isinstance(node, (int, float)):
        return node

    simplifier = Simplifier(skip_node_types)
    for _ in range(repeat_limit):
        old_node = node
        node = simplifier(node)
        if old_node is node:
            break

    # Apply rule-based simplification if enabled
    if enable_rules:
        from tilus.hidet.transforms.rule_based_simplifier import rule_based_simplify

        node = rule_based_simplify(node, skip_node_types=skip_node_types)

    return node


def simplify_to_int(node: Union[Expr, int], *, repeat_limit=10) -> int:
    if isinstance(node, int):
        return node
    node = simplify(node, repeat_limit=repeat_limit)
    if not (isinstance(node, Constant) and node.type.is_integer()):
        raise ValueError(f"Can not simplify expression {node} to an integer")
    return node.value


# --- Extra simplification utilities (merged from tilus extensions) ---


class Sum(Expr):
    def __init__(self, terms: list[Expr]):
        assert len(terms) >= 1
        self.terms: list[Expr] = terms


class ExprRewriterWithSum(ExprRewriter):
    def visit_dispatch(self, node):
        if isinstance(node, Sum):
            return self.visit_Sum(node)
        return super().visit_dispatch(node)

    def visit_Sum(self, e: Sum) -> Expr:
        terms = [self.visit(term) for term in e.terms]
        # flatten nested Sums
        flat_terms: List[Expr] = []
        for term in terms:
            if isinstance(term, Sum):
                flat_terms.extend(term.terms)
            else:
                flat_terms.append(term)
        return Sum(flat_terms)


class ConvertAddSubToSumRewriter(ExprRewriterWithSum):
    def visit_Add(self, e: Add) -> Expr:
        lhs = self.visit(e.a)
        rhs = self.visit(e.b)
        terms = []
        if isinstance(lhs, Sum):
            terms.extend(lhs.terms)
        else:
            terms.append(lhs)
        if isinstance(rhs, Sum):
            terms.extend(rhs.terms)
        else:
            terms.append(rhs)
        return Sum(terms)

    def visit_Sub(self, e: Sub) -> Expr:
        lhs = self.visit(e.a)
        rhs = self.visit(e.b)
        terms = []
        if isinstance(lhs, Sum):
            terms.extend(lhs.terms)
        else:
            terms.append(lhs)
        if isinstance(rhs, Sum):
            terms.extend([-term for term in rhs.terms])
        else:
            terms.append(-rhs)
        return Sum(terms)


class ConvertSumToAddSubRewriter(ExprRewriterWithSum):
    def visit_Sum(self, e: Sum) -> Expr:
        terms = [self.visit(term) for term in e.terms]
        assert len(terms) >= 1
        result = terms[0]
        for term in terms[1:]:
            if isinstance(term, Neg):
                result = Sub(result, term.a)
            else:
                result = Add(result, term)
        return result


class MergeTermSimplifier(ExprRewriterWithSum):
    def __init__(self):
        super().__init__()

    def decompose_linear_term(self, term: Expr) -> Optional[tuple[Constant, Var]]:
        if isinstance(term, Var):
            if isinstance(term.type, DataType):
                return Constant(value=1, const_type=term.type), term
        elif isinstance(term, Multiply):
            if isinstance(term.a, Constant) and isinstance(term.b, Var):
                return term.a, term.b
            elif isinstance(term.b, Constant) and isinstance(term.a, Var):
                return term.b, term.a
        elif isinstance(term, Neg) and isinstance(term.a, Multiply):
            optional_term = self.decompose_linear_term(term.a)
            if optional_term is not None:
                c, x = optional_term
                return -c, x
        return None

    def merge_linear_terms(self, terms: list[Expr]) -> bool:
        n = len(terms)
        for i in range(n):
            for j in range(i + 1, n):
                # check if terms[i] and terms[j] are like c1 * x and c2 * x, if so, merge them
                decomp_i = self.decompose_linear_term(terms[i])
                decomp_j = self.decompose_linear_term(terms[j])
                if decomp_i is None or decomp_j is None:
                    continue
                c1, x1 = decomp_i
                c2, x2 = decomp_j
                if x1 is x2:
                    # merge c1 * x1 and c2 * x2
                    assert isinstance(x1.type, DataType)
                    c = Constant(c1.value + c2.value, const_type=x1.type)  # type: ignore
                    if c.value == 0:
                        terms.pop(j)
                        terms.pop(i)
                    else:
                        terms[i] = Multiply(c, x1)
                        terms.pop(j)
                    return True
        return False

    def visit_Sum(self, e: Sum) -> Expr:
        terms = [self.visit(term) for term in e.terms]
        while self.merge_linear_terms(terms):
            pass
        if same_list(terms, e.terms):
            return e
        else:
            return Sum(terms)


class SwizzleCallSimplifier(ExprRewriter):
    def visit_Call(self, e: Call) -> Expr:
        if e.func_var.name == "swizzle":
            x, mbase, bbits, sshift = e.args
            if isinstance(x, Constant) and x.value == 0:
                return x
            else:
                return super().visit_Call(e)
        else:
            return super().visit_Call(e)


def _extra_simplify_expr(expr: Expr) -> Expr:
    converter_to_sum = ConvertAddSubToSumRewriter()
    converter_from_sum = ConvertSumToAddSubRewriter()
    merge_term_simplifier = MergeTermSimplifier()
    swizzle_simplifier = SwizzleCallSimplifier()

    expr = converter_to_sum(expr)
    expr = merge_term_simplifier(expr)
    expr = converter_from_sum(expr)
    expr = swizzle_simplifier(expr)

    return expr


def simplify_expr(expr: Expr) -> Expr:
    expr = simplify(expr)  # type: ignore
    expr = _extra_simplify_expr(expr)
    return expr
