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
from typing import List, Optional

from hidet.ir.expr import Add, Call, Constant, Expr, Multiply, Neg, Sub, Var
from hidet.ir.functors import ExprRewriter
from hidet.ir.tools import simplify as original_simplify
from hidet.ir.type import DataType
from hidet.utils import same_list


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
    expr = original_simplify(expr)  # type: ignore
    expr = _extra_simplify_expr(expr)
    return expr
