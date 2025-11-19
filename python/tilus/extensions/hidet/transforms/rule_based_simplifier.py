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
import itertools
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from hidet.ir import BitwiseXor, Constant, DataType, logical_and
from hidet.ir.dtypes import int32
from hidet.ir.expr import Add, Expr, Var
from hidet.ir.func import Function
from hidet.ir.tools import TypeInfer
from hidet.ir.type import sizeof
from hidet.transforms.base import FunctionPass
from hidet.transforms.rule_based_simplifier import BoundInfo, any_constant, any_expr, int_constant
from hidet.transforms.rule_based_simplifier import RuleBasedSimplifier as OriginalRuleBasedSimplifier
from hidet.utils import gcd, repeat_until_converge


class RuleBasedSimplifier(OriginalRuleBasedSimplifier):
    def __init__(self, var2bound: Optional[Dict[Var, BoundInfo]]):
        super().__init__()
        self.analyzer.bound.update(var2bound or {})  # type: ignore
        self.type_infer = TypeInfer()

        e1, e2 = any_expr(allow_const=False), any_expr(allow_const=False)
        c1, c2 = any_constant(), any_constant()
        ic1, ic2 = int_constant(), int_constant()
        ec1, ec2 = any_expr(allow_const=True), any_expr(allow_const=True)

        extra_patterns = [
            # bitwise xor
            (e1 ^ c1, e1, c1 == 0),
            ((e1 + c1) - c2, e1 + (c1 - c2), c1 >= c2),
            ((e1 + c1) - c2, e1 - (c2 - c1), c1 < c2),
            ((e1 + c1) // c2, e1 // c2 + c1 // c2, c1 % c2 == 0),
            ((e1 % c1), int32.zero, c1 == 1),
            ((e1 % c1) % c2, e1 % c2, c1 % c2 == 0),
            (e1 / c1 % c2, e1 % (c1 * c2) / c1),
            (e1 / c1 * c1 + e1 % c1, e1),
            (e1 % c1 / c2 * c2 + e1 % c2, e1 % c1, c1 % c2 == 0),
            (
                (e1 * ic1) ^ (e2 * ic2),
                (e1 * (ic1 // 2)) ^ (e2 * (ic2 // 2)) * 2,
                logical_and(ic1 % 2 == 0, ic2 % 2 == 0),
            ),
        ]

        extra_bound_patterns = [
            ((ec1, c1, c2), (ec1, c1, c2), lambda ec1, c1, c2: (ec1 * c1) // c2, lambda ec1, c1, c2: ec1 * (c1 // c2)),
        ]
        self.args.update({e1, e2, c1, c2, ic1, ic2, ec1, ec2})

        self.patterns.extend(extra_patterns)
        self.bound_patterns.extend(extra_bound_patterns)

    def simplify_xor_by_factor(self, e: BitwiseXor) -> BitwiseXor:
        # (a * 8) ^ (b * 16) -> (a ^ (b * 2)) * 8
        self.analyzer(e)
        lhs_set: Optional[Iterable[int]] = self.bound[e.a].candidate_set()
        rhs_set: Optional[Iterable[int]] = self.bound[e.b].candidate_set()

        if lhs_set is None or rhs_set is None:
            return super().visit_BitwiseXor(e)

        divisor = 0
        for candidate in itertools.chain(lhs_set, rhs_set):
            divisor = gcd(divisor, candidate)

        power_two_divisor = 1
        while divisor > 0 and divisor % 2 == 0:
            divisor //= 2
            power_two_divisor *= 2

        if power_two_divisor > 1:
            return (
                BitwiseXor(self.visit(e.a) // power_two_divisor, self.visit(e.b) // power_two_divisor)
                * power_two_divisor
            )
        else:
            return super().visit_BitwiseXor(e)

    def always_bits(self, e: Expr, bit_value: int) -> Optional[set[int]]:
        """
        Analyze the expression `e` and return a set of bits that are always `bit_value` (0 or 1) in the expression `e`.

        Returns a set of bits that are always `bit_value` (0 or 1) in the expression `e`.
        If the expression can not be analyzed, None is returned.
        """
        self.analyzer(e)
        dtype: DataType = self.type_infer(e)
        bound: BoundInfo = self.analyzer.bound[e]
        candidate_set = bound.candidate_set()
        if candidate_set is None:
            return None
        bits = set(range(sizeof(dtype) * 8))
        for candidate in candidate_set:
            for bit in list(bits):
                if ((candidate >> bit) & 1) != bit_value:
                    bits.discard(bit)
        return bits

    def simplify_xor_by_sum(self, e: BitwiseXor) -> BitwiseXor:
        # 10 ^ (a % 8) -> 2 ^ (a % 8) + 8
        if isinstance(e.a, Constant) and isinstance(e.b, Expr):
            e = BitwiseXor(e.b, e.a)  # ensure that a is the variable and b is the constant
        if not (isinstance(e.a, Expr) and isinstance(e.b, Constant)):
            return super().visit_BitwiseXor(e)
        a, b = e.a, e.b

        # find out the bits that are 0 across all candidates of a and are 1 for b
        a_zero_bits = self.always_bits(a, 0)
        b_one_bits = self.always_bits(b, 1)
        if a_zero_bits is None or b_one_bits is None:
            return super().visit_BitwiseXor(e)
        bits = a_zero_bits & b_one_bits
        if bits:
            extracted = 0
            for bit in bits:
                extracted |= 1 << bit
            dtype = self.type_infer(b)
            return BitwiseXor(a, b - dtype(extracted)) + dtype(extracted)
        else:
            return super().visit_BitwiseXor(e)

    def simplify_xor_v3(self, e: BitwiseXor) -> BitwiseXor:
        # (a + b) ^ c
        # where
        # 1. "a" or "b" is a constant
        # 2. always_ones(a + b) & always_zeros(c) != empty
        if isinstance(e.a, Add) and (isinstance(e.a.a, Constant) or isinstance(e.a.b, Constant)):
            add_expr = e.a
            other_expr = e.b
            add_on_left = True
        elif isinstance(e.b, Add) and (isinstance(e.b.a, Constant) or isinstance(e.b.b, Constant)):
            add_expr = e.b
            other_expr = e.a
            add_on_left = False
        else:
            return super().visit_BitwiseXor(e)

        one_bits = self.always_bits(add_expr, 1)
        zero_bits = self.always_bits(other_expr, 0)
        if one_bits is None or zero_bits is None:
            return super().visit_BitwiseXor(e)
        bits = one_bits & zero_bits
        if bits:
            extracted = 0
            for bit in bits:
                extracted |= 1 << bit
            dtype = self.type_infer(add_expr)
            if add_on_left:
                return BitwiseXor(add_expr - dtype(extracted), other_expr) + dtype(extracted)
            else:
                return BitwiseXor(other_expr, add_expr - dtype(extracted)) + dtype(extracted)
        else:
            return super().visit_BitwiseXor(e)

    def visit_BitwiseXor(self, e: BitwiseXor) -> Expr:
        return self.simplify_xor_v3(self.simplify_xor_by_sum(self.simplify_xor_by_factor(e)))


class RuleBasedSimplifyPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        simplifier = RuleBasedSimplifier(None)
        return repeat_until_converge(simplifier, func)


def bound_aware_simplify(
    exprs: Union[Expr, List[Expr]], var2bound: Dict[Var, Union[BoundInfo, int, Tuple[int, int]]]
) -> Any:
    normalized_var2bound = {}
    for var, bound in var2bound.items():
        if isinstance(bound, int):
            bound = BoundInfo(value=bound)
        elif isinstance(bound, tuple) and len(bound) == 2 and isinstance(bound[0], int) and isinstance(bound[1], int):
            bound = BoundInfo(min_value=bound[0], max_value=bound[1])
        elif isinstance(bound, BoundInfo):
            pass
        else:
            raise ValueError(bound)
        normalized_var2bound[var] = bound
    simplifier = RuleBasedSimplifier(normalized_var2bound)
    return repeat_until_converge(simplifier, exprs)


def rule_based_simplify_pass():
    return RuleBasedSimplifyPass()
