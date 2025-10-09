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

import logging
import operator
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Type

from hidet.ir import BitwiseXor, DataType, Div, Equal, Mod, TensorElement
from hidet.ir.dtypes import int32
from hidet.ir.expr import (
    Add,
    BinaryExpr,
    BitwiseAnd,
    BitwiseOr,
    Constant,
    Expr,
    LeftShift,
    LessEqual,
    LessThan,
    LogicalAnd,
    Multiply,
    RightShift,
    Sub,
    Call,
    Var,
)
from hidet.ir.functors import IRFunctor
from hidet.ir.layout import is_power_of_two
from hidet.ir.tools import collect
from hidet.utils import gcd

import tilus.logging
from tilus.ir.func import Analysis

logger = tilus.logging.get_logger(__name__)


def compute_value(a: DimensionInfo, b: DimensionInfo, op: Callable[[int, int], int]) -> Optional[int]:
    if a.value is not None and b.value is not None:
        return op(a.value, b.value)
    else:
        return None


class DimensionInfo:
    """
    Given one tensor, there will be multiple dimensions. For a given dimension, we could get many vectors for that
    dimension. For example, given a two-dimension tensor:
    [[0, 1, 2, 3],
     [4, 5, 6, 7]]
    It has two dimensions. For a dimension i of a tensor with shape=[s1, s2, ..., sn], we can extract n vectors where
      n = s1 * s2 * ... * sn / si.
    For example, the two-dimension tensor example:
        dimension 0: {[0, 1, 2, 3], [4, 5, 6, 7]}
        dimension 1: {[0, 4], [1, 5], [2, 6], [3, 7]}

    - length: the length of the dimension
    - value: if all elements of each vector are identical to a specific value, we store this value, otherwise None.
    - continuity: if we could split each vector into multiple same-length segments, and elements in each segment are
      contiguous, we use `continuity` to store the number of elements of the segments.
    - constancy: if we could split each vector into multiple same-length segments, and elements in each segment are
      identical, we use `constancy` to store the number of elements of the segments.
    - divisibility: obviously, it's impossible to have both `continuity` > 1 and `constancy` > 1. let segments[i] to be
      the i-th segment following the partition patterns of `continuity` or `constancy` (depends on which > 1). We use
      the divisibility to represent gcd(segments[i][0] for all segment i).

    In above two-dimension tensor example, we have
    - dimension 0: DimensionInfo(length=3, value=None, continuity=4, constancy=1, divisibility=4)
    - dimension 1: DimensionInfo(length=2, value=None, continuity=1, constancy=1, divisibility=1)

    Some other examples:

    0 1 2 3 8 9 10 11 12 13 14 15
    DimensionInfo(length=12, value=None, continuity=4, constancy=1, divisibility=4)

    2 2 2 2 4 4 4 4
    DimensionInfo(length=8, value=None, continuity=1, constancy=4, divisibility=2)

    3 3 3 3 3 3 3 3
    DimensionInfo(length=8, value=3, continuity=1, constancy=8, divisibility=3)

    8*a 8*a 8*a 8*a
    DimensionInfo(length=4, value=None, continuity=1, constancy=4, divisibility=8)
    """

    def __init__(self, value: Optional[int], continuity: int, constancy: int, divisibility: int):
        super().__init__()
        self.value: Optional[int] = value
        self.continuity: int = continuity
        self.constancy: int = constancy
        self.divisibility: int = divisibility

        assert divisibility >= 0 and constancy >= 0 and continuity >= 0

    def __add__(self, other):
        assert isinstance(other, DimensionInfo)
        lhs, rhs = self, other

        if lhs.constancy > 1 and rhs.constancy > 1:
            continuity = 1
            constancy = gcd(lhs.constancy, rhs.constancy)
            divisibility = gcd(lhs.divisibility, rhs.divisibility)
        elif lhs.continuity > 1 and rhs.constancy > 1:
            continuity = gcd(lhs.continuity, rhs.constancy)
            constancy = 1
            divisibility = gcd(continuity, lhs.divisibility, rhs.divisibility)
        elif lhs.constancy > 1 and rhs.continuity > 1:
            continuity = gcd(lhs.constancy, rhs.continuity)
            constancy = 1
            divisibility = gcd(continuity, lhs.divisibility, rhs.divisibility)
        else:
            continuity, constancy, divisibility = 1, 1, 1
        return DimensionInfo(
            compute_value(self, other, lambda a, b: a + b),
            continuity=continuity,
            constancy=constancy,
            divisibility=divisibility,
        )

    def __sub__(self, other):
        assert isinstance(other, DimensionInfo)
        add_result = self + other
        add_result.value = compute_value(self, other, lambda a, b: a - b)
        return add_result

    def __mul__(self, other):
        assert isinstance(other, DimensionInfo)

        if self.constancy > 1 and other.constancy > 1:
            continuity = 1
            constancy = gcd(self.constancy, other.constancy)
            divisibility = self.divisibility * other.divisibility
        elif self.constancy > 1 and other.continuity > 1:
            #   self: 8, 8, 8, 8, 16, 16, 16, 16
            #  other: 0, 1, 4, 6,  8,  9, 12, 13
            # result: 0, 8, 32, 48, 128, 144, 192, 208
            continuity = 1
            constancy = 1
            divisibility = self.divisibility
        elif self.continuity > 1 and other.constancy > 1:
            continuity = 1
            constancy = 1
            divisibility = other.divisibility
        else:
            continuity, constancy, divisibility = 1, 1, 1
        return DimensionInfo(
            value=compute_value(self, other, lambda a, b: a * b),
            continuity=continuity,
            constancy=constancy,
            divisibility=divisibility,
        )

    def __floordiv__(self, other):
        assert isinstance(other, DimensionInfo)
        value = None
        if other.value is not None and self.continuity % other.value == 0 and self.divisibility % other.value == 0:
            continuity = 1
            constancy = other.value
            divisibility = self.divisibility // other.value
        elif self.constancy > 1 and other.constancy > 1:
            continuity = 1
            constancy = gcd(self.constancy, other.constancy)
            divisibility = 1
        else:
            continuity, constancy, divisibility = 1, 1, 1
        return DimensionInfo(
            value=compute_value(self, other, lambda a, b: a // b) if value is None else value,
            continuity=continuity,
            constancy=constancy,
            divisibility=divisibility,
        )

    def __mod__(self, other):
        assert isinstance(other, DimensionInfo)
        value = None
        if other.value is not None and other.value == 1:
            continuity = 1
            constancy = other.constancy
            divisibility = 0
            value = 0
        elif other.value is not None and self.continuity % other.value == 0 and self.divisibility % other.value == 0:
            continuity = other.value
            constancy = 1
            divisibility = 0
        elif self.constancy > 1 and other.constancy > 1:
            continuity = 1
            constancy = gcd(self.constancy, other.constancy)
            divisibility = 1
        else:
            continuity, constancy, divisibility = 1, 1, 1
        return DimensionInfo(
            value=compute_value(self, other, lambda a, b: a % b) if value is None else value,
            continuity=continuity,
            constancy=constancy,
            divisibility=divisibility,
        )

    def __lt__(self, other):
        assert isinstance(other, DimensionInfo)
        if self.constancy > 1 and other.constancy > 1:
            continuity = 1
            constancy = gcd(self.constancy, other.constancy)
            divisibility = 1
        elif self.constancy > 1:
            continuity = 1
            constancy = gcd(self.constancy, self.divisibility, other.continuity, other.divisibility)
            divisibility = 1
        elif other.constancy > 1:
            continuity = 1
            constancy = gcd(other.constancy, other.divisibility, self.continuity, self.divisibility)
            divisibility = 1
        else:
            continuity, constancy, divisibility = 1, 1, 1
        return DimensionInfo(
            value=compute_value(self, other, lambda a, b: a < b),
            continuity=continuity,
            constancy=constancy,
            divisibility=divisibility,
        )

    def __le__(self, other):
        assert isinstance(other, DimensionInfo)
        if self.constancy > 1 and other.constancy > 1:
            continuity = 1
            constancy = gcd(self.constancy, other.constancy)
            divisibility = 1
        elif self.constancy > 1:
            continuity = 1
            constancy = gcd(self.constancy, self.divisibility, other.continuity, other.divisibility)
            divisibility = 1
        elif other.constancy > 1:
            continuity = 1
            constancy = gcd(other.constancy, other.divisibility, self.continuity, self.divisibility)
            divisibility = 1
        else:
            continuity, constancy, divisibility = 1, 1, 1
        return DimensionInfo(
            value=compute_value(self, other, lambda a, b: a <= b),
            continuity=continuity,
            constancy=constancy,
            divisibility=divisibility,
        )

    def __eq__(self, other):
        assert isinstance(other, DimensionInfo)
        if self.constancy > 1 and other.constancy > 1:
            continuity = 1
            constancy = gcd(self.constancy, other.constancy)
            divisibility = 1
        elif self.constancy > 1:
            continuity = 1
            constancy = gcd(self.constancy, self.divisibility, other.continuity, other.divisibility)
            divisibility = 1
        elif other.constancy > 1:
            continuity = 1
            constancy = gcd(other.constancy, other.divisibility, self.continuity, self.divisibility)
            divisibility = 1
        else:
            continuity, constancy, divisibility = 1, 1, 1
        return DimensionInfo(
            value=compute_value(self, other, lambda a, b: a == b),
            continuity=continuity,
            constancy=constancy,
            divisibility=divisibility,
        )

    def __and__(self, other):
        assert isinstance(other, DimensionInfo)
        if self.constancy > 1 and other.constancy > 1:
            continuity = 1
            constancy = gcd(self.constancy, other.constancy)
            divisibility = 1
        else:
            continuity, constancy, divisibility = 1, 1, 1
        return DimensionInfo(
            value=compute_value(self, other, lambda a, b: a and b),
            continuity=continuity,
            constancy=constancy,
            divisibility=divisibility,
        )

    def __xor__(self, other):
        assert isinstance(other, DimensionInfo)
        if self.constancy > 1 and other.constancy > 1:
            continuity = 1
            constancy = gcd(self.constancy, other.constancy)
            divisibility = 1
        elif (
            self.continuity > 1
            and is_power_of_two(self.continuity)
            and other.constancy > 1
            and is_power_of_two(other.constancy)
            and is_power_of_two(other.divisibility)
            and self.continuity % other.divisibility == 0
        ):
            continuity = gcd(self.continuity, other.constancy, other.divisibility)
            constancy = 1
            divisibility = gcd(self.divisibility, other.divisibility)
        elif (
            other.continuity > 1
            and is_power_of_two(other.continuity)
            and self.constancy > 1
            and is_power_of_two(self.constancy)
            and is_power_of_two(self.divisibility)
            and other.continuity % self.divisibility == 0
        ):
            continuity = gcd(self.constancy, self.divisibility, other.continuity)
            constancy = 1
            divisibility = gcd(self.divisibility, other.divisibility)
        else:
            continuity, constancy, divisibility = 1, 1, 1
        return DimensionInfo(
            value=compute_value(self, other, lambda a, b: a and b),
            continuity=continuity,
            constancy=constancy,
            divisibility=divisibility,
        )


class TensorInfo:
    """
    The tensor information for the tensors whose elements are integers.


    """

    def __init__(self, shape: Sequence[int], infos: Sequence[DimensionInfo]):
        self.shape: Sequence[int] = shape
        self.infos: Sequence[DimensionInfo] = infos

    def __str__(self):
        continuity = [dim.continuity for dim in self.infos]
        constancy = [dim.constancy for dim in self.infos]
        divisibility = [dim.divisibility for dim in self.infos]
        value = [dim.value for dim in self.infos]

        return "tensor(shape={}, continuity={}, constancy={}, divisibility={}, value={})".format(
            self.shape, continuity, constancy, divisibility, value
        )

    def __getitem__(self, item):
        return self.infos[item]

    def _binary(self, other, op):
        assert isinstance(other, TensorInfo)
        assert all(a == b for a, b in zip(self.shape, other.shape))
        return TensorInfo(shape=self.shape, infos=[op(a, b) for a, b in zip(self.infos, other.infos)])

    def __add__(self, other):
        return self._binary(other, operator.add)

    def __sub__(self, other):
        return self._binary(other, operator.sub)

    def __mul__(self, other):
        return self._binary(other, operator.mul)

    def __floordiv__(self, other):
        return self._binary(other, operator.floordiv)

    def __mod__(self, other):
        return self._binary(other, operator.mod)

    def __lt__(self, other):
        return self._binary(other, operator.lt)

    def __le__(self, other):
        return self._binary(other, operator.le)

    def __eq__(self, other):
        return self._binary(other, operator.eq)

    def __and__(self, other):
        return self._binary(other, operator.and_)

    def __xor__(self, other):
        return self._binary(other, operator.xor)

    @staticmethod
    def from_axis(shape: Sequence[int], dim: int) -> TensorInfo:
        infos: List[DimensionInfo] = []
        for i, extent in enumerate(shape):
            if i == dim:
                infos.append(DimensionInfo(value=None, continuity=extent, divisibility=0, constancy=1))
            else:
                if shape[dim] == 1:
                    infos.append(DimensionInfo(value=None, continuity=1, divisibility=0, constancy=extent))
                else:
                    infos.append(DimensionInfo(value=None, continuity=1, divisibility=1, constancy=extent))
        return TensorInfo(shape, infos)

    @staticmethod
    def from_constant(shape: Sequence[int], value: Optional[int]) -> TensorInfo:
        infos: List[DimensionInfo] = []

        for i, extent in enumerate(shape):
            infos.append(
                DimensionInfo(
                    value=value, continuity=1, constancy=extent, divisibility=abs(value) if value is not None else 1
                )
            )
        return TensorInfo(shape, infos)

    @staticmethod
    def from_divisibility(shape: Sequence[int], divisibility: int) -> TensorInfo:
        infos: List[DimensionInfo] = []

        for i, extent in enumerate(shape):
            infos.append(DimensionInfo(value=None, continuity=1, constancy=extent, divisibility=divisibility))
        return TensorInfo(shape, infos)

    @staticmethod
    def any(shape: Sequence[int]) -> TensorInfo:
        infos: List[DimensionInfo] = []

        for i in range(len(shape)):
            infos.append(DimensionInfo(value=None, continuity=1, constancy=1, divisibility=1))
        return TensorInfo(shape, infos)


class GridAnalyzer(IRFunctor):
    """
    Given a variable, it may have multiple potential values
    a := 8
    b := 16
       | a + 4
    c := b
       | a + b
    Let value(x) be the Value (with constancy and divisibility) of x.
    Then we have the following equations:
    value(a) = value(8)
    value(b) = merge(value(16), value(a) + value(4))
    value(c) = merge(value(b), value(a) + value(b))

    This class implements an iterative algorithm to solve the above equations to find a fixed point.
    """

    def __init__(self) -> None:
        super().__init__()
        self.shape: Sequence[int] = []
        self.var2info: Dict[Var, TensorInfo] = {}

    def analyze(
        self, axes: Sequence[Var], shape: Sequence[int], var2info: Mapping[Var, TensorInfo], expr: Expr
    ) -> TensorInfo:
        self.var2info.clear()
        self.var2info.update(var2info)

        # Initialize the value of each axis
        for dim, axis in enumerate(axes):
            self.var2info[axis] = TensorInfo.from_axis(shape, dim=dim)

        self.shape = shape
        info = self.visit(expr)
        return info

    def visit_Var(self, e: Var) -> TensorInfo:
        if e in self.var2info:
            return self.var2info[e]
        else:
            return TensorInfo.from_constant(self.shape, value=None)

    def visit_Constant(self, e: Constant) -> TensorInfo:
        if isinstance(e.type, DataType) and e.type.is_integer():
            assert isinstance(e.value, int)
            return TensorInfo.from_constant(self.shape, value=int(e.value))
        else:
            return TensorInfo.from_constant(self.shape, value=None)

    def visit_TensorElement(self, e: TensorElement) -> TensorInfo:
        # we do not do any assumption on tensor elements
        return TensorInfo.from_constant(self.shape, value=None)

    def visit_binary(self, e: BinaryExpr) -> TensorInfo:
        a = self.visit(e.a)
        b = self.visit(e.b)
        op_dict: Dict[Type[Expr], Callable] = {
            Add: operator.add,
            Sub: operator.sub,
            Multiply: operator.mul,
            Div: operator.floordiv,
            Mod: operator.mod,
            LessThan: operator.lt,
            LessEqual: operator.le,
            Equal: operator.eq,
            LogicalAnd: operator.and_,
            BitwiseXor: operator.xor,
        }
        if type(e) not in op_dict:
            raise NotImplementedError()
        c = op_dict[type(e)](a, b)

        return c

    def visit_Add(self, e: Add) -> TensorInfo:
        return self.visit_binary(e)

    def visit_Sub(self, e: Sub) -> TensorInfo:
        return self.visit_binary(e)

    def visit_Multiply(self, e: Multiply) -> TensorInfo:
        return self.visit_binary(e)

    def visit_Mod(self, e: Mod) -> TensorInfo:
        return self.visit_binary(e)

    def visit_Div(self, e: Div) -> TensorInfo:
        return self.visit_binary(e)

    def visit_LessThan(self, e: LessThan) -> TensorInfo:
        return self.visit_binary(e)

    def visit_LessEqual(self, e: LessEqual) -> TensorInfo:
        return self.visit_binary(e)

    def visit_And(self, e: LogicalAnd) -> TensorInfo:
        return self.visit_binary(e)

    def visit_BitwiseXor(self, e: BitwiseXor) -> TensorInfo:
        return self.visit_binary(e)

    def visit_Equal(self, e: Equal) -> TensorInfo:
        return self.visit_binary(e)

    def visit_RightShift(self, e: RightShift) -> TensorInfo:
        if isinstance(e.b, Constant):
            return self.visit_Div(Div(e.a, int32(2 ** int(e.b.value))))  # type: ignore
        return super().visit_RightShift(e)

    def visit_LeftShift(self, e: LeftShift) -> TensorInfo:
        if isinstance(e.b, Constant):
            return self.visit_Multiply(Multiply(e.a, int32(2 ** int(e.b.value))))  # type: ignore
        return super().visit_LeftShift(e)

    def visit_BitwiseAnd(self, e: BitwiseAnd) -> TensorInfo:
        return TensorInfo.any(self.shape)

    def visit_BitwiseOr(self, e: BitwiseOr) -> TensorInfo:
        return TensorInfo.any(self.shape)
    
    def visit_Call(self, e: Call):
        if e.func_var.name == "swizzle":
            # y = swizzle(x, mbase, bbits, sshift) = x ^ ((x & ((1 << bbits) - 1) << (mbase + sshift)) >> sshift)
            x, mbase, bbits, sshift = e.args
            if not all(isinstance(arg, Constant) for arg in [mbase, bbits, sshift]):
                raise NotImplementedError("Swizzle mbase, bbits, and sshift must be constants, got {}.".format(e.args[1:]))
            mbase = int(mbase)
            bbits = int(bbits)
            sshift = int(sshift)
            x_tensor_info: TensorInfo = self.visit(x)
            y_infos: list[DimensionInfo] = []
            for x_info in x_tensor_info.infos:
                y_infos.append(DimensionInfo(
                    value=x_info.value ^ ((x_info.value & (((1 << bbits) - 1) << (mbase + sshift))) >> sshift) if x_info.value is not None else None,
                    continuity=gcd(x_info.continuity, 1 << mbase),
                    constancy=gcd(x_info.constancy, 1 << mbase),
                    divisibility=gcd(x_info.divisibility, 1 << mbase),
                ))
            return TensorInfo(self.shape, y_infos)
        else:
            raise NotImplementedError("Function {} is not supported in grid analysis yet.".format(e.func_var.name))
    

def analyze_grid(
    shape: Sequence[int],
    axes: Sequence[Var],
    expr: Expr,
    *,
    var2info: Optional[Mapping[Var, TensorInfo]] = None,
    analysis: Optional[Analysis] = None,
) -> TensorInfo:
    """
    Given the mapping from axes -> value, we could construct a tensor with given shape. This function analyze the
    tensor information (TensorInfo) of this tensor.
    """
    analyzer = GridAnalyzer()

    if var2info is None:
        var2info = {}
    else:
        var2info = dict(var2info)

    if analysis is not None:
        used_vars: list[Var] = collect(expr, [Var], stop_when_found=True)
        for v in used_vars:
            if v in axes:
                continue
            if (
                v in analysis.lower_bound
                and v in analysis.upper_bound
                and analysis.lower_bound[v] == analysis.upper_bound[v]
            ):
                var2info[v] = TensorInfo.from_constant(shape=shape, value=analysis.lower_bound[v])
            elif v in analysis.divisibility:
                var2info[v] = TensorInfo.from_divisibility(shape=shape, divisibility=analysis.divisibility[v])

    ret = analyzer.analyze(axes, shape, var2info, expr)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("shape: %s, axes: %s, expr: %s", shape, axes, expr)
        for var in var2info:
            logger.debug("%s: %s", var, var2info[var])
        logger.debug("ret: %s", ret)

    return ret


if __name__ == "__main__":
    from hidet.ir.dtypes import int32
    from hidet.ir.expr import Var

    # i, = index_vars(1)
    # e = ((((i / 4) % 1) * 4) + ((i % 4) % 4))
    # ret = analyze_value(shape=[1024], axes=[i], var2info={}, expr=e)

    # i, r0, t = index_vars(3)
    # e = ((((((i / 2) % 2) * 8) + ((t % 32) / 4)) * 4096) + ( (r0 * 32) + (((((i / 4) * 4) + (t % 4)) * 2) + (i % 2))))
    # analyze_value(shape=[1024], axes=[i], var2info={}, expr=e)

    names = ["b0", "i", "b1", "r0", "j", "k", "u0"]
    b0, i, b1, r0, j, k, u0 = [Var(name, int32) for name in names]
    # e = logical_and(
    #         logical_and(((b0 + i) < 1), (((b1 * 16) + j) < 1)),
    #         ((((r0 * 32) + (u0 * 16)) + k) < 4096)
    # )
    # e = ((((r0 * 32) + (u0 * 16)) + k) < 4096)
    e = k < 4096
    analyze_grid(shape=[1, 16, 16], axes=[i, j, k], var2info={}, expr=e)
