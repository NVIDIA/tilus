from __future__ import annotations

import operator
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Type

from hidet.ir import BitwiseXor, DataType, Div, Equal, Mod
from hidet.ir.expr import Add, BinaryExpr, Constant, Expr, LessEqual, LessThan, LogicalAnd, Multiply, Sub, Var
from hidet.ir.functors import IRFunctor
from hidet.utils import gcd


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

        return "tensor(shape={}, continuity={}, divisibility={}, constancy={}, value={})".format(
            self.shape, continuity, divisibility, constancy, value
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

    This class implements an iterative algorithm to solve the above equations.
    """

    def __init__(self) -> None:
        super().__init__()
        self.shape: Sequence[int] = []
        self.var2info: Dict[Var, TensorInfo] = {}

    def analyze(
        self, axes: Sequence[Var], shape: Sequence[int], var2value: Mapping[Var, TensorInfo], expr: Expr
    ) -> TensorInfo:
        self.var2info.clear()
        self.var2info.update(var2value)

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


def analyze_grid(
    shape: Sequence[int], axes: Sequence[Var], var2info: Mapping[Var, TensorInfo], expr: Expr
) -> TensorInfo:
    """
    Given the mapping from axes -> value, we could construct a tensor with given shape. This function analyze the
    tensor information (TensorInfo) of this tensor.
    """
    analyzer = GridAnalyzer()
    ret = analyzer.analyze(axes, shape, var2info, expr)
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
