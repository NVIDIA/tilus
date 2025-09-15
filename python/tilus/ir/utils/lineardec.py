from __future__ import annotations
from typing import Sequence, Optional

from hidet.ir import Constant, Add, Sub, Multiply, Div, Mod
from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, Var
from hidet.ir.functors import ExprFunctor

class LinearDecompositionError(Exception):
    pass

class Linear:
    def __init__(self, c: list[Optional[Expr]]):
        self.c: list[Optional[Expr]] = c

    def __getitem__(self, i: int) -> Optional[Expr]:
        return self.c[i]

    def __setitem__(self, key, value):
        self.c[key] = value

    def __len__(self):
        return len(self.c) - 1

    def __add__(self, other):
        lhs = self.c
        rhs = other.c
        result = []
        for l, r in zip(lhs, rhs):
            if l is None and r is None:
                result.append(None)
            elif l is None:
                result.append(r)
            elif r is None:
                result.append(l)
            else:
                result.append(l + r)
        return Linear(result)

    def __sub__(self, other):
        lhs = self.c
        rhs = other.c
        result = []
        for l, r in zip(lhs, rhs):
            if l is None and r is None:
                result.append(None)
            elif l is None:
                result.append(-r)
            elif r is None:
                result.append(l)
            else:
                result.append(l - r)
        return Linear(result)

    def __mul__(self, other):
        lhs = self
        rhs = other
        result = self.empty(len(lhs))

        if lhs.is_empty():
            return result
        elif rhs.is_empty():
            return result
        elif lhs.is_constant():
            for i in range(len(rhs)):
                if rhs[i] is not None:
                    result[i] = lhs[-1] * rhs[i]
            return result
        elif rhs.is_constant():
            for i in range(len(lhs)):
                if lhs[i] is not None:
                    result[i] = lhs[i] * rhs[-1]
            return result
        else:
            raise LinearDecompositionError("Cannot multiply two non-constant linear expressions")

    def __floordiv__(self, other):
        raise NotImplementedError()

    def __mod__(self, other):
        raise NotImplementedError()

    def is_empty(self):
        return all(coef is None for coef in self.c)

    def is_constant(self) -> bool:
        if any(coef is not None for coef in self.c[:-1]):
            return False
        return True

    @staticmethod
    def empty(n: int):
        return Linear([None for _ in range(n + 1)])

    @staticmethod
    def from_constant(n: int, value: Expr) -> Linear:
        empty = Linear.empty(n)
        if isinstance(value, Constant) and int(value) == 0:
            return empty
        else:
            empty.c[n] = value
            return empty

    @staticmethod
    def from_variable(n: int, var_index: int) -> Linear:
        assert 0 <= var_index < n
        c: list[Optional[Expr]] = [None for _ in range(n + 1)]
        c[var_index] = int32.one
        return Linear(c)

class LinearDecomposer(ExprFunctor):
    def __init__(self, coordinates: Sequence[Var]):
        super().__init__()
        self.n: int = len(coordinates)
        self.coordinates: list[Var] = list(coordinates)

    def decompose(self, e: Expr) -> Linear:
        try:
            return self.visit(e)
        except NotImplementedError:
            raise LinearDecompositionError(
                f'Expression {e} is not linear with respect to coordinates {self.coordinates}'
            ) from e

    def visit_Var(self, e: Var):
        if e in self.coordinates:
            index = self.coordinates.index(e)
            return Linear.from_variable(self.n, index)
        else:
            # we treat variables not in coordinates as constants
            return Linear.from_constant(self.n, e)

    def visit_Constant(self, e: Constant):
        return Linear.from_constant(self.n, e)

    def visit_Add(self, e: Add):
        return self.visit(e.a) + self.visit(e.b)

    def visit_Sub(self, e: Sub):
        return self.visit(e.a) - self.visit(e.b)

    def visit_Multiply(self, e: Multiply):
        return self.visit(e.a) * self.visit(e.b)

    def visit_Div(self, e: Div):
        return self.visit(e.a) // self.visit(e.b)

    def visit_Mod(self, e: Mod):
        return self.visit(e.a) % self.visit(e.b)


def decompose_linear(expr: Expr, coordinates: Sequence[Var]) -> list[Expr]:
    """ Decompose a linear expression to a list of coefficients corresponding to the given coordinates.

    Given an expression `expr` and a list of variables `coordinates`, this function returns a list of coefficients
    `[c0, c1, c2, ..., cn]` such that:

    expr = c0 * coordinates[0] + c1 * coordinates[1] + ... + c_n-1 * coordinates[n-1] + cn

    or raises a `LinearDecompositionError` if the expression is not linear with respect to the given coordinates.

    When the constant term is zero, we do not include it in the result.

    Parameters
    ----------
    expr: Expr
        The expression to be decomposed.
    coordinates: Sequence[Var]
        The list of variables representing the coordinates.

    Returns
    -------
    seq: list[Expr]
        The list of coefficients corresponding to the coordinates, with the last element being the constant term if
        it is non-zero, or we do not include it in the seq (in this case the length of the seq is equal to
        len(coordinates)).
    """
    decomposer = LinearDecomposer(coordinates)
    linear = decomposer.decompose(expr)
    ret = [coef if coef is not None else int32.zero for coef in linear.c]
    if isinstance(ret[-1], Constant) and int(ret[-1]) == 0:
        ret.pop()
    return ret
