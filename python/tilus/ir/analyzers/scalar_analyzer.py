from __future__ import annotations

from typing import Mapping, Optional

from hidet.ir.expr import Add, Constant, Div, Mod, Multiply, Sub, Var
from hidet.ir.functors import IRFunctor as HidetIRFunctor
from hidet.ir.primitives.cuda.vars import blockIdx
from hidet.ir.type import DataType
from tilus.ir.func import Analysis, Function
from tilus.ir.stmt import AssignStmt, DeclareStmt, ForStmt
from tilus.ir.tools import collect
from tilus.utils import gcd


class ScalarSet:
    """
    The scalar set abstracts a set of integers that a scalar value could be.

    for each integer n, if it holds the following conditions:
        1) n is divisible by divisibility
        2) when lower_bound is not None, and n is greater than or equal to lower_bound
        3) when upper_bound is not None, and n is less than or equal to upper_bound
    Then, n is in the set represented by the scalar set object.

    We have the following examples:

    ScalarSet(divisibility=2, lower_bound=0, upper_bound=10) represents {0, 2, 4, 6, 8, 10}
    ScalarSet(divisibility=3, lower_bound=0, upper_bound=10) represents {0, 3, 6, 9}
    ScalarSet(divisibility=2, lower_bound=0) represents {0, 2, 4, 6, ...} all even numbers greater than or equal to 0
    ScalarSet(divisibility=1) represents all integers

    When we have two scalar sets: sa and sb, we could perform the following operations:
      `sa op sb`, where op could be +, -, *, //, %,
    Let sc' = {a op b for a, b in sa, sb},
    We define sc = one minimal set that includes sc' and could be represented as a scalar set object.
    Here minimal is under the set inclusion relation. When there are multiple minimal sets, we choose the one
    with the largest divisibility.
    """

    def __init__(self, divisibility: int = 1, lower_bound: Optional[int] = None, upper_bound: Optional[int] = None):
        self.divisibility: int = divisibility
        self.lower_bound: Optional[int] = lower_bound
        self.upper_bound: Optional[int] = upper_bound

        assert self.divisibility >= 1
        assert upper_bound is None or isinstance(upper_bound, int)
        assert lower_bound is None or isinstance(lower_bound, int)

    def __str__(self):
        items = []
        if self.divisibility != 1:
            items.append(f"divisibility={self.divisibility}")
        if self.lower_bound is not None:
            items.append(f"lower_bound={self.lower_bound}")
        if self.upper_bound is not None:
            items.append(f"upper_bound={self.upper_bound}")
        return f"ScalarSet({', '.join(items)})"

    def is_empty(self) -> bool:
        return (
            self.lower_bound is not None
            and self.upper_bound is not None
            and (
                self.lower_bound > self.upper_bound
                or self.upper_bound // self.divisibility * self.divisibility < self.lower_bound
            )
        )

    def is_constant(self) -> bool:
        return (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound == self.upper_bound
            and self.lower_bound % self.divisibility == 0
        )

    @staticmethod
    def empty_set() -> ScalarSet:
        return ScalarSet(lower_bound=0, upper_bound=-1)

    def __eq__(self, other: ScalarSet) -> bool:
        if self.is_empty() and other.is_empty():
            return True
        return (
            self.divisibility == other.divisibility
            and self.lower_bound == other.lower_bound
            and self.upper_bound == other.upper_bound
        )

    def __or__(self, other: ScalarSet) -> ScalarSet:
        if self.is_empty():
            return other
        return ScalarSet(
            divisibility=gcd(self.divisibility, other.divisibility),
            lower_bound=None
            if self.lower_bound is None or other.lower_bound is None
            else min(self.lower_bound, other.lower_bound),
            upper_bound=None
            if self.upper_bound is None or other.upper_bound is None
            else max(self.upper_bound, other.upper_bound),
        )

    def __add__(self, other: ScalarSet) -> ScalarSet:
        if self.is_empty() or other.is_empty():
            return self.empty_set()

        div = gcd(self.divisibility, other.divisibility)

        lb = None
        if self.lower_bound is not None and other.lower_bound is not None:
            lb = self.lower_bound + other.lower_bound

        ub = None
        if self.upper_bound is not None and other.upper_bound is not None:
            ub = self.upper_bound + other.upper_bound

        return ScalarSet(divisibility=div, lower_bound=lb, upper_bound=ub)

    def __sub__(self, other: ScalarSet) -> ScalarSet:
        if self.is_empty() or other.is_empty():
            return self.empty_set()

        div = gcd(self.divisibility, other.divisibility)

        lb = None
        if self.lower_bound is not None and other.upper_bound is not None:
            lb = self.lower_bound - other.upper_bound

        ub = None
        if self.upper_bound is not None and other.lower_bound is not None:
            ub = self.upper_bound - other.lower_bound

        return ScalarSet(divisibility=div, lower_bound=lb, upper_bound=ub)

    def __mul__(self, other: ScalarSet) -> ScalarSet:
        if self.is_empty() or other.is_empty():
            return ScalarSet(lower_bound=0, upper_bound=-1)  # empty set

        div = self.divisibility * other.divisibility

        # Calculate bounds accounting for signs
        bounds = []
        if self.lower_bound is not None and other.lower_bound is not None:
            bounds.append(self.lower_bound * other.lower_bound)
        if self.lower_bound is not None and other.upper_bound is not None:
            bounds.append(self.lower_bound * other.upper_bound)
        if self.upper_bound is not None and other.lower_bound is not None:
            bounds.append(self.upper_bound * other.lower_bound)
        if self.upper_bound is not None and other.upper_bound is not None:
            bounds.append(self.upper_bound * other.upper_bound)

        lb = min(bounds) if bounds else None
        ub = max(bounds) if bounds else None

        return ScalarSet(divisibility=div, lower_bound=lb, upper_bound=ub)

    def __floordiv__(self, other: ScalarSet) -> ScalarSet:
        if self.is_empty() or other.is_empty():
            return self.empty_set()

        if other.is_constant():
            other_value = other.lower_bound
            div, lu, rb = 1, None, None
            if other.lower_bound is not None:
                lu = self.lower_bound // other_value
            if other.upper_bound is not None:
                rb = self.upper_bound // other_value
            div = self.divisibility // gcd(self.divisibility, other.lower_bound)
            return ScalarSet(divisibility=div, lower_bound=lu, upper_bound=rb)
        else:
            # Calculate bounds (assuming positive numbers for simplicity)
            lb = None
            if self.lower_bound is not None and other.upper_bound is not None and other.upper_bound > 0:
                lb = self.lower_bound // other.upper_bound

            ub = None
            if self.upper_bound is not None and other.lower_bound is not None and other.lower_bound > 0:
                ub = self.upper_bound // other.lower_bound

            return ScalarSet(divisibility=1, lower_bound=lb, upper_bound=ub)

    def __mod__(self, other: ScalarSet) -> ScalarSet:
        if self.is_empty() or other.is_empty():
            return self.empty_set()

        if other.is_constant():
            # rhs is a constant
            mod_value = other.lower_bound
            lb, ub = 0, mod_value - 1
            if self.lower_bound is not None and self.upper_bound is not None:
                if self.lower_bound >= 0 and self.upper_bound < mod_value:
                    lb = max(lb, mod_value)
                    ub = min(ub, self.upper_bound)

            return ScalarSet(
                divisibility=self.divisibility // gcd(self.divisibility, mod_value),
                lower_bound=lb,
                upper_bound=ub,
            )
        else:
            ub = None
            if other.upper_bound is not None:
                ub = other.upper_bound - 1

            return ScalarSet(divisibility=1, lower_bound=0, upper_bound=ub)


class ScalarSetAnalyzer(HidetIRFunctor):
    def __init__(self, var2info: Mapping[Var, ScalarSet]):
        super().__init__()
        self.var2info = var2info

    def visit_Var(self, var: Var) -> ScalarSet:
        info = self.var2info.get(var, None)
        return info if info is not None else ScalarSet()

    def visit_Constant(self, constant: Constant) -> ScalarSet:
        if constant.type.is_integer():  # type: ignore
            return ScalarSet(divisibility=constant.value, lower_bound=constant.value, upper_bound=constant.value)  # type: ignore
        else:
            return ScalarSet()

    def visit_Add(self, e: Add) -> ScalarSet:
        return self.visit(e.a) + self.visit(e.b)

    def visit_Sub(self, e: Sub) -> ScalarSet:
        return self.visit(e.a) - self.visit(e.b)

    def visit_Multiply(self, e: Multiply) -> ScalarSet:
        return self.visit(e.a) * self.visit(e.b)

    def visit_Div(self, e: Div) -> ScalarSet:
        return self.visit(e.a) // self.visit(e.b)

    def visit_Mod(self, e: Mod) -> ScalarSet:
        return self.visit(e.a) % self.visit(e.b)


def analyze_scalar(func: Function) -> Function:
    var2set: dict[Var, ScalarSet] = {}

    # update the scalar set of parameters
    metadata = func.metadata
    for param in func.params:
        if param in metadata.param2divisibility:
            # we assume that the input parameters are non-negative
            var2set[param] = ScalarSet(divisibility=metadata.param2divisibility[param], lower_bound=0)

    # update the scalar set of built-in variables
    for i, var in enumerate([blockIdx.x, blockIdx.y, blockIdx.z]):  # type: ignore
        if isinstance(metadata.num_blocks[i], Constant):
            var2set[var] = ScalarSet(lower_bound=0, upper_bound=int(metadata.num_blocks[i]) - 1)
        else:
            var2set[var] = ScalarSet(lower_bound=0)

    # collect all the statements that manipulate integer scalar values
    stmts: list[DeclareStmt | ForStmt | AssignStmt] = []
    variables: list[Var] = []
    for stmt in collect(func, types=[DeclareStmt, ForStmt, AssignStmt]):
        if isinstance(stmt, AssignStmt) and isinstance(stmt.var.type, DataType) and stmt.var.type.is_integer():
            stmts.append(stmt)
        elif isinstance(stmt, DeclareStmt) and isinstance(stmt.var.type, DataType) and stmt.var.type.is_integer():
            stmts.append(stmt)
            variables.append(stmt.var)
        elif isinstance(stmt, ForStmt):
            stmts.append(stmt)
            variables.append(stmt.iter_var)

    # initialize the scalar set of variables defined in the function body to be empty set
    for var in variables:
        var2set[var] = ScalarSet(lower_bound=0, upper_bound=-1)  # empty set

    # for each variable, there might be multiple statements that define its possible values:
    #    var = expr1(...)
    #        | expr2(...)
    #        | expr3(...)
    # we iteratively update the scalar set of each variable:
    #    set[var] = set[var]
    #             | scalar_set(expr1)
    #             | scalar_set(expr2)
    #             | scalar_set(expr3)
    # until the scalar set of each variable does not change, i.e., we reach a fixed point.
    while True:
        updated = False
        for stmt in stmts:
            analyzer = ScalarSetAnalyzer(var2set)
            if isinstance(stmt, AssignStmt):
                union_set = var2set[stmt.var] | analyzer(stmt.value)
                if var2set[stmt.var] != union_set:
                    var2set[stmt.var] = union_set
                    updated = True
                    # print("update {}: {}".format(stmt.var, union_set))
            elif isinstance(stmt, DeclareStmt):
                if stmt.init is not None:
                    union_set = var2set[stmt.var] | analyzer(stmt.init)
                    if var2set[stmt.var] != union_set:
                        var2set[stmt.var] = union_set
                        updated = True
                        # print("update {}: {}".format(stmt.var, union_set))
            elif isinstance(stmt, ForStmt):
                extent_info: ScalarSet = analyzer.visit(stmt.extent)
                if extent_info.upper_bound is not None:
                    range_set = ScalarSet(lower_bound=0, upper_bound=extent_info.upper_bound - 1)
                    union_set = var2set[stmt.iter_var] | range_set
                    if var2set[stmt.iter_var] != union_set:
                        var2set[stmt.iter_var] = union_set
                        updated = True
                        # print("update {}: {}".format(stmt.iter_var, union_set))
            else:
                assert False
        if not updated:
            break

    # collect the final result
    analysis = Analysis.create(
        divisibility={var: var2set[var].divisibility for var in var2set if var2set[var].divisibility != 1},
        lower_bound={var: var2set[var].lower_bound for var in var2set if var2set[var].lower_bound is not None},
        upper_bound={var: var2set[var].upper_bound for var in var2set if var2set[var].upper_bound is not None},
    )
    # print(analysis.divisibility)
    # print(analysis.lower_bound)
    # print(analysis.upper_bound)
    return func.with_metadata(func.metadata.with_analysis(analysis))
