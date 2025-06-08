from __future__ import annotations

import logging
from typing import Optional, Sequence

from hidet.ir import Call
from hidet.ir.expr import Add, Constant, Div, Expr, Mod, Multiply, Sub, Var
from hidet.ir.functors import IRFunctor as HidetIRFunctor
from hidet.ir.tools import collect as hidet_collect
from hidet.ir.type import DataType

import tilus.logging
from tilus.ir.func import Analysis, Function
from tilus.ir.stmt import AssignStmt, DeclareStmt, ForStmt, LetStmt
from tilus.ir.tools import IRPrinter, collect
from tilus.utils import gcd

logger = tilus.logging.get_logger(__name__)

"""
The number of updates for each variable's scalar set before we give up and set the upper and/or lower bound to None.
"""
UPDATE_COUNT_LIMIT = 10


def c_style_div(a: int, b: int) -> int:
    """
    Perform C-style integer division, which rounds towards zero, instead of towards negative infinity.

    Python and C handle negative division differently, so we implement this function to ensure
    that the division behaves like C's integer division. In python, the `//` operator rounds towards negative infinity,
    while in C, it rounds towards zero. For example:
    >>> c_style_div(-5, 2)
    -2
    >>> c_style_div(5, -2)
    -2
    >>> c_style_div(-5, -2)
    2
    while in python:
    >>> (-5) // 2
    -3
    >>> 5 // (-2)
    -3
    >>> (-5) // (-2)
    2

    Parameters
    ----------
    a: int
        The dividend.
    b: int
        The divisor.

    Returns
    -------
    ret: int
        The result of the division, rounded towards zero.
    """
    if a * b < 0 and a % b != 0:
        return a // b + 1
    else:
        return a // b


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

        assert self.divisibility >= 0
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

    def __hash__(self):
        return hash((self.divisibility, self.lower_bound, self.upper_bound))

    def is_empty(self) -> bool:
        return (
            self.lower_bound is not None
            and self.upper_bound is not None
            and (
                self.lower_bound > self.upper_bound
                or (
                    self.divisibility != 0
                    and self.upper_bound // self.divisibility * self.divisibility < self.lower_bound
                )
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
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and other.lower_bound is not None
            and other.upper_bound is not None
        ):
            bounds = [
                self.lower_bound * other.lower_bound,
                self.lower_bound * other.upper_bound,
                self.upper_bound * other.lower_bound,
                self.upper_bound * other.upper_bound,
            ]
            lb = min(bounds) if bounds else None
            ub = max(bounds) if bounds else None
        elif (
            self.lower_bound is not None
            and self.lower_bound >= 0
            and other.lower_bound is not None
            and other.lower_bound is not None
        ):
            lb = self.lower_bound * other.lower_bound
            ub = None
        else:
            lb = ub = None

        return ScalarSet(divisibility=div, lower_bound=lb, upper_bound=ub)

    def __floordiv__(self, other: ScalarSet) -> ScalarSet:
        if self.is_empty() or other.is_empty():
            return self.empty_set()

        if other.is_constant():
            other_value = other.lower_bound
            div, lu, rb = 1, None, None
            if self.lower_bound is not None:
                lu = c_style_div(self.lower_bound, other_value)
            if self.upper_bound is not None:
                rb = c_style_div(self.upper_bound, other_value)
            div = self.divisibility // gcd(self.divisibility, abs(other_value))
            return ScalarSet(divisibility=div, lower_bound=lu, upper_bound=rb)
        else:
            # Calculate bounds (assuming positive numbers for simplicity)
            lb = None
            if self.lower_bound is not None and other.upper_bound is not None and other.upper_bound > 0:
                lb = c_style_div(self.lower_bound, other.upper_bound)

            ub = None
            if self.upper_bound is not None and other.lower_bound is not None and other.lower_bound > 0:
                ub = c_style_div(self.upper_bound, other.lower_bound)

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
                divisibility=self.divisibility // gcd(self.divisibility, abs(mod_value)),
                lower_bound=lb,
                upper_bound=ub,
            )
        else:
            ub = None
            if other.upper_bound is not None:
                ub = other.upper_bound - 1

            return ScalarSet(divisibility=1, lower_bound=0, upper_bound=ub)

    @staticmethod
    def min(lhs: ScalarSet, rhs: ScalarSet) -> ScalarSet:
        if lhs.is_empty() or rhs.is_empty():
            return ScalarSet.empty_set()
        result = ScalarSet()
        result.divisibility = gcd(lhs.divisibility, rhs.divisibility)

        if lhs.lower_bound is not None and rhs.lower_bound is not None:
            result.lower_bound = min(lhs.lower_bound, rhs.lower_bound)

        if lhs.upper_bound is not None and rhs.upper_bound is not None:
            result.upper_bound = min(lhs.upper_bound, rhs.upper_bound)
        return result

    @staticmethod
    def max(lhs: ScalarSet, rhs: ScalarSet) -> ScalarSet:
        if lhs.is_empty() or rhs.is_empty():
            return ScalarSet.empty_set()
        result = ScalarSet()
        result.divisibility = gcd(lhs.divisibility, rhs.divisibility)

        if lhs.lower_bound is not None and rhs.lower_bound is not None:
            result.lower_bound = max(lhs.lower_bound, rhs.lower_bound)

        if lhs.upper_bound is not None and rhs.upper_bound is not None:
            result.upper_bound = max(lhs.upper_bound, rhs.upper_bound)
        return result


class ScalarSetAnalyzer(HidetIRFunctor):
    def __init__(self, var2info: dict[Var, ScalarSet]):
        super().__init__()
        self.var2info = var2info

    def __call__(self, expr: Expr) -> ScalarSet:
        return super().visit(expr)

    def visit_Var(self, var: Var) -> ScalarSet:
        return self.var2info[var] if var in self.var2info else ScalarSet()

    def visit_Constant(self, constant: Constant) -> ScalarSet:
        if constant.type.is_integer():  # type: ignore
            value: int = int(constant.value)  # type: ignore
            return ScalarSet(divisibility=abs(value), lower_bound=value, upper_bound=value)
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

    def visit_Call(self, e: Call) -> ScalarSet:
        func_name = e.func_var.name
        if func_name == "generic_min":
            return ScalarSet.min(self.visit(e.args[0]), self.visit(e.args[1]))
        elif func_name == "generic_max":
            return ScalarSet.max(self.visit(e.args[0]), self.visit(e.args[1]))
        else:
            # a set contains all integers
            logger.warning("Unknown function call in scalar analysis: {}, fallback to universe set.".format(func_name))
            return ScalarSet()

    def visit_dispatch(self, node):
        if isinstance(node, (Var, Add, Sub, Multiply, Div, Mod, Constant, Call)):
            return super().visit_dispatch(node)
        elif isinstance(node, Expr):
            return ScalarSet()
        else:
            raise NotImplementedError(f"Unsupported node type: {type(node)}")


class AnalysisCache:
    def __init__(self, vars: Sequence[Var]):
        self.vars = vars
        self.expr2used: dict[Expr, list[Var]] = {}
        # expr to ((used var sets), result set)
        self.expr2set: dict[Expr, tuple[tuple[ScalarSet, ...], ScalarSet]] = {}


def has_smaller_lower_bound(lhs: ScalarSet, rhs: ScalarSet) -> bool:
    if lhs.lower_bound is None and rhs.lower_bound is not None:
        # -oo is smaller than any finite number
        return True
    if lhs.lower_bound is not None and rhs.lower_bound is not None and lhs.lower_bound < rhs.lower_bound:
        # lhs is smaller than rhs
        return True
    return False


def has_larger_upper_bound(lhs: ScalarSet, rhs: ScalarSet) -> bool:
    if lhs.upper_bound is None and rhs.upper_bound is not None:
        # +oo is larger than any finite number
        return True
    if lhs.upper_bound is not None and rhs.upper_bound is not None and lhs.upper_bound > rhs.upper_bound:
        # lhs is larger than rhs
        return True
    return False


def analyze_scalar_set(
    analyzer: ScalarSetAnalyzer, cache: AnalysisCache, var2set: dict[Var, ScalarSet], expr: Expr
) -> ScalarSet:
    if expr not in cache.expr2used:
        used_vars = hidet_collect(expr, node_types=Var)
        cache.expr2used[expr] = [var for var in used_vars if var in cache.vars]

    used_vars = cache.expr2used[expr]
    arg_sets = tuple(var2set[var] for var in used_vars)

    if expr not in cache.expr2set:
        result_set = analyzer.visit(expr)
        cache.expr2set[expr] = (arg_sets, result_set)
        return result_set
    else:
        cached_args_sets, cached_result_set = cache.expr2set[expr]
        if arg_sets != cached_args_sets:
            # The sets of the used variables have changed, we need to recompute the result set
            result_set = analyzer.visit(expr)
            cache.expr2set[expr] = (arg_sets, result_set)
            return result_set
        else:
            # The sets of the used variables have not changed, we can return the cached result set
            return cached_result_set


def analyze_scalar(func: Function) -> Function:
    var2set: dict[Var, ScalarSet] = {}
    lower_count: dict[Var, int] = {}
    upper_count: dict[Var, int] = {}

    # update the scalar set of parameters
    metadata = func.metadata
    for param in func.params:
        if param in metadata.param2divisibility:
            # we assume that the input parameters are non-negative
            var2set[param] = ScalarSet(divisibility=metadata.param2divisibility[param], lower_bound=0)

    # update the scalar set of built-in variables
    for i, var in enumerate(metadata.block_indices):  # type: ignore
        if isinstance(metadata.num_blocks[i], Constant):
            var2set[var] = ScalarSet(lower_bound=0, upper_bound=int(metadata.num_blocks[i]) - 1)
        else:
            var2set[var] = ScalarSet(lower_bound=0)

    # collect all the statements that manipulate integer scalar values
    stmts: list[DeclareStmt | LetStmt | ForStmt | AssignStmt] = []
    variables: list[Var] = []
    for stmt in collect(func, types=[DeclareStmt, LetStmt, ForStmt, AssignStmt]):
        if isinstance(stmt, AssignStmt):
            if isinstance(stmt.var.type, DataType) and stmt.var.type.is_integer():
                stmts.append(stmt)
        elif isinstance(stmt, DeclareStmt):
            if isinstance(stmt.var.type, DataType) and stmt.var.type.is_integer():
                stmts.append(stmt)
                variables.append(stmt.var)
        elif isinstance(stmt, ForStmt):
            stmts.append(stmt)
            variables.append(stmt.iter_var)
        elif isinstance(stmt, LetStmt):
            stmts.append(stmt)
            for bind_var in stmt.bind_vars:
                if isinstance(bind_var.type, DataType) and bind_var.type.is_integer():
                    variables.append(bind_var)
        else:
            raise NotImplementedError()

    # initialize the scalar set of variables defined in the function body to be empty set
    for var in variables:
        var2set[var] = ScalarSet(lower_bound=0, upper_bound=-1)  # empty set
    for var in var2set:
        lower_count[var] = 0
        upper_count[var] = 0

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
    # there might be case for self-referencing variables, e.g.,
    #    var = var + 1
    # in this case, we will not be able to reach a fixed point, thus we set a limit of iterations to update one
    # variable's upper and lower bounds. Once we reach the limit, we will set the upper bound and/or lower bound to None
    # to indicate that the variable is boundless on that dimension, and continue the analysis until we reach fix point.
    printer = IRPrinter()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Analyzing scalar sets for function {printer(func)}")
    while True:
        updated = False
        for stmt in stmts:
            analyzer = ScalarSetAnalyzer(var2set)
            var_list: list[Var] = []
            rhs_sets: list[ScalarSet] = []
            if isinstance(stmt, (AssignStmt, DeclareStmt, LetStmt)):
                value_list: list[Expr] = []
                if isinstance(stmt, AssignStmt):
                    var_list.append(stmt.var)
                    value_list.append(stmt.value)
                elif isinstance(stmt, DeclareStmt):
                    if stmt.init is not None:
                        var_list.append(stmt.var)
                        value_list.append(stmt.init)
                elif isinstance(stmt, LetStmt):
                    for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
                        if isinstance(bind_var.type, DataType) and bind_var.type.is_integer():
                            var_list.append(bind_var)
                            value_list.append(bind_value)
                else:
                    assert False
                for var, value in zip(var_list, value_list):
                    rhs_sets.append(analyzer.visit(value))
                    # rhs_sets.append(analyze_scalar_set(analyzer, cache, var2set, value))
            elif isinstance(stmt, ForStmt):
                extent_info: ScalarSet = analyzer.visit(stmt.extent)
                var_list.append(stmt.iter_var)
                if extent_info.upper_bound is not None:
                    rhs_sets.append(ScalarSet(lower_bound=0, upper_bound=extent_info.upper_bound - 1))
                else:
                    rhs_sets.append(ScalarSet(lower_bound=0))
            else:
                assert False
            original_sets = [var2set[var] for var in var_list]
            union_sets = [original_set | rhs_set for original_set, rhs_set in zip(original_sets, rhs_sets)]
            for var, original_set, union_set in zip(var_list, original_sets, union_sets):
                if union_set != original_set:
                    if has_smaller_lower_bound(union_set, original_set):
                        lower_count[var] += 1
                        if lower_count[var] > UPDATE_COUNT_LIMIT:
                            union_set.lower_bound = None
                    if has_larger_upper_bound(union_set, original_set):
                        upper_count[var] += 1
                        if upper_count[var] > UPDATE_COUNT_LIMIT:
                            union_set.upper_bound = None
                    var2set[var] = union_set
                    updated = True
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Updated scalar set for {printer(var)}: {original_set} -> {union_set}")
        if not updated:
            break

    # check to make sure there is no empty set in the final result
    empty_set_vars = []
    for var, scalar_set in var2set.items():
        if scalar_set.is_empty():
            empty_set_vars.append(var)
    if len(empty_set_vars) > 0:
        vars = ", ".join(str(printer(var)) for var in empty_set_vars)
        raise ValueError("Found the following variables with empty scalar sets: {}".format(vars))

    # collect the final result
    analysis = Analysis.create(
        divisibility={var: var2set[var].divisibility for var in var2set if var2set[var].divisibility != 1},
        lower_bound={var: var2set[var].lower_bound for var in var2set if var2set[var].lower_bound is not None},
        upper_bound={var: var2set[var].upper_bound for var in var2set if var2set[var].upper_bound is not None},
    )
    ret = func.with_metadata(func.metadata.with_analysis(analysis))

    return ret
