from typing import List, Optional, Sequence

from hidet.ir.expr import Dereference, Expr, Var, as_expr, cast, var
from hidet.ir.type import BaseType


def deref(v: Expr, derefed_type: Optional[BaseType] = None) -> Expr:
    if derefed_type is not None:
        v = cast(v, ~derefed_type)
    return Dereference(v)


def convert_sequence(seq: Sequence) -> tuple:
    return tuple(convert_sequence(item) if isinstance(item, Sequence) else as_expr(item) for item in seq)


def index_vars(num_vars: int) -> List[Var]:
    """Create a list of index variables with given number of variables.

    Parameters
    ----------
    num_vars: int
        The number of index variables to create.

    Returns
    -------
    ret: List[Var]
        The list of created index variables.
    """
    default_names = ["i", "j", "k", "p", "q", "r", "s", "t", "u", "v"]
    if num_vars < len(default_names):
        return [var(default_names[i]) for i in range(num_vars)]
    else:
        return [var("i") for _ in range(num_vars)]


def reinterpret(value: Expr, target_type: BaseType) -> Expr:
    return cast(~value, ~target_type)[0]
