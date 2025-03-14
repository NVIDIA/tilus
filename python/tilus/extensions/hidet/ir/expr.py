from typing import List, Union, Optional, Sequence
from hidet.ir.type import string_type, BaseType
from hidet.ir.dtypes import boolean, int32, default_float_dtype, default_int_dtype
from hidet.ir.expr import Var, Expr, var, Constant, cast, Dereference


def deref(v: Expr, derefed_type: Optional[BaseType] = None) -> Expr:
    if derefed_type is not None:
        v = cast(v, ~derefed_type)
    return Dereference(v)


def convert_to_expr(obj: Union[float, bool, int, str, Expr]) -> Expr:
    if isinstance(obj, Expr):
        return obj
    elif isinstance(obj, bool):
        return boolean.constant(obj)
    elif isinstance(obj, int):
        assert default_int_dtype.min_value <= obj <= default_int_dtype.max_value, obj
        return int32.constant(obj)
    elif isinstance(obj, float):
        return default_float_dtype.constant(obj)
    elif isinstance(obj, str):
        return Constant(obj, const_type=string_type())
    else:
        raise ValueError(obj)


def convert_sequence(seq: Sequence) -> tuple:
    return tuple(convert_sequence(item) if isinstance(item, Sequence) else convert_to_expr(item) for item in seq)


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
