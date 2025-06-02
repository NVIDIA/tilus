import numpy as np
from hidet.ir.expr import Var
from tilus.ir.utils.veceval import vectorized_evaluate


def test_vectorized_evaluate():
    from hidet.ir.dtypes import int32

    # Example usage
    x = Var("x", int32)
    y = Var("y", int32)
    expr = x * y + 2

    var2value = {x: np.array([1, 2, 3]), y: np.array([4, 5, 6])}

    result = vectorized_evaluate(expr, var2value)
    assert np.all(result == np.array([6, 12, 20]))
