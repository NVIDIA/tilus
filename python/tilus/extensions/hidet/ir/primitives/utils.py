from hidet.ir.func import Function
from hidet.ir.primitives import register_primitive_function

def register_primitive_function_decorator(
    fn: Function
):
    if not isinstance(fn, Function):
        raise TypeError(f'Expected a Function, but got {type(fn)}')
    register_primitive_function(fn.name, func_or_type=fn)
    return fn
