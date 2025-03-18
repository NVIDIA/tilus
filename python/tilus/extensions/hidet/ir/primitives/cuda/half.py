from typing import no_type_check
from hidet.ir.expr import Expr
from hidet.ir.func import Function
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.utils import initialize


@initialize()
def register_functions():
    from hidet.lang import attrs, script, asm, cast  # pylint: disable=import-outside-toplevel
    from hidet.lang.types import uint32, void_p

    @no_type_check
    @script
    def mul_f16x2_(d: void_p, a: uint32, b: uint32):
        attrs.func_kind = "cuda_internal"
        attrs.func_name = "mul_f16x2"

        asm("mul.rn.f16x2 %0, %1, %2;", outputs=[cast(d, ~uint32)[0]], inputs=[a, b], is_volatile=True)

    funcs = [mul_f16x2_]
    for func in funcs:
        assert isinstance(func, Function)
        register_primitive_function(name=func.name, func_or_type=func)


def mul_f16x2(d: Expr, a: Expr, b: Expr) -> Expr:
    """
    Multiply two f16x2 values and store the result in `d`.

    Expect `d` to be an uint32 pointer while `a` an `b` are uint32 values, all of them will be interpreted as f16x2.

    Parameters
    ----------
    d: Expr
        The pointer to the f16x2 result, stored with uint32 data type.
    a: Expr
        The first f16x2 operand stored with uint32 data type.
    b: Expr
        The second f16x2 operand stored with uint32 data type.
    """
    return call_primitive_func("mul_f16x2", args=[d, a, b])
