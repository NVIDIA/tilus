from typing import no_type_check

from hidet.ir.func import Function
from hidet.ir.primitives.func import call_primitive_func, register_primitive_function
from hidet.utils import initialize


@initialize()
def register_functions():
    from hidet.lang import asm, attrs, script  # pylint: disable=import-outside-toplevel

    @no_type_check
    @script
    def exit_primitive():
        attrs.func_kind = "cuda_internal"
        attrs.func_name = "cuda_exit"

        asm("exit;", outputs=[], inputs=[], is_volatile=True)

    assert isinstance(exit_primitive, Function)
    register_primitive_function(name=exit_primitive.name, func_or_type=exit_primitive)


def exit():
    return call_primitive_func("cuda_exit", args=[])
