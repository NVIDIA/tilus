from hidet.ir.func import Function
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.utils import initialize


@initialize()
def register_functions():
    from hidet.lang import attrs, script, asm  # pylint: disable=import-outside-toplevel

    @script
    def exit_primitive():
        attrs.func_kind = "cuda_internal"
        attrs.func_name = "cuda_exit"

        asm("exit;", outputs=[], inputs=[], is_volatile=True)

    assert isinstance(exit_primitive, Function)
    register_primitive_function(name=exit_primitive.name, func_or_type=exit_primitive)


def exit():
    return call_primitive_func("cuda_exit", args=[])
