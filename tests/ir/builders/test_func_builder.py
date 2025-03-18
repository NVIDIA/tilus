from tilus.ir.builders import FunctionBuilder
from tilus.ir.func import Function


def test_func_builder():
    fb = FunctionBuilder()

    with fb.function("hello_world", num_warps=1, params=[]):
        fb.num_blocks = 1
        fb.printf("Hello, world!\n")

    function = fb.flush_function()

    assert isinstance(function, Function)
